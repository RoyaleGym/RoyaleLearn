"""Generalised advantage estimation over the iteration's rectangle.

Two implementations of one recursion. ``gae_recursion`` is vectorised over slots -- one backward
loop of ``T`` steps over ``(R,)`` vectors, on the device -- and ``reference_gae`` is the readable
one, a python loop over a single slot. The pair exists so that the test can hold the fast one to
the slow one over every boundary case; without the test the pair is just two things to keep in
step.

Every array here is indexed the way the rectangle is: row ``t`` is one timestep. ``rewards[t]``
is what the action taken at ``t`` earned, ``values[t]`` is ``V(s_t)``, ``terminated[t]`` and
``truncated[t]`` say whether that action ended the episode, and ``values[t + 1]`` is the state
it led to. ``delta_t = r_t + gamma * boot_t - V(s_t)`` is then one timestep's temporal
difference and nothing has to be shifted on the way in.

Two rules decide everything that is subtle here:

**What a cell bootstraps from is decided by how its episode ended.** A terminated cell bootstraps
from zero, because there is no state after a finished battle. A truncated cell bootstraps from
``V(final_obs)``, because the battle was cut and not decided, and throwing away the value of the
position a step limit ended is throwing away the whole episode's tail. A cell that is neither --
including the last cycle of an iteration for a row still mid-episode -- bootstraps from the value
of the next cycle, which under SAME_STEP autoreset is the value of the true next observation,
because a row is replaced only when its episode ended and an episode that ended is flagged.

**The recursion breaks at every episode end, truncated or terminated alike.** Breaking only on
termination leaks the next episode's advantage backwards across a truncation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..api.advantage import AdvantageEstimator, AdvantageStats
from .returns import WelfordReturnScaler, discounted_returns

__all__ = ["GAE", "bootstrap_values", "gae_recursion", "reference_gae"]


def bootstrap_values(
    values: Tensor, final_values: Tensor, terminated: Tensor, truncated: Tensor
) -> Tensor:
    """``(T, R)``: what each cell's next-state value is, by how its episode ended."""
    zero = torch.zeros_like(final_values)
    return torch.where(
        terminated, zero, torch.where(truncated, final_values, values[1:])
    )


def gae_recursion(
    *,
    rewards: Tensor,
    values: Tensor,
    final_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Advantages and returns, ``(T, R)`` each, vectorised over slots.

    ``rewards`` is already scaled; ``values`` is ``(T+1, R)`` and its last row is the bootstrap
    cycle's.
    """
    cycles = rewards.shape[0]
    ended = terminated | truncated
    boot = bootstrap_values(values, final_values, terminated, truncated)
    alive = (~ended).to(rewards.dtype)
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(cycles - 1, -1, -1):
        delta = rewards[t] + gamma * boot[t] - values[t]
        carry = delta + gamma * lam * alive[t] * carry
        advantages[t] = carry
    return advantages, advantages + values[:cycles]


def reference_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    final_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The same recursion, written for a reader: python floats, one slot at a time.

    Arrays are ``(T, R)`` except ``values``, which is ``(T+1, R)``; a one-dimensional input is
    read as a single slot.
    """
    rewards = np.atleast_2d(np.asarray(rewards, dtype=np.float64).T).T
    values = np.atleast_2d(np.asarray(values, dtype=np.float64).T).T
    final_values = np.atleast_2d(np.asarray(final_values, dtype=np.float64).T).T
    terminated = np.atleast_2d(np.asarray(terminated, dtype=bool).T).T
    truncated = np.atleast_2d(np.asarray(truncated, dtype=bool).T).T
    cycles, slots = rewards.shape
    advantages = np.zeros((cycles, slots), dtype=np.float64)
    returns = np.zeros((cycles, slots), dtype=np.float64)
    for r in range(slots):
        carry = 0.0
        for t in range(cycles - 1, -1, -1):
            if terminated[t, r]:
                boot = 0.0
            elif truncated[t, r]:
                boot = float(final_values[t, r])
            else:
                boot = float(values[t + 1, r])
            delta = float(rewards[t, r]) + gamma * boot - float(values[t, r])
            if terminated[t, r] or truncated[t, r]:
                carry = delta
            else:
                carry = delta + gamma * lam * carry
            advantages[t, r] = carry
            returns[t, r] = carry + float(values[t, r])
    return advantages, returns


class GAE(AdvantageEstimator):
    """The shipped estimator: scale the rewards, then run the recursion.

    The scaler lives here rather than in the buffer because the scale is a property of the
    returns, which is a thing only the estimator computes; it is checkpointed through this
    component's own folder.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        standardize_rewards: bool = True,
        reward_clip: float = 10.0,
        scaler: WelfordReturnScaler | None = None,
    ) -> None:
        self.scaler = scaler or WelfordReturnScaler(
            standardize=standardize_rewards, clip=reward_clip
        )

    def compute(
        self,
        *,
        rewards: Tensor,
        values: Tensor,
        final_values: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        trainable: Tensor,
        gamma: float,
        lam: float,
    ) -> tuple[Tensor, Tensor, AdvantageStats]:
        """Advantages, returns and what the estimator saw.

        ``trainable`` selects the cells the return statistic learns from and the cells the
        reported numbers describe. It does not mask the recursion: a cell that is not trainable
        still sits between two that are, and the buffer has already zeroed a dead row's rewards
        and values and flagged it ended, so the loop runs over it inertly.
        """
        ended = terminated | truncated
        raw = discounted_returns(rewards, ended, gamma)
        selected = raw[trainable]
        self.scaler.update(selected)
        scaled, clipped_frac = self.scaler.scale(rewards, over=trainable)
        advantages, returns = gae_recursion(
            rewards=scaled,
            values=values,
            final_values=final_values,
            terminated=terminated,
            truncated=truncated,
            gamma=gamma,
            lam=lam,
        )
        stats = AdvantageStats(
            raw_return_mean=float(selected.mean().item()) if selected.numel() else 0.0,
            raw_return_std=float(selected.std(unbiased=True).item())
            if selected.numel() > 1
            else 0.0,
            reward_scale=self.scaler.divisor,
            clipped_reward_frac=clipped_frac,
        )
        return advantages, returns, stats

    # -- checkpoint --------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        self.scaler.save_checkpoint(folder)

    def load_checkpoint(self, folder: Path, *, strict: bool) -> None:
        self.scaler.load_checkpoint(folder, strict=strict)
