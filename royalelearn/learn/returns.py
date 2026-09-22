"""Reward scaling, and the running statistic it is scaled by.

Rewards are divided by the running standard deviation of the discounted returns the run has seen
so far, and then clipped. The mean is **never** subtracted: the reward is zero-sum between the two
seats, and subtracting a mean from it turns a loss into a smaller win and destroys the sign
structure the whole reward composition is built on.

The statistic is taken over the **unstandardised, unclipped** returns, which is the only version
of them that is comparable across iterations, and it is updated with Welford's method in its
batched form so that a hundred thousand samples an iteration cost one merge and carry no
cancellation. The triple is checkpointed, because a resume that starts the divisor again from one
puts a step change into the loss at the point the run was resumed.

This matters more here than it does in the reference learners: a discount approaching 0.999
inflates return magnitudes roughly tenfold over 0.99.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import torch
from torch import Tensor

from ..errors import CheckpointFormatError

__all__ = ["WelfordReturnScaler", "WelfordState", "discounted_returns"]


def discounted_returns(rewards: Tensor, ended: Tensor, gamma: float) -> Tensor:
    """The discounted return from every cell to the end of its own episode.

    ``ended`` is true where the episode ended AT that cell, terminated or truncated alike, and the
    sum is cut there: a return that ran on into the next episode would be a statistic about a
    quantity no policy ever receives.
    """
    cycles = rewards.shape[0]
    out = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    alive = (~ended).to(rewards.dtype)
    for t in range(cycles - 1, -1, -1):
        carry = rewards[t] + gamma * carry * alive[t]
        out[t] = carry
    return out


class WelfordState(msgspec.Struct):
    """The running triple, as it is checkpointed and logged.

    ``m2`` is the sum of squared deviations, so the variance is ``m2 / (count - 1)``: the sample
    variance rather than the population one, because the divisor is an estimate from a sample and
    a run's first iteration is a small one.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0


class WelfordReturnScaler:
    """The divisor rewards are scaled by, and the clip that follows it."""

    FORMAT_VERSION = 1

    #: The file the triple is checkpointed in.
    STATE_FILE = "return_scaler.json"

    def __init__(self, *, standardize: bool = True, clip: float = 10.0) -> None:
        self.standardize = standardize
        self.clip = clip
        self._state = WelfordState()

    # -- the statistic -----------------------------------------------------

    @property
    def count(self) -> int:
        return self._state.count

    @property
    def mean(self) -> float:
        """Recorded and logged; never subtracted from a reward."""
        return self._state.mean

    @property
    def variance(self) -> float:
        if self._state.count < 2:
            return 0.0
        return self._state.m2 / (self._state.count - 1)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def divisor(self) -> float:
        """What a reward is divided by: the running standard deviation, or one.

        One until there are two samples to take a sample variance from, and one whenever
        standardisation is off, so that the scaled reward is the reward and the metric row says
        so rather than reporting a scale nobody applied.
        """
        if not self.standardize or self._state.count < 2:
            return 1.0
        std = self.std
        return std if std > 0.0 else 1.0

    def update(self, values: Tensor | np.ndarray) -> None:
        """Merge a batch of raw returns into the running triple.

        Chan's parallel form rather than a per-sample loop: one merge for the whole batch, with
        the same result to floating-point rounding and none of the per-sample cost.
        """
        batch = _as_float64(values).reshape(-1)
        n = int(batch.size)
        if n == 0:
            return
        batch_mean = float(batch.mean())
        batch_m2 = float(((batch - batch_mean) ** 2).sum())
        state = self._state
        if state.count == 0:
            self._state = WelfordState(count=n, mean=batch_mean, m2=batch_m2)
            return
        total = state.count + n
        delta = batch_mean - state.mean
        mean = state.mean + delta * n / total
        m2 = state.m2 + batch_m2 + delta * delta * state.count * n / total
        self._state = WelfordState(count=total, mean=mean, m2=m2)

    # -- the scaling -------------------------------------------------------

    def scale(self, rewards: Tensor, *, over: Tensor | None = None) -> tuple[Tensor, float]:
        """``(rewards / divisor)`` clipped, and the share of it the clip bound touched.

        ``over`` restricts the reported share to the cells it selects -- the trainable ones,
        which are what every other reported statistic describes. It is a mask here rather than a
        second comparison at the caller because there is one clip bound and there should be one
        test against it; two that have to be kept in step disagree at the boundary.
        """
        divisor = self.divisor
        scaled = rewards / divisor if divisor != 1.0 else rewards.clone()
        touched = (scaled.abs() > self.clip).to(torch.float32)
        if over is not None:
            touched = touched[over]
        fraction = float(touched.mean().item()) if touched.numel() else 0.0
        return scaled.clamp_(-self.clip, self.clip), fraction

    # -- state -------------------------------------------------------------

    def state(self) -> WelfordState:
        return msgspec.structs.replace(self._state)

    def load_state(self, state: WelfordState) -> None:
        self._state = msgspec.structs.replace(state)

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": self.FORMAT_VERSION, "state": self._state}
        (folder / self.STATE_FILE).write_bytes(msgspec.json.encode(payload))

    def load_checkpoint(self, folder: Path, *, strict: bool) -> None:
        path = folder / self.STATE_FILE
        if not path.exists():
            if strict:
                raise CheckpointFormatError(f"the return scaler's state is not at {path}")
            print(f"no return scaler state at {path}; starting its statistics from nothing")
            self._state = WelfordState()
            return
        payload: dict[str, Any] = msgspec.json.decode(path.read_bytes())
        version = int(payload.get("format_version", 0))
        if version > self.FORMAT_VERSION:
            raise CheckpointFormatError(
                f"{path} was written at return-scaler format {version} and this build reads "
                f"{self.FORMAT_VERSION}"
            )
        self._state = msgspec.convert(payload["state"], WelfordState)


def _as_float64(values: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)
    return values.detach().to(device="cpu", dtype=torch.float64).numpy()
