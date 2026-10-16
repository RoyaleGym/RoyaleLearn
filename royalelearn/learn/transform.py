"""``ppo.policy_transform``: the policy the learner trains, as a function of its actor's logits.

Unset, the learner's policy is the masked softmax of the actor's logits. Set to
``plugin:<module>:<function>``, it is the masked softmax of ``f(logits, mask, vector, layout)``,
on every path the learner's policy is read: its seats in training battles, the overlap's
behaviour snapshot, the update's forward on either pairing and on a subset of rows, the ratio
check, the entropy terms and the extra actor terms. The stored log-probabilities are then the
transformed policy's, so the first ratio of an update is one as it is without it.

What is not the learner's training policy does not see it: a snapshot holds the actor's weights,
so pool members, seeds, the gate, evaluation, probes and ``Learner.play`` play the actor itself
under ``ladder.opponent_mode`` and ``ladder.release_mode``, the live learner included.

The function takes the head's float32 logits ``(B, n_actions)`` before any masking, the mask
``(B, n_actions)``, the observation vector ``(B, V)`` and the action layout, all on the run's
device, and returns float32 logits shaped like its input. It is differentiated, so it is built
from torch operations on its input; it must be finite on every legal action; and a row's output
depends on that row alone, because the rollout and the update batch rows differently. The first
batch it is handed is checked for all of that, and a transform that fails stops the run naming
it. The masking and the ``log_softmax`` come after it, so its output need not be normalised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from .decode import PLUGIN, import_plugin, parse_mode

if TYPE_CHECKING:
    from torch import Tensor

    from ..api.policy import ObsBatch

__all__ = ["PolicyLayout", "PolicyTransform", "transform_problems"]

#: How far apart a row's output may be when it is batched with other rows than before.
ROW_ATOL = 1e-5


class PolicyLayout(NamedTuple):
    """Where each action sits: the no-op, then ``hand_size * tiles`` grid actions slot by slot,
    then ``buttons`` ability buttons."""

    hand_size: int
    tiles: int
    buttons: int


def transform_problems(mode: Any, where: str) -> list[str]:
    """Why ``mode`` cannot name a transform, or nothing."""
    if not isinstance(mode, str):
        return [f"{where} must be a string, plugin:<module>:<function>"]
    try:
        kind, _ = parse_mode(mode)
    except ValueError as exc:
        return [f"{where}: {exc}"]
    if kind != PLUGIN:
        return [
            f"{where} is {mode!r}: a transform is a function of your own, "
            "plugin:<module>:<function>"
        ]
    return []


class PolicyTransform:
    """The function ``ppo.policy_transform`` names, bound to the run's action layout.

    Called with the actor's logits and the batch they were computed on, it returns the
    transformed logits. Shared by the model and its behaviour snapshots, so the first batch's
    check runs once a run.
    """

    __slots__ = ("_checked", "function", "layout", "mode")

    def __init__(self, mode: str, layout: PolicyLayout) -> None:
        problems = transform_problems(mode, "ppo.policy_transform")
        if problems:
            raise ValueError(problems[0])
        self.mode = mode
        self.layout = layout
        self.function = import_plugin(parse_mode(mode)[1], "policy transform")
        self._checked = False

    def __call__(self, logits: Tensor, obs: ObsBatch) -> Tensor:
        import torch

        out = self.function(logits, obs.mask, obs.vector, self.layout)
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != logits.shape
            or out.dtype is not torch.float32
        ):
            got = (
                f"{out.dtype} {tuple(out.shape)}"
                if isinstance(out, torch.Tensor)
                else type(out).__name__
            )
            raise ValueError(
                f"policy transform {self.mode!r} returned {got}; it must return float32 logits "
                f"shaped like its input, {tuple(logits.shape)}"
            )
        if logits.requires_grad and not out.requires_grad:
            raise ValueError(
                f"policy transform {self.mode!r} passes no gradient: the update differentiates "
                "it, so build its output from the input logits with torch operations"
            )
        if not self._checked and logits.shape[0] >= 2:
            self._checked = True
            self._check(logits.detach(), obs, out.detach())
        return out

    def _check(self, logits: Tensor, obs: ObsBatch, out: Tensor) -> None:
        """The first batch: finite on every legal action, and each row its own.

        A row is computed again in another order and in half the batch; a transform that mixes
        rows gives it other numbers there, and the rollout's batches are not the update's.
        """
        import torch

        mask = obs.mask
        bad = (~torch.isfinite(out) & mask).any(dim=1)
        if bool(bad.any()):
            rows = bad.nonzero().flatten()[:8].tolist()
            raise ValueError(
                f"policy transform {self.mode!r} gave a legal action a non-finite logit on rows "
                f"{rows}: every legal action needs a finite log-probability"
            )
        batch = int(logits.shape[0])
        with torch.no_grad():
            order = torch.arange(batch - 1, -1, -1, device=logits.device)
            half = torch.arange(batch // 2, device=logits.device)
            worst = 0.0
            for rows in (order, half):
                vector = None if obs.vector is None else obs.vector.index_select(0, rows)
                again = self.function(
                    logits.index_select(0, rows), mask.index_select(0, rows), vector, self.layout
                )
                gap = torch.where(
                    mask.index_select(0, rows),
                    (again - out.index_select(0, rows)).abs(),
                    torch.zeros_like(again),
                )
                worst = max(worst, float(gap.max()))
        if not worst <= ROW_ATOL:
            raise ValueError(
                f"policy transform {self.mode!r} gives a row other logits when the row is batched "
                f"with other rows (by up to {worst:.3g}): each row's output must depend on that "
                "row alone, because the rollout and the update batch rows differently"
            )
