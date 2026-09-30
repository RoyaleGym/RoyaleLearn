"""How a frozen policy picks its action: sampled, its single most likely action, or ``gtau``.

``stochastic`` samples the masked distribution, ``argmax`` takes its mode. On this action space
neither is how a cloned policy plays best: sampled, it picks a rare action often enough to play
badly, and its single most likely action is nearly always the no-op, so its mode barely plays.

``gtau:<x>`` decides in two steps, reading the masked distribution (an illegal action's
probability is zero before anything else is read):

1. play iff ``1 - p(no-op) > x``; otherwise the no-op;
2. if it plays, the group with the largest summed probability: each hand slot summed over its
   tiles, and each ability button on its own. For a slot, that slot's most likely tile; for a
   button, the press. With nothing legal but the no-op, the no-op.

Batched on the device, one ``argmax`` per step, so a whole round of frozen seats is decoded in the
forward that produced it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from .distribution import MaskedCategorical

__all__ = [
    "ARGMAX",
    "GTAU",
    "STOCHASTIC",
    "decode_actions",
    "gtau_actions",
    "mode_problems",
    "parse_mode",
]

STOCHASTIC, ARGMAX, GTAU = "stochastic", "argmax", "gtau"
_GTAU_PREFIX = GTAU + ":"


def parse_mode(mode: str) -> tuple[str, float]:
    """``(kind, threshold)`` of a decode mode; the threshold is 0 except for ``gtau:<x>``."""
    if mode in (STOCHASTIC, ARGMAX):
        return mode, 0.0
    if mode.startswith(_GTAU_PREFIX):
        try:
            threshold = float(mode[len(_GTAU_PREFIX) :])
        except ValueError:
            threshold = -1.0
        if 0.0 <= threshold < 1.0:
            return GTAU, threshold
    raise ValueError(
        f"decode mode {mode!r} is not 'stochastic', 'argmax' or 'gtau:<x>' with 0 <= x < 1"
    )


def mode_problems(mode: str, where: str) -> list[str]:
    try:
        parse_mode(mode)
    except ValueError as exc:
        return [f"{where}: {exc}"]
    return []


def gtau_actions(log_probs: Tensor, *, threshold: float, hand_size: int, tiles: int) -> Tensor:
    """``(B,)`` int64: the ``gtau`` decode of masked log-probabilities ``(B, n_actions)`` laid out
    as the no-op, ``hand_size * tiles`` grid actions, then any ability buttons."""
    import torch

    p = log_probs.exp()
    batch = p.shape[0]
    grid = 1 + hand_size * tiles
    per_tile = p[:, 1:grid].reshape(batch, hand_size, tiles)
    groups = torch.cat([per_tile.sum(-1), p[:, grid:]], dim=-1)
    best = groups.argmax(-1)
    slot = best.clamp(max=hand_size - 1)
    tile = per_tile.argmax(-1).gather(1, slot[:, None])[:, 0]
    action = torch.where(best < hand_size, 1 + slot * tiles + tile, grid + best - hand_size)
    play = ((1.0 - p[:, 0]) > threshold) & (groups.gather(1, best[:, None])[:, 0] > 0.0)
    return torch.where(play, action, torch.zeros_like(action))


def decode_actions(
    distribution: MaskedCategorical,
    mode: str,
    uniforms: Tensor,
    *,
    hand_size: int,
    tiles: int,
) -> Tensor:
    """``(B,)`` int64 actions of ``distribution`` under ``mode``; ``uniforms`` feed a sample."""
    kind, threshold = parse_mode(mode)
    if kind == STOCHASTIC:
        return distribution.sample(uniforms)
    if kind == ARGMAX:
        return distribution.mode()
    return gtau_actions(
        distribution.log_probs, threshold=threshold, hand_size=hand_size, tiles=tiles
    )
