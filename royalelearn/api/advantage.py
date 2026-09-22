"""How a return is turned into an advantage.

One ABC, because the estimator is the piece most likely to be replaced -- V-trace against a
frozen pool, a different lambda rule, a per-bucket normalisation -- and because the one thing it
must never do is carry its recursion across an episode boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import msgspec

from .checkpoint import Checkpointable

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor

__all__ = ["AdvantageEstimator", "AdvantageStats"]


class AdvantageStats(msgspec.Struct):
    """What the estimator saw, for the metric row.

    ``reward_scale`` is the divisor the return scaler applied and ``clipped_reward_frac`` is how
    much of the batch the clip bound touched: both are how a scaled reward stays legible after
    the scaling.
    """

    raw_return_mean: float
    raw_return_std: float
    reward_scale: float
    clipped_reward_frac: float


class AdvantageEstimator(ABC, Checkpointable):
    """Rewards and values in, advantages and returns out."""

    @abstractmethod
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
        """``rewards``/``terminated``/``truncated``/``trainable`` are ``(T, R)``; ``values`` is
        ``(T+1, R)``; ``final_values`` is ``(T, R)`` and is read only where truncated.
        Returns ``(advantages (T, R), returns (T, R), stats)``.

        Implementers MUST bootstrap a terminated cell from 0 and a truncated cell from
        ``final_values``, and MUST NOT carry the recursion across an episode boundary. A
        truncation is an episode that was cut, not one that was decided, and treating the two
        alike throws away the value of every position a step limit ended.
        """
