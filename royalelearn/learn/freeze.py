"""Holding the actor still while the critic warms up, and what the row says about it.

The mechanism is in the update: where the scheduled actor learning-rate scale is zero, the actor
runs no loss and its optimizer does not step (``learn/ppo.py``). What is here is the bookkeeping
that has to survive a resume: whether the last iteration was frozen, and when the actor was last
let go. It exists only on a run that schedules the scale, so a run that does not writes no key
and no state for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.metrics import Alarm

__all__ = ["FREEZE_ALARM_KEYS", "FreezeTracker", "freeze_alarms"]

#: What each of the freeze's alarms reads. They exist only on a run with the tracker, because
#: only such a run publishes the ``ppo/`` keys they watch.
FREEZE_ALARM_KEYS: dict[str, tuple[str, ...]] = {
    "actor_handoff": ("ppo/kl", "ppo/clip_fraction", "ppo/iterations_since_unfreeze"),
    "critic_unready": ("ppo/ev_at_unfreeze",),
}


def freeze_alarms(
    *, handoff_window: int, handoff_kl: float, handoff_clip: float, ev_at_unfreeze: float
) -> list[Alarm]:
    """The two alarms about letting the actor go. Both warn: a halted treatment run would be
    censored out of the comparison it is part of."""
    from ..metrics.alarms import MetricAlarm

    return [
        MetricAlarm(
            "actor_handoff",
            lambda kl, clip, since: since <= handoff_window
            and (kl > handoff_kl or clip > handoff_clip),
            keys=FREEZE_ALARM_KEYS["actor_handoff"],
            meaning=(
                "the first unfrozen iterations are moving the policy fast; the backoff acts on "
                "its own, and this says why"
            ),
        ),
        MetricAlarm(
            "critic_unready",
            lambda ev: ev < ev_at_unfreeze,
            keys=FREEZE_ALARM_KEYS["critic_unready"],
            meaning="the critic explained little of the return when the actor was unfrozen",
        ),
    ]


class FreezeTracker:
    """The freeze's state and its ``ppo/`` keys, on a run with an actor learning-rate schedule."""

    FORMAT_VERSION = 1

    def __init__(self) -> None:
        #: The first unfrozen iteration after the last frozen stretch, or None.
        self.unfrozen_at: int | None = None
        #: Whether the previous iteration was frozen. None before any iteration ran.
        self.last_frozen: bool | None = None

    def finish(
        self, *, iteration: int, scale: float, explained_variance: float
    ) -> dict[str, float]:
        """This iteration's freeze keys.

        ``ppo/ev_at_unfreeze`` is on the first unfrozen row after a frozen stretch only, and
        ``ppo/iterations_since_unfreeze`` on every unfrozen row after one.
        """
        frozen = scale == 0.0
        fields: dict[str, float] = {
            "ppo/actor_lr_scale": float(scale),
            "ppo/actor_frozen": 1.0 if frozen else 0.0,
        }
        if not frozen and self.last_frozen:
            self.unfrozen_at = int(iteration)
            fields["ppo/ev_at_unfreeze"] = float(explained_variance)
        if not frozen and self.unfrozen_at is not None:
            fields["ppo/iterations_since_unfreeze"] = float(iteration - self.unfrozen_at + 1)
        self.last_frozen = frozen
        return fields

    def state(self) -> dict[str, Any]:
        return {"unfrozen_at": self.unfrozen_at, "last_frozen": self.last_frozen}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.unfrozen_at = state.get("unfrozen_at")
        self.last_frozen = state.get("last_frozen")
