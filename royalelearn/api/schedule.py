"""Quantities that move with the run: the discount, the entropy coefficients, the learning
rates.

A schedule is a pure function of ``cumulative_env_steps``, so its position is restored by
restoring that counter rather than by replaying the run. ``ScheduleState`` is the set of values
one iteration was run at: it goes into the checkpoint, into the metric row and into the update,
so that what a number was at iteration ``k`` is a lookup rather than a recomputation from a
config that may since have been edited.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import msgspec

__all__ = ["Schedule", "ScheduleState"]


class Schedule(ABC):
    """One scheduled scalar."""

    @abstractmethod
    def value(self, env_steps: int) -> float: ...


class ScheduleState(msgspec.Struct, frozen=True):
    """Every scheduled quantity, evaluated once at the top of an iteration.

    Evaluated once and passed down, rather than evaluated where each is used: the discount the
    reward was computed with, the discount GAE used and the discount the metric row reports are
    then the same number by construction.
    """

    iteration: int
    cumulative_env_steps: int
    cumulative_timesteps: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    ent_coef_noop: float
    lr_actor: float
    lr_critic: float
    lr_backoff_events: int = 0
    #: The scheduled actor learning-rate scale at this clock, from the section that schedules it
    #: (``Extension.actor_lr_scale``): multiplies ``lr_actor``, and zero freezes the actor for the
    #: iteration (section 19.5). One on every run whose sections schedule none.
    actor_lr_scale: float = 1.0

    def credit_horizon_seconds(self, decision_ms: int) -> float:
        """``1 / (1 - gamma * lambda)`` decisions, in seconds.

        ``decision_ms`` comes from the environment, so the horizon follows a change in the
        decision granularity instead of quietly meaning something else. Logged every iteration,
        because it is the single highest-leverage number in the config and because it can
        otherwise be ten seconds by accident.
        """
        decisions = 1.0 / max(1e-12, 1.0 - self.gamma * self.gae_lambda)
        return decisions * decision_ms / 1000.0
