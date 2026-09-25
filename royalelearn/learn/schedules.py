"""The quantities that move with the run, and the one that moves in response to it.

A schedule is a pure function of ``cumulative_env_steps``. Its position is therefore restored by
restoring that counter, not by replaying the run, and an iteration's values are a lookup rather
than a recomputation against a config that may since have been edited. Every scheduled quantity is
evaluated once at the top of an iteration into a ``ScheduleState``, which goes into the update,
into the metric row and into the checkpoint, so the discount the reward was computed with, the
discount GAE used and the discount the row reports are the same number by construction.

The learning rate is the exception: it is not a function of the step count but of what the run
did, and ``LrBackoff`` is the whole of that rule -- four lines that prevent the class of blow-up
that costs a run, in place of computing a KL and acting on nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from ..api.schedule import Schedule, ScheduleState
from ..config import (
    ConstantSpec,
    GeometricSpec,
    LinearSpec,
    LrBackoffConfig,
    PiecewiseConstantSpec,
    ScheduleSpec,
)
from ..errors import CheckpointFormatError

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..config import RunConfig

__all__ = [
    "BackoffState",
    "Constant",
    "Geometric",
    "Linear",
    "LrBackoff",
    "PiecewiseConstant",
    "ScheduleSet",
    "build_schedule",
]


def _progress(env_steps: int, over_env_steps: int) -> float:
    """How far along an anneal ``env_steps`` is, in [0, 1].

    A non-positive span is a schedule that is already at its endpoint, which is how a config
    turns an anneal off without a second kind of schedule.
    """
    if over_env_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, env_steps / over_env_steps))


class Constant(Schedule):
    """A value that does not move."""

    def __init__(self, value: float) -> None:
        self.value_ = float(value)

    def value(self, env_steps: int) -> float:
        return self.value_


class Linear(Schedule):
    """``start`` to ``end`` in equal steps over ``over_env_steps``, then ``end`` forever."""

    def __init__(self, start: float, end: float, over_env_steps: int) -> None:
        self.start = float(start)
        self.end = float(end)
        self.over_env_steps = int(over_env_steps)

    def value(self, env_steps: int) -> float:
        p = _progress(env_steps, self.over_env_steps)
        return self.start + (self.end - self.start) * p


class Geometric(Schedule):
    """``start`` to ``end`` with the distance to one decaying by a constant factor.

    This is what a discount wants. Moving gamma itself in equal steps spends most of the anneal
    where it changes the horizon least: the horizon is ``1 / (1 - gamma)``, so what has to move
    geometrically is ``1 - gamma``. From 0.997 to 0.999 the horizon doubles at a constant rate
    rather than crawling and then leaping.

    It is therefore for discount-like quantities only -- a discount, a GAE lambda, anything
    whose interesting scale is its distance from one. On a coefficient that anneals toward zero,
    such as an entropy bonus, the curvature is the wrong way round and ``Linear`` is what that
    wants. Both endpoints must be below one, which every discount and every lambda is.
    """

    def __init__(self, start: float, end: float, over_env_steps: int) -> None:
        for name, value in (("start", start), ("end", end)):
            if value >= 1.0:
                raise ValueError(
                    f"a geometric schedule anneals the distance to one, so {name}={value} must "
                    "be below one"
                )
        self.start = float(start)
        self.end = float(end)
        self.over_env_steps = int(over_env_steps)

    def value(self, env_steps: int) -> float:
        p = _progress(env_steps, self.over_env_steps)
        # Both gaps are strictly positive: the constructor refuses an endpoint at or above one.
        gap_start = 1.0 - self.start
        gap_end = 1.0 - self.end
        return 1.0 - gap_start * (gap_end / gap_start) ** p


class PiecewiseConstant(Schedule):
    """``(env_steps, value)`` breakpoints, held between them.

    Before the first breakpoint the first value holds, so a schedule that starts at a non-zero
    step is a step up rather than an undefined region.
    """

    def __init__(self, points: list[tuple[int, float]]) -> None:
        if not points:
            raise ValueError("a piecewise schedule needs at least one breakpoint")
        ordered = sorted((int(step), float(value)) for step, value in points)
        self.points = tuple(ordered)

    def value(self, env_steps: int) -> float:
        current = self.points[0][1]
        for step, value in self.points:
            if env_steps < step:
                break
            current = value
        return current


def build_schedule(spec: ScheduleSpec) -> Schedule:
    """The schedule one config entry describes."""
    if isinstance(spec, ConstantSpec):
        return Constant(spec.value)
    if isinstance(spec, LinearSpec):
        return Linear(spec.start, spec.end, spec.over_env_steps)
    if isinstance(spec, GeometricSpec):
        return Geometric(spec.start, spec.end, spec.over_env_steps)
    if isinstance(spec, PiecewiseConstantSpec):
        return PiecewiseConstant(spec.points)
    raise TypeError(f"{type(spec).__name__} is not a schedule specification")


class BackoffState(msgspec.Struct):
    """What the backoff carries across a checkpoint: the rates it has reached and the counter."""

    lr_actor: float
    lr_critic: float
    consecutive_breaches: int = 0
    events: int = 0


class LrBackoff:
    """Drop both learning rates when the KL stays above its threshold.

    The counter is consecutive: one quiet iteration resets it, because what this guards against is
    a policy walking away from its behaviour distribution and not a single noisy update. Both
    rates move together and by the same factor -- the actor is what breached, and letting the
    critic run on at an unchanged rate against a slowed actor changes the balance between them for
    a reason nobody chose.
    """

    FORMAT_VERSION = 1

    #: The file the rates and the counter are checkpointed in.
    STATE_FILE = "lr_backoff.json"

    def __init__(self, config: LrBackoffConfig, *, lr_actor: float, lr_critic: float) -> None:
        self.config = config
        self._state = BackoffState(lr_actor=float(lr_actor), lr_critic=float(lr_critic))

    @property
    def lr_actor(self) -> float:
        return self._state.lr_actor

    @property
    def lr_critic(self) -> float:
        return self._state.lr_critic

    @property
    def events(self) -> int:
        return self._state.events

    @property
    def consecutive_breaches(self) -> int:
        return self._state.consecutive_breaches

    def observe(self, kl: float) -> bool:
        """Record one iteration's KL and return whether the rates were just dropped."""
        state = self._state
        if kl > self.config.kl_threshold:
            state.consecutive_breaches += 1
        else:
            state.consecutive_breaches = 0
        if state.consecutive_breaches < self.config.patience:
            return False
        floor = self.config.lr_min
        state.lr_actor = max(floor, state.lr_actor * self.config.factor)
        state.lr_critic = max(floor, state.lr_critic * self.config.factor)
        state.consecutive_breaches = 0
        state.events += 1
        return True

    def state(self) -> BackoffState:
        return msgspec.structs.replace(self._state)

    def load_state(self, state: BackoffState) -> None:
        self._state = msgspec.structs.replace(state)

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": self.FORMAT_VERSION, "state": self._state}
        (folder / self.STATE_FILE).write_bytes(msgspec.json.encode(payload))

    def load_checkpoint(self, folder: Path, *, strict: bool) -> None:
        path = folder / self.STATE_FILE
        if not path.exists():
            if strict:
                raise CheckpointFormatError(f"the learning-rate backoff's state is not at {path}")
            print(f"no learning-rate backoff state at {path}; keeping the configured rates")
            return
        payload: dict[str, Any] = msgspec.json.decode(path.read_bytes())
        version = int(payload.get("format_version", 0))
        if version > self.FORMAT_VERSION:
            raise CheckpointFormatError(
                f"{path} was written at backoff format {version} and this build reads "
                f"{self.FORMAT_VERSION}"
            )
        self._state = msgspec.convert(payload["state"], BackoffState)


class ScheduleSet:
    """Every scheduled quantity of a run, evaluated together.

    Evaluated once at the top of an iteration and passed down, rather than evaluated where each is
    used: two evaluations at different step counts inside one iteration would be two different
    discounts, and the metric row would report whichever was asked for last.
    """

    def __init__(
        self,
        *,
        gamma: Schedule,
        ent_coef: Schedule,
        ent_coef_noop: Schedule,
        gae_lambda: float,
        backoff: LrBackoff,
        actor_lr_scale: Schedule | None = None,
    ) -> None:
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.ent_coef_noop = ent_coef_noop
        self.gae_lambda = float(gae_lambda)
        self.backoff = backoff
        self.actor_lr_scale = actor_lr_scale if actor_lr_scale is not None else Constant(1.0)

    @classmethod
    def from_config(cls, config: RunConfig) -> ScheduleSet:
        return cls(
            gamma=build_schedule(config.advantage.gamma),
            ent_coef=build_schedule(config.ppo.ent_coef),
            ent_coef_noop=build_schedule(config.ppo.ent_coef_noop),
            gae_lambda=config.advantage.gae_lambda,
            backoff=LrBackoff(
                config.ppo.lr_backoff,
                lr_actor=config.ppo.lr_actor,
                lr_critic=config.ppo.lr_critic,
            ),
            actor_lr_scale=(
                build_schedule(config.imitation.actor_lr_scale)
                if config.imitation is not None and config.imitation.actor_lr_scale is not None
                else None
            ),
        )

    def state(
        self, *, iteration: int, cumulative_env_steps: int, cumulative_timesteps: int
    ) -> ScheduleState:
        """The values this iteration runs at."""
        steps = cumulative_env_steps
        return ScheduleState(
            iteration=iteration,
            cumulative_env_steps=steps,
            cumulative_timesteps=cumulative_timesteps,
            gamma=self.gamma.value(steps),
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef.value(steps),
            ent_coef_noop=self.ent_coef_noop.value(steps),
            lr_actor=self.backoff.lr_actor,
            lr_critic=self.backoff.lr_critic,
            lr_backoff_events=self.backoff.events,
            actor_lr_scale=self.actor_lr_scale.value(steps),
        )
