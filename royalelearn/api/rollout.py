"""Where experience comes from: the environment's description, the iteration's plan, and the
one seam between the learner and the world.

Nothing here holds a layout constant. ``EnvSpec`` is read from a running environment at
preflight -- every width, plane name, field offset and timing number -- because a width typed
into this repository would turn a RoyaleGym change into a silent wrong answer instead of a loud
one, and because the whole test suite then runs on ``MockEngine``, whose catalogue is not the
one a real run uses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import msgspec
import numpy as np

from ..rollout.envspec import EnvFactorySpec

if TYPE_CHECKING:  # pragma: no cover - the concrete buffer lives outside api/ and needs torch
    from .buffer import ExperienceBuffer

__all__ = [
    "BUCKETS",
    "EPISODE_END_DRAW",
    "EPISODE_END_LOSS",
    "EPISODE_END_NONE",
    "EPISODE_END_WIN",
    "GROUP_DEAD",
    "GROUP_LEARNER",
    "GROUP_SCRIPTED",
    "ROLE_MIRROR",
    "ROLE_POOL",
    "ROLE_SCRIPTED",
    "Assignment",
    "Close",
    "Defer",
    "EnvSpec",
    "EpisodeRecord",
    "ObsKeySpec",
    "Plan",
    "RolloutRound",
    "RolloutSource",
    "SetState",
    "SlotPlan",
    "Spaces",
    "Step",
    "WorkerCommand",
    "WorkerFailure",
]

#: What a battle is playing. The role decides the shape of the episode; the per-seat ``group``
#: below decides who supplies each seat's action.
ROLE_MIRROR = 0
ROLE_POOL = 1
ROLE_SCRIPTED = 2

#: A seat's controller, as carried per slot on every round. Non-negative values index the plan's
#: resident snapshots, so a batched forward per distinct policy is a group-by on this column.
GROUP_LEARNER = -1
GROUP_SCRIPTED = -2
GROUP_DEAD = -3

#: The rating context an episode is filed under, one per role.
BUCKETS: tuple[str, ...] = ("mirror", "pool", "scripted")

#: ``episode_end`` on a round, from the ending seat's own view.
EPISODE_END_NONE = 0
EPISODE_END_WIN = 1
EPISODE_END_LOSS = 2
EPISODE_END_DRAW = 3


class ObsKeySpec(msgspec.Struct, frozen=True):
    """One key of the environment's observation space, read from the space itself.

    ``low`` and ``high`` are per channel for a three-dimensional key and length one otherwise.
    A space that declares scalar bounds broadcasts them over the whole array, so the per-channel
    form is a reduction of the declared bounds over each plane rather than a second opinion about
    them: it is what the codec needs in order to decide a plane's storage (section 7.2).
    """

    shape: tuple[int, ...]
    dtype: str
    low: tuple[float, ...]
    high: tuple[float, ...]


class EnvSpec(msgspec.Struct, frozen=True):
    """Everything the learner must know about the environment before it builds anything.

    Every field here is READ from the running environment at preflight. The harness holds no
    layout constant of its own: a width, a plane count and a field offset are all environment
    facts, and typing one into this repo would make a RoyaleGym change a silent wrong answer
    instead of a loud one.
    """

    num_cards: int
    obs_space: dict[str, ObsKeySpec]
    vector_layout: tuple[tuple[str, int, int], ...]
    spatial_layout: tuple[tuple[str, bool], ...]
    frame_stack: int
    vector_size: int
    spatial_shape: tuple[int, int, int]
    n_actions: int
    hand_size: int
    tiles: tuple[int, int]
    decision_ms: int
    tick_ms: int
    decision_ticks: int
    regular_ticks: int
    overtime_ticks: int
    engine_build_digest: str
    obs_digest: str
    env_factory: EnvFactorySpec

    def __post_init__(self) -> None:
        # vector_size and spatial_shape are conveniences derived from obs_space, there so that
        # the common expressions read well. The space is the authority, and a disagreement
        # between the two is a construction-time assertion rather than a puzzle later.
        vector = self.obs_space["vector"]
        if (self.vector_size,) != vector.shape:
            raise ValueError(
                f"vector_size {self.vector_size} is not obs_space['vector'] {vector.shape}"
            )
        spatial = self.obs_space["spatial"]
        if self.spatial_shape != spatial.shape:
            raise ValueError(
                f"spatial_shape {self.spatial_shape} is not obs_space['spatial'] {spatial.shape}"
            )
        if self.spatial_shape[1:] != self.tiles:
            raise ValueError(
                f"spatial planes are {self.spatial_shape[1:]} tiles, tiles says {self.tiles}"
            )
        if len(self.spatial_layout) != self.spatial_shape[0]:
            raise ValueError(
                f"spatial_layout declares {len(self.spatial_layout)} planes, the space has "
                f"{self.spatial_shape[0]}"
            )
        covered = sum(size for _, _, size in self.vector_layout)
        if covered != self.vector_size:
            raise ValueError(f"vector_layout covers {covered} slots of {self.vector_size}")

    @property
    def n_planes(self) -> int:
        """Spatial planes the environment emits, static ones included."""
        return self.spatial_shape[0]

    @property
    def static_planes(self) -> tuple[str, ...]:
        """The planes the layout DECLARES static -- never the ones a sample found constant."""
        return tuple(name for name, static in self.spatial_layout if static)


class Assignment(msgspec.Struct, frozen=True):
    """What one battle's next episode is.

    Drawn by the ``Matchmaker`` at the battle's own episode boundary, from
    ``match/battle/{b}/ordinal/{k}``, and therefore a pure function of the master seed, the
    battle index and the reset ordinal.
    """

    battle: int
    ordinal: int
    role: int
    opponent_id: str | None
    group: tuple[int, int]
    learner_seat: int


class SlotPlan(msgspec.Struct, frozen=True):
    """The iteration's opening assignment table: what every battle is playing at the moment the
    iteration begins.

    Assignments change only at a battle's own episode boundary, where the parent draws a fresh
    one; nothing in an iteration's span reassigns a battle mid-episode, so a partially controlled
    trajectory cannot occur.
    """

    iteration: int
    n_battles: int
    n_slots: int
    assignment: tuple[Assignment, ...]
    resident_snapshots: tuple[str, ...]


@dataclass(slots=True)
class RolloutRound:
    """One shard-round of observations. Holds zero-copy views; not a Struct.

    The observations themselves are not here: the worker packed them straight into their final
    resting place in the experience buffer, and ``obs_rows`` says where each slot's row went.
    What crosses the boundary is the scalars and the episodes that ended.

    ``deploy_status`` is -1 for no command and 0 for an accepted one; 1..11 is a DeployStatus
    refusal, which under a correct mask cannot happen and is therefore a mask bug rather than a
    tolerance.
    """

    cycle: int
    shard: int
    slots: np.ndarray
    obs_rows: np.ndarray
    group: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    valid: np.ndarray
    deploy_status: np.ndarray
    tick: np.ndarray
    episode_end: np.ndarray
    episodes: list[EpisodeRecord] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


class EpisodeRecord(msgspec.Struct, frozen=True):
    """One seat's finished episode: what it was, how it went, and enough to replay it.

    The terminal scalars are copied straight out of the environment's ``final_info`` rather
    than reconstructed from the states the worker saw. The environment already computes them,
    and a second implementation of the same summary is a second answer. RoyaleGym's
    ``EPISODE_STAT_KEYS`` is the authority on the set.
    """

    slot: int
    worker: int
    shard: int
    battle: int
    seat: int
    ordinal: int
    episode_seed_path: str
    policy_id: str
    opponent_id: str
    bucket: str
    # the terminal scalars, copied straight out of the env's final_info
    episode_steps: int
    episode_ticks: int
    own_crowns: int
    enemy_crowns: int
    own_tower_hp_frac: float
    enemy_tower_hp_frac: float
    elixir_leak_steps: int
    # False if this seat's count of the opponent's elixir was ever caught disagreeing with the
    # bar it models, so the policy read an estimate in a slot documented as exact
    elixir_count_exact: bool
    # counted by the worker as the episode runs
    winner: int
    outcome: int
    cards_played: int
    illegal_commands: int
    undiscounted_return: float
    reward_terms: dict[str, float]


class WorkerCommand(msgspec.Struct, tag_field="kind", tag=str.upper):
    """What the parent hands a shard for one round. Six variants, one per round.

    A tagged union rather than a dict of flags: the worker's loop switches on the tag, so a
    command it does not know is a protocol failure at the boundary instead of a missing key deep
    inside a step.
    """


class Step(WorkerCommand):
    """Step this shard by one decision.

    ``group``, ``opponent_ix`` and ``learner_seat`` carry the parent's assignment for every
    battle whose episode started on the previous round, and are empty otherwise; the worker needs
    them only to know which slots it fills with a scripted action.
    """

    actions: np.ndarray
    gamma: float
    group: np.ndarray
    opponent_ix: np.ndarray
    learner_seat: np.ndarray


class Plan(WorkerCommand):
    """The iteration's opening table, once per iteration."""

    plan: SlotPlan


class SetState(WorkerCommand):
    """Start each battle from a recorded position; one blob per battle, None to leave it."""

    snapshots: tuple[bytes | None, ...]


class Spaces(WorkerCommand):
    """Re-read the spaces without stepping."""


class Defer(WorkerCommand):
    """Nothing this round; hand back the same data."""


class Close(WorkerCommand):
    """Shut the shard down."""


class WorkerFailure(msgspec.Struct, frozen=True):
    """A worker that stopped answering, as a value the coordinator can act on.

    ``kind`` is one of "exception", "crash", "timeout" or "protocol", and ``message`` is the
    child's traceback verbatim. Both reference learners treat a dead worker as a permanent silent
    hang; here it is a typed failure, which is what lets the farm restart the worker and the run
    continue with the loss recorded rather than hidden in a throughput dip.
    """

    worker: int
    shard: int
    cycle: int
    kind: str
    message: str


class RolloutSource(ABC):
    """Where experience comes from. The one seam between the learner and the world.

    Implementers may assume: ``begin_iteration`` precedes any ``next_round``; exactly one
    ``submit`` per ``next_round``; ``Step.actions`` are legal under the mask that was handed out;
    the caller does not retain a RolloutRound's views past the next ``next_round`` for the same
    shard.
    Implementers MUST guarantee: ``slots`` is ascending; every slot appears exactly once per
    cycle; a dead worker's slots arrive with ``valid=False`` rather than not arriving.
    """

    @abstractmethod
    def spec(self) -> EnvSpec: ...

    @abstractmethod
    def begin_iteration(self, plan: SlotPlan, buffer: ExperienceBuffer, iteration: int) -> None: ...

    @abstractmethod
    def next_round(self, timeout_s: float = 30.0) -> RolloutRound: ...

    @abstractmethod
    def submit(self, command: WorkerCommand) -> None: ...

    @abstractmethod
    def drain_failures(self) -> list[WorkerFailure]: ...

    @abstractmethod
    def restart(self, worker: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def stats(self) -> dict[str, float]:
        """Per-round timing: env_ms, wait_ms, codec_ms, idle_frac, bytes_out. Default: {}."""
        return {}
