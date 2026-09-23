"""The ladder: who plays whom, what the result means, and when a snapshot is admitted.

The authoritative rating is a batch fit over an append-only result log rather than an online
filter, so the same games produce the same numbers in any order, on any machine. Online Elo
stays as a dashboard readout and is never the gate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import msgspec

from .checkpoint import Checkpointable
from .rollout import Assignment, EpisodeRecord, SlotPlan

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..config import Geometry
    from ..ladder.evaluate import EvalRunner
    from ..ladder.pool import LadderPool
    from ..ladder.results import ResultView
    from .policy import Actor, ActorCritic

__all__ = [
    "ConditionResult",
    "EvictionPolicy",
    "GateDecision",
    "Matchmaker",
    "PromotionGate",
    "Rater",
    "RatingTable",
    "SnapshotStore",
]


class RatingTable(msgspec.Struct, frozen=True):
    """One fit of the whole result log.

    Ratings are in Elo units with the anchor pinned exactly, standard errors come from the
    inverse observed Fisher information, and ``transitivity_residual`` says how much of the
    result log a single scalar per player fails to explain -- which is the number that decides
    whether a scalar rating is lying.
    """

    rating: dict[str, float]
    se: dict[str, float]
    anchor: str
    draw_nu: float | None
    n_games: dict[str, int]
    transitivity_residual: float
    converged: bool
    iterations: int


class Matchmaker(ABC, Checkpointable):
    """What every battle plays next."""

    @abstractmethod
    def plan(
        self, iteration: int, pool: LadderPool, ratings: RatingTable, geometry: Geometry
    ) -> SlotPlan:
        """The iteration's opening table. One ``Assignment`` per battle, each drawn at that
        battle's current ordinal, so ``plan()`` is ``assign()`` applied across the geometry."""

    @abstractmethod
    def assign(
        self, battle: int, ordinal: int, pool: LadderPool, ratings: RatingTable
    ) -> Assignment:
        """The assignment for one battle's next episode.

        Draws from ``match/battle/{battle}/ordinal/{ordinal}``, so it is a pure function of the
        master seed and its two arguments: the same episode of the same battle always meets the
        same opponent, whichever iteration it happens to fall in and whichever worker holds it.
        """

    @abstractmethod
    def on_episode(self, record: EpisodeRecord) -> None:
        """Record a finished episode, for the mixture's own bookkeeping."""


class Rater(ABC, Checkpointable):
    """How a pile of results becomes a number per player."""

    @abstractmethod
    def fit(self, results: ResultView) -> RatingTable:
        """id -> (rating in Elo units, standard error).

        MUST be a pure function of ``results``: same games in, same numbers out, in any order,
        on any machine.
        """

    @abstractmethod
    def predict(self, a: str, b: str) -> float:
        """P(a scores against b), with draws counted as half a win."""

    @abstractmethod
    def transitivity_residual(self, results: ResultView) -> float:
        """How badly one number per player fits: near zero on transitive results, large on
        synthetic rock-paper-scissors."""


class ConditionResult(msgspec.Struct):
    """One gate condition, with the numbers that decided it rather than a bare pass or fail."""

    passed: bool
    n: int
    observed: float
    bound: float
    reference: float
    #: True when the condition was NOT PLAYED because the decision was already settled. A
    #: condition nobody measured is not one that failed and not one that passed, and the
    #: difference is the whole evidence for why a candidate was refused: "it also lost to the
    #: anchors" and "nobody asked" are different findings. ``passed`` is False and ``n`` is 0 on
    #: a skipped condition, so anything reading either without reading this reads a refusal.
    skipped: bool = False


class GateDecision(msgspec.Struct):
    """A candidate's audition, kept whole.

    ``admit`` puts the snapshot in the pool, ``promote`` makes it the champion, and ``cycle``
    is the case where a snapshot is worth playing against without being the best: the three are
    separate because a pool that only ever admits champions forgets everything it beat.
    """

    candidate: str
    champion: str
    admit: bool
    promote: bool
    cycle: bool
    conditions: dict[str, ConditionResult]
    eval_seed_set_sha: str
    wall_seconds: float


class PromotionGate(ABC):
    """Whether a candidate snapshot joins the pool, and whether it becomes the champion."""

    @abstractmethod
    def evaluate(self, candidate: str, pool: LadderPool, runner: EvalRunner) -> GateDecision: ...


class SnapshotStore(ABC):
    """Where frozen actors live. They outlive checkpoints: a rating means nothing without the
    player it rated."""

    @abstractmethod
    def put(self, snapshot_id: str, ac: ActorCritic, meta: dict[str, Any]) -> str: ...

    @abstractmethod
    def get(self, snapshot_id: str, device: Any) -> Actor:
        """LRU-cached: a shard-round touches at most ``ladder.max_resident_opponents`` of them,
        so a cache that never thrashes is a small one."""

    @abstractmethod
    def digest(self, snapshot_id: str) -> str: ...

    @abstractmethod
    def list(self) -> list[str]: ...


class EvictionPolicy(ABC):
    """Which snapshots the sampler stops drawing."""

    @abstractmethod
    def select_for_eviction(
        self, *, pool: LadderPool, ratings: RatingTable, max_sampled: int
    ) -> list[str]:
        """Ids to remove FROM THE SAMPLER. The archive and the result log are never touched:
        eviction is about sampling cost, not about forgetting evidence."""
