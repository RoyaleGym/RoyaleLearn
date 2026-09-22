"""The ladder as one object: who is in it, who the champion is, and where the evidence lives.

``LadderPool`` is the registry (RoyaleGym's ``OpponentPool``), the result log and the snapshot
archive held together, plus the small amount of state that is neither -- the champion chain, the
sampler's membership, and the counters a gate moves.

Two of ``OpponentPool``'s methods are deliberately never called from here.
``record_result`` keeps ``wins: float`` with a draw adding a half, which makes five wins and
five losses indistinguishable from ten draws although the two carry completely different
variance; ``results.py`` counts draws separately instead. ``_evict`` drops the *oldest*
snapshot, which is exactly the diversity the anti-forgetting floor exists to protect;
``eviction.py`` chooses by rating instead. What is used here is the part that is pure
bookkeeping over ids: the registry and its persistence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from ..api.ladder import GateDecision, RatingTable
from .results import KIND_EVAL, KIND_TRAIN, GameResult, ResultLog, ResultView

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from royalegym.selfplay import OpponentPool

    from ..api.ladder import SnapshotStore

__all__ = [
    "LEARNER_ID",
    "SCRIPTED_IDS",
    "SCRIPTED_NOOP",
    "SCRIPTED_RANDOM_LEGAL",
    "LadderPool",
    "PoolState",
]

#: The two scripted anchors. They exist before any snapshot does, which is what lets the rating
#: scale be pinned from the first game, and their score rate is the one measurement in the run
#: that never drifts: everything else is rated against a population that is itself improving.
SCRIPTED_NOOP = "scripted:noop"
SCRIPTED_RANDOM_LEGAL = "scripted:random_legal"
SCRIPTED_IDS: tuple[str, ...] = (SCRIPTED_NOOP, SCRIPTED_RANDOM_LEGAL)

#: The live policy's id in the result log and in the fit.
LEARNER_ID = "learner"


class PoolState(msgspec.Struct):
    """Everything about the ladder that is not a snapshot, a result or a rating.

    ``residency_epoch`` counts the moments at which the drawable set may legitimately change --
    a snapshot admitted, one evicted, a refit landing. The matchmaker's residency is addressed
    by it, so a battle's opponent is a pure function of the pool it was drawn against.
    """

    champion: str | None = None
    champion_chain: list[str] = msgspec.field(default_factory=list)
    v0: str | None = None
    sampled: list[str] = msgspec.field(default_factory=list)
    evicted: list[str] = msgspec.field(default_factory=list)
    residency_epoch: int = 0
    gate_attempts: int = 0
    gate_passes: int = 0
    consecutive_gate_failures: int = 0
    evictions: int = 0
    eval_games: int = 0


class LadderPool:
    """The registry, the log and the archive, and the champion chain over them."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        results: ResultLog,
        *,
        opponents: OpponentPool | None = None,
        snapshots: SnapshotStore | None = None,
        context: str = "",
        run_id: str = "",
        anchors: Sequence[str] = SCRIPTED_IDS,
    ) -> None:
        from royalegym.selfplay import OpponentPool as _OpponentPool

        self.results = results
        # max_size stays None: eviction here is a choice about which snapshots the sampler
        # draws from, and the registry's own rule would drop the oldest.
        self.opponents = opponents if opponents is not None else _OpponentPool(max_size=None)
        self.snapshots = snapshots
        self.context = context
        self.run_id = run_id
        self.anchors = tuple(anchors)
        self.state = PoolState()
        self._ratings: RatingTable | None = None
        self._view: ResultView | None = None
        self._view_size = -1
        for anchor in self.anchors:
            self.opponents.ensure(anchor, step=0)

    # -- membership ---------------------------------------------------------

    def members(self) -> tuple[str, ...]:
        """Every id the archive knows, anchors included. Nothing is ever removed from here."""
        return tuple(sorted(self.opponents.snapshots))

    def snapshot_ids(self) -> tuple[str, ...]:
        """Every archived snapshot: the members that are policies rather than scripts."""
        return tuple(member for member in self.members() if not self.is_anchor(member))

    def sampler(self) -> tuple[str, ...]:
        """What the matchmaker's pool bucket may draw, in a fixed order."""
        return tuple(self.state.sampled)

    def is_anchor(self, member: str) -> bool:
        return member in self.anchors

    def meta(self, member: str) -> dict[str, Any]:
        snapshot = self.opponents.snapshots.get(member)
        return dict(snapshot.meta) if snapshot is not None else {}

    def step_of(self, member: str) -> int:
        snapshot = self.opponents.snapshots.get(member)
        return int(snapshot.step) if snapshot is not None else 0

    @property
    def champion(self) -> str | None:
        return self.state.champion

    @property
    def champion_chain(self) -> tuple[str, ...]:
        """Every id that has ever been champion. None of them is ever evicted: the chain is the
        spine the ladder's own history is told along."""
        return tuple(self.state.champion_chain)

    @property
    def v0(self) -> str | None:
        """The run's first snapshot, which ``rating_above_v0`` is measured against."""
        return self.state.v0

    @property
    def residency_epoch(self) -> int:
        return self.state.residency_epoch

    @property
    def ratings(self) -> RatingTable | None:
        """The last fit, as the gate and the matchmaker see it."""
        return self._ratings

    def add(
        self,
        member: str,
        step: int,
        meta: Mapping[str, Any] | None = None,
        *,
        sampled: bool = True,
    ) -> None:
        """Register a snapshot. It joins the sampler unless it is being archived only."""
        if member not in self.opponents.snapshots:
            self.opponents.add(member, step=step, meta=dict(meta or {}))
        elif meta:
            self.opponents.snapshots[member].meta.update(dict(meta))
        if self.state.v0 is None and not self.is_anchor(member):
            self.state.v0 = member
        if sampled and member not in self.state.sampled:
            self.state.sampled.append(member)
            if member in self.state.evicted:
                self.state.evicted.remove(member)
        self.state.residency_epoch += 1

    def promote(self, member: str) -> None:
        """Make ``member`` the champion. The previous one stays in the pool and in the chain."""
        self.state.champion = member
        if member not in self.state.champion_chain:
            self.state.champion_chain.append(member)
        self.state.residency_epoch += 1

    def evict(self, members: Iterable[str]) -> tuple[str, ...]:
        """Remove ids FROM THE SAMPLER. The archive and the log are untouched."""
        removed: list[str] = []
        for member in members:
            if member in self.state.sampled:
                self.state.sampled.remove(member)
                self.state.evicted.append(member)
                removed.append(member)
        if removed:
            self.state.evictions += len(removed)
            self.state.residency_epoch += 1
        return tuple(removed)

    def apply(self, decision: GateDecision) -> None:
        """Act on one gate's verdict, and count it.

        Admission and promotion are separate because a pool that only ever admits champions
        forgets everything it beat: a candidate that beats the champion but collapses against
        the wider pool is a useful, diverse opponent and a detected cycle, not progress.
        """
        self.state.gate_attempts += 1
        if decision.admit:
            self.state.gate_passes += 1
            self.state.consecutive_gate_failures = 0
            meta: dict[str, Any] = {"cycle": decision.cycle}
            self.add(decision.candidate, step=self.step_of(decision.candidate), meta=meta)
            if decision.promote:
                self.promote(decision.candidate)
        else:
            self.state.consecutive_gate_failures += 1

    def note_refit(self, ratings: RatingTable) -> None:
        """Take a fresh fit. The drawable set is allowed to move here and nowhere else, so an
        assignment drawn inside an iteration cannot meet an opponent the iteration's plan did
        not carry."""
        self._ratings = ratings
        self.state.residency_epoch += 1

    # -- results ------------------------------------------------------------

    def record(self, games: Iterable[GameResult]) -> int:
        """Append results to the log. Evaluation games are counted; training games are not."""
        batch = list(games)
        written = self.results.extend(batch)
        self.state.eval_games += sum(1 for game in batch if game.kind == KIND_EVAL)
        return written

    def record_training_results(
        self, outcomes: Iterable[tuple[str, str, float]], *, iteration: int, seed_index: int = -1
    ) -> int:
        """File the training battles, tagged so the fit leaves them out.

        They are recorded because they are evidence about a pair and free to keep, and excluded
        by default because they are PFSP-selected: the mixture chooses hard matchups on purpose,
        so a rating fitted to them would be a rating of the curriculum.
        """
        return self.record(
            GameResult(
                a=a,
                b=b,
                score_a=score,
                seed_index=seed_index,
                side_a="blue",
                context=self.context,
                kind=KIND_TRAIN,
                run_id=self.run_id,
                iteration=iteration,
                wall="",
            )
            for a, b, score in outcomes
        )

    def eval_view(self) -> ResultView:
        """What the authoritative rating is fitted to: this context's evaluation games.

        Held until the log grows. A run asks for this view several times an iteration -- the
        fit, the gate's anchor reference, the metric row -- and re-reading a log of a hundred
        thousand battles each time would put a file scan on the iteration's critical path for a
        number that cannot have changed. The log is append-only, so its size is what says
        whether it has: that holds for the evaluation runner's writes as well as this object's.
        """
        size = self.results.path.stat().st_size if self.results.path.exists() else -1
        if self._view is None or size != self._view_size:
            self._view = self.results.eval_view(self.context or None)
            self._view_size = size
        return self._view

    # -- checkpoint ---------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.opponents.save(folder / "pool.json")
        (folder / "champion.json").write_bytes(msgspec.json.encode(self.state))
        (folder / "ratings.json").write_bytes(msgspec.json.encode(self._ratings))

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        from royalegym.selfplay import OpponentPool as _OpponentPool

        pool_path = folder / "pool.json"
        state_path = folder / "champion.json"
        ratings_path = folder / "ratings.json"
        for path in (pool_path, state_path):
            if not path.exists():
                if strict:
                    raise FileNotFoundError(str(path))
                print(f"no ladder state at {path}; starting the pool empty")
                return
        self._view = None
        self.opponents = _OpponentPool.load(pool_path, max_size=None)
        self.state = msgspec.json.decode(state_path.read_bytes(), type=PoolState)
        if ratings_path.exists():
            self._ratings = msgspec.json.decode(
                ratings_path.read_bytes(), type=RatingTable | None
            )
