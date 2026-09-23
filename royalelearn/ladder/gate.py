"""Whether a candidate snapshot joins the pool, and whether it becomes the champion.

Three conditions, all of which must hold, and each of which exists because of a specific way a
self-play pool goes wrong.

1. **It beats the champion** by enough to be sure. The Wilson 95% lower bound on the
   candidate's score rate over a thousand paired battles must clear 0.52, which in practice
   needs an observed 55.2%, about 35 Elo. A bound at 0.50 would admit anything merely *not
   worse*, and the pool would fill with lateral moves.
2. **No regression against the anchors.** This catches the policy that beats its recent
   ancestors by exploiting a blind spot they share while losing basic competence.
3. **No pool-wide collapse.** The candidate's observed mean against a stratified sample of the
   pool, against the champion's *fitted* mean over the same members. Fitted and not recorded,
   because the champion has not necessarily played those members on those seeds, and a
   condition that silently skipped the snapshots it had no games against would be weakest
   exactly where the pool is most diverse.

A candidate that passes the first two and fails the third is admitted **without** being
promoted, flagged ``cycle``: it is a useful diverse opponent and a detected cycle, not progress.
A candidate that fails the first is discarded and never retried -- its results stay in the log,
and a run of consecutive failures is exactly the plateau signal worth having.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import numpy as np

from ..api.ladder import ConditionResult, GateDecision, PromotionGate, Rater
from .pool import SCRIPTED_IDS
from .rating import wilson_interval

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..config import GateConfig
    from .evaluate import EvalRunner
    from .pool import LadderPool

__all__ = [
    "CONDITION_ANCHORS",
    "CONDITION_CHAMPION",
    "CONDITION_POOL",
    "WilsonGate",
    "failed_condition",
    "floor_decision",
    "gate_filename",
]

#: The three conditions, by the names a decision record and the metric row use.
CONDITION_CHAMPION = "beats_champion"
CONDITION_ANCHORS = "anchors"
CONDITION_POOL = "pool_collapse"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def gate_filename(candidate: str) -> str:
    """A snapshot id as a file name. Ids carry colons and at-signs; one of those is not a legal
    character in a Windows path, and this harness's default profile runs there."""
    return f"{_UNSAFE.sub('-', candidate)}.json"


def floor_decision(candidate: str, champion: str | None) -> GateDecision:
    """The unconditional admission that keeps a plateau from starving the pool.

    It admits and never promotes: the point is to keep putting fresh opponents in front of the
    learner while nothing is passing the gate, not to declare that something improved.
    """
    return GateDecision(
        candidate=candidate,
        champion=champion or "",
        admit=True,
        promote=False,
        cycle=False,
        conditions={},
        eval_seed_set_sha="",
        wall_seconds=0.0,
    )


def _skipped() -> ConditionResult:
    """A condition that was not played because the decision was already settled."""
    return ConditionResult(
        passed=False, n=0, observed=0.0, bound=0.0, reference=0.0, skipped=True
    )


class WilsonGate(PromotionGate):
    """The three conditions, evaluated in order, with the numbers that decided each."""

    def __init__(
        self,
        config: GateConfig,
        rater: Rater,
        *,
        anchors: Sequence[str] = SCRIPTED_IDS,
        gates_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.rater = rater
        self.anchors = tuple(anchors)
        self.gates_dir = Path(gates_dir) if gates_dir is not None else None

    def evaluate(
        self, candidate: str, pool: LadderPool, runner: EvalRunner
    ) -> GateDecision:
        started = time.perf_counter()
        champion = pool.champion
        if champion is None:
            # Nothing to beat. The first snapshot of a run is the pool, and everything after it
            # is measured against something.
            return self._record(
                GateDecision(
                    candidate=candidate,
                    champion="",
                    admit=True,
                    promote=True,
                    cycle=False,
                    conditions={},
                    eval_seed_set_sha=runner.seeds.sha(),
                    wall_seconds=time.perf_counter() - started,
                )
            )

        if candidate == champion:
            # The snapshot store is content-addressed, so a policy that has not moved since the
            # last candidate produces the same id -- and then the champion and the candidate are
            # one snapshot. There is nothing to decide: it is already in the pool and already
            # the champion. Asking anyway sends a battle of a snapshot against itself to the
            # result log, which refuses it, correctly, several frames further down.
            return self._record(
                GateDecision(
                    candidate=candidate,
                    champion=champion,
                    admit=True,
                    promote=True,
                    cycle=False,
                    conditions={},
                    eval_seed_set_sha=runner.seeds.sha(),
                    wall_seconds=time.perf_counter() - started,
                )
            )

        # Stop as soon as the decision is settled, and SAY which conditions were not played.
        # `admit` is `beats and anchors_held` and `promote` is `admit and no_collapse`, so a
        # failed champion condition fixes all three outcomes and everything after it is spent on
        # an answer already known. Measured 2026-09-23: an evaluation battle is 8.02 s with real
        # networks, so the 1,200 battles below the champion comparison are 2.7 hours per gate,
        # and a 500-iteration run fires six gates. Nobody had felt it because this path has never
        # executed: every gate any run has ever run was the unconditional admission of the first
        # snapshot into an empty pool.
        #
        # What is NOT done is stopping early INSIDE a comparison. The seeds are the first
        # `games // 2` of the frozen set, so a variable count means two gates played different
        # positions and stopped being comparable. That property is load-bearing and stays.
        conditions: dict[str, ConditionResult] = {}
        conditions[CONDITION_CHAMPION] = self._beats_champion(candidate, champion, runner)
        beats = conditions[CONDITION_CHAMPION].passed

        anchor_names = [f"{CONDITION_ANCHORS}:{anchor}" for anchor in self.anchors]
        if beats:
            for anchor, name in zip(self.anchors, anchor_names, strict=True):
                conditions[name] = self._anchor(candidate, champion, anchor, pool, runner)
        else:
            for name in anchor_names:
                conditions[name] = _skipped()
        anchors_held = beats and all(conditions[name].passed for name in anchor_names)

        admit = beats and anchors_held
        conditions[CONDITION_POOL] = (
            self._pool_collapse(candidate, champion, pool, runner) if admit else _skipped()
        )
        no_collapse = conditions[CONDITION_POOL].passed
        return self._record(
            GateDecision(
                candidate=candidate,
                champion=champion,
                admit=admit,
                promote=admit and no_collapse,
                cycle=admit and not no_collapse,
                conditions=conditions,
                eval_seed_set_sha=runner.seeds.sha(),
                wall_seconds=time.perf_counter() - started,
            )
        )

    # -- the conditions -----------------------------------------------------

    def _beats_champion(
        self, candidate: str, champion: str, runner: EvalRunner
    ) -> ConditionResult:
        comparison = runner.compare(candidate, champion, games=self.config.champion_games)
        lower, _upper = wilson_interval(comparison.score_a, comparison.n_games)
        return ConditionResult(
            passed=lower >= self.config.champion_lower_bound,
            n=comparison.n_games,
            observed=comparison.score_a,
            bound=lower,
            reference=self.config.champion_lower_bound,
        )

    def _anchor(
        self,
        candidate: str,
        champion: str,
        anchor: str,
        pool: LadderPool,
        runner: EvalRunner,
    ) -> ConditionResult:
        comparison = runner.compare(candidate, anchor, games=self.config.anchor_games)
        reference = self._champion_vs_anchor(champion, anchor, pool)
        bound = reference - self.config.anchor_tolerance_pp / 100.0
        return ConditionResult(
            passed=comparison.score_a >= bound,
            n=comparison.n_games,
            observed=comparison.score_a,
            bound=bound,
            reference=reference,
        )

    def _champion_vs_anchor(self, champion: str, anchor: str, pool: LadderPool) -> float:
        """What the champion scores against an anchor: its record if it has one, the fit
        otherwise. Every champion was gated against the anchors, so the record is the usual
        case and the fit covers a pool loaded from somewhere else."""
        record = pool.eval_view().record(champion, anchor)
        if record.games:
            return record.score_a
        return self.rater.predict(champion, anchor)

    def _pool_collapse(
        self, candidate: str, champion: str, pool: LadderPool, runner: EvalRunner
    ) -> ConditionResult:
        members = self.stratified_sample(champion, pool)
        if not members:
            return ConditionResult(passed=True, n=0, observed=0.0, bound=0.0, reference=0.0)
        observed: list[float] = []
        per_seed: list[float] = []
        for member in members:
            comparison = runner.compare(candidate, member, games=self.config.stratified_games)
            observed.append(comparison.score_a)
            per_seed.extend(score.x for score in comparison.seed_scores)
        values = np.array(per_seed, dtype=np.float64)
        mean = float(np.mean(observed))
        se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        reference = float(
            np.mean([self.rater.predict(champion, member) for member in members])
        )
        return ConditionResult(
            passed=mean >= reference - se,
            n=len(members) * self.config.stratified_games,
            observed=mean,
            bound=reference - se,
            reference=reference,
        )

    def stratified_sample(self, champion: str, pool: LadderPool) -> tuple[str, ...]:
        """Snapshots spanning the fitted rating range, the most even matchup from each stratum.

        Variance weighting and not ``hard``: this condition is a measurement, and a measurement
        wants the matchups that carry the most information per battle. The champion stands in
        for the candidate in the weighting because the candidate has no rating yet -- it is a
        fresh copy of the learner, and the champion is the closest rated thing to it.
        """
        ratings = pool.ratings
        excluded = {champion, *self.anchors}
        members = [member for member in pool.sampler() if member not in excluded]
        if not members:
            return ()
        wanted = min(self.config.stratified_snapshots, len(members))
        ranked = sorted(
            members,
            key=lambda member: (
                ratings.rating.get(member, 0.0) if ratings is not None else 0.0,
                member,
            ),
        )
        chosen: list[str] = []
        for stratum in range(wanted):
            low = stratum * len(ranked) // wanted
            high = max(low + 1, (stratum + 1) * len(ranked) // wanted)
            window = [member for member in ranked[low:high] if member not in chosen]
            if not window:
                continue
            chosen.append(max(window, key=lambda member: self._variance_weight(champion, member)))
        return tuple(chosen)

    def _variance_weight(self, champion: str, member: str) -> float:
        p = self.rater.predict(champion, member)
        return p * (1.0 - p)

    # -- the record ---------------------------------------------------------

    def _record(self, decision: GateDecision) -> GateDecision:
        if self.gates_dir is not None:
            self.gates_dir.mkdir(parents=True, exist_ok=True)
            path = self.gates_dir / gate_filename(decision.candidate)
            path.write_bytes(msgspec.json.encode(decision))
        return decision


def failed_condition(decision: GateDecision) -> str:
    """Which condition a decision turned on, for the metric row. ``"none"`` when all held."""
    for name, result in decision.conditions.items():
        if not result.passed:
            return name
    return "none"
