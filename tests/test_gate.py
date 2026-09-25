"""The gate, and the paired measurement it is made of.

The player here awards wins to a quota, so a comparison's observed rate is exactly the rate the
test asked for and the margin between 55.0% and 55.2% at a thousand battles -- which is the
whole margin the promotion rule runs on -- is asserted rather than sampled.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from royalelearn.api.ladder import RatingTable
from royalelearn.config import GateConfig
from royalelearn.errors import PreflightError
from royalelearn.ladder.evaluate import EvalRunner, SeedSet, bootstrap_interval, eval_seed_set
from royalelearn.ladder.gate import (
    CONDITION_ANCHORS,
    CONDITION_CHAMPION,
    CONDITION_POOL,
    WilsonGate,
    failed_condition,
    floor_decision,
)
from royalelearn.ladder.pool import SCRIPTED_NOOP, SCRIPTED_RANDOM_LEGAL, LadderPool
from royalelearn.ladder.results import ResultLog

SEED = 4242
CANDIDATE = "learner@12500000"
CHAMPION = "snap:champion"
POOL_MEMBERS = [f"snap:v{index}" for index in range(8)]


class QuotaPlayer:
    """Awards ``a`` exactly the share of battles the test asked for, in a fixed order.

    A sampled player would make every assertion about a bound a statement about a seed; this
    one makes the observed rate the number under test.
    """

    def __init__(self, rates: dict[str, float], default: float = 0.5) -> None:
        self.rates = rates
        self.default = default
        self.calls: dict[tuple[str, str], int] = {}
        self.played: list[tuple[str, str, int, int]] = []

    def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
        assert act_path.startswith("eval/match/")
        rate = self.rates.get(b, self.default)
        key = (a, b)
        index = self.calls.get(key, 0)
        self.calls[key] = index + 1
        self.played.append((a, b, seed, a_seat))
        return 1.0 if math.floor((index + 1) * rate) > math.floor(index * rate) else 0.0


class SwapPlayer:
    """Scores by side, so that pairing has something to remove."""

    def __init__(self, blue_wins: set[int], red_wins: set[int]) -> None:
        self.blue_wins = blue_wins
        self.red_wins = red_wins

    def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
        index = seed % 1000
        if a_seat == 0:
            return 1.0 if index in self.blue_wins else 0.0
        return 1.0 if index in self.red_wins else 0.0


class FlatRater:
    """A rater whose prediction is a table the test writes, so a gate condition can be aimed."""

    FORMAT_VERSION = 1

    def __init__(self, predictions: dict[str, float], default: float = 0.5) -> None:
        self.predictions = predictions
        self.default = default

    def fit(self, results):
        raise AssertionError("the gate must not refit")

    def predict(self, a: str, b: str) -> float:
        if a == CHAMPION:
            return self.predictions.get(b, self.default)
        return 1.0 - self.predictions.get(a, self.default)

    def transitivity_residual(self, results) -> float:
        return 0.0


def _runner(player, tmp_path, *, seeds: int = 600, log: ResultLog | None = None) -> EvalRunner:
    return EvalRunner(
        player,
        eval_seed_set(SEED, seeds),
        master_seed=SEED,
        context="ctx",
        run_id="run",
        log=log,
        bootstrap_resamples=200,
    )


def _pool(tmp_path) -> LadderPool:
    pool = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    pool.add(CHAMPION, step=1000)
    pool.promote(CHAMPION)
    for index, member in enumerate(POOL_MEMBERS):
        pool.add(member, step=index * 100)
    pool.note_refit(
        RatingTable(
            rating={member: index * 20.0 for index, member in enumerate(POOL_MEMBERS)},
            se=dict.fromkeys(POOL_MEMBERS, 10.0),
            anchor=SCRIPTED_NOOP,
            draw_nu=None,
            n_games=dict.fromkeys(POOL_MEMBERS, 100),
            transitivity_residual=0.0,
            converged=True,
            iterations=3,
        )
    )
    return pool


def _gated_pool(tmp_path) -> LadderPool:
    """A pool whose champion came through a gate, so it has a record against every anchor.

    What the stopping-logic tests always assumed without saying so. Until 2026-09-24 a champion
    with no record was held to the rater's prior instead, and these tests passed on that 0.5.
    """
    pool = _pool(tmp_path)
    pool.record(
        _anchor_games(CHAMPION, SCRIPTED_NOOP, wins=50, losses=50)
        + _anchor_games(CHAMPION, SCRIPTED_RANDOM_LEGAL, wins=50, losses=50)
    )
    return pool


def _rates(champion: float, anchors: float = 1.0, members: float = 0.6) -> dict[str, float]:
    rates = {CHAMPION: champion, SCRIPTED_NOOP: anchors, SCRIPTED_RANDOM_LEGAL: anchors}
    rates.update(dict.fromkeys(POOL_MEMBERS, members))
    return rates


# -- the paired measurement --------------------------------------------------


def test_a_pairing_plays_every_seed_from_both_sides(tmp_path) -> None:
    player = QuotaPlayer(_rates(0.6))
    runner = _runner(player, tmp_path)
    comparison = runner.compare(CANDIDATE, CHAMPION, games=20)
    assert comparison.n_seeds == 10
    assert comparison.n_games == 20
    seats = {(seed, seat) for _a, _b, seed, seat in player.played}
    assert len(seats) == 20
    assert {seat for _seed, seat in seats} == {0, 1}
    assert [score.seed_index for score in comparison.seed_scores] == list(range(10))


def test_the_unit_of_analysis_is_the_seed(tmp_path) -> None:
    """A seed scores in quarters, which is what pairing buys: the side is removed exactly."""
    player = SwapPlayer(blue_wins={0, 1, 2}, red_wins={1})
    runner = _runner(player, tmp_path, seeds=8)
    comparison = runner.compare("a", "b", games=8)
    values = {score.seed_index: score.x for score in comparison.seed_scores}
    assert set(values.values()) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert comparison.score_a == pytest.approx(float(np.mean(list(values.values()))))


def test_swapping_the_two_players_mirrors_the_measurement(tmp_path) -> None:
    class Mirrored:
        def play(self, *, a, b, seed, a_seat, act_path):
            winner = "x" if (seed // 7) % 3 else "y"
            return 1.0 if a == winner else 0.0

    runner = _runner(Mirrored(), tmp_path, seeds=40)
    forward = runner.compare("x", "y", games=40)
    backward = runner.compare("y", "x", games=40)
    assert forward.score_a == pytest.approx(1.0 - backward.score_a)
    # The bootstrap is addressed by the pairing and not by its order, so the two intervals are
    # exact mirrors rather than two samples of one.
    assert forward.lo == pytest.approx(1.0 - backward.hi)
    assert forward.hi == pytest.approx(1.0 - backward.lo)


def test_the_bootstrap_covers_a_known_rate_at_the_nominal_level() -> None:
    rng = np.random.default_rng(3)
    truth = 0.6
    covered = 0
    replications = 200
    for _ in range(replications):
        sample = rng.binomial(2, truth, size=60) / 2.0
        lo, hi = bootstrap_interval(sample, rng, 400)
        covered += lo <= truth <= hi
    assert 0.88 <= covered / replications <= 1.0


def test_a_swept_rung_does_not_report_a_zero_width_interval() -> None:
    """The case a reader watches most is the one a percentile bootstrap cannot speak about.

    Resampling a sample with no spread draws the same number every time, so twenty seeds all won
    published a 95% interval of [1.000, 1.000]: the row said the policy is CERTAIN to sweep the
    rung, off twenty battles. The probe's keys are read by a person directly, and a zero-width
    interval reads as certainty the sample does not support. A Wilson bound over the same seeds
    is what a proportion at the end of its range is for.
    """
    rng = np.random.default_rng(11)

    lo, hi = bootstrap_interval([1.0] * 20, rng, 10_000)
    assert hi == 1.0
    assert 0.8 < lo < 1.0, "twenty wins out of twenty is not certainty"

    lost_lo, lost_hi = bootstrap_interval([0.0] * 20, rng, 10_000)
    assert lost_lo == 0.0
    assert 0.0 < lost_hi < 0.2

    # A sample that DOES have spread is still the bootstrap's, unchanged.
    mixed_lo, mixed_hi = bootstrap_interval([1.0] * 10 + [0.5] * 10, rng, 10_000)
    assert mixed_lo == pytest.approx(0.65, abs=0.05)
    assert mixed_hi == pytest.approx(0.85, abs=0.05)


def test_the_paired_correlation_is_reported(tmp_path) -> None:
    both = SwapPlayer(blue_wins=set(range(1000)), red_wins=set(range(0, 1000, 2)))
    comparison = _runner(both, tmp_path, seeds=40).compare("a", "b", games=40)
    # Every seed goes the same way on blue, so there is nothing to correlate. This used to be
    # reported as 0.0, which does not say "undefined": it says "uncorrelated", and it is the
    # number hog26-10 published in every row. None is the answer that cannot be misread.
    assert comparison.rho is None


def test_a_correlation_that_exists_is_a_number(tmp_path) -> None:
    """The control: undefined is only for the degenerate case."""
    both = SwapPlayer(
        blue_wins=set(range(0, 1000, 2)) | set(range(1, 1000, 6)),
        red_wins=set(range(0, 1000, 2)),
    )
    comparison = _runner(both, tmp_path, seeds=40).compare("a", "b", games=40)
    assert comparison.rho is not None and -1.0 <= comparison.rho <= 1.0


def test_the_champion_condition_carries_its_own_correlation(tmp_path) -> None:
    """``ladder/paired_rho`` claims to set the GATE's effective sample size, which is the champion
    comparison's. It used to read whichever comparison the runner made last -- the eighth
    stratified member -- so the champion's is now kept on the condition that played it."""
    pool = _gated_pool(tmp_path)
    rates = _rates(0.7, members=0.9)
    decision = _gate(tmp_path).evaluate(CANDIDATE, pool, _runner(QuotaPlayer(rates), tmp_path))

    # An independent replay of the champion comparison alone: same player, same frozen seeds.
    alone = _runner(QuotaPlayer(rates), tmp_path).compare(CANDIDATE, CHAMPION, games=1000)
    # And the comparison a gate plays LAST, which is what used to be read.
    last = _runner(QuotaPlayer(rates), tmp_path).compare(CANDIDATE, POOL_MEMBERS[-1], games=100)

    assert alone.rho is not None, "the fixture must give the champion comparison a defined rho"
    assert decision.conditions[CONDITION_CHAMPION].rho == pytest.approx(alone.rho)
    assert last.rho != alone.rho, "the fixture cannot tell the champion's from the last one"


def test_a_pairing_of_two_different_observations_is_refused(tmp_path) -> None:
    digests = {"a": "1111", "b": "2222"}
    runner = EvalRunner(
        QuotaPlayer({}),
        eval_seed_set(SEED, 10),
        master_seed=SEED,
        obs_digest=digests.get,
    )
    with pytest.raises(PreflightError) as excinfo:
        runner.compare("a", "b", games=4)
    message = str(excinfo.value)
    assert "a" in message and "b" in message
    assert "1111" in message and "2222" in message


def test_every_battle_reaches_the_log_tagged_eval(tmp_path) -> None:
    log = ResultLog(tmp_path / "games.jsonl")
    runner = _runner(QuotaPlayer(_rates(0.6)), tmp_path, seeds=10, log=log)
    runner.compare(CANDIDATE, CHAMPION, games=20)
    log.close()
    games = log.read()
    assert len(games) == 20
    assert {game.kind for game in games} == {"eval"}
    assert {game.side_a for game in games} == {"blue", "red"}
    assert runner.games_played == 20


def test_a_comparison_needs_seeds_it_has(tmp_path) -> None:
    runner = _runner(QuotaPlayer({}), tmp_path, seeds=4)
    with pytest.raises(ValueError, match="frozen set holds"):
        runner.compare("a", "b", games=20)


def test_the_seed_set_is_frozen_and_hashed() -> None:
    first = eval_seed_set(SEED, 32)
    assert first == eval_seed_set(SEED, 32)
    assert first.sha() == eval_seed_set(SEED, 32).sha()
    assert first.sha() != eval_seed_set(SEED + 1, 32).sha()
    assert len(SeedSet(seeds=(1, 2, 3))) == 3


# -- the gate ----------------------------------------------------------------


def _gate(tmp_path, rater=None, **overrides) -> WilsonGate:
    settings = {
        "champion_games": 1000,
        "anchor_games": 200,
        "stratified_snapshots": 8,
        "stratified_games": 100,
    }
    settings.update(overrides)
    config = GateConfig(**settings)
    return WilsonGate(config, rater or FlatRater({}), gates_dir=tmp_path / "gates")


def test_an_observed_552_passes_and_550_fails(tmp_path) -> None:
    for rate, expected in ((0.552, True), (0.550, False)):
        pool = _pool(tmp_path / str(rate))
        gate = _gate(tmp_path / str(rate))
        runner = _runner(QuotaPlayer(_rates(rate)), tmp_path)
        decision = gate.evaluate(CANDIDATE, pool, runner)
        condition = decision.conditions[CONDITION_CHAMPION]
        assert condition.n == 1000
        assert condition.observed == pytest.approx(rate, abs=0.001)
        assert condition.passed is expected
        assert decision.admit is expected


def test_each_condition_fails_on_its_own_and_the_decision_names_it(tmp_path) -> None:
    pool = _pool(tmp_path / "beaten")
    decision = _gate(tmp_path / "beaten").evaluate(
        CANDIDATE, pool, _runner(QuotaPlayer(_rates(0.50)), tmp_path)
    )
    assert failed_condition(decision) == CONDITION_CHAMPION
    assert not decision.admit and not decision.promote and not decision.cycle

    pool = _pool(tmp_path / "anchor")
    # The champion's own record against the anchors is what the candidate is held to.
    pool.record(
        [
            game
            for game in _anchor_games(CHAMPION, SCRIPTED_NOOP, wins=100, losses=0)
            + _anchor_games(CHAMPION, SCRIPTED_RANDOM_LEGAL, wins=90, losses=10)
        ]
    )
    decision = _gate(tmp_path / "anchor").evaluate(
        CANDIDATE, pool, _runner(QuotaPlayer(_rates(0.60, anchors=0.5)), tmp_path)
    )
    assert failed_condition(decision).startswith("anchors:")
    assert not decision.admit

    pool = _pool(tmp_path / "collapse")
    rater = FlatRater(dict.fromkeys(POOL_MEMBERS, 0.9))
    decision = _gate(tmp_path / "collapse", rater).evaluate(
        CANDIDATE, pool, _runner(QuotaPlayer(_rates(0.60, members=0.30)), tmp_path)
    )
    assert failed_condition(decision) == CONDITION_POOL
    assert decision.admit and not decision.promote and decision.cycle


def test_all_three_conditions_passing_promotes(tmp_path) -> None:
    pool = _pool(tmp_path)
    rater = FlatRater(dict.fromkeys(POOL_MEMBERS, 0.5))
    decision = _gate(tmp_path, rater).evaluate(
        CANDIDATE, pool, _runner(QuotaPlayer(_rates(0.60)), tmp_path)
    )
    assert decision.admit and decision.promote and not decision.cycle
    assert failed_condition(decision) == "none"
    assert decision.eval_seed_set_sha == eval_seed_set(SEED, 600).sha()
    assert (tmp_path / "gates" / "learner-12500000.json").exists()


def test_condition_three_uses_the_champions_fit_and_not_its_record(tmp_path) -> None:
    """The champion has never played these eight, and the condition is still defined."""
    pool = _pool(tmp_path)
    assert pool.eval_view().record(CHAMPION, POOL_MEMBERS[0]).games == 0
    rater = FlatRater(dict.fromkeys(POOL_MEMBERS, 0.55))
    decision = _gate(tmp_path, rater).evaluate(
        CANDIDATE, pool, _runner(QuotaPlayer(_rates(0.60, members=0.62)), tmp_path)
    )
    condition = decision.conditions[CONDITION_POOL]
    assert condition.reference == pytest.approx(0.55)
    assert condition.n == 800
    assert condition.passed


def test_the_stratified_sample_spans_the_rating_range(tmp_path) -> None:
    pool = _pool(tmp_path)
    gate = _gate(tmp_path, stratified_snapshots=4)
    chosen = gate.stratified_sample(CHAMPION, pool)
    assert len(chosen) == 4
    assert CHAMPION not in chosen
    ratings = pool.ratings
    assert ratings is not None
    picked = sorted(ratings.rating[member] for member in chosen)
    everything = sorted(ratings.rating[member] for member in POOL_MEMBERS)
    assert picked[0] <= everything[1]
    assert picked[-1] >= everything[-2]


def test_the_first_candidate_of_a_run_is_admitted_unopposed(tmp_path) -> None:
    pool = LadderPool(ResultLog(tmp_path / "games.jsonl"), context="ctx")
    decision = _gate(tmp_path).evaluate(
        CANDIDATE, pool, _runner(QuotaPlayer({}), tmp_path)
    )
    assert decision.admit and decision.promote
    assert decision.conditions == {}
    pool.apply(decision, step=12_500_000)
    assert pool.champion == CANDIDATE
    assert pool.v0 == CANDIDATE


def test_a_floor_admission_never_promotes() -> None:
    decision = floor_decision(CANDIDATE, CHAMPION)
    assert decision.admit and not decision.promote and not decision.cycle


def test_the_pool_counts_what_the_gate_decided(tmp_path) -> None:
    pool = _pool(tmp_path)
    gate = _gate(tmp_path)
    failing = gate.evaluate(CANDIDATE, pool, _runner(QuotaPlayer(_rates(0.50)), tmp_path))
    pool.apply(failing, step=12_500_000)
    assert pool.state.consecutive_gate_failures == 1
    assert pool.state.gate_attempts == 1
    assert pool.champion == CHAMPION
    passing = gate.evaluate(CANDIDATE, pool, _runner(QuotaPlayer(_rates(0.60)), tmp_path))
    pool.apply(passing, step=12_500_000)
    assert pool.state.consecutive_gate_failures == 0
    assert pool.state.gate_passes == 1
    assert pool.champion == CANDIDATE
    assert CANDIDATE in pool.sampler()


def _anchor_games(a: str, b: str, *, wins: int, losses: int):
    from royalelearn.ladder.results import GameResult

    def game(score: float, index: int) -> GameResult:
        return GameResult(
            a=a,
            b=b,
            score_a=score,
            seed_index=index,
            side_a="blue",
            context="ctx",
            kind="eval",
            run_id="run",
            iteration=0,
            wall="",
        )

    return [game(1.0, index) for index in range(wins)] + [
        game(0.0, wins + index) for index in range(losses)
    ]


def test_a_candidate_that_is_already_the_champion_is_not_played_against_itself(tmp_path) -> None:
    """The snapshot store is content-addressed, so a policy that has not moved keeps its id.

    When it does, the champion and the candidate are one snapshot and there is nothing to
    decide. Asking anyway sends a battle of a snapshot against itself to the result log, which
    refuses it several frames further down -- a correct guard reporting a caller's mistake, and
    the first thing a stranger running `verify-resume` on the shipped smoke config met.
    """

    real = _runner(QuotaPlayer(_rates(0.6)), tmp_path)

    class Runner:
        """A runner that fails if the gate asks it to play anything."""

        seeds = real.seeds

        def compare(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("the gate evaluated a snapshot against itself")

        def against_scripted(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("the gate evaluated a snapshot against itself")

    pool = _pool(tmp_path)
    pool.promote(CHAMPION)
    decision = _gate(tmp_path).evaluate(CHAMPION, pool, Runner())

    assert decision.candidate == decision.champion == CHAMPION
    assert decision.admit and decision.promote and not decision.cycle
    assert decision.conditions == {}, "there is nothing to decide, so nothing is recorded"


# -- what a settled decision still pays for ----------------------------------


def _battles(player: QuotaPlayer) -> int:
    return len(player.played)


def test_a_candidate_that_loses_the_champion_stops_there(tmp_path) -> None:
    """2,200 battles, no short-circuit, and 1,200 of them after the answer is known.

    ``admit`` is ``beats and anchors_held`` and ``promote`` is ``admit and no_collapse``, so once
    the champion condition has failed neither the anchors nor the pool-collapse check can move
    any of the three outcomes. Measured 2026-09-23: an evaluation battle is 8.02 s with real
    networks, so those 1,200 battles are 2.7 hours per gate spent on a decision already made,
    and a 500-iteration run was going to pay it six times.
    """
    pool = _pool(tmp_path)
    player = QuotaPlayer(_rates(0.40))
    decision = _gate(tmp_path).evaluate(CANDIDATE, pool, _runner(player, tmp_path))

    assert not decision.admit and not decision.promote and not decision.cycle
    assert _battles(player) == 1000, (
        "the champion comparison is 1000 battles; anything more was spent after the decision"
    )


def test_a_skipped_condition_says_so_rather_than_being_absent(tmp_path) -> None:
    """A condition nobody measured is not a condition that failed, and not one that passed.

    The recorded conditions are the evidence for why a candidate was refused, and a reader has
    to be able to tell "it also lost to the anchors" from "nobody asked". Absence would read as
    the gate not having that condition at all.
    """
    pool = _pool(tmp_path)
    decision = _gate(tmp_path).evaluate(
        CANDIDATE, pool, _runner(QuotaPlayer(_rates(0.40)), tmp_path)
    )

    champion = decision.conditions[CONDITION_CHAMPION]
    assert not champion.passed and not champion.skipped and champion.n == 1000

    others = {
        name: result for name, result in decision.conditions.items() if name != CONDITION_CHAMPION
    }
    assert others, "the skipped conditions are still named"
    for name, result in others.items():
        assert result.skipped, f"{name} was not marked as skipped"
        assert result.n == 0, f"{name} claims to have measured {result.n} battles"
        assert not result.passed, f"{name} reads as having passed without being played"


def test_a_candidate_that_beats_everything_still_plays_everything(tmp_path) -> None:
    """The other half. Without it, "stop early" would pass by never playing anything."""
    pool = _gated_pool(tmp_path)
    player = QuotaPlayer(_rates(0.7))
    decision = _gate(tmp_path).evaluate(CANDIDATE, pool, _runner(player, tmp_path))

    assert decision.admit
    assert _battles(player) == 1000 + 2 * 200 + 8 * 100
    assert not any(result.skipped for result in decision.conditions.values())


def test_a_failed_anchor_stops_before_the_pool(tmp_path) -> None:
    """Beating the champion is not enough: a candidate that lost an anchor cannot be admitted.

    So the pool-collapse check, which is 800 of the 2,200, is spent on a settled decision too.
    """
    pool = _gated_pool(tmp_path)
    player = QuotaPlayer(_rates(0.7, anchors=0.0))
    decision = _gate(tmp_path).evaluate(CANDIDATE, pool, _runner(player, tmp_path))

    assert not decision.admit
    assert _battles(player) == 1000 + 2 * 200
    assert decision.conditions[CONDITION_POOL].skipped


def test_stopping_early_does_not_change_any_decision(tmp_path) -> None:
    """The property the saving rests on, over the whole grid rather than one case.

    For every combination of champion, anchor and pool outcomes, the three flags are what the
    unconditional gate produced. ``admit = beats and anchors_held`` and
    ``promote = admit and no_collapse`` are the definitions this is checking against, so a
    future condition that does NOT factor this way fails here rather than quietly costing
    hours or quietly changing an admission.
    """
    for champion_rate, beats in ((0.7, True), (0.4, False)):
        for anchor_rate, anchors_held in ((1.0, True), (0.0, False)):
            for member_rate in (0.9, 0.1):
                name = f"{champion_rate}-{anchor_rate}-{member_rate}"
                pool = _gated_pool(tmp_path / name)
                player = QuotaPlayer(
                    _rates(champion_rate, anchors=anchor_rate, members=member_rate)
                )
                decision = _gate(tmp_path / name).evaluate(
                    CANDIDATE, pool, _runner(player, tmp_path / name)
                )
                assert decision.admit == (beats and anchors_held), name
                assert decision.promote == (decision.admit and not decision.cycle), name
                if not decision.admit:
                    assert not decision.promote and not decision.cycle, name


# -- what a gate costs, before anybody pays it -------------------------------


def test_the_gate_says_how_many_battles_it_plays() -> None:
    """The number nobody had before 2026-09-23, when it turned out to be 29 hours.

    A gate's cost is four config fields multiplied together and nothing printed it, so the
    shipped cadence quietly implied six gates of 2,200 battles in a 500-iteration run. At a
    measured 8.02 s a battle that is 29 hours of evaluation against 4 hours of training.
    """
    from royalelearn.ladder.gate import gate_battles

    config = GateConfig(
        champion_games=1000, anchor_games=200, stratified_snapshots=8, stratified_games=100
    )
    assert gate_battles(config, anchors=2) == 1000 + 2 * 200 + 8 * 100 == 2200
    assert gate_battles(config, anchors=0) == 1000 + 800


def test_the_cost_it_reports_is_the_cost_it_pays(tmp_path) -> None:
    """The count against a real gate that plays everything, so the two cannot drift apart.

    A number computed from config in one place and paid in another is exactly the pairing that
    goes stale, and this is the whole point of publishing it.
    """
    from royalelearn.ladder.gate import gate_battles

    # A champion that came through a gate: the full price is the nominal one.
    pool = _gated_pool(tmp_path / "gated")
    player = QuotaPlayer(_rates(0.7))
    gate = _gate(tmp_path / "gated")
    gate.evaluate(CANDIDATE, pool, _runner(player, tmp_path))
    assert len(player.played) == gate_battles(gate.config, anchors=len(gate.anchors))

    # A champion admitted free, never measured: it plays each anchor once, and the report says so.
    pool = _pool(tmp_path / "free")
    player = QuotaPlayer(_rates(0.7))
    gate = _gate(tmp_path / "free")
    gate.evaluate(CANDIDATE, pool, _runner(player, tmp_path))
    assert len(player.played) == gate_battles(
        gate.config, anchors=len(gate.anchors), unmeasured_champion_anchors=len(gate.anchors)
    )


def test_the_preflight_says_which_side_of_the_boundary_a_run_falls_on() -> None:
    """The cadence is a boundary between two economics, and nothing printed it.

    Train's run was capped at 2,700,000 steps rather than 4,000,000 for exactly this reason:
    493 iterations is one candidate and the free admission, 730 is two and the second is a full
    gate. Nothing in the config says so, so preflight says it.
    """
    import pathlib

    import royalelearn.config as cfg
    from royalelearn.config import geometry
    from royalelearn.ladder.gate import gate_battles
    from royalelearn.ladder.pool import SCRIPTED_IDS
    from royalelearn.rollout.preflight import _ladder_cost_line

    config = cfg.load_config(
        pathlib.Path("examples/configs/train-hog26-8.json").read_bytes()
    )
    line = _ladder_cost_line(config, geometry(config))

    battles = gate_battles(config.ladder.gate, anchors=len(SCRIPTED_IDS))
    assert str(battles) in line, line
    assert "FIRST into an empty pool is free" in line
    assert f"every {config.ladder.probe_every_iterations} iterations" in line

    env_per_iteration = geometry(config).cycles * geometry(config).n_battles
    cadence = config.ladder.candidate_every_env_steps // env_per_iteration
    assert f"every {cadence} iterations" in line


def test_a_run_that_never_gates_says_that_instead() -> None:
    """A cadence of zero is off, and a line reporting hours for it would be a lie."""
    import msgspec

    import royalelearn.config as cfg
    from royalelearn.config import geometry
    from royalelearn.rollout.preflight import _ladder_cost_line

    config = cfg.RunConfig()
    config = msgspec.structs.replace(
        config, ladder=msgspec.structs.replace(config.ladder, candidate_every_env_steps=0)
    )
    line = _ladder_cost_line(config, geometry(config))
    assert "no candidates" in line
    assert "h at" not in line


# -- the two conditions that could not fail ------------------------------------------------


class PairPlayer:
    """Scores by (a, b), so the champion and a candidate can score differently on one anchor.

    ``QuotaPlayer`` keys by the opponent alone, which makes a champion's score against an anchor
    the candidate's by construction -- and a test built on it could not tell a measured reference
    from an assumed one.
    """

    def __init__(self, rates: dict[tuple[str, str], float], default: float = 0.5) -> None:
        self.rates = rates
        self.default = default
        self.calls: dict[tuple[str, str], int] = {}

    def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
        rate = self.rates.get((a, b), self.default)
        index = self.calls.get((a, b), 0)
        self.calls[(a, b)] = index + 1
        return 1.0 if math.floor((index + 1) * rate) > math.floor(index * rate) else 0.0


def _pair_rates(candidate_vs_anchor: float, champion_vs_anchor: float) -> dict:
    rates = {(CANDIDATE, CHAMPION): 0.60}
    for anchor in (SCRIPTED_NOOP, SCRIPTED_RANDOM_LEGAL):
        rates[(CANDIDATE, anchor)] = candidate_vs_anchor
        rates[(CHAMPION, anchor)] = champion_vs_anchor
    for member in POOL_MEMBERS:
        rates[(CANDIDATE, member)] = 0.6
    return rates


def test_an_unmeasured_champion_is_measured_against_the_anchor_not_assumed(tmp_path) -> None:
    """hog26-10's first real gate: a champion admitted free, never rated, never played an anchor.

    Its reference fell back to the rater's prior, 0.5, so the bound was 0.48 and any policy
    cleared it. Here the champion really scores 0.95 against the anchors and the candidate 0.70:
    measured, that is a regression; assumed, it was a pass.
    """
    pool = _pool(tmp_path)
    assert pool.eval_view().record(CHAMPION, SCRIPTED_NOOP).games == 0
    player = PairPlayer(_pair_rates(candidate_vs_anchor=0.70, champion_vs_anchor=0.95))
    # The pool's own log, as a real run wires it, so what the gate measures is where it looks.
    runner = _runner(player, tmp_path, log=pool.results)
    decision = _gate(tmp_path).evaluate(CANDIDATE, pool, runner)

    condition = decision.conditions[f"{CONDITION_ANCHORS}:{SCRIPTED_NOOP}"]
    assert condition.reference == pytest.approx(0.95, abs=0.01), "the reference was assumed"
    assert not condition.passed
    assert not decision.admit
    # measured once, and kept: the next gate reads the record instead of playing it again
    assert pool.eval_view().record(CHAMPION, SCRIPTED_NOOP).games == 200


def test_a_champion_with_a_record_is_not_played_against_the_anchor_again(tmp_path) -> None:
    pool = _pool(tmp_path)
    pool.record(
        _anchor_games(CHAMPION, SCRIPTED_NOOP, wins=95, losses=5)
        + _anchor_games(CHAMPION, SCRIPTED_RANDOM_LEGAL, wins=95, losses=5)
    )
    player = PairPlayer(_pair_rates(candidate_vs_anchor=0.99, champion_vs_anchor=0.95))
    _gate(tmp_path).evaluate(CANDIDATE, pool, _runner(player, tmp_path))
    assert (CHAMPION, SCRIPTED_NOOP) not in player.calls
    assert (CHAMPION, SCRIPTED_RANDOM_LEGAL) not in player.calls


def test_an_empty_pool_skips_the_collapse_check_and_says_why(tmp_path) -> None:
    """Nothing besides the champion and the anchors: there is nothing to have collapsed against.

    It used to record passed=True with n=0 -- a pass with no population, which is how "1,400 of
    2,200 battles" read as a full gate. Promotion still rests on the two conditions that could be
    measured; the record now says the third could not be.
    """
    pool = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    pool.add(CHAMPION, step=1000)
    pool.promote(CHAMPION)
    pool.record(
        _anchor_games(CHAMPION, SCRIPTED_NOOP, wins=50, losses=50)
        + _anchor_games(CHAMPION, SCRIPTED_RANDOM_LEGAL, wins=50, losses=50)
    )
    player = PairPlayer(_pair_rates(candidate_vs_anchor=0.9, champion_vs_anchor=0.5))
    decision = _gate(tmp_path).evaluate(CANDIDATE, pool, _runner(player, tmp_path))

    collapse = decision.conditions[CONDITION_POOL]
    assert collapse.skipped and not collapse.passed and collapse.n == 0
    assert "no" in collapse.reason and "pool" in collapse.reason, collapse.reason
    assert decision.admit and decision.promote and not decision.cycle
    assert failed_condition(decision) == "none", "a promoted candidate named a failed condition"


# -- the champion's step -------------------------------------------------------------------


def test_an_admitted_candidate_is_filed_at_the_step_it_was_taken(tmp_path) -> None:
    """``apply`` used to look the candidate's step up in the registry it was about to be added to.

    A new candidate is never there yet, so every admission was filed at step 0 and
    ``ladder/champion_step`` read 0 in every row of hog26-10. The caller knows the step -- it took
    the snapshot -- so it now has to say it, and there is no fallback that could guess.
    """
    pool = LadderPool(ResultLog(tmp_path / "games.jsonl"), context="ctx")
    decision = _gate(tmp_path).evaluate(CANDIDATE, pool, _runner(QuotaPlayer({}), tmp_path))
    pool.apply(decision, step=12_500_000)
    assert pool.champion == CANDIDATE
    assert pool.step_of(pool.champion) == 12_500_000


def test_apply_cannot_be_called_without_the_step() -> None:
    """The shape that made the bug possible is gone: the step is not optional."""
    import inspect

    parameter = inspect.signature(LadderPool.apply).parameters["step"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
