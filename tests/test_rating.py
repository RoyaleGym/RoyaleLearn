"""The rating is the number every decision in the ladder is made on, so it is tested as one.

The synthetic results here are drawn from the model the rater fits, which is what makes
"recovers the truth within its own standard errors" a statement about the implementation rather
than about the model. The two properties that are not statistical -- the fit is a pure function
of the results, and the anchor is pinned exactly -- are asserted to the last bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from royalelearn.ladder.rating import (
    ELO_SCALE,
    BradleyTerryDavidsonRater,
    EloReadout,
    elo_of_score,
    wilson_interval,
)
from royalelearn.ladder.results import GameResult, ResultView

ANCHOR = "scripted:noop"


def _game(a: str, b: str, score_a: float, index: int = 0) -> GameResult:
    return GameResult(
        a=a,
        b=b,
        score_a=score_a,
        seed_index=index,
        side_a="blue",
        context="ctx",
        kind="eval",
        run_id="run",
        iteration=0,
        wall="",
    )


def synthetic(
    true: dict[str, float], n: int, seed: int, nu: float = 0.0
) -> list[GameResult]:
    """``n`` battles between every pair, drawn from the Davidson model at ``true``."""
    rng = np.random.default_rng(seed)
    names = sorted(true)
    games: list[GameResult] = []
    for position, a in enumerate(names):
        for b in names[position + 1 :]:
            qa = math.exp(ELO_SCALE * true[a])
            qb = math.exp(ELO_SCALE * true[b])
            tie = nu * math.sqrt(qa * qb)
            total = qa + qb + tie
            wins_a, draws, wins_b = rng.multinomial(n, [qa / total, tie / total, qb / total])
            index = 0
            for score, count in ((1.0, wins_a), (0.5, draws), (0.0, wins_b)):
                for _ in range(int(count)):
                    games.append(_game(a, b, score, index))
                    index += 1
    return games


TRUE = {ANCHOR: 0.0, "alpha": 120.0, "beta": 260.0, "gamma": -180.0}

#: A population the size of a young pool. Eight players is enough that the fit has more to go
#: on than one opponent each, which is the regime the ladder actually runs in.
POPULATION = {
    ANCHOR: 0.0,
    "alpha": 120.0,
    "beta": 260.0,
    "gamma": -180.0,
    "delta": 60.0,
    "epsilon": -60.0,
    "zeta": 200.0,
    "eta": -250.0,
}


def test_recovers_known_strengths_at_two_thousand_battles_a_pair() -> None:
    errors: list[float] = []
    for seed in range(5):
        table = BradleyTerryDavidsonRater().fit(
            ResultView(synthetic(POPULATION, 2000, seed, nu=0.4))
        )
        assert table.converged
        assert table.iterations < 10
        for player, strength in POPULATION.items():
            if player == ANCHOR:
                continue
            errors.append(abs(table.rating[player] - strength))
            assert errors[-1] < 3.0 * table.se[player]
    assert sum(errors) / len(errors) < 5.0


def test_estimates_lie_inside_their_own_intervals() -> None:
    """The interval is the fit's own, so its coverage is a statement about this code."""
    covered = 0
    trials = 0
    for seed in range(40):
        table = BradleyTerryDavidsonRater().fit(ResultView(synthetic(TRUE, 300, seed, nu=0.4)))
        for player, strength in TRUE.items():
            if player == ANCHOR:
                continue
            trials += 1
            covered += abs(table.rating[player] - strength) <= 1.96 * table.se[player]
    assert covered / trials > 0.85


def test_the_fit_is_a_pure_function_of_the_results() -> None:
    games = synthetic(TRUE, 400, 3, nu=0.3)
    first = BradleyTerryDavidsonRater().fit(ResultView(games))
    shuffled = list(games)
    np.random.default_rng(7).shuffle(shuffled)
    second = BradleyTerryDavidsonRater().fit(ResultView(shuffled))
    assert first.rating == second.rating
    assert first.se == second.se
    assert first.draw_nu == second.draw_nu


def test_the_fit_does_not_depend_on_what_the_players_are_called() -> None:
    renaming = {ANCHOR: ANCHOR, "alpha": "zulu", "beta": "alpha", "gamma": "mike"}
    games = synthetic(TRUE, 400, 5, nu=0.3)
    plain = BradleyTerryDavidsonRater().fit(ResultView(games))
    renamed = BradleyTerryDavidsonRater().fit(
        ResultView([_game(renaming[g.a], renaming[g.b], g.score_a, g.seed_index) for g in games])
    )
    for original, alias in renaming.items():
        assert renamed.rating[alias] == pytest.approx(plain.rating[original], abs=1e-9)
        assert renamed.se[alias] == pytest.approx(plain.se[original], abs=1e-9)


def test_standard_errors_shrink_like_one_over_root_n() -> None:
    errors = {}
    for n in (250, 1000, 4000):
        table = BradleyTerryDavidsonRater().fit(ResultView(synthetic(TRUE, n, 2, nu=0.4)))
        errors[n] = table.se["alpha"]
    assert errors[1000] / errors[250] == pytest.approx(0.5, rel=0.15)
    assert errors[4000] / errors[1000] == pytest.approx(0.5, rel=0.15)


def test_the_anchor_is_pinned_exactly() -> None:
    table = BradleyTerryDavidsonRater(anchor=ANCHOR).fit(
        ResultView(synthetic(TRUE, 200, 4, nu=0.2))
    )
    assert table.anchor == ANCHOR
    assert table.rating[ANCHOR] == 0.0
    assert table.se[ANCHOR] == 0.0


def test_the_anchor_is_in_the_table_even_with_no_games() -> None:
    table = BradleyTerryDavidsonRater(anchor=ANCHOR).fit(
        ResultView([_game("alpha", "beta", 1.0, i) for i in range(40)])
    )
    assert table.rating[ANCHOR] == 0.0
    assert table.n_games[ANCHOR] == 0


def test_a_difference_uses_the_two_by_two_block() -> None:
    rater = BradleyTerryDavidsonRater()
    rater.fit(ResultView(synthetic(TRUE, 1500, 6, nu=0.4)))
    table = rater.table
    assert table is not None
    quadrature = math.hypot(table.se["alpha"], table.se["beta"])
    assert rater.difference_se("alpha", "beta") < quadrature
    # Against the pinned anchor there is no shared uncertainty to remove, so the difference is
    # the marginal exactly -- which is the cleanest statement that the block is being used.
    assert rater.difference_se("alpha", ANCHOR) == pytest.approx(table.se["alpha"], rel=1e-12)


def test_the_davidson_term_recovers_a_known_draw_rate() -> None:
    table = BradleyTerryDavidsonRater().fit(ResultView(synthetic(TRUE, 3000, 8, nu=0.75)))
    assert table.draw_nu is not None
    assert table.draw_nu == pytest.approx(0.75, rel=0.1)


def test_too_few_draws_falls_back_to_half_a_win() -> None:
    games = synthetic(TRUE, 500, 9, nu=0.0)
    view = ResultView(games)
    assert view.draw_rate() < 0.02
    table = BradleyTerryDavidsonRater().fit(view)
    assert table.draw_nu is None
    assert abs(table.rating["beta"] - TRUE["beta"]) < 15.0


def test_the_residual_is_near_zero_on_transitive_results() -> None:
    table = BradleyTerryDavidsonRater().fit(ResultView(synthetic(TRUE, 400, 12, nu=0.3)))
    assert table.transitivity_residual == 0.0


def test_the_residual_is_large_on_rock_paper_scissors() -> None:
    games = [
        _game(a, b, 1.0, index)
        for a, b in (("x", "y"), ("y", "z"), ("z", "x"))
        for index in range(100)
    ]
    rater = BradleyTerryDavidsonRater()
    table = rater.fit(ResultView(games))
    assert table.transitivity_residual == 1.0
    # A scalar rating cannot separate them at all, which is the point the number is making.
    assert table.rating["x"] == pytest.approx(table.rating["y"], abs=1.0)


def test_pairs_with_too_few_games_do_not_count_towards_the_residual() -> None:
    rater = BradleyTerryDavidsonRater(min_pair_games=30)
    games = [_game("x", "y", 1.0, index) for index in range(10)]
    games += [_game("y", "x", 1.0, index) for index in range(10)]
    assert rater.fit(ResultView(games)).transitivity_residual == 0.0


def test_wilson_matches_the_published_table() -> None:
    assert wilson_interval(0.550, 1000)[0] == pytest.approx(0.519, abs=0.0005)
    assert wilson_interval(0.552, 1000)[0] == pytest.approx(0.521, abs=0.0005)
    assert wilson_interval(0.575, 1000)[0] == pytest.approx(0.544, abs=0.0005)
    assert wilson_interval(0.550, 400)[0] == pytest.approx(0.501, abs=0.0005)
    assert wilson_interval(0.550, 1000)[0] < 0.52 <= wilson_interval(0.552, 1000)[0]


def test_score_rate_and_elo_agree_with_the_table() -> None:
    assert elo_of_score(0.55) == pytest.approx(35.0, abs=0.5)
    assert elo_of_score(0.75) == pytest.approx(191.0, abs=0.5)


def test_predict_is_the_model_and_counts_a_draw_as_half() -> None:
    rater = BradleyTerryDavidsonRater()
    rater.fit(ResultView(synthetic(TRUE, 1000, 13, nu=0.5)))
    table = rater.table
    assert table is not None and table.draw_nu is not None
    q_beta = 10.0 ** (table.rating["beta"] / 400.0)
    q_gamma = 10.0 ** (table.rating["gamma"] / 400.0)
    tie = table.draw_nu * math.sqrt(q_beta * q_gamma)
    expected = (q_beta + 0.5 * tie) / (q_beta + q_gamma + tie)
    assert rater.predict("beta", "gamma") == pytest.approx(expected, abs=1e-9)
    assert rater.predict("beta", "gamma") + rater.predict("gamma", "beta") == pytest.approx(1.0)


def test_an_unrated_player_reads_at_the_prior_mean() -> None:
    """Nothing is assumed about a player with no games; the anchor's own rating is the answer
    the prior gives before any evidence."""
    rater = BradleyTerryDavidsonRater()
    rater.fit(ResultView(synthetic(TRUE, 200, 15, nu=0.2)))
    assert rater.predict("never-played", ANCHOR) == pytest.approx(0.5)


def test_the_rater_round_trips_through_a_checkpoint(tmp_path) -> None:
    rater = BradleyTerryDavidsonRater()
    table = rater.fit(ResultView(synthetic(TRUE, 300, 14, nu=0.3)))
    rater.save_checkpoint(tmp_path)
    loaded = BradleyTerryDavidsonRater()
    loaded.load_checkpoint(tmp_path, strict=True)
    assert loaded.table == table
    assert loaded.predict("alpha", "beta") == pytest.approx(rater.predict("alpha", "beta"))


def test_the_elo_readout_is_zero_sum_and_round_trips(tmp_path) -> None:
    elo = EloReadout()
    before = elo.rating("alpha") + elo.rating("beta")
    elo.update("alpha", "beta", 1.0)
    assert elo.rating("alpha") + elo.rating("beta") == pytest.approx(before)
    assert elo.rating("alpha") > elo.rating("beta")
    elo.save_checkpoint(tmp_path)
    loaded = EloReadout()
    loaded.load_checkpoint(tmp_path, strict=True)
    assert loaded.ratings() == elo.ratings()
    assert loaded.games("alpha") == 1
