"""Eviction takes snapshots out of the sampler and out of nothing else.

The archive, the result log and the fit are untouched by every test here: a rating means
nothing without the player it rated, and what is bounded is the cost of drawing an opponent,
not the memory of having played one.
"""

from __future__ import annotations

import pytest

from royalelearn.api.ladder import RatingTable
from royalelearn.ladder.eviction import HallOfFameEviction
from royalelearn.ladder.pool import SCRIPTED_IDS, SCRIPTED_NOOP, LadderPool
from royalelearn.ladder.results import GameResult, ResultLog


def _ratings(members: dict[str, float], games: int = 100) -> RatingTable:
    return RatingTable(
        rating=dict(members),
        se=dict.fromkeys(members, 12.0),
        anchor=SCRIPTED_NOOP,
        draw_nu=None,
        n_games=dict.fromkeys(members, games),
        transitivity_residual=0.0,
        converged=True,
        iterations=4,
    )


@pytest.fixture
def populated(tmp_path) -> tuple[LadderPool, RatingTable]:
    pool = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    strengths: dict[str, float] = {}
    for index in range(20):
        member = f"snap:v{index}"
        pool.add(member, step=index * 1000, meta={"cycle": index % 5 == 0})
        strengths[member] = index * 25.0
    for anchor in SCRIPTED_IDS:
        pool.add(anchor, step=0)
        strengths[anchor] = 0.0
    pool.promote("snap:v9")
    pool.promote("snap:v19")
    return pool, _ratings(strengths)


def test_nothing_is_evicted_while_the_sampler_fits(populated) -> None:
    pool, ratings = populated
    policy = HallOfFameEviction()
    assert policy.select_for_eviction(pool=pool, ratings=ratings, max_sampled=100) == []


def test_anchors_v0_and_the_champion_chain_are_never_evicted(populated) -> None:
    pool, ratings = populated
    evicted = HallOfFameEviction().select_for_eviction(
        pool=pool, ratings=ratings, max_sampled=8
    )
    assert evicted
    for protected in (*SCRIPTED_IDS, pool.v0, *pool.champion_chain):
        assert protected not in evicted
    assert pool.v0 == "snap:v0"
    assert pool.champion_chain == ("snap:v9", "snap:v19")


def test_what_is_kept_spans_the_rating_range(populated) -> None:
    """Each equal-count rating stratum of the members that may go keeps exactly one of them.

    Counted over the unprotected members only. The protected ones -- the anchors, v0 and the
    champion chain -- stay whatever their rating, and here v0 is the lowest-rated member and
    the last champion the highest, so a check over everything kept passed whatever the strata
    chose: the audit's plant kept the three lowest and it still passed. Nor is "the lowest kept
    is near the bottom" the property: within a stratum a detected cycle is preferred, so the one
    kept need not be its extreme.
    """
    pool, ratings = populated
    max_sampled = 8
    evicted = set(
        HallOfFameEviction().select_for_eviction(
            pool=pool, ratings=ratings, max_sampled=max_sampled
        )
    )
    kept = [member for member in pool.sampler() if member not in evicted]
    assert len(kept) == max_sampled
    protected = {*SCRIPTED_IDS, pool.v0, *pool.champion_chain}
    optional = sorted(
        (member for member in pool.sampler() if member not in protected),
        key=lambda member: (ratings.rating[member], member),
    )
    budget = max_sampled - len(protected)
    assert (len(optional), budget) == (17, 3)
    strata = [
        optional[index * len(optional) // budget : (index + 1) * len(optional) // budget]
        for index in range(budget)
    ]
    per_stratum = [sum(member in stratum for member in kept) for stratum in strata]
    assert per_stratum == [1] * budget, (
        f"kept {sorted(set(kept) - protected)}; per stratum {per_stratum} of {strata}"
    )


def test_a_detected_cycle_is_preferred_within_its_stratum(tmp_path) -> None:
    pool = LadderPool(ResultLog(tmp_path / "games.jsonl"), context="ctx")
    strengths = {}
    for index in range(6):
        member = f"snap:v{index}"
        # Pairs of equal strength, one of each flagged: the stratum has to choose between them.
        pool.add(member, step=index, meta={"cycle": index % 2 == 1})
        strengths[member] = (index // 2) * 100.0
    ratings = _ratings(strengths)
    evicted = set(
        HallOfFameEviction().select_for_eviction(pool=pool, ratings=ratings, max_sampled=4)
    )
    kept = [member for member in pool.sampler() if member not in evicted]
    # v0 is protected whatever its flag; every other survivor is a flagged one.
    assert pool.v0 in kept
    for member in kept:
        if member != pool.v0:
            assert pool.meta(member)["cycle"] is True


def test_eviction_touches_neither_the_archive_nor_the_log(populated, tmp_path) -> None:
    pool, ratings = populated
    pool.record(
        [
            GameResult(
                a="learner",
                b="snap:v3",
                score_a=1.0,
                seed_index=index,
                side_a="blue",
                context="ctx",
                kind="eval",
                run_id="run",
                iteration=0,
                wall="",
            )
            for index in range(4)
        ]
    )
    members_before = pool.members()
    games_before = len(pool.eval_view())
    evicted = HallOfFameEviction().select_for_eviction(
        pool=pool, ratings=ratings, max_sampled=8
    )
    removed = pool.evict(evicted)
    assert set(removed) == set(evicted)
    assert pool.members() == members_before
    assert len(pool.eval_view()) == games_before
    assert pool.state.evictions == len(removed)
    for member in removed:
        assert member not in pool.sampler()
        assert member in pool.members()
        assert member in pool.state.evicted


def test_a_sampler_of_protected_members_only_evicts_the_rest(tmp_path) -> None:
    pool = LadderPool(ResultLog(tmp_path / "games.jsonl"), context="ctx")
    pool.add("snap:v0", step=0)
    pool.promote("snap:v0")
    for index in range(1, 5):
        pool.add(f"snap:v{index}", step=index)
    ratings = _ratings({f"snap:v{index}": index * 10.0 for index in range(5)})
    evicted = HallOfFameEviction().select_for_eviction(
        pool=pool, ratings=ratings, max_sampled=1
    )
    assert set(evicted) == {f"snap:v{index}" for index in range(1, 5)}
