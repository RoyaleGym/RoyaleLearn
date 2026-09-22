"""The mixture, and the property that makes a rerun a rerun.

An assignment is addressed by the battle and its reset ordinal, so the same episode of the same
battle meets the same opponent whatever the iteration length and whatever the worker count.
That is asserted directly here, by drawing the same assignment from matchmakers configured for
different geometries.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalelearn.api.rollout import (
    GROUP_LEARNER,
    GROUP_SCRIPTED,
    ROLE_MIRROR,
    ROLE_POOL,
    ROLE_SCRIPTED,
    EpisodeRecord,
)
from royalelearn.config import LadderConfig, RolloutConfig, RunConfig, geometry
from royalelearn.ladder.matchmaker import MixMatchmaker, pfsp_shape
from royalelearn.ladder.pool import SCRIPTED_IDS, LadderPool
from royalelearn.ladder.results import ResultLog

SEED = 20260922


@pytest.fixture
def pool(tmp_path) -> LadderPool:
    ladder = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    for index in range(6):
        ladder.add(f"snap:{index}", step=index * 1000)
    ladder.promote("snap:5")
    return ladder


def _record(battle: int, ordinal: int, seat: int = 0) -> EpisodeRecord:
    return EpisodeRecord(
        slot=battle * 2 + seat,
        worker=0,
        shard=0,
        battle=battle,
        seat=seat,
        ordinal=ordinal,
        episode_seed_path="env/worker/0/shard/0/gen/0",
        policy_id="learner",
        opponent_id="",
        bucket="mirror",
        episode_steps=10,
        episode_ticks=100,
        own_crowns=1,
        enemy_crowns=0,
        own_tower_hp_frac=1.0,
        enemy_tower_hp_frac=0.5,
        elixir_leak_steps=0,
        elixir_count_exact=True,
        winner=0,
        outcome=1,
        cards_played=4,
        illegal_commands=0,
        undiscounted_return=1.0,
        reward_terms={},
    )


def test_the_empirical_mixture_is_the_configured_one(pool: LadderPool) -> None:
    matchmaker = MixMatchmaker(SEED, LadderConfig())
    counts = {ROLE_MIRROR: 0, ROLE_POOL: 0, ROLE_SCRIPTED: 0}
    draws = 12_000
    for draw in range(draws):
        counts[matchmaker.assign(draw % 97, draw // 97, pool, None).role] += 1
    mirror, pool_share, scripted = LadderConfig().mix
    assert counts[ROLE_MIRROR] / draws == pytest.approx(mirror, abs=0.015)
    assert counts[ROLE_POOL] / draws == pytest.approx(pool_share, abs=0.015)
    assert counts[ROLE_SCRIPTED] / draws == pytest.approx(scripted, abs=0.015)


def test_a_pool_that_moves_inside_an_iteration_is_refused(pool: LadderPool) -> None:
    """A pool assignment names its opponent by its position in the plan's resident table.

    So an admission, an eviction or a refit landing between the plan and the assignments drawn
    against it would not fail -- it would file the episode under another snapshot's name. The
    matchmaker holds the epoch it planned at and says so instead.
    """
    from royalelearn.config import RolloutConfig, RunConfig, geometry

    matchmaker = MixMatchmaker(SEED, LadderConfig())
    geo = geometry(RunConfig(rollout=RolloutConfig(workers=1, games_per_worker=2)))
    matchmaker.plan(0, pool, None, geo)
    matchmaker.assign(0, matchmaker.ordinal(0), pool, None)
    pool.add("snap:new", step=99_000)
    with pytest.raises(RuntimeError, match="residency epoch"):
        matchmaker.assign(0, matchmaker.ordinal(0), pool, None)
    # And the next plan is what makes it drawable again.
    matchmaker.plan(1, pool, None, geo)
    matchmaker.assign(0, matchmaker.ordinal(0), pool, None)


def test_three_quarters_of_the_rows_are_kept_by_construction() -> None:
    matchmaker = MixMatchmaker(SEED, LadderConfig())
    assert matchmaker.expected_learner_row_fraction == pytest.approx(0.75)
    assert matchmaker.expected_discarded_rows_frac == pytest.approx(0.25)


def test_an_assignment_is_a_pure_function_of_its_arguments(pool: LadderPool) -> None:
    matchmaker = MixMatchmaker(SEED, LadderConfig())
    for battle, ordinal in ((0, 0), (3, 17), (11, 4)):
        first = matchmaker.assign(battle, ordinal, pool, None)
        assert matchmaker.assign(battle, ordinal, pool, None) == first
        assert MixMatchmaker(SEED, LadderConfig()).assign(battle, ordinal, pool, None) == first
    other = MixMatchmaker(SEED + 1, LadderConfig())
    table = [matchmaker.assign(battle, 0, pool, None) for battle in range(32)]
    assert [other.assign(battle, 0, pool, None) for battle in range(32)] != table


def test_the_same_episode_meets_the_same_opponent_on_any_geometry(pool: LadderPool) -> None:
    """A different worker count and a different iteration length are the same run."""
    wide = MixMatchmaker(SEED, LadderConfig())
    narrow = MixMatchmaker(SEED, LadderConfig())
    for battle in range(8):
        for ordinal in range(3):
            assert wide.assign(battle, ordinal, pool, None) == narrow.assign(
                battle, ordinal, pool, None
            )


def test_plan_is_assign_across_the_geometry(pool: LadderPool) -> None:
    config = RunConfig(rollout=RolloutConfig(workers=2, games_per_worker=4, shards_per_worker=2))
    shape = geometry(config)
    matchmaker = MixMatchmaker(SEED, config.ladder)
    plan = matchmaker.plan(iteration=3, pool=pool, ratings=None, geometry=shape)
    assert plan.n_battles == shape.n_battles
    assert plan.n_slots == shape.n_slots
    assert len(plan.assignment) == shape.n_battles
    for battle, assignment in enumerate(plan.assignment):
        assert assignment == matchmaker.assign(battle, matchmaker.ordinal(battle), pool, None)
    # The iteration is not part of the address, so the same table comes out at any iteration.
    assert matchmaker.plan(99, pool, None, shape).assignment == plan.assignment


def test_a_plan_names_at_most_the_resident_opponents(pool: LadderPool) -> None:
    config = LadderConfig(max_resident_opponents=2)
    matchmaker = MixMatchmaker(SEED, config)
    residents = matchmaker.residents(pool, None)
    assert len(residents) == config.max_resident_opponents
    named = {
        assignment.opponent_id
        for battle in range(64)
        for assignment in [matchmaker.assign(battle, 0, pool, None)]
        if assignment.role == ROLE_POOL
    }
    assert named <= set(residents)


def test_the_resident_set_moves_when_the_pool_does(pool: LadderPool) -> None:
    matchmaker = MixMatchmaker(SEED, LadderConfig(max_resident_opponents=2))
    before = matchmaker.residents(pool, None)
    epoch = pool.residency_epoch
    assert matchmaker.residents(pool, None) == before
    for index in range(6, 40):
        pool.add(f"snap:{index}", step=index * 1000)
    assert pool.residency_epoch > epoch
    assert matchmaker.residents(pool, None) != before


def test_every_seat_is_filled_and_the_learner_takes_each_side(pool: LadderPool) -> None:
    matchmaker = MixMatchmaker(SEED, LadderConfig())
    seats: dict[int, int] = {0: 0, 1: 0}
    for battle in range(400):
        assignment = matchmaker.assign(battle, 0, pool, None)
        if assignment.role == ROLE_MIRROR:
            assert assignment.group == (GROUP_LEARNER, GROUP_LEARNER)
            assert assignment.opponent_id is None
            continue
        seats[assignment.learner_seat] += 1
        assert assignment.group[assignment.learner_seat] == GROUP_LEARNER
        other = assignment.group[1 - assignment.learner_seat]
        if assignment.role == ROLE_SCRIPTED:
            assert other == GROUP_SCRIPTED
            assert assignment.opponent_id in SCRIPTED_IDS
        else:
            assert other >= 0
            assert matchmaker.residents(pool, None)[other] == assignment.opponent_id
    assert min(seats.values()) > 0.4 * sum(seats.values())


def test_an_assignment_is_binding_until_the_episode_ends(pool: LadderPool) -> None:
    matchmaker = MixMatchmaker(SEED, LadderConfig())
    assert matchmaker.ordinal(2) == 0
    first = matchmaker.assign(2, matchmaker.ordinal(2), pool, None)
    matchmaker.on_episode(_record(battle=2, ordinal=0, seat=0))
    matchmaker.on_episode(_record(battle=2, ordinal=0, seat=1))
    assert matchmaker.ordinal(2) == 1  # both seats reported one episode, not two
    # The next episode is a fresh draw; the one that just ended still answers as it did.
    assert matchmaker.assign(2, matchmaker.ordinal(2), pool, None).ordinal == 1
    assert matchmaker.assign(2, 0, pool, None) == first


def test_pool_battles_become_scripted_while_the_pool_is_empty(tmp_path) -> None:
    empty = LadderPool(ResultLog(tmp_path / "games.jsonl"))
    matchmaker = MixMatchmaker(SEED, LadderConfig())
    roles = {matchmaker.assign(battle, 0, empty, None).role for battle in range(200)}
    assert ROLE_POOL not in roles
    assert roles == {ROLE_MIRROR, ROLE_SCRIPTED}


def test_weights_are_floored_so_the_tail_stays_reachable(pool: LadderPool) -> None:
    config = LadderConfig(pfsp_uniform_floor=0.2, weight_floor_scale=0.01)
    matchmaker = MixMatchmaker(SEED, config)
    from royalelearn.api.ladder import RatingTable

    members = list(pool.sampler())
    # One member the learner beats out of sight: hard weighting alone would never draw it.
    ratings = RatingTable(
        rating={"learner": 1500.0, **{member: 1400.0 for member in members}, members[0]: -2000.0},
        se=dict.fromkeys([*members, "learner"], 10.0),
        anchor="scripted:noop",
        draw_nu=None,
        n_games=dict.fromkeys([*members, "learner"], 100),
        transitivity_residual=0.0,
        converged=True,
        iterations=4,
    )
    weights = matchmaker.weights(members, ratings, "hard")
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= config.weight_floor_scale / len(members) * 0.99
    assert weights[members.index(members[0])] > 0.0


def test_the_pfsp_shapes_are_the_documented_ones() -> None:
    p = np.array([0.1, 0.5, 0.9])
    assert pfsp_shape(p, "hard", 2.0) == pytest.approx((1 - p) ** 2)
    assert pfsp_shape(p, "variance", 2.0) == pytest.approx(p * (1 - p))
    assert pfsp_shape(p, "linear", 2.0) == pytest.approx(1 - p)
    # An opponent the learner never beats leaves "variance" with nothing; uniform is the honest
    # fallback rather than a division by zero.
    assert pfsp_shape(np.zeros(3), "variance", 2.0) == pytest.approx(np.ones(3))
    with pytest.raises(ValueError, match="unknown PFSP weighting"):
        pfsp_shape(p, "nonsense", 2.0)


def test_the_matchmaker_round_trips_through_a_checkpoint(pool: LadderPool, tmp_path) -> None:
    matchmaker = MixMatchmaker(SEED, LadderConfig())
    for battle in range(4):
        for ordinal in range(battle):
            matchmaker.on_episode(_record(battle=battle, ordinal=ordinal))
    matchmaker.save_checkpoint(tmp_path / "matchmaker")
    loaded = MixMatchmaker(SEED, LadderConfig())
    loaded.load_checkpoint(tmp_path / "matchmaker", strict=True)
    assert [loaded.ordinal(battle) for battle in range(4)] == [
        matchmaker.ordinal(battle) for battle in range(4)
    ]
