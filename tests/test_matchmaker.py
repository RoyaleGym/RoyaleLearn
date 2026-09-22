"""The mixture, and the property that makes a rerun a rerun.

The role is a property of the battle slot: a fixed number of slots are mirror, a fixed number
pool and a fixed number scripted. So an iteration collects the same number of learner rows on
every cycle and the mixture is exact rather than sampled. The opponent and the learner's seat
are still drawn per episode, addressed by the battle and its reset ordinal, so the same episode
of the same battle meets the same opponent whatever the iteration length.
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
from royalelearn.config import (
    LadderConfig,
    RolloutConfig,
    RunConfig,
    geometry,
    laptop,
    role_counts,
)
from royalelearn.ladder.matchmaker import MixMatchmaker, pfsp_shape
from royalelearn.ladder.pool import SCRIPTED_IDS, LadderPool
from royalelearn.ladder.results import ResultLog

SEED = 20260922

#: The master seeds every exactness claim is made over. Thirty rather than one, because the
#: defect this file exists to keep out was invisible on the seed that happened to ship.
MASTER_SEEDS = tuple(range(30))

#: ``(workers, games_per_worker, shards_per_worker, mix)``. An odd battle count, a single
#: battle, an all-mirror mixture and shares that do not divide the battles are all here on
#: purpose: each of them is a way for a count to stop summing to the rectangle.
GEOMETRIES = (
    (3, 32, 2, (0.50, 0.35, 0.15)),
    (1, 7, 1, (0.50, 0.35, 0.15)),
    (1, 11, 1, (0.50, 0.35, 0.15)),
    (2, 5, 1, (1.0, 0.0, 0.0)),
    (1, 1, 1, (0.50, 0.35, 0.15)),
    (1, 7, 1, (0.33, 0.33, 0.34)),
    (2, 12, 4, (0.0, 0.5, 0.5)),
)


@pytest.fixture
def pool(tmp_path) -> LadderPool:
    ladder = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    for index in range(6):
        ladder.add(f"snap:{index}", step=index * 1000)
    ladder.promote("snap:5")
    return ladder


def _pool(tmp_path, name: str, members: int = 6) -> LadderPool:
    """A pool of its own, so one test can hold several at different residency epochs."""
    ladder = LadderPool(ResultLog(tmp_path / name / "games.jsonl"), context="ctx")
    for index in range(members):
        ladder.add(f"snap:{index}", step=index * 1000)
    return ladder


def _config(workers: int, games: int, shards: int, mix: tuple[float, float, float]) -> RunConfig:
    return RunConfig(
        rollout=RolloutConfig(workers=workers, games_per_worker=games, shards_per_worker=shards),
        ladder=LadderConfig(mix=mix),
    )


def _rows_this_cycle(matchmaker: MixMatchmaker, pool: LadderPool, n_battles: int) -> int:
    """The learner rows one cycle of the rectangle produces, read off the matchmaker itself.

    Two for a mirror battle -- the learner sits in both seats -- and one for every other.
    """
    rows = 0
    for battle in range(n_battles):
        assignment = matchmaker.assign(battle, matchmaker.ordinal(battle), pool, None)
        rows += 2 if assignment.role == ROLE_MIRROR else 1
    return rows


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


@pytest.mark.parametrize("workers,games,shards,mix", GEOMETRIES)
def test_the_mixture_is_a_count_of_slots_and_not_a_draw(
    tmp_path, workers: int, games: int, shards: int, mix: tuple[float, float, float]
) -> None:
    """Every role's slot count is within one battle of its share, on every seed.

    A share is a count here, not a probability, so "within one battle" is what integer
    arithmetic costs and nothing more: there is no sampling error left to be within.
    """
    shape = geometry(_config(workers, games, shards, mix))
    for seed in MASTER_SEEDS:
        ladder = _pool(tmp_path, f"count-{seed}-{workers}-{games}-{mix[0]}")
        matchmaker = MixMatchmaker(seed, LadderConfig(mix=mix), n_battles=shape.n_battles)
        counts = {ROLE_MIRROR: 0, ROLE_POOL: 0, ROLE_SCRIPTED: 0}
        for battle in range(shape.n_battles):
            counts[matchmaker.assign(battle, 0, ladder, None).role] += 1
        assert sum(counts.values()) == shape.n_battles
        for role, share in enumerate(mix):
            assert abs(counts[role] - share * shape.n_battles) <= 1.0


def test_a_role_does_not_move_with_the_ordinal_the_pool_or_the_ratings(tmp_path) -> None:
    """The count stops being exact the moment a role can change inside an iteration.

    ``assign`` is called mid-iteration, at each battle's own episode boundary, and the pool and
    the ratings move at iteration boundaries. If any of those moved a role, the rows collected
    in a cycle would not be the rows the iteration was sized for.
    """
    from royalelearn.api.ladder import RatingTable

    shape = geometry(_config(3, 32, 2, (0.50, 0.35, 0.15)))
    small = _pool(tmp_path, "small", members=3)
    large = _pool(tmp_path, "large", members=40)
    matchmaker = MixMatchmaker(SEED, LadderConfig(), n_battles=shape.n_battles)
    baseline = [matchmaker.assign(b, 0, small, None).role for b in range(shape.n_battles)]

    ratings = RatingTable(
        rating={"learner": 1500.0, **{f"snap:{i}": 1400.0 + i for i in range(40)}},
        se=dict.fromkeys(["learner", *(f"snap:{i}" for i in range(40))], 10.0),
        anchor="scripted:noop",
        draw_nu=None,
        n_games=dict.fromkeys(["learner", *(f"snap:{i}" for i in range(40))], 100),
        transitivity_residual=0.0,
        converged=True,
        iterations=4,
    )
    for ordinal in (0, 1, 7, 512):
        for ladder, table in ((small, None), (large, None), (large, ratings)):
            roles = [
                matchmaker.assign(b, ordinal, ladder, table).role
                for b in range(shape.n_battles)
            ]
            assert roles == baseline


@pytest.mark.parametrize("workers,games,shards,mix", GEOMETRIES)
def test_every_cycle_collects_the_same_learner_rows(
    tmp_path, workers: int, games: int, shards: int, mix: tuple[float, float, float]
) -> None:
    """The iteration's size is exact, cycle by cycle.

    Episodes end at different moments in different battles, so the table the rectangle is
    filling changes constantly inside an iteration. What must not change is how many of its
    rows are the learner's.
    """
    config = _config(workers, games, shards, mix)
    shape = geometry(config)
    for seed in MASTER_SEEDS:
        ladder = _pool(tmp_path, f"cycle-{seed}-{workers}-{games}-{mix[0]}")
        matchmaker = MixMatchmaker(seed, config.ladder, n_battles=shape.n_battles)
        ends = np.random.default_rng(seed).random((16, shape.n_battles)) < 0.3
        total = 0
        for cycle in range(16):
            assert _rows_this_cycle(matchmaker, ladder, shape.n_battles) == shape.learner_rows
            total += shape.learner_rows
            for battle in np.flatnonzero(ends[cycle]):
                matchmaker.on_episode(_record(battle=int(battle), ordinal=cycle))
        assert total == 16 * shape.learner_rows
        assert shape.cycles * shape.learner_rows == shape.cycles * matchmaker.learner_rows


def test_the_shipped_iteration_reaches_its_floor_on_every_master_seed(tmp_path) -> None:
    """The failure this whole design exists to remove.

    A role drawn per episode makes the mirror count Binomial(n_battles, mix[0]); at the laptop
    profile that is 96 draws with a standard deviation of 4.9 against 0.2% of slack, so about
    one master seed in four sized an iteration below ``MIN_TRAINABLE_FRACTION`` and halted the
    run. The count is now the same on every seed, and the margin is the whole 0.2% again.
    """
    from royalelearn.coordinator import MIN_TRAINABLE_FRACTION

    config = laptop()
    shape = geometry(config)
    floor = config.ppo.timesteps_per_iteration * MIN_TRAINABLE_FRACTION
    collected: dict[int, int] = {}
    for seed in MASTER_SEEDS:
        ladder = _pool(tmp_path, f"floor-{seed}")
        matchmaker = MixMatchmaker(seed, config.ladder, n_battles=shape.n_battles)
        rows = _rows_this_cycle(matchmaker, ladder, shape.n_battles)
        collected[seed] = shape.cycles * rows
    assert set(collected.values()) == {shape.cycles * shape.learner_rows}
    short = {seed: rows for seed, rows in collected.items() if rows < floor}
    assert not short, f"{len(short)} of {len(MASTER_SEEDS)} seeds collect below {floor:.0f}"


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
    """96 battles, 48 of them mirror: 144 of the 192 slots are the learner's, exactly."""
    shape = geometry(laptop())
    matchmaker = MixMatchmaker(SEED, LadderConfig(), n_battles=shape.n_battles)
    assert matchmaker.learner_rows == 144
    assert matchmaker.expected_learner_row_fraction == 0.75
    assert matchmaker.expected_discarded_rows_frac == 0.25
    assert matchmaker.learner_rows == shape.learner_rows


def test_an_odd_rectangle_still_divides_into_whole_battles() -> None:
    """Eleven battles cannot be split 50/35/15, and the counts still sum to eleven.

    Each count is the nearest whole number of battles to its share, and the shares are resolved
    in order so that the three of them add up to the rectangle rather than to eleven and a bit.
    """
    assert role_counts((0.50, 0.35, 0.15), 11) == (6, 4, 1)
    assert sum(role_counts((0.50, 0.35, 0.15), 11)) == 11
    # A single battle: half of one battle is one battle, and the other two shares round away.
    assert role_counts((0.50, 0.35, 0.15), 1) == (1, 0, 0)
    # An all-mirror mixture keeps every slot.
    assert role_counts((1.0, 0.0, 0.0), 10) == (10, 0, 0)
    # A share small enough to round to zero does: three battles cannot hold a tenth of one.
    assert role_counts((0.50, 0.40, 0.10), 3) == (2, 1, 0)
    assert role_counts((0.0, 0.5, 0.5), 5) == (0, 3, 2)


def test_an_assignment_is_a_pure_function_of_its_arguments(pool: LadderPool) -> None:
    matchmaker = MixMatchmaker(SEED, LadderConfig(), n_battles=32)
    for battle, ordinal in ((0, 0), (3, 17), (11, 4)):
        first = matchmaker.assign(battle, ordinal, pool, None)
        assert matchmaker.assign(battle, ordinal, pool, None) == first
        twin = MixMatchmaker(SEED, LadderConfig(), n_battles=32)
        assert twin.assign(battle, ordinal, pool, None) == first
    other = MixMatchmaker(SEED + 1, LadderConfig(), n_battles=32)
    table = [matchmaker.assign(battle, 0, pool, None) for battle in range(32)]
    assert [other.assign(battle, 0, pool, None) for battle in range(32)] != table


def test_the_role_layout_is_a_permutation_and_not_the_first_m_indices() -> None:
    """A role that always landed on the low battle indices would always land on the same shard.

    Battles map to workers and shards in blocks, so `the first M are mirror` would put every
    mirror battle on the first worker or two: one worker would carry all the double-seat
    episodes and the rest would carry none, and a shard-round would be as long as the slowest.
    """
    shape = geometry(laptop())
    layouts = set()
    for seed in MASTER_SEEDS:
        matchmaker = MixMatchmaker(seed, LadderConfig(), n_battles=shape.n_battles)
        layout = tuple(matchmaker.role_of(battle) for battle in range(shape.n_battles))
        layouts.add(layout)
        block = shape.n_battles // shape.workers
        per_worker = [
            sum(1 for role in layout[w * block : (w + 1) * block] if role == ROLE_MIRROR)
            for w in range(shape.workers)
        ]
        # Every worker carries mirror battles, and none carries them all.
        assert min(per_worker) > 0
        assert max(per_worker) < sum(per_worker)
    # And the layout is the seed's own: thirty seeds, thirty layouts.
    assert len(layouts) == len(MASTER_SEEDS)


def test_the_seat_a_battle_plays_is_the_same_on_any_geometry(pool: LadderPool) -> None:
    """A different worker count and a different iteration length are the same run.

    The role cannot be: exactly half of 96 battles and exactly half of 192 battles are not the
    same set of battle indices, and exactness across the rectangle is what buys the guarantee.
    Everything drawn per episode is still addressed by the battle and its ordinal, so a battle
    that draws a seat draws the same one however wide the rectangle is.
    """
    wide = MixMatchmaker(SEED, LadderConfig(), n_battles=512)
    narrow = MixMatchmaker(SEED, LadderConfig(), n_battles=8)
    for battle in range(8):
        for ordinal in range(3):
            here = wide.assign(battle, ordinal, pool, None)
            there = narrow.assign(battle, ordinal, pool, None)
            if here.role == ROLE_MIRROR or there.role == ROLE_MIRROR:
                continue
            assert here.learner_seat == there.learner_seat
            if here.role == there.role:
                assert here.opponent_id == there.opponent_id


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
    matchmaker = MixMatchmaker(SEED, config, n_battles=64)
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
    matchmaker = MixMatchmaker(SEED, LadderConfig(), n_battles=400)
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


def test_one_slot_still_draws_its_seat_and_its_opponent_afresh(pool: LadderPool) -> None:
    """The slot's role is fixed; what it plays inside that role is not.

    A pool slot that always took the blue seat against the same snapshot would be one matchup
    repeated 228 times an iteration, which is the correlation the per-episode draw exists to
    avoid. So the two things the fix does not touch are asserted over the episodes of a single
    slot rather than across slots.
    """
    matchmaker = MixMatchmaker(SEED, LadderConfig(max_resident_opponents=4), n_battles=96)
    checked = 0
    for battle in range(96):
        if matchmaker.role_of(battle) == ROLE_MIRROR:
            continue
        seats: set[int] = set()
        opponents: set[str | None] = set()
        for ordinal in range(60):
            assignment = matchmaker.assign(battle, ordinal, pool, None)
            seats.add(assignment.learner_seat)
            opponents.add(assignment.opponent_id)
        assert seats == {0, 1}
        assert len(opponents) > 1
        checked += 1
    assert checked > 0


def test_an_assignment_is_binding_until_the_episode_ends(pool: LadderPool) -> None:
    matchmaker = MixMatchmaker(SEED, LadderConfig(), n_battles=8)
    assert matchmaker.ordinal(2) == 0
    first = matchmaker.assign(2, matchmaker.ordinal(2), pool, None)
    matchmaker.on_episode(_record(battle=2, ordinal=0, seat=0))
    matchmaker.on_episode(_record(battle=2, ordinal=0, seat=1))
    assert matchmaker.ordinal(2) == 1  # both seats reported one episode, not two
    # The next episode is a fresh draw; the one that just ended still answers as it did.
    assert matchmaker.assign(2, matchmaker.ordinal(2), pool, None).ordinal == 1
    assert matchmaker.assign(2, 0, pool, None) == first


def test_pool_battles_become_scripted_while_the_pool_is_empty(tmp_path) -> None:
    """And the substitution keeps the seat count, so the iteration is still exactly sized."""
    empty = LadderPool(ResultLog(tmp_path / "games.jsonl"))
    matchmaker = MixMatchmaker(SEED, LadderConfig(), n_battles=200)
    roles = {matchmaker.assign(battle, 0, empty, None).role for battle in range(200)}
    assert ROLE_POOL not in roles
    assert roles == {ROLE_MIRROR, ROLE_SCRIPTED}
    assert _rows_this_cycle(matchmaker, empty, 200) == matchmaker.learner_rows


def test_weights_are_floored_so_the_tail_stays_reachable(pool: LadderPool) -> None:
    config = LadderConfig(pfsp_uniform_floor=0.2, weight_floor_scale=0.01)
    matchmaker = MixMatchmaker(SEED, config, n_battles=8)
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
    matchmaker = MixMatchmaker(SEED, LadderConfig(), n_battles=96)
    for battle in range(4):
        for ordinal in range(battle):
            matchmaker.on_episode(_record(battle=battle, ordinal=ordinal))
    matchmaker.save_checkpoint(tmp_path / "matchmaker")
    loaded = MixMatchmaker(SEED, LadderConfig())
    loaded.load_checkpoint(tmp_path / "matchmaker", strict=True)
    assert [loaded.ordinal(battle) for battle in range(4)] == [
        matchmaker.ordinal(battle) for battle in range(4)
    ]
    # The rectangle rides in the checkpoint, so the resumed run's roles are the ones it left.
    assert loaded.n_battles == 96
    assert [loaded.role_of(battle) for battle in range(96)] == [
        matchmaker.role_of(battle) for battle in range(96)
    ]
    assert loaded.learner_rows == matchmaker.learner_rows


def test_a_checkpoint_written_before_the_rectangle_was_stored_still_loads(tmp_path) -> None:
    """A format-1 checkpoint carries the ordinals and no battle count.

    It loads: the ordinals are what a resume cannot recompute, and the rectangle comes back at
    the first ``plan`` of the resumed run, which happens before any assignment is drawn.
    """
    import msgspec

    folder = tmp_path / "matchmaker"
    folder.mkdir(parents=True)
    (folder / "matchmaker.json").write_bytes(msgspec.json.encode({"ordinal": {"2": 5}}))
    loaded = MixMatchmaker(SEED, LadderConfig())
    loaded.load_checkpoint(folder, strict=True)
    assert loaded.ordinal(2) == 5
    assert loaded.n_battles is None
    shape = geometry(_config(1, 4, 1, (0.50, 0.35, 0.15)))
    loaded.plan(0, _pool(tmp_path, "resumed"), None, shape)
    assert loaded.n_battles == shape.n_battles
    assert loaded.learner_rows == shape.learner_rows
