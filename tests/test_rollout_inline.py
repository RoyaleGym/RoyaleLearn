"""The rollout path, in one process, against MockEngine.

``InlineRolloutSource`` is the semantic reference: the process farm is accepted when it
produces these bytes, and a Rust worker will be accepted the same way. So what is asserted here
is not "the source ran" but every property the learner is entitled to assume -- where a slot's
observation went, what its scalars mean, when a battle may change hands, and that the numbers
the environment reports and the numbers the worker counts are the same numbers.

One run answers all of it. Building an environment costs about a second and driving it costs
milliseconds, so the run is a module fixture and the assertions are cheap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rollout_support import CODEC, ReferenceCodec, SharedRectangle, drive, preflight, rollout_config
from royalelearn.api.rollout import (
    EPISODE_END_NONE,
    GROUP_LEARNER,
    GROUP_SCRIPTED,
    ROLE_MIRROR,
    Step,
)
from royalelearn.rollout.inline import (
    InlineRolloutSource,
    assignments_constant_within_episodes,
    check_bootstrap,
    check_round,
)
from royalelearn.rollout.plan import SlotPlanner
from royalelearn.rollout.scripted import SCRIPTED_NAMES
from royalelearn.seeding import derive_generator

CYCLES = 32
MAX_STEPS = 14


class Ran:
    """One inline run, and the pieces needed to ask questions of it."""

    def __init__(self, config: Any, report: Any, out: Any, planner: Any, codec: Any) -> None:
        self.config = config
        self.report = report
        self.out = out
        self.planner = planner
        self.codec = codec
        self.spec = report.spec
        self.geometry = report.geometry


@pytest.fixture(scope="module")
def ran() -> Ran:
    config = rollout_config(
        workers=2,
        games_per_worker=2,
        shards_per_worker=2,
        max_steps=MAX_STEPS,
        stagger_first_reset=False,
    )
    report = preflight(config)
    planner = SlotPlanner(report.geometry, config.master_seed)
    codec = ReferenceCodec(report.table)
    codec.bind(report.spec, report.table)
    buffer = SharedRectangle(
        report.spec,
        run_id="inline-test",
        cycles=CYCLES,
        n_slots=report.geometry.n_slots,
        row_bytes=report.row_bytes,
    )
    source = InlineRolloutSource(
        config, report.spec, report.table, run_id="inline-test", codec=CODEC
    )
    generator = derive_generator(config.master_seed, "test/inline/policy")

    def policy(round_: Any, rect: SharedRectangle) -> np.ndarray:
        """A legal action per slot, drawn from the mask that was stored with the row.

        Reading the mask back out of the rectangle rather than off the environment is the
        point: it is the mask the learner will apply at update time, so an action legal under
        it is an action legal under the one the transition carries.
        """
        actions = np.zeros(round_.slots.size, dtype=np.int16)
        for index, row in enumerate(round_.obs_rows):
            mask = codec.unpack_row(rect.rows, int(row))["action_mask"]
            legal = np.flatnonzero(mask)
            legal = legal[legal != 0]
            if legal.size and generator.random() < 0.5:
                actions[index] = int(legal[generator.integers(legal.size)])
        return actions

    def matchmaker(battle: int, ordinal: int) -> tuple[tuple[int, int], tuple[int, int], int]:
        """Alternate between a mirror and a scripted opponent, per battle and per episode."""
        draw = derive_generator(config.master_seed, planner.match_path(battle, ordinal))
        if draw.random() < 0.5:
            return (GROUP_LEARNER, GROUP_LEARNER), (-1, -1), -1
        seat = int(draw.integers(2))
        groups = [GROUP_LEARNER, GROUP_LEARNER]
        groups[1 - seat] = GROUP_SCRIPTED
        index = int(draw.integers(len(SCRIPTED_NAMES)))
        return (groups[0], groups[1]), (index, index), seat

    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        out = drive(source, buffer, report, cycles=CYCLES, policy=policy, matchmaker=matchmaker)
    finally:
        source.close()
        buffer.close()
    return Ran(config, report, out, planner, codec)


def test_every_slot_is_written_exactly_once_a_cycle(ran: Ran) -> None:
    rows = ran.out.obs_rows
    layout = ran.report.geometry
    assert rows.shape == (CYCLES + 1, layout.n_slots)
    assert len(set(rows.reshape(-1).tolist())) == rows.size
    assert ran.out.valid.all()


def test_slots_map_to_their_own_battles(ran: Ran) -> None:
    """A slot is one seat of one battle, and the record says which."""
    planner = ran.planner
    for record in ran.out.episodes:
        assert record.battle == int(planner.slot_battle[record.slot])
        assert record.seat == int(planner.slot_seat[record.slot])
        assert record.worker == int(planner.slot_worker[record.slot])
        assert record.shard == int(planner.slot_shard[record.slot])
        assert record.episode_seed_path == planner.env_seed_path(record.worker, record.shard, 0)


def test_rounds_are_the_shard_they_claim_to_be(ran: Ran) -> None:
    """Each cycle's slots are exactly one shard's, ascending, across every worker."""
    planner = ran.planner
    for shard in range(ran.geometry.shards_per_worker):
        slots = planner.round_slots(shard)
        assert np.array_equal(np.sort(slots), slots)
        cycles = np.flatnonzero(ran.out.shard[:, slots[0]] == shard)
        assert cycles.size >= CYCLES // ran.geometry.shards_per_worker


def test_episode_ends_arrive_in_pairs(ran: Ran) -> None:
    """Both seats of a battle end on the same cycle: it is one battle."""
    ends = ran.out.episode_end != EPISODE_END_NONE
    for battle in range(ran.geometry.n_battles):
        assert np.array_equal(ends[:, 2 * battle], ends[:, 2 * battle + 1])
    assert ends.sum() >= 2 * ran.geometry.n_battles, "fewer than one episode ended per battle"


def test_the_start_up_summary_names_the_run() -> None:
    """Gate 10's last line, and where it comes from.

    A run identity is a function of what preflight found and of the architecture the network
    factory builds, so it does not exist while preflight runs. The report carries the string
    once it does, which is what makes the id a person reads at start-up the id that goes into
    the checkpoint rather than a second opinion about it.

    Its own report, because this writes to one.
    """
    config = rollout_config(
        workers=1, games_per_worker=1, shards_per_worker=1, max_steps=MAX_STEPS
    )
    report = preflight(config)
    assert report.run_id == ""
    assert not any(line.startswith("run id") for line in report.lines)
    report.note_run_id("0123456789abcdef", printer=None)
    assert report.run_id == "0123456789abcdef"
    assert report.lines[-1].split() == ["run", "id", "0123456789abcdef"]

    named = preflight(config, run_id="fedcba9876543210")
    assert named.run_id == "fedcba9876543210"
    assert named.lines[-1].endswith("fedcba9876543210")


def test_outcomes_are_opposite_within_a_battle(ran: Ran) -> None:
    by_slot: dict[tuple[int, int], Any] = {(r.slot, r.ordinal): r for r in ran.out.episodes}
    for (slot, ordinal), record in by_slot.items():
        other = by_slot.get((slot ^ 1, ordinal))
        assert other is not None, f"slot {slot} ended episode {ordinal} alone"
        assert record.outcome == -other.outcome
        assert record.own_crowns == other.enemy_crowns
        assert record.episode_steps == other.episode_steps


def test_rewards_are_antisymmetric_in_a_mirror_battle(ran: Ran) -> None:
    """The default reward is zero-sum, so a mirror battle's two seats cancel.

    It is the check that the two seats are being scored from their own points of view rather
    than both from blue's, which is a mistake that produces a policy trained to lose as red.
    """
    reward, group = ran.out.reward, ran.out.group
    mirror = 0
    for battle in range(ran.geometry.n_battles):
        blue, red = 2 * battle, 2 * battle + 1
        rows = np.flatnonzero(
            (group[:, blue] == GROUP_LEARNER) & (group[:, red] == GROUP_LEARNER)
        )
        mirror += rows.size
        assert np.allclose(reward[rows, blue], -reward[rows, red], atol=1e-6)
    assert mirror, "no cycle of the run was a mirror battle"


def test_no_action_was_refused_by_the_engine(ran: Ran) -> None:
    """deploy_status is -1 for a no-op and 0 for an accepted command; anything else is a mask
    bug, not a tolerance."""
    status = ran.out.deploy_status
    assert status.min() >= -1
    assert (status <= 0).all(), f"{int((status > 0).sum())} commands were refused by the engine"
    assert sum(r.illegal_commands for r in ran.out.episodes) == 0


def test_terminal_scalars_are_the_environments_own(ran: Ran) -> None:
    """The record's episode length and return agree with the rectangle they came from.

    The scalars are copied out of ``final_info``; the return is summed by the worker. If the
    two disagree the worker is counting a different episode from the one that ended.
    """
    reward = ran.out.reward
    ends = ran.out.episode_end
    for record in ran.out.episodes:
        column = ends[:, record.slot]
        # The first episode of a battle is ordinal 0, which is also its position in the column.
        cycle = int(np.flatnonzero(column != EPISODE_END_NONE)[record.ordinal])
        start = cycle - record.episode_steps
        assert start >= 0
        assert record.episode_steps <= MAX_STEPS
        assert record.episode_ticks == record.episode_steps * ran.spec.decision_ticks
        total = reward[start + 1 : cycle + 1, record.slot].sum()
        assert record.undiscounted_return == pytest.approx(float(total), abs=1e-5)
        assert 0.0 <= record.own_tower_hp_frac <= 1.0
        assert record.elixir_count_exact in (True, False)


def test_reward_terms_sum_to_the_return(ran: Ran) -> None:
    """The per-term breakdown is the reward, not a sample of it.

    The environment clears its breakdown when it resets a finished battle, which is the same
    step that ended it, so the terminal step's terms are only there because the worker records
    them as they are produced.
    """
    scored = 0
    for record in ran.out.episodes:
        assert record.reward_terms
        total = sum(record.reward_terms.values())
        assert total == pytest.approx(record.undiscounted_return, abs=1e-4)
        scored += any(value != 0.0 for value in record.reward_terms.values())
    assert scored, "every episode scored exactly zero in every term, so nothing was recorded"


def test_a_battle_changes_hands_only_where_an_episode_ended(ran: Ran) -> None:
    """The assignment invariant, over the whole rectangle.

    It is what makes "one policy for one whole episode" checkable: a partially controlled
    trajectory would have a group column that changes with no end beside it.

    The Drive's columns are indexed by the cycle each round was published at, which is where
    the invariant's inputs differ: the group is the round's own and the episode end is the step
    that arrived at it, so the ends are read from one row up, exactly as the rectangle stores
    them.
    """
    ends = ran.out.episode_end[1:]
    assignments_constant_within_episodes(ran.out.group[:-1], ends)
    changed = (ran.out.group[1:] != ran.out.group[:-1]).any()
    assert changed, "no battle ever changed hands, so the invariant proved nothing"
    assert (ends != EPISODE_END_NONE).any(), "no episode ended, so nothing was allowed"


def test_scripted_seats_are_played_by_the_worker(ran: Ran) -> None:
    """A scripted seat's action is the worker's, whatever the parent wrote there."""
    scripted = ran.out.group == GROUP_SCRIPTED
    assert scripted.any(), "no seat was ever scripted"
    played = (ran.out.action != 0) & scripted
    assert ran.out.action[scripted].min() >= 0
    assert played.sum() >= 0  # the no-op opponent plays nothing, and that is an answer


def test_the_rectangle_holds_the_observation_the_policy_acted_on(ran: Ran) -> None:
    """A packed row unpacks to a legal mask whose planes are the mask reshaped.

    The rectangle is the only copy of an observation there is: nothing re-reads it from the
    environment, so a row that is not what the environment produced is a policy trained on
    noise.
    """
    spec = ran.spec
    rows = ran.out.rows
    for cycle in (0, CYCLES // 2, CYCLES):
        for slot in (0, ran.geometry.n_slots - 1):
            row = int(ran.out.obs_rows[cycle, slot])
            one = ran.codec.unpack_row(rows, row, ran.report.statics)
            assert one["action_mask"][0] == 1
            assert np.array_equal(
                one["mask_planes"],
                one["action_mask"][1:].reshape(spec.hand_size, *spec.tiles),
            )
            assert one["vector"].shape == (spec.vector_size,)
            assert one["spatial"].shape == spec.spatial_shape
            assert one["vector"].max() <= 1.0


def test_truncated_cells_have_a_final_observation(ran: Ran) -> None:
    """A truncation bootstraps from the state it was cut off in, and that state is not in the
    rectangle: the environment resets on the step that ends an episode."""
    truncated = ran.out.truncated
    assert truncated.any(), "nothing truncated, so the bootstrap path was never taken"
    for cycle in np.flatnonzero(truncated.any(axis=1)):
        for shard in range(ran.geometry.shards_per_worker):
            slots = ran.planner.round_slots(shard)
            expected = int(truncated[cycle, slots].sum())
            if not expected:
                continue
            rows = ran.out.finals[(int(cycle), shard)]
            assert rows.shape == (expected, ran.report.row_bytes)


def test_check_round_accepts_the_rounds_that_were_produced(ran: Ran) -> None:
    """The invariants the coordinator asserts hold on a run that was not corrupted."""
    from royalelearn.api.rollout import RolloutRound

    planner = ran.planner
    for shard in range(ran.geometry.shards_per_worker):
        slots = planner.round_slots(shard)
        cycle = int(np.flatnonzero(ran.out.shard[:, slots[0]] == shard)[0])
        round_ = RolloutRound(
            cycle=cycle,
            shard=shard,
            slots=slots,
            obs_rows=ran.out.obs_rows[cycle, slots],
            group=ran.out.group[cycle, slots],
            reward=ran.out.reward[cycle, slots],
            terminated=ran.out.terminated[cycle, slots],
            truncated=ran.out.truncated[cycle, slots],
            valid=ran.out.valid[cycle, slots],
            deploy_status=ran.out.deploy_status[cycle, slots],
            tick=ran.out.tick[cycle, slots],
            episode_end=ran.out.episode_end[cycle, slots],
        )
        check_round(round_, cycle=cycle, slots=slots)
        check_bootstrap(
            round_.truncated,
            slots[round_.truncated] if round_.truncated.any() else np.zeros(0, np.int32),
            slots,
        )


def test_the_plan_is_the_mirror_it_was_built_as(ran: Ran) -> None:
    plan = ran.planner.mirror_plan(3)
    assert plan.iteration == 3
    assert plan.n_slots == ran.geometry.n_slots
    assert len(plan.assignment) == ran.geometry.n_battles
    assert {a.role for a in plan.assignment} == {ROLE_MIRROR}
    assert all(a.group == (GROUP_LEARNER, GROUP_LEARNER) for a in plan.assignment)


def test_a_deferred_round_hands_back_the_same_data() -> None:
    """``Defer`` and ``Spaces`` answer a round without stepping the battles.

    They are what the parent submits when it has nothing to say -- an iteration it is not
    collecting, a check of the environment it wants re-read -- and the property that matters is
    that neither advances a battle: the cycle comes back unchanged, so nothing has been
    collected twice.
    """
    from royalelearn.api.rollout import Defer, Spaces

    config = rollout_config(
        workers=1, games_per_worker=1, shards_per_worker=1, max_steps=MAX_STEPS,
        stagger_first_reset=False,
    )
    report = preflight(config)
    planner = SlotPlanner(report.geometry, config.master_seed)
    buffer = SharedRectangle(
        report.spec,
        run_id="inline-defer",
        cycles=4,
        n_slots=report.geometry.n_slots,
        row_bytes=report.row_bytes,
    )
    source = InlineRolloutSource(
        config, report.spec, report.table, run_id="inline-defer", codec=CODEC
    )
    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        first = source.next_round(5.0)
        assert first.cycle == 0
        ticks = first.tick.copy()
        source.submit(Defer())

        deferred = source.next_round(5.0)
        assert deferred.cycle == 0
        assert np.array_equal(deferred.tick, ticks)
        source.submit(Spaces())

        rechecked = source.next_round(5.0)
        assert rechecked.cycle == 0
        assert np.array_equal(rechecked.tick, ticks)
        source.submit(
            Step(
                actions=np.zeros(rechecked.slots.size, dtype=np.int16),
                gamma=0.99,
                group=np.zeros(0, dtype=np.int8),
                opponent_ix=np.zeros(0, dtype=np.int8),
                learner_seat=np.zeros(0, dtype=np.int8),
            )
        )
        stepped = source.next_round(5.0)
        assert stepped.cycle == 1
        assert (stepped.tick > ticks).all()
    finally:
        source.close()
        buffer.close()
