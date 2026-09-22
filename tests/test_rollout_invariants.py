"""The invariants a round has to satisfy, checked by corrupting one.

Each of these catches a failure that is otherwise silent. A dropped cycle writes a transition
into the wrong row of the rectangle; a round of the wrong width leaves a column of it never
written; an action illegal under its own stored mask produces ``log pi = -inf`` at the first
minibatch; a truncated cell with no final observation is bootstrapped from a different battle;
an assignment that lands mid-episode makes a trajectory that two policies played look like one.

None of them needs an environment. They are assertions about arrays, so what the tests do is
build the arrays a healthy round would have, break one thing, and check that the break is named.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from royalelearn.api.rollout import (
    EPISODE_END_DRAW,
    EPISODE_END_NONE,
    EPISODE_END_WIN,
    GROUP_LEARNER,
    GROUP_SCRIPTED,
    RolloutRound,
)
from royalelearn.config import geometry
from royalelearn.rollout.inline import (
    assignments_constant_within_episodes,
    check_actions_legal,
    check_bootstrap,
    check_round,
)
from royalelearn.rollout.plan import SlotPlanner


@pytest.fixture
def planner(run_config: Any) -> SlotPlanner:
    return SlotPlanner(geometry(run_config), run_config.master_seed)


def healthy_round(planner: SlotPlanner, shard: int = 0, cycle: int = 3) -> RolloutRound:
    """A round with nothing wrong with it, to break one thing at a time."""
    slots = planner.round_slots(shard)
    n = slots.size
    return RolloutRound(
        cycle=cycle,
        shard=shard,
        slots=slots,
        obs_rows=(cycle * planner.n_slots + slots).astype(np.int32),
        group=np.full(n, GROUP_LEARNER, dtype=np.int8),
        reward=np.zeros(n, dtype=np.float32),
        terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool),
        valid=np.ones(n, dtype=bool),
        deploy_status=np.full(n, -1, dtype=np.int8),
        tick=np.zeros(n, dtype=np.int32),
        episode_end=np.zeros(n, dtype=np.int8),
    )


def test_a_healthy_round_passes(planner: SlotPlanner) -> None:
    round_ = healthy_round(planner)
    check_round(round_, cycle=round_.cycle, slots=planner.round_slots(round_.shard))
    check_bootstrap(round_.truncated, np.zeros(0, dtype=np.int32), round_.slots)


def test_a_dropped_cycle_is_named(planner: SlotPlanner) -> None:
    round_ = healthy_round(planner, cycle=3)
    with pytest.raises(ValueError, match="cycle 4 was due"):
        check_round(round_, cycle=4, slots=planner.round_slots(round_.shard))


def test_a_round_of_the_wrong_width_is_named(planner: SlotPlanner) -> None:
    round_ = healthy_round(planner)
    with pytest.raises(ValueError, match="covered"):
        check_round(round_, cycle=round_.cycle, slots=round_.slots[:-1])


def test_slots_out_of_order_are_refused(planner: SlotPlanner) -> None:
    """Ascending slot order is what makes an inference batch a function of the slot index
    rather than of which worker answered first."""
    round_ = healthy_round(planner)
    shuffled = round_.slots.copy()
    shuffled[[0, -1]] = shuffled[[-1, 0]]
    round_.slots = shuffled
    with pytest.raises(ValueError, match=f"ascending order, at cycle {round_.cycle}"):
        check_round(round_, cycle=round_.cycle, slots=planner.round_slots(round_.shard))


def test_an_action_illegal_under_its_own_mask_is_named(planner: SlotPlanner, env_spec: Any) -> None:
    slots = planner.round_slots(0)
    mask = np.zeros((slots.size, env_spec.n_actions), dtype=bool)
    mask[:, 0] = True
    mask[:, 7] = True
    actions = np.zeros(slots.size, dtype=np.int16)
    check_actions_legal(actions, mask, slots)
    actions[1] = 9
    with pytest.raises(ValueError, match=f"slot {int(slots[1])} took action 9"):
        check_actions_legal(actions, mask, slots)


def test_a_truncation_with_no_final_row_is_named(planner: SlotPlanner) -> None:
    round_ = healthy_round(planner)
    round_.truncated[2] = True
    with pytest.raises(ValueError, match="truncation"):
        check_bootstrap(round_.truncated, np.zeros(0, dtype=np.int32), round_.slots)
    check_bootstrap(round_.truncated, round_.slots[[2]], round_.slots)


def test_a_final_row_with_no_truncation_is_named(planner: SlotPlanner) -> None:
    round_ = healthy_round(planner)
    with pytest.raises(ValueError, match="final row"):
        check_bootstrap(round_.truncated, round_.slots[[0]], round_.slots)


def test_an_assignment_inside_an_episode_is_named(planner: SlotPlanner) -> None:
    """A battle may change hands only on the cycle whose scalars report an episode end."""
    cycles, slots = 6, planner.n_slots
    group = np.full((cycles, slots), GROUP_LEARNER, dtype=np.int8)
    ends = np.full((cycles, slots), EPISODE_END_NONE, dtype=np.int8)
    ends[3, 0] = EPISODE_END_WIN
    ends[3, 1] = EPISODE_END_DRAW
    group[3:, 0] = GROUP_SCRIPTED
    group[3:, 1] = GROUP_SCRIPTED
    assignments_constant_within_episodes(group, ends)

    group[2:, 0] = GROUP_SCRIPTED
    with pytest.raises(ValueError, match="inside an episode"):
        assignments_constant_within_episodes(group, ends)


def test_the_invariant_refuses_arrays_of_different_shapes(planner: SlotPlanner) -> None:
    group = np.zeros((4, planner.n_slots), dtype=np.int8)
    with pytest.raises(ValueError, match="cycles, slots"):
        assignments_constant_within_episodes(group, np.zeros((3, planner.n_slots), dtype=np.int8))


def test_the_slot_map_covers_every_slot_exactly_once(planner: SlotPlanner) -> None:
    """Every slot appears in exactly one shard-round, and a cycle is all of them."""
    seen = np.concatenate(
        [planner.round_slots(shard) for shard in range(planner.shards_per_worker)]
    )
    assert np.array_equal(np.sort(seen), np.arange(planner.n_slots))
    for shard in range(planner.shards_per_worker):
        slots = planner.round_slots(shard)
        assert (planner.slot_shard[slots] == shard).all()
        assert slots.size == planner.slots_per_shard * planner.workers


def test_a_slot_is_one_seat_of_one_battle(planner: SlotPlanner) -> None:
    for slot in range(planner.n_slots):
        battle = int(planner.slot_battle[slot])
        seat = int(planner.slot_seat[slot])
        assert planner.slot_of(battle, seat) == slot
        assert int(planner.battle_worker[battle]) == int(planner.slot_worker[slot])
        assert int(planner.battle_shard[battle]) == int(planner.slot_shard[slot])


def test_a_shards_slots_are_the_seats_of_its_battles(planner: SlotPlanner) -> None:
    for worker in range(planner.workers):
        for shard in range(planner.shards_per_worker):
            battles = planner.shard_battles(worker, shard)
            slots = planner.shard_slots(worker, shard)
            assert np.array_equal(slots, np.repeat(2 * battles, 2) + np.tile([0, 1], battles.size))


def test_a_seed_is_a_function_of_its_name(planner: SlotPlanner, run_config: Any) -> None:
    """Every draw is name-addressed, so adding a consumer never moves an existing stream."""
    twin = SlotPlanner(geometry(run_config), run_config.master_seed)
    assert planner.env_seed(0, 0, 0) == twin.env_seed(0, 0, 0)
    assert planner.env_seed(0, 0, 0) != planner.env_seed(0, 0, 1)
    assert planner.match_seed(1, 0) != planner.match_seed(1, 1)
    assert np.array_equal(planner.uniforms(2, 5), twin.uniforms(2, 5))
    assert not np.array_equal(planner.uniforms(2, 5), planner.uniforms(2, 6))
    assert planner.uniforms(0, 0).shape == (planner.n_slots,)
    assert ((planner.uniforms(0, 0) >= 0) & (planner.uniforms(0, 0) < 1)).all()
