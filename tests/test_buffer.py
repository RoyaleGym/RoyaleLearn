"""The experience rectangle: what a round writes, what a minibatch gathers, and what neither
does.

The rectangle is built here at a handful of cycles and a handful of slots, and filled the way a
worker fills it -- by packing real observations into the shared block at the row a cell's index
names. Nothing in this file knows a width: the row size is the codec's, the plane count is the
space's, and the footprint is checked against the arithmetic recomputed from both.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from royalelearn.api.rollout import (
    EPISODE_END_LOSS,
    EPISODE_END_NONE,
    EPISODE_END_WIN,
    GROUP_DEAD,
    GROUP_LEARNER,
    EnvSpec,
)
from royalelearn.seeding import derive_generator

torch = pytest.importorskip("torch")


from royalelearn.learn.buffer import RectBuffer  # noqa: E402

# The rectangle harness is defined once, in ``royalelearn.testing``, where a package built on
# RoyaleLearn finds it too; the names are imported here, where the other test modules find them.
from royalelearn.testing import (  # noqa: E402
    CYCLES,
    SEED,
    SLOTS,
    plan_for,
    round_for,
)
from royalelearn.testing import RectFixture as Fixture  # noqa: E402

BITS_PER_BYTE = 8


@pytest.fixture(scope="module")
def observations(mock_env_spec: Any) -> list[dict[str, np.ndarray]]:
    """Real observations to fill the rectangle with, from this module's own environment."""
    env = mock_env_spec.build_vec(2)
    try:
        batches = [env.reset(seed=5)[0]]
        actions = np.zeros(env.num_envs, dtype=np.int64)
        for _ in range(16):
            batches.append(env.step(actions)[0])
    finally:
        env.close()
    return [{key: value[0] for key, value in batch.items()} for batch in batches]


def record_clean_iteration(buffer: RectBuffer, slots: np.ndarray, cycles: int = CYCLES) -> None:
    """Every round of an iteration that ended no episode and lost no worker.

    The trailing round at cycle ``T`` is one of them. It has no row of its own, so it is handed
    no action, but it is where row ``T - 1``'s reward and validity come from and a rectangle
    filled without it has a last row nothing completed.
    """
    for cycle in range(cycles + 1):
        trailing = cycle == cycles
        buffer.record_round(
            round_for(cycle, slots, rows=slots),
            actions=None if trailing else np.zeros(slots.size, dtype=np.int16),
            log_probs=None if trailing else np.zeros(slots.size, dtype=np.float32),
        )


@pytest.fixture
def rect(env_spec: EnvSpec, observations: list[dict[str, np.ndarray]]) -> Iterator[Fixture]:
    built = Fixture(env_spec, observations)
    try:
        yield built
    finally:
        built.close()


# --------------------------------------------------------------------------
# The rectangle
# --------------------------------------------------------------------------


def test_the_footprint_is_the_arithmetic_of_the_space(rect: Fixture) -> None:
    spec = rect.spec
    layout = rect.buffer.layout
    planes, tiles_y, tiles_x = spec.spatial_shape
    cells = tiles_y * tiles_x
    stored_bytes = len(rect.codec.layout.u8_planes)
    stored_halves = len(rect.codec.layout.f16_planes)
    assert stored_bytes + stored_halves + len(rect.codec.layout.static_planes) == planes
    row_bytes = (
        cells * (stored_bytes + 2 * stored_halves)
        + 2 * spec.vector_size
        + math.ceil(spec.n_actions / BITS_PER_BYTE)
    )
    assert layout.row_bytes == row_bytes
    assert layout.rows == layout.cycles + spec.frame_stack
    assert layout.obs_bytes == layout.rows * layout.n_slots * row_bytes
    assert rect.buffer.shm.size >= layout.obs_offset + layout.obs_bytes


def test_the_handle_carries_what_a_worker_computes_its_offsets_from(rect: Fixture) -> None:
    handle = rect.buffer.shared_handle()
    assert handle.row_bytes == rect.codec.row_bytes(rect.spec)
    assert handle.codec_version == rect.codec.codec_version
    assert handle.frame_stack == rect.spec.frame_stack
    rebuilt = handle.layout()
    assert rebuilt.total_bytes == rect.buffer.layout.total_bytes
    assert rebuilt.cell_offset(2, 3) == rect.buffer.layout.cell_offset(2, 3)


def test_a_cell_s_row_is_its_slot_index(rect: Fixture) -> None:
    """No second map from slot to row: the index IS the slot index, offset by the history."""
    layout = rect.buffer.layout
    for cycle in range(layout.cycles + 1):
        for slot in range(layout.n_slots):
            assert layout.row_index(cycle, slot) == (
                cycle + layout.history_rows
            ) * layout.n_slots + slot


# --------------------------------------------------------------------------
# What a round writes
# --------------------------------------------------------------------------


def test_record_round_writes_exactly_the_cells_it_was_given(rect: Fixture) -> None:
    buffer = rect.buffer
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    slots = np.array([1, 4, 5], dtype=np.int64)
    rewards = np.array([0.5, -0.5, 2.0], dtype=np.float32)
    terminated = np.array([False, True, False])
    buffer.record_round(
        round_for(
            2,
            slots,
            rows=np.array([buffer.layout.row_index(2, int(s)) for s in slots]),
            reward=rewards,
            terminated=terminated,
        ),
        actions=np.array([3, 0, 7], dtype=np.int16),
        log_probs=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
    )
    # The step this round reports finished at cycle 2 and began at cycle 1, so its reward and
    # its flag belong beside the action that earned them, one row down.
    assert np.array_equal(buffer.reward[1, slots], rewards)
    assert np.array_equal(buffer.terminated[1, slots], terminated)
    assert buffer.valid[1, slots].all()
    assert np.array_equal(buffer.action[2, slots], np.array([3, 0, 7]))
    untouched = np.setdiff1d(np.arange(SLOTS), slots)
    assert not buffer.valid[1, untouched].any()
    assert not buffer.valid[np.arange(CYCLES) != 1].any()
    assert np.array_equal(buffer.group[2, untouched], np.full(untouched.size, GROUP_DEAD))


def test_every_scalar_of_a_round_lands_on_the_row_it_describes(rect: Fixture) -> None:
    """Each of a round's columns, and which side of the boundary it is on.

    A round carries a state and the step that arrived at it, and those are two timesteps. The
    observation, the tick that dates it, the group holding the seat and the action taken there
    are the round's own cycle. The reward, the two done flags, the deploy status, the episode
    end and the validity of the publication all describe the step, which began one cycle
    earlier and is the step the row below acted.

    Every quantity names the cycle it came from, so a column stored one row out reads as
    another cycle's number rather than as a plausible one.
    """
    buffer = rect.buffer
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    slots = np.arange(SLOTS, dtype=np.int64)
    #: The round at this cycle reports an ended episode, and nothing else does.
    won, lost, dead = 3, 5, 2
    for cycle in range(CYCLES + 1):
        ends = np.zeros(SLOTS, dtype=np.int8)
        if cycle == won:
            ends[:] = EPISODE_END_WIN
        elif cycle == lost:
            ends[:] = EPISODE_END_LOSS
        buffer.record_round(
            round_for(
                cycle,
                slots,
                rows=np.array([buffer.layout.row_index(cycle, int(s)) for s in slots]),
                group=GROUP_LEARNER if cycle <= won else 0,
                reward=np.full(SLOTS, 100.0 + cycle, dtype=np.float32),
                terminated=np.full(SLOTS, cycle == won),
                truncated=np.full(SLOTS, cycle == lost),
                valid=cycle != dead,
                episode_end=ends,
                tick=np.full(SLOTS, 10 * cycle, dtype=np.int32),
                deploy_status=cycle,
            ),
            actions=np.full(SLOTS, 20 + cycle, dtype=np.int16),
            log_probs=np.full(SLOTS, -1.0 * cycle, dtype=np.float32),
        )
    rows = np.arange(CYCLES)
    # Arriving: the row that acted is one below the cycle that reported.
    assert np.array_equal(buffer.reward[:CYCLES, 0], 100.0 + rows + 1)
    assert np.array_equal(buffer.deploy_status[:CYCLES, 0], rows + 1)
    assert np.array_equal(np.flatnonzero(buffer.terminated[:CYCLES, 0]), [won - 1])
    assert np.array_equal(np.flatnonzero(buffer.truncated[:CYCLES, 0]), [lost - 1])
    assert buffer.episode_end[won - 1, 0] == EPISODE_END_WIN
    assert buffer.episode_end[lost - 1, 0] == EPISODE_END_LOSS
    assert np.array_equal(
        np.flatnonzero(buffer.episode_end[:, 0] != EPISODE_END_NONE), [won - 1, lost - 1]
    )
    assert np.array_equal(np.flatnonzero(~buffer.valid[:CYCLES, 0]), [dead - 1])
    # Own: the state the round published, and what was done in it.
    assert np.array_equal(buffer.tick[:CYCLES, 0], 10 * rows)
    assert np.array_equal(buffer.action[:CYCLES, 0], 20 + rows)
    assert np.array_equal(buffer.log_prob[:CYCLES, 0], -1.0 * rows)
    assert np.array_equal(buffer.group[:CYCLES, 0] == GROUP_LEARNER, rows <= won)


def test_the_trailing_round_is_the_only_source_of_the_last_rows_scalars(
    rect: Fixture,
) -> None:
    """Cycle ``T`` carries the bootstrap observation and the reward of the step before it.

    Dropping it costs every iteration its last transition, which is a reward the environment
    gave and the objective then never sees.
    """
    buffer = rect.buffer
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    slots = np.arange(SLOTS, dtype=np.int64)
    buffer.record_round(
        round_for(CYCLES, slots, rows=slots, reward=np.full(SLOTS, 9.0, dtype=np.float32)),
        actions=None,
        log_probs=None,
    )
    assert np.array_equal(buffer.reward[CYCLES - 1], np.full(SLOTS, 9.0))
    assert buffer.valid[CYCLES - 1].all()
    assert not buffer.valid[: CYCLES - 1].any()
    # It has no row of its own, so nothing it says about its own cycle is stored.
    assert not buffer.action.any()
    assert not buffer.tick.any()
    with pytest.raises(IndexError):
        buffer.record_round(
            round_for(CYCLES + 1, slots, rows=slots),
            actions=np.zeros(SLOTS, dtype=np.int16),
            log_probs=np.zeros(SLOTS, dtype=np.float32),
        )


def test_the_advantage_of_a_terminated_row_is_its_own_reward_against_its_own_value(
    rect: Fixture,
) -> None:
    """``A = r - V(s)`` on the row a termination ended, with rewards and values it can name.

    A terminated cell bootstraps from zero and the recursion breaks there, so its advantage is
    exactly one reward less one value. Both are distinct per cycle, so the equality holds only
    if the reward the estimator was handed and the value beside it are the same timestep's --
    which is the pairing the whole rectangle exists to get right.
    """
    from royalelearn.learn.gae import gae_recursion

    buffer = rect.buffer
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    slots = np.arange(SLOTS, dtype=np.int64)
    ended = 3
    for cycle in range(CYCLES + 1):
        ends = np.full(SLOTS, EPISODE_END_WIN if cycle == ended + 1 else EPISODE_END_NONE, np.int8)
        buffer.record_round(
            round_for(
                cycle,
                slots,
                rows=slots,
                reward=np.full(SLOTS, float(cycle), dtype=np.float32),
                terminated=np.full(SLOTS, cycle == ended + 1),
                episode_end=ends,
            ),
            actions=None if cycle == CYCLES else np.zeros(SLOTS, dtype=np.int16),
            log_probs=None if cycle == CYCLES else np.zeros(SLOTS, dtype=np.float32),
        )
    values = 100.0 + torch.arange((CYCLES + 1) * SLOTS, dtype=torch.float32).reshape(
        CYCLES + 1, SLOTS
    )
    buffer.set_values(values)
    inputs = buffer.advantage_inputs()
    advantages, _ = gae_recursion(
        rewards=inputs.rewards,
        values=inputs.values,
        final_values=inputs.final_values,
        terminated=inputs.terminated,
        truncated=inputs.truncated,
        gamma=0.9,
        lam=0.95,
    )
    for slot in range(SLOTS):
        reward = float(ended + 1)
        assert float(buffer.reward[ended, slot]) == reward
        expected = reward - float(values[ended, slot])
        assert advantages[ended, slot].item() == pytest.approx(expected, abs=1e-4)


def test_an_iteration_clears_what_the_last_one_wrote(rect: Fixture) -> None:
    buffer = rect.buffer
    slots = np.arange(SLOTS, dtype=np.int64)
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    buffer.record_round(
        round_for(0, slots, rows=slots, reward=np.full(SLOTS, 3.0, dtype=np.float32)),
        actions=np.full(SLOTS, 5, dtype=np.int16),
        log_probs=np.zeros(SLOTS, dtype=np.float32),
    )
    buffer.begin_iteration(plan_for(SLOTS, iteration=1), CYCLES)
    assert not buffer.reward.any()
    assert not buffer.action.any()
    assert not buffer.valid.any()
    assert (buffer.group == GROUP_DEAD).all()


# --------------------------------------------------------------------------
# What reaches the update
# --------------------------------------------------------------------------


def fill_iteration(rect: Fixture, *, opponent_slots: tuple[int, ...] = (), dead_from: int = -1):
    """Collect a whole iteration: every slot every cycle, some of them an opponent's.

    Every round of the iteration, the trailing one at cycle ``T`` included: it is where row
    ``T - 1``'s reward and validity come from, so a rectangle filled without it has a last row
    nothing ever completed.

    ``dead_from`` is the first ROW that is not collected. The round that would have completed
    it is the one at the cycle above, so that is the round this stops publishing at.
    """
    buffer = rect.buffer
    rect.fill()
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    rng = derive_generator(SEED, "test/buffer/fill")
    slots = np.arange(SLOTS, dtype=np.int64)
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    for slot in opponent_slots:
        groups[slot] = 0  # a resident snapshot's index: not the learner
    for cycle in range(CYCLES + 1):
        trailing = cycle == CYCLES
        for slot in slots:
            record = round_for(
                cycle,
                np.array([slot], dtype=np.int64),
                rows=np.array([buffer.layout.row_index(cycle, int(slot))]),
                group=int(groups[slot]),
                reward=rng.normal(size=1).astype(np.float32),
                valid=not (0 <= dead_from <= cycle - 1),
            )
            buffer.record_round(
                record,
                actions=None if trailing else rng.integers(0, 2, size=1).astype(np.int16),
                log_probs=None if trailing else rng.normal(size=1).astype(np.float32),
            )
    return buffer


def test_what_decides_a_trainable_cell_is_the_group_and_the_validity(rect: Fixture) -> None:
    opponents = (2, 3)
    buffer = fill_iteration(rect, opponent_slots=opponents)
    trainable = buffer.trainable()
    assert trainable.shape == (CYCLES, SLOTS)
    for slot in range(SLOTS):
        assert bool(trainable[:, slot].all()) == (slot not in opponents)
    assert not trainable[:, list(opponents)].any()
    assert trainable.sum() == CYCLES * (SLOTS - len(opponents))
    assert torch.equal(buffer.trainable_mask(), torch.from_numpy(trainable))


def test_the_count_of_legal_actions_is_stored_beside_the_cell_it_describes(
    rect: Fixture,
) -> None:
    """Whether a decision had a choice at all is a property of one row, and it is asked per row.

    The count comes off the same mask the whole-iteration critic pass already unpacks, so it is
    stored rather than recomputed: anything downstream that wanted it would otherwise unpack the
    observation a second time to find out how many actions the cell's mask left. It is cleared
    when an iteration opens, with every other scalar -- a count carried over from the previous
    rectangle would describe rows that are no longer in it.
    """
    from royalelearn.learn.inference import RectGather

    buffer = fill_iteration(rect)
    slots = np.arange(SLOTS, dtype=np.int64)
    gather = RectGather(buffer, rows=SLOTS)
    counts = np.stack(
        [
            gather.observations(np.full(SLOTS, cycle, dtype=np.int64), slots)
            .mask.sum(-1)
            .numpy()
            for cycle in range(CYCLES + 1)
        ]
    )

    buffer.set_n_legal(torch.from_numpy(counts))

    assert buffer.n_legal.shape == (CYCLES, SLOTS)
    assert buffer.n_legal.dtype == np.int16
    assert (buffer.n_legal == counts[:CYCLES]).all()
    assert int(buffer.n_legal.min()) > 1, (
        "every cell of this rectangle had a choice, so a column of ones would agree with a "
        "column that was never written"
    )

    buffer.begin_iteration(plan_for(SLOTS, iteration=1), CYCLES)

    assert not buffer.n_legal.any()


def test_an_opponent_s_rows_are_stored_and_simply_not_trained_on(rect: Fixture) -> None:
    """Every row of the rectangle is stored, including the seats a frozen policy played."""
    buffer = fill_iteration(rect, opponent_slots=(2,))
    row = buffer.layout.row_index(0, 2)
    assert buffer.obs_view[row].any()
    assert not buffer.trainable()[:, 2].any()


def test_a_dead_worker_s_cells_are_invalid_and_do_not_reach_the_update(rect: Fixture) -> None:
    buffer = fill_iteration(rect, dead_from=3)
    trainable = buffer.trainable()
    assert trainable[:3].all()
    assert not trainable[3:].any()


# --------------------------------------------------------------------------
# Minibatching
# --------------------------------------------------------------------------


def collect(buffer: RectBuffer, batch_size: int, minibatch_size: int, epochs: int):
    def rng_for_epoch(epoch: int) -> np.random.Generator:
        return derive_generator(SEED, f"ppo/minibatch/iteration/0/epoch/{epoch}")

    return list(buffer.batches(batch_size, minibatch_size, epochs, rng_for_epoch))


def test_every_trainable_cell_is_trained_on_once_an_epoch_and_nothing_is_dropped(
    rect: Fixture,
) -> None:
    buffer = fill_iteration(rect)
    epochs, batch_size, minibatch_size = 3, 7, 3
    batches = collect(buffer, batch_size, minibatch_size, epochs)
    seen: list[int] = []
    for batch in batches:
        for minibatch in batch:
            seen.extend(minibatch.cells.cpu().numpy().tolist())
    cells = np.flatnonzero(buffer.trainable().reshape(-1))
    counts = np.bincount(np.asarray(seen), minlength=CYCLES * SLOTS)
    assert counts[cells].tolist() == [epochs] * cells.size
    assert counts.sum() == epochs * cells.size


def test_an_epoch_with_nothing_trainable_yields_no_batches(rect: Fixture) -> None:
    """The claim is written in three documents and was held by nothing.

    ``api/buffer.py``'s contract, the implementation's own docstring and the coordinator all say
    an epoch with no trainable rows yields no batches. The integrator restored the behaviour this
    replaced -- one empty batch per epoch -- and the whole suite stayed green, which is how a
    contract quietly stops being true. It is about the contract rather than about a run: the
    coordinator's invariants refuse an iteration this short long before the update sees it.
    """
    buffer = fill_iteration(rect)
    buffer.group[: buffer.cycles] = GROUP_DEAD
    assert int(buffer.trainable().sum()) == 0

    assert collect(buffer, 7, 3, 3) == []


def test_an_epoch_is_cut_into_whole_batches_with_the_rows_over_spread_across_them(
    rect: Fixture,
) -> None:
    """Every optimizer step sees at least a full batch, and the count is a property of the config.

    Cutting at every ``batch_size`` rows left the rows over as a batch of their own, and a batch
    is one optimizer step whatever it holds: a few dozen leftover rows moved the weights as far
    as a full batch did, on a gradient estimated from a fraction of the sample. Worse for
    comparing runs, the number of steps then moved between iterations as the trainable count
    crossed a multiple of the batch size.
    """
    buffer = fill_iteration(rect)
    epochs, batch_size = 2, 7
    total = int(buffer.trainable().sum())
    per_epoch = total // batch_size
    assert per_epoch >= 2 and total % batch_size, "this fixture must have rows over to spread"

    batches = collect(buffer, batch_size, 3, epochs)
    assert len(batches) == epochs * per_epoch
    sizes = [batch.n_samples for batch in batches]
    assert min(sizes) >= batch_size
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == epochs * total

    # An epoch that cannot fill one batch is one batch, which is what it already was.
    assert len(collect(buffer, total * 2, 3, 1)) == 1


def test_every_minibatch_is_weighted_by_its_share_of_its_own_batch(rect: Fixture) -> None:
    buffer = fill_iteration(rect)
    batch_size, minibatch_size = 7, 3
    for batch in collect(buffer, batch_size, minibatch_size, 1):
        weights = []
        counts = []
        for minibatch in batch:
            weights.append(minibatch.weight)
            counts.append(minibatch.n)
            assert minibatch.weight == pytest.approx(minibatch.n / batch.n_samples)
        assert sum(counts) == batch.n_samples
        assert sum(weights) == pytest.approx(1.0)
        assert len(batch) == math.ceil(batch.n_samples / minibatch_size)


def test_a_minibatch_carries_the_scalars_of_its_own_cells(rect: Fixture) -> None:
    buffer = fill_iteration(rect)
    buffer.set_values(torch.zeros((CYCLES + 1, SLOTS)))
    buffer.set_advantages(
        torch.from_numpy(buffer.reward.copy()), torch.from_numpy(buffer.reward.copy() * 2)
    )
    for batch in collect(buffer, 7, 3, 1):
        for minibatch in batch:
            cells = minibatch.cells.cpu().numpy()
            assert np.allclose(
                minibatch.log_probs.cpu().numpy(), buffer.log_prob.reshape(-1)[cells]
            )
            assert np.allclose(
                minibatch.actions.cpu().numpy(), buffer.action.reshape(-1)[cells]
            )
            assert np.allclose(
                minibatch.advantages.cpu().numpy(), buffer.advantage.reshape(-1)[cells]
            )
            assert np.allclose(minibatch.returns.cpu().numpy(), buffer.ret.reshape(-1)[cells])


def test_a_minibatch_s_observations_are_the_rows_its_cells_name(rect: Fixture) -> None:
    buffer = fill_iteration(rect)
    spec = rect.spec
    for batch in collect(buffer, 7, 3, 1):
        for minibatch in batch:
            cells = minibatch.cells.cpu().numpy()
            spatial = minibatch.obs.spatial.cpu().numpy()
            assert spatial.shape == (minibatch.n, spec.n_planes, *spec.tiles)
            assert minibatch.obs.vector.shape == (minibatch.n, spec.vector_size)
            assert minibatch.obs.mask.shape == (minibatch.n, spec.n_actions)
            for row, cell in enumerate(cells):
                index = int(cell) % len(rect.observations)
                expected = rect.observations[index]["spatial"]
                byte_planes = list(rect.codec.layout.u8_planes)
                assert np.array_equal(spatial[row, byte_planes], expected[byte_planes])
            assert minibatch.obs.mask[:, 0].all()


def test_the_staging_ring_does_not_alias(rect: Fixture) -> None:
    buffer = fill_iteration(rect)
    batches = collect(buffer, 7, 3, 1)
    ring = buffer._ring
    assert ring is not None
    pointers = [slab.data_ptr() for slab in ring.slabs]
    assert len(set(pointers)) == len(pointers)
    kept = [
        (minibatch.cells.clone(), minibatch.obs.spatial.clone())
        for batch in batches
        for minibatch in batch
    ]
    for cells, spatial in kept:
        assert cells.numel() == spatial.shape[0]
    drawn = [ring.next()[0].data_ptr() for _ in range(len(ring.slabs))]
    assert len(set(drawn)) == len(ring.slabs)


def test_the_advantage_inputs_are_the_columns_the_estimator_takes(rect: Fixture) -> None:
    buffer = fill_iteration(rect)
    values = torch.arange((CYCLES + 1) * SLOTS, dtype=torch.float32).reshape(CYCLES + 1, SLOTS)
    buffer.set_values(values)
    cells = np.array([[1, 0], [2, 3]], dtype=np.int64)
    buffer.set_final_values(cells, torch.tensor([4.0, 5.0]))
    inputs = buffer.advantage_inputs()
    assert inputs.values.shape == (CYCLES + 1, SLOTS)
    assert torch.equal(inputs.values, values)
    assert inputs.final_values[1, 0].item() == pytest.approx(4.0)
    assert inputs.final_values[2, 3].item() == pytest.approx(5.0)
    assert inputs.rewards.shape == (CYCLES, SLOTS)
    assert inputs.trainable.dtype == torch.bool


def test_the_truncated_cells_are_the_ones_the_critic_is_run_on(rect: Fixture) -> None:
    """The cell is the one that was truncated, not the one the truncation was reported at.

    ``V(final_obs)`` is stored against it, and the estimator reads the flag and that value at
    the same index, so a cell one row out bootstraps a row from a state it never reached.
    """
    buffer = rect.buffer
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    slots = np.array([0, 1], dtype=np.int64)
    buffer.record_round(
        round_for(
            2,
            slots,
            rows=slots,
            truncated=np.array([True, False]),
        ),
        actions=np.zeros(2, dtype=np.int16),
        log_probs=np.zeros(2, dtype=np.float32),
    )
    assert buffer.truncated_cells().tolist() == [[1, 0]]


# --------------------------------------------------------------------------
# The frame stack is a gather
# --------------------------------------------------------------------------


def test_at_a_frame_stack_of_one_a_cell_is_its_own_row(
    env_spec: EnvSpec, observations: list[dict[str, np.ndarray]]
) -> None:
    built = Fixture(env_spec, observations, frame_stack=1)
    try:
        assert built.buffer.layout.history_rows == 0
        assert built.buffer.layout.rows == CYCLES + 1
        buffer = fill_iteration(built)
        for batch in collect(buffer, 8, 4, 1):
            for minibatch in batch:
                assert minibatch.obs.spatial.shape[1] == env_spec.n_planes
    finally:
        built.close()


def test_a_stacked_cell_gathers_the_row_below_it_and_zero_fills_across_an_episode(
    env_spec: EnvSpec, observations: list[dict[str, np.ndarray]]
) -> None:
    built = Fixture(env_spec, observations, frame_stack=2)
    try:
        buffer = built.buffer
        built.fill()
        buffer.begin_iteration(plan_for(SLOTS), CYCLES)
        slots = np.arange(SLOTS, dtype=np.int64)
        # The episode of every slot ends ON row 1 -- reported by the round at cycle 2, which is
        # the round the ending step arrived at -- so row 2 opens a new one and has no history.
        ends_at = 2
        for cycle in range(CYCLES + 1):
            trailing = cycle == CYCLES
            ends = np.full(SLOTS, 1 if cycle == ends_at else 0, dtype=np.int8)
            buffer.record_round(
                round_for(
                    cycle,
                    slots,
                    rows=slots,
                    episode_end=ends,
                    terminated=np.full(SLOTS, cycle == ends_at),
                ),
                actions=None if trailing else np.zeros(SLOTS, dtype=np.int16),
                log_probs=None if trailing else np.zeros(SLOTS, dtype=np.float32),
            )
        planes = env_spec.n_planes
        gathered: dict[int, Any] = {}
        for batch in collect(buffer, 64, 64, 1):
            for minibatch in batch:
                for row, cell in enumerate(minibatch.cells.cpu().numpy()):
                    gathered[int(cell)] = minibatch.obs.spatial[row]
        for slot in range(SLOTS):
            first = gathered[1 * SLOTS + slot]
            assert first[planes:].abs().sum() > 0, "cycle 1 stacks the row of cycle 0"
            after = gathered[2 * SLOTS + slot]
            assert after[planes:].abs().sum() == 0, "cycle 2 is a new episode: a zero history"
            opening = gathered[0 * SLOTS + slot]
            assert opening[planes:].abs().sum() == 0, "there was no previous iteration"
    finally:
        built.close()


def test_the_history_rows_are_carried_down_when_an_iteration_opens(
    env_spec: EnvSpec, observations: list[dict[str, np.ndarray]]
) -> None:
    built = Fixture(env_spec, observations, frame_stack=2)
    try:
        buffer = built.buffer
        built.fill()
        slots = np.arange(SLOTS, dtype=np.int64)
        buffer.begin_iteration(plan_for(SLOTS), CYCLES)
        record_clean_iteration(buffer, slots)
        last = buffer.obs_view[buffer.layout.row_index(CYCLES - 1, 0)].copy()
        buffer.begin_iteration(plan_for(SLOTS, iteration=1), CYCLES)
        carried = buffer.obs_view[buffer.layout.row_index(-1, 0)].copy()
        assert np.array_equal(carried, last)
    finally:
        built.close()


def test_a_checkpoint_round_trips_the_history_it_carries(
    env_spec: EnvSpec, observations: list[dict[str, np.ndarray]], tmp_path: Any
) -> None:
    """A resumed run stacks the rows its predecessor collected, not a zero frame.

    The bytes alone are not the history: a carried row is stacked only where the gather can see
    that it was collected and that no episode ended on it. So the test does not stop at the
    bytes -- it opens the first iteration of the resumed run, which is where a rectangle that
    carried its own empty cycles down would overwrite them, and gathers cycle zero.
    """
    from pathlib import Path

    folder = Path(str(tmp_path)) / "buffer"
    built = Fixture(env_spec, observations, frame_stack=2)
    other = None
    slots = np.arange(SLOTS, dtype=np.int64)
    try:
        built.fill()
        buffer = built.buffer
        buffer.begin_iteration(plan_for(SLOTS), CYCLES)
        record_clean_iteration(buffer, slots)
        # Opening the second iteration is what carries the first one's last cycles down.
        buffer.begin_iteration(plan_for(SLOTS, iteration=1), CYCLES)
        rows = buffer.history_rows * SLOTS
        saved = buffer.obs_view[:rows].copy()
        assert saved.any(), "the carry put a collected row below cycle zero"
        buffer.save_checkpoint(folder)

        other = Fixture(env_spec, observations, frame_stack=2)
        other.fill()
        other.buffer.load_checkpoint(folder, strict=True)
        assert np.array_equal(other.buffer.obs_view[:rows].copy(), saved)
        assert other.buffer.iteration == buffer.iteration

        other.buffer.begin_iteration(plan_for(SLOTS, iteration=2), CYCLES)
        assert np.array_equal(other.buffer.obs_view[:rows].copy(), saved)
        record_clean_iteration(other.buffer, slots)
        planes = env_spec.n_planes
        gathered: dict[int, Any] = {}
        for batch in collect(other.buffer, 64, 64, 1):
            for minibatch in batch:
                for row, cell in enumerate(minibatch.cells.cpu().numpy()):
                    gathered[int(cell)] = minibatch.obs.spatial[row]
        for slot in range(SLOTS):
            opening = gathered[slot]
            assert opening[planes:].abs().sum() > 0, "cycle 0 stacks the carried row"
    finally:
        built.close()
        if other is not None:
            other.close()
