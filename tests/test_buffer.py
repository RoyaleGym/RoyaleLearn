"""The experience rectangle: what a round writes, what a minibatch gathers, and what neither
does.

The rectangle is built here at a handful of cycles and a handful of slots, and filled the way a
worker fills it -- by packing real observations into the shared block at the row a cell's index
names. Nothing in this file knows a width: the row size is the codec's, the plane count is the
space's, and the footprint is checked against the arithmetic recomputed from both.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from royalelearn.api.rollout import (
    GROUP_DEAD,
    GROUP_LEARNER,
    ROLE_MIRROR,
    Assignment,
    EnvSpec,
    RolloutRound,
    SlotPlan,
)
from royalelearn.rollout.codec import SpatialObsCodec
from royalelearn.seeding import derive_generator

torch = pytest.importorskip("torch")

import msgspec  # noqa: E402

from royalelearn.learn.buffer import RectBuffer  # noqa: E402

SEED = 20260921
CYCLES = 6
SLOTS = 8
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


def plan_for(slots: int, *, iteration: int = 0) -> SlotPlan:
    """A mirror plan: every battle is the learner against itself."""
    assignments = tuple(
        Assignment(
            battle=slot // 2,
            ordinal=0,
            role=ROLE_MIRROR,
            opponent_id=None,
            group=(GROUP_LEARNER, GROUP_LEARNER),
            learner_seat=slot % 2,
        )
        for slot in range(slots)
    )
    return SlotPlan(
        iteration=iteration,
        n_battles=slots // 2,
        n_slots=slots,
        assignment=assignments,
        resident_snapshots=(),
    )


def round_for(
    cycle: int,
    slots: np.ndarray,
    *,
    rows: np.ndarray,
    group: int = GROUP_LEARNER,
    reward: np.ndarray | None = None,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
    valid: bool = True,
    episode_end: np.ndarray | None = None,
    tick: np.ndarray | None = None,
) -> RolloutRound:
    count = slots.size
    zeros = np.zeros(count, dtype=bool)
    return RolloutRound(
        cycle=cycle,
        shard=0,
        slots=slots,
        obs_rows=rows,
        group=np.full(count, group, dtype=np.int8),
        reward=reward if reward is not None else np.zeros(count, dtype=np.float32),
        terminated=terminated if terminated is not None else zeros,
        truncated=truncated if truncated is not None else zeros,
        valid=np.full(count, valid, dtype=bool),
        deploy_status=np.full(count, -1, dtype=np.int8),
        tick=tick if tick is not None else np.full(count, cycle, dtype=np.int32),
        episode_end=episode_end
        if episode_end is not None
        else np.zeros(count, dtype=np.int8),
    )


class Fixture:
    """A rectangle, its codec, and the rows a worker would have written into it."""

    def __init__(
        self,
        spec: EnvSpec,
        observations: list[dict[str, np.ndarray]],
        *,
        cycles: int = CYCLES,
        slots: int = SLOTS,
        frame_stack: int = 1,
    ) -> None:
        self.spec = msgspec.structs.replace(spec, frame_stack=frame_stack)
        self.codec = SpatialObsCodec()
        # A handful of rows is all this file needs a table for -- it is how a row is sized and
        # packed. What a table may be decided from for a run is asserted in tests/test_codec.py.
        self.codec.table(self.spec, observations, min_states=0)
        self.observations = observations
        self.buffer = RectBuffer(
            self.spec,
            self.codec,
            run_id=uuid.uuid4().hex[:12],
            cycles=cycles,
            n_slots=slots,
        )
        self.buffer.set_static_planes(self.codec.static_planes(observations[0]))

    def fill(self, cycles: int | None = None) -> None:
        """Pack an observation into every cell of the rectangle, as a worker would."""
        layout = self.buffer.layout
        view = memoryview(self.buffer.shm.buf)[layout.obs_offset :]
        try:
            top = layout.cycles if cycles is None else cycles
            for cycle in range(top + 1):
                for slot in range(layout.n_slots):
                    index = (cycle * layout.n_slots + slot) % len(self.observations)
                    self.codec.pack(
                        self.observations[index], view, layout.row_index(cycle, slot)
                    )
        finally:
            view.release()

    def close(self) -> None:
        self.buffer.close()


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
    assert np.array_equal(buffer.reward[2, slots], rewards)
    assert np.array_equal(buffer.terminated[2, slots], terminated)
    assert np.array_equal(buffer.action[2, slots], np.array([3, 0, 7]))
    assert buffer.valid[2, slots].all()
    untouched = np.setdiff1d(np.arange(SLOTS), slots)
    assert not buffer.valid[2, untouched].any()
    assert not buffer.valid[np.arange(CYCLES) != 2].any()
    assert np.array_equal(buffer.group[2, untouched], np.full(untouched.size, GROUP_DEAD))


def test_the_bootstrap_cycle_carries_observations_and_no_scalars(rect: Fixture) -> None:
    buffer = rect.buffer
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    slots = np.arange(SLOTS, dtype=np.int64)
    buffer.record_round(
        round_for(CYCLES, slots, rows=slots),
        actions=np.zeros(SLOTS, dtype=np.int16),
        log_probs=np.zeros(SLOTS, dtype=np.float32),
    )
    assert not buffer.valid.any()
    with pytest.raises(IndexError):
        buffer.record_round(
            round_for(CYCLES + 1, slots, rows=slots),
            actions=np.zeros(SLOTS, dtype=np.int16),
            log_probs=np.zeros(SLOTS, dtype=np.float32),
        )


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
    """Collect a whole iteration: every slot every cycle, some of them an opponent's."""
    buffer = rect.buffer
    rect.fill()
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    rng = derive_generator(SEED, "test/buffer/fill")
    slots = np.arange(SLOTS, dtype=np.int64)
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    for slot in opponent_slots:
        groups[slot] = 0  # a resident snapshot's index: not the learner
    for cycle in range(CYCLES):
        for slot in slots:
            record = round_for(
                cycle,
                np.array([slot], dtype=np.int64),
                rows=np.array([buffer.layout.row_index(cycle, int(slot))]),
                group=int(groups[slot]),
                reward=rng.normal(size=1).astype(np.float32),
                valid=not (0 <= dead_from <= cycle),
            )
            buffer.record_round(
                record,
                actions=rng.integers(0, 2, size=1).astype(np.int16),
                log_probs=rng.normal(size=1).astype(np.float32),
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


def test_a_batch_never_straddles_an_epoch(rect: Fixture) -> None:
    buffer = fill_iteration(rect)
    epochs, batch_size = 2, 7
    total = int(buffer.trainable().sum())
    batches = collect(buffer, batch_size, 3, epochs)
    per_epoch = math.ceil(total / batch_size)
    assert len(batches) == epochs * per_epoch
    sizes = [batch.n_samples for batch in batches]
    remainder = total - batch_size * (per_epoch - 1)
    expected = [batch_size] * (per_epoch - 1) + [remainder]
    assert sizes == expected * epochs
    assert sum(sizes) == epochs * total


def test_the_remainder_is_a_smaller_batch_weighted_by_its_true_count(rect: Fixture) -> None:
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
    buffer = rect.buffer
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    slots = np.array([0, 1], dtype=np.int64)
    buffer.record_round(
        round_for(
            1,
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
        for cycle in range(CYCLES):
            # The episode of every slot ends at cycle 1, so cycle 2 has no history to stack.
            ends = np.full(SLOTS, 1 if cycle == 1 else 0, dtype=np.int8)
            buffer.record_round(
                round_for(
                    cycle,
                    slots,
                    rows=slots,
                    episode_end=ends,
                    terminated=np.full(SLOTS, cycle == 1),
                ),
                actions=np.zeros(SLOTS, dtype=np.int16),
                log_probs=np.zeros(SLOTS, dtype=np.float32),
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
        for cycle in range(CYCLES):
            buffer.record_round(
                round_for(cycle, slots, rows=slots),
                actions=np.zeros(SLOTS, dtype=np.int16),
                log_probs=np.zeros(SLOTS, dtype=np.float32),
            )
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
    actions, log_probs = np.zeros(SLOTS, dtype=np.int16), np.zeros(SLOTS, dtype=np.float32)
    try:
        built.fill()
        buffer = built.buffer
        buffer.begin_iteration(plan_for(SLOTS), CYCLES)
        for cycle in range(CYCLES):
            buffer.record_round(round_for(cycle, slots, rows=slots), actions, log_probs)
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
        for cycle in range(CYCLES):
            other.buffer.record_round(round_for(cycle, slots, rows=slots), actions, log_probs)
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
