"""Frame stacking: which rows a stacked cell gathers, and which it leaves alone.

The stack is a gather over the rectangle rather than a second copy, so what has to be asserted
is that the rows it reaches are the ones it means -- the previous decision of the same seat, and
at a stack of one the cell's own row and nothing else. The existing rectangle tests say whether
a history is present; these say what it is, which is the part a stride off by one slot or one
cycle would still pass.

Nothing here knows a width: the plane count and the vector width are the space's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from royalelearn.api.rollout import EnvSpec

torch = pytest.importorskip("torch")

from test_buffer import CYCLES, SLOTS, Fixture, collect, plan_for, round_for  # noqa: E402

#: Engine ticks between two decisions of one seat in these rectangles. Any spacing does: what a
#: stack has to show is that consecutive frames are consecutive decisions, whatever the gap.
DECISION_TICKS = 5


@pytest.fixture(scope="module")
def observations(mock_env_spec: Any) -> list[dict[str, np.ndarray]]:
    """Real observations to fill the rectangles with, from this module's own environment.

    One per cell and all different, which is what lets a gather that landed on the wrong row be
    told from one that landed on the right one.
    """
    env = mock_env_spec.build_vec(2)
    try:
        batches = [env.reset(seed=5)[0]]
        actions = np.zeros(env.num_envs, dtype=np.int64)
        for _ in range(CYCLES * SLOTS):
            batches.append(env.step(actions)[0])
    finally:
        env.close()
    return [{key: value[0] for key, value in batch.items()} for batch in batches]


def _play(built: Fixture, *, ticks: int = 1) -> Any:
    """Fill every cell, open an iteration and record a clean round at every cycle.

    The trailing round at cycle ``T`` is one of them: it has no row of its own, but it is what
    completes row ``T - 1``, and a row nothing completed is not a row the update gathers.
    """
    built.fill()
    buffer = built.buffer
    buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    slots = np.arange(SLOTS, dtype=np.int64)
    for cycle in range(CYCLES + 1):
        trailing = cycle == CYCLES
        buffer.record_round(
            round_for(
                cycle,
                slots,
                rows=slots,
                tick=np.full(SLOTS, cycle * ticks, dtype=np.int32),
            ),
            actions=None if trailing else np.zeros(SLOTS, dtype=np.int16),
            log_probs=None if trailing else np.zeros(SLOTS, dtype=np.float32),
        )
    return buffer


def _gathered(buffer: Any) -> dict[int, Any]:
    """Every trainable cell's stacked observation, by cell index."""
    out: dict[int, Any] = {}
    for batch in collect(buffer, 64, 64, 1):
        for minibatch in batch:
            for row, cell in enumerate(minibatch.cells.cpu().numpy()):
                out[int(cell)] = minibatch.obs.spatial[row]
    return out


def test_a_stacked_frame_is_the_same_seats_previous_decision(
    env_spec: EnvSpec, observations: list[dict[str, np.ndarray]]
) -> None:
    """Frame one of cell ``(t, r)`` is frame zero of cell ``(t - 1, r)``, and one decision older.

    The rectangle is filled with a different observation in every cell, so a gather that landed
    one slot over or one cycle out would produce a live-looking frame belonging to somebody
    else. Comparing the two gathers rather than the packed bytes is what makes this a statement
    about the stack and not about the packing.
    """
    built = Fixture(env_spec, observations, frame_stack=2)
    try:
        buffer = _play(built, ticks=DECISION_TICKS)
        planes = env_spec.n_planes
        gathered = _gathered(buffer)
        for cycle in range(1, CYCLES):
            for slot in range(SLOTS):
                cell = cycle * SLOTS + slot
                below = (cycle - 1) * SLOTS + slot
                assert torch.equal(gathered[cell][planes:], gathered[below][:planes])
                elapsed = int(buffer.tick[cycle, slot]) - int(buffer.tick[cycle - 1, slot])
                assert elapsed == DECISION_TICKS, "two stacked frames are consecutive decisions"
    finally:
        built.close()


def test_at_a_stack_of_one_a_cell_gathers_its_own_row_and_nothing_else(
    env_spec: EnvSpec, observations: list[dict[str, np.ndarray]]
) -> None:
    """The unstacked observation is exactly what the stacked one carries in front of its history.

    Two rectangles over the same rows, one at ``k = 1`` and one at ``k = 2``: every cell's
    single frame is the leading frame of its stacked twin, to the value.
    """
    planes = env_spec.n_planes
    single = Fixture(env_spec, observations, frame_stack=1)
    stacked = Fixture(env_spec, observations, frame_stack=2)
    try:
        one = _gathered(_play(single))
        two = _gathered(_play(stacked))
        assert set(one) == set(two)
        for cell, frame in one.items():
            assert frame.shape[0] == planes
            assert torch.equal(frame, two[cell][:planes])
    finally:
        single.close()
        stacked.close()


def test_the_vector_is_the_current_frames_and_is_not_stacked(
    env_spec: EnvSpec, observations: list[dict[str, np.ndarray]]
) -> None:
    """Elixir, hand and clock are already the present state; a stale copy of them is noise."""
    built = Fixture(env_spec, observations, frame_stack=2)
    try:
        buffer = _play(built)
        frames = built.spec.frame_stack
        seen = 0
        for batch in collect(buffer, 64, 64, 1):
            for minibatch in batch:
                assert minibatch.obs.vector.shape == (minibatch.n, env_spec.vector_size)
                assert minibatch.obs.spatial.shape[1] == frames * env_spec.n_planes
                assert minibatch.obs.mask.shape == (minibatch.n, env_spec.n_actions)
                seen += minibatch.n
        assert seen == int(buffer.trainable().sum())
    finally:
        built.close()
