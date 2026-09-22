"""Which row of the rectangle a transition's scalars are stored at.

A round carries a state and the step that arrived at it, and those two things belong to
different timesteps: the observation is the round's own, the reward and the two done flags and
the deploy status are the step that began one cycle earlier. The rectangle stores one timestep
per row -- the observation, the action taken in it, the reward that action earned, and the flag
that says whether it ended the episode -- so the scalars a round carries go one row below the
observation it carries.

Everything here is graded against the environment rather than against the buffer. The reward
function names the tick each step ended at, so every row can be checked against its own
observation's clock; the episode records come out of the environment's own terminal statistics;
and the deploy status is the engine's answer to the command in that row, which is -1 exactly
when the row played no card. A test that compared the buffer to the collector would agree with
itself whichever way round the two were.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from rollout_support import CODEC, ReferenceCodec, preflight, rollout_config
from royalelearn.api.rollout import (
    EPISODE_END_NONE,
    GROUP_LEARNER,
    Step,
)
from royalelearn.rollout.envspec import ComponentSpec
from royalelearn.rollout.inline import (
    InlineRolloutSource,
    assignments_constant_within_episodes,
)
from royalelearn.rollout.plan import SlotPlanner

torch = pytest.importorskip("torch")

from royalelearn.learn.buffer import RectBuffer  # noqa: E402
from royalelearn.learn.gae import gae_recursion  # noqa: E402

#: Decisions an episode is cut off after. Seven, so that three whole episodes fit in the
#: rectangle and the third of them ends on its very last row -- which is the row whose scalars
#: only the trailing round can supply.
MAX_STEPS = 7
CYCLES = 3 * MAX_STEPS
GAMMA = 0.97

REWARD = ComponentSpec(cls="rollout_support.TickReward")


class Collected:
    """One iteration, collected into a real rectangle the way the coordinator collects it."""

    def __init__(self) -> None:
        self.config = rollout_config(
            workers=1,
            games_per_worker=2,
            shards_per_worker=1,
            max_steps=MAX_STEPS,
            reward=REWARD,
            stagger_first_reset=False,
        )
        self.report = preflight(self.config)
        self.spec = self.report.spec
        self.geometry = self.report.geometry
        self.planner = SlotPlanner(self.geometry, self.config.master_seed)
        self.codec = ReferenceCodec(self.report.table)
        self.codec.bind(self.spec, self.report.table)
        self.buffer = RectBuffer(
            self.spec,
            self.codec,
            run_id="align",
            cycles=CYCLES,
            n_slots=self.geometry.n_slots,
        )
        self.source = InlineRolloutSource(
            self.config, self.spec, self.report.table, run_id="align", codec=CODEC
        )
        self.noop = 0
        #: The packed observation each truncated row was bootstrapped from, by (row, slot).
        self.finals: dict[tuple[int, int], np.ndarray] = {}
        self.episodes: list[Any] = []

    # -- collecting ---------------------------------------------------------

    def _actions(self, round_: Any) -> np.ndarray:
        """Play a card wherever one is legal, so that the deploy status is not all one value."""
        actions = np.zeros(round_.slots.size, dtype=np.int16)
        for index, row in enumerate(round_.obs_rows):
            mask = self.codec.unpack_row(self.buffer.obs_view, int(row))["action_mask"]
            legal = np.flatnonzero(mask)
            legal = legal[legal != self.noop]
            if legal.size:
                actions[index] = int(legal[legal.size // 2])
        return actions

    def run(self) -> None:
        plan = self.planner.mirror_plan(0)
        self.buffer.begin_iteration(plan, CYCLES)
        self.source.begin_iteration(plan, self.buffer, 0)
        empty = np.zeros(0, dtype=np.int8)
        for _cycle in range(CYCLES):
            for _shard in range(self.geometry.shards_per_worker):
                round_ = self.source.next_round(30.0)
                actions = self._actions(round_)
                self.source.submit(
                    Step(
                        actions=actions,
                        gamma=GAMMA,
                        group=empty,
                        opponent_ix=empty,
                        learner_seat=empty,
                    )
                )
                self.buffer.record_round(
                    round_, actions, np.zeros(actions.size, dtype=np.float32)
                )
                self._take_finals(round_)
                self.episodes.extend(round_.episodes)
        for round_ in self.source.finish_iteration():
            self.buffer.record_round(round_, None, None)
            self._take_finals(round_)
            self.episodes.extend(round_.episodes)

    def _take_finals(self, round_: Any) -> None:
        slots, rows = self.source.final_rows(round_)
        for index, slot in enumerate(np.asarray(slots).tolist()):
            self.finals[(round_.cycle - 1, int(slot))] = np.array(rows[index], copy=True)

    def close(self) -> None:
        self.source.close()
        self.buffer.close()

    # -- what the assertions read -------------------------------------------

    @property
    def decision_ticks(self) -> int:
        return int(self.spec.decision_ticks)

    def end_rows(self, slot: int) -> np.ndarray:
        """The rows at which an episode of this slot ended, ascending."""
        return np.flatnonzero(self.buffer.episode_end[:, slot] != EPISODE_END_NONE)

    def whole_episodes(self, slot: int) -> list[tuple[int, int]]:
        """``(first row, last row)`` of every episode that started and ended in this rectangle.

        The first episode of the rectangle is whole too here -- the shards do not stagger -- but
        it is read off the ends like the rest rather than assumed, so a rectangle that opened
        mid-episode would simply offer one episode fewer.
        """
        ends = self.end_rows(slot)
        return [(int(a) + 1, int(b)) for a, b in itertools.pairwise(ends)]


@pytest.fixture(scope="module")
def collected() -> Any:
    run = Collected()
    try:
        run.run()
        yield run
    finally:
        run.close()


# --------------------------------------------------------------------------
# One row, one timestep
# --------------------------------------------------------------------------


def test_a_rows_reward_is_the_one_its_own_action_earned(collected: Any) -> None:
    """Every row's reward is the tick its own step ended at, and no other row's.

    The reward function returns the engine tick the step reached, so a row whose observation
    stands at tick ``c`` and whose action was played there must hold ``c + decision_ticks``.
    The half distinguishes the two seats of a battle, so a column that took its neighbour's
    rewards fails here as loudly as one that took its neighbour's cycle.
    """
    buffer = collected.buffer
    step = collected.decision_ticks
    for slot in range(collected.geometry.n_slots):
        seat = slot % 2
        expected = buffer.tick[:CYCLES, slot].astype(np.float64) + step + 0.5 * seat
        stored = buffer.reward[:CYCLES, slot].astype(np.float64)
        wrong = np.flatnonzero(stored != expected)
        assert wrong.size == 0, (
            f"slot {slot}: {wrong.size} row(s) hold a reward another row earned; row "
            f"{int(wrong[0])} stands at tick {int(buffer.tick[wrong[0], slot])} and holds "
            f"{stored[wrong[0]]}, where its own step earned {expected[wrong[0]]}"
        )


def test_a_rows_deploy_status_is_the_engines_answer_to_its_own_command(
    collected: Any,
) -> None:
    """-1 is "this row played no card", and the row that played one is the row that has it.

    Off the rows that ended an episode. The vec env resets inside the step that ends one, so
    the batched info that step returns is already the fresh episode's and the status of a
    terminal command reaches the harness only through the episode record. A row that ended an
    episode therefore reads -1 whatever it played -- the environment's reporting, not this
    rectangle's alignment -- and it is excluded here rather than quietly passed over.
    """
    buffer = collected.buffer
    ends = buffer.episode_end[:CYCLES] != EPISODE_END_NONE
    played = (buffer.action[:CYCLES] != collected.noop) & ~ends
    answered = (buffer.deploy_status[:CYCLES] != -1) & ~ends
    assert played.sum() >= CYCLES // 2, "too few rows played a card for the status to say much"
    assert not played.all(), "every row played a card, so the status proved nothing"
    assert np.array_equal(played, answered), (
        f"{int((played != answered).sum())} row(s) carry the engine's answer to another row's "
        "command"
    )
    assert (buffer.deploy_status[:CYCLES][ends] == -1).all()


def test_the_tick_is_the_rows_own_observation_and_not_the_steps(collected: Any) -> None:
    """A new episode's first row reads tick zero: the clock belongs to the state, not the step.

    It is the one scalar of a round that does NOT move down a row, and saying so with the
    rectangle in hand is what stops the fix from sliding the whole record.
    """
    buffer = collected.buffer
    step = collected.decision_ticks
    for slot in range(collected.geometry.n_slots):
        for first, last in collected.whole_episodes(slot):
            ticks = buffer.tick[first : last + 1, slot]
            assert ticks[0] == 0, "an episode's first row does not read tick zero"
            assert np.array_equal(ticks, np.arange(ticks.size) * step)


# --------------------------------------------------------------------------
# Where an episode ends
# --------------------------------------------------------------------------


def test_the_flag_that_ended_an_episode_sits_on_the_last_row_of_that_episode(
    collected: Any,
) -> None:
    """The row whose action ended the episode carries the flag, and the next row carries none.

    Put the flag on the first row of the NEXT episode instead and the estimator cuts the
    trajectory one step late: the ending row bootstraps from a state in another battle, and the
    row that opens the new episode is told it has no future.
    """
    buffer = collected.buffer
    ends = 0
    for slot in range(collected.geometry.n_slots):
        for first, last in collected.whole_episodes(slot):
            ends += 1
            done = buffer.terminated[:CYCLES, slot] | buffer.truncated[:CYCLES, slot]
            assert done[last], f"slot {slot}: the episode ending on row {last} has no flag there"
            inside = done[first:last]
            assert not inside.any(), f"slot {slot}: a done flag sits inside an episode"
            assert buffer.tick[last, slot] == (MAX_STEPS - 1) * collected.decision_ticks
    assert ends >= 2, "fewer than two whole episodes, so nothing was pinned"


def test_an_episodes_stored_rewards_sum_to_the_return_the_environment_reported(
    collected: Any,
) -> None:
    """The rows an episode owns are exactly the rows whose rewards make up its return.

    ``undiscounted_return`` is summed by the worker as the episode runs, out of the same
    rewards, so this is the rectangle checked against the environment's own total rather than
    against a second walk of the rectangle.
    """
    buffer = collected.buffer
    by_slot: dict[int, list[Any]] = {}
    for record in collected.episodes:
        by_slot.setdefault(record.slot, []).append(record)
    checked = 0
    for slot, records in by_slot.items():
        ends = collected.end_rows(slot)
        first_ordinal = records[0].ordinal
        for record in records:
            position = record.ordinal - first_ordinal
            if position == 0 or position >= ends.size:
                continue  # the rectangle opened inside this one, so it does not own every row
            last = int(ends[position])
            first = int(ends[position - 1]) + 1
            assert last - first + 1 == record.episode_steps
            total = float(buffer.reward[first : last + 1, slot].sum())
            assert total == record.undiscounted_return
            checked += 1
    assert checked >= 2, "fewer than two whole episodes were checked"


def test_the_last_transition_of_the_iteration_is_kept(collected: Any) -> None:
    """The trailing round is the only place row ``T-1``'s reward can come from.

    Drop it and every iteration trains its last cycle on a reward of zero, which is a reward
    the environment never gave.
    """
    buffer = collected.buffer
    last = CYCLES - 1
    assert buffer.valid[last].all(), "the last row of the rectangle was never completed"
    expected = buffer.tick[last].astype(np.float64) + collected.decision_ticks
    expected += 0.5 * (np.arange(collected.geometry.n_slots) % 2)
    assert np.array_equal(buffer.reward[last].astype(np.float64), expected)
    assert (buffer.terminated[last] | buffer.truncated[last]).any(), (
        "no episode ended on the rectangle's last row, so the trailing round's flags were not "
        "under test"
    )


def test_a_battle_changes_hands_only_where_an_episode_ended(collected: Any) -> None:
    """The invariant the coordinator asserts, over a rectangle in the same convention it is."""
    buffer = collected.buffer
    assignments_constant_within_episodes(buffer.group[:CYCLES], buffer.episode_end)
    assert (buffer.group[:CYCLES] == GROUP_LEARNER).all()


# --------------------------------------------------------------------------
# End to end: the advantage of the row that ended
# --------------------------------------------------------------------------


def test_the_advantage_of_a_truncated_row_pairs_its_own_reward_with_its_own_value(
    collected: Any,
) -> None:
    """``A = r + gamma * V(final) - V(s)`` for the row the truncation ended, to the value.

    The values are the test's own distinct numbers and the reward is the environment's, so the
    equality holds only if the row's reward, its truncation flag, its bootstrap value and its
    own value are all the same timestep's. It is the pairing the estimator actually consumes,
    which is one step further than asserting what the buffer stores.
    """
    buffer = collected.buffer
    slots = collected.geometry.n_slots
    # Distinct per cell and distinct from any reward: V(s_t) at cell (t, r) is 1000 + 10t + r,
    # so a value taken from the row above or the column over is a different number.
    grid = 1000.0 + 10.0 * np.arange(CYCLES + 1)[:, None] + np.arange(slots)[None, :]
    buffer.set_values(torch.from_numpy(grid.astype(np.float32)))
    cells = buffer.truncated_cells()
    assert cells.size, "nothing truncated, so the bootstrap path was never taken"
    finals = np.array(
        [7000.0 + 10.0 * cycle + slot for cycle, slot in cells.tolist()], dtype=np.float32
    )
    buffer.set_final_values(cells, torch.from_numpy(finals))
    inputs = buffer.advantage_inputs()
    advantages, _ = gae_recursion(
        rewards=inputs.rewards,
        values=inputs.values,
        final_values=inputs.final_values,
        terminated=inputs.terminated,
        truncated=inputs.truncated,
        gamma=GAMMA,
        lam=0.95,
    )
    got = advantages.numpy()
    for index, (cycle, slot) in enumerate(cells.tolist()):
        reward = float(buffer.reward[cycle, slot])
        expected = reward + GAMMA * float(finals[index]) - grid[cycle, slot]
        assert got[cycle, slot] == pytest.approx(expected, abs=1e-2), (
            f"the advantage of the truncated cell ({cycle}, {slot}) is not its own reward "
            "against its own value"
        )
        assert (cycle, slot) in collected.finals, (
            "the observation this row was bootstrapped from was never handed over"
        )
