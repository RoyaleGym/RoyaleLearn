"""What a worker process must be, and what the farm must do when one stops being it.

Two subjects. The first is hygiene: a worker holds environments and no policy, it pinned its
BLAS threads before numpy loaded, and the only way to know either is to ask the child itself.
The second is failure: both reference learners treat a dead worker as a permanent silent hang,
and everything below exists so that this one does not -- an exception becomes a typed failure
with the child's traceback, a restart is deterministic and counted, and a wait that would never
end ends at its deadline instead.

Marked ``slow``: every test here spawns processes and builds environments inside them.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from rollout_support import CODEC, SUPPORT_MODULE, SharedRectangle, preflight, rollout_config
from royalelearn.api.rollout import GROUP_DEAD, Step
from royalelearn.determinism import BLAS_THREAD_VARS
from royalelearn.errors import WorkerTimeout
from royalelearn.rollout.envspec import ComponentSpec
from royalelearn.rollout.farm import ProcessRolloutSource
from royalelearn.rollout.plan import SlotPlanner

pytestmark = pytest.mark.slow

CYCLES = 12
MAX_STEPS = 40


def _sources(reward: ComponentSpec | None = None, **overrides: Any) -> tuple[Any, Any, Any, Any]:
    """A farm and the rectangle it writes into, plus the report its workers were built from.

    Preflight runs against the healthy environment even when the farm is given a broken one:
    the gates are about the observation and the mask, and a component that fails on the fourth
    step of a battle is not something a start-up check is meant to find. The warm-up is off for
    the same reason -- a fault that arrived before the first round would be a start-up failure,
    which is a different subject from a worker that dies mid-iteration.
    """
    healthy = rollout_config(
        workers=2,
        games_per_worker=1,
        shards_per_worker=1,
        max_steps=MAX_STEPS,
        stagger_first_reset=False,
        **overrides,
    )
    report = preflight(healthy)
    config = rollout_config(
        workers=2,
        games_per_worker=1,
        shards_per_worker=1,
        max_steps=MAX_STEPS,
        reward=reward,
        stagger_first_reset=False,
        **overrides,
    )
    buffer = SharedRectangle(
        report.spec,
        run_id=f"hygiene-{int(time.time() * 1000) % 100000}",
        cycles=CYCLES,
        n_slots=report.geometry.n_slots,
        row_bytes=report.row_bytes,
    )
    source = ProcessRolloutSource(
        config, report.spec, report.table, run_id=buffer.layout.run_id, codec=CODEC
    )
    planner = SlotPlanner(report.geometry, config.master_seed)
    return source, buffer, report, planner


def _step(source: Any, timeout: float = 30.0) -> Any:
    """One round, answered with no-ops."""
    round_ = source.next_round(timeout)
    source.submit(
        Step(
            actions=np.zeros(round_.slots.size, dtype=np.int16),
            gamma=0.99,
            group=np.zeros(0, dtype=np.int8),
            opponent_ix=np.zeros(0, dtype=np.int8),
            learner_seat=np.zeros(0, dtype=np.int8),
        )
    )
    return round_


def test_a_worker_holds_no_policy_and_pinned_its_threads() -> None:
    """The child's own report of its interpreter, which is the only place it can come from.

    A worker that imported torch would cost about three hundred megabytes resident and three
    seconds of start-up for a library it never calls; a worker that imported numpy before the
    thread variables were set would have a BLAS pool sized by however busy the machine was.
    """
    source, buffer, report, planner = _sources()
    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        reports = source.startup_reports()
        assert len(reports) == report.geometry.workers
        for index, said in enumerate(reports):
            assert said is not None
            assert said.worker == index
            assert said.generation == 0
            assert not said.torch_loaded, "a rollout worker imported torch"
            assert not said.numpy_preloaded, "numpy was imported before the worker's preamble"
            assert said.thread_env == dict.fromkeys(BLAS_THREAD_VARS, "1")
            assert said.shards == report.geometry.shards_per_worker
            assert said.pid > 0
        assert len({said.pid for said in reports}) == len(reports)
    finally:
        source.close()
        buffer.close()


def test_an_exception_arrives_as_a_failure_with_its_traceback() -> None:
    """A worker that raises hands over what happened, and its slots go on arriving, invalid.

    The rectangle needs no special case for the cycles a dead worker missed: those cells are
    marked invalid and the learner drops them, which is a measured loss rather than a silent
    one. The environment given to the workers fails on the fourth step of a battle, so both of
    them fail: what is under test is the report, not which of them made it.
    """
    source, buffer, report, planner = _sources(
        reward=ComponentSpec(
            f"{SUPPORT_MODULE}.ExplodingReward", {"at_step": 3, "message": "a deliberate fault"}
        )
    )
    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        failures: list[Any] = []
        dead_round = None
        for _ in range(CYCLES):
            round_ = _step(source)
            found = source.drain_failures()
            if found:
                failures = found
                dead_round = round_
                break
        assert failures, "a worker raised and the farm noticed nothing"
        for failure in failures:
            assert failure.kind == "exception"
            assert "a deliberate fault" in failure.message
            assert "Traceback" in failure.message
            assert 0 <= failure.worker < report.geometry.workers
            # The failure knows which cycle the worker died on even when the round cannot: a
            # round every worker failed on has no cycle, because no worker published one.
            assert failure.cycle >= 0

        assert dead_round is not None
        for failure in failures:
            block = planner.slot_worker[dead_round.slots] == failure.worker
            assert not dead_round.valid[block].any()
            assert (dead_round.group[block] == GROUP_DEAD).all()

        worker = failures[0].worker
        assert source.restarts[worker] == 0
        source.restart(worker)
        assert source.restarts[worker] == 1
        assert source.generation[worker] == 1
        assert source.startup_reports()[worker].generation == 1
        assert source.rejoining[worker], "a restarted worker rejoins at the next plan"
    finally:
        source.close()
        buffer.close()


def test_a_dead_worker_is_restarted_and_the_run_goes_on() -> None:
    """One worker's process is killed outright, and the rest of the farm carries the run.

    This is the case the rectangle was shaped for: the dead worker's cells arrive marked
    invalid rather than not arriving, the live worker keeps publishing on its own schedule, and
    the restarted worker draws from the next generation of its own seed so the run is still a
    function of the master seed and a restart count.
    """
    source, buffer, report, planner = _sources(round_timeout_s=3.0)
    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        _step(source, timeout=10.0)
        victim = report.geometry.workers - 1
        source.workers[victim].process.terminate()

        with pytest.raises(WorkerTimeout) as raised:
            for _ in range(CYCLES):
                _step(source, timeout=3.0)
        assert raised.value.worker == victim
        failures = source.drain_failures()
        assert failures and failures[0].kind == "crash"
        assert "exit code" in failures[0].message

        after = _step(source, timeout=10.0)
        block = planner.slot_worker[after.slots] == victim
        assert not after.valid[block].any(), "a dead worker's slots stopped arriving"
        assert after.valid[~block].all(), "a live worker's slots were dropped with it"
        assert (after.group[block] == GROUP_DEAD).all()

        source.restart(victim)
        assert source.generation[victim] == 1
        assert source.rejoining[victim]
        source.begin_iteration(planner.mirror_plan(1), buffer, 1)
        rejoined = _step(source, timeout=10.0)
        assert rejoined.valid.all(), "the restarted worker did not rejoin"
        assert rejoined.cycle == 0
    finally:
        source.close()
        buffer.close()


def test_a_worker_that_stops_answering_times_out() -> None:
    """Every wait has a deadline, and what a deadline produces is a WorkerTimeout.

    The reference learners block forever here. The timeout names the worker, the shard and the
    cycle, and the failure it leaves behind says whether the process is still alive -- which is
    the difference between a hang to debug and a crash to restart.
    """
    hang = 5.0
    source, buffer, report, planner = _sources(
        reward=ComponentSpec(f"{SUPPORT_MODULE}.HangingReward", {"at_step": 2, "seconds": hang}),
        round_timeout_s=1.5,
    )
    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        started = time.monotonic()
        with pytest.raises(WorkerTimeout) as raised:
            for _ in range(CYCLES):
                _step(source, timeout=1.5)
        waited = time.monotonic() - started
        assert waited < hang + 5.0, "the wait outlasted the hang it was supposed to cut short"
        assert raised.value.timeout_s == pytest.approx(1.5)
        assert 0 <= raised.value.worker < report.geometry.workers
        failures = source.drain_failures()
        assert failures and failures[0].kind in ("timeout", "crash")
        assert "alive" in failures[0].message
    finally:
        source.close()
        buffer.close()
