"""What a worker process must be, and what the farm must do when one stops being it.

Two subjects. The first is hygiene: a worker holds environments and no policy, it pinned its
BLAS threads before numpy loaded, and the only way to know either is to ask the child itself;
and a worker with nothing to do sleeps, which only the operating system can say. The second is
failure: both reference learners treat a dead worker as a permanent silent hang, and everything
below exists so that this one does not -- an exception becomes a typed failure with the child's
traceback, a restart is deterministic and counted, and a wait that would never end ends at its
deadline instead.

Marked ``slow``: every test here spawns processes and builds environments inside them.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rollout_support import CODEC, SUPPORT_MODULE, SharedRectangle, preflight, rollout_config
from royalelearn.api.rollout import GROUP_DEAD, Defer, Step
from royalelearn.determinism import BLAS_THREAD_VARS
from royalelearn.errors import WorkerTimeout
from royalelearn.rollout.envspec import ComponentSpec
from royalelearn.rollout.farm import ProcessRolloutSource
from royalelearn.rollout.plan import SlotPlanner

pytestmark = pytest.mark.slow

CYCLES = 12
MAX_STEPS = 40

#: How long a worker is left with nothing to do, and the share of a core it may spend on that.
#: A worker that sleeps through the wait uses a fraction of a percent; one that spins uses
#: whatever the machine will give it.
IDLE_S = 2.0
IDLE_CPU_SHARE = 0.10

#: The access right that lets one process read another's times and nothing else.
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _cpu_seconds(pid: int) -> float:
    """The user and kernel time another process has used so far, in seconds.

    Read from the operating system, because a worker that spins is in no position to report
    that it does. No dependency the suite already has reads another process's times, and the
    two platforms that matter take a few lines each: Windows answers ``GetProcessTimes`` on a
    handle opened for querying, Linux keeps the ticks in ``/proc``. Anywhere else the test that
    needs this is skipped rather than guessed at.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        opened = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not opened:
            raise ctypes.WinError(ctypes.get_last_error())
        handle = wintypes.HANDLE(opened)
        times = [wintypes.FILETIME() for _ in range(4)]
        try:
            if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel32.CloseHandle(handle)
        _created, _exited, kernel, user = times
        ticks = sum(t.dwHighDateTime << 32 | t.dwLowDateTime for t in (kernel, user))
        return ticks / 1e7  # FILETIME counts 100 ns intervals
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        pytest.skip("this platform has no dependency-free way to read another process's times")
    # The command name in field two may hold spaces and parentheses; everything after its
    # closing parenthesis is space-separated, starting at field three.
    fields = stat.read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def _sources(
    reward: ComponentSpec | None = None, shards: int = 1, **overrides: Any
) -> tuple[Any, Any, Any, Any]:
    """A farm and the rectangle it writes into, plus the report its workers were built from.

    Preflight runs against the healthy environment even when the farm is given a broken one:
    the gates are about the observation and the mask, and a component that fails on the fourth
    step of a battle is not something a start-up check is meant to find. The warm-up is off for
    the same reason -- a fault that arrived before the first round would be a start-up failure,
    which is a different subject from a worker that dies mid-iteration.
    """
    healthy = rollout_config(
        workers=2,
        games_per_worker=shards,
        shards_per_worker=shards,
        max_steps=MAX_STEPS,
        stagger_first_reset=False,
        **overrides,
    )
    report = preflight(healthy)
    config = rollout_config(
        workers=2,
        games_per_worker=shards,
        shards_per_worker=shards,
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


def _parent_binary(source: Any) -> str:
    """What the parent's own environment states about its engine binary, for this config."""
    from royalelearn.rollout.envspec import engine_binary

    env = source.config.env.factory(tuple(source.config.extra_component_modules))()
    try:
        return engine_binary(env.config())
    finally:
        env.close()


def _step(source: Any, timeout: float = 30.0) -> Any:
    """One round, answered with no-ops."""
    round_ = source.next_round(timeout)
    _step_answer(source, round_)
    return round_


def _step_answer(source: Any, round_: Any) -> None:
    """Answer a round already taken with no-ops."""
    source.submit(
        Step(
            actions=np.zeros(round_.slots.size, dtype=np.int16),
            gamma=0.99,
            group=np.zeros(0, dtype=np.int8),
            opponent_ix=np.zeros(0, dtype=np.int8),
            learner_seat=np.zeros(0, dtype=np.int8),
        )
    )


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


def test_a_worker_with_nothing_to_do_sleeps() -> None:
    """A worker waiting for a command that is not coming yet costs next to no CPU.

    That wait is where every worker sits through the update, which is most of an iteration,
    and a worker that spins through it takes a core the update's own threads would have had.
    Two shards, because that is the shape a run uses and the one whose wait moves from one
    shard's semaphore to the other's. The farm is then asked for another iteration, because a
    wait that sleeps well and misses the command that ends it is worse than one that spins.
    """
    source, buffer, report, planner = _sources(shards=2)
    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        for _ in range(report.geometry.shards_per_worker):
            _step(source)
        # Every shard has published its trailing round, and every worker is now waiting for a
        # plan: exactly where a run leaves them while it updates the policy.
        source.finish_iteration()
        pids = [child.process.pid for child in source.workers]
        before = [_cpu_seconds(pid) for pid in pids]
        started = time.perf_counter()
        time.sleep(IDLE_S)
        wall = time.perf_counter() - started
        after = [_cpu_seconds(pid) for pid in pids]
        shares = [(end - start) / wall for start, end in zip(before, after, strict=True)]
        assert max(shares) < IDLE_CPU_SHARE, (
            "workers with nothing to do used "
            + ", ".join(f"{share:.0%}" for share in shares)
            + f" of a core each over {wall:.1f}s"
        )

        source.begin_iteration(planner.mirror_plan(1), buffer, 1)
        answered = _step(source, timeout=5.0)
        assert answered.cycle == 0
        assert answered.valid.all(), "a worker that had been idle did not answer its plan"
    finally:
        source.close()
        buffer.close()


def test_a_command_out_of_turn_is_still_taken() -> None:
    """A worker sleeps on the shard whose command is due, and still takes one sent to another.

    A run sends its commands in turn, shard after shard, so that is the semaphore a worker
    sleeps on. A deferral breaks the turn: it hands the same shard back. A second one goes to
    that shard while the worker is asleep on the other, and the parent then waits on that shard
    and on nothing else, so nothing it sends will wake the worker. A worker that only ever
    watched the shard whose turn it was would never take that deferral, and the round would end
    at its deadline instead of one sleep later.
    """
    source, buffer, _report, planner = _sources(shards=2, round_timeout_s=5.0)
    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        # A shard's rounds are refilled in place, so what is compared is read off as it arrives.
        first = source.next_round(5.0)
        shard, cycle = first.shard, first.cycle
        source.submit(Defer())
        source.next_round(5.0)
        source.submit(Defer())
        again = source.next_round(5.0)
        assert (again.shard, again.cycle) == (shard, cycle)
        assert again.valid.all()

        # And the turn comes back: the step goes to the same shard, the next to the other.
        _step_answer(source, again)
        other = _step(source, timeout=5.0)
        assert other.shard != shard
        stepped = source.next_round(5.0)
        assert (stepped.shard, stepped.cycle) == (shard, cycle + 1)
        assert stepped.valid.all()
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


def test_an_idle_control_word_is_not_read_as_a_command() -> None:
    """``STATE_IDLE`` is what the cell holds before either side writes it for this parity.

    It differs from the state the shard published, so a wait that takes "different" for
    "answered" returns on it and hands the dispatcher a command word of zero -- which is not a
    command it knows, and is refused as a protocol violation. It is not an unknown command; it
    is no command yet, and those want opposite handling.
    """
    from royalelearn.rollout.layout import STATE_ACTIONS_READY, STATE_IDLE, STATE_OBS_READY
    from royalelearn.rollout.worker import _wait_command

    class Word:
        def __init__(self, value: int) -> None:
            self.value = value

        def read_command(self, _parity: int) -> tuple[int, float]:
            return self.value, 0.0

    class NoWait:
        @staticmethod
        def acquire(timeout: float = 0.0) -> bool:
            return False

    idle = _wait_command(Word(STATE_IDLE), 0, NoWait(), 0.0, STATE_OBS_READY, STATE_IDLE)
    assert idle is False, "an unwritten control cell is not the parent answering"

    mine = _wait_command(Word(STATE_OBS_READY), 0, NoWait(), 0.0, STATE_OBS_READY, STATE_IDLE)
    assert mine is False, "the shard's own published state is not a command either"

    real = _wait_command(
        Word(STATE_ACTIONS_READY), 0, NoWait(), 0.0, STATE_OBS_READY, STATE_IDLE
    )
    assert real is True, "a command the parent really wrote must still be taken"


class _Word:
    """A shard whose control word holds whatever the test puts in it."""

    def __init__(self, value: int) -> None:
        self.value = value

    def read_command(self, _parity: int) -> tuple[int, float]:
        return self.value, 0.0


class _Tokens:
    """A semaphore's count, and what the parent does while the worker is asleep on it."""

    def __init__(self, count: int, while_asleep: Any = None) -> None:
        self.count = count
        self.sleeps = 0
        self.while_asleep = while_asleep

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        if block:
            self.sleeps += 1
            if self.while_asleep is not None:
                self.while_asleep(self)
        if self.count:
            self.count -= 1
            return True
        return False


def test_a_command_takes_the_token_that_announced_it() -> None:
    """The parent releases one token per command; a wait that sees the command takes it.

    Left behind, the token wakes the next wait on the shard with nothing in the word. That wake
    is survivable -- the word is re-read -- but a semaphore that gains a token each round a spin
    answers is a sleep that has stopped sleeping.
    """
    from royalelearn.rollout.layout import STATE_ACTIONS_READY, STATE_IDLE, STATE_OBS_READY
    from royalelearn.rollout.worker import _wait_command

    tokens = _Tokens(1)
    seen = _wait_command(
        _Word(STATE_ACTIONS_READY), 0, tokens, 0.0, STATE_OBS_READY, STATE_IDLE
    )
    assert seen is True
    assert tokens.count == 0, "the token that announced a command was left for a later wait"
    assert tokens.sleeps == 0, "a command already in the word was slept on"


def test_a_token_left_by_an_earlier_command_is_slept_through() -> None:
    """A wake with nothing in the word is not an answer, and the wait goes back to sleep.

    A spin can see a command between the parent's write and its release, so a token can
    arrive after the wait it belonged to has returned, and wake the next one early. That wait
    sleeps again on the same shard until its own command or the end of ``SLEEP_S``. Returning
    at the wake would report that no command came in a sleep that was cut short, and send the
    worker round its other shards and through another spin for nothing.
    """
    from royalelearn.rollout.layout import STATE_ACTIONS_READY, STATE_IDLE, STATE_OBS_READY
    from royalelearn.rollout.worker import _wait_command

    word = _Word(STATE_OBS_READY)

    def parent_answers_during_the_second_sleep(tokens: _Tokens) -> None:
        if tokens.sleeps == 2:
            word.value = STATE_ACTIONS_READY
            tokens.count += 1

    tokens = _Tokens(1, parent_answers_during_the_second_sleep)
    seen = _wait_command(word, 0, tokens, 0.0, STATE_OBS_READY, STATE_IDLE)
    assert seen is True, "a stale token's wake was taken as the answer"
    assert tokens.sleeps == 2
    assert tokens.count == 0


def test_each_worker_states_the_engine_binary_it_loaded() -> None:
    """Computed in the child, from its own environment, and not a default that happens to agree.

    On MockEngine both sides say "not stated", so a worker that reported nothing would still
    match. This engine states a binary, so the report has to carry it to pass.
    """
    from rollout_support import STAMPED_BINARY

    source, buffer, _report, planner = _sources(engine="rollout_support.StampedMockEngine")
    try:
        assert _parent_binary(source) == STAMPED_BINARY
        source.expected_engine_binary = STAMPED_BINARY
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        for said in source.startup_reports():
            assert said is not None
            assert said.engine_binary_sha256 == STAMPED_BINARY
    finally:
        source.close()
        buffer.close()


def test_workers_on_another_binary_than_the_identity_are_refused_at_start() -> None:
    """The window between the parent measuring the engine and the workers loading it."""
    from royalelearn.errors import PreflightError

    source, buffer, _report, planner = _sources(engine="rollout_support.StampedMockEngine")
    try:
        source.expected_engine_binary = "0123456789abcdef"
        with pytest.raises(PreflightError, match="0123456789abcdef"):
            source.begin_iteration(planner.mirror_plan(0), buffer, 0)
    finally:
        source.close()
        buffer.close()


def test_a_replacement_on_another_engine_binary_is_refused_and_stopped() -> None:
    """The case a rebuild during a run produces, through the real restart path.

    The parent measured the engine at start; a worker restarted later loads whatever file is on
    disk by then. Here the run's identity names a binary no worker has, so the replacement comes
    up healthy and on the wrong engine -- which must be refused in those words, not reported as
    a replacement that failed to start, and must not be left running.
    """
    from royalelearn.errors import PreflightError

    source, buffer, report, planner = _sources(round_timeout_s=3.0)
    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        _step(source, timeout=10.0)
        victim = report.geometry.workers - 1
        source.expected_engine_binary = "0123456789abcdef"
        source.workers[victim].process.terminate()
        with pytest.raises(WorkerTimeout):
            for _ in range(CYCLES):
                _step(source, timeout=3.0)
        source.drain_failures()

        with pytest.raises(PreflightError) as refused:
            source.restart(victim)
        message = str(refused.value)
        assert "restarted rollout worker loaded a different engine binary" in message
        assert "0123456789abcdef" in message
        assert "its replacement did not" not in message, "a healthy replacement was called a crash"
        replacement = source.workers[victim].process
        replacement.join(timeout=10.0)
        assert not replacement.is_alive(), "the refused replacement was left running"
    finally:
        source.close()
        buffer.close()


def test_a_replacement_that_cannot_start_says_the_worker_had_already_started() -> None:
    """"Failed to start" and "stopped being able to start" have opposite causes.

    They have the same symptom -- no startup report -- and the reader's next action differs by
    everything: debug the configuration, or wait for whatever changed to change back. A worker
    that came up once in THIS process proves the configuration was sound, so a replacement that
    cannot come up means something moved underneath the run.

    It is not hypothetical. On 2026-09-22 a sibling repository rebuilt its engine against edited
    calibration data and ``RustEngine()`` stopped constructing across the whole workspace; four
    sessions each had to be told that the failure was not their own code. The engine is not
    reachable from this test, so the environment is broken the same way a child sees it: its
    reward component names a module that is not there, which is a construction failure in the
    worker, before the first round, exactly like an engine that will not build.
    """
    import msgspec

    from royalelearn.errors import PreflightError

    source, buffer, report, planner = _sources(round_timeout_s=3.0)
    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        _step(source, timeout=10.0)
        victim = report.geometry.workers - 1
        # It came up once. That is the whole premise of the message under test, so it is
        # asserted rather than assumed.
        assert source.startup_reports()[victim] is not None
        source.workers[victim].process.terminate()
        with pytest.raises(WorkerTimeout):
            for _ in range(CYCLES):
                _step(source, timeout=3.0)
        source.drain_failures()

        # What the child is given changes under the run, which is what a rebuilt engine or a
        # moved file does. The parent is untouched.
        source.config = msgspec.structs.replace(
            source.config,
            env=msgspec.structs.replace(
                source.config.env,
                reward_fn=ComponentSpec("royalelearn.no_such_module.NoSuchReward", {}),
            ),
        )

        with pytest.raises(PreflightError) as raised:
            source.restart(victim)
        message = str(raised.value)
        assert "came up once in this process" in message
        assert "CHANGED under the run" in message
        assert f"worker {victim}" in message
        # And the child's own text is still carried: the discrimination is added to the cause,
        # not substituted for it.
        assert "no_such_module" in message or "NoSuchReward" in message
    finally:
        source.close()
        buffer.close()
