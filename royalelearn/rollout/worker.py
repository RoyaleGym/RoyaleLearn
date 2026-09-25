"""A rollout worker's process: the preamble, the loop, and nothing else.

The preamble is why this module imports almost nothing. A worker pins its BLAS thread counts
and ignores Ctrl-C **before numpy is imported**, and numpy arrives the moment anything that
uses it does, so the module holding the child's entry point has to reach the first line of that
entry point without having pulled numpy in behind it. Everything the worker actually does
therefore lives in ``rollout.inline`` -- one shard runner, driven here by a child process and
in the learner's own process by the inline source -- and is imported below, after the preamble
has run.

The loop is the whole protocol:

    wait for a command on the parity of the round this shard last published
    apply it -- a plan, an assignment table and a step, a state, a deferral, a close
    publish the result

Shards alternate strictly, so a worker holding two of them steps one while the parent runs
inference on the other. The worker sleeps only on the shard whose turn it is -- the one the
parent answers next when it answers them in order, which is every command of a run -- and
glances at the others on its way round, so a command that arrives out of turn is still taken,
at most ``SLEEP_S`` late.

The control word is the signal and the semaphores are only a way of sleeping until it changes.
The parent releases one token per command, after writing the word, and a wait that sees the
word takes that token with it, so a semaphore holds one token per command not yet answered. A
spin can see the word between the parent's write and its release, and then the token arrives
after the wait has returned; it wakes a later wait with no new command in the word. So the word
is re-read after every wake, and a wake is never taken as the answer.

The child holds no policy and never imports torch. ``tests/test_worker_hygiene.py`` asserts
both from the report this file sends once its environments are up.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any

import msgspec

from ..determinism import BLAS_THREAD_VARS, apply_blas_thread_env
from .envspec import NOT_STATED, engine_binary

#: How often a worker asks whether its parent is still there. Often enough that a killed run
#: does not leave a tree behind for long, rarely enough to cost nothing in a round.
PARENT_CHECK_S = 2.0

#: The longest a worker sleeps on one shard's semaphore before it looks at its other shards
#: and at its parent again. The parent's release ends the sleep at once, so this is not the
#: latency of a command on the shard being slept on; it is what an out-of-turn command waits at
#: most, and what an idle worker pays one wake and one spin for. The update leaves every worker
#: idle for most of an iteration, and at fifty milliseconds an idle worker wakes at most twenty
#: times a second.
SLEEP_S = 0.05

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Semaphore

__all__ = ["StartupReport", "preamble", "worker_main"]


class StartupReport(msgspec.Struct, frozen=True):
    """What a worker says about itself once its environments are up.

    Asking the child is the only way to know what its interpreter looks like, and what is
    asked is exactly what a worker promises: the thread variables were set before numpy, and no
    policy came with it.
    """

    worker: int
    generation: int
    pid: int
    numpy_preloaded: bool
    torch_loaded: bool
    thread_env: dict[str, str]
    shards: int
    battles: int
    #: The engine binary THIS process loaded, stated rather than assumed. A worker restarted
    #: after a rebuild loads the new file, and the parent's measurement at start cannot see it.
    engine_binary_sha256: str = NOT_STATED


def preamble() -> None:
    """Pin the thread counts and hand Ctrl-C back to the parent.

    The thread counts are read by the BLAS library when numpy loads it, so this is early or it
    is nothing. SIGINT is ignored because an interrupt at the terminal reaches every process in
    the group, and the parent is the one that decides a run is over: it closes its workers in
    order, writing a checkpoint on the way out.
    """
    apply_blas_thread_env()
    # Only the main thread of a process may install a handler; an inline source is not one.
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def worker_main(
    payload: bytes,
    obs_ready: list[Semaphore],
    actions_ready: list[Semaphore],
    outbox: Queue,
    inbox: Queue,
) -> None:
    """One worker process, from its first import to its last publication."""
    numpy_preloaded = "numpy" in sys.modules
    preamble()

    from multiprocessing import shared_memory

    from .inline import (
        COMMAND_PLAN,
        COMMAND_SET_STATE,
        CYCLE_UNBOUND,
        PlanMessage,
        ShardRunner,
        WorkerConfig,
        build_codec,
    )
    from .layout import STATE_IDLE, STATE_OBS_READY
    from .plan import SlotPlanner

    config = msgspec.msgpack.decode(payload, type=WorkerConfig)
    shards: list[ShardRunner] = []
    buffer = None
    control = None
    try:
        buffer = shared_memory.SharedMemory(name=config.buffer.name)
        control = shared_memory.SharedMemory(name=config.control.name)
        planner = SlotPlanner(config.geometry, config.master_seed)
        codec = build_codec(
            config.codec, config.spec, config.codec_table, config.extra_component_modules
        )
        for shard in range(config.geometry.shards_per_worker):
            runner = ShardRunner(
                config,
                shard,
                planner=planner,
                codec=codec,
                buffer=buffer.buf,
                control=control.buf,
                viser="env" if (config.viser and shard == 0) else None,
                recorder=config.recorder if shard == 0 else None,
            )
            runner.start()
            shards.append(runner)
    except BaseException:
        outbox.put(("crash", 0, traceback.format_exc()))
        _close(shards, buffer, control)
        raise

    outbox.put(
        (
            "start",
            0,
            msgspec.msgpack.encode(
                StartupReport(
                    worker=config.worker,
                    generation=config.generation,
                    pid=os.getpid(),
                    numpy_preloaded=numpy_preloaded,
                    torch_loaded="torch" in sys.modules,
                    thread_env={name: os.environ.get(name, "") for name in BLAS_THREAD_VARS},
                    shards=len(shards),
                    battles=config.geometry.games_per_shard * len(shards),
                    engine_binary_sha256=engine_binary(shards[0].vec.envs[0].config())
                    if shards
                    else NOT_STATED,
                )
            ),
        )
    )
    for shard, runner in enumerate(shards):
        runner.publish(CYCLE_UNBOUND)
        obs_ready[shard].release()

    spin_s = max(0.0, config.spin_us / 1e6)
    pending: dict[int, list[bytes]] = {shard: [] for shard in range(len(shards))}
    # The shard whose command the parent sends next when it answers in order. Every command of
    # a run is in order -- the plans of an iteration go out shard by shard, and so do the steps
    # -- so this is the one worth sleeping on.
    turn = 0
    running = True
    parent = multiprocessing.parent_process()
    next_parent_check = time.monotonic() + PARENT_CHECK_S
    while running:
        # A worker outlives its parent silently otherwise. The parent normally closes the farm
        # on its way out, but it cannot when it is killed rather than asked, and what is left
        # is a tree of processes holding a viewer port and a share of the memory -- which then
        # makes the next run more likely to lose a worker to memory pressure, and harder to
        # attribute when it does. Asking whether the parent is alive is exact, where an idle
        # timeout would be a guess: a gate legitimately leaves these workers untouched for
        # minutes at a time.
        now = time.monotonic()
        if now >= next_parent_check:
            next_parent_check = now + PARENT_CHECK_S
            if parent is not None and not parent.is_alive():
                break
        for shard, runner in enumerate(shards):
            parity = runner.open_parity()
            # Sleeping on a shard whose turn it is not would leave the command that is due on
            # the other one waiting out the sleep; a glance costs one read.
            due = shard == turn
            if not _wait_command(
                runner,
                parity,
                actions_ready[shard],
                spin_s if due else 0.0,
                STATE_OBS_READY,
                STATE_IDLE,
                sleep_s=SLEEP_S if due else 0.0,
            ):
                continue
            turn = (shard + 1) % len(shards)
            command, gamma = runner.read_command(parity)
            try:
                message: PlanMessage | None = None
                if command == COMMAND_PLAN:
                    message = msgspec.msgpack.decode(_take(inbox, shard, pending), type=PlanMessage)
                elif command == COMMAND_SET_STATE:
                    runner.pending_snapshots = msgspec.msgpack.decode(
                        _take(inbox, shard, pending), type=tuple[bytes | None, ...]
                    )
                running = runner.handle(command, gamma, parity, message)
            except BaseException as exc:  # the parent is told, and then the child stops
                runner.publish_error(_failure_text(exc))
                obs_ready[shard].release()
                _close(shards, buffer, control)
                return
            for record in runner.episodes:
                outbox.put(("episode", shard, msgspec.msgpack.encode(record)))
            obs_ready[shard].release()
            if not running:
                break
    _close(shards, buffer, control)


def _failure_text(exc: BaseException) -> str:
    """The error first, then the frames that led to it.

    The error region is half a kilobyte and a traceback is longer than that, so what has to
    survive the truncation is the line saying what happened rather than the first few frames of
    how it was reached.
    """
    return f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def _wait_command(
    runner: Any,
    parity: int,
    semaphore: Semaphore,
    spin_s: float,
    published_state: int,
    idle_state: int,
    sleep_s: float = SLEEP_S,
) -> bool:
    """True once the parent has written a command into this shard's open control word.

    The shard published its own state into that word, so anything else in it is the parent's
    answer -- except the one value that is nobody's answer. ``STATE_IDLE`` is what the cell
    holds before either side has written it for this parity, and it differs from the published
    state without being a command, so taking "different" as "answered" hands the dispatcher a
    word of zero and it refuses it as a command it does not know. It is not an unknown command;
    it is no command yet, and the two want opposite handling: an unknown word is a protocol
    violation and should be loud, an idle cell means keep waiting.

    Spin first -- a round trip through the operating system is tens of microseconds and a spin
    is about two -- and then sleep on the semaphore until the parent's release, for at most
    ``sleep_s``; zero looks once and does not sleep. The spin's deadline is read off
    ``perf_counter``. ``monotonic`` is the tick count on Windows before Python 3.13 and moves
    in steps of 15.6 ms, so a spin of a hundred microseconds timed by it lasts until the next
    step. With a one-millisecond sleep after it, an idle worker spun for about half its time on
    the default system timer and for about 94% of it on a one-millisecond one.

    A command's token is taken along with the command, so it cannot wake a later wait. Taking
    one never steals the next command's: the parent writes that only once it has read the
    publication answering this one. A token can still arrive late -- the spin saw the word
    between the parent's write and its release -- and then it wakes the next wait on this
    shard with no command in the word, which sleeps again rather than giving up its turn.
    """
    deadline = time.perf_counter() + spin_s
    while True:
        if _answered(runner, parity, published_state, idle_state):
            semaphore.acquire(False)
            return True
        if time.perf_counter() >= deadline:
            break
    if sleep_s <= 0.0:
        return False
    while semaphore.acquire(timeout=sleep_s):
        if _answered(runner, parity, published_state, idle_state):
            return True
    # The sleep ran out, and the command may have landed as it did.
    if _answered(runner, parity, published_state, idle_state):
        semaphore.acquire(False)
        return True
    return False


def _answered(runner: Any, parity: int, published_state: int, idle_state: int) -> bool:
    state, _ = runner.read_command(parity)
    return state not in (published_state, idle_state)


def _take(inbox: Queue, shard: int, pending: dict[int, list[bytes]]) -> bytes:
    """The message belonging to the command this shard just read.

    A command that carries one sends it before the control word that announces it, so it is in
    the queue by the time the word is visible -- but both shards share the queue, and the
    other shard's message may be at the front of it. Messages are tagged, and one for the
    shard that is not being served waits in ``pending`` for its own turn.
    """
    queued = pending[shard]
    while not queued:
        where, payload = inbox.get(timeout=30.0)
        pending[int(where)].append(payload)
    return queued.pop(0)


def _close(shards: list[Any], buffer: Any, control: Any) -> None:
    """Give back everything this process holds. Shutdown reports nothing and refuses nothing."""
    for runner in shards:
        with contextlib.suppress(Exception):
            runner.close()
    for segment in (buffer, control):
        if segment is not None:
            with contextlib.suppress(Exception):
                segment.close()
