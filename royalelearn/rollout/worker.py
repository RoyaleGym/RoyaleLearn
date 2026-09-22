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

Shards alternate strictly, and a shard whose command has not arrived is skipped rather than
waited on, so a worker holding two of them steps one while the parent runs inference on the
other.

The control word is the signal and the semaphores are only a way of sleeping until it changes.
A semaphore may be left with a token when the spin sees the word first, so a wait can return
without a command having arrived; the word is re-read after every wake and the wait is never
taken as the answer.

The child holds no policy and never imports torch. ``tests/test_worker_hygiene.py`` asserts
both from the report this file sends once its environments are up.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any

import msgspec

from ..determinism import BLAS_THREAD_VARS, apply_blas_thread_env

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
    from .layout import STATE_OBS_READY
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
                )
            ),
        )
    )
    for shard, runner in enumerate(shards):
        runner.publish(CYCLE_UNBOUND)
        obs_ready[shard].release()

    spin_s = max(0.0, config.spin_us / 1e6)
    pending: dict[int, list[bytes]] = {shard: [] for shard in range(len(shards))}
    running = True
    while running:
        for shard, runner in enumerate(shards):
            parity = runner.open_parity()
            if not _wait_command(runner, parity, actions_ready[shard], spin_s, STATE_OBS_READY):
                continue
            command, gamma = runner.read_command(parity)
            try:
                message: PlanMessage | None = None
                if command == COMMAND_PLAN:
                    message = msgspec.msgpack.decode(
                        _take(inbox, shard, pending), type=PlanMessage
                    )
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
    runner: Any, parity: int, semaphore: Semaphore, spin_s: float, published_state: int
) -> bool:
    """True once the parent has written a command into this shard's open control word.

    The shard published its own state into that word, so anything else in it is the parent's
    answer. Spin first -- a round trip through the operating system is tens of microseconds and
    a spin is about two -- and then sleep briefly on the semaphore rather than burning a core
    while the parent is running inference.
    """
    deadline = time.monotonic() + spin_s
    while True:
        state, _ = runner.read_command(parity)
        if state != published_state:
            return True
        if time.monotonic() >= deadline:
            break
    semaphore.acquire(timeout=0.001)
    state, _ = runner.read_command(parity)
    return state != published_state


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
