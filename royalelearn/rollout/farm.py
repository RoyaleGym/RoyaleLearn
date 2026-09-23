"""The worker farm: K processes, the rounds they publish, and what happens when one dies.

The farm is a driver, not a second implementation. Every byte a worker writes is written by
the shard runner in ``rollout.inline``; this module spawns the processes that hold them, waits
on the control words they publish, writes back the actions and the assignment table, and turns
a worker that has stopped answering into a typed failure the coordinator can act on.

Four properties neither reference learner has, and all four are here rather than in a comment:

* every wait has a deadline, and a deadline that passes is a ``WorkerTimeout`` naming the
  worker, the shard and the cycle;
* ``is_alive()`` is checked when a deadline passes, and a dead child's exit code is recorded;
* a restart is deterministic -- the new worker draws from
  ``env/worker/{w}/shard/{s}/gen/{g+1}``, so the run is still a function of the master seed and
  a restart count that is checkpointed;
* a dead worker's slots keep arriving, marked invalid, so the rectangle needs no special case
  for the cycles it missed.

Start method is ``spawn`` everywhere. Windows has no ``fork``, the parent has CUDA
initialised so ``fork`` would be wrong in any case, and forcing one method means the worker's
import-time environment is the same on both platforms.
"""

from __future__ import annotations

import contextlib
import queue
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np

from ..api.rollout import EpisodeRecord, WorkerFailure
from ..errors import PreflightError, WorkerTimeout
from .inline import (
    COMMAND_CLOSE,
    PlanMessage,
    RolloutSourceBase,
    WorkerConfig,
    _cycle_of,
    _gamma_to_word,
)
from .layout import (
    CONTROL,
    ERR_NONE,
    STATE_OBS_READY,
    BufferHandle,
    ControlLayout,
    control_segment_name,
    read_error,
)
from .worker import StartupReport, worker_main

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.buffer import CodecTable
    from ..api.rollout import EnvSpec
    from ..config import Geometry

__all__ = ["ProcessRolloutSource"]

#: How long a worker is given to build its environments and say so. Not the round timeout:
#: constructing an engine decodes the calibration data, and K of them at once decode it K
#: times, which is what ``rollout.launch_delay_s`` spreads out rather than removes.
STARTUP_TIMEOUT_S = 120.0

#: How long the parent sleeps on a semaphore before re-reading the control word it is really
#: waiting on. The word is the truth; the semaphore only keeps the wait from being a spin.
POLL_S = 0.002


class _Worker:
    """One child process and the handles the parent talks to it through."""

    def __init__(self, index: int, layout: ControlLayout, segment: Any) -> None:
        self.index = index
        self.layout = layout
        self.segment = segment
        self.process: Any = None
        self.obs_ready: list[Any] = []
        self.actions_ready: list[Any] = []
        self.outbox: Any = None
        self.inbox: Any = None
        #: Publications the parent has already answered, per shard. The child's parity is a
        #: function of this and nothing else, so neither side counts rounds the other cannot
        #: see.
        self.sent: list[int] = []
        self.pending: dict[int, list[bytes]] = {}
        self.report: StartupReport | None = None


class ProcessRolloutSource(RolloutSourceBase):
    """``K`` worker processes, each holding ``shards_per_worker`` vec envs."""

    def __init__(
        self,
        config: Any,
        spec: EnvSpec,
        codec_table: CodecTable,
        *,
        run_id: str,
        codec: str = "royalelearn.rollout.codec.SpatialObsCodec",
        geometry: Geometry | None = None,
        viser: bool = False,
    ) -> None:
        super().__init__(
            config, spec, codec_table, run_id=run_id, codec=codec, geometry=geometry
        )
        import atexit
        import multiprocessing

        self.viser = viser
        self.context = multiprocessing.get_context("spawn")
        self.workers: list[_Worker] = []
        self._started = False
        self._handle: BufferHandle | None = None
        atexit.register(self.close)

    # -- start-up -----------------------------------------------------------

    def _start(self, handle: BufferHandle) -> None:
        if self._started:
            return
        self._handle = handle
        for index in range(self.geometry.workers):
            self.workers.append(self._build_worker(index))
        for worker in self.workers:
            self._spawn(worker, handle)
            if self.config.rollout.launch_delay_s > 0 and worker.index + 1 < len(self.workers):
                time.sleep(self.config.rollout.launch_delay_s)
        for worker in self.workers:
            worker.report = self._await_report(worker)
        self._started = True

    def _build_worker(self, index: int) -> _Worker:
        from multiprocessing import shared_memory

        layout = ControlLayout(
            run_id=self.run_id,
            worker=index,
            shards=self.geometry.shards_per_worker,
            slots_per_shard=self.geometry.slots_per_shard,
            row_bytes=self.row_bytes,
        )
        name = control_segment_name(self.run_id, index)
        attempt = 0
        while True:
            try:
                segment = shared_memory.SharedMemory(
                    create=True, size=layout.total_bytes, name=name
                )
                break
            except FileExistsError:
                # A segment of this name outlived the process that made it. Take the next
                # name rather than the existing block: the worker is told the name, so it
                # costs nothing, and attaching to somebody else's memory would cost a great
                # deal.
                attempt += 1
                name = f"{control_segment_name(self.run_id, index)}-{attempt}"
                if attempt > 16:  # pragma: no cover - sixteen stale segments is a symptom
                    raise
        return _Worker(index, layout, segment)

    def _spawn(self, worker: _Worker, handle: BufferHandle) -> None:
        shards = self.geometry.shards_per_worker
        worker.obs_ready = [self.context.Semaphore(0) for _ in range(shards)]
        worker.actions_ready = [self.context.Semaphore(0) for _ in range(shards)]
        worker.outbox = self.context.Queue()
        worker.inbox = self.context.Queue()
        worker.sent = [0] * shards
        worker.pending = {shard: [] for shard in range(shards)}
        payload = msgspec.msgpack.encode(self._worker_config(worker, handle))
        worker.process = self.context.Process(
            target=worker_main,
            args=(payload, worker.obs_ready, worker.actions_ready, worker.outbox, worker.inbox),
            name=f"royalelearn-worker-{worker.index}",
            daemon=True,
        )
        worker.process.start()
        self.live[worker.index] = True
        self.rejoining[worker.index] = False

    def _worker_config(self, worker: _Worker, handle: BufferHandle) -> WorkerConfig:
        rollout = self.config.rollout
        return WorkerConfig(
            worker=worker.index,
            generation=self.generation[worker.index],
            run_id=self.run_id,
            master_seed=self.config.master_seed,
            geometry=self.geometry,
            env=self.config.env,
            spec=self._spec,
            codec=self.codec_path,
            codec_table=self.codec_table,
            buffer=handle,
            control=worker.layout.handle(worker.segment.name),
            extra_component_modules=self.extra_modules,
            spin_us=rollout.spin_us,
            stagger_first_reset=rollout.stagger_first_reset,
            viser=self.viser and worker.index == 0,
            ordinals=self.ordinals,
        )

    def _await_report(self, worker: _Worker) -> StartupReport:
        kind, _, payload = self._take(worker, STARTUP_TIMEOUT_S)
        if kind == "start":
            return msgspec.msgpack.decode(payload, type=StartupReport)
        raise PreflightError(
            f"rollout worker {worker.index} did not come up:\n{payload}"
        )

    def _take(self, worker: _Worker, timeout_s: float) -> tuple[str, int, Any]:
        """The next thing a worker says, or an error naming what became of it.

        The wait is in slices so that a child which has already died is noticed when it dies
        rather than when the deadline passes. A process that exits immediately after putting a
        message can be gone before the message arrives, so a dead child is given one more wait
        before it is declared silent.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                return worker.outbox.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                pass
            if not worker.process.is_alive():
                with contextlib.suppress(queue.Empty):
                    return worker.outbox.get(timeout=1.0)
                raise PreflightError(
                    f"rollout worker {worker.index} exited before it said anything; its exit "
                    f"code is {worker.process.exitcode!r}"
                )
            if time.monotonic() >= deadline:
                raise PreflightError(
                    f"rollout worker {worker.index} said nothing for {timeout_s:.0f}s and is "
                    "still running"
                )

    # -- the driver ---------------------------------------------------------

    def _parity(self, worker: int, shard: int) -> int:
        return self.workers[worker].sent[shard] % 2

    def _control_layout(self, worker: int) -> ControlLayout:
        return self.workers[worker].layout

    def _control_memory(self, worker: int) -> Any:
        return self.workers[worker].segment.buf

    def _control_word(self, worker: _Worker, shard: int) -> dict[str, Any]:
        view = worker.layout.view(
            worker.segment.buf, shard, self._parity(worker.index, shard), "control"
        )
        return CONTROL.unpack_from(view, 0)

    def _await_publication(self, shard: int, timeout_s: float) -> None:
        """Wait until every live worker has published this shard, or turn the wait into a failure.

        Every clock here is ``perf_counter``. ``monotonic`` moves in steps of 15.6 ms on Windows
        before Python 3.13, which makes a spin of a hundred microseconds last until the next
        step. A worker releases one token per publication, after writing the word; a spin that
        sees the word takes its token too, so the next wait on this shard sleeps rather than
        waking on it. One the spin could not take yet is drained by a later wait, which wakes,
        reads a word that is not ready, and sleeps again.
        """
        started_ns = time.perf_counter_ns()
        deadline = time.perf_counter() + timeout_s
        spin_s = max(0.0, self.config.rollout.spin_us / 1e6)
        for worker in self.workers:
            if not self._alive(worker.index):
                continue
            semaphore = worker.obs_ready[shard]
            spin_until = time.perf_counter() + spin_s
            woke = False
            while True:
                if int(self._control_word(worker, shard)["state"]) == STATE_OBS_READY:
                    if not woke:
                        semaphore.acquire(False)
                    break
                now = time.perf_counter()
                if now < spin_until:
                    continue
                if now >= deadline:
                    self._timeout(worker, shard, timeout_s)
                woke = semaphore.acquire(timeout=POLL_S)
                if not worker.process.is_alive():
                    # A child that has gone is not going to publish, so the round does not
                    # wait out the deadline for it. The state word is read once more first: a
                    # worker can publish and exit inside one wake, and that publication is as
                    # good as any other.
                    state = int(self._control_word(worker, shard)["state"])
                    if state == STATE_OBS_READY:
                        break
                    self._timeout(worker, shard, timeout_s)
        self._wait_ns += time.perf_counter_ns() - started_ns

    def _timeout(self, worker: _Worker, shard: int, timeout_s: float) -> None:
        """Turn a deadline into a typed failure, and say what the child was doing."""
        alive = worker.process.is_alive()
        exitcode = worker.process.exitcode
        self.live[worker.index] = False
        cycle = _cycle_of(self._control_word(worker, shard))
        self.failures.append(
            WorkerFailure(
                worker=worker.index,
                shard=shard,
                cycle=cycle,
                kind="timeout" if alive else "crash",
                message=(
                    f"no publication for {timeout_s:.1f}s; the process is "
                    f"{'alive' if alive else f'gone with exit code {exitcode!r}'}"
                ),
            )
        )
        raise WorkerTimeout(worker.index, shard, cycle, timeout_s)

    def _read_publication(self, worker: int, shard: int) -> tuple[dict[str, Any], np.ndarray]:
        child = self.workers[worker]
        parity = self._parity(worker, shard)
        word = self._control_word(child, shard)
        return word, child.layout.scalars(child.segment.buf, shard, parity)

    def _read_episodes(self, worker: int, shard: int, count: int) -> list[EpisodeRecord]:
        child = self.workers[worker]
        pending = child.pending[shard]
        deadline = time.monotonic() + self.config.rollout.round_timeout_s
        while len(pending) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout(
                    worker, shard, self._cycle, self.config.rollout.round_timeout_s
                )
            try:
                kind, where, payload = child.outbox.get(timeout=min(remaining, POLL_S * 10))
            except queue.Empty:
                continue
            if kind != "episode":  # pragma: no cover - the child sends nothing else here
                raise ValueError(f"worker {worker} sent {kind!r} in the middle of a round")
            child.pending[int(where)].append(payload)
        records = [
            msgspec.msgpack.decode(blob, type=EpisodeRecord) for blob in pending[:count]
        ]
        del pending[:count]
        return records

    def _read_finals(self, worker: int, shard: int, count: int) -> np.ndarray:
        child = self.workers[worker]
        view = child.layout.view(
            child.segment.buf, shard, self._parity(worker, shard), "finals"
        )
        return np.frombuffer(view, dtype=np.uint8, count=count * self.row_bytes).reshape(
            count, self.row_bytes
        )

    def _send(
        self, shard: int, command: int, gamma: float, message: PlanMessage | None
    ) -> None:
        for child in self.workers:
            if not self._alive(child.index):
                continue
            parity = self._parity(child.index, shard)
            if message is not None:
                child.inbox.put((shard, msgspec.msgpack.encode(message)))
            view = child.layout.view(child.segment.buf, shard, parity, "control")
            CONTROL.pack_into(
                view,
                0,
                {
                    "cycle": (1 << 64) - 1,
                    "t_env_ns": _gamma_to_word(gamma),
                    "state": command,
                    "n_slots": self.geometry.slots_per_shard,
                    "err_code": ERR_NONE,
                    "err_len": 0,
                },
            )
            child.actions_ready[shard].release()
            child.sent[shard] += 1

    def _write_snapshots(self, shard: int, snapshots: Sequence[bytes | None]) -> None:
        for child in self.workers:
            if not self._alive(child.index):
                continue
            battles = self.planner.shard_battles(child.index, shard)
            child.inbox.put(
                (
                    shard,
                    msgspec.msgpack.encode(
                        tuple(
                            snapshots[int(b)] if int(b) < len(snapshots) else None
                            for b in battles
                        )
                    ),
                )
            )

    def _fail_worker(self, worker: int, shard: int, word: dict[str, Any]) -> None:
        child = self.workers[worker]
        self.live[worker] = False
        view = child.layout.view(
            child.segment.buf, shard, self._parity(worker, shard), "error"
        )
        self.failures.append(
            WorkerFailure(
                worker=worker,
                shard=shard,
                cycle=_cycle_of(word),
                kind="exception",
                message=read_error(view, int(word["err_len"])),
            )
        )

    # -- failure ------------------------------------------------------------

    def restart(self, worker: int) -> None:
        """Replace one worker with a fresh process at the next generation.

        The restarted worker rejoins at the next iteration boundary: it has reset its battles
        from a new seed, so its slots hold a different match from the one the rectangle's
        earlier cycles were collecting, and the cells in between are the ones
        ``health/rows_dropped_dead_worker`` counts.
        """
        if self._handle is None:  # pragma: no cover - a source starts before it can fail
            raise RuntimeError("restart() before the first iteration")
        child = self.workers[worker]
        self._terminate(child)
        self.generation[worker] += 1
        self.restarts[worker] += 1
        if self.restarts[worker] > self.config.rollout.max_restarts_per_worker:
            raise RuntimeError(
                f"rollout worker {worker} has been restarted {self.restarts[worker]} times, "
                f"and rollout.max_restarts_per_worker is "
                f"{self.config.rollout.max_restarts_per_worker}: the failure is not transient"
            )
        self._spawn(child, self._handle)
        try:
            child.report = self._await_report(child)
        except PreflightError as exc:
            # "Failed to start" and "stopped being able to start" have the same symptom and
            # opposite causes, and the reader's next action is opposite too: debug the config,
            # or wait for whatever changed to change back. This worker came up once in this
            # process, so the configuration was sound and something moved under the run. On
            # 2026-09-22 four sessions each had to be TOLD that a RustEngine which would not
            # construct was a sibling rebuilding rather than their own code.
            raise PreflightError(
                f"rollout worker {worker} came up once in this process (generation "
                f"{self.generation[worker] - 1}) and its replacement did not. The configuration "
                "started successfully at least once, so this is something that CHANGED under "
                "the run -- an engine rebuilt against different data, a file moved, a device "
                f"taken -- rather than a configuration that was never going to work.\n{exc}"
            ) from exc
        # Up, and out of the iteration: its battles started from a new seed, so the matches its
        # slots were collecting are gone and its cells arrive invalid until the next plan.
        self.live[worker] = False
        self.rejoining[worker] = True
        if self.plan is not None:
            self._write_plan(worker, self.plan)

    def _terminate(self, child: _Worker) -> None:
        process = child.process
        self.live[child.index] = False
        if process is None:
            return
        with contextlib.suppress(OSError, ValueError):  # a process already reaped
            if process.is_alive():
                process.terminate()
            process.join(timeout=self.config.rollout.round_timeout_s)
            if process.is_alive():  # pragma: no cover - a child that ignores terminate
                process.kill()
                process.join(timeout=1.0)
        for channel in (child.outbox, child.inbox):
            if channel is not None:
                channel.close()
                channel.cancel_join_thread()
        child.outbox = None
        child.inbox = None

    def close(self) -> None:
        """Close every worker, in order, and give up on the ones that will not go.

        Idempotent, and registered with ``atexit`` as well as called from the coordinator's
        ``finally``: a run that dies holding 3 GB of vec envs should not need the operator to
        find them.
        """
        if self.closed:
            return
        self.closed = True
        for shard in range(self.geometry.shards_per_worker):
            for child in self.workers:
                if child.process is None or not child.process.is_alive():
                    continue
                # A segment already gone is a worker already gone, which is what this asks for.
                with contextlib.suppress(ValueError, OSError):
                    self._send_one(child, shard, COMMAND_CLOSE)
        for child in self.workers:
            if child.process is not None and child.process.is_alive():
                child.process.join(timeout=10.0)
            self._terminate(child)
            if child.segment is not None:
                child.segment.close()
                # unlink is a no-op on Windows, where a segment lives as long as a handle to
                # it does, and is what actually frees it everywhere else.
                with contextlib.suppress(FileNotFoundError, OSError):
                    child.segment.unlink()
                child.segment = None

    def _send_one(self, child: _Worker, shard: int, command: int) -> None:
        parity = self._parity(child.index, shard)
        view = child.layout.view(child.segment.buf, shard, parity, "control")
        CONTROL.pack_into(
            view,
            0,
            {
                "cycle": (1 << 64) - 1,
                "t_env_ns": 0,
                "state": command,
                "n_slots": self.geometry.slots_per_shard,
                "err_code": ERR_NONE,
                "err_len": 0,
            },
        )
        child.actions_ready[shard].release()
        child.sent[shard] += 1

    def startup_reports(self) -> list[StartupReport | None]:
        """What each worker said about its interpreter once its environments were up."""
        return [child.report for child in self.workers]
