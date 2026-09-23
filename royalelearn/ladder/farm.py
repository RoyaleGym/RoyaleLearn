"""Evaluation battles across processes, because a gate is 2,200 of them one at a time.

MEASURED 2026-09-23, with the real engine: one evaluation battle is 8.02 s network against
network and 0.195 s scripted against scripted. A gate plays the champion comparison, the anchors
and the pool-collapse check -- 2,200 battles at the shipped settings -- in the parent, serially,
between the update and the metric row. That is 4.9 hours, and at the shipped candidate cadence a
500-iteration run fires six of them: 29 hours of evaluation against 4 hours of training.

Nothing about those battles requires the parent. Every one is fully determined by
``(a, b, seed, a_seat, act_path)``: the environment is reset from the seed and the acting uniforms
are derived from the master seed and the stream path, so no battle can see another and the order
they are played in is not part of the measurement. ``EvalRunner`` names the whole list first and
hands it to ``play_all``, and a player that offers ``play_many`` is given all of them at once.

WHAT THIS CANNOT PLAY, and says so rather than substituting. A probe asks for ``learner@{step}``,
which is the LIVE model, and a worker process does not have one -- its weights change every
iteration and shipping them per battle would cost more than the battle. Those requests go to the
fallback player in the parent, which is where they were already. A gate needs none of them: both
sides of every comparison it runs are frozen snapshots or scripted, and both load from disk.

THE CONTRACT IS THE RETURN ORDER, NOT THE PLAY ORDER. Requests come back indexed, so a farm and a
serial player produce the identical ``Comparison`` and the identical rows in the result log. That
is asserted in ``tests/test_eval_dispatch.py`` against a player that deliberately plays its batch
back to front, and it was pinned there before this module existed.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

# These three are imported for real rather than under TYPE_CHECKING: msgspec resolves a
# struct's annotations when it DECODES one, which happens inside a worker where nothing else
# has imported them. None of the three pulls in torch.
from ..api.rollout import EnvSpec
from ..config import NetConfig
from ..rollout.envspec import EnvFactorySpec
from .evaluate import BattleRequest
from .snapshots import SnapshotSpec

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .evaluate import BattlePlayer

__all__ = ["EvalFarm", "EvalWorkerConfig", "eval_worker_main"]

#: How long a worker is given to build its environment and say it is ready. Constructing an engine
#: decodes the calibration data, and K of them at once decode it K times.
STARTUP_TIMEOUT_S = 180.0

#: How long a worker gets to act on its stop word before it is signalled. It is finishing at most
#: one battle, measured at 8.02 s.
STOP_WORD_TIMEOUT_S = 10.0

#: How long a signalled worker gets to actually die before the next signal. Nothing is running at
#: this point; this is the operating system's latency, not the worker's work.
SIGNAL_TIMEOUT_S = 5.0

#: How long one battle may take before the farm calls the worker dead. A full match with no
#: truncation was measured at 8 s with networks on both sides; this is twenty times that, because
#: the number that matters is "clearly hung" rather than "slower than expected".
BATTLE_TIMEOUT_S = 180.0


class EvalWorkerConfig(msgspec.Struct, frozen=True):
    """Everything a worker needs to play a battle, and nothing that changes during a run.

    Every field is msgspec-encodable on purpose: a worker is spawned rather than forked, so this
    crosses as bytes and a closure could not. The live model is absent for the same reason it is
    absent from ``EvalActors`` in a worker -- it changes every iteration and is not a constant.
    """

    spec: EnvSpec
    net: NetConfig
    env: EnvFactorySpec
    extra_modules: tuple[str, ...]
    snapshot_root: str
    template: SnapshotSpec
    master_seed: int
    release_mode: str
    max_decisions: int
    max_resident: int


def _build_player(config: EvalWorkerConfig) -> Any:
    """The worker's own player, built from the same classes the parent uses.

    Imported here rather than at module scope so that importing this module costs nothing in a
    process that only wants to talk to a farm, and so a parent without torch can still build one.
    """
    from ..coordinator import EnvBattlePlayer
    from .actors import EvalActors
    from .snapshots import DiskSnapshotStore

    def build_actor(device: Any) -> Any:
        from ..learn.actor_critic import ClashActor
        from ..learn.nets import ClashTrunk, PointerPolicyHead, resolve_dtype

        actor = ClashActor(
            ClashTrunk(config.spec, config.net),
            PointerPolicyHead(config.spec, config.net),
            resolve_dtype(config.net.autocast_dtype),
        )
        return actor.to(device)

    store = DiskSnapshotStore(
        Path(config.snapshot_root),
        template=config.template,
        build=build_actor,
        max_resident=config.max_resident,
    )
    actors = EvalActors(
        config.spec,
        config.net,
        store,
        device="cpu",
        release_mode=config.release_mode,
        model=None,
    )
    return EnvBattlePlayer(
        config.spec,
        config.env,
        actors,
        master_seed=config.master_seed,
        extra_modules=config.extra_modules,
        device="cpu",
        release_mode=config.release_mode,
        max_decisions=config.max_decisions,
    )


def eval_worker_main(payload: bytes, inbox: Any, outbox: Any) -> None:
    """One worker: build a player, then answer battles until told to stop.

    CPU on purpose. A batch-of-one forward is latency-bound rather than throughput-bound, and K
    workers sharing one device would serialise on it anyway; the point of the farm is the K
    environments, which are the expensive half.
    """
    config = msgspec.msgpack.decode(payload, type=EvalWorkerConfig)
    try:
        player = _build_player(config)
    except BaseException as exc:  # the parent decides what a failed start means
        import traceback

        outbox.put(("failed", -1, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        return
    outbox.put(("ready", -1, ""))
    while True:
        item = inbox.get()
        if item is None:
            break
        index, request = item
        try:
            score = player.play(
                a=request.a,
                b=request.b,
                seed=request.seed,
                a_seat=request.a_seat,
                act_path=request.act_path,
            )
        except BaseException as exc:  # reported, not swallowed
            import traceback

            outbox.put(("failed", index, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
            continue
        outbox.put(("score", index, float(score)))
    player.close()


class EvalFarm:
    """``workers`` processes playing evaluation battles, behind the ``BattlePlayer`` seam.

    Built lazily: a run that never gates never pays for the processes, and a run that gates once
    pays for them once. ``close`` is idempotent and is called from the coordinator's own close.
    """

    def __init__(
        self,
        config: EvalWorkerConfig,
        *,
        workers: int,
        fallback: BattlePlayer,
        printer: Any = None,
    ) -> None:
        self.config = config
        self.workers = max(1, int(workers))
        self.fallback = fallback
        self.printer = printer or (lambda _line: None)
        self._procs: list[Any] = []
        self._inbox: Any = None
        self._outbox: Any = None
        self.battles_played = 0
        #: Workers that survived every signal ``close`` has. Kept, because the defect this
        #: replaced was that they were dropped and could not be asked about afterwards.
        self.unkilled: list[Any] = []
        self.terminate_failures = 0

    # -- the player seam ----------------------------------------------------

    def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
        """One battle, in the parent. The farm is for batches; a single battle is not worth one."""
        return self.fallback.play(a=a, b=b, seed=seed, a_seat=a_seat, act_path=act_path)

    def play_many(self, requests: list[BattleRequest]) -> list[float]:
        """Every request's score, in the order the requests were given."""
        from .pool import is_learner

        if not requests:
            return []
        live = [
            index
            for index, request in enumerate(requests)
            if is_learner(request.a) or is_learner(request.b)
        ]
        if live:
            # A probe, or anything else naming the live weights. The parent has the model and the
            # workers do not, and a snapshot would answer a different question.
            return [
                self.play(
                    a=request.a,
                    b=request.b,
                    seed=request.seed,
                    a_seat=request.a_seat,
                    act_path=request.act_path,
                )
                for request in requests
            ]
        self._start()
        return self._dispatch(requests)

    # -- the processes ------------------------------------------------------

    def _start(self) -> None:
        if self._procs:
            return
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        self._inbox = context.Queue()
        self._outbox = context.Queue()
        payload = msgspec.msgpack.encode(self.config)
        started = time.perf_counter()
        for index in range(self.workers):
            process = context.Process(
                target=eval_worker_main,
                args=(payload, self._inbox, self._outbox),
                name=f"royalelearn-eval-{index}",
                daemon=True,
            )
            process.start()
            self._procs.append(process)
        for _ in range(self.workers):
            kind, _index, detail = self._take(STARTUP_TIMEOUT_S)
            if kind != "ready":
                self.close()
                raise RuntimeError(f"an evaluation worker did not come up:\n{detail}")
        self.printer(
            f"eval farm     {self.workers} workers up in {time.perf_counter() - started:.1f}s"
        )

    def _dispatch(self, requests: list[BattleRequest]) -> list[float]:
        assert self._inbox is not None
        for index, request in enumerate(requests):
            self._inbox.put((index, request))
        scores: list[float | None] = [None] * len(requests)
        for _ in range(len(requests)):
            kind, index, detail = self._take(BATTLE_TIMEOUT_S)
            if kind != "score":
                self.close()
                raise RuntimeError(
                    f"an evaluation worker failed on battle {index} "
                    f"({requests[index].a} vs {requests[index].b}):\n{detail}"
                )
            scores[index] = float(detail)
        self.battles_played += len(requests)
        missing = [index for index, score in enumerate(scores) if score is None]
        if missing:  # pragma: no cover - the loop above counts, so this cannot be reached
            raise RuntimeError(f"battles {missing} were never answered")
        return [float(score) for score in scores]  # type: ignore[arg-type]

    def _take(self, timeout_s: float) -> tuple[str, int, Any]:
        """The next thing a worker says, or a failure naming what became of them.

        The wait is in slices so that a batch whose workers have all died is noticed when they die
        rather than when the deadline passes.
        """
        assert self._outbox is not None
        deadline = time.perf_counter() + timeout_s
        while True:
            try:
                return self._outbox.get(timeout=0.2)
            except queue.Empty:
                pass
            alive = [process for process in self._procs if process.is_alive()]
            if not alive:
                codes = [process.exitcode for process in self._procs]
                raise RuntimeError(
                    f"every evaluation worker exited (codes {codes}) with battles outstanding"
                )
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"no evaluation worker answered within {timeout_s:.0f}s; "
                    f"{len(alive)} of {len(self._procs)} still alive"
                )

    def close(self) -> None:
        """Stop the workers, and say so only if they stopped.

        THE OLD VERSION COULD NOT TELL YOU WHICH HAPPENED. It joined, called ``terminate`` on
        anything still alive, and then cleared ``_procs`` on the next line -- so a worker that
        ignored SIGTERM was both unreported and unreachable, and a clean shutdown and a leaked
        process produced the same output: none. A worker blocked inside a native engine call is
        exactly the case that ignores a stop word AND a terminate, and it is the case this farm
        spends its whole life in.

        So each stage is tried because the one before it did not take, the handles of anything
        that survived all three are kept on ``unkilled`` rather than dropped, and the count is a
        number somebody can read. They are ``daemon=True``, so a leak ends when the run does --
        that bounds the cost at the length of a run on a box that troughs at 337 MB free, which is
        a reason to report it rather than a reason not to.
        """
        if not self._procs:
            return
        delivered = 0
        if self._inbox is not None:
            for _ in self._procs:
                with _suppress():
                    self._inbox.put(None)
                    delivered += 1
        leaked = []
        for process in self._procs:
            if _stop(process):
                continue
            leaked.append(process)
        if leaked:
            self.terminate_failures += len(leaked)
            self.unkilled.extend(leaked)
            # The stop word is put under _suppress, so a broken queue is as silent as a
            # delivered one. Saying "survived a stop word" when none went out is a WRONG cause
            # rather than a missing one: it reads as a complete explanation and sends the reader
            # to the workers when the queue is what broke.
            missed = len(self._procs) - delivered
            note = (
                f" The stop word did not reach {missed} of {len(self._procs)}, so this may be "
                "the queue rather than the workers."
                if missed
                else ""
            )
            self.printer(
                f"{len(leaked)} evaluation worker(s) survived every signal close() has: "
                f"{', '.join(p.name for p in leaked)}. They hold memory until this run "
                f"ends.{note}"
            )
        self._procs = []
        self._inbox = None
        self._outbox = None


def _stop(process: Any) -> bool:
    """Stop word, SIGTERM, SIGKILL, in that order, each because the one before did not take.

    Returns whether the process is gone. A handle that RAISES counts as not gone rather than
    propagating: ``close`` runs in teardown under the coordinator's blanket suppress, so an
    exception here would abort the loop over the remaining workers and then be swallowed, leaving
    more processes alive and nothing said about any of them.
    """
    for stage, timeout in (
        (None, STOP_WORD_TIMEOUT_S),
        ("terminate", SIGNAL_TIMEOUT_S),
        ("kill", SIGNAL_TIMEOUT_S),
    ):
        try:
            if stage is not None:
                getattr(process, stage)()
            process.join(timeout=timeout)
            if not process.is_alive():
                return True
        except Exception:
            return False
    return False


def _suppress() -> Any:
    import contextlib

    return contextlib.suppress(Exception)
