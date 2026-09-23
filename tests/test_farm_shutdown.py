"""Does ``EvalFarm.close`` know whether it actually stopped its workers?

The shutdown path used to be ``join(10); if is_alive(): terminate()`` followed by
``self._procs = []``. Every branch of that is silent. A worker blocked inside a native engine call
ignores both its stop word and SIGTERM, and dropping the handles on the next line removed the only
object that could have been asked afterwards. "Stopped four workers" and "leaked four workers"
produced identical output: nothing.

The workers are ``daemon=True``, so a leak ends when the run does. That is the difference between
a leak that costs the box hours and one that costs it forever -- not the difference between a
defect and none. A gate runs every few iterations on a machine whose training run troughs at
337 MB free, and eval workers that never came back are the cheapest available explanation for an
OOM nobody can place.

These tests do not spawn anything. They hand the farm processes that behave badly on purpose.
"""

from __future__ import annotations

from typing import Any

from royalelearn.ladder.farm import EvalFarm


class _Process:
    """A worker that dies when told -- at whichever stage it is set to obey."""

    def __init__(self, name: str, *, obeys: str) -> None:
        self.name = name
        self.obeys = obeys  # "stop_word", "terminate", "kill", or "nothing"
        self.alive = True
        self.terminated = 0
        self.killed = 0
        self.joins = 0

    def join(self, timeout: float | None = None) -> None:
        self.joins += 1
        if self.obeys == "stop_word":
            self.alive = False

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated += 1
        if self.obeys == "terminate":
            self.alive = False

    def kill(self) -> None:
        self.killed += 1
        if self.obeys == "kill":
            self.alive = False


class _Inbox:
    """A queue that takes its stop words, like a started farm's."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    def put(self, item: Any) -> None:
        self.sent.append(item)


def _farm(*procs: _Process) -> tuple[EvalFarm, list[str]]:
    lines: list[str] = []
    farm = EvalFarm.__new__(EvalFarm)
    farm.printer = lines.append
    farm.terminate_failures = 0
    farm.unkilled = []
    farm._procs = list(procs)
    farm._inbox = _Inbox()
    farm._outbox = None
    return farm, lines


def test_a_worker_that_ignores_everything_is_counted_and_named() -> None:
    """The failure the old code could not express."""
    stuck = _Process("royalelearn-eval-0", obeys="nothing")
    farm, lines = _farm(stuck)

    farm.close()

    assert farm.terminate_failures == 1
    assert farm.unkilled == [stuck], "the handle was dropped, so nobody can ask it anything"
    assert stuck.terminated == 1, "SIGTERM was never tried"
    assert stuck.killed == 1, "SIGKILL was never tried; terminate alone is what failed before"
    assert any("royalelearn-eval-0" in line for line in lines), (
        "close() said nothing, which is exactly how the old version failed"
    )


def test_a_worker_that_stops_is_not_reported() -> None:
    """The control. A counter that always fires measures the code, not the run."""
    good = _Process("royalelearn-eval-0", obeys="stop_word")
    farm, lines = _farm(good)

    farm.close()

    assert farm.terminate_failures == 0
    assert farm.unkilled == []
    assert good.terminated == 0, "a worker that took its stop word was signalled anyway"
    assert good.killed == 0
    assert lines == []


def test_escalation_stops_at_the_stage_that_works() -> None:
    """Each stage is tried only because the one before it did not take."""
    polite = _Process("a", obeys="terminate")
    stubborn = _Process("b", obeys="kill")
    farm, _ = _farm(polite, stubborn)

    farm.close()

    assert (polite.terminated, polite.killed) == (1, 0)
    assert (stubborn.terminated, stubborn.killed) == (1, 1)
    assert farm.terminate_failures == 0, "both processes did stop; neither is a failure"


def test_a_mixed_shutdown_counts_only_the_leak() -> None:
    """Distinct outcomes in one call, so a miscount cannot hide behind a uniform batch."""
    procs = [
        _Process("a", obeys="stop_word"),
        _Process("b", obeys="kill"),
        _Process("c", obeys="nothing"),
        _Process("d", obeys="terminate"),
    ]
    farm, lines = _farm(*procs)

    farm.close()

    assert farm.terminate_failures == 1
    assert [p.name for p in farm.unkilled] == ["c"]
    assert [(p.terminated, p.killed) for p in procs] == [(0, 0), (1, 1), (1, 1), (1, 0)]
    assert len(lines) == 1


def test_close_is_still_idempotent_and_clears_the_queues() -> None:
    """The second call must not re-signal a process it already gave up on."""
    stuck = _Process("royalelearn-eval-0", obeys="nothing")
    farm, lines = _farm(stuck)

    farm.close()
    farm.close()

    assert farm.terminate_failures == 1, "the same leak was counted twice"
    assert (stuck.terminated, stuck.killed) == (1, 1), "the second close signalled it again"
    assert len(lines) == 1
    assert farm._inbox is None and farm._outbox is None


def test_a_leak_does_not_block_the_next_gate() -> None:
    """``_start`` returns early when ``_procs`` is non-empty, so the leak must not live there."""
    stuck = _Process("royalelearn-eval-0", obeys="nothing")
    farm, _ = _farm(stuck)

    farm.close()

    assert farm._procs == [], "a leaked worker left in _procs makes the next gate start none at all"


def test_the_stop_word_goes_out_before_any_signal() -> None:
    """Signalling first would make the polite path untestable and the logs wrong."""
    sent: list[Any] = []
    order: list[str] = []

    class _Watching(_Inbox):
        def put(self, item: Any) -> None:
            sent.append(item)
            order.append("stop_word")

    class _Watched(_Process):
        def terminate(self) -> None:
            order.append("terminate")
            super().terminate()

    proc = _Watched("royalelearn-eval-0", obeys="kill")
    farm, _ = _farm(proc)
    farm._inbox = _Watching()

    farm.close()

    assert sent == [None]
    assert order == ["stop_word", "terminate"]


def test_a_stop_word_that_never_went_out_is_not_blamed_on_the_workers() -> None:
    """The message must not charge a worker with ignoring something it never received.

    A wrong cause is worse than a missing one: it is a complete sentence, it arrives with the
    credibility of a real failure, and it sends the reader to the workers when the queue is what
    broke.
    """

    class _BrokenInbox(_Inbox):
        def put(self, item: Any) -> None:
            raise OSError("the handle is closed")

    stuck = _Process("royalelearn-eval-0", obeys="nothing")
    farm, lines = _farm(stuck)
    farm._inbox = _BrokenInbox()

    farm.close()

    assert farm.terminate_failures == 1
    assert "did not reach 1 of 1" in lines[0], "close() blamed the worker for the queue"
    assert "survived a stop word" not in lines[0]


def test_a_delivered_stop_word_is_not_second_guessed() -> None:
    """The control. A note that always appears names nothing."""
    stuck = _Process("royalelearn-eval-0", obeys="nothing")
    farm, lines = _farm(stuck)

    farm.close()

    assert farm.terminate_failures == 1
    assert "did not reach" not in lines[0]
