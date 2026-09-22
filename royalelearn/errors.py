"""The exceptions the harness raises, and what each one means operationally.

Every one of them is a refusal to continue with something that would otherwise produce a
number nobody could trust: a run started against a stale engine, a resume into a different
identity, a checkpoint whose bytes no longer hash to what the manifest says. They are
separate types rather than one because the caller does different things with them --
``PreflightError`` aborts a start-up, ``WorkerTimeout`` restarts one worker, ``AlarmHalt``
writes a bundle and exits non-zero -- and because a bare ``RuntimeError`` in a log six weeks
later says nothing about which of those happened.
"""

from __future__ import annotations

from collections.abc import Sequence


class RoyaleLearnError(Exception):
    """Base of every error this package raises deliberately."""


class PreflightError(RoyaleLearnError):
    """A start-up gate refused the run (section 7.7 of ``docs/harness-spec.md``).

    Raised before the first cycle, never during one: a mask disagreement, a missing vector
    field, an observation space the codec cannot store, a RAM projection over budget. The
    message names the thing that is wrong and, where there is one, the command that fixes it.
    """


class StaleEngineBuild(PreflightError):
    """The compiled engine was built from data that differs from the data on disk.

    Carries the engine's own list of differing keys verbatim: the harness hashes no data file
    of its own, because a second opinion about what the engine is running on is exactly what a
    stale build looks like.
    """

    def __init__(
        self, message: str, differences: Sequence[str] = (), rebuild_command: str = ""
    ) -> None:
        self.differences = tuple(differences)
        self.rebuild_command = rebuild_command
        if rebuild_command:
            message = f"{message}\nrebuild with: {rebuild_command}"
        super().__init__(message)


class IdentityMismatch(RoyaleLearnError):
    """A resume was asked for into a run identity that is not the checkpoint's.

    ``differences`` is ``field -> (checkpoint value, current value)`` for every identity field
    that moved. All of them are reported at once: fixing them one error message at a time is
    several minutes per field on a machine that takes a minute to build an env.
    """

    def __init__(self, differences: dict[str, tuple[object, object]]) -> None:
        self.differences = dict(differences)
        lines = [
            f"  {name}: checkpoint {was!r}, now {now!r}"
            for name, (was, now) in sorted(differences.items())
        ]
        super().__init__(
            "the run identity differs from the checkpoint's in "
            f"{len(differences)} field(s):\n" + "\n".join(lines)
        )


class CheckpointFormatError(RoyaleLearnError):
    """A checkpoint cannot be read as what it claims to be.

    A manifest from a newer format version, a file whose sha256 does not match the manifest, a
    component folder that is not there under ``strict=True``. A truncated write from a crash is
    an error here rather than a silently wrong resume.
    """


class WorkerTimeout(RoyaleLearnError):
    """A rollout worker did not answer within ``rollout.round_timeout_s``.

    Every wait in the farm has a timeout and this is what one becomes, so that a dead child is a
    typed failure the coordinator restarts rather than a block with no end.
    """

    def __init__(self, worker: int, shard: int, cycle: int, timeout_s: float) -> None:
        self.worker = worker
        self.shard = shard
        self.cycle = cycle
        self.timeout_s = timeout_s
        super().__init__(
            f"worker {worker} shard {shard} did not publish cycle {cycle} within {timeout_s:.1f}s"
        )


class AlarmHalt(RoyaleLearnError):
    """A halt-severity alarm fired (section 13.3).

    Raised after the checkpoint and the diagnostic bundle have been written, so the process
    exits non-zero with the evidence already on disk.
    """

    def __init__(self, alarm: str, message: str, iteration: int, bundle: str | None = None) -> None:
        self.alarm = alarm
        self.iteration = iteration
        self.bundle = bundle
        where = f"; bundle at {bundle}" if bundle else ""
        super().__init__(f"alarm {alarm!r} at iteration {iteration}: {message}{where}")
