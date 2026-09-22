"""Where numbers go, and what makes one an alarm.

One row per iteration, flat, merged from all three sources -- the environment's round scalars
and episode records, the learner's ``UpdateResult``, the ladder's rating table and gate
decision. A sink writes it somewhere; an alarm reads it and decides whether the run should stop.

``metrics/schema.py`` is the single source of truth for what a key means. A sink never indexes a
fixed key list, so a new metric cannot crash the console, and a run that emits a key the schema
does not know is a test failure rather than an undocumented column.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from .checkpoint import Checkpointable
from .rollout import EpisodeRecord

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..identity import RunIdentity

__all__ = ["Alarm", "AlarmResult", "MetricRow", "MetricValue", "MetricsSink"]

#: What may appear in a metric row. Strings are there for the few keys that are labels --
#: ``run/determinism_tier``, ``ladder/champion_id``, ``run/state_digest`` -- and are written to
#: the same row rather than to a second stream, so one line describes one iteration completely.
MetricValue = float | int | str | bool | None

#: One iteration, flat, keyed ``group/name``.
MetricRow = Mapping[str, MetricValue]


class MetricsSink(ABC, Checkpointable):
    """Somewhere a row goes."""

    @abstractmethod
    def open(self, *, identity: RunIdentity, config_json: str, run_dir: Path) -> None: ...

    @abstractmethod
    def write(self, row: MetricRow) -> None:
        """One flat dict per iteration. Implementers MUST NOT mutate the row and MUST NOT raise
        on an unknown key."""

    def write_episodes(self, rows: Sequence[EpisodeRecord]) -> None:
        """Finished episodes. Default: ignore."""

    def write_alarms(self, alarms: Sequence[AlarmResult]) -> None:
        """Alarms that fired this iteration. Default: ignore."""

    def write_artifact(self, name: str, path: Path) -> None:
        """A file produced this iteration -- a heatmap, a bundle. Default: ignore."""

    @abstractmethod
    def close(self) -> None: ...


class AlarmResult(msgspec.Struct, frozen=True):
    """One alarm's verdict for one iteration.

    ``consecutive`` is how many iterations in a row the predicate has held, so a row written
    before the patience is spent still records that something was building.
    """

    name: str
    severity: str
    iteration: int
    fired: bool
    consecutive: int
    message: str
    values: dict[str, float]


class Alarm(ABC):
    """A predicate over a metric row, with a severity and a patience.

    ``severity="warn"`` logs and writes an ``alarms.jsonl`` row; ``severity="halt"``
    additionally writes a checkpoint and a diagnostic bundle, then raises ``AlarmHalt``. An
    alarm can stop a run and can never alter a value, which is why its thresholds are recorded
    but excluded from the run identity.
    """

    name: str
    severity: str
    patience: int
    #: The metric keys this alarm reads. They must exist in ``metrics.schema``; the test that
    #: says so is what keeps an alarm from watching a key nobody emits any more.
    keys: tuple[str, ...]

    @abstractmethod
    def holds(self, row: MetricRow) -> bool:
        """Whether the condition is true for this row, ignoring patience."""

    def message(self, row: MetricRow) -> str:
        """What to say when it fires. The default names the keys and their values."""
        parts = [f"{key}={row.get(key)!r}" for key in self.keys]
        return f"{self.name}: " + ", ".join(parts)
