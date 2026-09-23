"""Where a row goes: a file, a console, and a fan-out over both.

``JsonlSink`` is always installed and is never optional. It needs no service and no account, it
is replayable, and it is the file the resume test compares byte for byte -- a harness whose
only record of a run is in somebody's web dashboard cannot prove that a resume continued the
curve.

Every sink here iterates the row rather than indexing a fixed list of keys, so adding a metric
cannot crash a sink, and a sink that does not understand a key writes it anyway.
"""

from __future__ import annotations

import contextlib
import gzip
import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import msgspec

from ..api.metrics import AlarmResult, MetricRow, MetricsSink
from ..api.rollout import EpisodeRecord
from . import schema

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..identity import RunIdentity

__all__ = [
    "ALARMS_NAME",
    "EPISODES_NAME",
    "METRICS_NAME",
    "CompositeSink",
    "ConsoleSink",
    "JsonlSink",
    "build_sinks",
]

METRICS_NAME = "metrics.jsonl"
EPISODES_NAME = "episodes.jsonl"
ALARMS_NAME = "alarms.jsonl"
IDENTITY_NAME = "identity.json"
CONFIG_NAME = "config.json"


class JsonlSink(MetricsSink):
    """One msgspec-encoded line per iteration, plus the episode and alarm streams.

    It also writes the run's ``identity.json`` and ``config.json`` once, because it is the sink
    that is handed both and the only one that always exists: a run directory without them is a
    directory of numbers nobody can attribute to a configuration.
    """

    FORMAT_VERSION = 1

    def __init__(self, *, flush_every: int = 1, keep_episode_log_iterations: int = 0) -> None:
        self.flush_every = max(1, int(flush_every))
        self.keep_episode_log_iterations = int(keep_episode_log_iterations)
        self.run_dir: Path | None = None
        self._encoder = msgspec.json.Encoder()
        self._handles: dict[str, BinaryIO] = {}
        self._rows = 0
        self._iterations = 0
        #: Whether the episode log has been archived yet. A flag rather than an equality on the
        #: iteration count, so an attempt that could not run is tried again next iteration
        #: instead of being spent.
        self._compacted = False
        #: Attempts that could not complete, for anyone asking why the log is still here.
        self.compaction_failures = 0

    def open(self, *, identity: RunIdentity, config_json: str, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        identity_path = self.run_dir / IDENTITY_NAME
        if not identity_path.exists():
            identity_path.write_bytes(msgspec.json.encode(identity))
        config_path = self.run_dir / CONFIG_NAME
        if not config_path.exists():
            config_path.write_text(config_json, encoding="utf-8")

    def write(self, row: MetricRow) -> None:
        self._iterations += 1
        self._write(METRICS_NAME, [dict(row)])
        self._maybe_compact()

    def write_episodes(self, rows: Sequence[EpisodeRecord]) -> None:
        if rows:
            self._write(EPISODES_NAME, rows)

    def write_alarms(self, alarms: Sequence[AlarmResult]) -> None:
        if alarms:
            self._write(ALARMS_NAME, alarms)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self._handles.clear()

    # -- checkpoint ---------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "jsonl.json").write_bytes(
            msgspec.json.encode(
                {"iterations": self._iterations, "compacted": self._compacted}
            )
        )

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        path = folder / "jsonl.json"
        if not path.exists():
            if strict:
                raise FileNotFoundError(str(path))
            return
        state = msgspec.json.decode(path.read_bytes())
        self._iterations = int(state["iterations"])
        # Absent in checkpoints written before 2026-09-23. False is right for those: the old
        # trigger was an equality on the iteration count, so a resume past the threshold never
        # compacted anyway, and one late archive is better than none.
        self._compacted = bool(state.get("compacted", False))

    # -- internals ----------------------------------------------------------

    def _write(self, name: str, records: Iterable[Any]) -> None:
        handle = self._handle(name)
        for record in records:
            handle.write(self._encoder.encode(record))
            handle.write(b"\n")
        self._rows += 1
        if self._rows % self.flush_every == 0:
            handle.flush()

    def _handle(self, name: str) -> BinaryIO:
        handle = self._handles.get(name)
        if handle is None:
            if self.run_dir is None:
                raise RuntimeError("the JSONL sink was written to before it was opened")
            handle = (self.run_dir / name).open("ab")
            self._handles[name] = handle
        return handle

    def _maybe_compact(self) -> None:
        """Gzip the episode log once it is long enough to be an archive rather than a tail.

        Episodes are the largest stream a run writes and the one nobody reads until something
        looks wrong, which is exactly the shape of a file that should be compressed in place.

        ARCHIVING MUST NOT BE ABLE TO END A RUN, and before 2026-09-23 it could. On Windows
        ``unlink`` raises ``PermissionError`` while any other process holds a handle on the
        file -- and reading ``episodes.jsonl`` during a run is what the train session's job
        requires. That exception came out of ``write``, which the coordinator calls between
        setting ``_learner_ahead_of_rows`` and clearing it, so the run fell over AND its
        emergency checkpoint was suppressed. It was not one-shot either: the iteration counter
        is restored from the pre-threshold checkpoint, so a resumed run reached the same line
        and died again.

        THE RENAME IS THE PROBE. Moving the file aside costs nothing and fails in exactly the
        same way the delete would, so a held file is discovered before the expensive copy rather
        than after it -- and the original is out of the way before the archive exists, which is
        what stops a complete ``episodes.jsonl`` sitting beside a complete
        ``episodes.jsonl.gz`` that claims to have replaced it. If the copy then fails, the
        original goes back.
        """
        keep = self.keep_episode_log_iterations
        if not keep or self._compacted or self._iterations < keep or self.run_dir is None:
            return
        handle = self._handles.pop(EPISODES_NAME, None)
        if handle is not None:
            handle.flush()
            handle.close()
        source = self.run_dir / EPISODES_NAME
        if not source.exists():
            self._compacted = True
            return
        staged = source.with_suffix(".jsonl.compacting")
        archive = source.with_suffix(".jsonl.gz")
        try:
            os.replace(source, staged)
        except OSError:
            # Somebody is holding it. Nothing has moved, the log keeps growing, and the next
            # iteration tries again.
            self.compaction_failures += 1
            return
        try:
            with staged.open("rb") as raw, gzip.open(archive, "wb") as out:
                shutil.copyfileobj(raw, out)
            staged.unlink()
        except OSError:
            self.compaction_failures += 1
            with contextlib.suppress(OSError):
                archive.unlink()
            with contextlib.suppress(OSError):
                os.replace(staged, source)
            return
        self._compacted = True


class ConsoleSink(MetricsSink):
    """The per-iteration block, printed by iterating the row.

    Grouped by the part of the key before the slash, in the schema's own order, with anything
    the schema does not know printed last under its own group: a metric added this morning
    appears on the console this morning.
    """

    FORMAT_VERSION = 1

    def __init__(self, *, every: int = 1, width: int = 34, stream: Any = None) -> None:
        self.every = max(1, int(every))
        self.width = int(width)
        self.stream = stream
        self._iterations = 0

    def open(self, *, identity: RunIdentity, config_json: str, run_dir: Path) -> None:
        self._print(f"run {identity.master_seed} in {run_dir}")

    def write(self, row: MetricRow) -> None:
        self._iterations += 1
        if self._iterations % self.every:
            return
        groups: dict[str, list[tuple[str, Any]]] = {}
        for key, value in row.items():
            group, _, name = key.partition("/")
            groups.setdefault(group, []).append((name or key, value))
        order = [group for group in schema.groups() if group in groups]
        order += [group for group in groups if group not in order]
        iteration = row.get("run/iteration", self._iterations)
        self._print(f"-- iteration {iteration} " + "-" * 40)
        for group in order:
            self._print(f"  {group}")
            for name, value in groups[group]:
                self._print(f"    {name:<{self.width}} {_format(value)}")

    def close(self) -> None:
        return

    def save_checkpoint(self, folder: Path) -> None:
        return

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        return

    def _print(self, line: str) -> None:
        print(line, file=self.stream)


class CompositeSink(MetricsSink):
    """Fans one row out to several sinks.

    A sink that raises takes the iteration down with it, which is deliberate: a metrics sink
    that silently stops writing is how a run becomes unattributable, and the one sink that must
    never be optional is the file.
    """

    FORMAT_VERSION = 1

    def __init__(self, sinks: Sequence[MetricsSink]) -> None:
        self.sinks = list(sinks)

    def open(self, *, identity: RunIdentity, config_json: str, run_dir: Path) -> None:
        for sink in self.sinks:
            sink.open(identity=identity, config_json=config_json, run_dir=run_dir)

    def write(self, row: MetricRow) -> None:
        for sink in self.sinks:
            sink.write(row)

    def write_episodes(self, rows: Sequence[EpisodeRecord]) -> None:
        for sink in self.sinks:
            sink.write_episodes(rows)

    def write_alarms(self, alarms: Sequence[AlarmResult]) -> None:
        for sink in self.sinks:
            sink.write_alarms(alarms)

    def write_artifact(self, name: str, path: Path) -> None:
        for sink in self.sinks:
            sink.write_artifact(name, path)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for position, sink in enumerate(self.sinks):
            sink.save_checkpoint(folder / f"{position}-{type(sink).__name__}")

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        for position, sink in enumerate(self.sinks):
            sink.load_checkpoint(folder / f"{position}-{type(sink).__name__}", strict=strict)


def _format(value: Any) -> str:
    """A value as the console shows it: four significant figures, thousands on an integer."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def build_sinks(specs: Iterable[Any], **options: Any) -> MetricsSink:
    """The composite a config's sink list describes.

    ``jsonl`` is installed whether or not the list names it. Everything else is optional, and a
    disabled sink is left out rather than constructed and told to do nothing -- except the
    wandb decorator, which is a pass-through by design and is the one sink whose absence would
    change the shape of the stack.

    ``options`` carries the settings that live on the config beside the sink list rather than
    inside a spec's own ``options`` -- ``build_sinks(specs, jsonl={"keep_episode_log_iterations":
    ...})``. They are defaults: a spec that names the same key wins, and a sink the list does
    not mention still gets them when it is installed anyway.
    """
    from .viser_sink import ViserSink
    from .wandb_sink import WandbSink

    sinks: list[MetricsSink] = []
    wandb_spec = None
    for spec in specs:
        kind = spec.kind if hasattr(spec, "kind") else str(spec)
        enabled = getattr(spec, "enabled", True)
        settings: dict[str, Any] = dict(options.get(kind, {}))
        settings.update(dict(getattr(spec, "options", {}) or {}))
        if kind == "wandb":
            wandb_spec = (enabled, settings)
            continue
        if not enabled:
            continue
        if kind == "jsonl":
            sinks.append(JsonlSink(**settings))
        elif kind == "console":
            sinks.append(ConsoleSink(**settings))
        elif kind == "viser":
            sinks.append(ViserSink(**settings))
        else:
            raise ValueError(f"unknown metrics sink {kind!r}")
    if not any(isinstance(sink, JsonlSink) for sink in sinks):
        sinks.insert(0, JsonlSink(**options.get("jsonl", {})))
    composite = CompositeSink(sinks)
    if wandb_spec is not None:
        enabled, settings = wandb_spec
        return WandbSink(composite, enable=enabled, **settings)
    return composite
