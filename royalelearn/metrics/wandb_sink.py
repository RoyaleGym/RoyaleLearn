"""Weights and Biases, as a decorator over whatever is underneath it.

A decorator and not a peer, for two reasons. The run must keep working when wandb is not
installed, not configured or not reachable, so the sink that owns the file has to be inside
this one rather than beside it. And the wandb run id has to survive into the checkpoint: a
resumed run that starts a second wandb run has two half curves and no way to join them, so the
id is checkpointed here and handed back to ``wandb.init(id=..., resume="allow")``.

``enable=False`` makes it a pure pass-through: nothing is imported, nothing is sent, and the
inner sink sees exactly what it would have seen without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from ..api.metrics import AlarmResult, MetricRow, MetricsSink
from ..api.rollout import EpisodeRecord
from .records import flatten

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..identity import RunIdentity

__all__ = ["WandbSink"]


class WandbSink(MetricsSink):
    """Sends the row to wandb and passes it down unchanged."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        inner: MetricsSink,
        *,
        enable: bool = False,
        project: str = "royalelearn",
        entity: str | None = None,
        tags: Sequence[str] = (),
        mode: str | None = None,
    ) -> None:
        self.inner = inner
        self.enable = bool(enable)
        self.project = project
        self.entity = entity
        self.tags = tuple(tags)
        self.mode = mode
        self.run_id: str | None = None
        self._run: Any = None

    def open(self, *, identity: RunIdentity, config_json: str, run_dir: Path) -> None:
        self.inner.open(identity=identity, config_json=config_json, run_dir=run_dir)
        if not self.enable:
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - exercised only with the extra absent
            raise RuntimeError(
                "the wandb sink is enabled and wandb is not installed: "
                "pip install 'royalelearn[wandb]', or disable the sink in the config"
            ) from exc
        # The identity is in the wandb config as well as the resolved config: what a run IS
        # belongs beside the numbers, not only in a file on the machine that produced them.
        config = {
            "config": msgspec.json.decode(config_json),
            "identity": msgspec.json.decode(msgspec.json.encode(identity)),
        }
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            tags=list(self.tags),
            mode=self.mode,
            dir=str(run_dir),
            id=self.run_id,
            resume="allow",
            config=config,
        )
        self.run_id = getattr(self._run, "id", self.run_id)

    def write(self, row: MetricRow) -> None:
        if self._run is not None:
            # Flattened on the way out: wandb panels are named by a flat key, and a nested
            # mapping would arrive as one opaque column.
            self._run.log(flatten(dict(row)), step=int(row.get("run/iteration") or 0))
        self.inner.write(row)

    def write_episodes(self, rows: Sequence[EpisodeRecord]) -> None:
        self.inner.write_episodes(rows)

    def write_alarms(self, alarms: Sequence[AlarmResult]) -> None:
        if self._run is not None:
            for alarm in alarms:
                self._run.log(
                    {f"alarm/{alarm.name}": 1.0 if alarm.fired else 0.0},
                    step=alarm.iteration,
                )
        self.inner.write_alarms(alarms)

    def write_artifact(self, name: str, path: Path) -> None:
        if self._run is not None:
            self._run.save(str(path), base_path=str(path.parent), policy="now")
        self.inner.write_artifact(name, path)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
        self.inner.close()

    # -- checkpoint ---------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        """The run id here, the decorated sink's state in a folder of its own.

        Nesting rather than flattening: the decorator's state and the file sink's state are
        different components that happen to be stacked, and a checkpoint that mixed them could
        not be read by a stack assembled differently.
        """
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "wandb.json").write_bytes(msgspec.json.encode({"run_id": self.run_id}))
        self.inner.save_checkpoint(folder / "inner")

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        path = folder / "wandb.json"
        if path.exists():
            self.run_id = msgspec.json.decode(path.read_bytes()).get("run_id")
        elif strict:
            raise FileNotFoundError(str(path))
        else:
            print(f"no wandb state at {path}; this resume starts a new wandb run")
        self.inner.load_checkpoint(folder / "inner", strict=strict)
