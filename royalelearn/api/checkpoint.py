"""What a checkpoint is: components that save themselves, a store that writes them atomically,
and a manifest that says what was written and proves the bytes are still what was written.

Every component owns its folder and its own pair of methods. Adding a component to a checkpoint
is adding a folder name and a dict entry; there is no central serialiser to edit, and no place
where a component's state is described twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import msgspec

# Imported at run time, not behind TYPE_CHECKING: Manifest carries a RunIdentity, and msgspec
# resolves a Struct's annotations when a decoder for it is built, so a name that exists only
# for a type checker would make a manifest unreadable.
from ..identity import RunIdentity

__all__ = ["CheckpointStore", "Checkpointable", "Manifest", "RngState"]


@runtime_checkable
class Checkpointable(Protocol):
    """Anything that goes into a checkpoint folder of its own.

    ``FORMAT_VERSION`` is the component's own, not the checkpoint's: a component may change its
    files without the checkpoint format moving, and the manifest records each one so that a
    load can say which component it could not read.
    """

    FORMAT_VERSION: int

    def save_checkpoint(self, folder: Path) -> None: ...

    def load_checkpoint(self, folder: Path, *, strict: bool) -> None:
        """With ``strict=False``, a missing file prints the exact path it wanted and continues
        with a default. With ``strict=True`` it raises. ``strict`` defaults to True on resume:
        tolerate-everything is right for a research tool and wrong for a harness that promises
        the curve continues."""


class Manifest(msgspec.Struct):
    """What one checkpoint is, beside the folders that hold it.

    ``config`` is the resolved config verbatim, so a checkpoint is self-describing without the
    run directory around it, and ``files`` is every file in the checkpoint with its sha256, so a
    truncated write from a crash six weeks ago is an error on load rather than a silently wrong
    resume.
    """

    format_version: int
    run_id: str
    run_name: str
    identity: RunIdentity
    config: dict[str, Any]
    config_hash: str
    iteration: int
    cumulative_env_steps: int
    cumulative_timesteps: int
    cumulative_model_updates: int
    wall_seconds: float
    created_unix_ns: int
    state_digest: str
    component_versions: dict[str, int]
    files: dict[str, str]
    #: Seconds spent gating so far, and the last gate's decision: run state, like the wall clock.
    #: None on a manifest written before either was recorded.
    gate_seconds: float | None = None
    #: Stored as plain data and converted by the coordinator, so this module need not import
    #: the ladder's types: ``api.ladder`` already imports this one.
    last_decision: dict[str, Any] | None = None


class RngState(msgspec.Struct):
    """Every random stream's position at the moment a checkpoint was written.

    One reference learner saves none of this; the other re-seeds from the run's INITIAL seed on
    load, so a run resumed at ten million steps draws the same action noise it drew at step
    zero. Because every stream here is name-addressed, restoring the iteration counter and the
    shard positions restores the stream, not merely the parameters.
    """

    master_seed: int
    torch_cpu: str
    torch_cuda: list[str]
    python_random: list[Any]
    numpy_minibatch: dict[str, Any]
    iteration: int
    shard_streams: list[dict[str, Any]]
    eval_seed_set_sha: str


class CheckpointStore(ABC):
    """Where checkpoints live, and the only thing that writes or reads them."""

    @abstractmethod
    def write(self, components: Mapping[str, Checkpointable], manifest: Manifest) -> Path:
        """Write atomically: into ``<name>.partial/``, fsync each file, then ``os.replace``.

        The directory itself is not fsynced. ``os.fsync`` on a directory handle is a POSIX
        guarantee and raises on Windows, where this harness's default profile runs, so the
        durability step is per file and the atomic step is the rename -- which is atomic on both
        platforms, and is the property recovery actually needs.
        """

    @abstractmethod
    def read(
        self, path: Path, components: Mapping[str, Checkpointable], *, strict: bool
    ) -> Manifest:
        """Verify every hash in the manifest, then load each component. A mismatch raises
        ``CheckpointFormatError`` naming the first file that failed."""

    @abstractmethod
    def latest(self, run_dir: Path) -> Path | None:
        """The newest checkpoint, from the run index rather than from parsing directory names:
        ``int(x) for x in os.listdir(...)`` crashes on a stray file, and the save is called from
        the crash handler, which is where the user most needs it not to."""

    @abstractmethod
    def prune(self, run_dir: Path, keep: int) -> list[Path]:
        """Remove all but the newest ``keep``, and return what was removed."""
