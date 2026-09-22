"""Checkpoints: per-component folders, a manifest that proves the bytes, and an atomic write.

Every component owns its folder and its own pair of methods, so adding one to a checkpoint is
adding a folder name and a dict entry rather than editing a central serialiser. Nothing about a
component's state is described in two places.

**Atomicity.** Everything is written into ``<name>.partial/``, every file is fsynced, and then
the directory is renamed into place. The directory itself is not fsynced: ``os.fsync`` on a
directory handle is a POSIX guarantee and raises on Windows, where this harness's default
profile runs, so the durability step is per file and the atomic step is the rename -- which is
atomic on both platforms and is the property recovery actually needs. A crash mid-write leaves
the previous checkpoint intact and the partial directory obviously named.

**No pickle on any path this module writes.** safetensors for weights, msgspec JSON for
everything else, and ``torch.load(weights_only=True)`` wherever a component reads an optimizer
state back. A checkpoint is loaded six weeks later from a directory nobody has looked at since;
it must not be able to run code.
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from .api.checkpoint import Checkpointable, CheckpointStore, Manifest, RngState
from .errors import CheckpointFormatError, IdentityMismatch
from .identity import RunIdentity, identity_differences

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "INDEX_NAME",
    "MANIFEST_NAME",
    "PARTIAL_SUFFIX",
    "CheckpointIndex",
    "DirCheckpointStore",
    "IndexEntry",
    "RngComponent",
    "check_described",
    "check_resume",
    "config_differences",
    "sha256_of",
]

CHECKPOINT_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.json"
PARTIAL_SUFFIX = ".partial"
CHECKPOINTS_DIR = "checkpoints"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """The hash the manifest records, read in chunks so a buffer file costs no memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class IndexEntry(msgspec.Struct):
    """One checkpoint, as the index knows it."""

    name: str
    iteration: int
    cumulative_env_steps: int
    created_unix_ns: int


class CheckpointIndex(msgspec.Struct):
    """The run's checkpoints, newest last.

    ``latest()`` reads this rather than parsing directory names. ``int(x) for x in
    os.listdir(...)`` crashes the save on any stray file, and the save is called from the crash
    handler, which is exactly where the user most needs it not to.
    """

    format_version: int = CHECKPOINT_FORMAT_VERSION
    entries: list[IndexEntry] = msgspec.field(default_factory=list)


class RngComponent:
    """Every random stream's position, as one folder of a checkpoint.

    One reference learner saves none of this; the other re-seeds all three generators from the
    run's *initial* seed on load, so a run resumed at ten million steps draws the same action
    noise it drew at step zero. Because every stream in this harness is name-addressed,
    restoring the iteration counter and the shard positions restores the stream itself rather
    than merely the parameters -- the torch and python states below are for the few draws that
    are not name-addressed, such as a dropout mask.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        master_seed: int,
        generator: np.random.Generator | None = None,
        eval_seed_set_sha: str = "",
    ) -> None:
        self.master_seed = int(master_seed)
        self.generator = generator
        self.eval_seed_set_sha = eval_seed_set_sha
        self.iteration = 0
        self.shard_streams: list[dict[str, Any]] = []

    def capture(self) -> RngState:
        """Everything that would have to be true again for the next draw to be the same one."""
        torch_cpu = ""
        torch_cuda: list[str] = []
        torch = _torch()
        if torch is not None:
            torch_cpu = torch.get_rng_state().numpy().tobytes().hex()
            if torch.cuda.is_available():
                torch_cuda = [
                    state.numpy().tobytes().hex() for state in torch.cuda.get_rng_state_all()
                ]
        return RngState(
            master_seed=self.master_seed,
            torch_cpu=torch_cpu,
            torch_cuda=torch_cuda,
            python_random=_jsonable(random.getstate()),
            numpy_minibatch=(
                dict(self.generator.bit_generator.state) if self.generator is not None else {}
            ),
            iteration=self.iteration,
            shard_streams=list(self.shard_streams),
            eval_seed_set_sha=self.eval_seed_set_sha,
        )

    def restore(self, state: RngState) -> None:
        self.master_seed = state.master_seed
        self.iteration = state.iteration
        self.shard_streams = list(state.shard_streams)
        self.eval_seed_set_sha = state.eval_seed_set_sha
        random.setstate(_tupled(state.python_random))
        if self.generator is not None and state.numpy_minibatch:
            self.generator.bit_generator.state = state.numpy_minibatch
        torch = _torch()
        if torch is not None and state.torch_cpu:
            torch.set_rng_state(_byte_tensor(torch, state.torch_cpu))
            if state.torch_cuda and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [_byte_tensor(torch, hexed) for hexed in state.torch_cuda]
                )

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "rng.json").write_bytes(msgspec.json.encode(self.capture()))

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        path = folder / "rng.json"
        if not path.exists():
            if strict:
                raise CheckpointFormatError(f"no RNG state at {path}")
            print(f"no RNG state at {path}; the streams restart from the master seed")
            return
        self.restore(msgspec.json.decode(path.read_bytes(), type=RngState))


class DirCheckpointStore(CheckpointStore):
    """One directory per checkpoint, named by the cumulative env step that produced it."""

    def __init__(self, run_dir: str | Path, *, keep: int = 10) -> None:
        self.run_dir = Path(run_dir)
        self.keep = int(keep)

    def folder(self, run_dir: Path | None = None) -> Path:
        return (run_dir or self.run_dir) / CHECKPOINTS_DIR

    # -- writing ------------------------------------------------------------

    def write(self, components: Mapping[str, Checkpointable], manifest: Manifest) -> Path:
        root = self.folder()
        root.mkdir(parents=True, exist_ok=True)
        name = f"{manifest.cumulative_env_steps:012d}"
        partial = root / f"{name}{PARTIAL_SUFFIX}"
        final = root / name
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)

        for component_name, component in components.items():
            component.save_checkpoint(partial / component_name)

        files = {
            path.relative_to(partial).as_posix(): sha256_of(path)
            for path in sorted(partial.rglob("*"))
            if path.is_file()
        }
        written = msgspec.structs.replace(
            manifest,
            format_version=manifest.format_version or CHECKPOINT_FORMAT_VERSION,
            component_versions={
                component_name: int(getattr(component, "FORMAT_VERSION", 1))
                for component_name, component in components.items()
            },
            files=files,
        )
        (partial / MANIFEST_NAME).write_bytes(msgspec.json.encode(written))
        for path in sorted(partial.rglob("*")):
            if path.is_file():
                _fsync_file(path)

        # os.replace over an existing directory is not portable, so the previous copy of this
        # same step goes first. It is the same step, so nothing is lost that the new one does
        # not have; a crash between the two leaves the older checkpoints untouched.
        if final.exists():
            shutil.rmtree(final)
        os.replace(partial, final)
        self._index_append(
            root,
            IndexEntry(
                name=name,
                iteration=written.iteration,
                cumulative_env_steps=written.cumulative_env_steps,
                created_unix_ns=written.created_unix_ns or time.time_ns(),
            ),
        )
        return final

    # -- reading ------------------------------------------------------------

    def read(
        self, path: Path, components: Mapping[str, Checkpointable], *, strict: bool = True
    ) -> Manifest:
        path = Path(path)
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.exists():
            raise CheckpointFormatError(f"no {MANIFEST_NAME} in {path}")
        try:
            manifest = msgspec.json.decode(manifest_path.read_bytes(), type=Manifest)
        except msgspec.ValidationError as exc:
            raise CheckpointFormatError(f"{manifest_path} is not a manifest: {exc}") from exc
        if manifest.format_version > CHECKPOINT_FORMAT_VERSION:
            raise CheckpointFormatError(
                f"{manifest_path} is checkpoint format {manifest.format_version} and this "
                f"royalelearn reads {CHECKPOINT_FORMAT_VERSION}"
            )
        self.verify(path, manifest)
        for component_name, component in components.items():
            folder = path / component_name
            if not folder.is_dir():
                if strict:
                    raise CheckpointFormatError(
                        f"the checkpoint at {path} has no {component_name!r} folder"
                    )
                print(f"no {component_name!r} in {path}; continuing with its default state")
                continue
            component.load_checkpoint(folder, strict=strict)
        return manifest

    def verify(self, path: Path, manifest: Manifest) -> None:
        """Every file in the manifest, hashed and compared. Names the first that fails.

        This is what turns a truncated write from a crash six weeks ago into an error at the
        moment of the resume rather than into a run that continues from something else.
        """
        for relative in sorted(manifest.files):
            expected = manifest.files[relative]
            candidate = path / relative
            if not candidate.exists():
                raise CheckpointFormatError(f"{candidate} is in the manifest and not on disk")
            actual = sha256_of(candidate)
            if actual != expected:
                raise CheckpointFormatError(
                    f"{candidate} hashes to {actual[:16]} and the manifest says "
                    f"{expected[:16]}: the checkpoint is not what was written"
                )

    def latest(self, run_dir: Path | None = None) -> Path | None:
        entries = self._entries(self.folder(run_dir))
        if not entries:
            return None
        newest = max(entries, key=lambda e: (e.cumulative_env_steps, e.created_unix_ns))
        return self.folder(run_dir) / newest.name

    def prune(self, run_dir: Path | None = None, keep: int | None = None) -> list[Path]:
        """Keep the newest ``keep`` and remove the rest, along with any abandoned partial."""
        root = self.folder(run_dir)
        limit = self.keep if keep is None else int(keep)
        entries = sorted(
            self._entries(root), key=lambda e: (e.cumulative_env_steps, e.created_unix_ns)
        )
        removed: list[Path] = []
        for entry in entries[: max(0, len(entries) - limit)]:
            folder = root / entry.name
            if folder.is_dir():
                shutil.rmtree(folder)
            removed.append(folder)
        if root.is_dir():
            for partial in sorted(root.glob(f"*{PARTIAL_SUFFIX}")):
                if partial.is_dir():
                    shutil.rmtree(partial)
                    removed.append(partial)
        kept = {entry.name for entry in entries[max(0, len(entries) - limit) :]}
        self._index_write(root, [entry for entry in entries if entry.name in kept])
        return removed

    # -- the index ----------------------------------------------------------

    def _entries(self, root: Path) -> list[IndexEntry]:
        """What the index says, or what the directory says when there is no index.

        The fallback reads each candidate's own manifest rather than its name: a stray file or
        a half-written folder is skipped, and nothing is ever inferred from a directory name.
        """
        if not root.is_dir():
            return []
        path = root / INDEX_NAME
        if path.exists():
            index = msgspec.json.decode(path.read_bytes(), type=CheckpointIndex)
            return [entry for entry in index.entries if (root / entry.name).is_dir()]
        entries: list[IndexEntry] = []
        for candidate in sorted(root.iterdir()):
            manifest_path = candidate / MANIFEST_NAME
            if not candidate.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = msgspec.json.decode(manifest_path.read_bytes(), type=Manifest)
            except msgspec.DecodeError:
                continue
            entries.append(
                IndexEntry(
                    name=candidate.name,
                    iteration=manifest.iteration,
                    cumulative_env_steps=manifest.cumulative_env_steps,
                    created_unix_ns=manifest.created_unix_ns,
                )
            )
        return entries

    def _index_append(self, root: Path, entry: IndexEntry) -> None:
        entries = [e for e in self._entries(root) if e.name != entry.name]
        entries.append(entry)
        self._index_write(root, entries)

    def _index_write(self, root: Path, entries: Iterable[IndexEntry]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        index = CheckpointIndex(entries=list(entries))
        path = root / INDEX_NAME
        path.write_bytes(msgspec.json.encode(index))
        _fsync_file(path)


def config_differences(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Every leaf in which two configs differ, keyed by its dotted path.

    Over the whole tree rather than the top level, so that a changed learning rate reads as
    ``ppo.lr_actor`` instead of as the whole ``ppo`` block having moved.
    """
    differences: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(before) | set(after)):
        left, right = before.get(key), after.get(key)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for name, pair in config_differences(left, right).items():
                differences[f"{key}.{name}"] = pair
        elif left != right:
            differences[key] = (left, right)
    return differences


def check_resume(
    manifest: Manifest,
    identity: RunIdentity,
    config: Mapping[str, Any],
    *,
    allow_drift: bool = False,
) -> dict[str, tuple[Any, Any]]:
    """What a resume has to agree about, checked once before anything is loaded.

    A difference in an identity field is a refusal rather than a warning: a policy trained on
    one catalogue's observation vector cannot load into another's, and the failure that follows
    is a shape error deep inside a forward pass rather than a sentence naming the field. Every
    other difference is printed and continued past, because a run that resumes with a different
    checkpoint interval is the same run.

    Returns the config differences it printed. ``allow_drift`` records the identity differences
    instead of refusing them, which is what ``--allow-identity-drift`` is for.
    """
    drift = identity_differences(manifest.identity, identity)
    if drift and not allow_drift:
        raise IdentityMismatch(drift)
    differences = config_differences(manifest.config, config)
    for name, (was, now) in differences.items():
        print(f"config differs from the checkpoint's: {name}: {was!r} -> {now!r}")
    return differences


class _RowIdentity(msgspec.Struct):
    """The two fields of a metric row a checkpoint answers to; the decoder skips the rest."""

    iteration: int = msgspec.field(default=-1, name="run/iteration")
    state_digest: str = msgspec.field(default="", name="run/state_digest")


def check_described(manifest: Manifest, rows: Path) -> None:
    """Refuse a checkpoint whose learner no metric row of its iteration describes.

    A checkpoint's weights are supposed to be ones the run's record reports: the row of the
    iteration its manifest names carries the same state digest. One that breaks this holds a
    learner trained past its row -- which is what the emergency save wrote while the update
    ran before the batch was judged -- and resuming it continues the run from a state its
    record does not contain, under counters that belong to a different learner.

    Any row of that iteration will do: a run resumed from an earlier checkpoint writes the
    iterations after it a second time, and each copy describes a learner that existed. What
    cannot be compared is let through rather than refused -- no metric file, no row of that
    iteration, a line torn by a crash mid-write -- because an absence says nothing about the
    weights. Past iteration zero, which never has a row, it is printed: a resume the record
    could not vouch for should not read like one it did.
    """
    described: set[str] = set()
    if rows.is_file():
        decoder = msgspec.json.Decoder(_RowIdentity)
        with rows.open("rb") as handle:
            for line in handle:
                try:
                    row = decoder.decode(line)
                except msgspec.DecodeError:
                    continue
                if row.iteration == manifest.iteration and row.state_digest:
                    described.add(row.state_digest)
    if manifest.state_digest in described:
        return
    if not described:
        if manifest.iteration:
            print(
                f"no metrics row of iteration {manifest.iteration} in {rows}: the checkpoint's "
                f"learner is not checked against the run's record"
            )
        return
    recorded = ", ".join(sorted(digest[:16] for digest in described))
    raise CheckpointFormatError(
        f"the checkpoint of iteration {manifest.iteration} holds learner "
        f"{manifest.state_digest[:16]} and no metrics row describes it: {rows} records "
        f"{recorded} for that iteration. Its weights were trained past the row its counters "
        f"belong to; resume from an earlier checkpoint with --checkpoint"
    )


def _fsync_file(path: Path) -> None:
    """Durability, one file at a time. The directory handle is deliberately not fsynced.

    Opened for update rather than for reading: Windows commits a handle's buffers and refuses
    to do it through one that cannot write.
    """
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _byte_tensor(torch: Any, hexed: str) -> Any:
    return torch.frombuffer(bytearray.fromhex(hexed), dtype=torch.uint8).clone()


def _jsonable(state: Any) -> Any:
    """Python's random state as JSON: tuples become lists, everything else stands."""
    if isinstance(state, tuple):
        return [_jsonable(item) for item in state]
    return state


def _tupled(state: Any) -> Any:
    """The inverse: ``random.setstate`` insists on the tuples it handed out."""
    if isinstance(state, list):
        return tuple(_tupled(item) for item in state)
    return state
