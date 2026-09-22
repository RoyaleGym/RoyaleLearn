"""Where frozen actors live.

Snapshots outlive checkpoints, because a rating means nothing without the player it rated, and
they are archived forever: at about a megabyte each, five hundred of them is under a gigabyte.
What is bounded is how many are resident on the device at once.

**No pickled module.** Unpickling an ``nn.Module`` is arbitrary code execution on load, and one
class rename invalidates the whole pool -- a pool that has to be thrown away is a ladder with no
history. What is stored is the actor's tensors in safetensors and a JSON description beside
them, and a load whose ``arch_digest``, ``obs_digest`` or ``codec_table_digest`` disagrees with
the current run is refused with the field named rather than shape-errored halfway through.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from ..api.ladder import SnapshotStore
from ..errors import IdentityMismatch
from ..rollout.envspec import JsonValue

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.policy import Actor, ActorCritic

__all__ = [
    "COMPATIBILITY_FIELDS",
    "INDEX_NAME",
    "SNAPSHOT_FORMAT_VERSION",
    "SPEC_NAME",
    "WEIGHTS_NAME",
    "DiskSnapshotStore",
    "SnapshotSpec",
    "check_compatible",
]

SNAPSHOT_FORMAT_VERSION = 1
WEIGHTS_NAME = "actor.safetensors"
SPEC_NAME = "spec.json"
INDEX_NAME = "index.json"

#: The fields a snapshot and the run loading it must agree on. Everything else in the spec is
#: description; these three decide whether the tensors mean the same thing.
COMPATIBILITY_FIELDS: tuple[str, ...] = ("arch_digest", "obs_digest", "codec_table_digest")


class SnapshotSpec(msgspec.Struct, frozen=True):
    """What is in the folder beside the weights.

    Enough to refuse an incompatible load by name, and enough that the archive is readable
    without the run that wrote it.
    """

    snapshot_id: str
    arch_digest: str = ""
    obs_digest: str = ""
    action_digest: str = ""
    codec_version: int = 0
    codec_table_digest: str = ""
    frame_stack: int = 1
    num_cards: int = 0
    vector_size: int = 0
    step: int = 0
    context: str = ""
    run_id: str = ""
    format_version: int = SNAPSHOT_FORMAT_VERSION
    meta: dict[str, JsonValue] = msgspec.field(default_factory=dict)


def check_compatible(stored: SnapshotSpec, current: SnapshotSpec) -> None:
    """Refuse a snapshot the current run cannot read, naming every field that differs.

    All of them at once: fixing an incompatibility one error message at a time is several
    minutes per field on a machine that takes a minute to build an environment.
    """
    differences = {
        field: (getattr(stored, field), getattr(current, field))
        for field in COMPATIBILITY_FIELDS
        if getattr(stored, field) and getattr(stored, field) != getattr(current, field)
    }
    if differences:
        raise IdentityMismatch(differences)


class DiskSnapshotStore(SnapshotStore):
    """Content-addressed folders under ``snapshots/``, with an index from id to digest.

    Content-addressed because two ids for the same weights are one file, and because the folder
    name is then a checksum of what is in it. The index is what makes ``snap:v17`` a name a
    human can use for it.

    ``max_resident`` is ``ladder.max_resident_opponents`` plus two: a shard-round touches at
    most the residents, and the two spare places are the champion and the candidate a gate is
    auditioning, which are loaded beside them.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        *,
        template: SnapshotSpec | None = None,
        build: Callable[[Any], Actor] | None = None,
        max_resident: int = 4,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.template = template
        self.build = build
        self.max_resident = max(1, int(max_resident))
        self._cache: OrderedDict[tuple[str, str], Actor] = OrderedDict()
        self._index: dict[str, str] = {}
        self._read_index()

    # -- writing ------------------------------------------------------------

    def put(self, snapshot_id: str, ac: ActorCritic, meta: Mapping[str, Any]) -> str:
        """Freeze the actor's weights under their own digest. Returns that digest."""
        payload = _encode_tensors(ac.actor_state_dict_fp16())
        digest = hashlib.sha256(payload).hexdigest()[:16]
        folder = self.root / digest
        folder.mkdir(parents=True, exist_ok=True)
        (folder / WEIGHTS_NAME).write_bytes(payload)
        spec = msgspec.structs.replace(
            self.template or SnapshotSpec(snapshot_id=snapshot_id),
            snapshot_id=snapshot_id,
            step=int(meta.get("step", 0)),
            meta={key: value for key, value in meta.items() if key != "step"},
        )
        (folder / SPEC_NAME).write_bytes(msgspec.json.encode(spec))
        self._index[snapshot_id] = digest
        self._write_index()
        return digest

    # -- reading ------------------------------------------------------------

    def list(self) -> list[str]:
        """Every id in the archive, sorted."""
        return sorted(self._index)

    def digest(self, snapshot_id: str) -> str:
        """The content address of one snapshot."""
        return self._folder(snapshot_id).name

    def spec(self, snapshot_id: str) -> SnapshotSpec:
        """The description stored beside the weights."""
        path = self._folder(snapshot_id) / SPEC_NAME
        return msgspec.json.decode(path.read_bytes(), type=SnapshotSpec)

    def obs_digest(self, snapshot_id: str) -> str:
        """What this snapshot saw, for the evaluation runner's pairing check."""
        return self.spec(snapshot_id).obs_digest

    def get(self, snapshot_id: str, device: Any) -> Actor:
        """A resident actor, built once and kept.

        The cache holds ``max_resident`` of them, which a shard-round never exceeds, so it
        never thrashes: the alternative -- a module rebuilt per round -- costs a construction
        and a host-to-device copy for every batch of a few hundred rows.
        """
        key = (snapshot_id, str(device))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if self.build is None:
            raise RuntimeError(
                "this snapshot store can archive and describe snapshots but not load them: "
                "it was built without a way to construct an actor"
            )
        folder = self._folder(snapshot_id)
        if self.template is not None:
            check_compatible(
                msgspec.json.decode((folder / SPEC_NAME).read_bytes(), type=SnapshotSpec),
                self.template,
            )
        actor = self.build(device)
        state = _decode_tensors((folder / WEIGHTS_NAME).read_bytes(), device)
        actor.load_state_dict(state)  # type: ignore[attr-defined]
        actor.eval()  # type: ignore[attr-defined]
        self._cache[key] = actor
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_resident:
            self._cache.popitem(last=False)
        return actor

    @property
    def resident(self) -> tuple[str, ...]:
        """What is on the device right now, oldest first."""
        return tuple(snapshot_id for snapshot_id, _device in self._cache)

    # -- the index ----------------------------------------------------------

    def _folder(self, snapshot_id: str) -> Path:
        digest = self._index.get(snapshot_id, snapshot_id)
        folder = self.root / digest
        if not folder.is_dir():
            raise KeyError(f"no snapshot {snapshot_id!r} under {self.root}")
        return folder

    def _read_index(self) -> None:
        path = self.root / INDEX_NAME
        if path.exists():
            stored = msgspec.json.decode(path.read_bytes(), type=dict[str, str])
            self._index = dict(stored)

    def _write_index(self) -> None:
        (self.root / INDEX_NAME).write_bytes(msgspec.json.encode(self._index))


def _encode_tensors(state: Mapping[str, Any]) -> bytes:
    """The actor's tensors as safetensors bytes, hashed as they are written.

    As bytes rather than straight to a file so that the digest is of exactly what lands on
    disk: a content address computed from anything else is a second opinion.
    """
    from safetensors.torch import save

    return save({name: tensor.contiguous() for name, tensor in state.items()})


def _decode_tensors(payload: bytes, device: Any) -> dict[str, Any]:
    from safetensors.torch import load

    tensors = load(payload)
    return {name: tensor.to(device) for name, tensor in tensors.items()}
