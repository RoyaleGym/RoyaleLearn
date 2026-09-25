"""Demonstration shards: rows stored exactly, keyed to the engine that produced them (19.10).

A shard directory is a ``manifest.json`` and part files, one compressed array per column. The
rows are what the environment emitted, stored exactly -- the spatial planes as ``uint16`` value x
1000 with a refusal for any value that does not survive the round trip -- and quantised only when
they are read, by the reading run's own codec. So a row reaches a cloned actor through exactly the
pack and unpack a rollout row takes into the update, and the shards outlive a change of codec
table.

The directory is named by an ENGINE KEY over everything that decides what the engine does with a
command: the compiled file, the data compiled into it, the calibration values as loaded, the
engine's constructor parameters, the card catalogue and the observation layout. Rows replayed on
another engine are another dataset, and the reader refuses them unless told why not to.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec
import numpy as np

from ..errors import PreflightError
from ..rollout.envspec import digest_of
from .split import is_validation

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor

    from ..api.policy import ObsBatch
    from ..api.rollout import EnvSpec
    from ..config import RunConfig
    from ..learn.rows import RowCodec

__all__ = [
    "FLAG_OTHER_COMMAND",
    "FLAG_PROJECTED",
    "MANIFEST_NAME",
    "SHARD_FORMAT_VERSION",
    "PartInfo",
    "ShardBatch",
    "ShardContext",
    "ShardManifest",
    "ShardReader",
    "ShardWriter",
]

SHARD_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
#: The driver's own flag bits. A producer names more in ``flag_names``, from bit 8 up.
FLAG_PROJECTED = 1 << 0
FLAG_OTHER_COMMAND = 1 << 1
DRIVER_FLAGS: dict[int, str] = {FLAG_PROJECTED: "projected", FLAG_OTHER_COMMAND: "other_command"}
#: The spatial planes' fixed point: value x 1000 as uint16.
SPATIAL_SCALE = 1000
_UINT16_MAX = 65535


# --------------------------------------------------------------------------
# What the rows were produced against
# --------------------------------------------------------------------------


class ShardContext(msgspec.Struct, frozen=True, kw_only=True):
    """The environment a set of rows belongs to, as the reading run can check it."""

    obs_digest: str
    action_digest: str
    catalogue_sha256: str
    card_names: list[str]
    binary_sha256: str
    build_digest: str
    calibration_digest: str
    #: The engine's own constructor parameters, as its ``config()`` states them. What the
    #: calibration digest does not cover: a card level, a path search.
    engine_params: dict[str, Any]
    spatial_shape: tuple[int, int, int]
    spatial_planes: list[str]
    static_planes: list[int]
    vector_size: int
    n_actions: int
    card_id_planes: int
    #: The environment's whole ``config()``, without its truncation: a record, not a key.
    env_config: dict[str, Any]

    @property
    def engine_key(self) -> str:
        """First sixteen hex of the sha256 over what decides the engine's behaviour."""
        params = {k: v for k, v in self.engine_params.items() if k != "cards"}
        return digest_of(
            {
                "binary_sha256": self.binary_sha256,
                "build_digest": self.build_digest,
                "calibration_digest": self.calibration_digest,
                "catalogue_sha256": self.catalogue_sha256,
                "obs_digest": self.obs_digest,
                "engine_params": params,
            }
        )[:16]

    @classmethod
    def from_parts(
        cls,
        spec: EnvSpec,
        env_config: Mapping[str, Any],
        cards: Sequence[Any],
        action_digest: str,
    ) -> ShardContext:
        """From an environment already built: its spec, its ``config()`` and its catalogue."""
        from ..identity import engine_build

        build = engine_build(env_config, cards)
        engine = env_config.get("engine") or {}
        card_ids = spec.obs_space.get("card_ids")
        return cls(
            obs_digest=spec.obs_digest,
            action_digest=action_digest,
            catalogue_sha256=build.catalogue_sha256,
            card_names=[str(card.name) for card in cards],
            binary_sha256=build.binary_sha256,
            build_digest=build.build_digest,
            calibration_digest=build.calibration_digest,
            engine_params=dict(engine.get("params") or {}),
            spatial_shape=tuple(spec.spatial_shape),  # type: ignore[arg-type]
            spatial_planes=[name for name, _ in spec.spatial_layout],
            static_planes=[i for i, (_, static) in enumerate(spec.spatial_layout) if static],
            vector_size=int(spec.vector_size),
            n_actions=int(spec.n_actions),
            card_id_planes=int(card_ids.shape[0]) if card_ids is not None else 0,
            env_config={k: v for k, v in env_config.items() if k != "truncation_cond"},
        )

    @classmethod
    def of_config(cls, config: RunConfig) -> ShardContext:
        """The context of a run's own environment, built once and closed."""
        from ..identity import action_digest_of
        from ..rollout.envspec import build_env, read_env_spec

        vec = build_env(config.env, tuple(config.extra_component_modules))
        try:
            spec = read_env_spec(vec, config.env, frame_stack=1, seed=0)
            env = vec.envs[0]
            return cls.from_parts(
                spec, env.config(), env.engine.cards(), action_digest_of(config.env, spec)
            )
        finally:
            vec.close()


class PartInfo(msgspec.Struct, frozen=True):
    name: str
    sha256: str
    rows: int
    validation_rows: int


class ShardManifest(msgspec.Struct, kw_only=True):
    format_version: int = SHARD_FORMAT_VERSION
    engine_key: str
    context: ShardContext
    packages: dict[str, str]
    producer: dict[str, Any] = {}
    #: Bit (as a decimal string) to name, for every flag a row may carry.
    flag_names: dict[str, str] = {}
    parts: list[PartInfo] = []
    rows: int = 0
    validation_rows: int = 0
    flag_counts: dict[str, int] = {}


def _packages() -> dict[str, str]:
    from ..identity import royalegym_provenance
    from ..version import __version__, git_describe

    gym_version, gym_git = royalegym_provenance()
    return {
        "royalelearn": f"{__version__} ({git_describe()})",
        "royalegym": f"{gym_version} ({gym_git})",
    }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


class ShardWriter:
    """Rows in, part files and a manifest out, under ``<root>/<engine key>/``.

    A directory that already holds files is refused: two writers into one key are one dataset
    nobody wrote. The manifest is written last, by ``close``, so a directory without one is a
    write that did not finish, and the reader refuses it.
    """

    def __init__(
        self,
        root: str | Path,
        context: ShardContext,
        *,
        producer: Mapping[str, Any] | None = None,
        flag_names: Mapping[int, str] | None = None,
        rows_per_part: int = 4096,
    ) -> None:
        names = dict(DRIVER_FLAGS)
        for bit, name in (flag_names or {}).items():
            if bit in DRIVER_FLAGS:
                raise PreflightError(f"flag bit {bit} is the driver's own ({DRIVER_FLAGS[bit]})")
            if bit < 1 << 8 or bit & (bit - 1):
                raise PreflightError(f"a producer's flag is one bit from 1 << 8 up, not {bit}")
            names[int(bit)] = str(name)
        self.context = context
        self.directory = Path(root) / context.engine_key
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(
                f"{self.directory} already holds files; a shard directory is written once"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows_per_part = max(1, int(rows_per_part))
        self.manifest = ShardManifest(
            engine_key=context.engine_key,
            context=context,
            packages=_packages(),
            producer=dict(producer or {}),
            flag_names={str(bit): name for bit, name in sorted(names.items())},
        )
        self._pending: dict[str, list[Any]] = {}
        self._flags = dict(names)
        self._closed = False

    def add(
        self,
        obs: Mapping[str, np.ndarray],
        *,
        action: int,
        weight: float,
        group: int,
        seat: int,
        tick: int,
        flags: int = 0,
        reward: float = 0.0,
    ) -> None:
        """One row. The observation is the environment's own, unquantised."""
        ctx = self.context
        spatial = np.asarray(obs["spatial"], dtype=np.float32)
        if spatial.shape != ctx.spatial_shape:
            raise ValueError(f"spatial {spatial.shape}, the context says {ctx.spatial_shape}")
        pending = self._pending
        pending.setdefault("spatial", []).append(self._exact(spatial, tick))
        vector = np.asarray(obs["vector"], dtype=np.float32).reshape(-1)
        if vector.size != ctx.vector_size:
            raise ValueError(f"vector of {vector.size}, the context says {ctx.vector_size}")
        pending.setdefault("vector", []).append(vector.astype(np.float16))
        mask = np.asarray(obs["action_mask"]).astype(bool).reshape(-1)
        if mask.size != ctx.n_actions:
            raise ValueError(f"mask of {mask.size}, the context says {ctx.n_actions}")
        if not mask[int(action)]:
            raise ValueError(f"action {action} is not legal under the row's own mask")
        pending.setdefault("mask", []).append(np.packbits(mask, bitorder="little"))
        if ctx.card_id_planes:
            ids = np.asarray(obs["card_ids"])
            if ids.min() < 0 or ids.max() > 255:
                raise ValueError("a card id does not fit a byte")
            pending.setdefault("card_ids", []).append(ids.astype(np.uint8))
        unknown = int(flags) & ~sum(self._flags)
        if unknown:
            raise ValueError(f"flag bits {unknown:#x} are not named in flag_names")
        for column, value in (
            ("action", np.int16(action)),
            ("weight", np.float32(weight)),
            ("flags", np.uint32(flags)),
            ("group", np.uint64(group)),
            ("seat", np.uint8(seat)),
            ("tick", np.uint32(tick)),
            ("reward", np.float32(reward)),
        ):
            pending.setdefault(column, []).append(value)
        if len(pending["action"]) >= self.rows_per_part:
            self._flush()

    def _exact(self, spatial: np.ndarray, tick: int) -> np.ndarray:
        """``uint16`` value x 1000, refused wherever that does not give the value back."""
        quantum = np.rint(spatial.astype(np.float64) * SPATIAL_SCALE)
        stored = np.clip(quantum, 0, _UINT16_MAX).astype(np.uint16)
        back = stored.astype(np.float32) / np.float32(SPATIAL_SCALE)
        wrong = (quantum < 0) | (quantum > _UINT16_MAX) | (back != spatial)
        if wrong.any():
            plane, y, x = (int(v) for v in np.argwhere(wrong)[0])
            name = self.context.spatial_planes[plane]
            raise PreflightError(
                f"spatial plane {plane} ({name!r}) holds {float(spatial[plane, y, x])!r} at "
                f"tile ({x}, {y}), tick {tick}, which uint16 value x {SPATIAL_SCALE} does not "
                f"store exactly ({int(wrong.sum())} cells in this row). A shard stores rows "
                "exactly and quantises them only when a run reads them"
            )
        return stored

    def _flush(self) -> None:
        pending = self._pending
        if not pending.get("action"):
            return
        index = len(self.manifest.parts)
        name = f"part-{index:05d}.npz"
        path = self.directory / name
        arrays = {column: np.stack(values) for column, values in pending.items()}
        np.savez_compressed(path, **arrays)
        rows = int(arrays["action"].shape[0])
        held = int(is_validation(arrays["group"]).sum())
        self.manifest.parts.append(
            PartInfo(
                name=name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=rows,
                validation_rows=held,
            )
        )
        self.manifest.rows += rows
        self.manifest.validation_rows += held
        for bit, flag in self._flags.items():
            count = int(((arrays["flags"] & np.uint32(bit)) != 0).sum())
            self.manifest.flag_counts[flag] = self.manifest.flag_counts.get(flag, 0) + count
        self._pending = {}

    def close(self) -> Path:
        """Flush the last part and write the manifest. Returns the directory."""
        if not self._closed:
            self._flush()
            (self.directory / MANIFEST_NAME).write_bytes(
                msgspec.json.format(msgspec.json.encode(self.manifest), indent=1)
            )
            self._closed = True
        return self.directory

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, kind: object, *_rest: object) -> None:
        # A write that raised is left without a manifest, which the reader refuses.
        if kind is None:
            self.close()


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


class ShardBatch(NamedTuple):
    """One batch, decoded through the reading run's codec."""

    obs: ObsBatch
    actions: Tensor
    weights: Tensor
    flags: Tensor
    groups: np.ndarray


#: Fields a reader and the rows must agree on, whatever the engine.
_MUST_MATCH = (
    "catalogue_sha256",
    "card_names",
    "obs_digest",
    "action_digest",
    "spatial_shape",
    "vector_size",
    "n_actions",
    "card_id_planes",
)


class ShardReader:
    """A shard directory, checked against the reading run's own context."""

    def __init__(
        self,
        directory: str | Path,
        context: ShardContext,
        *,
        allow_engine_mismatch: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        path = self.directory / MANIFEST_NAME
        if not path.is_file():
            raise PreflightError(
                f"{self.directory} has no {MANIFEST_NAME}: a write that did not finish, or not a "
                "shard directory"
            )
        blob = path.read_bytes()
        self.manifest = msgspec.json.decode(blob, type=ShardManifest)
        if self.manifest.format_version > SHARD_FORMAT_VERSION:
            raise PreflightError(
                f"{path} is shard format {self.manifest.format_version}; this build reads "
                f"{SHARD_FORMAT_VERSION}"
            )
        stored = self.manifest.context
        differences = {
            name: (getattr(stored, name), getattr(context, name))
            for name in _MUST_MATCH
            if getattr(stored, name) != getattr(context, name)
        }
        if differences:
            named = "; ".join(
                f"{name}: rows {a!r}, run {b!r}" for name, (a, b) in differences.items()
            )
            raise PreflightError(
                f"{self.directory} holds rows of another environment. {named}"[:2000]
            )
        mismatch = None
        if stored.engine_key != context.engine_key:
            if not allow_engine_mismatch:
                changed = [
                    name
                    for name in ("binary_sha256", "build_digest", "calibration_digest")
                    if getattr(stored, name) != getattr(context, name)
                ]
                if stored.engine_params != context.engine_params:
                    changed.append("engine_params")
                raise PreflightError(
                    f"{self.directory} was replayed on engine {stored.engine_key} and this run is "
                    f"on {context.engine_key} (differing: {', '.join(changed) or 'engine key'}). "
                    "Rows replayed on another engine are another dataset. Pass "
                    "allow_engine_mismatch with the reason to read them anyway; the reason is "
                    "recorded with anything trained from them"
                )
            mismatch = str(allow_engine_mismatch)
        #: What anything trained from these rows records about them.
        self.provenance: dict[str, Any] = {
            "directory": str(self.directory),
            "manifest_sha256": hashlib.sha256(blob).hexdigest(),
            "engine_key": stored.engine_key,
            "engine_mismatch": mismatch,
            "rows": self.manifest.rows,
            "validation_rows": self.manifest.validation_rows,
        }

    # -- one part --------------------------------------------------------------

    def part(self, index: int) -> dict[str, np.ndarray]:
        """One part's columns, its file checked against the manifest first."""
        info = self.manifest.parts[index]
        path = self.directory / info.name
        blob = path.read_bytes()
        found = hashlib.sha256(blob).hexdigest()
        if found != info.sha256:
            raise PreflightError(
                f"{path} has sha256 {found} and the manifest says {info.sha256}: the part changed "
                "after the shard was written"
            )
        import io

        with np.load(io.BytesIO(blob)) as arrays:
            return {name: arrays[name] for name in arrays.files}

    def observations(self, columns: Mapping[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
        """The environment's observations back, exactly: spatial from its fixed point, the
        vector as stored, the mask unpacked."""
        ctx = self.manifest.context
        spatial = columns["spatial"].astype(np.float32) / np.float32(SPATIAL_SCALE)
        vector = columns["vector"].astype(np.float32)
        mask = np.unpackbits(columns["mask"], axis=-1, bitorder="little")[:, : ctx.n_actions]
        out: list[dict[str, np.ndarray]] = []
        for row in range(spatial.shape[0]):
            obs = {"spatial": spatial[row], "vector": vector[row], "action_mask": mask[row]}
            if ctx.card_id_planes:
                obs["card_ids"] = columns["card_ids"][row]
            out.append(obs)
        return out

    def packed(
        self, index: int, codec: RowCodec, *, split: str = "all"
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """One part as the reading run's codec packs it, and the scalar columns beside it.

        The static planes are not in a packed row: the codec supplies the run's own on decode.
        So they are checked here against the run's, and a part whose arena differs is refused
        rather than silently shown the run's.
        """
        columns = self.part(index)
        keep = _split_rows(columns["group"], split)
        observations = [
            obs for obs, kept in zip(self.observations(columns), keep, strict=True) if kept
        ]
        statics = codec.statics.cpu().numpy()
        for obs in observations[:1]:
            own = np.asarray(codec.codec.static_planes(obs), dtype=np.float32)
            if own.shape != statics.shape or not np.array_equal(own, statics):
                raise PreflightError(
                    f"{self.directory / self.manifest.parts[index].name}: the rows' static planes "
                    "differ from this run's, so the codec would show them another arena"
                )
        scalars = {
            name: columns[name][keep]
            for name in ("action", "weight", "flags", "group", "seat", "tick", "reward")
        }
        return codec.pack(observations), scalars

    # -- batches ---------------------------------------------------------------

    def batches(
        self,
        codec: RowCodec,
        *,
        split: str,
        batch_rows: int,
        seed: int,
        epoch: int = 0,
        window_rows: int = 8192,
        weight_of: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
    ) -> Iterator[ShardBatch]:
        """Every row of ``split`` once, in batches, shuffled within a window of parts.

        The order is a function of ``seed`` and ``epoch`` alone: the parts are taken in a seeded
        order and rows are permuted within windows of ``window_rows``, so memory is a window and
        not a dataset. ``weight_of`` may rescale each row's weight from its columns, for a pass
        that focuses on some rows.
        """
        import torch

        rng = np.random.default_rng([int(seed), int(epoch)])
        order = rng.permutation(len(self.manifest.parts))
        rows: list[np.ndarray] = []
        scalars: list[dict[str, np.ndarray]] = []
        held = 0
        device = codec.device

        def emit(block: np.ndarray, columns: dict[str, np.ndarray]) -> Iterator[ShardBatch]:
            permutation = rng.permutation(block.shape[0])
            block = block[permutation]
            columns = {name: value[permutation] for name, value in columns.items()}
            weights = columns["weight"].astype(np.float32)
            if weight_of is not None:
                weights = weights * np.asarray(weight_of(columns), dtype=np.float32)
            for start in range(0, block.shape[0], batch_rows):
                stop = start + batch_rows
                yield ShardBatch(
                    obs=codec.decode(block[start:stop]),
                    actions=torch.from_numpy(columns["action"][start:stop].astype(np.int64)).to(
                        device
                    ),
                    weights=torch.from_numpy(weights[start:stop]).to(device),
                    flags=torch.from_numpy(columns["flags"][start:stop].astype(np.int64)).to(
                        device
                    ),
                    groups=columns["group"][start:stop],
                )

        for index in order:
            packed, columns = self.packed(int(index), codec, split=split)
            rows.append(packed)
            scalars.append(columns)
            held += packed.shape[0]
            if held >= window_rows:
                block = np.concatenate(rows)
                merged = {k: np.concatenate([c[k] for c in scalars]) for k in scalars[0]}
                yield from emit(block, merged)
                rows, scalars, held = [], [], 0
        if held:
            block = np.concatenate(rows)
            merged = {k: np.concatenate([c[k] for c in scalars]) for k in scalars[0]}
            yield from emit(block, merged)


def _split_rows(groups: np.ndarray, split: str) -> np.ndarray:
    if split == "all":
        return np.ones(groups.shape[0], dtype=bool)
    held = is_validation(groups)
    if split == "validation":
        return held
    if split == "train":
        return ~held
    raise ValueError(f"split {split!r} is not 'train', 'validation' or 'all'")
