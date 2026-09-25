"""How one observation becomes a row of bytes, and how a batch of rows becomes tensors again.

The rule is written here; the table it produces is not. Which storage each spatial plane gets is
decided at preflight from the observation space's declared bounds and a sample of real
observations, so a plane added, removed or rescaled upstream reaches the right storage without a
line changing in this file. ``codec_version`` is the version of the RULE. The ``CodecTable`` is
data: it is printed at preflight, hashed into the run identity, sent to the workers and written
into every snapshot, and two runs whose tables differ are not comparable.

The rule, per key:

``spatial``, per plane
    A plane ``ObsBuilder.spatial_layout()`` declares static is not stored at all; it is held once
    per seat and scattered back in at unpack. A plane whose declared bounds fit an unsigned byte
    and whose sampled values are whole numbers is stored as ``uint8`` and is exact. Anything else
    is stored as ``float16``.
``mask_planes``
    Never stored. RoyaleGym builds it as ``action_mask[1:]`` reshaped, so the learner reshapes
    the stored mask at unpack; storing the planes would spend a sixth of the row on a reshape.
``vector``
    ``float16``. The builder bounds the whole vector in [0, 1], so the granularity is far below
    anything the policy can act on.
``action_mask``
    Bit-packed, least significant bit first, and exact.
``card_ids`` (only when the builder was asked for card identity)
    ``uint8``, exact, in a region of its own after the mask. It is never a plane of ``spatial``:
    a card id stored as a scaled half and read back as 6.997 would be embedded as card 6. Absent,
    the row is byte-for-byte what it always was.

Any other key is REFUSED at bind. The codec used to take the keys it knew and ignore the rest,
which would have dropped card identity on the floor the day the builder started sending it.

Existence is decided from the declaration and storage from the sample, and the two must not be
confused. The tower planes are constant across any sample in which no tower falls, so a rule that
promoted "unchanged on a thousand states" to "static" would freeze a tower at full health for the
rest of a run and the policy would never see one die.

``torch`` is imported inside ``unpack_to_device`` and nowhere else. Packing happens in the rollout
workers, which are numpy-only processes; unpacking happens on the learner's device.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ..api.buffer import MIN_TABLE_STATES, CodecTable, ObsCodec
from ..errors import PreflightError

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor

    from ..api.policy import ObsBatch
    from ..api.rollout import EnvSpec

__all__ = [
    "CODEC_VERSION",
    "MIN_TABLE_STATES",
    "STORAGE",
    "STORAGE_DERIVED",
    "STORAGE_FLOAT16",
    "STORAGE_STATIC",
    "STORAGE_UINT8",
    "RowLayout",
    "SpatialObsCodec",
]

#: The version of the RULE above. The table it produces travels separately.
CODEC_VERSION = 1

#: A plane stored as a whole number in one byte, exactly, with the divisor the table records.
STORAGE_UINT8 = "uint8"
#: A plane stored as half precision.
STORAGE_FLOAT16 = "float16"
#: A plane the layout declares static: held once per seat, never per row.
STORAGE_STATIC = "static"
#: A plane the learner can reconstruct from something else in the row. Nothing on the shipped
#: observation is one; the mask planes, which are derived, are not spatial planes.
STORAGE_DERIVED = "derived"

STORAGE: tuple[str, ...] = (STORAGE_UINT8, STORAGE_FLOAT16, STORAGE_STATIC, STORAGE_DERIVED)

#: The largest value one byte holds. The range of the storage type, not a fact about the arena.
_UINT8_MAX = 255
#: The key card identity arrives under, and every key this codec stores or derives.
CARD_IDS = "card_ids"
KNOWN_KEYS: frozenset[str] = frozenset(
    {"spatial", "vector", "action_mask", "mask_planes", CARD_IDS}
)
_BITS_PER_BYTE = 8

_F16 = np.dtype("<f2")


def _half(raw: Tensor) -> Tensor:
    """Reinterpret a run of bytes as half precision.

    A row is an odd number of bytes whenever the packed mask is, so a slice of one can begin on an
    odd byte or carry an odd row stride, neither of which is an address a half can be read from.
    The copy that fixes that happens only where it is needed: a gathered batch is sliced across
    its rows, so the contiguous call has already moved those bytes into storage of their own.
    """
    import torch

    raw = raw.contiguous()
    size = _F16.itemsize
    if raw.storage_offset() % size or any(stride % size for stride in raw.stride()[:-1]):
        raw = raw.clone(memory_format=torch.contiguous_format)
    return raw.view(torch.float16)


def _count_states(spec: EnvSpec, sample: Sequence[dict[str, np.ndarray]]) -> int:
    """How many observations a sample holds, counting a batched entry as the rows it carries.

    The shape of one state comes from the spec, so an entry that is one observation and an
    entry that is a batch of them are counted the same way and neither has to be declared.
    """
    planes, tiles_y, tiles_x = spec.spatial_shape
    per_state = planes * tiles_y * tiles_x
    return sum(int(np.asarray(entry["spatial"]).size // per_state) for entry in sample)


def _align2(offset: int) -> int:
    """Round up to the next even offset.

    ``float16`` is read back as a tensor rather than byte by byte, so each half-precision region
    starts on its own element boundary. With an even number of tiles in a plane the rounding never
    adds a byte, which is why the row size is exactly the arithmetic in ``docs/harness-spec.md``
    section 2.1 on both catalogues.
    """
    return offset + (offset & 1)


class RowLayout(NamedTuple):
    """Where each part of a packed row begins and ends, in bytes from the row's start.

    Every number is computed from the ``EnvSpec`` and the ``CodecTable``; none is written down.
    Both sides of the process boundary compute it the same way from the same two values, so a
    disagreement about a row is a refused attach rather than two processes reading different
    bytes.
    """

    u8_planes: tuple[int, ...]
    f16_planes: tuple[int, ...]
    static_planes: tuple[int, ...]
    u8_divisor: tuple[float, ...]
    cells: int
    u8_start: int
    u8_stop: int
    f16_start: int
    f16_stop: int
    vector_start: int
    vector_stop: int
    mask_start: int
    mask_stop: int
    row_bytes: int
    mask_bytes: int
    n_actions: int
    planes: int
    tiles: tuple[int, int]
    vector_size: int
    hand_size: int
    frame_stack: int
    #: The card-identity region: ``ids_planes`` bytes per cell, after the mask. Zero planes and
    #: an empty span when the observation carries none.
    ids_planes: int = 0
    ids_start: int = 0
    ids_stop: int = 0


class SpatialObsCodec(ObsCodec):
    """The shipped codec: quantise a dict observation into one row, and a batch of rows back.

    A codec is bound to one ``EnvSpec`` and one ``CodecTable`` before it packs or unpacks
    anything. The learner computes the table at preflight and hands it to the workers as data,
    which is what makes a worker in another language a drop-in: it needs this file's rule only to
    the extent of reproducing the offsets, and those follow from the table it is given.
    """

    def __init__(self, table: CodecTable | None = None) -> None:
        self._table = table
        self._spec: EnvSpec | None = None
        self._layout: RowLayout | None = None
        self._scratch: np.ndarray | None = None
        self._clipped = 0
        self._shift_cache: dict[Any, Any] = {}

    # -- the table ---------------------------------------------------------

    def table(
        self,
        spec: EnvSpec,
        sample: Sequence[dict[str, np.ndarray]],
        *,
        min_states: int = MIN_TABLE_STATES,
    ) -> CodecTable:
        """Decide every plane's storage, bind the result, and return it.

        ``sample`` is real observations, either one per entry or a batch per entry; a plane is
        admitted to ``uint8`` only if every sampled value of it is a whole number inside the byte
        range the space declares for that plane. Too small a sample is refused rather than
        defaulted, because a default here is a silent decision about the size of every row in
        the run.
        """
        states = _count_states(spec, sample)
        if states < min_states:
            raise PreflightError(
                f"the codec table is decided from real observations and this sample holds "
                f"{states} of them against a minimum of {min_states}. A plane that is only "
                f"fractional once units are on the board would be admitted to uint8 by a "
                f"sample that never reached one, and rounded for the rest of the run; step "
                f"the environment under real play before asking for a table."
            )
        planes = np.arange(spec.n_planes)
        integral = self._integral_planes(spec, sample)
        declared = spec.obs_space["spatial"]
        rows: list[tuple[str, str, float]] = []
        for index in planes:
            name, static = spec.spatial_layout[index]
            if static:
                rows.append((name, STORAGE_STATIC, 1.0))
                continue
            low = declared.low[index]
            high = declared.high[index]
            fits_a_byte = low >= 0.0 and high <= _UINT8_MAX
            if fits_a_byte and integral[index]:
                # Stored as the value itself, so the divisor that takes the byte back to what the
                # environment produced is one. The declared bound is what admitted the plane.
                rows.append((name, STORAGE_UINT8, 1.0))
            else:
                rows.append((name, STORAGE_FLOAT16, 1.0))
        table = CodecTable(
            plane=tuple(rows),
            vector=STORAGE_FLOAT16,
            mask="bitpack",
            ids=STORAGE_UINT8 if CARD_IDS in spec.obs_space else None,
        )
        self.bind(spec, table)
        return table

    @staticmethod
    def _integral_planes(
        spec: EnvSpec, sample: Sequence[dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Per plane: whether every sampled value of it is a whole number."""
        integral = np.ones(spec.n_planes, dtype=bool)
        for observation in sample:
            spatial = np.asarray(observation["spatial"])
            cells = spec.spatial_shape[1] * spec.spatial_shape[2]
            flat = spatial.reshape(-1, spec.n_planes, cells)
            whole = np.all(flat == np.rint(flat), axis=(0, 2))
            integral &= whole
        return integral

    def bind(self, spec: EnvSpec, table: CodecTable | None = None) -> CodecTable:
        """Bind this codec to an environment and a table, and compute the row's offsets.

        A worker is handed the table over the boundary and binds it here rather than deciding one
        of its own: two processes deciding the same table independently is two answers.
        """
        bound = table if table is not None else self._table
        if bound is None:
            raise PreflightError(
                "this codec has no table; compute one with table(spec, sample) at preflight or "
                "construct the codec with the table the learner decided"
            )
        unknown = sorted(set(spec.obs_space) - KNOWN_KEYS)
        if unknown:
            raise PreflightError(
                f"the observation carries {', '.join(unknown)}, which this codec does not store. "
                "It would have been dropped from every row and never reached the network; teach "
                "the codec the key, or build the observation without it."
            )
        has_ids = CARD_IDS in spec.obs_space
        if has_ids != (bound.ids is not None):
            raise PreflightError(
                f"the observation {'carries' if has_ids else 'has no'} {CARD_IDS} and the codec "
                f"table says {bound.ids!r}; a worker and the learner handed these would disagree "
                "about every row"
            )
        if has_ids:
            if bound.ids != STORAGE_UINT8:
                raise PreflightError(
                    f"{CARD_IDS} is stored as {bound.ids!r}; only 'uint8' is exact"
                )
            vocabulary = int(np.max(np.asarray(spec.obs_space[CARD_IDS].high))) + 1
            if vocabulary > _UINT8_MAX + 1:
                raise PreflightError(
                    f"{CARD_IDS} has a vocabulary of {vocabulary}, which does not fit one byte. "
                    "A wider id would wrap silently into another card; widen the region first."
                )
        if len(bound.plane) != spec.n_planes:
            raise PreflightError(
                f"the codec table describes {len(bound.plane)} planes and the observation space "
                f"has {spec.n_planes}"
            )
        for index, ((name, storage, _), (declared_name, _)) in enumerate(
            zip(bound.plane, spec.spatial_layout, strict=True)
        ):
            if name != declared_name:
                raise PreflightError(
                    f"the codec table names plane {index} {name!r} and the observation's layout "
                    f"names it {declared_name!r}"
                )
            if storage not in STORAGE:
                raise PreflightError(
                    f"plane {name!r} is stored as {storage!r}, which is not one of "
                    f"{', '.join(STORAGE)}"
                )
        self._table = bound
        self._spec = spec
        self._layout = self._compute_layout(spec, bound)
        self._scratch = np.empty(
            (len(self._layout.u8_planes), *spec.spatial_shape[1:]), dtype=np.float32
        )
        return bound

    @staticmethod
    def _compute_layout(spec: EnvSpec, table: CodecTable) -> RowLayout:
        by_storage = {name: [] for name in STORAGE}
        for index, (_, storage, _) in enumerate(table.plane):
            by_storage[storage].append(index)
        u8 = tuple(by_storage[STORAGE_UINT8])
        f16 = tuple(by_storage[STORAGE_FLOAT16])
        static = tuple(by_storage[STORAGE_STATIC])
        cells = spec.spatial_shape[1] * spec.spatial_shape[2]
        u8_start = 0
        u8_stop = u8_start + len(u8) * cells
        f16_start = _align2(u8_stop)
        f16_stop = f16_start + len(f16) * cells * _F16.itemsize
        vector_start = _align2(f16_stop)
        vector_stop = vector_start + spec.vector_size * _F16.itemsize
        mask_bytes = math.ceil(spec.n_actions / _BITS_PER_BYTE)
        mask_start = vector_stop
        mask_stop = mask_start + mask_bytes
        ids_planes = spec.obs_space[CARD_IDS].shape[0] if CARD_IDS in spec.obs_space else 0
        ids_start = mask_stop
        ids_stop = ids_start + ids_planes * cells
        return RowLayout(
            u8_planes=u8,
            f16_planes=f16,
            static_planes=static,
            u8_divisor=tuple(float(table.plane[i][2]) for i in u8),
            cells=cells,
            u8_start=u8_start,
            u8_stop=u8_stop,
            f16_start=f16_start,
            f16_stop=f16_stop,
            vector_start=vector_start,
            vector_stop=vector_stop,
            mask_start=mask_start,
            mask_stop=mask_stop,
            row_bytes=ids_stop,
            mask_bytes=mask_bytes,
            n_actions=spec.n_actions,
            planes=spec.n_planes,
            tiles=spec.spatial_shape[1:],
            vector_size=spec.vector_size,
            hand_size=spec.hand_size,
            frame_stack=spec.frame_stack,
            ids_planes=ids_planes,
            ids_start=ids_start,
            ids_stop=ids_stop,
        )

    @property
    def codec_table(self) -> CodecTable:
        """The table this codec is bound to."""
        if self._table is None:
            raise PreflightError("this codec has no table; call table() or bind() first")
        return self._table

    @property
    def layout(self) -> RowLayout:
        """Where each part of a row lives, for anything that has to agree about a row."""
        if self._layout is None:
            raise PreflightError("this codec is not bound to an environment; call bind() first")
        return self._layout

    @property
    def codec_version(self) -> int:
        return CODEC_VERSION

    @property
    def clipped(self) -> int:
        """Values clipped on the way into a ``uint8`` plane since the last reset.

        Reported as ``health/obs_codec_clipped``. Non-zero is a bug report and not a tolerance:
        the plane was admitted to a byte on a declared bound, so a clip means the bound was wrong.
        """
        return self._clipped

    def reset_clipped(self) -> int:
        """Read the counter and zero it, which is how a per-round total is taken."""
        count, self._clipped = self._clipped, 0
        return count

    def row_bytes(self, spec: EnvSpec) -> int:
        """The size of one packed row, for the environment this codec is bound to."""
        layout = self.layout
        if self._spec is not None and (
            spec.spatial_shape != self._spec.spatial_shape
            or spec.vector_size != self._spec.vector_size
            or spec.n_actions != self._spec.n_actions
        ):
            raise PreflightError(
                "this codec is bound to an observation of "
                f"{self._spec.spatial_shape} / {self._spec.vector_size} / "
                f"{self._spec.n_actions} and was asked for the row size of "
                f"{spec.spatial_shape} / {spec.vector_size} / {spec.n_actions}"
            )
        return layout.row_bytes

    # -- packing -----------------------------------------------------------

    def pack(self, obs: dict[str, np.ndarray], out: memoryview, row: int) -> None:
        """Write one observation into row ``row`` of the rectangle ``out`` covers.

        ``out`` is the observation area of the shared segment and the row is written in place, so
        an observation is copied once, by the process that produced it, into the memory the
        learner will gather it from.
        """
        layout = self.layout
        destination = np.frombuffer(out, dtype=np.uint8)
        start = row * layout.row_bytes
        if start < 0 or start + layout.row_bytes > destination.size:
            raise IndexError(
                f"row {row} of {layout.row_bytes} B does not fit the {destination.size} B this "
                "view covers"
            )
        target = destination[start : start + layout.row_bytes]
        spatial = np.asarray(obs["spatial"])

        if layout.u8_planes:
            scratch = self._scratch
            assert scratch is not None  # bind() allocates it beside the layout
            block = spatial[list(layout.u8_planes)]
            over = int(np.count_nonzero((block > _UINT8_MAX) | (block < 0)))
            if over:
                self._clipped += over
            np.clip(block, 0, _UINT8_MAX, out=scratch)
            np.rint(scratch, out=scratch)
            target[layout.u8_start : layout.u8_stop] = scratch.reshape(-1).astype(np.uint8)

        if layout.f16_planes:
            half = spatial[list(layout.f16_planes)].astype(_F16).reshape(-1)
            target[layout.f16_start : layout.f16_stop] = half.view(np.uint8)

        vector = np.asarray(obs["vector"], dtype=np.float32).astype(_F16).reshape(-1)
        target[layout.vector_start : layout.vector_stop] = vector.view(np.uint8)

        mask = np.asarray(obs["action_mask"]).astype(bool, copy=False).reshape(-1)
        target[layout.mask_start : layout.mask_stop] = np.packbits(mask, bitorder="little")

        if layout.ids_planes:
            ids = np.asarray(obs[CARD_IDS])
            if ids.dtype != np.uint8:
                # bind() proved the vocabulary fits a byte; a value outside it is a bug report.
                over = int(np.count_nonzero((ids < 0) | (ids > _UINT8_MAX)))
                if over:
                    self._clipped += over
                ids = np.clip(ids, 0, _UINT8_MAX).astype(np.uint8)
            target[layout.ids_start : layout.ids_stop] = ids.reshape(-1)

    def static_planes(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """The declared-static planes of one observation, as ``float32``.

        Held once per seat for the whole run. Declared, never inferred: a plane that happens to be
        constant on a sample is still stored per row.
        """
        layout = self.layout
        spatial = np.asarray(obs["spatial"], dtype=np.float32)
        return np.ascontiguousarray(spatial[list(layout.static_planes)])

    # -- unpacking ---------------------------------------------------------

    def unpack_to_device(self, raw: Tensor, statics: Tensor, out: ObsBatch) -> None:
        """Turn a gathered batch of rows into the tensors the networks read.

        ``raw`` is ``(B, k, row_bytes)`` of packed rows, frame 0 being the cell itself and frame
        ``j`` the row ``j`` cycles before it; ``statics`` is ``(static planes, H, W)``. A frame
        whose bytes are all zero decodes to a frame of zeros, statics included: that is how the
        buffer says "no history here", and it is unambiguous because a real row always has the
        no-op bit of its mask set.

        Everything is written into ``out``'s tensors, which the caller allocates once and reuses.
        """
        import torch

        layout = self.layout
        if raw.ndim == 2:
            raw = raw.unsqueeze(1)
        batch, frames, row_bytes = raw.shape
        if row_bytes != layout.row_bytes:
            raise ValueError(
                f"a row of this observation is {layout.row_bytes} B and the batch carries "
                f"{row_bytes} B"
            )
        tiles_y, tiles_x = layout.tiles
        spatial = out.spatial.view(batch, frames, layout.planes, tiles_y, tiles_x)

        if layout.u8_planes:
            block = (
                raw[:, :, layout.u8_start : layout.u8_stop]
                .reshape(batch, frames, len(layout.u8_planes), tiles_y, tiles_x)
                .to(spatial.dtype)
            )
            divisor = torch.as_tensor(
                layout.u8_divisor, dtype=spatial.dtype, device=spatial.device
            )
            if not bool(torch.all(divisor == 1)):
                block = block / divisor.view(1, 1, -1, 1, 1)
            spatial[:, :, list(layout.u8_planes)] = block

        if layout.f16_planes:
            half = _half(raw[:, :, layout.f16_start : layout.f16_stop]).reshape(
                batch, frames, len(layout.f16_planes), tiles_y, tiles_x
            )
            spatial[:, :, list(layout.f16_planes)] = half.to(spatial.dtype)

        if layout.static_planes:
            spatial[:, :, list(layout.static_planes)] = statics.to(spatial.dtype).view(
                1, 1, len(layout.static_planes), tiles_y, tiles_x
            )

        bits = self._mask_bits(raw, layout)
        out.mask.copy_(bits[:, 0].to(out.mask.dtype))
        out.mask_planes.view(
            batch, frames, layout.hand_size, tiles_y, tiles_x
        ).copy_(
            bits[:, :, 1:].reshape(batch, frames, layout.hand_size, tiles_y, tiles_x)
        )

        out.vector.copy_(_half(raw[:, 0, layout.vector_start : layout.vector_stop]))

        if layout.ids_planes:
            if out.card_ids is None:
                raise ValueError(
                    f"this batch was allocated without {CARD_IDS} and its rows carry them; the "
                    "planes would be read and thrown away"
                )
            out.card_ids.view(batch, frames, layout.ids_planes, tiles_y, tiles_x).copy_(
                raw[:, :, layout.ids_start : layout.ids_stop].reshape(
                    batch, frames, layout.ids_planes, tiles_y, tiles_x
                )
            )

        live = (raw != 0).any(dim=-1).view(batch, frames, 1, 1, 1)
        spatial.mul_(live.to(spatial.dtype))

    def _mask_bits(self, raw: Tensor, layout: RowLayout) -> Tensor:
        """``(B, k, n_actions)`` of the stored mask, unpacked least significant bit first."""
        import torch

        batch, frames = raw.shape[0], raw.shape[1]
        packed = raw[:, :, layout.mask_start : layout.mask_stop]
        key = (raw.device, raw.dtype)
        shifts = self._shift_cache.get(key)
        if shifts is None:
            shifts = torch.arange(_BITS_PER_BYTE, device=raw.device, dtype=raw.dtype)
            self._shift_cache[key] = shifts
        bits = torch.bitwise_right_shift(packed.unsqueeze(-1), shifts).bitwise_and_(1)
        return bits.reshape(batch, frames, -1)[:, :, : layout.n_actions]
