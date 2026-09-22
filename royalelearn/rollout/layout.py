"""The process boundary, as a byte layout.

This module is the cross-language contract. Both sides compute the same offsets from the same
``EnvSpec`` and the same row size; nothing is pickled on the hot path, nothing is
length-prefixed, and there is no stream to desynchronise. A worker written in Rust is a drop-in
the day it exists: it writes these bytes, flips the same events, and the differential test
against the Python worker is its acceptance criterion.

Two segments:

Segment A, one per run, ``royalelearn-buf-<run_id>``
    a 128-byte header and then the experience rectangle itself -- the learner owns it, the
    workers write observation rows into it, and the learner never copies an observation on the
    CPU except into its pinned staging ring.

Segment B, one per worker, ``royalelearn-ctl-<run_id>-<w>``
    the scalars and control words of every shard that worker holds, double-buffered by parity
    so that the parent can be reading the round it was handed while the child fills the next.

``LAYOUT_VERSION`` changes when any offset does. It is carried in the header, checked at attach,
and pinned by a golden record in ``tests/test_layout.py``, so that a change to it is a decision
rather than a surprise.

The ``Record`` definitions below, not any prose, are the byte order: within a record the fields
are ordered widest first, so that every scalar lands on its natural alignment and no padding is
needed to reach one. ``CONTROL`` and ``SCALARS`` therefore read in a different order from the
lists they were specified as, with the same field set and the same total size.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.rollout import EnvSpec

__all__ = [
    "ACTION_BYTES",
    "ALIGN",
    "ASSIGN_BYTES",
    "CONTROL",
    "ERROR_BYTES",
    "ERR_CRASH",
    "ERR_EXCEPTION",
    "ERR_NONE",
    "FLAG_TERMINATED",
    "FLAG_TRUNCATED",
    "FLAG_VALID",
    "HEADER",
    "LAYOUT_VERSION",
    "MAGIC",
    "PARITIES",
    "PLAN",
    "SCALARS",
    "SCALARS_DTYPE",
    "STATE_ACTIONS_READY",
    "STATE_CLOSED",
    "STATE_IDLE",
    "STATE_OBS_READY",
    "BufferHandle",
    "BufferLayout",
    "ControlHandle",
    "ControlLayout",
    "Record",
    "buffer_segment_name",
    "control_segment_name",
    "read_error",
    "write_error",
]

LAYOUT_VERSION = 1
MAGIC = b"RLRN"
#: Every region starts on a cache line. It costs a few hundred bytes a worker and keeps two
#: sides writing neighbouring records off the same line.
ALIGN = 64
#: The control and scalar regions are doubled: the parent reads one parity while the child fills
#: the other, which is what makes a round hand-off a flag flip rather than a copy.
PARITIES = 2

ERROR_BYTES = 512
ACTION_BYTES = 2
ASSIGN_BYTES = 4

#: ``control.state``: whose turn it is on this shard, in the order a round goes round.
STATE_IDLE = 0
STATE_OBS_READY = 1
STATE_ACTIONS_READY = 2
STATE_CLOSED = 3

#: ``control.err_code``. Either non-zero value becomes a ``WorkerFailure`` in the parent.
ERR_NONE = 0
ERR_EXCEPTION = 1
ERR_CRASH = 2

#: Bits of the scalar record's ``flags`` byte.
FLAG_TERMINATED = 1
FLAG_TRUNCATED = 2
FLAG_VALID = 4


class FieldSpec(NamedTuple):
    """One field of a fixed record: its name, its struct code, its offset and its size."""

    name: str
    code: str
    offset: int
    size: int


class Record:
    """A fixed-size binary record built from a field list.

    Offsets are computed from the list rather than written down, every scalar is checked to land
    on its natural alignment, and the trailing space up to ``size`` is reserved: a later field
    fits into the reserve without moving anything before it, which is the difference between
    adding a field and changing ``LAYOUT_VERSION``.
    """

    def __init__(
        self, name: str, fields: Sequence[tuple[str, str]], size: int | None = None
    ) -> None:
        self.name = name
        offset = 0
        specs: list[FieldSpec] = []
        for field_name, code in fields:
            width = struct.calcsize("<" + code)
            if code[-1] not in ("s", "x") and offset % width:
                raise ValueError(
                    f"{name}.{field_name} ({code}) at offset {offset} is not {width}-byte aligned"
                )
            specs.append(FieldSpec(field_name, code, offset, width))
            offset += width
        self.packed_size = offset
        self.size = self.packed_size if size is None else size
        if self.size < self.packed_size:
            raise ValueError(f"{name} needs {self.packed_size} B, was given {self.size}")
        self.fields: tuple[FieldSpec, ...] = tuple(specs)
        self._by_name = {f.name: f for f in specs}
        self._struct = struct.Struct("<" + "".join(f.code for f in specs))

    def __getitem__(self, field_name: str) -> FieldSpec:
        return self._by_name[field_name]

    def offset_of(self, field_name: str) -> int:
        return self._by_name[field_name].offset

    def pack_into(self, buffer: Any, base: int, values: Mapping[str, Any]) -> None:
        self._struct.pack_into(buffer, base, *(values[f.name] for f in self.fields))

    def unpack_from(self, buffer: Any, base: int = 0) -> dict[str, Any]:
        raw = self._struct.unpack_from(buffer, base)
        return {f.name: v for f, v in zip(self.fields, raw, strict=True)}

    def describe(self) -> dict[str, tuple[int, int]]:
        """``field -> (offset, size)``, for the golden record."""
        return {f.name: (f.offset, f.size) for f in self.fields}


#: Segment A's header. Everything a second process needs to compute the same offsets, so that
#: attaching to a segment checks agreement instead of assuming it.
HEADER = Record(
    "header",
    [
        ("magic", "4s"),
        ("layout_version", "I"),
        ("codec_version", "I"),
        ("cycles", "I"),
        ("n_slots", "I"),
        ("row_bytes", "I"),
        ("frame_stack", "I"),
        ("history_rows", "I"),
        ("obs_offset", "Q"),
        ("obs_bytes", "Q"),
        ("total_bytes", "Q"),
        ("run_id", "16s"),
    ],
    size=128,
)

#: One shard's control word, per parity.
CONTROL = Record(
    "control",
    [
        ("cycle", "Q"),
        ("t_env_ns", "Q"),
        ("state", "I"),
        ("n_slots", "I"),
        ("err_code", "I"),
        ("err_len", "I"),
    ],
    size=64,
)

#: One slot's scalars for one round. The order is alignment, not importance.
SCALARS = Record(
    "scalars",
    [
        ("reward", "f"),
        ("tick", "i"),
        ("episode_steps", "i"),
        ("cards_played", "h"),
        ("elixir_leak", "h"),
        ("group", "b"),
        ("flags", "B"),
        ("deploy_status", "b"),
        ("episode_end", "b"),
    ],
    size=32,
)

#: The iteration's opening table, one row per slot.
PLAN = Record(
    "plan",
    [
        ("seed", "Q"),
        ("role", "b"),
        ("group", "b"),
        ("opponent_ix", "b"),
    ],
    size=24,
)

#: The parent's per-slot assignment for the battles whose episode started on the previous round.
ASSIGN = Record(
    "assign",
    [
        ("group", "b"),
        ("opponent_ix", "b"),
        ("learner_seat", "b"),
    ],
    size=ASSIGN_BYTES,
)

#: ``SCALARS`` as a numpy dtype, so both sides read the region as one structured array rather
#: than unpacking a row at a time. The itemsize is asserted against the record below.
SCALARS_DTYPE = np.dtype(
    {
        "names": [f.name for f in SCALARS.fields],
        "formats": ["<f4", "<i4", "<i4", "<i2", "<i2", "i1", "u1", "i1", "i1"],
        "offsets": [f.offset for f in SCALARS.fields],
        "itemsize": SCALARS.size,
    }
)
if SCALARS_DTYPE.itemsize != SCALARS.size:  # pragma: no cover - a typo in the table above
    raise RuntimeError("SCALARS_DTYPE does not match the SCALARS record")


def _aligned(offset: int) -> int:
    return (offset + ALIGN - 1) // ALIGN * ALIGN


def buffer_segment_name(run_id: str) -> str:
    return f"royalelearn-buf-{run_id}"


def control_segment_name(run_id: str, worker: int) -> str:
    return f"royalelearn-ctl-{run_id}-{worker}"


def write_error(view: memoryview, text: str) -> int:
    """Put a traceback in an error region and return its length in bytes.

    Truncated on a character boundary, because a region that ends mid-sequence decodes to
    nothing useful in the one place -- a crashed worker -- where the text is all there is.
    """
    blob = text.encode("utf-8")
    if len(blob) > ERROR_BYTES:
        blob = blob[:ERROR_BYTES]
        while blob and (blob[-1] & 0xC0) == 0x80:
            blob = blob[:-1]
        if blob and blob[-1] >= 0xC0:
            blob = blob[:-1]
    view[: len(blob)] = blob
    return len(blob)


def read_error(view: memoryview, length: int) -> str:
    return bytes(view[: max(0, min(length, ERROR_BYTES))]).decode("utf-8", errors="replace")


class BufferHandle(msgspec.Struct, frozen=True):
    """What a worker is told about segment A: its name and the numbers the offsets follow from.

    The offsets themselves are not sent. Both sides compute them from these, which is what makes
    a disagreement a refusal at attach rather than two processes reading different rows.
    """

    name: str
    size: int
    run_id: str
    cycles: int
    n_slots: int
    row_bytes: int
    codec_version: int
    frame_stack: int
    layout_version: int = LAYOUT_VERSION

    def layout(self) -> BufferLayout:
        return BufferLayout(
            run_id=self.run_id,
            cycles=self.cycles,
            n_slots=self.n_slots,
            row_bytes=self.row_bytes,
            codec_version=self.codec_version,
            frame_stack=self.frame_stack,
        )


class BufferLayout:
    """Segment A: the experience rectangle.

    ``(cycles + frame_stack) x n_slots`` rows of ``row_bytes``, cycle-major, with
    ``index(t, r) = (t + frame_stack - 1) * n_slots + r`` for ``t`` in
    ``[-(frame_stack - 1), cycles]``. Cycle ``cycles`` holds observations only -- it is the
    bootstrap row -- and the rows below zero are the history carried down from the previous
    iteration, so an iteration boundary is not a discontinuity in what the policy sees.
    """

    def __init__(
        self,
        *,
        run_id: str,
        cycles: int,
        n_slots: int,
        row_bytes: int,
        codec_version: int,
        frame_stack: int = 1,
    ) -> None:
        if min(cycles, n_slots, row_bytes, frame_stack) < 1:
            raise ValueError(
                f"cycles={cycles}, n_slots={n_slots}, row_bytes={row_bytes}, "
                f"frame_stack={frame_stack}: all must be positive"
            )
        self.run_id = run_id
        self.cycles = cycles
        self.n_slots = n_slots
        self.row_bytes = row_bytes
        self.codec_version = codec_version
        self.frame_stack = frame_stack
        self.history_rows = frame_stack - 1
        self.rows = cycles + frame_stack
        self.obs_offset = _aligned(HEADER.size)
        self.obs_bytes = self.rows * n_slots * row_bytes
        self.total_bytes = self.obs_offset + self.obs_bytes

    @classmethod
    def from_spec(
        cls,
        spec: EnvSpec,
        *,
        run_id: str,
        cycles: int,
        n_slots: int,
        row_bytes: int,
        codec_version: int,
    ) -> BufferLayout:
        """The layout for one iteration of one run. ``row_bytes`` comes from the codec, which
        decides it from the same ``EnvSpec``; the frame stack comes from the spec."""
        return cls(
            run_id=run_id,
            cycles=cycles,
            n_slots=n_slots,
            row_bytes=row_bytes,
            codec_version=codec_version,
            frame_stack=spec.frame_stack,
        )

    def handle(self, name: str | None = None) -> BufferHandle:
        return BufferHandle(
            name=name or buffer_segment_name(self.run_id),
            size=self.total_bytes,
            run_id=self.run_id,
            cycles=self.cycles,
            n_slots=self.n_slots,
            row_bytes=self.row_bytes,
            codec_version=self.codec_version,
            frame_stack=self.frame_stack,
        )

    def row_index(self, cycle: int, slot: int = 0) -> int:
        """The row number of cell ``(cycle, slot)`` in the rectangle."""
        if not -self.history_rows <= cycle <= self.cycles:
            raise IndexError(
                f"cycle {cycle} outside [{-self.history_rows}, {self.cycles}] of this layout"
            )
        if not 0 <= slot < self.n_slots:
            raise IndexError(f"slot {slot} outside [0, {self.n_slots})")
        return (cycle + self.history_rows) * self.n_slots + slot

    def cell_offset(self, cycle: int, slot: int) -> int:
        """Where cell ``(cycle, slot)``'s packed observation starts in the segment."""
        return self.obs_offset + self.row_index(cycle, slot) * self.row_bytes

    def write_header(self, buffer: Any) -> None:
        if len(buffer) < self.total_bytes:
            raise ValueError(
                f"segment is {len(buffer)} B, this layout needs {self.total_bytes} B"
            )
        HEADER.pack_into(
            buffer,
            0,
            {
                "magic": MAGIC,
                "layout_version": LAYOUT_VERSION,
                "codec_version": self.codec_version,
                "cycles": self.cycles,
                "n_slots": self.n_slots,
                "row_bytes": self.row_bytes,
                "frame_stack": self.frame_stack,
                "history_rows": self.history_rows,
                "obs_offset": self.obs_offset,
                "obs_bytes": self.obs_bytes,
                "total_bytes": self.total_bytes,
                "run_id": self.run_id.encode("utf-8")[:16],
            },
        )

    def check_header(self, buffer: Any) -> None:
        """Refuse a segment that is not this layout's, at attach rather than at first write."""
        if len(buffer) < self.total_bytes:
            raise ValueError(
                f"segment is {len(buffer)} B, this layout needs {self.total_bytes} B"
            )
        header = HEADER.unpack_from(buffer, 0)
        if header["magic"] != MAGIC:
            raise ValueError(f"segment does not start with {MAGIC!r}")
        if header["layout_version"] != LAYOUT_VERSION:
            raise ValueError(
                f"segment was written at layout version {header['layout_version']}, "
                f"this build is {LAYOUT_VERSION}"
            )
        for field_name, mine in (
            ("codec_version", self.codec_version),
            ("cycles", self.cycles),
            ("n_slots", self.n_slots),
            ("row_bytes", self.row_bytes),
            ("frame_stack", self.frame_stack),
        ):
            if header[field_name] != mine:
                raise ValueError(
                    f"segment header {field_name}={header[field_name]}, this layout says {mine}"
                )

    def obs_view(self, buffer: Any) -> np.ndarray:
        """The rectangle as ``uint8[rows, row_bytes]``, sharing the segment's memory."""
        array = np.frombuffer(buffer, dtype=np.uint8, count=self.obs_bytes, offset=self.obs_offset)
        return array.reshape(self.rows, self.row_bytes)

    def describe(self) -> dict[str, Any]:
        """The golden record: every number a second implementation has to agree on."""
        return {
            "layout_version": LAYOUT_VERSION,
            "header": HEADER.describe(),
            "header_bytes": HEADER.size,
            "obs_offset": self.obs_offset,
            "obs_bytes": self.obs_bytes,
            "rows": self.rows,
            "history_rows": self.history_rows,
            "total_bytes": self.total_bytes,
        }


class ControlHandle(msgspec.Struct, frozen=True):
    """What a worker is told about its own segment B."""

    name: str
    size: int
    run_id: str
    worker: int
    shards: int
    slots_per_shard: int
    row_bytes: int
    finals_per_round: int
    layout_version: int = LAYOUT_VERSION

    def layout(self) -> ControlLayout:
        return ControlLayout(
            run_id=self.run_id,
            worker=self.worker,
            shards=self.shards,
            slots_per_shard=self.slots_per_shard,
            row_bytes=self.row_bytes,
            finals_per_round=self.finals_per_round,
        )


class ControlLayout:
    """Segment B: one worker's control words, scalars, final observations and actions.

    Per shard and parity, in this order and each aligned to a cache line: the control word, the
    error text, the scalar rows, the final observations of the rows that truncated, the actions
    the parent wrote, and the assignments for the battles that just started an episode. The
    iteration's opening table for this worker's slots sits once at the end.

    ``finals_per_round`` defaults to every slot of a shard. The bound has to hold in the worst
    case -- a step limit can truncate every battle of a shard on the same round -- and at a
    thirteen-kilobyte row that is under a megabyte a worker, which is not worth a tighter bound
    that could be exceeded.
    """

    def __init__(
        self,
        *,
        run_id: str,
        worker: int,
        shards: int,
        slots_per_shard: int,
        row_bytes: int,
        finals_per_round: int | None = None,
    ) -> None:
        if min(shards, slots_per_shard, row_bytes) < 1:
            raise ValueError(
                f"shards={shards}, slots_per_shard={slots_per_shard}, row_bytes={row_bytes}: "
                "all must be positive"
            )
        self.run_id = run_id
        self.worker = worker
        self.shards = shards
        self.slots_per_shard = slots_per_shard
        self.row_bytes = row_bytes
        self.finals_per_round = slots_per_shard if finals_per_round is None else finals_per_round
        self.n_slots = shards * slots_per_shard

        self._regions: dict[tuple[int, int, str], tuple[int, int]] = {}
        offset = 0
        for shard in range(shards):
            for parity in range(PARITIES):
                for region, size in (
                    ("control", CONTROL.size),
                    ("error", ERROR_BYTES),
                    ("scalars", slots_per_shard * SCALARS.size),
                    ("finals", self.finals_per_round * row_bytes),
                    ("actions", slots_per_shard * ACTION_BYTES),
                    ("assign", slots_per_shard * ASSIGN_BYTES),
                ):
                    offset = _aligned(offset)
                    self._regions[(shard, parity, region)] = (offset, size)
                    offset += size
        self.plan_offset = _aligned(offset)
        self.plan_bytes = self.n_slots * PLAN.size
        self.total_bytes = self.plan_offset + self.plan_bytes

    def handle(self, name: str | None = None) -> ControlHandle:
        return ControlHandle(
            name=name or control_segment_name(self.run_id, self.worker),
            size=self.total_bytes,
            run_id=self.run_id,
            worker=self.worker,
            shards=self.shards,
            slots_per_shard=self.slots_per_shard,
            row_bytes=self.row_bytes,
            finals_per_round=self.finals_per_round,
        )

    def region(self, shard: int, parity: int, name: str) -> tuple[int, int]:
        """``(offset, size)`` of one region. An unknown name is a KeyError, not a guess."""
        try:
            return self._regions[(shard, parity % PARITIES, name)]
        except KeyError:
            raise KeyError(
                f"no region {name!r} for shard {shard} of worker {self.worker}"
            ) from None

    def view(self, buffer: Any, shard: int, parity: int, name: str) -> memoryview:
        if len(buffer) < self.total_bytes:
            raise ValueError(
                f"segment is {len(buffer)} B, this layout needs {self.total_bytes} B"
            )
        offset, size = self.region(shard, parity, name)
        return memoryview(buffer)[offset : offset + size]

    def scalars(self, buffer: Any, shard: int, parity: int) -> np.ndarray:
        """One shard-round's scalar rows as a structured array over the segment's memory."""
        offset, _ = self.region(shard, parity, "scalars")
        return np.frombuffer(
            buffer, dtype=SCALARS_DTYPE, count=self.slots_per_shard, offset=offset
        )

    def actions(self, buffer: Any, shard: int, parity: int) -> np.ndarray:
        offset, _ = self.region(shard, parity, "actions")
        return np.frombuffer(buffer, dtype="<i2", count=self.slots_per_shard, offset=offset)

    def describe(self) -> dict[str, Any]:
        """The golden record: every region of every shard and parity, plus the record layouts."""
        regions = {
            f"shard{shard}/parity{parity}/{name}": value
            for (shard, parity, name), value in self._regions.items()
        }
        return {
            "layout_version": LAYOUT_VERSION,
            "control": CONTROL.describe(),
            "scalars": SCALARS.describe(),
            "plan": PLAN.describe(),
            "assign": ASSIGN.describe(),
            "regions": regions,
            "plan_offset": self.plan_offset,
            "plan_bytes": self.plan_bytes,
            "total_bytes": self.total_bytes,
        }
