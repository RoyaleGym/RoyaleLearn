"""The shared-memory byte layout: self-consistent, pinned against a golden record, and loud
about a block that is too small.

The golden records below are what makes ``LAYOUT_VERSION`` mean something. They are written for
made-up dimensions rather than any environment's, so a change to an observation's width does not
touch this file and a change to an offset cannot pass unnoticed.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalelearn.rollout import layout as L

CYCLES = 4
SLOTS = 8
ROW_BYTES = 1024
RUN_ID = "0123456789abcdef"

GOLDEN_HEADER = {
    "magic": (0, 4),
    "layout_version": (4, 4),
    "codec_version": (8, 4),
    "cycles": (12, 4),
    "n_slots": (16, 4),
    "row_bytes": (20, 4),
    "frame_stack": (24, 4),
    "history_rows": (28, 4),
    "obs_offset": (32, 8),
    "obs_bytes": (40, 8),
    "total_bytes": (48, 8),
    "run_id": (56, 16),
}

GOLDEN_CONTROL = {
    "cycle": (0, 8),
    "t_env_ns": (8, 8),
    "state": (16, 4),
    "n_slots": (20, 4),
    "err_code": (24, 4),
    "err_len": (28, 4),
}

GOLDEN_SCALARS = {
    "reward": (0, 4),
    "tick": (4, 4),
    "episode_steps": (8, 4),
    "cards_played": (12, 2),
    "elixir_leak": (14, 2),
    "group": (16, 1),
    "flags": (17, 1),
    "deploy_status": (18, 1),
    "episode_end": (19, 1),
}

GOLDEN_PLAN = {"seed": (0, 8), "role": (8, 1), "group": (9, 1), "opponent_ix": (10, 1)}
GOLDEN_ASSIGN = {"group": (0, 1), "opponent_ix": (1, 1), "learner_seat": (2, 1)}


@pytest.fixture
def buffer_layout() -> L.BufferLayout:
    return L.BufferLayout(
        run_id=RUN_ID,
        cycles=CYCLES,
        n_slots=SLOTS,
        row_bytes=ROW_BYTES,
        codec_version=1,
        frame_stack=2,
    )


@pytest.fixture
def control_layout() -> L.ControlLayout:
    return L.ControlLayout(
        run_id=RUN_ID, worker=0, shards=2, slots_per_shard=SLOTS // 2, row_bytes=ROW_BYTES
    )


def test_the_version_is_one() -> None:
    assert L.LAYOUT_VERSION == 1


def test_record_offsets_are_golden() -> None:
    assert L.HEADER.describe() == GOLDEN_HEADER
    assert L.HEADER.size == 128
    assert L.CONTROL.describe() == GOLDEN_CONTROL
    assert L.CONTROL.size == 64
    assert L.SCALARS.describe() == GOLDEN_SCALARS
    assert L.SCALARS.size == 32
    assert L.PLAN.describe() == GOLDEN_PLAN
    assert L.PLAN.size == 24
    assert L.ASSIGN.describe() == GOLDEN_ASSIGN
    assert L.ASSIGN.size == L.ASSIGN_BYTES


def test_the_buffer_layout_is_golden(buffer_layout: L.BufferLayout) -> None:
    assert buffer_layout.describe() == {
        "layout_version": 1,
        "header": GOLDEN_HEADER,
        "header_bytes": 128,
        "obs_offset": 128,
        "obs_bytes": (CYCLES + 2) * SLOTS * ROW_BYTES,
        "rows": CYCLES + 2,
        "history_rows": 1,
        "total_bytes": 128 + (CYCLES + 2) * SLOTS * ROW_BYTES,
    }


def test_the_control_layout_is_golden(control_layout: L.ControlLayout) -> None:
    described = control_layout.describe()
    assert described["regions"] == {
        "shard0/parity0/control": (0, 64),
        "shard0/parity0/error": (64, 512),
        "shard0/parity0/scalars": (576, 128),
        "shard0/parity0/finals": (704, 4096),
        "shard0/parity0/actions": (4800, 8),
        "shard0/parity0/assign": (4864, 16),
        "shard0/parity1/control": (4928, 64),
        "shard0/parity1/error": (4992, 512),
        "shard0/parity1/scalars": (5504, 128),
        "shard0/parity1/finals": (5632, 4096),
        "shard0/parity1/actions": (9728, 8),
        "shard0/parity1/assign": (9792, 16),
        "shard1/parity0/control": (9856, 64),
        "shard1/parity0/error": (9920, 512),
        "shard1/parity0/scalars": (10432, 128),
        "shard1/parity0/finals": (10560, 4096),
        "shard1/parity0/actions": (14656, 8),
        "shard1/parity0/assign": (14720, 16),
        "shard1/parity1/control": (14784, 64),
        "shard1/parity1/error": (14848, 512),
        "shard1/parity1/scalars": (15360, 128),
        "shard1/parity1/finals": (15488, 4096),
        "shard1/parity1/actions": (19584, 8),
        "shard1/parity1/assign": (19648, 16),
    }
    assert described["plan_offset"] == 19712
    assert described["plan_bytes"] == SLOTS * L.PLAN.size
    assert described["total_bytes"] == 19712 + SLOTS * L.PLAN.size


def test_regions_are_aligned_disjoint_and_inside_the_segment(
    control_layout: L.ControlLayout,
) -> None:
    spans = sorted(control_layout.describe()["regions"].values())
    end = 0
    for offset, size in spans:
        assert offset % L.ALIGN == 0
        assert offset >= end
        end = offset + size
    assert end <= control_layout.plan_offset
    assert control_layout.total_bytes >= control_layout.plan_offset + control_layout.plan_bytes


def test_the_rectangle_indexes_history_then_cycles_then_the_bootstrap_row(
    buffer_layout: L.BufferLayout,
) -> None:
    assert buffer_layout.row_index(-1, 0) == 0
    assert buffer_layout.row_index(0, 0) == SLOTS
    assert buffer_layout.row_index(0, 3) == SLOTS + 3
    assert buffer_layout.row_index(CYCLES, SLOTS - 1) == buffer_layout.rows * SLOTS - 1
    assert buffer_layout.cell_offset(0, 1) == 128 + (SLOTS + 1) * ROW_BYTES
    with pytest.raises(IndexError):
        buffer_layout.row_index(-2, 0)
    with pytest.raises(IndexError):
        buffer_layout.row_index(CYCLES + 1, 0)
    with pytest.raises(IndexError):
        buffer_layout.row_index(0, SLOTS)


def test_a_frame_stack_of_one_has_no_history_rows() -> None:
    plain = L.BufferLayout(
        run_id=RUN_ID, cycles=CYCLES, n_slots=SLOTS, row_bytes=ROW_BYTES, codec_version=1
    )
    assert plain.history_rows == 0
    assert plain.rows == CYCLES + 1
    assert plain.row_index(0, 0) == 0


def test_a_short_block_raises_at_construction_not_at_first_write(
    buffer_layout: L.BufferLayout, control_layout: L.ControlLayout
) -> None:
    short = bytearray(buffer_layout.total_bytes - 1)
    with pytest.raises(ValueError, match="needs"):
        buffer_layout.write_header(short)
    with pytest.raises(ValueError, match="needs"):
        buffer_layout.check_header(short)
    with pytest.raises(ValueError, match="needs"):
        control_layout.view(bytearray(control_layout.total_bytes - 1), 0, 0, "scalars")


def test_a_header_from_another_layout_is_refused(buffer_layout: L.BufferLayout) -> None:
    block = bytearray(buffer_layout.total_bytes)
    buffer_layout.write_header(block)
    buffer_layout.check_header(block)

    other = L.BufferLayout(
        run_id=RUN_ID,
        cycles=CYCLES,
        n_slots=SLOTS,
        row_bytes=ROW_BYTES,
        codec_version=2,
        frame_stack=2,
    )
    with pytest.raises(ValueError, match="codec_version"):
        other.check_header(block)

    block[0] = 0
    with pytest.raises(ValueError, match="RLRN"):
        buffer_layout.check_header(block)


def test_the_header_reports_what_the_layout_computed(buffer_layout: L.BufferLayout) -> None:
    block = bytearray(buffer_layout.total_bytes)
    buffer_layout.write_header(block)
    header = L.HEADER.unpack_from(block, 0)
    assert header["magic"] == L.MAGIC
    assert header["cycles"] == CYCLES
    assert header["n_slots"] == SLOTS
    assert header["row_bytes"] == ROW_BYTES
    assert header["obs_offset"] == buffer_layout.obs_offset
    assert header["obs_bytes"] == buffer_layout.obs_bytes
    assert header["run_id"].decode("utf-8") == RUN_ID


def test_a_handle_carries_the_numbers_and_not_the_offsets(buffer_layout: L.BufferLayout) -> None:
    """Both sides compute the offsets; only the inputs cross."""
    handle = buffer_layout.handle()
    decoded = msgspec.json.decode(msgspec.json.encode(handle), type=L.BufferHandle)
    assert decoded == handle
    rebuilt = decoded.layout()
    assert rebuilt.describe() == buffer_layout.describe()
    assert handle.name == L.buffer_segment_name(RUN_ID)


def test_a_control_handle_round_trips(control_layout: L.ControlLayout) -> None:
    handle = control_layout.handle()
    decoded = msgspec.json.decode(msgspec.json.encode(handle), type=L.ControlHandle)
    assert decoded.layout().describe() == control_layout.describe()
    assert handle.name == L.control_segment_name(RUN_ID, 0)


def test_the_error_protocol_round_trips_a_traceback(control_layout: L.ControlLayout) -> None:
    block = bytearray(control_layout.total_bytes)
    view = control_layout.view(block, 1, 0, "error")
    text = "Traceback (most recent call last):\n  File \"worker.py\", line 3\nValueError: boom\n"
    length = L.write_error(view, text)
    assert L.read_error(control_layout.view(block, 1, 0, "error"), length) == text
    # Another shard's region is untouched.
    assert L.read_error(control_layout.view(block, 0, 0, "error"), length) == "\x00" * length


def test_a_long_error_is_cut_on_a_character_boundary(control_layout: L.ControlLayout) -> None:
    block = bytearray(control_layout.total_bytes)
    view = control_layout.view(block, 0, 1, "error")
    text = "é" * L.ERROR_BYTES  # two bytes each, so the cut lands mid-character
    length = L.write_error(view, text)
    assert length <= L.ERROR_BYTES
    recovered = L.read_error(control_layout.view(block, 0, 1, "error"), length)
    assert recovered == "é" * (length // 2)
    assert "�" not in recovered


def test_the_scalar_dtype_matches_the_record() -> None:
    assert L.SCALARS_DTYPE.itemsize == L.SCALARS.size
    assert {name: L.SCALARS_DTYPE.fields[name][1] for name in L.SCALARS_DTYPE.names} == {
        name: offset for name, (offset, _) in L.SCALARS.describe().items()
    }


def test_the_scalars_view_writes_into_the_segment(control_layout: L.ControlLayout) -> None:
    block = bytearray(control_layout.total_bytes)
    scalars = control_layout.scalars(block, 0, 0)
    assert scalars.shape == (control_layout.slots_per_shard,)
    scalars["reward"][:] = np.arange(control_layout.slots_per_shard, dtype=np.float32)
    scalars["flags"][:] = L.FLAG_VALID
    again = control_layout.scalars(block, 0, 0)
    assert again["reward"].tolist() == list(range(control_layout.slots_per_shard))
    assert (again["flags"] & L.FLAG_VALID).all()
    assert not control_layout.scalars(block, 1, 0)["reward"].any()


def test_actions_are_int16_per_slot(control_layout: L.ControlLayout) -> None:
    block = bytearray(control_layout.total_bytes)
    actions = control_layout.actions(block, 1, 1)
    actions[:] = 1234
    assert control_layout.actions(block, 1, 1).tolist() == [1234] * control_layout.slots_per_shard
    assert control_layout.actions(block, 1, 0).tolist() == [0] * control_layout.slots_per_shard


def test_parity_wraps_and_an_unknown_region_is_a_key_error(
    control_layout: L.ControlLayout,
) -> None:
    assert control_layout.region(0, 2, "control") == control_layout.region(0, 0, "control")
    with pytest.raises(KeyError, match="finalz"):
        control_layout.region(0, 0, "finalz")


def test_the_layout_follows_the_spec_it_is_built_from(env_spec: object) -> None:
    """``from_spec`` takes the frame stack off the environment description, not an argument."""
    import msgspec as _msgspec

    stacked = _msgspec.structs.replace(env_spec, frame_stack=3)
    built = L.BufferLayout.from_spec(
        stacked, run_id=RUN_ID, cycles=CYCLES, n_slots=SLOTS, row_bytes=ROW_BYTES, codec_version=1
    )
    assert built.frame_stack == 3
    assert built.rows == CYCLES + 3


def test_a_zero_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        L.BufferLayout(run_id=RUN_ID, cycles=0, n_slots=SLOTS, row_bytes=1, codec_version=1)
    with pytest.raises(ValueError, match="positive"):
        L.ControlLayout(run_id=RUN_ID, worker=0, shards=0, slots_per_shard=1, row_bytes=1)


def test_a_misaligned_record_is_a_construction_error() -> None:
    with pytest.raises(ValueError, match="aligned"):
        L.Record("bad", [("small", "b"), ("wide", "Q")])
