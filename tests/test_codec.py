"""The observation codec: what the rule decides, and what survives the round trip.

Every width here comes from the observation space or from the codec's own layout. The row size is
checked against the arithmetic of ``docs/harness-spec.md`` section 2.1 recomputed from the space,
on MockEngine's catalogue and on a wider one, so a test that passes on both cannot be carrying a
number from either.
"""

from __future__ import annotations

import math
from typing import Any

import msgspec
import numpy as np
import pytest

from conftest import synthetic_env_spec
from royalelearn.api.buffer import MIN_TABLE_STATES
from royalelearn.api.rollout import EnvSpec
from royalelearn.errors import PreflightError
from royalelearn.rollout.codec import (
    STORAGE_FLOAT16,
    STORAGE_STATIC,
    STORAGE_UINT8,
    SpatialObsCodec,
)
from royalelearn.rollout.scripted import RANDOM_LEGAL_NOOP_PROB
from royalelearn.seeding import PREFLIGHT_SAMPLE, derive_generator, stream_path

# A catalogue several times MockEngine's, so the row arithmetic runs against two sets of widths.
WIDER_CATALOGUE = 65

# Half precision carries ten bits of mantissa, so a value in the builder's [0, 1] range comes
# back within this of where it went in.
HALF_RESOLUTION = 2.0**-10

BITS_PER_BYTE = 8

SEED = 20260921


@pytest.fixture(scope="module")
def rollout(mock_env_spec: Any) -> list[dict[str, np.ndarray]]:
    """Enough real observations to decide a table from, from an environment of this module's own.

    Its own because these are stepped: a session-wide environment that other tests read would
    not be where they left it. Played with legal cards rather than walked with the no-op,
    because storage is decided from what the sample contains: an idle walk never puts a unit on
    the board or takes a tower below full, and those are exactly the states in which a plane
    that is only sometimes fractional shows that it is.
    """
    env = mock_env_spec.build_vec(2)
    generator = derive_generator(SEED, stream_path(PREFLIGHT_SAMPLE))
    noop = int(env.envs[0].action_parser.noop())
    try:
        batch = env.reset(seed=11)[0]
        observations = [batch]
        while len(observations) * env.num_envs < MIN_TABLE_STATES:
            actions = np.full(env.num_envs, noop, dtype=np.int64)
            for index in range(env.num_envs):
                legal = np.flatnonzero(batch["action_mask"][index])
                legal = legal[legal != noop]
                if legal.size and generator.random() >= RANDOM_LEGAL_NOOP_PROB:
                    actions[index] = int(legal[generator.integers(legal.size)])
            batch = env.step(actions)[0]
            observations.append(batch)
    finally:
        env.close()
    return observations


@pytest.fixture(scope="module")
def sample(rollout: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    """One seat's observations, as the codec packs them."""
    return [{key: value[0] for key, value in batch.items()} for batch in rollout]


@pytest.fixture
def codec(env_spec: EnvSpec, rollout: list[dict[str, np.ndarray]]) -> SpatialObsCodec:
    built = SpatialObsCodec()
    built.table(env_spec, rollout)
    return built


def row_bytes_from_the_space(spec: EnvSpec, codec: SpatialObsCodec) -> int:
    """Section 2.1's arithmetic, recomputed from the space rather than quoted."""
    layout = codec.layout
    planes, tiles_y, tiles_x = spec.spatial_shape
    cells = tiles_y * tiles_x
    stored_as_bytes = len(layout.u8_planes)
    stored_as_halves = len(layout.f16_planes)
    assert stored_as_bytes + stored_as_halves + len(layout.static_planes) == planes
    return (
        cells * (stored_as_bytes + 2 * stored_as_halves)
        + 2 * spec.vector_size
        + math.ceil(spec.n_actions / BITS_PER_BYTE)
    )


def unpack(
    codec: SpatialObsCodec, spec: EnvSpec, rows: np.ndarray, statics: np.ndarray
) -> Any:
    """One batch of packed rows, decoded into freshly allocated tensors."""
    import torch

    from royalelearn.api.policy import ObsBatch

    count = rows.shape[0]
    frames = spec.frame_stack
    planes, tiles_y, tiles_x = spec.spatial_shape
    out = ObsBatch(
        spatial=torch.zeros((count, frames * planes, tiles_y, tiles_x)),
        mask_planes=torch.zeros((count, frames * spec.hand_size, tiles_y, tiles_x)),
        vector=torch.zeros((count, spec.vector_size)),
        mask=torch.zeros((count, spec.n_actions), dtype=torch.bool),
    )
    raw = torch.from_numpy(np.array(rows, dtype=np.uint8)).reshape(count, frames, -1)
    codec.unpack_to_device(raw, torch.from_numpy(statics), out)
    return out


def pack_all(
    codec: SpatialObsCodec, observations: list[dict[str, np.ndarray]]
) -> np.ndarray:
    row_bytes = codec.layout.row_bytes
    block = bytearray(row_bytes * len(observations))
    view = memoryview(block)
    for index, observation in enumerate(observations):
        codec.pack(observation, view, index)
    return np.frombuffer(bytes(block), dtype=np.uint8).reshape(len(observations), row_bytes)


# --------------------------------------------------------------------------
# The table is decided from the space and the sample
# --------------------------------------------------------------------------


def test_the_table_has_one_entry_per_plane_in_the_layout_s_order(
    codec: SpatialObsCodec, env_spec: EnvSpec
) -> None:
    table = codec.codec_table
    assert len(table.plane) == env_spec.n_planes
    assert [name for name, _, _ in table.plane] == [name for name, _ in env_spec.spatial_layout]
    assert table.vector == STORAGE_FLOAT16
    assert table.mask == "bitpack"


def test_a_plane_the_layout_declares_static_is_not_stored(
    codec: SpatialObsCodec, env_spec: EnvSpec
) -> None:
    declared = set(env_spec.static_planes)
    assert declared
    stored = {name for name, storage, _ in codec.codec_table.plane if storage != STORAGE_STATIC}
    assert declared.isdisjoint(stored)
    assert len(codec.layout.static_planes) == len(declared)


def test_a_plane_constant_on_the_sample_is_still_stored(
    codec: SpatialObsCodec, env_spec: EnvSpec, sample: list[dict[str, np.ndarray]]
) -> None:
    """Existence is decided from the declaration, storage from the sample.

    The tower planes are constant across any sample in which no tower falls; promoting them to
    static would freeze a tower at full health for the rest of a run.
    """
    spatial = np.stack([observation["spatial"] for observation in sample])
    flat = spatial.reshape(spatial.shape[0], env_spec.n_planes, -1)
    constant = flat.max(axis=(0, 2)) == flat.min(axis=(0, 2))
    storage = {name: kind for name, kind, _ in codec.codec_table.plane}
    unchanging = [
        name
        for index, (name, static) in enumerate(env_spec.spatial_layout)
        if constant[index] and not static
    ]
    assert unchanging, "the sample changed every non-static plane; this test proves nothing"
    for name in unchanging:
        assert storage[name] != STORAGE_STATIC


def test_a_declared_bound_over_a_byte_sends_a_plane_to_half_precision(
    env_spec: EnvSpec, rollout: list[dict[str, np.ndarray]]
) -> None:
    index = next(i for i, (_, static) in enumerate(env_spec.spatial_layout) if not static)
    spatial = env_spec.obs_space["spatial"]
    raised = list(spatial.high)
    raised[index] = 256.0
    wider = msgspec.structs.replace(
        env_spec,
        obs_space={
            **env_spec.obs_space,
            "spatial": msgspec.structs.replace(spatial, high=tuple(raised)),
        },
    )
    under = SpatialObsCodec().table(env_spec, rollout)
    over = SpatialObsCodec().table(wider, rollout)
    assert under.plane[index][1] == STORAGE_UINT8
    assert over.plane[index][1] == STORAGE_FLOAT16


def test_a_plane_that_is_not_whole_numbers_goes_to_half_precision(
    codec: SpatialObsCodec, env_spec: EnvSpec, sample: list[dict[str, np.ndarray]]
) -> None:
    spatial = np.stack([observation["spatial"] for observation in sample])
    fractional = {
        name
        for index, (name, static) in enumerate(env_spec.spatial_layout)
        if not static and np.any(spatial[:, index] != np.rint(spatial[:, index]))
    }
    assert fractional, "nothing in the sample was fractional; this test proves nothing"
    storage = {name: kind for name, kind, _ in codec.codec_table.plane}
    for name in fractional:
        assert storage[name] == STORAGE_FLOAT16


def test_a_sample_too_small_to_decide_a_table_from_is_refused(
    env_spec: EnvSpec, rollout: list[dict[str, np.ndarray]]
) -> None:
    """The row size of the whole run follows from this one decision.

    A short sample and an idle one fail the same way: neither has been in the states that decide
    a plane's storage, and a table decided from one of them rounds a plane for the rest of the
    run without anything being raised. The count is refused because it is the part that can be
    checked.
    """
    with pytest.raises(PreflightError, match="0 of them"):
        SpatialObsCodec().table(env_spec, [])
    with pytest.raises(PreflightError, match=f"minimum of {MIN_TABLE_STATES}"):
        SpatialObsCodec().table(env_spec, rollout[:1])
    # And the sample this module decides its own tables from clears it.
    SpatialObsCodec().table(env_spec, rollout)


def test_the_digest_moves_with_the_table(codec: SpatialObsCodec) -> None:
    table = codec.codec_table
    moved = msgspec.structs.replace(
        table, plane=((table.plane[0][0], STORAGE_FLOAT16, 1.0), *table.plane[1:])
    )
    assert table.digest() == msgspec.structs.replace(table).digest()
    assert table.digest() != moved.digest()


def test_the_codec_version_is_the_rule_s_and_reaches_the_identity(
    codec: SpatialObsCodec,
) -> None:
    from royalelearn.identity import RunIdentity

    assert isinstance(codec.codec_version, int)
    names = {field.name for field in msgspec.structs.fields(RunIdentity)}
    assert {"codec_version", "codec_table_digest"} <= names


# --------------------------------------------------------------------------
# The row
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_cards", [None, WIDER_CATALOGUE])
def test_the_row_size_is_the_arithmetic_of_the_space(
    env_spec: EnvSpec, rollout: list[dict[str, np.ndarray]], num_cards: int | None
) -> None:
    spec = env_spec if num_cards is None else synthetic_env_spec(env_spec, num_cards)
    built = SpatialObsCodec()
    built.table(spec, rollout)
    assert built.row_bytes(spec) == row_bytes_from_the_space(spec, built)


def test_the_two_catalogues_do_not_agree_on_the_row_size(
    env_spec: EnvSpec, rollout: list[dict[str, np.ndarray]]
) -> None:
    """The check above would prove nothing if both catalogues packed the same row."""
    wider = synthetic_env_spec(env_spec, WIDER_CATALOGUE)
    narrow = SpatialObsCodec()
    narrow.table(env_spec, rollout)
    wide = SpatialObsCodec()
    wide.table(wider, rollout)
    assert wide.row_bytes(wider) != narrow.row_bytes(env_spec)


def test_the_mask_planes_are_not_stored(codec: SpatialObsCodec, env_spec: EnvSpec) -> None:
    planes, tiles_y, tiles_x = env_spec.spatial_shape
    cells = tiles_y * tiles_x
    layout = codec.layout
    assert layout.row_bytes < cells * planes * np.float32().itemsize
    assert layout.mask_stop - layout.mask_start == math.ceil(env_spec.n_actions / BITS_PER_BYTE)


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


def test_every_byte_plane_and_the_mask_round_trip_exactly(
    codec: SpatialObsCodec, env_spec: EnvSpec, sample: list[dict[str, np.ndarray]]
) -> None:
    pytest.importorskip("torch")
    rows = pack_all(codec, sample)
    statics = codec.static_planes(sample[0])
    out = unpack(codec, env_spec, rows, statics)
    spatial = out.spatial.numpy().reshape(len(sample), env_spec.n_planes, *env_spec.tiles)
    original = np.stack([observation["spatial"] for observation in sample])
    byte_planes = list(codec.layout.u8_planes)
    assert byte_planes
    assert np.array_equal(spatial[:, byte_planes], original[:, byte_planes])
    masks = np.stack([observation["action_mask"] for observation in sample])
    assert np.array_equal(out.mask.numpy().astype(np.int8), masks)


def test_the_static_planes_come_back_from_the_side_channel(
    codec: SpatialObsCodec, env_spec: EnvSpec, sample: list[dict[str, np.ndarray]]
) -> None:
    pytest.importorskip("torch")
    rows = pack_all(codec, sample)
    statics = codec.static_planes(sample[0])
    out = unpack(codec, env_spec, rows, statics)
    spatial = out.spatial.numpy().reshape(len(sample), env_spec.n_planes, *env_spec.tiles)
    original = np.stack([observation["spatial"] for observation in sample])
    static_planes = list(codec.layout.static_planes)
    assert np.array_equal(spatial[:, static_planes], original[:, static_planes])


def test_the_half_precision_parts_round_trip_to_half_s_resolution(
    codec: SpatialObsCodec, env_spec: EnvSpec, sample: list[dict[str, np.ndarray]]
) -> None:
    pytest.importorskip("torch")
    rows = pack_all(codec, sample)
    out = unpack(codec, env_spec, rows, codec.static_planes(sample[0]))
    original = np.stack([observation["spatial"] for observation in sample])
    half_planes = list(codec.layout.f16_planes)
    assert half_planes
    spatial = out.spatial.numpy().reshape(len(sample), env_spec.n_planes, *env_spec.tiles)
    got = spatial[:, half_planes]
    want = original[:, half_planes]
    assert np.all(np.abs(got - want) <= HALF_RESOLUTION * np.maximum(np.abs(want), 1.0))
    vectors = np.stack([observation["vector"] for observation in sample])
    assert np.all(np.abs(out.vector.numpy() - vectors) <= HALF_RESOLUTION)


def test_the_mask_planes_are_the_environment_s_reshaped_mask(
    codec: SpatialObsCodec, env_spec: EnvSpec, sample: list[dict[str, np.ndarray]]
) -> None:
    pytest.importorskip("torch")
    rows = pack_all(codec, sample)
    out = unpack(codec, env_spec, rows, codec.static_planes(sample[0]))
    planes = out.mask_planes.numpy().astype(np.int8)
    expected = np.stack([observation["mask_planes"] for observation in sample])
    assert np.array_equal(planes.reshape(expected.shape), expected)


def test_a_row_of_zero_bytes_decodes_to_a_zero_frame(
    codec: SpatialObsCodec, env_spec: EnvSpec, sample: list[dict[str, np.ndarray]]
) -> None:
    """How the buffer says "no history here"; unambiguous because a real row always has the
    no-op bit of its mask set."""
    pytest.importorskip("torch")
    rows = np.zeros((1, codec.layout.row_bytes), dtype=np.uint8)
    out = unpack(codec, env_spec, rows, codec.static_planes(sample[0]))
    assert not out.spatial.any()
    assert not out.mask_planes.any()
    assert not out.mask.any()


# --------------------------------------------------------------------------
# The clipping counter
# --------------------------------------------------------------------------


def test_a_value_over_a_declared_bound_is_clipped_and_counted(
    codec: SpatialObsCodec, sample: list[dict[str, np.ndarray]]
) -> None:
    observation = {key: value.copy() for key, value in sample[0].items()}
    plane = codec.layout.u8_planes[0]
    observation["spatial"][plane].flat[0] = 4096.0
    block = bytearray(codec.layout.row_bytes)
    assert codec.reset_clipped() == 0
    codec.pack(observation, memoryview(block), 0)
    assert codec.clipped == 1
    assert np.frombuffer(bytes(block), dtype=np.uint8)[0] == np.iinfo(np.uint8).max
    assert codec.reset_clipped() == 1
    assert codec.clipped == 0


def test_packing_refuses_a_row_outside_the_block(
    codec: SpatialObsCodec, sample: list[dict[str, np.ndarray]]
) -> None:
    block = bytearray(codec.layout.row_bytes)
    with pytest.raises(IndexError):
        codec.pack(sample[0], memoryview(block), 1)


def test_an_unbound_codec_refuses_to_say_how_big_a_row_is(env_spec: EnvSpec) -> None:
    with pytest.raises(PreflightError, match="not bound"):
        SpatialObsCodec().row_bytes(env_spec)


def test_a_table_from_another_observation_is_refused(
    codec: SpatialObsCodec, env_spec: EnvSpec
) -> None:
    table = codec.codec_table
    short = msgspec.structs.replace(table, plane=table.plane[:-1])
    with pytest.raises(PreflightError, match="planes"):
        SpatialObsCodec().bind(env_spec, short)
    renamed = msgspec.structs.replace(
        table, plane=(("not_a_plane", STORAGE_UINT8, 1.0), *table.plane[1:])
    )
    with pytest.raises(PreflightError, match="names it"):
        SpatialObsCodec().bind(env_spec, renamed)
