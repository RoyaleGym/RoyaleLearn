"""Each tile's unit type reaches the network beside its producing card, exactly, when asked.

RoyaleGym's ``SpatialObsBuilder(unit_identity=True)`` adds ``unit_ids``: ``[2, H, W]`` uint8, 0
empty, 1 + the unit's index in the engine's unit-type list. It is another vocabulary from the card
ids (the Witch's skeletons are the card "Witch" and the unit "Skeleton"), so it is stored in a
region of its own, carried in ``ObsBatch.unit_ids`` and embedded in a table of its own. Without
it every row, table and digest is what it was. The unit list is positional; the builder's config
names it, and the observation digest covers that config, so a renumbered list is another run.
"""

from __future__ import annotations

from typing import Any

import msgspec
import numpy as np
import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import EntityKind
from royalelearn.api.buffer import MIN_TABLE_STATES
from royalelearn.errors import PreflightError
from royalelearn.rollout.codec import SpatialObsCodec
from royalelearn.rollout.envspec import ComponentSpec, read_env_spec

pytest.importorskip("royalegym.obs", reason="needs royalegym")
from royalegym.obs import UNIT_ID_OFFSET

UNITS = ["Archer", "Barbarian", "Giant", "Goblin", "KingTower", "Knight", "PrincessTower"]


class UnitTypedMock(MockEngine):
    """A MockEngine that says each entity's own unit type, as the engine does from 0.1.17."""

    def unit_types(self) -> list[str]:
        return list(UNITS)

    def state(self) -> Any:
        s = super().state()

        def typed(e: Any) -> Any:
            if e.kind == EntityKind.KING_TOWER:
                unit = "KingTower"
            elif e.kind == EntityKind.PRINCESS_TOWER:
                unit = "PrincessTower"
            else:
                unit = UNITS[(e.card_id + 2) % len(UNITS)]
            return msgspec.structs.replace(e, unit_type=UNITS.index(unit))

        return msgspec.structs.replace(s, entities=[typed(e) for e in s.entities])


ON = {"card_identity": True, "unit_identity": True}
#: Where the engine above lives, for the component allowlist.
MODULES = ("test_unit_identity.",)


def _with(factory_spec: Any, **switches: Any) -> Any:
    builder = factory_spec.obs_builder
    return msgspec.structs.replace(
        factory_spec,
        engine=ComponentSpec("test_unit_identity.UnitTypedMock", {}),
        obs_builder=ComponentSpec(builder.cls, {**builder.kwargs, **switches}),
    )


def _spec(factory: Any) -> Any:
    vec = factory.build_vec(2, MODULES)
    try:
        return read_env_spec(vec, factory, frame_stack=1)
    finally:
        vec.close()


@pytest.fixture(scope="module")
def unit_factory(mock_env_spec: Any) -> Any:
    return _with(mock_env_spec, **ON)


@pytest.fixture(scope="module")
def unit_spec(unit_factory: Any) -> Any:
    return _spec(unit_factory)


@pytest.fixture(scope="module")
def unit_rollout(unit_factory: Any) -> list[dict[str, np.ndarray]]:
    env = unit_factory.build_vec(2, MODULES)
    rng = np.random.default_rng(9)
    noop = int(env.envs[0].action_parser.noop())
    try:
        batch = env.reset(seed=3)[0]
        observations = [batch]
        while len(observations) * env.num_envs < MIN_TABLE_STATES:
            actions = np.full(env.num_envs, noop, dtype=np.int64)
            for index in range(env.num_envs):
                legal = np.flatnonzero(batch["action_mask"][index])
                legal = legal[legal != noop]
                if legal.size and rng.random() < 0.3:
                    actions[index] = int(legal[rng.integers(legal.size)])
            batch = env.step(actions)[0]
            observations.append(batch)
    finally:
        env.close()
    return observations


def _vocab(spec: Any) -> int:
    return int(np.max(np.asarray(spec.obs_space["unit_ids"].high))) + 1


def _codec(spec: Any, rollout: Any) -> SpatialObsCodec:
    codec = SpatialObsCodec()
    codec.table(spec, rollout)
    return codec


def test_the_space_is_its_own_vocabulary(unit_spec: Any, unit_rollout: Any) -> None:
    assert unit_spec.obs_space["unit_ids"].shape[0] == 2
    assert _vocab(unit_spec) == len(UNITS) + UNIT_ID_OFFSET
    seat = unit_rollout[-1]["unit_ids"][0]
    assert int(seat.max()) >= UNIT_ID_OFFSET, "no unit reached the board"


def test_unit_ids_round_trip_exactly_beside_the_card_ids(
    unit_spec: Any, unit_rollout: Any
) -> None:
    import torch

    from royalelearn.learn.buffer import empty_obs

    codec = _codec(unit_spec, unit_rollout)
    assert codec.codec_table.unit_ids == "uint8"
    seat = {key: value[0] for key, value in unit_rollout[-1].items()}
    shape = seat["unit_ids"].shape
    seat["unit_ids"] = (np.arange(np.prod(shape)) % _vocab(unit_spec)).astype(np.uint8)
    seat["unit_ids"] = seat["unit_ids"].reshape(shape)
    block = bytearray(codec.layout.row_bytes)
    codec.pack(seat, memoryview(block), 0)
    out = empty_obs(unit_spec, 1, frames=1, device="cpu")
    raw = torch.frombuffer(block, dtype=torch.uint8).reshape(1, 1, -1)
    codec.unpack_to_device(raw, torch.from_numpy(codec.static_planes(seat)), out)
    assert np.array_equal(out.unit_ids[0].numpy(), seat["unit_ids"].astype(np.int64))
    assert np.array_equal(out.card_ids[0].numpy(), seat["card_ids"].astype(np.int64))


def test_a_table_and_a_space_that_disagree_about_unit_ids_are_refused(unit_spec: Any) -> None:
    from royalelearn.api.buffer import CodecTable

    table = CodecTable(
        plane=tuple((name, "float16", 1.0) for name, _ in unit_spec.spatial_layout),
        vector="float16",
        mask="bitpack",
        ids="uint8",
    )
    with pytest.raises(PreflightError, match="unit_ids"):
        SpatialObsCodec().bind(unit_spec, table)


def _net() -> Any:
    from royalelearn.config import NetConfig

    return NetConfig(device="cpu", autocast_dtype="float32")


def _build(spec: Any) -> Any:
    from royalelearn.learn.nets import DefaultNetworkFactory

    return DefaultNetworkFactory(4242).build(spec, _net(), "cpu")


def _batch(spec: Any, units: Any) -> Any:
    import torch

    from royalelearn.learn.buffer import empty_obs

    obs = empty_obs(spec, 1, frames=1, device="cpu")
    for tensor in (obs.spatial, obs.mask_planes, obs.vector):
        tensor.zero_()
    obs.mask.fill_(True)
    obs.card_ids.zero_()
    return obs._replace(unit_ids=units if units is None else units.to(torch.int64))


def test_the_trunk_embeds_the_unit_types_in_a_table_of_their_own(unit_spec: Any) -> None:
    import torch

    trunk = _build(unit_spec).actor.trunk
    assert trunk.unit_ids_embed is not None
    assert trunk.unit_ids_embed is not trunk.card_ids_embed
    assert trunk.unit_ids_embed.num_embeddings == _vocab(unit_spec)
    assert trunk.unit_ids_embed.padding_idx == 0
    height, width = unit_spec.tiles
    knight = torch.zeros(1, 2, height, width, dtype=torch.int64)
    giant = knight.clone()
    knight[0, 1, 10, 9] = UNIT_ID_OFFSET + UNITS.index("Knight")
    giant[0, 1, 10, 9] = UNIT_ID_OFFSET + UNITS.index("Giant")
    with torch.no_grad():
        a, b = trunk(_batch(unit_spec, knight)), trunk(_batch(unit_spec, giant))
    assert not torch.allclose(a, b), "the unit planes never reached the features"
    with pytest.raises(ValueError, match="unit_ids"):
        trunk(_batch(unit_spec, None))


def test_unit_identity_is_another_architecture(unit_spec: Any, unit_factory: Any) -> None:
    from royalelearn.learn.nets import DefaultNetworkFactory

    cards_only = msgspec.structs.replace(
        unit_factory,
        obs_builder=ComponentSpec(
            unit_factory.obs_builder.cls,
            {k: v for k, v in unit_factory.obs_builder.kwargs.items() if k != "unit_identity"},
        ),
    )
    factory = DefaultNetworkFactory(4242)
    assert factory.arch_digest(unit_spec, _net()) != factory.arch_digest(_spec(cards_only), _net())


def test_a_real_iteration_collects_stores_and_trains_on_unit_ids(tmp_path: Any) -> None:
    import torch

    from royalelearn.testing import coordinator, tiny_config

    config = tiny_config(tmp_path)
    config = msgspec.structs.replace(
        config, env=_with(config.env, **ON), extra_component_modules=list(MODULES)
    )
    with coordinator(config) as run:
        before = run.model.actor.trunk.unit_ids_embed.weight.detach().clone()
        run.iterate()
        after = run.model.actor.trunk.unit_ids_embed.weight.detach()
        assert run.codec.layout.unit_planes == 2
        run._probe_backward(torch)
    assert not torch.equal(before, after), "the unit embedding never received a gradient"
    assert torch.all(after[0] == 0), "the empty tile's embedding moved"


def test_a_vocabulary_past_a_byte_is_stored_in_two(unit_spec: Any, unit_rollout: Any) -> None:
    """If the engine's unit list passes 255 types the planes widen; the codec follows the space.
    Plant: keep one byte and every id past 255 comes back wrapped."""
    import torch

    from royalelearn.learn.buffer import empty_obs

    wide = dict(unit_spec.obs_space)
    key = wide["unit_ids"]
    wide["unit_ids"] = msgspec.structs.replace(key, dtype="uint16", high=(299.0,) * len(key.high))
    spec = msgspec.structs.replace(unit_spec, obs_space=wide)
    codec = _codec(spec, unit_rollout)
    assert codec.codec_table.unit_ids == "uint16" and codec.layout.unit_bytes == 2
    seat = {key: value[0] for key, value in unit_rollout[-1].items()}
    shape = seat["unit_ids"].shape
    seat["unit_ids"] = (np.arange(np.prod(shape)) % 300).astype(np.uint16).reshape(shape)
    block = bytearray(codec.layout.row_bytes)
    codec.pack(seat, memoryview(block), 0)
    out = empty_obs(spec, 1, frames=1, device="cpu")
    raw = torch.frombuffer(block, dtype=torch.uint8).reshape(1, 1, -1)
    codec.unpack_to_device(raw, torch.from_numpy(codec.static_planes(seat)), out)
    assert np.array_equal(out.unit_ids[0].numpy(), seat["unit_ids"].astype(np.int64))
