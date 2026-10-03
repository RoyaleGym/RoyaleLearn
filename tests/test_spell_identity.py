"""The spells' ids reach the network beside the cards', exactly, and only when asked.

RoyaleGym's ``SpatialObsBuilder(spell_identity=True)`` adds ``spell_ids``: ``[4, H, W]`` uint8 in
the ``card_ids`` vocabulary (own spell centre, enemy spell centre, own aim, enemy aim once seen).
It needs ``card_identity``. RoyaleLearn carries it as more id planes after ``card_ids``: in the
codec's id region, in ``ObsBatch.card_ids``, and through the one card embedding table into the
trunk. Without it every row, table and digest is what it was.
"""

from __future__ import annotations

from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn.api.buffer import MIN_TABLE_STATES
from royalelearn.errors import PreflightError
from royalelearn.obs_layout import id_planes, id_stack
from royalelearn.rollout.codec import SpatialObsCodec
from royalelearn.rollout.envspec import ComponentSpec, read_env_spec

SPELLS_ON = {"card_identity": True, "spell_identity": True, "spell_aim_after_ticks": 20}


def _with(factory_spec: Any, **switches: Any) -> Any:
    builder = factory_spec.obs_builder
    return msgspec.structs.replace(
        factory_spec, obs_builder=ComponentSpec(builder.cls, {**builder.kwargs, **switches})
    )


@pytest.fixture(scope="module")
def spell_factory(mock_env_spec: Any) -> Any:
    return _with(mock_env_spec, **SPELLS_ON)


@pytest.fixture(scope="module")
def cards_factory(mock_env_spec: Any) -> Any:
    """The same observation without the spell planes (the aim plane stays: it is spatial)."""
    return _with(mock_env_spec, card_identity=True, spell_aim_after_ticks=20)


def _spec(factory: Any) -> Any:
    vec = factory.build_vec(2)
    try:
        return read_env_spec(vec, factory, frame_stack=1)
    finally:
        vec.close()


@pytest.fixture(scope="module")
def spell_spec(spell_factory: Any) -> Any:
    return _spec(spell_factory)


@pytest.fixture(scope="module")
def spell_rollout(spell_factory: Any) -> list[dict[str, np.ndarray]]:
    env = spell_factory.build_vec(2)
    rng = np.random.default_rng(3)
    noop = int(env.envs[0].action_parser.noop())
    try:
        batch = env.reset(seed=5)[0]
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


def _vocab(spec: Any, key: str = "card_ids") -> int:
    return int(np.max(np.asarray(spec.obs_space[key].high))) + 1


def _codec(spec: Any, rollout: Any) -> SpatialObsCodec:
    codec = SpatialObsCodec()
    codec.table(spec, rollout)
    return codec


def _pack_one(codec: SpatialObsCodec, observation: dict[str, np.ndarray]) -> bytes:
    block = bytearray(codec.layout.row_bytes)
    codec.pack(observation, memoryview(block), 0)
    return bytes(block)


# -- the space and the helpers -------------------------------------------------------------


def test_the_spell_planes_share_the_card_vocabulary(spell_spec: Any) -> None:
    assert spell_spec.obs_space["spell_ids"].shape[0] == 4
    assert _vocab(spell_spec, "spell_ids") == _vocab(spell_spec)
    assert id_planes(spell_spec.obs_space) == 2 + 4


def test_the_id_block_is_card_ids_then_spell_ids(spell_rollout: Any) -> None:
    seat = {key: value[0] for key, value in spell_rollout[-1].items()}
    block = id_stack(seat)
    assert np.array_equal(block[:2], seat["card_ids"])
    assert np.array_equal(block[2:], seat["spell_ids"])
    cards_only = {k: v for k, v in seat.items() if k != "spell_ids"}
    assert id_stack(cards_only) is cards_only["card_ids"], "no spells: the card planes as they are"
    assert id_stack({"spatial": seat["spatial"]}) is None


# -- the codec -------------------------------------------------------------------------------


def test_spell_ids_round_trip_exactly_after_the_card_ids(
    spell_spec: Any, spell_rollout: Any
) -> None:
    """Every id in the vocabulary on the spell planes, read back as integers in place."""
    import torch

    from royalelearn.api.policy import ObsBatch

    codec = _codec(spell_spec, spell_rollout)
    seat = {key: value[0] for key, value in spell_rollout[-1].items()}
    vocab = _vocab(spell_spec)
    shape = seat["spell_ids"].shape
    seat["spell_ids"] = (np.arange(seat["spell_ids"].size) % vocab).astype(np.uint8).reshape(shape)
    seat["card_ids"] = np.full_like(seat["card_ids"], 3)
    row = np.frombuffer(_pack_one(codec, seat), dtype=np.uint8)[None]
    planes, tiles_y, tiles_x = spell_spec.spatial_shape
    out = ObsBatch(
        spatial=torch.zeros((1, planes, tiles_y, tiles_x)),
        mask_planes=torch.zeros((1, spell_spec.hand_size, tiles_y, tiles_x)),
        vector=torch.zeros((1, spell_spec.vector_size)),
        mask=torch.zeros((1, spell_spec.n_actions), dtype=torch.bool),
        card_ids=torch.full((1, 6, tiles_y, tiles_x), -1, dtype=torch.int64),
    )
    raw = torch.from_numpy(np.array(row)).reshape(1, 1, -1)
    codec.unpack_to_device(raw, torch.from_numpy(codec.static_planes(seat)), out)
    assert np.array_equal(out.card_ids[0, :2].numpy(), seat["card_ids"].astype(np.int64))
    assert np.array_equal(out.card_ids[0, 2:].numpy(), seat["spell_ids"].astype(np.int64))


def test_the_spell_row_is_the_card_row_plus_one_byte_per_spell_cell(
    spell_spec: Any, spell_rollout: Any, cards_factory: Any
) -> None:
    cards_spec = _spec(cards_factory)
    spells = _codec(spell_spec, spell_rollout)
    strip = [{k: v for k, v in batch.items() if k != "spell_ids"} for batch in spell_rollout]
    cards = _codec(cards_spec, strip)
    cells = spell_spec.tiles[0] * spell_spec.tiles[1]
    assert spells.layout.row_bytes == cards.layout.row_bytes + 4 * cells
    assert spells.codec_table == cards.codec_table, "it says how ids are stored, not how many"


def test_spell_ids_without_card_ids_are_refused(spell_spec: Any) -> None:
    obs_space = {k: v for k, v in spell_spec.obs_space.items() if k != "card_ids"}
    spec = msgspec.structs.replace(spell_spec, obs_space=obs_space)
    table = msgspec.structs.replace(_codec_table(spell_spec), ids=None)
    with pytest.raises(PreflightError, match="spell_ids without card_ids"):
        SpatialObsCodec().bind(spec, table)


def test_spell_ids_in_another_vocabulary_are_refused(spell_spec: Any) -> None:
    import gymnasium as gym

    obs_space = dict(spell_spec.obs_space)
    spell = obs_space["spell_ids"]
    obs_space["spell_ids"] = gym.spaces.Box(0, int(np.max(spell.high)) + 1, spell.shape, np.uint8)
    spec = msgspec.structs.replace(spell_spec, obs_space=obs_space)
    with pytest.raises(PreflightError, match="one vocabulary"):
        SpatialObsCodec().bind(spec, _codec_table(spell_spec))


def _codec_table(spec: Any) -> Any:
    from royalelearn.api.buffer import CodecTable

    return CodecTable(
        plane=tuple((name, "float16", 1.0) for name, _ in spec.spatial_layout),
        vector="float16",
        mask="bitpack",
        ids="uint8",
    )


# -- the network -----------------------------------------------------------------------------


def _build(spec: Any) -> Any:
    from royalelearn.learn.nets import DefaultNetworkFactory

    return DefaultNetworkFactory(4242).build(spec, _net(), "cpu")


def _batch(spec: Any, ids: Any) -> Any:
    import torch

    from royalelearn.api.policy import ObsBatch

    height, width = spec.tiles
    return ObsBatch(
        spatial=torch.zeros(1, spec.n_planes, height, width),
        mask_planes=torch.zeros(1, spec.hand_size, height, width),
        vector=torch.zeros(1, spec.vector_size),
        mask=torch.ones(1, spec.n_actions, dtype=torch.bool),
        card_ids=ids,
    )


def test_the_trunk_reads_the_spell_planes_through_the_card_table(spell_spec: Any) -> None:
    import torch

    trunk = _build(spell_spec).actor.trunk
    assert trunk.card_ids_embed.num_embeddings == _vocab(spell_spec)
    height, width = spell_spec.tiles
    fireball = torch.zeros(1, 6, height, width, dtype=torch.int64)
    arrows = fireball.clone()
    fireball[0, 3, 20, 9] = 5
    arrows[0, 3, 20, 9] = 6
    with torch.no_grad():
        a, b = trunk(_batch(spell_spec, fireball)), trunk(_batch(spell_spec, arrows))
    assert not torch.allclose(a, b), "the spell planes never reached the features"
    with pytest.raises(ValueError, match="input channels"):
        trunk(_batch(spell_spec, fireball[:, :2]))


def test_spell_identity_is_another_architecture(
    spell_spec: Any, cards_factory: Any
) -> None:
    from royalelearn.learn.nets import DefaultNetworkFactory

    factory = DefaultNetworkFactory(4242)
    cards_spec = _spec(cards_factory)
    assert factory.arch_digest(spell_spec, _net()) != factory.arch_digest(cards_spec, _net())


def _net() -> Any:
    from royalelearn.config import NetConfig

    return NetConfig(device="cpu", autocast_dtype="float32")


# -- the ladder's actors and the run ---------------------------------------------------------


def test_the_ladder_actor_stacks_the_spell_planes_after_the_card_planes(
    spell_spec: Any, spell_rollout: Any
) -> None:
    from royalelearn.ladder.actors import EvalActors

    actors = EvalActors(spell_spec, net=None, snapshots=None)  # type: ignore[arg-type]
    seat = {key: value[0] for key, value in spell_rollout[-1].items()}
    seat["spell_ids"] = np.full_like(seat["spell_ids"], 7)
    batch = actors.obs_batch(seat, None)
    assert tuple(batch.card_ids.shape[1:2]) == (6,)
    assert int(batch.card_ids[0, 2:].min()) == 7


def test_a_real_iteration_collects_stores_and_trains_on_spell_ids(tmp_path: Any) -> None:
    """Collection, the rectangle, unpack, the network and the update, with the spells on."""
    import torch

    from test_coordinator import coordinator, tiny_config

    config = tiny_config(tmp_path)
    config = msgspec.structs.replace(config, env=_with(config.env, **SPELLS_ON))
    with coordinator(config) as run:
        assert run.model.actor.trunk.in_channels > 0
        before = run.model.actor.trunk.card_ids_embed.weight.detach().clone()
        run.iterate()
        after = run.model.actor.trunk.card_ids_embed.weight.detach()
        assert run.codec.layout.ids_planes == 6
        run._probe_backward(torch)
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)
