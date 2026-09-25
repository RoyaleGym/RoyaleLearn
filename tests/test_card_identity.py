"""D2's learner half: the card on each tile reaches the network, exactly, and only when asked.

RoyaleGym ef574ec added ``SpatialObsBuilder(card_identity=True)``: an integer plane per side,
``card_ids`` ``[2, H, W]`` uint8, 0 empty, 1 crown tower, 2 + card id for a unit of that card, and
``enemy_last_card`` at the end of the vector. Without it a policy cannot tell a Giant from a
Knight on the same tile, and no throughput comparison would ever have shown that ceiling.

Three properties, each a way this could go wrong quietly:

- **Exact.** A card id stored as a scaled half and read back as 6.997 is embedded as card 6. The
  codec stores ``card_ids`` in a region of its own, as bytes, and decodes it to integers.
- **Nothing dropped.** The codec used to take the three keys it knew and ignore the rest, so a
  switched-on builder's planes would have been discarded before the network ever saw them. A key
  the codec does not store is now a refusal at bind, not a silence.
- **Nothing moves while it is off.** Every checkpoint and ladder snapshot on disk is keyed by an
  arch digest and a codec table digest. With the switch off both are byte-identical to what they
  were, so no run's saves become unloadable because a feature it never used was added.
"""

from __future__ import annotations

from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn.api.buffer import MIN_TABLE_STATES
from royalelearn.errors import PreflightError
from royalelearn.rollout.codec import SpatialObsCodec
from royalelearn.rollout.envspec import ComponentSpec, canonical_json, read_env_spec

#: Taken on 011960d, before any of this existed: the flag-off architecture must not move.
GOLDEN_ARCH_DIGEST = "f244fcb743483eac65b728eae118ffae7ebea5198da351890d13e4734169772f"
GOLDEN_PARAMETERS = 767301


def _switched_on(factory_spec: Any) -> Any:
    builder = factory_spec.obs_builder
    return msgspec.structs.replace(
        factory_spec,
        obs_builder=ComponentSpec(builder.cls, {**builder.kwargs, "card_identity": True}),
    )


@pytest.fixture(scope="module")
def ids_factory(mock_env_spec: Any) -> Any:
    return _switched_on(mock_env_spec)


@pytest.fixture(scope="module")
def ids_spec(ids_factory: Any) -> Any:
    vec = ids_factory.build_vec(2)
    try:
        return read_env_spec(vec, ids_factory, frame_stack=1)
    finally:
        vec.close()


@pytest.fixture(scope="module")
def ids_rollout(ids_factory: Any) -> list[dict[str, np.ndarray]]:
    """Real observations with units on the board, so some tiles carry card ids."""
    env = ids_factory.build_vec(2)
    rng = np.random.default_rng(7)
    noop = int(env.envs[0].action_parser.noop())
    try:
        batch = env.reset(seed=11)[0]
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
    return int(np.max(np.asarray(spec.obs_space["card_ids"].high))) + 1


# -- the space ---------------------------------------------------------------------------


def test_the_vocabulary_comes_from_the_space_and_is_the_catalogue_plus_two(ids_spec: Any) -> None:
    """0 empty, 1 crown tower, 2 + id: the catalogue plus two, read off the space, not a literal."""
    assert _vocab(ids_spec) == ids_spec.num_cards + 2


# -- the codec ---------------------------------------------------------------------------


def _codec(spec: Any, rollout: Any) -> SpatialObsCodec:
    codec = SpatialObsCodec()
    codec.table(spec, rollout)
    return codec


def _unpack(codec: SpatialObsCodec, spec: Any, rows: np.ndarray, statics: np.ndarray) -> Any:
    import torch

    from royalelearn.api.policy import ObsBatch

    count, frames = rows.shape[0], spec.frame_stack
    planes, tiles_y, tiles_x = spec.spatial_shape
    ids_planes = spec.obs_space["card_ids"].shape[0]
    out = ObsBatch(
        spatial=torch.zeros((count, frames * planes, tiles_y, tiles_x)),
        mask_planes=torch.zeros((count, frames * spec.hand_size, tiles_y, tiles_x)),
        vector=torch.zeros((count, spec.vector_size)),
        mask=torch.zeros((count, spec.n_actions), dtype=torch.bool),
        card_ids=torch.full((count, frames * ids_planes, tiles_y, tiles_x), -1, dtype=torch.int64),
    )
    raw = torch.from_numpy(np.array(rows, dtype=np.uint8)).reshape(count, frames, -1)
    codec.unpack_to_device(raw, torch.from_numpy(statics), out)
    return out


def test_card_ids_round_trip_exactly(ids_spec: Any, ids_rollout: Any) -> None:
    """Every id on every tile, as an integer, including the ones a half would have bent."""
    codec = _codec(ids_spec, ids_rollout)
    seats = [{key: value[0] for key, value in batch.items()} for batch in ids_rollout]
    # The real rollout plus one observation that holds every id in the vocabulary, so the test
    # does not depend on which cards the mock happened to play.
    every = dict(seats[-1])
    vocab = _vocab(ids_spec)
    every["card_ids"] = (
        (np.arange(every["card_ids"].size) % vocab)
        .astype(np.uint8)
        .reshape(every["card_ids"].shape)
    )
    seats.append(every)
    assert any(np.any(s["card_ids"] >= 2) for s in seats[:-1]), "no unit reached the board"

    rows = np.stack([np.frombuffer(_pack_one(codec, s), dtype=np.uint8) for s in seats], axis=0)
    out = _unpack(codec, ids_spec, rows, codec.static_planes(seats[0]))
    for index, seat in enumerate(seats):
        assert np.array_equal(out.card_ids[index].numpy(), seat["card_ids"].astype(np.int64))
    assert out.card_ids.dtype.is_floating_point is False


def _pack_one(codec: SpatialObsCodec, observation: dict[str, np.ndarray]) -> bytes:
    block = bytearray(codec.layout.row_bytes)
    codec.pack(observation, memoryview(block), 0)
    return bytes(block)


def test_the_table_records_the_ids_region_and_says_it_is_exact(
    ids_spec: Any, ids_rollout: Any
) -> None:
    table = _codec(ids_spec, ids_rollout).codec_table
    assert table.ids == "uint8"


def test_with_the_switch_off_the_row_and_the_table_are_what_they_were(
    env_spec: Any, ids_spec: Any, ids_rollout: Any
) -> None:
    """The flag-off table encodes with no ids key at all, so its digest cannot have moved."""
    from test_codec import row_bytes_from_the_space

    off = SpatialObsCodec()
    table = off.table(env_spec, _off_sample(env_spec))
    assert table.ids is None
    assert b'"ids"' not in canonical_json(table)
    assert off.layout.row_bytes == row_bytes_from_the_space(env_spec, off)


def _off_sample(env_spec: Any) -> list[dict[str, np.ndarray]]:
    from royalelearn.config import MOCK_ENGINE, default_env_spec

    env = default_env_spec(MOCK_ENGINE).build_vec(2)
    try:
        batch = env.reset(seed=3)[0]
        out = [batch]
        noop = int(env.envs[0].action_parser.noop())
        while len(out) * env.num_envs < MIN_TABLE_STATES:
            batch = env.step(np.full(env.num_envs, noop, dtype=np.int64))[0]
            out.append(batch)
        return out
    finally:
        env.close()


def test_the_on_row_is_the_off_row_plus_one_byte_per_ids_cell(
    ids_spec: Any, ids_rollout: Any
) -> None:
    codec = _codec(ids_spec, ids_rollout)
    layout = codec.layout
    planes = ids_spec.obs_space["card_ids"].shape[0]
    assert layout.ids_stop - layout.ids_start == planes * layout.cells
    assert layout.ids_start == layout.mask_stop, "the new region goes after everything that was"
    assert layout.row_bytes == layout.ids_stop


def test_a_key_the_codec_does_not_store_is_refused_not_dropped(ids_spec: Any) -> None:
    """Before this, an unknown key was ignored at pack time and never reached the network."""
    obs_space = dict(ids_spec.obs_space)
    obs_space["mystery"] = obs_space["card_ids"]
    spec = msgspec.structs.replace(ids_spec, obs_space=obs_space)
    with pytest.raises(PreflightError, match="mystery"):
        SpatialObsCodec().bind(spec, _any_table(ids_spec))


def _any_table(spec: Any) -> Any:
    from royalelearn.api.buffer import CodecTable

    return CodecTable(
        plane=tuple((name, "float16", 1.0) for name, _ in spec.spatial_layout),
        vector="float16",
        mask="bitpack",
        ids="uint8",
    )


def test_a_vocabulary_that_does_not_fit_a_byte_is_refused(ids_spec: Any) -> None:
    """At 254 loadable cards the ids stop fitting one byte. That must be a refusal, not a wrap."""
    space = ids_spec.obs_space["card_ids"]
    wide = msgspec.structs.replace(space, high=[300.0] * len(np.ravel(space.high)))
    spec = msgspec.structs.replace(ids_spec, obs_space={**ids_spec.obs_space, "card_ids": wide})
    with pytest.raises(PreflightError, match="byte"):
        SpatialObsCodec().bind(spec, _any_table(ids_spec))


def test_a_table_and_a_space_that_disagree_about_ids_are_refused(
    env_spec: Any, ids_spec: Any
) -> None:
    """A worker handed an off table for an on space would write rows the learner reads wrong."""
    from royalelearn.api.buffer import CodecTable

    off_table = CodecTable(
        plane=tuple((name, "float16", 1.0) for name, _ in ids_spec.spatial_layout),
        vector="float16",
        mask="bitpack",
    )
    with pytest.raises(PreflightError, match="card_ids"):
        SpatialObsCodec().bind(ids_spec, off_table)


# -- the network -------------------------------------------------------------------------


def _net() -> Any:
    from royalelearn.config import NetConfig

    return NetConfig(device="cpu", autocast_dtype="float32")


def _build(spec: Any) -> Any:
    from royalelearn.learn.nets import DefaultNetworkFactory

    return DefaultNetworkFactory(4242).build(spec, _net(), "cpu")


def test_with_the_switch_off_the_network_is_the_one_every_save_was_made_with(
    env_spec: Any,
) -> None:
    from royalelearn.learn.nets import DefaultNetworkFactory

    assert DefaultNetworkFactory(4242).arch_digest(env_spec, _net()) == GOLDEN_ARCH_DIGEST
    assert sum(p.numel() for p in _build(env_spec).parameters()) == GOLDEN_PARAMETERS


def _batch(spec: Any, ids: Any) -> Any:
    import torch

    from royalelearn.api.policy import ObsBatch

    rows, frames = (1 if ids is None else ids.shape[0]), spec.frame_stack
    height, width = spec.tiles
    mask = torch.zeros(rows, spec.n_actions, dtype=torch.bool)
    mask[:, 0] = True
    return ObsBatch(
        spatial=torch.zeros(rows, frames * spec.spatial_shape[0], height, width),
        mask_planes=torch.zeros(rows, frames * spec.hand_size, height, width),
        vector=torch.zeros(rows, spec.vector_size),
        mask=mask,
        card_ids=ids,
    )


def test_the_trunk_embeds_the_ids_sized_from_the_space(ids_spec: Any) -> None:
    import torch

    trunk = _build(ids_spec).actor.trunk
    assert trunk.card_ids_embed is not None
    assert trunk.card_ids_embed.num_embeddings == _vocab(ids_spec)
    assert trunk.card_ids_embed.padding_idx == 0, "an empty tile is no card, and learns nothing"
    height, width = ids_spec.tiles
    ids = torch.zeros(1, 2, height, width, dtype=torch.int64)
    assert trunk(_batch(ids_spec, ids)).shape[1] == trunk.channels


def test_a_different_card_on_one_tile_changes_what_the_trunk_sees(ids_spec: Any) -> None:
    """The planes are not decoration: one tile's card moves the features."""
    import torch

    trunk = _build(ids_spec).actor.trunk
    height, width = ids_spec.tiles
    knight = torch.zeros(1, 2, height, width, dtype=torch.int64)
    giant = knight.clone()
    knight[0, 1, 10, 9] = 2 + 0
    giant[0, 1, 10, 9] = 2 + 1
    with torch.no_grad():
        a, b = trunk(_batch(ids_spec, knight)), trunk(_batch(ids_spec, giant))
    assert not torch.allclose(a, b)


def test_a_batch_without_ids_for_a_network_that_reads_them_is_refused(ids_spec: Any) -> None:
    trunk = _build(ids_spec).actor.trunk
    with pytest.raises(ValueError, match="card_ids"):
        trunk(_batch(ids_spec, None))


def test_a_renumbered_catalogue_is_a_different_architecture(ids_spec: Any) -> None:
    """Catalogue ids are positional. A saved embedding table read against a renumbered catalogue
    would be the same weights playing a different game; the digest makes it a refusal."""
    from royalelearn.learn.nets import DefaultNetworkFactory

    space = ids_spec.obs_space["card_ids"]
    grown = msgspec.structs.replace(space, high=[h + 1.0 for h in np.ravel(space.high).tolist()])
    renumbered = msgspec.structs.replace(
        ids_spec, obs_space={**ids_spec.obs_space, "card_ids": grown}
    )
    factory = DefaultNetworkFactory(4242)
    with pytest.raises(PreflightError, match="arch_digest"):
        factory.check_digest(renumbered, _net(), factory.arch_digest(ids_spec, _net()))


def test_the_actor_can_select_rows_of_a_batch_that_carries_ids(ids_spec: Any) -> None:
    """``actor_critic`` rebuilt a batch from every tensor in it; an optional one broke that."""
    import torch

    model = _build(ids_spec)
    height, width = ids_spec.tiles
    ids = torch.zeros(3, 2, height, width, dtype=torch.int64)
    ids[2, 0, 5, 5] = 3
    batch = _batch(ids_spec, ids)
    rows = torch.tensor([0, 2])
    with torch.no_grad():
        chosen = model.distribution_on_rows(batch, rows)
        whole = model.actor.distribution(batch)
    # The rows it chose are the rows it was asked for, ids included: row 2 carries a card and
    # row 1 does not, so a selection that dropped or misaligned the ids would disagree here.
    assert torch.allclose(chosen.log_probs, whole.log_probs[rows])


# -- the evaluation actors -------------------------------------------------------------------


class _Spec:
    """The fields ``obs_batch`` reads, at a frame stack of two."""

    hand_size = 1
    tiles = (2, 3)
    n_actions = 1 + hand_size * tiles[0] * tiles[1]
    vector_size = 2
    frame_stack = 2


def _frame(fill: float, legal_tile: int, card: int) -> dict[str, Any]:
    mask = np.zeros(_Spec.n_actions, dtype=bool)
    mask[0] = True
    mask[1 + legal_tile] = True
    ids = np.zeros((2, *_Spec.tiles), dtype=np.uint8)
    ids[1].reshape(-1)[legal_tile] = card
    return {
        "spatial": np.full((1, *_Spec.tiles), fill, dtype=np.float32),
        "vector": np.zeros(_Spec.vector_size, dtype=np.float32),
        "action_mask": mask,
        "card_ids": ids,
    }


def _actors() -> Any:
    from royalelearn.ladder.actors import EvalActors

    return EvalActors(_Spec(), net=None, snapshots=None)  # type: ignore[arg-type]


def test_an_older_frame_keeps_its_own_ids_and_its_own_mask() -> None:
    """What the rectangle decodes for an older frame, and what training therefore saw.

    The history used to keep the board alone, so an older frame's card ids would not have
    existed and its mask planes were zeros. Two frames with different legal tiles and different
    cards make both visible.
    """
    history: list[Any] = []
    actors = _actors()
    actors.obs_batch(_frame(1.0, legal_tile=0, card=5), history)
    batch = actors.obs_batch(_frame(2.0, legal_tile=4, card=9), history)

    ids = batch.card_ids[0].numpy()
    assert ids.shape == (4, *_Spec.tiles)
    assert ids[1].reshape(-1)[4] == 9, "the current frame's enemy plane"
    assert ids[3].reshape(-1)[0] == 5, "the previous frame's enemy plane, not zeros"
    planes = batch.mask_planes[0].numpy()
    assert planes[0].reshape(-1)[4] == 1.0 and planes[0].reshape(-1)[0] == 0.0
    assert planes[1].reshape(-1)[0] == 1.0, "the previous frame's own mask, not an all-zero one"


def test_an_observation_without_ids_gives_a_batch_without_them() -> None:
    frame = _frame(1.0, legal_tile=0, card=5)
    del frame["card_ids"]
    assert _actors().obs_batch(frame, None).card_ids is None


# -- end to end ------------------------------------------------------------------------------


def test_a_real_iteration_collects_stores_and_trains_on_card_ids(tmp_path: Any) -> None:
    """Collection, the rectangle, unpack, the network and the update, with the switch on.

    Every piece above is tested on its own; this is the one place they have to agree about a
    row. It also proves the codec's refusal of unknown keys does not refuse the key it now
    stores, which a unit test built against the codec alone could not.
    """
    import torch

    from test_coordinator import coordinator, tiny_config

    config = tiny_config(tmp_path)
    config = msgspec.structs.replace(config, env=_switched_on(config.env))
    with coordinator(config) as run:
        assert run.model.actor.trunk.card_ids_embed is not None
        before = run.model.actor.trunk.card_ids_embed.weight.detach().clone()
        run.iterate()
        after = run.model.actor.trunk.card_ids_embed.weight.detach()
        assert run.codec.codec_table.ids == "uint8"
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after), "the id embedding never received a gradient"
    assert torch.all(after[0] == 0), "the empty tile's embedding moved"


def test_a_table_that_claims_ids_for_a_space_without_them_is_refused(env_spec: Any) -> None:
    """The other direction, and the one only the agreement check can catch.

    A space WITH ids and a table without them is also refused by the storage check that follows,
    so testing that direction alone let the agreement check be deleted with the suite green.
    """
    from royalelearn.api.buffer import CodecTable

    claims_ids = CodecTable(
        plane=tuple((name, "float16", 1.0) for name, _ in env_spec.spatial_layout),
        vector="float16",
        mask="bitpack",
        ids="uint8",
    )
    with pytest.raises(PreflightError, match="card_ids"):
        SpatialObsCodec().bind(env_spec, claims_ids)


def test_the_memory_probe_builds_the_batch_a_real_minibatch_is(tmp_path: Any) -> None:
    """The probe runs only on CUDA, so no CPU run would ever show it missing the id planes.

    Called directly here, on the CPU, with the switch on: the forward it measures is the one a
    real minibatch takes, so a probe batch without card_ids is a refusal from the network.
    """
    import torch

    from test_coordinator import coordinator, tiny_config

    config = tiny_config(tmp_path)
    config = msgspec.structs.replace(config, env=_switched_on(config.env))
    with coordinator(config) as run:
        run._probe_backward(torch)
