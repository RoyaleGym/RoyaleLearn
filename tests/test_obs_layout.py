"""Reading the environment: the spec the harness builds everything from, the allow-list a spec's
class paths pass through, and the vector fields the policy head resolves by name.

Every assertion about a width compares against what the environment itself reports. Nothing in
this file knows how wide a vector is, and the two catalogues it runs on have different widths,
so a test that passes on both cannot contain a literal from either.
"""

from __future__ import annotations

import inspect

import msgspec
import pytest

from conftest import synthetic_env_spec
from royalelearn import obs_layout
from royalelearn.api.rollout import EnvSpec
from royalelearn.errors import PreflightError
from royalelearn.rollout.envspec import (
    ALLOWED_MODULE_PREFIXES,
    EnvFactorySpec,
    read_env_spec,
    resolve_component,
)

# A catalogue several times MockEngine's, so that every check below runs against two different
# sets of widths. The number is a fixture, not a fact about any environment.
WIDER_CATALOGUE = 65


# --------------------------------------------------------------------------
# The spec, read off a running environment
# --------------------------------------------------------------------------


def test_the_spec_agrees_with_the_observation_space(env_spec: EnvSpec, vec_env: object) -> None:
    space = vec_env.single_observation_space
    assert set(env_spec.obs_space) == set(space.spaces)
    for key, sub in space.spaces.items():
        assert env_spec.obs_space[key].shape == tuple(sub.shape)
        assert env_spec.obs_space[key].dtype == str(sub.dtype)
    assert env_spec.vector_size == space["vector"].shape[0]
    assert env_spec.spatial_shape == tuple(space["spatial"].shape)
    assert env_spec.n_actions == int(vec_env.single_action_space.n)


def test_the_spatial_bounds_are_per_channel(env_spec: EnvSpec, vec_env: object) -> None:
    spatial = env_spec.obs_space["spatial"]
    declared = vec_env.single_observation_space["spatial"]
    assert len(spatial.high) == env_spec.n_planes
    assert len(spatial.low) == env_spec.n_planes
    assert max(spatial.high) == pytest.approx(float(declared.high.max()))
    assert min(spatial.low) == pytest.approx(float(declared.low.min()))
    assert len(env_spec.obs_space["vector"].high) == 1


def test_the_vector_layout_is_contiguous_and_covers_the_vector(env_spec: EnvSpec) -> None:
    offset = 0
    for _, start, size in env_spec.vector_layout:
        assert start == offset
        assert size > 0
        offset += size
    assert offset == env_spec.vector_size


def test_the_layouts_come_from_the_builder(env_spec: EnvSpec, vec_env: object) -> None:
    builder = vec_env.envs[0].obs_builder
    assert env_spec.spatial_layout == tuple(builder.spatial_layout())
    assert [name for name, _, _ in env_spec.vector_layout] == [
        field.key for field in builder.vector_layout()
    ]
    assert env_spec.static_planes == tuple(
        name for name, static in builder.spatial_layout() if static
    )
    assert 0 < len(env_spec.static_planes) < env_spec.n_planes


def test_the_timing_fields_come_from_the_env_and_the_engine(
    env_spec: EnvSpec, vec_env: object
) -> None:
    config = vec_env.envs[0].config()
    state = vec_env.envs[0].engine.state()
    assert env_spec.decision_ms == config["decision_ms"]
    assert env_spec.decision_ticks == config["decision_ticks"]
    assert env_spec.decision_ticks * env_spec.tick_ms == env_spec.decision_ms
    assert env_spec.tick_ms == state.tick_ms
    assert env_spec.regular_ticks == state.regular_ticks
    assert env_spec.overtime_ticks == state.overtime_ticks
    assert env_spec.num_cards == len(vec_env.envs[0].engine.cards())


def test_the_action_space_matches_the_mask_planes(env_spec: EnvSpec) -> None:
    hand, tiles_y, tiles_x = env_spec.obs_space["mask_planes"].shape
    assert env_spec.hand_size == hand
    assert env_spec.tiles == (tiles_y, tiles_x)
    assert env_spec.n_actions == 1 + hand * tiles_y * tiles_x
    assert env_spec.spatial_shape[1:] == env_spec.tiles


def test_a_spec_that_disagrees_with_its_space_is_refused(env_spec: EnvSpec) -> None:
    with pytest.raises(ValueError, match="vector_size"):
        msgspec.structs.replace(env_spec, vector_size=env_spec.vector_size + 1)
    with pytest.raises(ValueError, match="vector_layout covers"):
        msgspec.structs.replace(env_spec, vector_layout=env_spec.vector_layout[:-1])
    with pytest.raises(ValueError, match="spatial_layout declares"):
        msgspec.structs.replace(env_spec, spatial_layout=env_spec.spatial_layout[:-1])


def test_the_obs_digest_moves_with_the_frame_stack(
    vec_env: object, mock_env_spec: EnvFactorySpec, env_spec: EnvSpec
) -> None:
    stacked = read_env_spec(vec_env, mock_env_spec, frame_stack=2)
    assert stacked.frame_stack == 2
    assert stacked.obs_digest != env_spec.obs_digest


def test_the_obs_digest_moves_with_a_reveal(
    mock_env_spec: EnvFactorySpec, env_spec: EnvSpec
) -> None:
    """``Reveal.enemy_elixir`` changes where a slot's number comes from without changing any
    width, so a digest that only hashed the shapes would call the two environments the same."""
    from royalegym.env import ClashSelfPlayVecEnv, EnvFactory
    from royalegym.mock_engine import MockEngine
    from royalegym.obs import Reveal, SpatialObsBuilder

    factory = EnvFactory(
        engine=MockEngine,
        obs_builder=(SpatialObsBuilder, {"reveal": Reveal(enemy_elixir=True)}),
        decision_ms=mock_env_spec.decision_ms,
    )
    revealed_env = ClashSelfPlayVecEnv(1, factory, viser=None)
    try:
        revealed = read_env_spec(revealed_env, mock_env_spec, frame_stack=1)
    finally:
        revealed_env.close()
    assert revealed.vector_size == env_spec.vector_size
    assert revealed.obs_digest != env_spec.obs_digest


def test_reading_the_spec_resets_the_environment_first(
    mock_env_spec: EnvFactorySpec, env_spec: EnvSpec
) -> None:
    """The engine's timing is read off a live battle, so the reset comes before the read."""
    fresh = mock_env_spec.build_vec(1)
    try:
        assert fresh.envs[0].agents == []
        read = read_env_spec(fresh, mock_env_spec, frame_stack=1)
        assert fresh.envs[0].agents
    finally:
        fresh.close()
    assert read.decision_ticks == env_spec.decision_ticks
    assert read.tick_ms == env_spec.tick_ms
    assert read.regular_ticks == env_spec.regular_ticks


def test_the_spec_carries_the_factory_it_was_built_from(
    env_spec: EnvSpec, mock_env_spec: EnvFactorySpec
) -> None:
    assert env_spec.env_factory == mock_env_spec
    assert env_spec.env_factory.digest() == mock_env_spec.digest()


# --------------------------------------------------------------------------
# The allow-list, which is the whole distance between a config file and a class
# --------------------------------------------------------------------------


def test_every_allowed_prefix_ends_in_a_dot() -> None:
    # Without the dot, startswith would admit royalegymevil as well as royalegym.
    assert all(prefix.endswith(".") for prefix in ALLOWED_MODULE_PREFIXES)


@pytest.mark.parametrize(
    "path",
    ["os.system", "subprocess.run", "royalegymevil.obs.SpatialObsBuilder", "builtins.eval"],
)
def test_a_component_off_the_allow_list_is_refused(path: str) -> None:
    with pytest.raises(PreflightError) as caught:
        resolve_component(path)
    assert path in str(caught.value)


def test_an_allowed_component_resolves_to_its_class() -> None:
    from royalegym.mock_engine import MockEngine

    assert resolve_component("royalegym.mock_engine.MockEngine") is MockEngine


def test_a_module_resolves_only_once_it_is_named_in_extra_modules() -> None:
    import os

    with pytest.raises(PreflightError):
        resolve_component("os.getpid")
    assert resolve_component("os.getpid", extra_modules=("os.",)) is os.getpid


def test_an_allowed_module_that_has_no_such_attribute_is_a_preflight_error() -> None:
    with pytest.raises(PreflightError) as caught:
        resolve_component("royalegym.mock_engine.NotAClass")
    assert "NotAClass" in str(caught.value)


# --------------------------------------------------------------------------
# The vector fields the pointer head needs
# --------------------------------------------------------------------------


def test_the_hand_fields_match_the_builders_own_offsets(
    env_spec: EnvSpec, vec_env: object
) -> None:
    offsets = vec_env.envs[0].obs_builder.vector_offsets()
    resolved = obs_layout.resolve_fields(env_spec)
    for name, span in resolved.items():
        assert span == offsets[name]


@pytest.mark.parametrize("num_cards", [None, WIDER_CATALOGUE])
def test_the_fields_are_resolved_by_name_on_either_catalogue(
    env_spec: EnvSpec, num_cards: int | None
) -> None:
    from royalegym.obs import vector_offsets

    spec = env_spec if num_cards is None else synthetic_env_spec(env_spec, num_cards)
    expected = vector_offsets(spec.num_cards)
    resolved = obs_layout.resolve_fields(spec)
    assert resolved == {name: expected[name] for name in obs_layout.REQUIRED_FIELDS}


def test_the_two_catalogues_do_not_agree_on_the_offsets(env_spec: EnvSpec) -> None:
    """The check above would prove nothing if both catalogues laid the vector out alike."""
    wider = synthetic_env_spec(env_spec, WIDER_CATALOGUE)
    assert wider.vector_size != env_spec.vector_size
    assert obs_layout.field_slice(wider, obs_layout.HAND_COST) != obs_layout.field_slice(
        env_spec, obs_layout.HAND_COST
    )


@pytest.mark.parametrize("num_cards", [None, WIDER_CATALOGUE])
def test_the_hand_one_hot_unfolds_to_the_catalogue_plus_an_empty_slot(
    env_spec: EnvSpec, num_cards: int | None
) -> None:
    spec = env_spec if num_cards is None else synthetic_env_spec(env_spec, num_cards)
    fields = obs_layout.hand_fields(spec)
    assert fields.hand_size == spec.hand_size
    assert fields.onehot_width == spec.num_cards + 1
    assert fields.card_onehot.stop - fields.card_onehot.start == (
        fields.hand_size * fields.onehot_width
    )
    assert fields.cost.stop - fields.cost.start == spec.hand_size
    assert fields.affordable.stop - fields.affordable.start == spec.hand_size


def test_a_missing_field_is_a_preflight_error_naming_it(env_spec: EnvSpec) -> None:
    renamed = msgspec.structs.replace(
        env_spec,
        vector_layout=tuple(
            ("renamed_by_a_refactor", offset, size)
            if name == obs_layout.HAND_AFFORDABLE
            else (name, offset, size)
            for name, offset, size in env_spec.vector_layout
        ),
    )
    with pytest.raises(PreflightError) as excinfo:
        obs_layout.hand_fields(renamed)
    assert obs_layout.HAND_AFFORDABLE in str(excinfo.value)
    assert "renamed_by_a_refactor" in str(excinfo.value)


def test_a_hand_field_of_the_wrong_width_is_refused(env_spec: EnvSpec) -> None:
    layout = []
    offset = 0
    for name, _, size in env_spec.vector_layout:
        width = size + 1 if name == obs_layout.HAND_COST else size
        layout.append((name, offset, width))
        offset += width
    widened = msgspec.structs.replace(
        env_spec,
        vector_layout=tuple(layout),
        vector_size=offset,
        obs_space={
            **env_spec.obs_space,
            "vector": msgspec.structs.replace(env_spec.obs_space["vector"], shape=(offset,)),
        },
    )
    with pytest.raises(PreflightError, match=obs_layout.HAND_COST):
        obs_layout.hand_fields(widened)


def test_no_offset_is_computed_from_the_vector_width() -> None:
    """The module resolves names; it never does arithmetic on how wide the vector is."""
    source = inspect.getsource(obs_layout)
    assert "vector_size" not in source
    assert "num_cards" not in source.split('"""')[-1]
