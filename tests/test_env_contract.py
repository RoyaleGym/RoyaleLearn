"""Every layout fact the harness relies on, checked against the environment itself.

This is the cheapest file in the suite and the most valuable one. The harness holds no width,
no plane count and no field offset of its own: it reads all of them from RoyaleGym at start-up,
which means a change there is not a failure here unless it is a change to something the harness
*relies* on. Each assertion below names the file it depends on, so that when one of them does
change, what breaks is this file, loudly, with the dependency written next to it -- instead of a
training run that quietly optimises the wrong index.

Nothing here asserts a number. The catalogue under these tests is MockEngine's, which is a
quarter the size of a real run's, so an assertion that passed on a literal from either would
fail on the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture(scope="module")
def short_env() -> Any:
    """Two battles that end after a handful of decisions, so episode ends are cheap to reach.

    Its own env rather than the session's: these tests step it, and a stepped environment is
    not the one another test read its spec off.
    """
    from royalelearn import config as cfg

    factory = cfg.default_env_spec(cfg.MOCK_ENGINE, max_steps=3)
    env = factory.build_vec(2, viser=None)
    yield env
    env.close()


# -- royalegym/action.py ----------------------------------------------------


def test_action_space_is_the_grid_plus_the_noop(env_spec: Any, vec_env: Any) -> None:
    """royalegym/action.py: Discrete(1 + hand_size * tiles_y * tiles_x), index 0 the no-op."""
    parser = vec_env.envs[0].action_parser
    tiles_y, tiles_x = env_spec.tiles
    assert int(parser.space.n) == 1 + env_spec.hand_size * tiles_y * tiles_x
    assert int(parser.noop()) == 0
    assert parser.parse(parser.noop(), vec_env.envs[0].engine.state(), 0) is None


def test_encode_is_the_c_order_index(env_spec: Any, vec_env: Any) -> None:
    """royalegym/action.py: a one-hot over (hand, y, x) read in C order encodes to that action.

    Exhaustive over every non-no-op action, because this is the identity the pointer head's
    output and the environment's input agree on, and a transposition in it is a policy that
    plays a different card somewhere else from the one it learned.
    """
    parser = vec_env.envs[0].action_parser
    tiles_y, tiles_x = env_spec.tiles
    shape = (env_spec.hand_size, tiles_y, tiles_x)
    slots, ys, xs = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    expected = np.ravel_multi_index((slots, ys, xs), shape).reshape(-1) + 1
    encoded = np.array(
        [
            parser.encode(int(s), int(x), int(y))
            for s, y, x in zip(slots.reshape(-1), ys.reshape(-1), xs.reshape(-1), strict=True)
        ]
    )
    assert np.array_equal(encoded, expected)
    assert encoded.size == env_spec.n_actions - 1


def test_decode_inverts_encode(env_spec: Any, vec_env: Any) -> None:
    """royalegym/action.py: the parser's own round trip, over the whole space."""
    parser = vec_env.envs[0].action_parser
    for action in range(1, env_spec.n_actions):
        slot, x, y = parser.decode(action)
        assert parser.encode(slot, x, y) == action


# -- royalegym/obs.py -------------------------------------------------------


def test_mask_planes_are_the_mask_reshaped(env_spec: Any, short_env: Any) -> None:
    """royalegym/obs.py: mask_planes == action_mask[1:].reshape(hand, tiles_y, tiles_x).

    The codec never stores the planes because of this: they are the mask it already stores,
    and the learner reshapes them at unpack.
    """
    obs, _ = short_env.reset(seed=11)
    assert "mask_planes" in obs
    shape = (env_spec.hand_size, *env_spec.tiles)
    for row in range(short_env.num_envs):
        assert np.array_equal(obs["mask_planes"][row], obs["action_mask"][row][1:].reshape(shape))


def test_vector_layout_is_contiguous_and_complete(env_spec: Any, vec_env: Any) -> None:
    """royalegym/obs.py: vector_layout() covers the whole vector with no gap and no overlap."""
    offset = 0
    for name, start, size in env_spec.vector_layout:
        assert start == offset, f"vector field {name} starts at {start}, not {offset}"
        assert size > 0
        offset += size
    assert offset == env_spec.vector_size == vec_env.single_observation_space["vector"].shape[0]


def test_vector_layout_declares_the_hand_fields(env_spec: Any) -> None:
    """royalegym/obs.py: the three fields the pointer head resolves by name are all there."""
    from royalelearn.obs_layout import REQUIRED_FIELDS, hand_fields

    declared = {name for name, _, _ in env_spec.vector_layout}
    assert set(REQUIRED_FIELDS) <= declared
    hand = hand_fields(env_spec)
    assert hand.hand_size == env_spec.hand_size
    assert hand.onehot_width * hand.hand_size == hand.card_onehot.stop - hand.card_onehot.start


def test_spatial_layout_matches_the_space(env_spec: Any) -> None:
    """royalegym/obs.py: one (name, static) entry per plane, and at least one static plane.

    A static plane is one the LAYOUT declares static. The codec holds those once per seat and
    stores every other plane per row, including the ones that happen not to move: a tower plane
    is constant in any sample in which no tower falls.
    """
    assert len(env_spec.spatial_layout) == env_spec.spatial_shape[0]
    names = [name for name, _ in env_spec.spatial_layout]
    assert len(set(names)) == len(names)
    assert env_spec.static_planes, "no plane is declared static; the codec stores every row"


def test_spatial_bounds_are_per_channel(env_spec: Any, vec_env: Any) -> None:
    """royalegym/obs.py: the space's bounds reach every cell, and a plane declares one bound.

    The codec decides a plane's storage from its declared bound, so what it needs is one number
    per plane; a bound that varied within a plane would not be a bound it could use.
    """
    space = vec_env.single_observation_space["spatial"]
    high = np.broadcast_to(space.high, env_spec.spatial_shape)
    low = np.broadcast_to(space.low, env_spec.spatial_shape)
    for plane in range(env_spec.spatial_shape[0]):
        assert np.ptp(high[plane]) == 0
        assert np.ptp(low[plane]) == 0
    assert len(env_spec.obs_space["spatial"].high) == env_spec.spatial_shape[0]


def test_vector_is_bounded_in_the_unit_interval(env_spec: Any) -> None:
    """royalegym/obs.py: the builder clips the vector, which is why the codec stores fp16."""
    vector = env_spec.obs_space["vector"]
    assert vector.low == (0.0,)
    assert vector.high == (1.0,)


# -- royalegym/env.py -------------------------------------------------------


def test_autoreset_is_same_step_with_a_final_observation(short_env: Any) -> None:
    """royalegym/env.py: SAME_STEP, so the observation after an end is the next episode's.

    It is why a truncated transition needs its final observation carried separately: what the
    rectangle holds at that cell is already the next battle.
    """
    from gymnasium.vector import AutoresetMode

    assert short_env.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    short_env.reset(seed=5)
    infos = _step_until_end(short_env)
    assert "final_obs" in infos and "_final_obs" in infos
    ended = np.flatnonzero(np.asarray(infos["_final_obs"], dtype=bool))
    assert ended.size
    for row in ended:
        assert set(infos["final_obs"][row]) == set(short_env.single_observation_space.spaces)


def test_terminal_statistics_arrive_under_final_info(short_env: Any) -> None:
    """royalegym/env.py: EPISODE_STAT_KEYS in final_info, with a validity mask per key.

    The worker copies them rather than recomputing them, and the set is whatever
    EPISODE_STAT_KEYS says it is -- a key added there arrives in the episode record unchanged.
    """
    from royalegym.env import EPISODE_STAT_KEYS

    short_env.reset(seed=6)
    infos = _step_until_end(short_env)
    final = infos["final_info"]
    for key in EPISODE_STAT_KEYS:
        assert key in final, f"{key} is in EPISODE_STAT_KEYS and not in final_info"
        assert f"_{key}" in final, f"{key} has no validity mask in final_info"
        assert len(final[key]) == short_env.num_envs
    assert np.asarray(final["_episode_steps"], dtype=bool).any()


def test_tick_is_in_every_info_and_survives_into_final_info(short_env: Any) -> None:
    """royalegym/env.py: info["tick"] every step, and in final_info on the last one."""
    _, infos = short_env.reset(seed=7)
    assert "tick" in infos
    infos = _step_until_end(short_env)
    assert "tick" in infos
    assert "tick" in infos["final_info"]
    assert "deploy_status" in infos


def test_episodes_end_in_pairs(short_env: Any) -> None:
    """royalegym/env.py: a battle's two seats end together, because it is one battle."""
    short_env.reset(seed=8)
    infos = _step_until_end(short_env)
    ended = np.asarray(infos["_final_info"], dtype=bool)
    for game in range(short_env.num_games):
        assert ended[2 * game] == ended[2 * game + 1]


def test_mask_allows_the_noop_in_every_state(short_env: Any) -> None:
    """royalegym/action.py: the no-op is always legal, including in a finished battle.

    Every masked distribution in the harness rests on it: a state with no legal action is a
    distribution over nothing, and what that produces is a NaN.
    """
    noop = int(short_env.envs[0].action_parser.noop())
    obs, _ = short_env.reset(seed=9)
    for _ in range(12):
        assert (obs["action_mask"][:, noop] == 1).all()
        obs, _, _, _, infos = short_env.step(np.zeros(short_env.num_envs, dtype=np.int64))
        finals = infos.get("final_obs")
        if finals is not None:
            for row in np.flatnonzero(np.asarray(infos["_final_obs"], dtype=bool)):
                assert finals[row]["action_mask"][noop] == 1


def test_config_reports_every_documented_key(short_env: Any) -> None:
    """royalegym/env.py: config() is what the identity, the ladder context and preflight read."""
    short_env.reset(seed=10)
    config = short_env.envs[0].config()
    for key in (
        "env",
        "decision_ms",
        "decision_ticks",
        "reveal",
        "engine",
        "obs_builder",
        "action_parser",
        "reward_fn",
        "termination_cond",
        "truncation_cond",
        "state_mutator",
        "calibration_digest",
        "build_digest",
    ):
        assert key in config, f"config() no longer reports {key!r}"
    assert set(config["engine"]) >= {"class", "params"}
    assert config["decision_ticks"] >= 1


def test_the_env_factory_recipe_builds_a_vec_env(env_spec: Any, mock_env_spec: Any) -> None:
    """royalegym/env.py: EnvFactory takes class-and-kwargs recipes and pickles.

    It is how a worker is handed an environment: a description crosses the spawn boundary, not
    a built env and not a closure.
    """
    import pickle

    factory = mock_env_spec.factory()
    pickle.dumps(factory)
    env = factory()
    try:
        assert int(env.action_space("blue").n) == env_spec.n_actions
    finally:
        env.close()


def test_a_spec_with_no_truncation_builds_an_env(mock_env_spec: Any) -> None:
    """royalegym/env.py: EnvFactory keeps every component key it is handed and builds it.

    So a condition that is turned off has to be left out of the recipe rather than passed as
    None, and an environment with no truncation is exactly the shape evaluation asks for: a full
    match under real rules, with no step cap to cut it short.
    """
    import msgspec

    spec = msgspec.structs.replace(mock_env_spec, truncation=[])
    env = spec.build_vec(1, viser=None)
    try:
        env.reset(seed=3)
        assert env.envs[0].config()["truncation_cond"] is None
        for _ in range(8):
            _, _, _, truncated, _ = env.step(np.zeros(env.num_envs, dtype=np.int64))
            assert not np.any(truncated)
    finally:
        env.close()


def _step_until_end(vec: Any, limit: int = 16) -> dict[str, Any]:
    """Step with no-ops until an episode ends, and hand back that step's infos."""
    for _ in range(limit):
        _, _, terminated, truncated, infos = vec.step(np.zeros(vec.num_envs, dtype=np.int64))
        if np.any(terminated) or np.any(truncated):
            return infos
    raise AssertionError(f"no episode ended within {limit} steps of a step-limited env")
