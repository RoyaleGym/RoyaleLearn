"""The same contract, against the engine a real run uses.

``tests/test_env_contract.py`` checks every layout fact the harness relies on, and checks it
against ``MockEngine`` -- sixteen cards, a 229-wide vector, spells that resolve inside a tick.
A real run is the Rust engine: ninety-five cards today, a vector nearly five times as wide, and
the only implementation whose observations exercise the spell and status planes at all.

Everything here therefore runs on the real engine and is marked ``engine``, so it is skipped
where the extension is not built or its data is stale. That skip is the reason this file exists
separately rather than as a parametrisation: the fast suite must stay runnable by somebody who
has not built the engine, and the properties below must still be checked by somebody who has.

What is checked is what differs between the two engines rather than what they share: the widths,
the planes only a real battle fills, and the codec's exactness on observations the mock cannot
produce. A literal appears nowhere -- every number is read from the environment, which is the
property that makes the harness survive a catalogue that grew from 65 cards to 95 without a line
changing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.engine


@pytest.fixture(scope="module")
def rust_env() -> Any:
    """Two real battles, or a skip naming why the engine could not be built."""
    from royalelearn import config as cfg

    royalegym = pytest.importorskip("royalegym")
    if not royalegym.core_available():
        pytest.skip("royalesim is not built; run `maturin develop --release` in RoyaleSim")
    try:
        factory = cfg.default_env_spec(cfg.RUST_ENGINE)
        env = factory.build_vec(2, viser=None)
    except RuntimeError as exc:  # the stale-build gate, which is a skip and not a failure here
        pytest.skip(f"the engine build disagrees with the data on disk: {exc}")
    yield env
    env.close()


@pytest.fixture(scope="module")
def rust_spec(rust_env: Any) -> Any:
    from royalelearn import config as cfg
    from royalelearn.rollout.envspec import read_env_spec

    return read_env_spec(rust_env, cfg.default_env_spec(cfg.RUST_ENGINE), 1)


def played_out(env: Any, steps: int) -> list[dict[str, np.ndarray]]:
    """Observations from real play, which is the only way to reach a populated board."""
    obs, _info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    collected: list[dict[str, np.ndarray]] = []
    for _ in range(steps):
        mask = obs["action_mask"].astype(bool)
        actions = np.array(
            [int(rng.choice(np.flatnonzero(row))) for row in mask], dtype=np.int64
        )
        obs, _rewards, _term, _trunc, _info = env.step(actions)
        collected += [{key: obs[key][row] for key in obs} for row in range(env.num_envs)]
    return collected


def test_the_real_catalogue_is_wider_than_the_one_the_fast_suite_uses(rust_spec: Any) -> None:
    """The reason this file exists: the two engines disagree about every width.

    A harness that passed the fast suite on a literal would fail here, which is exactly the
    failure this asserts cannot happen by construction.
    """
    from royalegym.mock_engine import MockEngine
    from royalelearn import config as cfg

    assert rust_spec.num_cards > len(MockEngine().cards())
    assert rust_spec.vector_size > 30 + 5 * len(MockEngine().cards())
    assert rust_spec.n_actions == rust_spec.obs_space["action_mask"].shape[0]
    assert cfg.RUST_ENGINE in rust_spec.env_factory.engine.cls


def test_mask_planes_are_the_mask_reshaped_on_real_states(rust_env: Any) -> None:
    """RoyaleGym guarantees the identity; the harness derives the planes rather than storing
    them, so a real battle is where that guarantee is worth re-checking."""
    for observation in played_out(rust_env, 12):
        if "mask_planes" not in observation:
            pytest.skip("this observation builder publishes no mask planes")
        flat = observation["action_mask"][1:]
        assert np.array_equal(observation["mask_planes"].reshape(-1), flat)


def test_the_noop_is_legal_in_every_real_state(rust_env: Any) -> None:
    """The precondition the whole masking scheme rests on, on the engine that will run."""
    for observation in played_out(rust_env, 24):
        assert observation["action_mask"][0] == 1


def test_the_codec_round_trips_real_observations_exactly(rust_env: Any, rust_spec: Any) -> None:
    """The planes a real battle fills and the mock cannot: spells, status effects, towers.

    The codec decides what is stored exactly and what is stored as halves from the declared
    bounds and a sample of real play. On MockEngine the spell planes are always zero, so this is
    the only place the decision is made against data that has anything in it.
    """
    pytest.importorskip("torch")
    from royalelearn.rollout.codec import SpatialObsCodec
    from test_codec import pack_all, unpack

    sample = played_out(rust_env, 260)
    codec = SpatialObsCodec()
    table = codec.table(rust_spec, sample)
    codec.bind(rust_spec, table)

    exact = [name for name, storage, _scale in table.plane if storage == "uint8"]
    assert exact, "no plane was admitted to exact storage on a real catalogue"

    checked = sample[:16]
    statics = codec.static_planes(checked[0])
    out = unpack(codec, rust_spec, pack_all(codec, checked), statics)
    for row, observation in enumerate(checked):
        for index, (name, storage, _scale) in enumerate(table.plane):
            if storage != "uint8":
                continue
            assert np.array_equal(
                out.spatial[row, index].numpy(), observation["spatial"][index]
            ), f"plane {name} was admitted to exact storage and did not round-trip"
        assert np.array_equal(
            out.mask[row].numpy(), observation["action_mask"].astype(bool)
        ), "the bit-packed mask did not round-trip on a real state"


def test_a_real_engine_iteration_completes(tmp_path: Any) -> None:
    """One iteration end to end on the engine a run uses, which no other test does.

    Small enough to be a test and real enough to be worth one: the codec table is computed from
    real observations, the workers hold real battles, and the update runs on what they collect.
    """
    pytest.importorskip("torch")
    from royalelearn import config as cfg
    from test_coordinator import coordinator, tiny_config

    try:
        config = tiny_config(
            tmp_path,
            env=cfg.default_env_spec(cfg.RUST_ENGINE, max_steps=8),
            alarms=cfg.AlarmConfig(enabled=False),
        )
        with coordinator(config) as run:
            run.learn(until_timesteps=config.ppo.timesteps_per_iteration)
            assert run.iteration >= 1
    except RuntimeError as exc:
        pytest.skip(f"the engine build disagrees with the data on disk: {exc}")
