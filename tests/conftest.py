"""Fixtures the suite shares: a MockEngine environment, the spec read off it, and a small
config to run things against.

Everything here is MockEngine. Its catalogue is a quarter the size of the full one, so its
observation vector is nowhere near the width of a real run's, which is the standing check that
no width is written down anywhere in the harness: a test that passes on both catalogues cannot
contain a literal from either.

The package is imported inside the fixtures rather than at module scope. A conftest is imported
before anything is collected, and a module-scope import here would load half the package into
every test session -- including the one that checks what importing ``royalelearn`` pulls in.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from royalelearn.api.rollout import EnvSpec
    from royalelearn.config import RunConfig
    from royalelearn.rollout.envspec import EnvFactorySpec


@pytest.fixture(scope="session")
def mock_env_spec() -> EnvFactorySpec:
    """The environment description the suite runs against."""
    from royalelearn.config import MOCK_ENGINE, default_env_spec

    return default_env_spec(MOCK_ENGINE)


@pytest.fixture(scope="session")
def vec_env(mock_env_spec: EnvFactorySpec) -> Iterator[Any]:
    """Two MockEngine battles, four seats. Built once: a MockEngine costs a CSV parse."""
    env = mock_env_spec.build_vec(2)
    yield env
    env.close()


@pytest.fixture(scope="session")
def env_spec(vec_env: Any, mock_env_spec: EnvFactorySpec) -> EnvSpec:
    """The spec read off that environment, at a frame stack of one."""
    from royalelearn.rollout.envspec import read_env_spec

    return read_env_spec(vec_env, mock_env_spec, frame_stack=1)


@pytest.fixture
def run_config(mock_env_spec: EnvFactorySpec) -> RunConfig:
    """A config small enough to build in a test: one worker, two battles, a tiny network.

    ``timesteps_per_iteration`` is set so that the rectangle is a handful of cycles rather than
    a few hundred, and the network is the smallest the architecture allows; nothing else moves,
    so a test that reads a default still reads the shipped one.
    """
    from royalelearn import config

    return config.RunConfig(
        run_name="test",
        env=mock_env_spec,
        rollout=config.RolloutConfig(
            source="inline", workers=1, games_per_worker=2, shards_per_worker=1
        ),
        net=config.NetConfig(
            channels=8, blocks=1, norm_groups=4, card_embed=8, value_hidden=16, device="cpu"
        ),
        ppo=config.PPOConfig(
            n_epochs=2, timesteps_per_iteration=24, batch_size=16, minibatch_size=8
        ),
        determinism=config.DeterminismConfig(tier="throughput"),
    )


@pytest.fixture
def torch() -> Any:
    """The learner's torch, or a skip. Nothing in ``api/`` needs it; the networks do."""
    return pytest.importorskip("torch")


def synthetic_env_spec(env_spec: EnvSpec, num_cards: int) -> EnvSpec:
    """The same environment with a catalogue of another size, without building one.

    The widths come from ``royalegym.obs.vector_layout`` -- the definition of the layout -- so
    the result is what an environment on that catalogue would report rather than an arithmetic
    guess at it, and a test can run against two catalogues whose widths differ without either
    number appearing in this repository.
    """
    import msgspec

    from royalegym.obs import vector_layout

    offset = 0
    layout: list[tuple[str, int, int]] = []
    for field in vector_layout(num_cards):
        layout.append((field.key, offset, int(field.size)))
        offset += int(field.size)
    vector = msgspec.structs.replace(env_spec.obs_space["vector"], shape=(offset,))
    return msgspec.structs.replace(
        env_spec,
        num_cards=num_cards,
        obs_space={**env_spec.obs_space, "vector": vector},
        vector_layout=tuple(layout),
        vector_size=offset,
    )
