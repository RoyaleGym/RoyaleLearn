"""``net.critic_channels`` and ``net.critic_blocks``: a critic trunk of its own width and depth.

With separate trunks the critic is a whole second trunk, and under ``ppo.forced_rows``
"critic_only" it runs on every row while the actor runs only on the rows with a choice, so most
of an update's compute can be the critic's. These set its trunk, and its value head, apart from
the actor's. Unset, the critic is the actor's size, as it always was, and neither field is in
the config's encoding or the architecture's digest.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from royalelearn.errors import PreflightError

torch = pytest.importorskip("torch")

from royalelearn import config as cfg  # noqa: E402
from royalelearn.learn.nets import DefaultNetworkFactory  # noqa: E402
from royalelearn.testing import ARCH  # noqa: E402


def _arch(**fields: Any) -> Any:
    return msgspec.structs.replace(ARCH, separate_trunks=True, **fields)


def _sizes(model: Any) -> tuple[int, int]:
    """Parameters in the actor's trunk and in the critic's."""
    actor = sum(p.numel() for p in model.actor.trunk.parameters())
    critic = sum(p.numel() for p in model.critic.trunk.parameters())
    return actor, critic


def test_unset_the_critic_is_the_actors_size_and_nothing_is_encoded() -> None:
    """Plant: give either field a plain default, and every config hash and digest moves."""
    assert b"critic_" not in msgspec.json.encode(cfg.NetConfig())
    assert b"critic_" not in msgspec.json.encode(ARCH)


def test_the_critic_trunk_takes_its_own_width_and_depth(env_spec: Any) -> None:
    """Plant: build the critic from the actor's arch, and both trunks come out the same size."""
    narrow = _arch(critic_channels=ARCH.channels // 2, critic_blocks=0)
    model = DefaultNetworkFactory(7).build(env_spec, narrow, "cpu")
    same = DefaultNetworkFactory(7).build(env_spec, _arch(), "cpu")
    assert _sizes(same)[0] == _sizes(same)[1]
    assert _sizes(model)[0] == _sizes(same)[0]
    assert _sizes(model)[1] < _sizes(same)[1] // 2
    assert model.critic.trunk.channels == ARCH.channels // 2
    assert len(model.critic.trunk.body) == 0


def test_a_narrower_critic_is_another_architecture(env_spec: Any) -> None:
    factory = DefaultNetworkFactory(7)
    plain = factory.arch_digest(env_spec, _arch())
    assert factory.arch_digest(env_spec, _arch(critic_channels=ARCH.channels // 2)) != plain
    assert factory.arch_digest(env_spec, _arch(critic_blocks=0)) != plain


@pytest.mark.parametrize(
    ("fields", "names"),
    [
        ({"separate_trunks": False, "critic_channels": 4}, "net.separate_trunks"),
        ({"critic_channels": 6}, "net.critic_channels"),  # norm_groups 4 does not divide it
        ({"critic_channels": 0}, "net.critic_channels"),
        ({"critic_blocks": -1}, "net.critic_blocks"),
    ],
)
def test_a_critic_shape_that_cannot_be_built_is_refused_by_name(
    env_spec: Any, fields: dict[str, Any], names: str
) -> None:
    arch = msgspec.structs.replace(ARCH, **{"separate_trunks": True, **fields})
    with pytest.raises(PreflightError, match=names.replace(".", r"\.")):
        DefaultNetworkFactory(7).build(env_spec, arch, "cpu")


def test_a_run_with_a_narrower_critic_trains_and_resumes(tmp_path: Any) -> None:
    """One real iteration, a checkpoint, and a second process's worth of carrying on."""
    from royalelearn.testing import coordinator, tiny_config

    base = tiny_config(tmp_path)
    config = tiny_config(
        tmp_path,
        net=msgspec.structs.replace(base.net, critic_channels=4, critic_blocks=0),
        checkpoint=cfg.CheckpointConfig(every_env_steps=1, keep=2),
    )
    with coordinator(config) as run:
        run.iterate()
        assert run.model.critic.trunk.channels == 4
        run_dir = run.run_dir
    from royalelearn.checkpoint import DirCheckpointStore

    latest = DirCheckpointStore(run_dir).latest()
    assert latest is not None
    with coordinator(config, resume=latest, run_dir=run_dir) as resumed:
        resumed.iterate()
        assert resumed.model.critic.trunk.channels == 4
