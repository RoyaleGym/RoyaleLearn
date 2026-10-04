"""``net.trunk_stride`` 2: the residual body on a quarter of the positions, the heads on every tile.

The stem keeps the board's resolution; a strided convolution takes it to half in each direction,
the body runs there, and its output is upsampled and added back to the stem's. The trunk's output
is the shape it always was, so the heads are untouched and the pointer head still reads one
feature per tile. At 1 the network, its draws and its digest are what they were before the field.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from royalelearn.errors import PreflightError
from test_nets import MASTER_SEED, _build, arch, fake_obs


def _digest(spec: Any, config: Any) -> str:
    from royalelearn.learn.nets import DefaultNetworkFactory

    return DefaultNetworkFactory(MASTER_SEED).arch_digest(spec, config)


def test_the_trunk_output_keeps_the_board_and_the_body_runs_at_half(
    torch: Any, env_spec: Any
) -> None:
    model = _build(env_spec, arch(trunk_stride=2, blocks=2))
    trunk = model.actor.trunk
    seen: list[tuple[int, ...]] = []
    hooks = [
        block.register_forward_hook(lambda _m, _i, out: seen.append(tuple(out.shape[2:])))
        for block in trunk.body
    ]
    obs = fake_obs(torch, env_spec, batch=3, seed=1)
    features = trunk(obs)
    for hook in hooks:
        hook.remove()
    height, width = env_spec.tiles
    assert tuple(features.shape) == (3, trunk.channels, height, width)
    assert seen == [(height // 2, width // 2)] * 2
    logits = model.actor.head(features, obs)
    assert logits.shape == (3, env_spec.n_actions)


def test_at_stride_one_nothing_moves(torch: Any, env_spec: Any) -> None:
    """The digest, the parameters and the draws of the network every save was made with."""
    from royalelearn.config import NetConfig

    plain, explicit = arch(), arch(trunk_stride=1)
    assert _digest(env_spec, plain) == _digest(env_spec, explicit)
    a, b = _build(env_spec, plain), _build(env_spec, explicit)
    assert a.actor.trunk.down is None
    for (name, x), (_, y) in zip(
        a.state_dict().items(), b.state_dict().items(), strict=True
    ):
        assert torch.equal(x, y), name
    assert "trunk_stride" not in msgspec.json.encode(
        _arch_struct(NetConfig(trunk_stride=1))
    ).decode()


def _arch_struct(config: Any) -> Any:
    from royalelearn.learn.nets import _arch_for_digest

    return _arch_for_digest(config)


def test_stride_two_is_another_architecture_with_one_more_conv(torch: Any, env_spec: Any) -> None:
    one, two = arch(), arch(trunk_stride=2)
    assert _digest(env_spec, one) != _digest(env_spec, two)
    count = lambda model: sum(p.numel() for p in model.actor.trunk.parameters())  # noqa: E731
    channels = one.channels
    down = channels * channels * 9 + channels + 2 * channels  # the conv, its bias, its norm
    assert count(_build(env_spec, two)) == count(_build(env_spec, one)) + down


def test_the_body_learns_through_the_upsampling(torch: Any, env_spec: Any) -> None:
    model = _build(env_spec, arch(trunk_stride=2))
    obs = fake_obs(torch, env_spec, batch=4, seed=2)
    logits = model.actor.head(model.actor.trunk(obs), obs)
    logits.masked_fill(~obs.mask, 0.0).sum().backward()
    for name, parameter in model.actor.trunk.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("stride", [0, 3, 4])
def test_a_stride_other_than_one_or_two_is_refused(env_spec: Any, stride: int) -> None:
    with pytest.raises(PreflightError, match="trunk_stride"):
        _build(env_spec, arch(trunk_stride=stride))


def test_a_real_iteration_trains_at_stride_two(tmp_path: Any) -> None:
    from test_coordinator import coordinator, tiny_config

    config = tiny_config(tmp_path)
    config = msgspec.structs.replace(
        config, net=msgspec.structs.replace(config.net, trunk_stride=2)
    )
    with coordinator(config) as run:
        assert run.model.actor.trunk.down is not None
        run.iterate()
