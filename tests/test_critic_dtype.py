"""``net.critic_autocast_dtype``: the critic's trunk at its own precision, its value head at
float32.

The critic never enters the importance ratio, so a run that keeps its actor at float32 for the
ratio invariant can train the critic -- most of an update under ``forced_rows`` "critic_only" --
on the bfloat16 path. The actor is untouched, and so are its digests: a precision is not a shape.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

from royalelearn import config as cfg  # noqa: E402
from royalelearn.errors import PreflightError  # noqa: E402
from royalelearn.learn.nets import DefaultNetworkFactory  # noqa: E402
from royalelearn.testing import ARCH  # noqa: E402
from test_float32_head import _obs  # noqa: E402


def _model(spec: Any, **fields: Any) -> Any:
    arch = msgspec.structs.replace(ARCH, autocast_dtype="float32", **fields)
    return DefaultNetworkFactory(11).build(spec, arch, "cpu")


def test_unset_nothing_is_encoded() -> None:
    assert b"critic_autocast_dtype" not in msgspec.json.encode(cfg.NetConfig())
    assert cfg.critic_autocast_dtype_of(cfg.NetConfig()) is None


def test_the_critic_trunk_runs_at_its_dtype_and_the_value_head_at_float32(env_spec: Any) -> None:
    """Plant: leave the head under the autocast, and its input is bfloat16; ignore the field,
    and the trunk's output is float32."""
    model = _model(env_spec, critic_autocast_dtype="bfloat16")
    seen: dict[str, torch.dtype] = {}
    model.critic.trunk.register_forward_hook(
        lambda _m, _i, out: seen.__setitem__("trunk", out.dtype)
    )
    model.critic.head.register_forward_pre_hook(
        lambda _m, args: seen.__setitem__("head_in", args[0].dtype)
    )
    with torch.no_grad():
        values = model.value(_obs(env_spec))
    assert seen == {"trunk": torch.bfloat16, "head_in": torch.float32}
    assert values.dtype == torch.float32


def test_the_actor_and_the_digests_are_untouched(env_spec: Any) -> None:
    plain = _model(env_spec)
    critic_bf16 = _model(env_spec, critic_autocast_dtype="bfloat16")
    obs = _obs(env_spec)
    with torch.no_grad():
        assert torch.equal(plain.actor.logits(obs), critic_bf16.actor.logits(obs))
        near = plain.value(obs)
        far = critic_bf16.value(obs)
    assert not torch.equal(near, far)
    assert torch.allclose(near, far, atol=0.05, rtol=0.05)
    assert plain.arch_digest == critic_bf16.arch_digest


def test_with_one_trunk_it_is_refused(env_spec: Any) -> None:
    with pytest.raises(PreflightError, match=r"needs net\.separate_trunks"):
        _model(env_spec, critic_autocast_dtype="bfloat16", separate_trunks=False)


def test_a_run_with_a_bfloat16_critic_keeps_the_ratio_exact(tmp_path: Any) -> None:
    """The ratio is the actor's alone, so it stays exactly one at the first epoch."""
    from royalelearn.testing import coordinator, tiny_config

    base = tiny_config(tmp_path)
    config = tiny_config(
        tmp_path, net=msgspec.structs.replace(base.net, critic_autocast_dtype="bfloat16")
    )
    with coordinator(config) as run:
        assert run.model.critic.head_float32
        for _ in range(2):
            run.iterate()
        assert all(row["ppo/ratio_max_abs_dev"] == 0.0 for row in run.rows)



def test_this_module_runs_without_onednn() -> None:
    """conftest's guard: a module that runs bfloat16 on the processor runs without oneDNN."""
    assert torch.backends.mkldnn.enabled is False


#: Runs bfloat16 on the processor, so conftest runs it without oneDNN (see there why).
CPU_BFLOAT16 = True
