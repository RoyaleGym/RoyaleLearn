"""``net.policy_head_float32``: the policy head in float32 while the trunk runs under autocast.

Under bfloat16 a logit is rounded to bfloat16's spacing, a thirty-second just under 8, and the
rollout's forward and the update's do not always round it the same way, because their batches
differ. An error in the largest logit reaches every action's log-probability, scaled by that
action's probability. That is the importance-ratio noise the preflight guard predicts and
refuses at ``noop_bias`` 8. With this set, the trunk keeps bfloat16, where nearly all the
compute is, and the head computes the logits from its features in float32.

Unset, the head runs at the autocast dtype as before, and nothing is encoded or digested.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

from royalelearn import config as cfg  # noqa: E402
from royalelearn.learn.nets import DefaultNetworkFactory  # noqa: E402
from royalelearn.testing import ARCH  # noqa: E402


def _obs(spec: Any, rows: int = 16, seed: int = 5) -> Any:
    from royalelearn.api.policy import ObsBatch

    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, spec.n_actions), generator=generator) < 0.05
    mask[:, 0] = True
    grid = spec.n_grid_actions
    return ObsBatch(
        spatial=torch.rand(
            (rows, spec.frame_stack * spec.spatial_shape[0], *spec.tiles), generator=generator
        ),
        vector=torch.rand((rows, spec.vector_size), generator=generator),
        mask=mask,
        mask_planes=mask[:, 1:grid].float().view(rows, spec.hand_size, *spec.tiles),
    )


def _actor(spec: Any, **fields: Any) -> Any:
    arch = msgspec.structs.replace(ARCH, autocast_dtype="bfloat16", noop_bias=8.0, **fields)
    return DefaultNetworkFactory(11).build(spec, arch, "cpu").actor


def _on_grid(values: Any, spacing: float) -> bool:
    scaled = values / spacing
    return bool(torch.equal(scaled, torch.round(scaled)))


def test_unset_nothing_is_encoded() -> None:
    assert b"policy_head_float32" not in msgspec.json.encode(cfg.NetConfig())


def test_the_head_computes_its_logits_in_float32(env_spec: Any) -> None:
    """At a no-op logit near 8, bfloat16 can only say multiples of 1/32 (1/16 from 8 up).
    Plant: ignore the flag, and the logits come back on that grid."""
    obs = _obs(env_spec)
    with torch.no_grad():
        rounded = _actor(env_spec)(obs)[:, 0]
        exact = _actor(env_spec, policy_head_float32=True)(obs)[:, 0]
    assert rounded.dtype == torch.float32
    assert float(rounded.min()) > 6.0 and float(rounded.max()) < 16.0
    assert _on_grid(rounded, 1 / 32)
    assert not _on_grid(exact, 1 / 32)


def test_the_float32_head_reads_the_trunks_own_features(env_spec: Any) -> None:
    """The trunk is untouched: the head's logits are the float32 head over the trunk's autocast
    features, exactly. Plant: run the trunk in float32 too, and they differ."""
    actor = _actor(env_spec, policy_head_float32=True)
    obs = _obs(env_spec)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        features = actor.trunk(obs)
    assert features.dtype == torch.bfloat16
    expected = actor.head(features.float(), obs)
    assert torch.equal(actor(obs), expected)


def test_the_ratio_guard_predicts_from_the_heads_precision(env_spec: Any) -> None:
    """noop_bias 8 under bfloat16 is refused before a run starts; with the head in float32 the
    logits carry float32's spacing, and it is not."""
    from royalelearn.errors import PreflightError
    from royalelearn.rollout.preflight import _ratio_precision_gate, logit_precision_name

    def config(**fields: Any) -> Any:
        net = msgspec.structs.replace(ARCH, autocast_dtype="bfloat16", noop_bias=8.0, **fields)
        return cfg.RunConfig(net=net)

    lines: list[str] = []
    assert logit_precision_name(config()) == "bfloat16"
    with pytest.raises(PreflightError, match="policy_head_float32"):
        _ratio_precision_gate(config(), env_spec, lines.append)
    assert logit_precision_name(config(policy_head_float32=True)) == "float32"
    _ratio_precision_gate(config(policy_head_float32=True), env_spec, lines.append)
    assert any("float32" in line for line in lines)


def test_a_float32_head_is_another_architecture(env_spec: Any) -> None:
    factory = DefaultNetworkFactory(11)
    plain = msgspec.structs.replace(ARCH, autocast_dtype="bfloat16")
    assert factory.arch_digest(env_spec, plain) != factory.arch_digest(
        env_spec, msgspec.structs.replace(plain, policy_head_float32=True)
    )
