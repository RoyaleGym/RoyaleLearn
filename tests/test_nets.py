"""The networks: the shapes they are forced into, the gradients that must not exist, and the
digest that stops a load from being a shape error.

Everything here is computed from the ``EnvSpec``. The suite runs the same assertions against
MockEngine's catalogue and against a wider one whose observation vector is several times as long,
so an assertion that happened to contain a width from either would fail on the other.
"""

from __future__ import annotations

import math
from typing import Any

import msgspec
import pytest

from conftest import synthetic_env_spec
from royalelearn.api.policy import ObsBatch
from royalelearn.config import NetConfig
from royalelearn.errors import PreflightError
from royalelearn.obs_layout import hand_fields

#: A catalogue several times MockEngine's, so that no width can be shared between the two cases.
WIDER_CATALOGUE = 65
#: The smallest network the architecture allows, which is what every test that does not measure
#: parameter counts is built with.
SMALL = {"channels": 8, "blocks": 1, "norm_groups": 4, "card_embed": 8, "value_hidden": 16}
MASTER_SEED = 4242


def arch(**overrides: Any) -> NetConfig:
    return NetConfig(**{**SMALL, "device": "cpu", "autocast_dtype": "float32", **overrides})


def variants(env_spec: Any) -> dict[str, Any]:
    """The four environments every shape assertion runs against."""
    wider = synthetic_env_spec(env_spec, WIDER_CATALOGUE)
    return {
        "mock/k1": env_spec,
        "mock/k2": msgspec.structs.replace(env_spec, frame_stack=2),
        "wide/k1": wider,
        "wide/k2": msgspec.structs.replace(wider, frame_stack=2),
    }


def _build(spec: Any, config: NetConfig, seed: int = MASTER_SEED) -> Any:
    """A fresh pair every time: a test that backpropagates must not inherit another's
    gradients."""
    from royalelearn.learn.nets import DefaultNetworkFactory

    return DefaultNetworkFactory(seed).build(spec, config, "cpu")


def fake_obs(torch: Any, spec: Any, batch: int, seed: int, density: float = 0.3) -> ObsBatch:
    """One batch shaped exactly as the codec delivers it, with a mask the environment could
    produce: the no-op always legal, the planes the flat mask reshaped."""
    generator = torch.Generator().manual_seed(seed)
    slots = spec.obs_space["mask_planes"].shape[0]
    height, width = spec.tiles
    stack = spec.frame_stack
    mask = torch.rand(batch, spec.n_actions, generator=generator) < density
    mask[:, 0] = True
    planes = mask[:, 1:].view(batch, slots, height, width).float()
    return ObsBatch(
        spatial=torch.rand(
            batch, stack * spec.spatial_shape[0], height, width, generator=generator
        ),
        mask_planes=planes.repeat(1, stack, 1, 1),
        vector=torch.rand(batch, spec.vector_size, generator=generator),
        mask=mask,
    )


@pytest.mark.parametrize("variant", ["mock/k1", "mock/k2", "wide/k1", "wide/k2"])
@pytest.mark.parametrize("batch", [1, 7, 512])
def test_shapes_and_dtypes(torch: Any, env_spec: Any, variant: str, batch: int) -> None:
    spec = variants(env_spec)[variant]
    model = _build(spec, arch())
    obs = fake_obs(torch, spec, batch, seed=1)

    logits = model.logits(obs)
    assert logits.shape == (batch, spec.n_actions)
    assert logits.dtype is torch.float32

    value = model.value(obs)
    assert value.shape == (batch,)
    assert value.dtype is torch.float32

    result = model.act(obs, torch.rand(batch, generator=torch.Generator().manual_seed(2)))
    assert result.actions.shape == (batch,)
    assert result.actions.dtype is torch.int64
    assert bool(obs.mask.gather(-1, result.actions.unsqueeze(-1)).all())
    for name in ("log_probs", "entropy", "p_noop"):
        field = getattr(result, name)
        assert field.shape == (batch,) and field.dtype is torch.float32
    assert result.n_legal.dtype is torch.int64

    back = model.backprop(obs, result.actions)
    for name in ("log_probs", "entropy", "noop_entropy", "values"):
        field = getattr(back, name)
        assert field.shape == (batch,) and field.dtype is torch.float32
    assert back.n_legal.dtype is torch.int64
    assert torch.equal(back.n_legal, result.n_legal)


@pytest.mark.parametrize("variant", ["mock/k1", "mock/k2", "wide/k1", "wide/k2"])
def test_the_stem_takes_the_channels_the_observation_supplies(
    torch: Any, env_spec: Any, variant: str
) -> None:
    """``k*(S + P) + 2 + E``, every term read off the environment."""
    spec = variants(env_spec)[variant]
    config = arch()
    model = _build(spec, config)
    planes = spec.spatial_shape[0]
    slots = spec.obs_space["mask_planes"].shape[0]
    expected = spec.frame_stack * (planes + slots) + 2 + config.vector_embed
    assert model.actor.trunk.in_channels == expected
    assert model.actor.trunk.stem.weight.shape[1] == expected


def test_coord_conv_can_be_turned_off(torch: Any, env_spec: Any) -> None:
    model = _build(env_spec, arch(coord_conv=False))
    planes = env_spec.spatial_shape[0]
    slots = env_spec.obs_space["mask_planes"].shape[0]
    assert model.actor.trunk.in_channels == planes + slots + NetConfig().vector_embed
    assert model.logits(fake_obs(torch, env_spec, 4, seed=3)).shape[0] == 4


def test_logits_are_float32_under_autocast(torch: Any, env_spec: Any) -> None:
    """The trunk really does run in bf16 -- otherwise this asserts nothing -- and the logits
    still arrive as float32, which is what makes the masking and the log_softmax exact."""
    model = _build(env_spec, arch(autocast_dtype="bfloat16"))
    obs = fake_obs(torch, env_spec, 4, seed=4)
    with torch.no_grad():
        with model.actor.autocast():
            features = model.actor.trunk(obs)
        assert features.dtype is torch.bfloat16
        logits = model.logits(obs)
        value = model.value(obs)
    assert logits.dtype is torch.float32
    assert value.dtype is torch.float32
    assert torch.isfinite(logits).all()


def test_a_masked_logit_gets_exactly_no_gradient(torch: Any, env_spec: Any) -> None:
    """Through the real head, not through a synthetic tensor: what is being checked is that the
    fill happens before anything differentiable touches the logit."""
    model = _build(env_spec, arch())
    obs = fake_obs(torch, env_spec, 6, seed=5, density=0.2)
    logits = model.logits(obs)
    retained = logits.detach().requires_grad_(True)
    from royalelearn.learn.distribution import MaskedCategorical

    distribution = MaskedCategorical(retained, obs.mask)
    distribution.log_probs.sum().backward()
    assert retained.grad is not None
    assert torch.equal(retained.grad[~obs.mask], torch.zeros_like(retained.grad[~obs.mask]))
    assert bool((retained.grad[obs.mask] != 0).any())


@pytest.mark.parametrize("separate_trunks", [True, False])
def test_the_entropy_term_never_reaches_the_critics_parameters(
    torch: Any, env_spec: Any, separate_trunks: bool
) -> None:
    """Separate trunks make it structural; with a shared trunk the critic's parameters are the
    value head's alone, and the trunk is listed with the actor because it carries the policy
    gradient. Either way the entropy bonus cannot move the value function."""
    model = _build(env_spec, arch(separate_trunks=separate_trunks))
    model.zero_grad(set_to_none=True)
    obs = fake_obs(torch, env_spec, 8, seed=6)
    actions = model.act(obs, torch.rand(8, generator=torch.Generator().manual_seed(7))).actions
    back = model.backprop(obs, actions)
    (back.entropy.mean() + back.noop_entropy.mean()).backward()

    critic = list(model.critic_parameters())
    assert critic
    for parameter in critic:
        assert parameter.grad is None or bool((parameter.grad == 0).all())
    actor = list(model.actor_parameters())
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in actor)
    # The two iterators are disjoint, so an optimizer built from both does not count a tensor
    # twice and a per-group learning rate means what it says.
    assert not ({id(p) for p in actor} & {id(p) for p in critic})


@pytest.mark.parametrize("separate_trunks", [True, False])
def test_the_value_loss_does_reach_the_critic(
    torch: Any, env_spec: Any, separate_trunks: bool
) -> None:
    model = _build(env_spec, arch(separate_trunks=separate_trunks))
    model.zero_grad(set_to_none=True)
    obs = fake_obs(torch, env_spec, 8, seed=8)
    model.value(obs).pow(2).mean().backward()
    assert any(
        p.grad is not None and bool((p.grad != 0).any()) for p in model.critic_parameters()
    )


def test_initialisation_gains_are_what_was_asked_for(torch: Any, env_spec: Any) -> None:
    """Orthogonal rows scaled by the gain, so ``W W^T`` is ``gain^2 I``; zero biases everywhere;
    a near-zero policy head, which is what makes the opening policy near uniform over the legal
    set rather than confident about a tile."""
    from royalelearn.learn import nets

    model = _build(env_spec, arch())
    actor, critic = model.actor, model.critic

    def gram(weight: Any) -> Any:
        flat = weight.detach().reshape(weight.shape[0], -1)
        return flat @ flat.T

    hidden = [
        actor.trunk.vector_embed.weight,
        actor.trunk.stem.weight,
        actor.trunk.body[0].conv1.weight,
        actor.trunk.body[0].conv2.weight,
        actor.head.feature.weight,
        critic.head.vector_embed.weight,
        critic.head.hidden.weight,
    ]
    for weight in hidden:
        expected = torch.eye(weight.shape[0]) * nets.HIDDEN_GAIN**2
        assert torch.allclose(gram(weight), expected, atol=1e-4)
    for weight in (actor.head.query.weight, actor.head.query_bias.weight, actor.head.noop.weight):
        expected = torch.eye(weight.shape[0]) * nets.HEAD_GAIN**2
        assert torch.allclose(gram(weight), expected, atol=1e-8)
    assert torch.allclose(
        gram(critic.head.out.weight), torch.eye(1) * nets.VALUE_GAIN**2, atol=1e-6
    )
    for module in model.modules():
        bias = getattr(module, "bias", None)
        if bias is not None and not isinstance(module, torch.nn.GroupNorm):
            assert torch.equal(bias.detach(), torch.zeros_like(bias))
    table = actor.head.card_embed.weight.detach()
    assert table.shape[0] == hand_fields(env_spec).onehot_width, (
        "the card table is one row per one-hot slot the layout declares"
    )
    assert nets.EMBED_STD * 0.6 < float(table.std()) < nets.EMBED_STD * 1.6


def test_the_opening_policy_is_near_uniform_over_the_legal_set(
    torch: Any, env_spec: Any
) -> None:
    """The consequence of the head gain, and the reason ``noop_bias`` can default to zero: the
    agent opens by spending elixir at random tiles, and the mask then makes it wait."""
    model = _build(env_spec, arch())
    obs = fake_obs(torch, env_spec, 4, seed=9)
    distribution = model.actor.distribution(obs)
    n_legal = distribution.n_legal().double()
    assert torch.allclose(distribution.entropy().double(), n_legal.log(), atol=0.1)


def test_two_builds_of_one_seed_are_identical_and_two_seeds_are_not(
    torch: Any, env_spec: Any
) -> None:
    """The initial weights are a function of the master seed alone: drawn on the CPU from
    ``torch/init`` and moved afterwards, never from a global generator."""
    from royalelearn.learn.nets import DefaultNetworkFactory

    config = arch()
    first = DefaultNetworkFactory(MASTER_SEED).build(env_spec, config, "cpu")
    torch.rand(1000)  # a global draw between the two builds must change nothing
    second = DefaultNetworkFactory(MASTER_SEED).build(env_spec, config, "cpu")
    other = DefaultNetworkFactory(MASTER_SEED + 1).build(env_spec, config, "cpu")
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name]), name
    assert not torch.equal(
        first.actor.head.card_embed.weight, other.actor.head.card_embed.weight
    )


def test_parameter_counts_are_what_the_budget_says(torch: Any, env_spec: Any) -> None:
    """The shipped architecture on the wider catalogue, against the figures the throughput
    budget was written from. Within five per cent, because the table counts convolution and
    linear weights and this counts every parameter there is."""
    spec = synthetic_env_spec(env_spec, WIDER_CATALOGUE)
    model = _build(spec, NetConfig(device="cpu", autocast_dtype="float32"))
    actor = sum(p.numel() for p in model.actor_parameters())
    critic = sum(p.numel() for p in model.critic_parameters())
    assert math.isclose(actor, 400_000, rel_tol=0.05), actor
    assert math.isclose(critic, 422_000, rel_tol=0.05), critic


def test_a_snapshot_is_the_actors_weights_in_half_precision(torch: Any, env_spec: Any) -> None:
    model = _build(env_spec, arch())
    snapshot = model.actor_state_dict_fp16()
    assert snapshot
    assert set(snapshot) == set(model.actor.state_dict())
    for tensor in snapshot.values():
        assert tensor.dtype is torch.float16
        assert tensor.device.type == "cpu"
    assert not any(name.startswith(("critic", "value_head")) for name in snapshot)
    assert not any("coords" in name for name in snapshot), (
        "the coordinate planes are geometry, not weights"
    )
    # It loads back into an actor of the same architecture, which is what the pool does.
    fresh = _build(env_spec, arch(), seed=MASTER_SEED + 5)
    fresh.actor.load_state_dict({k: v.float() for k, v in snapshot.items()})
    obs = fake_obs(torch, env_spec, 3, seed=10)
    with torch.no_grad():
        assert torch.allclose(fresh.logits(obs), model.logits(obs), atol=1e-3)


def test_arch_digest_covers_what_decides_a_shape(torch: Any, env_spec: Any) -> None:
    from royalelearn.learn.nets import DefaultNetworkFactory

    factory = DefaultNetworkFactory(MASTER_SEED)
    config = arch()
    digest = factory.arch_digest(env_spec, config)
    assert digest == factory.arch_digest(env_spec, config)
    assert digest == DefaultNetworkFactory(MASTER_SEED + 1).arch_digest(env_spec, config)
    assert _build(env_spec, config).arch_digest == digest

    stacked = msgspec.structs.replace(env_spec, frame_stack=2)
    assert factory.arch_digest(stacked, config) != digest
    assert factory.arch_digest(synthetic_env_spec(env_spec, WIDER_CATALOGUE), config) != digest
    for changed in (
        arch(channels=16, card_embed=16),
        arch(blocks=2),
        arch(vector_embed=8),
        arch(separate_trunks=False),
        arch(noop_bias=1.0),
    ):
        assert factory.arch_digest(env_spec, changed) != digest


def test_a_mismatched_digest_is_refused_by_name(torch: Any, env_spec: Any) -> None:
    """A snapshot from another architecture must be turned away with a sentence, not discovered
    as a shape error out of the middle of a load."""
    from royalelearn.learn.nets import DefaultNetworkFactory

    factory = DefaultNetworkFactory(MASTER_SEED)
    config = arch()
    stale = factory.arch_digest(msgspec.structs.replace(env_spec, frame_stack=2), config)
    factory.check_digest(env_spec, config, factory.arch_digest(env_spec, config))
    with pytest.raises(PreflightError) as caught:
        factory.check_digest(env_spec, config, stale, source="the pool snapshot")
    message = str(caught.value)
    assert "the pool snapshot" in message
    assert stale in message and factory.arch_digest(env_spec, config) in message


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"card_embed": 7}, "card_embed"),
        ({"norm_groups": 5}, "norm_groups"),
        ({"policy_head": "flat"}, "policy_head"),
        ({"logit_scale": "sqrt_c"}, "logit_scale"),
        ({"init": "xavier"}, "init"),
        ({"autocast_dtype": "float8"}, "autocast dtype"),
    ],
)
def test_an_architecture_that_cannot_be_built_says_which_field(
    torch: Any, env_spec: Any, overrides: dict[str, Any], expected: str
) -> None:
    from royalelearn.learn.nets import DefaultNetworkFactory

    with pytest.raises(PreflightError, match=expected):
        DefaultNetworkFactory(MASTER_SEED).build(env_spec, arch(**overrides), "cpu")


def test_a_behaviour_snapshot_is_frozen_at_the_boundary(torch: Any, env_spec: Any) -> None:
    """What overlapped collection samples from: the actor as it was when the iteration began,
    so the stored log-probabilities all come from one parameter vector."""
    from royalelearn.learn.actor_critic import BehaviourSnapshot

    model = _build(env_spec, arch(), seed=MASTER_SEED + 9)
    obs = fake_obs(torch, env_spec, 5, seed=11)
    snapshot = BehaviourSnapshot.of(model)
    before = snapshot.logits(obs)
    with torch.no_grad():
        for parameter in model.actor_parameters():
            parameter.add_(0.5)
    assert torch.equal(snapshot.logits(obs), before)
    assert not torch.allclose(model.logits(obs), before)
    assert all(not p.requires_grad for p in snapshot.actor.parameters())
