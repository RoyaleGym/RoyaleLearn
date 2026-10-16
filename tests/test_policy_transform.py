"""``ppo.policy_transform``: the policy the learner trains is a function of its actor's logits.

Set, the learner's seats sample the masked softmax of ``f(logits, mask, vector, layout)``, the
update differentiates that same policy, and the stored log-probabilities are its own, so the
first ratio of an update is one as it is without a transform. Every path that reads the
learner's policy reads it; the actor, which is what a snapshot holds and what evaluation plays,
does not. Unset, nothing is encoded and nothing moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

from royalelearn import config as cfg  # noqa: E402
from royalelearn.learn.distribution import MaskedCategorical  # noqa: E402
from royalelearn.learn.nets import DefaultNetworkFactory  # noqa: E402
from royalelearn.learn.transform import PolicyLayout, PolicyTransform  # noqa: E402
from royalelearn.testing import ARCH, coordinator, tiny_config  # noqa: E402

SHARPEN = "plugin:transform_plugins:sharpen"


def _layout(spec: Any) -> PolicyLayout:
    tiles = spec.tiles[0] * spec.tiles[1]
    return PolicyLayout(spec.hand_size, tiles, spec.n_actions - 1 - spec.hand_size * tiles)


def _obs(spec: Any, rows: int = 12, seed: int = 3) -> Any:
    from royalelearn.api.policy import ObsBatch

    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, spec.n_actions), generator=generator) < 0.3
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


def _model(spec: Any, mode: str | None = SHARPEN, **fields: Any) -> Any:
    model = DefaultNetworkFactory(7).build(spec, msgspec.structs.replace(ARCH, **fields), "cpu")
    if mode is not None:
        model.policy_transform = PolicyTransform(mode, _layout(spec))
    return model


def test_unset_nothing_is_encoded() -> None:
    assert b"policy_transform" not in msgspec.json.encode(cfg.PPOConfig())
    assert cfg.policy_transform_of(cfg.PPOConfig()) is None


@pytest.mark.parametrize("value", ["stochastic", "argmax", "plugin:transform_plugins", "temperature"])
def test_a_transform_is_a_function_of_your_own(tmp_path: Path, value: str) -> None:
    base = tiny_config(tmp_path)
    config = tiny_config(tmp_path, ppo=msgspec.structs.replace(base.ppo, policy_transform=value))
    problems = [p for p in cfg.check_consistency(config) if "ppo.policy_transform" in p]
    assert problems, value


@pytest.mark.parametrize("separate_trunks", [True, False])
def test_every_learner_path_reads_the_transformed_policy(
    env_spec: Any, separate_trunks: bool
) -> None:
    """Plant: leave the transform out of any one path and its log-probabilities are the
    actor's."""
    from royalelearn.learn.actor_critic import BehaviourSnapshot

    model = _model(env_spec, separate_trunks=separate_trunks)
    obs = _obs(env_spec)
    raw = MaskedCategorical(model.actor.logits(obs).detach(), obs.mask).log_probs
    want = MaskedCategorical(model.actor.logits(obs).detach() * 2.0, obs.mask).log_probs
    assert not torch.allclose(raw, want)

    def same(got: Any, expected: Any) -> None:
        assert torch.allclose(got.detach(), expected, atol=1e-6), (got - expected).abs().max()

    same(model.distribution(obs).log_probs, want)
    uniforms = torch.rand(obs.mask.shape[0], generator=torch.Generator().manual_seed(1))
    acted = model.act(obs, uniforms)
    same(acted.log_probs, want.gather(1, acted.actions[:, None])[:, 0])
    backprop = model.backprop(obs, acted.actions)
    same(backprop.log_probs, want.gather(1, acted.actions[:, None])[:, 0])
    same(backprop.distribution.log_probs, want)
    same(BehaviourSnapshot.of(model).distribution(obs).log_probs, want)
    if separate_trunks:
        rows = torch.tensor([1, 4, 7])
        same(model.distribution_on_rows(obs, rows).log_probs, want.index_select(0, rows))
    # What a snapshot of it holds, and what evaluation plays, is the actor's own.
    same(model.actor.distribution(obs).log_probs, raw)


def test_evaluation_plays_the_actor_of_the_live_learner(
    env_spec: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plant: hand evaluation the pairing, and the live learner plays the transformed policy
    while every snapshot of it plays the actor."""
    from royalelearn.ladder.actors import EvalActors
    from royalelearn.ladder.pool import LEARNER_ID

    model = _model(env_spec)
    played: list[Any] = []
    monkeypatch.setattr(EvalActors, "policy", lambda self, actor: played.append(actor))
    EvalActors(env_spec, ARCH, snapshots=None, model=model)(LEARNER_ID)  # type: ignore[arg-type]
    assert played == [model.actor]


def test_the_shared_trunk_update_runs_the_head_as_the_rollout_does(env_spec: Any) -> None:
    """Under ``net.policy_head_float32`` the actor's forward runs the head in float32 on the
    trunk's features; the shared-trunk update ran it under the autocast. Plant: that, and the
    update's logits are not the rollout's."""
    model = _model(
        env_spec,
        None,
        separate_trunks=False,
        autocast_dtype="bfloat16",
        policy_head_float32=True,
        noop_bias=8.0,
    )
    obs = _obs(env_spec, rows=16)
    actions = torch.zeros(16, dtype=torch.int64)
    with torch.no_grad():
        rollout = model.distribution(obs).log_prob(actions)
        update = model.backprop(obs, actions).log_probs
    assert torch.equal(rollout, update), (rollout - update).abs().max()


@pytest.mark.parametrize(
    ("name", "says"),
    [
        ("narrower", "shaped like its input"),
        ("float64", "float32 logits"),
        ("detached", "passes no gradient"),
        ("not_a_number", "non-finite"),
        ("mixes_rows", "batched with other rows"),
    ],
)
def test_a_transform_that_is_not_a_policy_is_refused(env_spec: Any, name: str, says: str) -> None:
    model = _model(env_spec, f"plugin:transform_plugins:{name}")
    with pytest.raises(ValueError, match=says):
        model.backprop(_obs(env_spec), torch.zeros(12, dtype=torch.int64))


def test_a_transform_that_cannot_be_found_stops_the_run(tmp_path: Path) -> None:
    from royalelearn.errors import PreflightError

    base = tiny_config(tmp_path)
    config = tiny_config(
        tmp_path,
        ppo=msgspec.structs.replace(base.ppo, policy_transform="plugin:transform_plugins:none"),
    )
    with pytest.raises(PreflightError, match="has no function 'none'"), coordinator(config):
        pass


@pytest.mark.parametrize("overlap", [False, True])
def test_a_run_trains_the_transformed_policy(tmp_path: Path, overlap: bool) -> None:
    """Two iterations with the ratio invariant at float32's tolerance on every one: the stored
    log-probabilities are the transformed policy's, on the live actor and on the overlap's
    behaviour snapshot. The run's identity names the transform's code."""
    base = tiny_config(tmp_path)
    config = tiny_config(
        tmp_path,
        ppo=msgspec.structs.replace(base.ppo, policy_transform="plugin:transform_plugins:gate"),
        rollout=msgspec.structs.replace(base.rollout, overlap=overlap),
    )
    lines: list[str] = []
    with coordinator(config, printer=lines.append) as run:
        run.iterate()
        run.iterate()
        assert "transform_plugins" in (run.identity.user_code or {})
    assert any(line.startswith("policy        plugin:transform_plugins:gate") for line in lines)


def test_a_transform_the_update_leaves_out_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ratio invariant is what holds every path to the transform: an update that reads the
    actor's own policy while the rollout sampled the transformed one stops the run, naming the
    transform."""
    from royalelearn.learn.actor_critic import SeparateActorCritic

    # The update's forward and its ratio check build the policy here; the rollout reads logits.
    monkeypatch.setattr(
        SeparateActorCritic,
        "distribution",
        lambda self, obs: MaskedCategorical(self.actor.logits(obs), obs.mask),
    )
    base = tiny_config(tmp_path)
    config = tiny_config(tmp_path, ppo=msgspec.structs.replace(base.ppo, policy_transform=SHARPEN))
    refused = pytest.raises(AssertionError, match=r"ppo\.policy_transform is")
    with refused, coordinator(config) as run:
        run.iterate()
