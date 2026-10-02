"""``net.policy_head = "factored"``: wait or act, then which card or button, then which tile.

The head writes the same flat action space the pointer head writes, as log-probabilities:

* ``log p(no-op) = log P(WAIT)``;
* ``log p(slot s, tile t) = log P(ACT) + log P(s | ACT) + log P(t | s)``;
* ``log p(button b) = log P(ACT) + log P(b | ACT)``.

A slot is a candidate iff any of its tiles is legal, a button iff its mask bit is set, and with no
candidate the gate waits. So everything downstream of the logits -- the masked distribution, its
sample, ``gtau`` -- is unchanged. The entropy splits exactly into a gate, a candidate and a tile
term by the chain rule, and ``ppo.entropy_coef_stages`` can weight the three apart.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn import config as cfg
from royalelearn.errors import PreflightError
from royalelearn.testing import coordinator, read_env_spec, tiny_config

torch = pytest.importorskip("torch")

SMALL = {"channels": 8, "blocks": 1, "norm_groups": 4, "card_embed": 8, "value_hidden": 16}


def _button_env() -> cfg.EnvFactorySpec:
    env = cfg.default_env_spec(cfg.MOCK_ENGINE, max_steps=6)
    parser = msgspec.structs.replace(env.action_parser, kwargs={"ability_buttons": True})
    return msgspec.structs.replace(env, action_parser=parser)


@pytest.fixture(scope="module")
def spec() -> Any:
    return read_env_spec(_button_env())


def _arch(**overrides: Any) -> cfg.NetConfig:
    settings = {**SMALL, "device": "cpu", "autocast_dtype": "float32", "policy_head": "factored"}
    return cfg.NetConfig(**{**settings, **overrides})


def _model(spec: Any, seed: int = 4242, **overrides: Any) -> Any:
    from royalelearn.learn.nets import DefaultNetworkFactory

    return DefaultNetworkFactory(seed).build(spec, _arch(**overrides), "cpu")


def _obs(spec: Any, rows: int, seed: int, density: float = 0.05) -> Any:
    """Random states, with some rows whose hand slots have no legal tile at all and some with a
    button ready, so that every candidate rule is exercised."""
    from royalelearn.api.policy import ObsBatch

    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, spec.n_actions), generator=generator) < density
    grid = spec.n_grid_actions
    tiles = spec.tiles[0] * spec.tiles[1]
    slots = mask[:, 1:grid].view(rows, spec.hand_size, tiles)
    slots[::3, 1] = False  # slot 1 has nothing legal on every third row
    mask[1::4, grid:] = False  # no button ready on some rows
    mask[2::4, grid] = True  # button 0 ready on others
    mask[5] = False  # one row with nothing but the no-op
    mask[:, 0] = True
    return ObsBatch(
        spatial=torch.rand(
            (rows, spec.frame_stack * spec.spatial_shape[0], *spec.tiles), generator=generator
        ),
        vector=torch.rand((rows, spec.vector_size), generator=generator),
        mask=mask,
        mask_planes=mask[:, 1:grid].float().view(rows, spec.hand_size, *spec.tiles),
    )


def _perturb(model: Any, seed: int = 9) -> None:
    """Weights away from their initial gains, so the stages are far from uniform."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.actor.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)


def _distribution(model: Any, obs: Any) -> Any:
    from royalelearn.learn.distribution import MaskedCategorical

    return MaskedCategorical(model.actor.logits(obs).float(), obs.mask)


def test_the_flat_log_probs_are_the_chain_of_the_three_stages(spec: Any) -> None:
    model = _model(spec)
    _perturb(model)
    obs = _obs(spec, 12, seed=1)
    head = model.actor.head
    with torch.no_grad():
        stages = head.stages(model.actor.trunk(obs), obs)
        lp = _distribution(model, obs).log_probs
    grid, hand = spec.n_grid_actions, spec.hand_size
    tiles = spec.tiles[0] * spec.tiles[1]
    legal = obs.mask
    want = torch.full_like(lp, -math.inf)
    want[:, 0] = stages.wait
    for s in range(hand):
        want[:, 1 + s * tiles : 1 + (s + 1) * tiles] = (
            stages.act[:, None] + stages.candidate[:, s, None] + stages.tile[:, s]
        )
    for b in range(spec.n_buttons):
        want[:, grid + b] = stages.act + stages.candidate[:, hand + b]
    p = lp.exp()
    assert torch.allclose(p.sum(-1), torch.ones(12), atol=1e-5)
    assert bool((p[~legal] == 0).all()), "an illegal action has probability"
    assert torch.allclose(lp[legal], want[legal], atol=1e-5)


def test_a_candidate_is_a_slot_with_a_legal_tile_or_a_ready_button(spec: Any) -> None:
    model = _model(spec)
    _perturb(model)
    obs = _obs(spec, 12, seed=2)
    with torch.no_grad():
        stages = model.actor.head.stages(model.actor.trunk(obs), obs)
    grid, hand = spec.n_grid_actions, spec.hand_size
    tiles = spec.tiles[0] * spec.tiles[1]
    slot_legal = obs.mask[:, 1:grid].view(12, hand, tiles).any(-1)
    candidate_legal = torch.cat([slot_legal, obs.mask[:, grid:]], dim=-1)
    p_candidate = stages.candidate.exp()
    # On a row with no candidate the gate waits, so its candidate stage is never reached.
    some = candidate_legal.any(-1)
    assert float(stages.act[~some].exp().max()) == 0.0
    assert bool((p_candidate[some][~candidate_legal[some]] == 0).all())
    assert torch.allclose(p_candidate[some].sum(-1), torch.ones(int(some.sum())), atol=1e-5)
    assert bool(slot_legal[::3, 1].logical_not().all()), "the fixture lost its empty slot"


def test_with_nothing_to_play_it_waits(spec: Any) -> None:
    model = _model(spec)
    _perturb(model)
    obs = _obs(spec, 12, seed=3)
    with torch.no_grad():
        lp = _distribution(model, obs).log_probs
    assert int(obs.mask[5].sum()) == 1
    assert float(lp[5, 0]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("act_init", [0.1, 0.3])
def test_the_gate_opens_at_factored_act_init(spec: Any, act_init: float) -> None:
    model = _model(spec, factored_act_init=act_init)
    obs = _obs(spec, 64, seed=4)
    with torch.no_grad():
        p_noop = _distribution(model, obs).log_probs[:, 0].exp()
    playable = obs.mask[:, 1:].any(-1)
    assert float((1.0 - p_noop[playable]).mean()) == pytest.approx(act_init, abs=0.02)


@pytest.mark.parametrize("head", ["pointer", "factored"])
def test_the_stage_entropies_sum_to_the_entropy_exactly(spec: Any, head: str) -> None:
    model = _model(spec, policy_head=head)
    _perturb(model)
    obs = _obs(spec, 16, seed=5, density=0.2)
    with torch.no_grad():
        dist = _distribution(model, obs)
        gate, candidate, tile = dist.stage_entropies(
            spec.hand_size, spec.tiles[0] * spec.tiles[1]
        )
        total = dist.entropy()
        p = dist.p_noop()
    assert torch.allclose(gate + candidate + tile, total, atol=1e-5)
    xlogy = torch.special.xlogy
    binary = -(xlogy(p, p) + xlogy(1 - p, 1 - p))
    assert torch.allclose(gate, binary, atol=1e-5)
    assert bool((candidate > 0).any()) and bool((tile > 0).any())
    assert bool((gate >= 0).all() and (candidate >= -1e-6).all() and (tile >= -1e-6).all())


def test_the_pointer_digest_does_not_read_the_gate_init(spec: Any) -> None:
    from royalelearn.learn.nets import DefaultNetworkFactory

    factory = DefaultNetworkFactory(1)
    pointer = _arch(policy_head="pointer")
    digest = factory.arch_digest(spec, pointer)
    assert factory.arch_digest(spec, _arch(policy_head="pointer", factored_act_init=0.4)) == digest
    factored = factory.arch_digest(spec, _arch())
    assert factored != digest
    # A starting value, not a shape: a clone made with another one still loads.
    assert factory.arch_digest(spec, _arch(factored_act_init=0.4)) == factored


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"noop_bias": 1.0}, "noop_bias"),
        ({"factored_act_init": 0.0}, "factored_act_init"),
        ({"factored_act_init": 1.0}, "factored_act_init"),
    ],
)
def test_what_the_factored_head_cannot_mean_is_refused(
    spec: Any, overrides: dict[str, Any], expected: str
) -> None:
    with pytest.raises(PreflightError, match=expected):
        _model(spec, **overrides)


def test_entropy_coef_stages_weights_the_three_terms(spec: Any) -> None:
    from royalelearn.learn.ppo import entropy_bonus

    model = _model(spec)
    _perturb(model)
    obs = _obs(spec, 8, seed=6, density=0.2)
    hand, tiles = spec.hand_size, spec.tiles[0] * spec.tiles[1]
    with torch.no_grad():
        dist = _distribution(model, obs)
        gate, candidate, tile = dist.stage_entropies(hand, tiles)
        single = entropy_bonus(dist, 0.01, None, hand_size=hand, tiles=tiles)
        staged = entropy_bonus(dist, 0.01, (0.5, 0.25, 0.125), hand_size=hand, tiles=tiles)
    assert torch.allclose(single, 0.01 * dist.entropy())
    assert torch.allclose(staged, 0.5 * gate + 0.25 * candidate + 0.125 * tile)


def test_bad_stage_coefficients_are_refused() -> None:
    for bad in ((0.1, 0.1), (0.1, -0.1, 0.1)):
        config = cfg.RunConfig(ppo=cfg.PPOConfig(entropy_coef_stages=bad))
        assert [p for p in cfg.check_consistency(config) if "entropy_coef_stages" in p], bad


def test_the_gradients_stay_finite_where_nothing_can_be_played(spec: Any) -> None:
    from royalelearn.learn.ppo import entropy_bonus

    model = _model(spec)
    obs = _obs(spec, 12, seed=7)
    dist = _distribution(model, obs)
    hand, tiles = spec.hand_size, spec.tiles[0] * spec.tiles[1]
    loss = -(entropy_bonus(dist, 0.0, (1.0, 1.0, 1.0), hand_size=hand, tiles=tiles).sum())
    loss = loss - dist.log_prob(torch.zeros(12, dtype=torch.long)).sum()
    loss.backward()
    grads = [p.grad for p in model.actor.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


def _staged(tmp_path: Path, stages: Any, forced_rows: str = "all", ent: float = 0.0) -> Any:
    base = tiny_config(tmp_path)
    return tiny_config(
        tmp_path,
        env=_button_env(),
        net=msgspec.structs.replace(base.net, policy_head="factored"),
        ppo=msgspec.structs.replace(
            base.ppo,
            entropy_coef_stages=stages,
            forced_rows=forced_rows,
            ent_coef=cfg.ConstantSpec(ent),
            ent_coef_noop=cfg.ConstantSpec(0.0),
        ),
    )


def _actor_after_one_update(config: Any) -> list[Any]:
    with coordinator(config) as run:
        run.iterate()
        return [p.detach().clone() for p in run.model.actor_parameters()]


@pytest.mark.parametrize("forced_rows", ["all", "critic_only"])
def test_the_stage_coefficients_reach_the_update(tmp_path: Path, forced_rows: str) -> None:
    plain = _actor_after_one_update(_staged(tmp_path / "a", None, forced_rows))
    zeros = _actor_after_one_update(_staged(tmp_path / "b", (0.0, 0.0, 0.0), forced_rows))
    gate = _actor_after_one_update(_staged(tmp_path / "c", (1.0, 0.0, 0.0), forced_rows))
    assert all(torch.equal(a, b) for a, b in zip(plain, zeros, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(zeros, gate, strict=True))


@pytest.mark.parametrize("forced_rows", ["all", "critic_only"])
def test_a_factored_run_collects_updates_and_logs_the_stages(
    tmp_path: Path, forced_rows: str
) -> None:
    config = _staged(tmp_path, (0.01, 0.005, 0.002), forced_rows, ent=0.01)
    with coordinator(config) as run:
        for _ in range(2):
            run.iterate()
        rows = list(run.rows)
    last = rows[-1]
    stages = [last[f"ppo/entropy_{k}"] for k in ("gate", "candidate", "tile")]
    assert sum(stages) == pytest.approx(last["ppo/entropy"], rel=1e-4, abs=1e-6)
    assert 0.0 < last["ppo/p_act"] < 1.0


def test_a_pointer_run_logs_no_stage_keys(tmp_path: Path) -> None:
    with coordinator(tiny_config(tmp_path)) as run:
        run.iterate()
        rows = list(run.rows)
    stage_keys = {"ppo/entropy_gate", "ppo/entropy_candidate", "ppo/entropy_tile", "ppo/p_act"}
    assert not stage_keys & set(rows[-1])


def test_a_factored_snapshot_names_its_head_and_plays_as_the_live_actor(tmp_path: Path) -> None:
    import json

    from royalelearn.learn.distribution import MaskedCategorical

    config = _staged(tmp_path, None)
    config = msgspec.structs.replace(
        config, net=msgspec.structs.replace(config.net, factored_act_init=0.2)
    )
    with coordinator(config) as run:
        assert run.artifact_spec().meta == {"policy_head": "factored", "factored_act_init": 0.2}
        run.snapshot_store.put("snap:test", run.model, {"step": 3, "note": "x"})
        folder = run.snapshot_store._folder("snap:test")
        meta = json.loads((folder / "spec.json").read_text(encoding="utf-8"))["meta"]
        assert meta == {"policy_head": "factored", "factored_act_init": 0.2, "note": "x"}
        frozen = run.snapshot_store.get("snap:test", "cpu")
        obs = _obs(run.spec, 8, seed=8)
        with torch.no_grad():
            live = MaskedCategorical(run.model.actor.logits(obs).float(), obs.mask).log_probs
            got = MaskedCategorical(frozen.logits(obs).float(), obs.mask).log_probs
    legal = obs.mask
    # The pool keeps half precision, so the frozen copy agrees to that precision.
    assert torch.allclose(got[legal], live[legal], atol=1e-2)


def test_a_pointer_runs_artifact_spec_has_no_head_meta(tmp_path: Path) -> None:
    with coordinator(tiny_config(tmp_path)) as run:
        assert run.artifact_spec().meta == {}
