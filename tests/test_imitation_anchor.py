"""Sections 19.5-19.9: the freeze, the reference-KL anchor and its adaptive coefficient.

What is held here, each by a test that has been seen failing on a plant:

- the KL is the forward one, exact over the legal set, and its chain-rule parts sum to it;
- the coefficient moves by the rule of 19.8 and nothing else;
- the anchor's contribution to the actor's gradient is exactly lambda * grad(sum KL) * scale,
  it leaves minibatch size a pure memory knob, and it is the same under every forced-row arm;
- a zero coefficient leaves the update bit-identical to one without the block (the blind
  control), in the update and over three iterations of a run;
- a frozen actor does not move, its optimizer holds no moments, the critic trains, and the
  row leaves the actor's keys out;
- lambda survives a checkpoint.
"""

# ruff: noqa: F811 - ``rect`` is test_ppo's fixture, imported by name so pytest finds it here.
from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import config as cfg
from royalelearn.errors import PreflightError
from royalelearn.imitation.regularisers import (
    AdaptiveCoefficient,
    ReferenceKL,
    RowFilter,
    joint_kl,
    joint_kl_parts,
    log1mexp,
    noop_kl,
)
from royalelearn.metrics.behaviour import play_rate_by_elixir

torch = pytest.importorskip("torch")

from imitation_support import seeded_artifact, with_imitation  # noqa: E402
from royalelearn.imitation.references import FieldMLPReference, SnapshotReference  # noqa: E402
from test_coordinator import coordinator, tiny_config  # noqa: E402
from test_inference import build_model  # noqa: E402
from test_ppo import (  # noqa: E402
    CONFIG,
    FORCED_CELLS,
    SAMPLES,
    SCHEDULE,
    _Recording,
    collect,
    observations,  # noqa: F401 - a fixture
    plant_forced,
    rect,  # noqa: F401 - a fixture
    update_for,
)

# --------------------------------------------------------------------------
# The algebra
# --------------------------------------------------------------------------

HAND, TILES = 4, 6


def _random_pair(seed: int, batch: int = 64) -> tuple[Any, Any, Any]:
    """Two masked log-probability tables over a small action layout, with a random legal set."""
    from royalelearn.learn.distribution import MaskedCategorical

    generator = torch.Generator().manual_seed(seed)
    width = 1 + HAND * TILES
    mask = torch.rand(batch, width, generator=generator) < 0.4
    mask[:, 0] = True
    ref = MaskedCategorical(torch.randn(batch, width, generator=generator) * 2.0, mask)
    pol = MaskedCategorical(torch.randn(batch, width, generator=generator) * 2.0, mask)
    return ref.log_probs, pol.log_probs, mask


def test_the_kl_is_the_forward_one_over_the_legal_set() -> None:
    ref, pol, mask = _random_pair(1)
    p, q = ref.exp().numpy(), pol.exp().numpy()
    legal = mask.numpy()
    by_hand = np.array(
        [
            sum(p[b, a] * (math.log(p[b, a]) - math.log(q[b, a])) for a in np.flatnonzero(legal[b]))
            for b in range(p.shape[0])
        ]
    )
    np.testing.assert_allclose(joint_kl(ref, pol, mask).numpy(), by_hand, rtol=1e-4, atol=1e-6)
    reverse = joint_kl(pol, ref, mask).numpy()
    assert not np.allclose(reverse, by_hand), "the two directions agree, so nothing is tested"
    assert torch.allclose(joint_kl(ref, ref, mask), torch.zeros(ref.shape[0]), atol=1e-6)


def test_the_chain_rule_parts_sum_to_the_joint_kl() -> None:
    """The spec's identity. Plant: weight the tile term by p(tile | slot), not p(slot, tile)."""
    for seed in range(4):
        ref, pol, mask = _random_pair(seed)
        parts = joint_kl_parts(ref, pol, mask, hand_size=HAND, tiles=TILES)
        total = parts.noop + parts.card + parts.tile
        np.testing.assert_allclose(
            total.numpy(), joint_kl(ref, pol, mask).numpy(), rtol=1e-4, atol=1e-5
        )
        for part in parts:
            assert bool((part > -1e-5).all()), "a chain-rule part of a KL is never negative"


def test_the_noop_marginal_is_the_bernoulli_kl() -> None:
    ref, pol, _ = _random_pair(5)
    p, q = ref[:, 0].exp().double(), pol[:, 0].exp().double()
    by_hand = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    got = noop_kl(ref[:, 0], log1mexp(ref[:, 0]), pol[:, 0], log1mexp(pol[:, 0]))
    np.testing.assert_allclose(got.double().numpy(), by_hand.numpy(), rtol=1e-4, atol=1e-6)


def test_log1mexp_holds_its_precision_at_both_ends() -> None:
    near_one = torch.tensor([-1e-6, -1e-3])  # p(no-op) near one: the play mass is tiny
    near_zero = torch.tensor([-20.0, -5.0])
    for values in (near_one, near_zero):
        exact = torch.log1p(-values.double().exp())
        np.testing.assert_allclose(log1mexp(values).double().numpy(), exact.numpy(), rtol=1e-5)


# --------------------------------------------------------------------------
# The coefficient
# --------------------------------------------------------------------------


def _coef(**overrides: Any) -> AdaptiveCoefficient:
    spec = {"start": 1.0, "max": 10.0, "up": 2.0, "down": 4.0, "band": 1.5, **overrides}
    return AdaptiveCoefficient(msgspec.convert(spec, type=cfg.CoefSpec))


@pytest.mark.parametrize(
    ("measured", "expected"),
    [
        (1.51, 2.0),  # above band * budget: up
        (1.49, 1.0),  # inside the band: no move
        (0.67, 1.0),  # inside the band from below
        (0.66, 0.25),  # below budget / band: down
        (None, 1.0),  # nothing measured: no move
    ],
)
def test_the_coefficient_moves_by_the_rule(measured: float | None, expected: float) -> None:
    coefficient = _coef()
    assert coefficient.observe(measured, 1.0, 0) == pytest.approx(expected)
    assert coefficient.value == pytest.approx(expected)


def test_the_coefficient_is_held_between_its_floor_and_its_ceiling() -> None:
    falling = {"kind": "piecewise", "points": [[0, 0.5], [100, 0.01]]}
    coefficient = _coef(start=8.0, min=falling)
    assert coefficient.observe(100.0, 1.0, 0) == 10.0  # capped
    for _ in range(5):
        coefficient.observe(0.0, 1.0, 0)
    assert coefficient.value == 0.5  # held on the protected floor
    assert coefficient.current(200) == 0.5  # the floor fell, the stored value has not yet moved
    coefficient.observe(0.0, 1.0, 200)
    assert coefficient.value == pytest.approx(0.125)  # and now it can fall on its own


# --------------------------------------------------------------------------
# The row filter
# --------------------------------------------------------------------------


def test_a_row_is_left_out_only_when_every_condition_holds(env_spec: Any) -> None:
    from royalelearn.obs_layout import field_slice

    clock = field_slice(env_spec, "clock")
    conditions = [
        msgspec.convert({"field": "clock", "index": 1, "op": ">=", "value": 0.5}, cfg.RowCondition),
        msgspec.convert({"field": "clock", "index": 2, "op": "<=", "value": 0.5}, cfg.RowCondition),
    ]
    rows = RowFilter(conditions, env_spec, what="test")
    vector = torch.zeros(4, env_spec.vector_size)
    vector[:, clock.start + 1] = torch.tensor([1.0, 1.0, 0.0, 1.0])
    vector[:, clock.start + 2] = torch.tensor([0.4, 0.6, 0.4, 0.5])
    assert rows.keep(vector).tolist() == [0.0, 1.0, 1.0, 0.0]


def test_a_condition_outside_its_field_is_refused(env_spec: Any) -> None:
    beyond = msgspec.convert(
        {"field": "clock", "index": 3, "op": "<", "value": 0}, cfg.RowCondition
    )
    with pytest.raises(PreflightError, match="outside field 'clock'"):
        RowFilter([beyond], env_spec, what="test")
    unknown = msgspec.convert({"field": "no_such", "op": "<", "value": 0}, cfg.RowCondition)
    with pytest.raises(PreflightError, match="no field named 'no_such'"):
        RowFilter([unknown], env_spec, what="test")


# --------------------------------------------------------------------------
# The update
# --------------------------------------------------------------------------


def _term(
    spec: Any,
    reference: Any,
    *,
    start: float = 1.0,
    budget: float = 0.1,
    factor: str = "joint",
    **coef: Any,
) -> ReferenceKL:
    regulariser = msgspec.convert(
        {
            "kind": "reference_kl",
            "name": "bc",
            "reference": "bc",
            "factor": factor,
            "budget": {"kind": "constant", "value": budget},
            "coef": {"start": start, **coef},
        },
        type=cfg.ReferenceKLSpec,
    )
    return ReferenceKL(regulariser, reference, spec)


def _reference(spec: Any, seed: int = 99) -> SnapshotReference:
    """Another seed's actor with its weights tripled: two freshly seeded actors are both near
    uniform, and a KL of a few millionths would test float32 rounding rather than the term."""
    actor = build_model(spec, seed=seed).actor
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.mul_(3.0)
    return SnapshotReference("bc", actor)


def _flat(recorded: Sequence[Any]) -> Any:
    assert len(recorded) == 1
    return recorded[0]


@pytest.mark.parametrize("arm", cfg.FORCED_ROW_ARMS)
def test_with_the_anchor_minibatch_size_is_still_a_pure_memory_knob(rect: Any, arm: str) -> None:
    model = build_model(rect.spec)
    plant_forced(rect, FORCED_CELLS)
    collect(rect, model)
    config = msgspec.structs.replace(CONFIG, forced_rows=arm)
    recorded = []
    for minibatch in (SAMPLES, SAMPLES // 4):
        update = update_for(
            build_model(rect.spec),
            msgspec.structs.replace(config, minibatch_size=minibatch),
            optimizer_factory=_Recording,
            extra_actor_terms=(_term(rect.spec, _reference(rect.spec), start=3.0),),
        )
        update.step(rect.buffer, SCHEDULE)
        recorded.append(_flat(update.actor_optimizer.recorded))
    scale = float(recorded[0].abs().max())
    assert float((recorded[0] - recorded[1]).abs().max()) <= 1e-5 * scale


def test_the_anchor_is_the_same_gradient_under_all_and_critic_only(rect: Any) -> None:
    model = build_model(rect.spec)
    plant_forced(rect, FORCED_CELLS)
    collect(rect, model)
    recorded = []
    for arm in ("all", "critic_only"):
        update = update_for(
            build_model(rect.spec),
            msgspec.structs.replace(CONFIG, forced_rows=arm),
            optimizer_factory=_Recording,
            extra_actor_terms=(_term(rect.spec, _reference(rect.spec), start=3.0),),
        )
        update.step(rect.buffer, SCHEDULE)
        recorded.append(_flat(update.actor_optimizer.recorded))
    scale = float(recorded[0].abs().max())
    assert float((recorded[0] - recorded[1]).abs().max()) <= 1e-5 * scale


def _anchor_gradient(rect: Any, reference: Any) -> Any:
    """lambda-free: grad over the actor of sum KL / N on every choice row, by autograd directly."""
    from royalelearn.learn.buffer import empty_obs

    model = build_model(rect.spec)
    buffer = rect.buffer
    cycles, slots = np.nonzero(buffer.trainable())
    rows, _ = buffer._stack_rows(cycles, slots)
    raw = torch.from_numpy(np.ascontiguousarray(buffer.obs_view[rows[:, 0]]))
    obs = empty_obs(rect.spec, raw.shape[0], frames=1, device="cpu")
    buffer.codec.unpack_to_device(raw, buffer._statics(), obs)
    choice = obs.mask.sum(-1) > 1
    policy = model.actor.distribution(obs).log_probs
    with torch.no_grad():
        ref = reference.log_probs(obs)
    kl = (joint_kl(ref, policy, obs.mask) * choice).sum() / int(cycles.size)
    grads = torch.autograd.grad(kl, list(model.actor_parameters()), allow_unused=True)
    return torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, model.actor_parameters(), strict=True)
        ]
    )


def test_the_anchor_adds_exactly_lambda_times_the_kl_gradient(rect: Any) -> None:
    """The term's whole contribution, against autograd on the KL alone.

    Plant: scaling the term by the minibatch weight twice, or summing over forced rows too, moves
    the difference away from lambda * grad."""
    model = build_model(rect.spec)
    collect(rect, model)
    reference = _reference(rect.spec)
    lam = 2.5
    got = {}
    for name, terms in (("plain", ()), ("anchor", (_term(rect.spec, reference, start=lam),))):
        update = update_for(
            build_model(rect.spec), CONFIG, optimizer_factory=_Recording, extra_actor_terms=terms
        )
        update.step(rect.buffer, SCHEDULE)
        got[name] = _flat(update.actor_optimizer.recorded)
    expected = lam * _anchor_gradient(rect, reference)
    difference = got["anchor"] - got["plain"]
    scale = float(expected.abs().max())
    assert scale > 0.0
    assert float((difference - expected).abs().max()) <= 1e-4 * scale


def test_a_zero_coefficient_is_the_update_without_the_block(rect: Any) -> None:
    """The blind control, in one update: bit for bit, and the term really was computed.

    Plant: a coefficient that leaks a thousandth past zero must break it."""
    model = build_model(rect.spec)
    collect(rect, model)
    got = {}
    term = _term(rect.spec, _reference(rect.spec), start=0.0, max=0.0)
    for name, terms in (("plain", ()), ("blind", (term,))):
        update = update_for(
            build_model(rect.spec), CONFIG, optimizer_factory=_Recording, extra_actor_terms=terms
        )
        result = update.step(rect.buffer, SCHEDULE)
        got[name] = _flat(update.actor_optimizer.recorded)
    assert torch.equal(got["plain"], got["blind"])
    assert result.extra["imitation/bc/kl"] > 0.0
    assert result.extra["imitation/bc/lambda"] == 0.0


def test_the_reported_parts_and_the_coefficient_after_one_update(rect: Any) -> None:
    model = build_model(rect.spec)
    collect(rect, model)
    term = _term(rect.spec, _reference(rect.spec), start=1.0, budget=0.0, up=1.5, max=10.0)
    update = update_for(build_model(rect.spec), CONFIG, extra_actor_terms=(term,))
    result = update.step(rect.buffer, SCHEDULE)
    row = result.extra
    parts = row["imitation/bc/kl_noop"] + row["imitation/bc/kl_card"] + row["imitation/bc/kl_tile"]
    # Float32 sums over every covered row: exact to a part in a thousand here, and the exact
    # identity is held on its own in test_the_chain_rule_parts_sum_to_the_joint_kl.
    assert parts == pytest.approx(row["imitation/bc/kl"], rel=1e-3)
    assert row["imitation/bc/lambda"] == 1.0
    assert row["imitation/bc/budget"] == 0.0
    assert row["imitation/bc/rows_frac"] == 1.0
    assert 0.0 <= row["imitation/bc/top1_agree"] <= 1.0
    assert row["imitation/bc/grad_ratio"] > 0.0
    # A KL above a zero budget moved lambda up, for the next iteration.
    assert term.coef.value == pytest.approx(1.5)


def test_the_gradient_ratio_is_measured_before_the_coefficient(rect: Any) -> None:
    """``grad_ratio`` is for choosing coef.start, so it is the gradient of the UNSCALED KL
    against the policy term's: the same number at lambda 2.5 as at lambda 1, bit for bit, since
    it is taken on the first minibatch before any step.

    Plant: measure it on what the loss adds, lambda included, and it reads 2.5 times higher."""
    model = build_model(rect.spec)
    collect(rect, model)
    reference = _reference(rect.spec)
    ratios = {}
    for lam in (1.0, 2.5):
        update = update_for(
            build_model(rect.spec),
            CONFIG,
            extra_actor_terms=(_term(rect.spec, reference, start=lam),),
        )
        ratios[lam] = update.step(rect.buffer, SCHEDULE).extra["imitation/bc/grad_ratio"]
    assert ratios[1.0] > 0.0
    assert ratios[2.5] == ratios[1.0]


def test_the_gradient_ratio_is_zero_while_the_policy_is_the_reference(rect: Any) -> None:
    model = build_model(rect.spec)
    collect(rect, model)
    learner = build_model(rect.spec)
    same = SnapshotReference("bc", copy.deepcopy(learner.actor))
    update = update_for(learner, CONFIG, extra_actor_terms=(_term(rect.spec, same),))
    row = update.step(rect.buffer, SCHEDULE).extra
    assert row["imitation/bc/kl"] == pytest.approx(0.0, abs=1e-6)
    assert row["imitation/bc/grad_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_a_play_wait_reference_anchors_only_the_play_wait_split(rect: Any) -> None:
    model = build_model(rect.spec)
    collect(rect, model)
    from royalelearn.obs_layout import field_slice

    module = torch.nn.Sequential(torch.nn.Linear(1, 1))
    with torch.no_grad():
        module[0].weight.fill_(3.0)
        module[0].bias.fill_(-1.0)
    reference = FieldMLPReference(
        "bc",
        module,
        [field_slice(rect.spec, "own_elixir")],
        torch.zeros(1),
        torch.ones(1),
    )
    update = update_for(
        build_model(rect.spec),
        CONFIG,
        extra_actor_terms=(_term(rect.spec, reference, factor="noop_marginal"),),
    )
    row = update.step(rect.buffer, SCHEDULE).extra
    assert "imitation/bc/kl_card" not in row and "imitation/bc/kl_tile" not in row
    assert row["imitation/bc/kl_noop"] == pytest.approx(row["imitation/bc/kl"])
    assert row["imitation/bc/kl"] > 0.0


# -- the freeze ---------------------------------------------------------------------------


FROZEN = msgspec.structs.replace(SCHEDULE, actor_lr_scale=0.0)


@pytest.mark.parametrize("separate", [True, False])
def test_a_frozen_actor_does_not_move_and_its_optimizer_holds_nothing(
    rect: Any, separate: bool
) -> None:
    """Under a shared trunk the critic's loss reaches the trunk, which is the actor's: it is the
    case where stepping the actor's optimizer would move a frozen actor.

    Plant: step the actor's optimizer on a frozen iteration; the shared-trunk case must fail."""
    from royalelearn.learn.nets import DefaultNetworkFactory
    from test_inference import ARCH, SEED

    arch = msgspec.structs.replace(ARCH, separate_trunks=separate)
    model = DefaultNetworkFactory(SEED).build(rect.spec, arch, "cpu")
    collect(rect, build_model(rect.spec))
    before_actor = [p.detach().clone() for p in model.actor_parameters()]
    before_critic = [p.detach().clone() for p in model.critic_parameters()]
    update = update_for(model, msgspec.structs.replace(CONFIG, debug_assert_iterations=1))
    result = update.step(rect.buffer, FROZEN)
    for was, now in zip(before_actor, model.actor_parameters(), strict=True):
        assert torch.equal(was, now)
    assert any(
        not torch.equal(was, now)
        for was, now in zip(before_critic, model.critic_parameters(), strict=True)
    )
    assert update.actor_optimizer.state_dict()["state"] == {}
    assert not result.actor_trained
    assert result.ratio_max_abs_dev == pytest.approx(0.0, abs=1e-6)


def test_a_frozen_row_leaves_the_actors_keys_out() -> None:
    from royalelearn.api.update import UpdateResult
    from royalelearn.metrics import schema
    from royalelearn.metrics.records import update_fields

    base = {field: 0.0 for field in UpdateResult.__struct_fields__}
    base.update(kl_by_epoch=[0.0], clip_fraction_by_epoch=[0.0], extra={})
    base.update(n_minibatches=1, n_optimizer_steps=1, n_samples=1, actor_rows=1, actor_forwards=1)
    trained = update_fields(msgspec.convert({**base, "actor_trained": True}, UpdateResult))
    frozen = update_fields(msgspec.convert({**base, "actor_trained": False}, UpdateResult))
    assert set(schema.ACTOR_UPDATE_KEYS) <= set(trained)
    assert not set(schema.ACTOR_UPDATE_KEYS) & set(frozen)
    assert "ppo/kl_epoch0" in trained and "ppo/kl_epoch0" not in frozen
    assert "ppo/value_loss" in frozen and "ppo/grad_norm_critic" in frozen


def test_the_coefficient_and_the_freeze_survive_a_checkpoint(rect: Any, tmp_path: Path) -> None:
    from royalelearn.learn.freeze import FreezeTracker

    model = build_model(rect.spec)
    collect(rect, model)
    term = _term(rect.spec, _reference(rect.spec), start=1.0, budget=0.0, up=2.0)
    freeze = FreezeTracker()
    update = update_for(model, CONFIG, extra_actor_terms=(term,), freeze=freeze)
    update.step(rect.buffer, FROZEN)
    update.step(rect.buffer, SCHEDULE)
    update.save_checkpoint(tmp_path)
    fresh = _term(rect.spec, _reference(rect.spec), start=1.0, budget=0.0, up=2.0)
    thawed = FreezeTracker()
    restored = update_for(build_model(rect.spec), CONFIG, extra_actor_terms=(fresh,), freeze=thawed)
    restored.load_checkpoint(tmp_path, strict=True)
    assert fresh.coef.value == 2.0
    assert fresh.state() == term.state()
    assert thawed.state() == freeze.state() == {"unfrozen_at": 0, "last_frozen": False}


# --------------------------------------------------------------------------
# A run
# --------------------------------------------------------------------------


def _learner_state(run: Any) -> list[Any]:
    tensors = [t.detach().clone() for t in run.model.state_dict().values()]
    for optimizer in run.update.optimizers:
        for state in optimizer.state_dict()["state"].values():
            tensors.extend(v.clone() for v in state.values() if torch.is_tensor(v))
    return tensors


def _three_iterations(config: cfg.RunConfig) -> tuple[list[Any], np.ndarray, list[dict]]:
    with coordinator(config) as run:
        for _ in range(3):
            run.iterate()
        return _learner_state(run), run.buffer.action[: run.buffer.cycles].copy(), list(run.rows)


def _reference_block(folder: Path, digest: str, **regulariser: Any) -> dict[str, Any]:
    return {
        "references": {"bc": {"kind": "snapshot", "path": str(folder), "sha256": digest}},
        "regularisers": [
            {
                "kind": "reference_kl",
                "name": "bc",
                "reference": "bc",
                "budget": {"kind": "constant", "value": 0.1},
                "coef": {"start": 0.0, "max": 0.0},
                **regulariser,
            }
        ],
    }


def test_the_blind_control_over_three_iterations(tmp_path: Path) -> None:
    """Section 19.15: the block with every coefficient zero is the run without it, bit for bit.

    The reference is another seed's actor, so the KL it reports is not zero and the term's whole
    path -- the reference forward, the KL, the measurement, the gradient-ratio pass -- ran."""
    folder = tmp_path / "other-seed"
    digest = seeded_artifact(tiny_config(tmp_path / "donor", master_seed=77), folder, coordinator)
    plain = _three_iterations(tiny_config(tmp_path / "plain"))
    blind_config = with_imitation(
        tiny_config(tmp_path / "blind"), **_reference_block(folder, digest)
    )
    blind = _three_iterations(blind_config)
    assert len(plain[0]) == len(blind[0])
    for one, other in zip(plain[0], blind[0], strict=True):
        assert torch.equal(one, other)
    assert np.array_equal(plain[1], blind[1])
    assert all(row["imitation/bc/kl"] > 0.0 for row in blind[2])


def test_a_run_freezes_then_unfreezes_and_says_so(tmp_path: Path) -> None:
    from royalelearn.metrics import schema
    from royalelearn.metrics.records import unknown_keys

    config = with_imitation(
        tiny_config(tmp_path / "frozen"),
        actor_lr_scale={"kind": "piecewise", "points": [[0, 0.0], [1, 1.0]]},
    )
    with coordinator(config) as run:
        seeded = [p.detach().clone() for p in run.model.actor_parameters()]
        run.iterate()
        frozen_row = dict(run.rows[-1])
        for was, now in zip(seeded, run.model.actor_parameters(), strict=True):
            assert torch.equal(was, now)
        run.iterate()
        thawed_row = dict(run.rows[-1])
        run.iterate()
        later_row = dict(run.rows[-1])
        fired = [result.name for result in run.alarm_rows]
        run_schema = run.schema
    assert frozen_row["ppo/actor_frozen"] == 1.0
    assert not set(schema.ACTOR_UPDATE_KEYS) & set(frozen_row)
    assert "ppo/value_loss" in frozen_row and "ppo/ratio_max_abs_dev" in frozen_row
    assert thawed_row["ppo/actor_frozen"] == 0.0
    assert set(schema.ACTOR_UPDATE_KEYS) <= set(thawed_row)
    assert thawed_row["ppo/ev_at_unfreeze"] == thawed_row["ppo/explained_variance"]
    assert thawed_row["ppo/iterations_since_unfreeze"] == 1.0
    assert "ppo/ev_at_unfreeze" not in later_row
    assert later_row["ppo/iterations_since_unfreeze"] == 2.0
    assert "kl_dead" not in fired
    for row in (frozen_row, thawed_row, later_row):
        assert not unknown_keys(row, run_schema)
        assert not any(key.startswith("imitation/") for key in row)


def test_lambda_carries_across_a_resume(tmp_path: Path) -> None:
    folder = tmp_path / "other-seed"
    digest = seeded_artifact(tiny_config(tmp_path / "donor", master_seed=77), folder, coordinator)
    block = _reference_block(
        folder,
        digest,
        budget={"kind": "constant", "value": 0.0},
        coef={"start": 0.5, "max": 10.0, "up": 2.0},
    )
    config = with_imitation(tiny_config(tmp_path / "run"), **block)
    with coordinator(config) as run:
        run.iterate()
        run.iterate()
        assert [row["imitation/bc/lambda"] for row in run.rows] == [0.5, 1.0]
        saved = run.checkpoint()
    with coordinator(config, resume=saved) as resumed:
        resumed.iterate()
        assert resumed.rows[-1]["imitation/bc/lambda"] == 2.0


def test_a_reference_that_fails_its_own_probe_is_refused(tmp_path: Path) -> None:
    def nudge(tensors: dict[str, Any]) -> None:
        name = next(n for n, t in tensors.items() if t.is_floating_point() and t.numel() > 1)
        tensors[name].view(-1)[0] += 0.5

    folder = tmp_path / "nudged"
    digest = seeded_artifact(tiny_config(tmp_path / "donor"), folder, coordinator, edit=nudge)
    config = with_imitation(tiny_config(tmp_path / "run"), **_reference_block(folder, digest))
    with pytest.raises(PreflightError, match="probe-logit self-test failed"), coordinator(config):
        pass


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------


def test_play_rate_by_elixir_counts_choice_rows_at_each_whole_elixir() -> None:
    elixir = np.array([2.998, 3.2, 3.9, 5.0, 5.0, 7.0])  # 2.998 is a bar of 3 in half precision
    legal = np.array([5, 5, 5, 5, 1, 3])
    actions = np.array([0, 7, 9, 0, 0, 4])
    fields = play_rate_by_elixir(elixir, legal, actions)
    assert fields == {
        "env/play_rate_by_elixir/3": pytest.approx(2 / 3),
        "env/play_rate_by_elixir/5": 0.0,
        "env/play_rate_by_elixir/7": 1.0,
    }
