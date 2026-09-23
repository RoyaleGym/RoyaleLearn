"""The update: the algebra of the loss, and the properties the loop has to have.

Two kinds of test live here and they are kept apart on purpose.

The first kind is arithmetic. The surrogate, the dual clip, the KL estimator and the clip
fraction are pure functions of a ratio and an advantage, so they are checked against values
worked out on paper on a batch written down by hand. A test that recomputed them with the same
expression would agree with any bug in it.

The second kind is the loop's properties, checked by running the real update over a real
rectangle filled from a real environment: that the accumulated gradient of k minibatches is the
one-batch gradient, which is the whole of what makes ``minibatch_size`` a memory knob and
nothing else; that the mask that came out of the buffer is the mask that is applied; and that
the learning-rate backoff fires when it says it does and not before.

Nothing here writes down a width, an action count or a plane index. The rectangle's shape is
the environment's and the network is built from the spec read off it, so the whole file runs on
MockEngine's catalogue and would run on the full one.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn.api.schedule import ScheduleState

torch = pytest.importorskip("torch")

from royalelearn.config import LrBackoffConfig, PPOConfig  # noqa: E402
from royalelearn.learn.gae import GAE  # noqa: E402
from royalelearn.learn.inference import BatchedInference  # noqa: E402
from royalelearn.learn.ppo import (  # noqa: E402
    PPOUpdate,
    approx_kl,
    chunked_critic_pass,
    clipped_fraction,
    dual_clipped_fraction,
    explained_variance,
    standardise,
    surrogate,
    value_error,
)
from royalelearn.learn.schedules import LrBackoff  # noqa: E402
from test_buffer import CYCLES, SEED, SLOTS, Fixture, plan_for, round_for  # noqa: E402
from test_inference import build_model  # noqa: E402

#: Every trainable cell of the rectangle in one batch, so that "the one-batch gradient" is a
#: thing this fixture actually has.
SAMPLES = CYCLES * SLOTS

CONFIG = PPOConfig(
    n_epochs=1,
    timesteps_per_iteration=SAMPLES,
    batch_size=SAMPLES,
    minibatch_size=SAMPLES,
    critic_chunk=SLOTS * 2,
    # The clip bound is not what this file is about, and leaving it at its shipped value would
    # make "the accumulated gradient is the one-batch gradient" a statement about the clip.
    max_grad_norm=1e9,
    debug_assert_iterations=1,
    check_ratio_invariant_every=0,
)

SCHEDULE = ScheduleState(
    iteration=0,
    cumulative_env_steps=0,
    cumulative_timesteps=0,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    ent_coef_noop=0.02,
    lr_actor=2e-4,
    lr_critic=2e-4,
)


@pytest.fixture(scope="module")
def observations(mock_env_spec: Any) -> list[dict[str, np.ndarray]]:
    """Real observations, all different, to fill the rectangle with."""
    env = mock_env_spec.build_vec(2)
    try:
        batches = [env.reset(seed=3)[0]]
        actions = np.zeros(env.num_envs, dtype=np.int64)
        for _ in range(SAMPLES):
            batches.append(env.step(actions)[0])
    finally:
        env.close()
    return [{key: value[0] for key, value in batch.items()} for batch in batches]


@pytest.fixture
def rect(env_spec: Any, observations: list[dict[str, np.ndarray]]) -> Any:
    built = Fixture(env_spec, observations)
    built.fill()
    try:
        yield built
    finally:
        built.close()


def collect(built: Fixture, model: Any, *, iteration: int = 0) -> Any:
    """One iteration played into the rectangle, the way the coordinator plays one.

    The actions and their log-probabilities come from the real inference path against the real
    stored masks, which is what makes the importance ratio at the first minibatch meaningful
    rather than a comparison against a column of zeros.
    """
    buffer = built.buffer
    plan = plan_for(SLOTS, iteration=iteration)
    buffer.begin_iteration(plan, CYCLES)
    engine = BatchedInference(buffer, model, master_seed=SEED)
    engine.begin_iteration(plan)
    slots = np.arange(SLOTS, dtype=np.int64)
    rewards = np.random.default_rng(SEED).normal(size=(CYCLES + 1, SLOTS)).astype(np.float32)
    # Every round of the iteration, the trailing one included: it carries the bootstrap
    # observation and the last cycle's reward, and without it the last row is never completed.
    for cycle in range(CYCLES + 1):
        trailing = cycle == CYCLES
        terminated = np.zeros(SLOTS, dtype=bool)
        if cycle == CYCLES - 2:
            terminated[0] = True
        played = round_for(
            cycle,
            slots,
            rows=np.array([buffer.layout.row_index(cycle, int(s)) for s in slots]),
            reward=rewards[cycle],
            terminated=terminated,
        )
        if trailing:
            buffer.record_round(played, None, None)
            continue
        answer = engine.act(played)
        buffer.record_round(played, answer.actions, answer.log_probs)
    return buffer


def update_for(model: Any, config: PPOConfig = CONFIG, **kwargs: Any) -> PPOUpdate:
    """An update over a scaler that does not standardise, so a reward is its own number."""
    return PPOUpdate(
        model,
        GAE(standardize_rewards=False),
        config,
        master_seed=SEED,
        **kwargs,
    )


class _Recording(torch.optim.SGD):
    """An optimizer that records the gradient it was handed and changes nothing.

    A real optimizer would answer a different question: Adam divides by the square root of the
    second moment, so two gradients that agree to a part in a million can still produce
    parameters that differ by the whole step size where a coordinate's gradient is near zero.
    What has to be equal is the gradient.
    """

    def __init__(self, params: Any, **kwargs: Any) -> None:
        super().__init__(list(params), lr=float(kwargs.get("lr", 0.0)))
        self.recorded: list[torch.Tensor] = []

    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        flat = [
            parameter.grad.reshape(-1)
            if parameter.grad is not None
            else torch.zeros(parameter.numel())
            for group in self.param_groups
            for parameter in group["params"]
        ]
        self.recorded.append(torch.cat(flat).clone())


# --------------------------------------------------------------------------
# The algebra of the loss
# --------------------------------------------------------------------------


def test_the_value_clip_is_a_floor_on_the_error_and_not_a_cap_on_the_step() -> None:
    """``ppo.value_clipping`` had no reader at all, so a config could ask for it and not get it.

    What it should do, when it is on: measure the prediction twice, once as it is and once pulled
    back to within ``clip_range`` of the value the batch was collected under, and optimise the
    LARGER error. A move that overshoots is then graded on the far side of the clip, so the critic
    cannot be rewarded for going somewhere the batch does not support.
    """
    import torch

    old = torch.tensor([0.0, 0.0, 0.0])
    returns = torch.tensor([1.0, 1.0, -1.0])
    # Within the clip the two branches agree, so the error is the plain one.
    inside = torch.tensor([0.1, -0.1, 0.05])
    assert value_error(inside, old, returns, clip_range=0.2) == pytest.approx(
        float(value_error(inside, old, returns, clip_range=None))
    )
    # Outside it they do not, and the larger of the two is taken.
    over = torch.tensor([0.9, 0.9, 0.9])
    plain = float(value_error(over, old, returns, clip_range=None))
    clipped = float(value_error(over, old, returns, clip_range=0.2))
    assert clipped > plain
    pulled = old + (over - old).clamp(-0.2, 0.2)
    by_hand = torch.max((over - returns).square(), (pulled - returns).square()).mean()
    assert clipped == pytest.approx(float(by_hand))
    # A clip wide enough to bind on nothing is the unclipped error exactly.
    assert value_error(over, old, returns, clip_range=1e9) == pytest.approx(plain)


def test_the_kl_estimator_is_the_hand_computed_one() -> None:
    """Schulman's k3, sample by sample, against the same expression evaluated on paper."""
    ratios = [0.5, 0.9, 1.0, 1.3, 1.25]
    expected = [(r - 1.0) - math.log(r) for r in ratios]

    computed = approx_kl(torch.tensor(ratios, dtype=torch.float64))

    assert computed.tolist() == pytest.approx(expected, abs=1e-12)
    assert float(computed.mean()) == pytest.approx(sum(expected) / len(expected), abs=1e-12)
    # Non-negative for every ratio, which is what lets one threshold mean one thing.
    assert bool((computed >= 0).all())


def test_the_clip_fraction_counts_the_samples_outside_the_band() -> None:
    """Three of these five are more than a fifth away from one: 0.5, 1.3 and 1.25."""
    ratios = torch.tensor([0.5, 0.9, 1.0, 1.3, 1.25])

    assert float(clipped_fraction(ratios, 0.2).mean()) == pytest.approx(3.0 / 5.0)
    assert clipped_fraction(ratios, 0.2).tolist() == [1.0, 0.0, 0.0, 1.0, 1.0]


def test_at_a_ratio_of_one_the_policy_loss_is_minus_the_mean_advantage() -> None:
    """Nothing is clipped where nothing has moved, so the objective is the plain one."""
    advantages = torch.tensor([1.5, -2.0, 0.0, 0.25, -0.75])
    ratio = torch.ones_like(advantages)

    _surr, dual = surrogate(ratio, advantages, clip_range=0.2, dual_clip_c=3.0)

    assert float(-dual.mean()) == pytest.approx(float(-advantages.mean()), abs=1e-7)


def test_the_dual_clip_binds_only_where_the_advantage_is_negative() -> None:
    """The standard minimum already bounds a positive advantage; it is the negative one that
    runs away, and the lower bound is what stops it."""
    ratio = torch.tensor([0.1, 4.0, 9.0, 0.5, 6.0, 1.0])
    advantages = torch.tensor([2.0, 2.0, -1.0, -3.0, -2.0, -5.0])

    surr, dual = surrogate(ratio, advantages, clip_range=0.2, dual_clip_c=3.0)
    bound = dual != surr

    assert not bool((bound & (advantages >= 0)).any())
    assert bool(bound.any()), "this batch was chosen to make the bound bind somewhere"
    assert torch.equal(dual[bound], 3.0 * advantages[bound])
    assert torch.equal(dual[~bound], surr[~bound])
    fraction = dual_clipped_fraction(advantages, surr, dual_clip_c=3.0)
    assert fraction.tolist() == bound.float().tolist()


def test_the_dual_clip_is_the_hand_computed_value_on_one_sample() -> None:
    """``r = 9``, ``A = -1``, ``eps = 0.2``, ``c = 3``: the clipped surrogate is ``-1.2``, the
    unclipped one is ``-9``, the minimum is ``-9``, and the floor puts it back at ``-3``."""
    surr, dual = surrogate(
        torch.tensor([9.0]), torch.tensor([-1.0]), clip_range=0.2, dual_clip_c=3.0
    )

    assert float(surr) == pytest.approx(-9.0)
    assert float(dual) == pytest.approx(-3.0)


def test_standardisation_is_taken_over_the_trainable_cells_alone() -> None:
    """An opponent's rows are in the rectangle and are not trained on; letting them set the
    scale would make the gradient depend on the iteration's opponent mix."""
    advantages = torch.tensor([1.0, 2.0, 3.0, 100.0])
    mask = torch.tensor([True, True, True, False])

    standardised = standardise(advantages, mask)
    selected = advantages[mask]

    assert float(standardised[mask].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(standardised[0]) == pytest.approx(
        float((advantages[0] - selected.mean()) / (selected.std() + 1e-8)), abs=1e-6
    )


def test_explained_variance_is_one_for_a_perfect_critic_and_zero_for_the_mean() -> None:
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert explained_variance(returns, returns) == pytest.approx(1.0)
    assert explained_variance(returns, torch.full_like(returns, 2.5)) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_an_update_reports_what_its_two_pre_epoch_phases_cost(rect: Fixture) -> None:
    """Both were published as a literal 0.0, which reads as "free" rather than "unmeasured".

    They are the critic's pass over every collected cell and the advantage recursion over the
    rectangle, and they both run before the first epoch does. A reader deciding where to spend an
    optimisation was told the update was entirely epochs.
    """
    model = build_model(rect.spec)
    collect(rect, model)

    result = update_for(model).step(rect.buffer, SCHEDULE)

    assert result.critic_pass_seconds > 0.0
    assert result.gae_seconds > 0.0
    assert result.critic_pass_seconds + result.gae_seconds < result.seconds


def test_turning_value_clipping_on_reaches_the_critic(rect: Fixture) -> None:
    """The unit test above grades the arithmetic; this one grades the wiring.

    ``ppo.value_clipping`` sat in the config with no reader, so the honest test is not that the
    clipped error can be computed somewhere, it is that asking for it changes where the critic
    ends up. It binds only once the critic has moved inside the update -- the values the batch
    carries are the critic's own predictions from the pass that opens the update, so on the first
    pass the two branches are equal by construction -- which is why this runs several epochs with
    a real optimizer rather than reading one gradient. Both arms use the same clip range, so the
    only difference between them is the value branch; the actor's loss does not see the flag.
    """
    config = msgspec.structs.replace(CONFIG, n_epochs=3, clip_range=1e-4)
    model = build_model(rect.spec)
    collect(rect, model)

    plain = update_for(build_model(rect.spec), config)
    clipping = update_for(
        build_model(rect.spec), msgspec.structs.replace(config, value_clipping=True)
    )
    before = [p.detach().clone() for p in clipping.critic_params]

    plain.step(rect.buffer, SCHEDULE)
    clipping.step(rect.buffer, SCHEDULE)

    pairs = zip(clipping.critic_params, before, strict=True)
    moved = max(float((now.detach() - was).abs().max()) for now, was in pairs)
    assert moved > 0.0, "the critic did not move at all, so this test proves nothing"
    drift = max(
        float((one.detach() - other.detach()).abs().max())
        for one, other in zip(plain.critic_params, clipping.critic_params, strict=True)
    )
    assert drift > 0.0, "the flag did not reach the critic's loss"


def test_accumulated_minibatch_gradients_are_the_one_batch_gradient(rect: Fixture) -> None:
    """``minibatch_size`` is a pure memory knob, and this is the whole of what that means.

    One batch of every trainable cell against four minibatches accumulating into the same
    step, from the same weights over the same rectangle. On a four-gigabyte device this is not
    a nicety: it is the mechanism that makes the run fit.
    """
    model = build_model(rect.spec)
    collect(rect, model)
    whole = update_for(build_model(rect.spec), optimizer_factory=_Recording)
    split = update_for(
        build_model(rect.spec),
        msgspec.structs.replace(CONFIG, minibatch_size=SAMPLES // 4),
        optimizer_factory=_Recording,
    )

    whole.step(rect.buffer, SCHEDULE)
    split.step(rect.buffer, SCHEDULE)

    for one, many in (
        (whole.actor_optimizer, split.actor_optimizer),
        (whole.critic_optimizer, split.critic_optimizer),
    ):
        assert len(one.recorded) == len(many.recorded) == 1
        scale = float(one.recorded[0].abs().max())
        assert scale > 0.0
        # The two are equal in exact arithmetic and summed in a different order in float32,
        # so what is asserted is that they agree to a small fraction of the gradient's own
        # size. An absolute tolerance would be a statement about this fixture's magnitudes,
        # and it would move with the thread count, which changes the order of the reduction.
        drift = float((one.recorded[0] - many.recorded[0]).abs().max())
        assert drift <= 1e-5 * scale, f"gradients differ by {drift:.3g} on a scale of {scale:.3g}"


def test_one_optimizer_step_per_batch_whatever_the_minibatch_size(rect: Fixture) -> None:
    model = build_model(rect.spec)
    collect(rect, model)
    config = msgspec.structs.replace(CONFIG, n_epochs=2, batch_size=SAMPLES // 2)
    update = update_for(model, msgspec.structs.replace(config, minibatch_size=SAMPLES // 8))

    result = update.step(rect.buffer, SCHEDULE)

    assert result.n_optimizer_steps == 4, "two epochs of two batches"
    assert result.n_minibatches == 16
    assert update.model_updates == 4


def test_nothing_collected_is_left_out_of_the_update(rect: Fixture) -> None:
    """The remainder of an epoch is a smaller batch, weighted by its true sample count, and not
    a discarded one."""
    model = build_model(rect.spec)
    collect(rect, model)
    config = msgspec.structs.replace(
        CONFIG, n_epochs=3, batch_size=SAMPLES // 5, minibatch_size=7
    )

    result = update_for(model, config).step(rect.buffer, SCHEDULE)

    assert result.n_samples == SAMPLES
    assert result.samples_unused_frac == 0.0


def test_the_ratio_is_one_at_the_first_minibatch_of_a_fresh_iteration(rect: Fixture) -> None:
    """The parameters are the ones that acted and the bytes are the ones they acted on, so the
    two forwards are the same forward. Nothing here relaxes the tolerance."""
    model = build_model(rect.spec)
    collect(rect, model)

    result = update_for(model).step(rect.buffer, SCHEDULE)

    assert result.ratio_max_abs_dev <= CONFIG.ratio_atol["float32"]


def test_a_weight_version_mismatch_trips_the_ratio_invariant(rect: Fixture) -> None:
    """The failure the check exists for: the policy that acted is not the policy being
    updated. It moves the ratio by a quantity of order one, not of order a percent."""
    model = build_model(rect.spec)
    collect(rect, model)
    with torch.no_grad():
        for parameter in model.actor_parameters():
            parameter.add_(0.5)

    with pytest.raises(AssertionError, match="importance ratio deviates"):
        update_for(model).step(rect.buffer, SCHEDULE)


def test_the_mask_assert_fires_when_an_action_is_not_legal_under_its_own_mask(
    rect: Fixture,
) -> None:
    """A rollout that sampled outside its mask, or an update shown somebody else's mask. Both
    are silent for hours otherwise: unmasked at update pins the clip fraction at one, and a
    wrong mask gives a log-probability of ``finfo.min`` and then NaN."""
    from royalelearn.learn.inference import RectGather

    model = build_model(rect.spec)
    buffer = collect(rect, model)
    obs = RectGather(buffer, rows=1).observations(np.array([0]), np.array([0]))
    illegal = int(torch.nonzero(~obs.mask[0]).reshape(-1)[0])
    buffer.action[0, 0] = illegal

    with pytest.raises(AssertionError, match="stored mask forbids"):
        update_for(model).step(buffer, SCHEDULE)


def test_the_asserts_stop_after_the_iterations_they_were_asked_for(rect: Fixture) -> None:
    """They are a start-up gate on a run's wiring, not a per-sample cost for its whole life."""
    from royalelearn.learn.inference import RectGather

    model = build_model(rect.spec)
    buffer = collect(rect, model)
    obs = RectGather(buffer, rows=1).observations(np.array([0]), np.array([0]))
    buffer.action[0, 0] = int(torch.nonzero(~obs.mask[0]).reshape(-1)[0])
    config = msgspec.structs.replace(
        CONFIG, debug_assert_iterations=0, check_ratio_invariant_every=0
    )

    update_for(model, config).step(buffer, msgspec.structs.replace(SCHEDULE, iteration=5))


def test_the_kl_and_the_clip_fraction_are_reported_per_epoch(rect: Fixture) -> None:
    """"Epoch three's clip fraction is more than twice epoch one's" is the rule for lowering
    ``n_epochs``, and an average over the epochs cannot answer it."""
    model = build_model(rect.spec)
    collect(rect, model)
    config = msgspec.structs.replace(CONFIG, n_epochs=3)

    result = update_for(model, config).step(rect.buffer, SCHEDULE)

    assert len(result.kl_by_epoch) == 3
    assert len(result.clip_fraction_by_epoch) == 3
    assert result.kl_by_epoch[0] == pytest.approx(0.0, abs=1e-6), "nothing has moved yet"
    assert result.kl_by_epoch[-1] > result.kl_by_epoch[0]


def test_the_last_epoch_is_reported_when_the_batch_size_does_not_divide_the_rows(
    rect: Fixture,
) -> None:
    """The epoch a batch is labelled with is counted, and it was counted a second way.

    The update divided the trainable rows by the batch size and rounded UP, which was right while
    an epoch ended in a remainder batch. Since the rows over are spread across whole batches there
    are FEWER batches than that ceiling, so the labels ran ahead of the batches and the last epoch
    was never reached: with three epochs the diagnostics for the third stayed at their initial
    value and the progress line stopped saying "epoch 3/3". The test above cannot see it, because
    its batch size divides its fixture exactly and the two counts agree. This one picks a size
    that does not.

    It is a diagnostics defect and not a learning one: the label reaches the per-epoch numbers and
    the progress line, never a gradient. But "epoch three's clip fraction against epoch one's" is
    the documented rule for lowering n_epochs, so a reader would act on it.
    """
    from royalelearn.learn.buffer import batch_count

    model = build_model(rect.spec)
    collect(rect, model)
    rows = int(rect.buffer.trainable().sum())
    size = (rows // 3) + 1
    assert rows % size, "this test needs a batch size that does not divide the rows"
    config = msgspec.structs.replace(CONFIG, n_epochs=3, batch_size=size)
    assert batch_count(rows, size) < -(-rows // size), "and one the old ceiling overcounts"

    result = update_for(model, config).step(rect.buffer, SCHEDULE)

    assert len(result.kl_by_epoch) == 3
    assert result.kl_by_epoch[-1] != 0.0, "the last epoch was never labelled"


def test_the_entropy_bonus_reaches_no_critic_parameter(rect: Fixture) -> None:
    """Separate trunks make it structural rather than a matter of remembering to detach: there
    is no tensor between the two, so the gradient has no path to take."""
    from royalelearn.learn.inference import RectGather

    model = build_model(rect.spec)
    collect(rect, model)
    obs = RectGather(rect.buffer, rows=SLOTS).observations(
        np.zeros(SLOTS, dtype=np.int64), np.arange(SLOTS, dtype=np.int64)
    )
    actions = torch.from_numpy(rect.buffer.action[0].astype(np.int64))

    result = model.backprop(obs, actions)
    result.entropy.mean().backward()

    for parameter in model.critic_parameters():
        assert parameter.grad is None or bool((parameter.grad == 0).all())


def test_the_critic_pass_covers_the_bootstrap_cycle(rect: Fixture) -> None:
    """The last cycle's advantage needs the value of the observation after it, and that row is
    in the rectangle for exactly this reason."""
    from royalelearn.learn.inference import RectGather

    model = build_model(rect.spec)
    buffer = collect(rect, model)
    gather = RectGather(buffer, rows=SLOTS * 2)

    values = chunked_critic_pass(buffer, model, gather, chunk=SLOTS * 2)

    assert values.shape == (CYCLES + 1, SLOTS)
    assert bool(torch.isfinite(values).all())
    assert float(values.abs().sum()) > 0.0


def test_a_cell_no_worker_wrote_has_no_value(rect: Fixture) -> None:
    """A dead worker's rows fall out of the recursion without a special case: no reward, no
    value and the episode flagged ended, so the loop runs over them inertly."""
    from royalelearn.learn.inference import RectGather

    model = build_model(rect.spec)
    buffer = collect(rect, model)
    buffer.valid[1, 3] = False
    gather = RectGather(buffer, rows=SLOTS * 2)

    values = chunked_critic_pass(buffer, model, gather, chunk=SLOTS * 2)

    assert float(values[1, 3]) == 0.0


# --------------------------------------------------------------------------
# The observation a truncated episode ended on
# --------------------------------------------------------------------------


def packed(built: Fixture, observation: dict[str, np.ndarray]) -> np.ndarray:
    """One observation as the worker would have written it into the rectangle."""
    row = bytearray(built.codec.row_bytes(built.spec))
    view = memoryview(row)
    try:
        built.codec.pack(observation, view, 0)
    finally:
        view.release()
    return np.frombuffer(bytes(row), dtype=np.uint8)


def cut_iteration(built: Fixture, model: Any, *, cycle: int, slot: int) -> Any:
    """An iteration in which one seat's episode was cut rather than decided.

    ``cycle`` is the ROW that was truncated. The round that reports a cut is the one the cut
    step arrived at, which is the cycle above it.
    """
    buffer = built.buffer
    plan = plan_for(SLOTS)
    buffer.begin_iteration(plan, CYCLES)
    engine = BatchedInference(buffer, model, master_seed=SEED)
    engine.begin_iteration(plan)
    slots = np.arange(SLOTS, dtype=np.int64)
    for step in range(CYCLES + 1):
        truncated = np.zeros(SLOTS, dtype=bool)
        truncated[slot] = step == cycle + 1
        played = round_for(
            step,
            slots,
            rows=np.array([buffer.layout.row_index(step, int(s)) for s in slots]),
            truncated=truncated,
        )
        if step == CYCLES:
            buffer.record_round(played, None, None)
            continue
        answer = engine.act(played)
        buffer.record_round(played, answer.actions, answer.log_probs)
    return buffer


def test_a_truncated_cell_is_bootstrapped_from_the_observation_it_ended_on(
    rect: Fixture, observations: list[dict[str, np.ndarray]]
) -> None:
    """What the rectangle holds at a truncated cell is the FIRST observation of the next
    episode, because the environment resets on the step that ends one. The value that belongs
    there is the one of the position the cut interrupted, and it arrives as packed bytes.

    The observation handed over here is also the one the rectangle happens to hold at another
    cell, so what its value should be is a number this test already has.
    """
    twin = 5
    cycle, slot = 2, 3
    model = build_model(rect.spec)
    buffer = cut_iteration(rect, model, cycle=cycle, slot=slot)
    update = update_for(model)
    update.record_final_observations(
        cycle, np.array([slot]), packed(rect, observations[twin])[None, :]
    )

    update.step(buffer, SCHEDULE)

    assert buffer.final_value[cycle, slot] == pytest.approx(
        float(buffer.value[twin // SLOTS, twin % SLOTS]), abs=1e-5
    )


def test_a_final_observation_for_a_cell_that_was_not_cut_is_refused(
    rect: Fixture, observations: list[dict[str, np.ndarray]]
) -> None:
    """A final observation belongs to the transition that was cut. Anywhere else it would be
    bootstrapping a live trajectory off a state it never reached."""
    model = build_model(rect.spec)
    buffer = cut_iteration(rect, model, cycle=2, slot=3)
    update = update_for(model)
    update.record_final_observations(
        1, np.array([4]), packed(rect, observations[0])[None, :]
    )

    with pytest.raises(ValueError, match="marked truncated"):
        update.step(buffer, SCHEDULE)


# --------------------------------------------------------------------------
# The learning-rate backoff
# --------------------------------------------------------------------------


def _breaching_update(rect: Fixture, backoff: LrBackoff) -> PPOUpdate:
    model = build_model(rect.spec)
    collect(rect, model)
    config = msgspec.structs.replace(
        CONFIG, debug_assert_iterations=0, check_ratio_invariant_every=0
    )
    return update_for(model, config, backoff=backoff)


def test_the_backoff_fires_after_exactly_the_patience_it_was_given(rect: Fixture) -> None:
    """Consecutive breaches, because what this guards against is a policy walking away from its
    behaviour distribution and not one noisy update."""
    # A threshold below zero: the k3 estimator is non-negative, so every iteration breaches.
    backoff = LrBackoff(
        LrBackoffConfig(kl_threshold=-1.0, patience=3, factor=0.5, lr_min=1e-9),
        lr_actor=2e-4,
        lr_critic=4e-4,
    )
    update = _breaching_update(rect, backoff)

    for iteration in range(2):
        update.step(rect.buffer, msgspec.structs.replace(SCHEDULE, iteration=iteration))
        assert backoff.lr_actor == 2e-4
        assert backoff.events == 0

    update.step(rect.buffer, msgspec.structs.replace(SCHEDULE, iteration=2))

    assert backoff.events == 1
    assert backoff.lr_actor == 1e-4
    assert backoff.lr_critic == 2e-4
    assert backoff.consecutive_breaches == 0


def test_a_quiet_iteration_resets_the_counter(rect: Fixture) -> None:
    backoff = LrBackoff(
        LrBackoffConfig(kl_threshold=1e9, patience=1, factor=0.5, lr_min=1e-9),
        lr_actor=2e-4,
        lr_critic=2e-4,
    )
    update = _breaching_update(rect, backoff)

    update.step(rect.buffer, SCHEDULE)

    assert backoff.events == 0
    assert backoff.consecutive_breaches == 0
    assert backoff.lr_actor == 2e-4


def test_the_backoff_floors_at_lr_min(rect: Fixture) -> None:
    backoff = LrBackoff(
        LrBackoffConfig(kl_threshold=-1.0, patience=1, factor=0.5, lr_min=1.5e-4),
        lr_actor=2e-4,
        lr_critic=2e-4,
    )
    update = _breaching_update(rect, backoff)

    for iteration in range(3):
        update.step(rect.buffer, msgspec.structs.replace(SCHEDULE, iteration=iteration))

    assert backoff.events == 3
    assert backoff.lr_actor == 1.5e-4
    assert backoff.lr_critic == 1.5e-4


def test_the_rates_the_schedule_carries_are_the_rates_the_optimizers_run_at(
    rect: Fixture,
) -> None:
    model = build_model(rect.spec)
    collect(rect, model)
    update = update_for(model)

    update.step(rect.buffer, msgspec.structs.replace(SCHEDULE, lr_actor=7e-5, lr_critic=3e-5))

    assert all(group["lr"] == 7e-5 for group in update.actor_optimizer.param_groups)
    assert all(group["lr"] == 3e-5 for group in update.critic_optimizer.param_groups)


# --------------------------------------------------------------------------
# The checkpoint
# --------------------------------------------------------------------------


def test_the_optimizer_state_survives_a_round_trip(rect: Fixture, tmp_path: Path) -> None:
    model = build_model(rect.spec)
    collect(rect, model)
    update = update_for(model)
    update.step(rect.buffer, SCHEDULE)
    update.save_checkpoint(tmp_path)

    restored = update_for(build_model(rect.spec))
    restored.load_checkpoint(tmp_path, strict=True)

    assert restored.model_updates == update.model_updates
    saved = update.actor_optimizer.state_dict()["state"]
    loaded = restored.actor_optimizer.state_dict()["state"]
    assert set(saved) == set(loaded)
    for key in saved:
        assert torch.allclose(saved[key]["exp_avg"], loaded[key]["exp_avg"])


def test_a_configured_rate_is_restored_over_a_loaded_one(rect: Fixture, tmp_path: Path) -> None:
    """Changing the learning rate for the resumed half of a run has to take effect, and an
    optimizer state dict carries the rate the run was saved at."""
    model = build_model(rect.spec)
    collect(rect, model)
    update = update_for(model)
    update.step(rect.buffer, msgspec.structs.replace(SCHEDULE, lr_actor=9e-4))
    update.save_checkpoint(tmp_path)

    restored = update_for(build_model(rect.spec), msgspec.structs.replace(CONFIG, lr_actor=1e-5))
    restored.load_checkpoint(tmp_path, strict=True)

    assert all(group["lr"] == 1e-5 for group in restored.actor_optimizer.param_groups)


def test_a_missing_optimizer_state_is_an_error_on_a_strict_load(
    rect: Fixture, tmp_path: Path
) -> None:
    """Tolerate-everything is right for a research tool and wrong for a harness that promises
    the curve continues: a resume that silently restarts Adam's moments is a visible step in
    the loss at the point the run was resumed."""
    from royalelearn.errors import CheckpointFormatError

    update = update_for(build_model(rect.spec))

    with pytest.raises(CheckpointFormatError, match="optimizer state"):
        update.load_checkpoint(tmp_path, strict=True)

    update.load_checkpoint(tmp_path, strict=False)
    assert update.model_updates == 0


def test_the_play_wait_entropy_ignores_rows_that_had_no_choice(rect: Fixture) -> None:
    """A decision the elixir bar cannot afford has a binary entropy of zero by construction.

    On the shipped environment most decisions are exactly that, so a mean over every row
    reports the elixir curve rather than the policy: it sits near zero on a healthy run, a
    floor set on it fires permanently, and a gate that really had collapsed would move it by a
    fraction of what it moves on the rows that had a choice.
    """
    import torch

    from royalelearn.learn.ppo import _Diagnostics

    device = torch.device("cpu")

    def record(diagnostics: _Diagnostics, noop: list[float], legal: list[int]) -> None:
        count = len(noop)
        result = SimpleNamespace(
            log_probs=torch.zeros(count),
            entropy=torch.zeros(count),
            noop_entropy=torch.tensor(noop),
            values=torch.zeros(count),
            n_legal=torch.tensor(legal),
        )
        diagnostics.minibatch(
            epoch=0,
            n=count,
            ratio=torch.ones(count),
            advantages=torch.zeros(count),
            surr=torch.zeros(count),
            dual=torch.zeros(count),
            result=result,
            value_loss=torch.zeros(()),
            clip_range=0.2,
            dual_clip_c=3.0,
        )

    # Three rows had a choice and carried 0.6 nats; seventeen could only wait.
    conditioned = _Diagnostics(1, device)
    record(conditioned, [0.6] * 3 + [0.0] * 17, [40] * 3 + [1] * 17)
    out = conditioned.result(
        n_samples=20, epochs=1, explained=0.0, update_actor=0.0, update_critic=0.0, seconds=0.0
    )

    assert out.noop_entropy == pytest.approx(0.6, abs=1e-6), (
        "the play/wait entropy averaged over rows that had no play to choose"
    )


def test_every_ratio_diagnostic_ignores_rows_that_could_not_move(rect: Fixture) -> None:
    """A forced row's ratio is exactly one: the distribution is a point mass, before and after.

    So it contributes zero KL and zero clipping structurally, and an unconditioned mean is the
    choice-bearing mean times a fraction the elixir economy sets. That fraction moves as the
    policy learns to hold elixir and again in overtime, so a threshold against it is wrong in a
    way no re-tuning fixes. It reaches past the dashboard: the learning-rate backoff reads this
    KL, and a diluted one puts the brake out of reach by the same factor.
    """
    import torch

    from royalelearn.learn.ppo import _Diagnostics

    device = torch.device("cpu")
    diagnostics = _Diagnostics(1, device)
    count = 20
    # Four rows had a choice and moved; sixteen could only wait, and their ratio is one.
    chose = [True] * 4 + [False] * 16
    ratio = torch.tensor([1.5] * 4 + [1.0] * 16)
    result = SimpleNamespace(
        log_probs=torch.zeros(count),
        entropy=torch.zeros(count),
        noop_entropy=torch.zeros(count),
        values=torch.zeros(count),
        n_legal=torch.tensor([40 if c else 1 for c in chose]),
    )
    diagnostics.minibatch(
        epoch=0,
        n=count,
        ratio=ratio,
        advantages=torch.zeros(count),
        surr=torch.zeros(count),
        dual=torch.zeros(count),
        result=result,
        value_loss=torch.zeros(()),
        clip_range=0.2,
        dual_clip_c=3.0,
    )
    out = diagnostics.result(
        n_samples=count, epochs=1, explained=0.0, update_actor=0.0, update_critic=0.0, seconds=0.0
    )

    from royalelearn.learn.ppo import approx_kl

    expected_kl = float(approx_kl(torch.tensor([1.5])).item())
    assert out.kl == pytest.approx(expected_kl, rel=1e-5), (
        "the KL was averaged over rows whose ratio is one by construction"
    )
    assert out.clip_fraction == pytest.approx(1.0, abs=1e-6), (
        "every row that could move was outside the clip band, so the fraction is one"
    )
