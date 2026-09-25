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
from typing import Any

import msgspec
import numpy as np
import pytest

from royalegym.action import NOOP

torch = pytest.importorskip("torch")

from royalelearn.config import FORCED_ROW_ARMS, LrBackoffConfig  # noqa: E402
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

# The update harness is defined once, in ``royalelearn.testing``.
from royalelearn.testing import (  # noqa: E402
    FORCED_CELLS,
    SAMPLES,
    collect,
    plant_forced,
    update_for,
)
from royalelearn.testing import PPO_CONFIG as CONFIG  # noqa: E402
from royalelearn.testing import PPO_SCHEDULE as SCHEDULE  # noqa: E402
from royalelearn.testing import RecordingSGD as _Recording  # noqa: E402
from royalelearn.testing import observations as fresh_observations  # noqa: E402
from test_buffer import CYCLES, SEED, SLOTS, Fixture, plan_for, round_for  # noqa: E402
from test_inference import build_model  # noqa: E402


@pytest.fixture(scope="module")
def observations(mock_env_spec: Any) -> list[dict[str, np.ndarray]]:
    """Real observations, all different, to fill the rectangle with."""
    return fresh_observations(mock_env_spec, seed=3, steps=SAMPLES)


@pytest.fixture
def rect(env_spec: Any, observations: list[dict[str, np.ndarray]]) -> Any:
    built = Fixture(env_spec, observations)
    built.fill()
    try:
        yield built
    finally:
        built.close()


#: Two slots whose every cell is forced. They are the padding an arm either divides by or does
#: not: their rows reach the critic and the recursion and carry no policy gradient at all.
PADDING_SLOTS = (SLOTS - 2, SLOTS - 1)
PADDING_CELLS = [(cycle, slot) for cycle in range(CYCLES) for slot in PADDING_SLOTS]


def count_rows(module: Any) -> list[int]:
    """Record how many rows each forward of this module is handed, in order.

    The row count is the whole of what a skipping arm changes about the actor, and it is not
    visible in a gradient: a forward over every row and a forward over the rows that can move
    produce the same gradient, which is the point. So it is counted at the module.
    """
    seen: list[int] = []
    inner = module.forward

    def forward(obs: Any) -> Any:
        seen.append(int(obs.mask.shape[0]))
        return inner(obs)

    module.forward = forward
    return seen


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


@pytest.mark.parametrize("arm", FORCED_ROW_ARMS)
def test_accumulated_minibatch_gradients_are_the_one_batch_gradient(
    rect: Fixture, arm: str
) -> None:
    """``minibatch_size`` is a pure memory knob, and this is the whole of what that means.

    One batch of every trainable cell against four minibatches accumulating into the same
    step, from the same weights over the same rectangle. On a four-gigabyte device this is not
    a nicety: it is the mechanism that makes the run fit.

    It holds under every value of ``ppo.forced_rows`` because every denominator any of them uses
    is a property of the BATCH: the batch's row count, or the count of its rows that had a
    choice. A denominator taken per minibatch would make the gradient a function of the
    partition, which is the property this test exists to deny.
    """
    model = build_model(rect.spec)
    plant_forced(rect, FORCED_CELLS)
    collect(rect, model)
    config = msgspec.structs.replace(CONFIG, forced_rows=arm)
    whole = update_for(build_model(rect.spec), config, optimizer_factory=_Recording)
    split = update_for(
        build_model(rect.spec),
        msgspec.structs.replace(config, minibatch_size=SAMPLES // 4),
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


@pytest.mark.parametrize("minibatch_size", [SAMPLES, SAMPLES // 4])
def test_letting_the_actor_skip_forced_rows_lands_on_the_gradient_all_produces(
    rect: Fixture, minibatch_size: int
) -> None:
    """``critic_only`` is a speed change and nothing else, which is a claim about a gradient.

    A row whose mask leaves one action has a log-probability of exactly zero under any
    parameters, so its ratio is exactly one, its surrogate is its own advantage, its entropy is
    zero and its play/wait entropy is a clamped constant. It therefore adds a constant to each of
    the actor's three numerators and a full row to its denominator. ``critic_only`` leaves the
    row out of the numerator sums and keeps the same denominator, so the gradient is the same
    one in exact arithmetic and the same one to rounding in float32.

    Two arms over one rectangle, from identical weights. The equality is asserted against the
    gradient's own scale rather than an absolute tolerance, because the two are summed in a
    different order and the order moves with the thread count.

    The row counts are asserted too, and they are why this test cannot pass by doing nothing:
    the gradients of two arms that both trained on every row would also agree. The critic's
    forward sees the whole rectangle under both.
    """
    model = build_model(rect.spec)
    plant_forced(rect, FORCED_CELLS)
    collect(rect, model)
    config = msgspec.structs.replace(CONFIG, minibatch_size=minibatch_size)
    every_model, skipping_model = build_model(rect.spec), build_model(rect.spec)
    every_actor, every_critic = count_rows(every_model.actor), count_rows(every_model.critic)
    skip_actor, skip_critic = count_rows(skipping_model.actor), count_rows(skipping_model.critic)
    every = update_for(every_model, config, optimizer_factory=_Recording)
    skipping = update_for(
        skipping_model,
        msgspec.structs.replace(config, forced_rows="critic_only"),
        optimizer_factory=_Recording,
    )

    plain = every.step(rect.buffer, SCHEDULE)
    skipped = skipping.step(rect.buffer, SCHEDULE)

    rows = int(rect.buffer.trainable().sum())
    choice = int(((rect.buffer.n_legal > 1) & rect.buffer.trainable()).sum())
    assert 0 < choice < rows, "this rectangle needs both kinds of row for the test to mean any"
    if minibatch_size < rows:
        assert choice % minibatch_size, (
            "this partition needs choice rows that cross a minibatch boundary, or the skip is "
            "only ever tested on whole minibatches"
        )
    # The critic pass runs the critic over the rectangle and the bootstrap cycle as well, so
    # what is compared is the epochs' share: every row of every epoch, under both arms.
    assert sum(skip_critic[-len(skip_critic) :]) == sum(every_critic[-len(every_critic) :])
    assert sum(every_actor) == rows * config.n_epochs
    assert sum(skip_actor) == choice * config.n_epochs

    for one, other in (
        (every.actor_optimizer, skipping.actor_optimizer),
        (every.critic_optimizer, skipping.critic_optimizer),
    ):
        assert len(one.recorded) == len(other.recorded) >= 1
        for mine, theirs in zip(one.recorded, other.recorded, strict=True):
            scale = float(mine.abs().max())
            assert scale > 0.0
            drift = float((mine - theirs).abs().max())
            assert drift <= 1e-5 * scale, (
                f"gradients differ by {drift:.3g} on a scale of {scale:.3g}"
            )

    # And the numbers the update reports about itself, which a reader compares across the arms.
    assert skipped.value_loss == pytest.approx(plain.value_loss, rel=1e-6)
    assert skipped.explained_variance == pytest.approx(plain.explained_variance, rel=1e-6)
    # An absolute tolerance, and it is not a weaker statement. `ppo/policy_loss` is a mean over
    # the batch of terms of unit scale after standardisation, and those terms nearly cancel, so
    # the key reads about 1e-7 here and a relative tolerance on it is a statement about float32's
    # last bit. What it would catch is the failure worth catching: leaving the skipped rows out
    # of this key entirely puts it at the mean of the choice rows' advantages alone, about 0.1 on
    # this fixture, five orders above this bound.
    assert skipped.policy_loss == pytest.approx(plain.policy_loss, abs=1e-6)
    for key in ("kl", "clip_fraction", "entropy", "noop_entropy", "entropy_normalised"):
        assert getattr(skipped, key) == pytest.approx(getattr(plain, key), rel=1e-5), key


@pytest.mark.parametrize("arm", ["critic_only", "critic_only_choice_mean"])
def test_only_the_choice_mean_ignores_how_much_padding_a_batch_carries(
    rect: Fixture, arm: str
) -> None:
    """This is the whole of what ``critic_only_choice_mean`` is for, in one number.

    Two slots are made entirely forced and then, in the second run, taken out of the update by
    marking them somebody else's seat. Nothing else moves: the recursion is per slot and does
    not read the trainable flag, so the cells that did have a choice get the same advantages and
    returns either way, which this asserts rather than assumes.

    Under ``critic_only`` the actor's denominator is the batch, so removing padding scales its
    gradient by the ratio of the two row counts -- 48 rows to 36 here, which is the factor the
    elixir bar moves on a real run as the policy learns to hold elixir and again in overtime.
    Under ``critic_only_choice_mean`` the denominator is the rows that had a choice, and those
    did not change, so the gradient is the same one to rounding.

    Standardisation is off so that the two runs share one set of advantages; which population
    sets the scale is the next test's subject, not this one's.
    """
    model = build_model(rect.spec)
    plant_forced(rect, PADDING_CELLS)
    buffer = collect(rect, model)
    config = msgspec.structs.replace(
        CONFIG, forced_rows=arm, advantage_standardization=False
    )
    padded = update_for(build_model(rect.spec), config, optimizer_factory=_Recording)

    padded.step(buffer, SCHEDULE)
    with_padding = padded.actor_optimizer.recorded[0]
    kept = np.array([slot not in PADDING_SLOTS for slot in range(SLOTS)])
    advantages = buffer.advantage[:CYCLES, kept].copy()
    returns = buffer.ret[:CYCLES, kept].copy()

    buffer.group[:, list(PADDING_SLOTS)] = 0  # a resident snapshot's index: not the learner
    bare = update_for(build_model(rect.spec), config, optimizer_factory=_Recording)
    bare.step(buffer, SCHEDULE)
    without = bare.actor_optimizer.recorded[0]

    assert np.array_equal(buffer.advantage[:CYCLES, kept], advantages), (
        "the cells that had a choice were given different advantages, so the two runs are not "
        "the same experiment"
    )
    assert np.array_equal(buffer.ret[:CYCLES, kept], returns)
    rows, choice = SAMPLES, int(kept.sum()) * CYCLES
    assert (rows, choice) == (48, 36)
    scale = float(without.abs().max())
    expected = without if arm == "critic_only_choice_mean" else without * (choice / rows)
    drift = float((with_padding - expected).abs().max())
    assert drift <= 1e-5 * scale, (
        f"the actor's gradient with padding is {drift:.3g} from the expected one on a scale of "
        f"{scale:.3g}"
    )
    if arm == "critic_only":
        # And the factor is not one, or the assertion above would hold for either arm.
        assert float((with_padding - without).abs().max()) > 1e-3 * scale


@pytest.mark.parametrize("arm", FORCED_ROW_ARMS)
def test_the_advantages_are_scaled_by_the_cells_that_will_read_them(
    rect: Fixture, arm: str
) -> None:
    """``standardise`` centres and scales over the cells that reach the update, and under the
    choice mean a forced cell no longer does.

    It reaches the critic, which never reads an advantage. Leaving the forced cells in the
    statistics would leave the actor's baseline and its scale following the elixir bar through
    the back door, after the denominator had been taken off it -- and it would move the entropy
    terms' weight against the policy term by the same ratio.
    """
    model = build_model(rect.spec)
    plant_forced(rect, FORCED_CELLS)
    buffer = collect(rect, model)

    update_for(
        build_model(rect.spec), msgspec.structs.replace(CONFIG, forced_rows=arm)
    ).step(buffer, SCHEDULE)

    trainable = buffer.trainable()
    choice = trainable & (buffer.n_legal[:CYCLES] > 1)
    forced = trainable & ~choice
    assert choice.any() and forced.any()
    population = choice if arm == "critic_only_choice_mean" else trainable
    selected = buffer.advantage[:CYCLES][population]

    assert float(selected.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(selected.std(ddof=1)) == pytest.approx(1.0, abs=1e-6)
    # The other population is not the centred one, so the assertion above names which was used.
    other = forced if arm == "critic_only_choice_mean" else choice
    assert abs(float(buffer.advantage[:CYCLES][other].mean())) > 1e-3


def test_the_two_skipping_arms_standardise_over_different_cells(rect: Fixture) -> None:
    """The arms are told apart by the numbers they produce and not only by a population count."""
    model = build_model(rect.spec)
    plant_forced(rect, FORCED_CELLS)
    buffer = collect(rect, model)

    update_for(
        build_model(rect.spec), msgspec.structs.replace(CONFIG, forced_rows="critic_only")
    ).step(buffer, SCHEDULE)
    batch_scaled = buffer.advantage[:CYCLES].copy()
    update_for(
        build_model(rect.spec),
        msgspec.structs.replace(CONFIG, forced_rows="critic_only_choice_mean"),
    ).step(buffer, SCHEDULE)

    assert not np.allclose(batch_scaled, buffer.advantage[:CYCLES], atol=1e-4)


def test_the_choice_mean_moves_the_actor_by_the_ratio_of_the_denominators(
    rect: Fixture,
) -> None:
    """Section 18 item 8's own test: the parameter delta after a real Adam step, not the
    gradient norm.

    Adam largely cancels a uniform rescale of the loss, because the first moment and the root of
    the second scale together, so a gradient-norm comparison between these two arms reports a
    clean factor while the weights move almost identically. What is left of the rescale is
    ``eps``, and on the laptop's optimizer state almost every actor coordinate sits below it.

    So the factor is measured in both regimes. With ``adam_eps`` far above the largest gradient
    coordinate the step is ``lr * g / eps``, linear in the gradient, and the choice mean moves
    the actor by exactly the ratio of the two denominators. With ``adam_eps`` at 1e-12 the first
    step is ``lr * sign(g)`` and the two arms land in the same place. That pair is what shows
    the difference is eps and not the gradient.

    Advantage standardisation is off, so the only thing between the two arms is the denominator.
    The learning rate is raised for this test alone: what is compared is a difference of
    parameters, and at the shipped rate the step is a ten-millionth of a weight, which float32
    cannot subtract without quantising the answer to a percent of itself.
    """
    from torch.nn.utils import parameters_to_vector

    model = build_model(rect.spec)
    plant_forced(rect, FORCED_CELLS)
    buffer = collect(rect, model)
    base = msgspec.structs.replace(CONFIG, advantage_standardization=False)
    sched = msgspec.structs.replace(SCHEDULE, lr_actor=0.5, lr_critic=0.5)
    probe = update_for(
        build_model(rect.spec),
        msgspec.structs.replace(base, forced_rows="critic_only"),
        optimizer_factory=_Recording,
    )
    probe.step(buffer, sched)
    largest = float(probe.actor_optimizer.recorded[0].abs().max())
    assert largest > 0.0

    def deltas(arm: str, eps: float) -> tuple[Any, Any]:
        update = update_for(
            build_model(rect.spec),
            msgspec.structs.replace(base, forced_rows=arm, adam_eps=eps),
        )
        before = parameters_to_vector(update.actor_params).detach().clone()
        before_critic = parameters_to_vector(update.critic_params).detach().clone()
        update.step(buffer, sched)
        return (
            parameters_to_vector(update.actor_params).detach() - before,
            parameters_to_vector(update.critic_params).detach() - before_critic,
        )

    rows = int(buffer.trainable().sum())
    choice = int(((buffer.n_legal[:CYCLES] > 1) & buffer.trainable()).sum())
    assert 0 < choice < rows
    expected = rows / choice

    bound_batch, bound_batch_critic = deltas("critic_only", 10_000.0 * largest)
    bound_choice, bound_choice_critic = deltas("critic_only_choice_mean", 10_000.0 * largest)
    assert float(bound_choice.norm() / bound_batch.norm()) == pytest.approx(expected, rel=1e-3)
    # And it is the same step scaled, not a different one of the right size.
    assert float((bound_choice - bound_batch * expected).abs().max()) <= 1e-2 * float(
        bound_choice.abs().max()
    )

    sign_batch, sign_batch_critic = deltas("critic_only", 1e-12)
    sign_choice, sign_choice_critic = deltas("critic_only_choice_mean", 1e-12)
    assert float(sign_choice.norm() / sign_batch.norm()) == pytest.approx(1.0, rel=1e-3), (
        "away from the eps-bound regime Adam's first step is the rate times the sign of the "
        "gradient, so a denominator cannot reach it"
    )
    assert torch.equal(sign_choice.sign(), sign_batch.sign())
    assert torch.allclose(sign_choice, sign_batch, rtol=1e-2, atol=1e-9)

    for mine, theirs in (
        (bound_batch_critic, bound_choice_critic),
        (sign_batch_critic, sign_choice_critic),
    ):
        assert torch.equal(mine, theirs), "the critic's step is not this field's business"


@pytest.mark.parametrize("arm", FORCED_ROW_ARMS)
def test_an_update_says_how_many_rows_its_actor_actually_ran_on(
    rect: Fixture, arm: str
) -> None:
    """Two arms that differ only in what the actor was shown look identical in every other key.

    The point of the whole field is a row count, and a row count is the one thing a gradient, a
    loss and a KL cannot show: ``critic_only`` is built so that they do not move. So the count
    is published, and ``ppo/forced_frac`` is published beside it from the column rather than
    from the rollout's sample, so that a reader can check one against the other and check both
    against the arm the run identity names.
    """
    model = build_model(rect.spec)
    plant_forced(rect, FORCED_CELLS)
    buffer = collect(rect, model)
    config = msgspec.structs.replace(CONFIG, n_epochs=3, forced_rows=arm)
    counted = build_model(rect.spec)
    seen = count_rows(counted.actor)

    result = update_for(counted, config).step(buffer, SCHEDULE)

    rows = int(buffer.trainable().sum())
    choice = int(((buffer.n_legal[:CYCLES] > 1) & buffer.trainable()).sum())
    expected = rows if arm == "all" else choice
    assert result.n_samples == rows
    assert result.actor_rows == expected * config.n_epochs == sum(seen)
    assert result.actor_forwards == len(seen)
    assert result.forced_frac == pytest.approx(1.0 - choice / rows)
    # The identity phase 1 of section 18.1 checks, in the arm it checks it in. Under `all` the
    # actor is shown every trainable row of every epoch and the forced fraction is a report
    # about the batch rather than about what was skipped.
    epochs = config.n_epochs
    share = 1.0 if arm == "all" else 1.0 - result.forced_frac
    assert result.actor_rows == pytest.approx(result.n_samples * epochs * share)


def test_a_batch_with_no_choice_row_still_steps_both_optimizers(rect: Fixture) -> None:
    """Under ``all`` such a batch hands the actor a gradient of zeros, and Adam does not ignore
    one.

    The step Adam takes on a zero gradient is the remains of the ones before it: the first moment
    decays towards zero and the parameters move. If the skipping arm left the gradients at None
    instead, torch would skip those parameters entirely and the two arms' optimizer trajectories
    would part on the first batch that held no choice -- which on the shipped geometry is about
    one remainder batch in five.

    The whole rectangle is forced here, so every batch is that batch and the actor never runs.
    """
    model = build_model(rect.spec)
    plant_forced(rect, [(cycle, slot) for cycle in range(CYCLES) for slot in range(SLOTS)])
    collect(rect, model)
    config = msgspec.structs.replace(CONFIG, n_epochs=2, batch_size=SAMPLES // 2)
    every_model, skipping_model = build_model(rect.spec), build_model(rect.spec)
    seen = count_rows(skipping_model.actor)
    every = update_for(every_model, config)
    skipping = update_for(
        skipping_model, msgspec.structs.replace(config, forced_rows="critic_only")
    )

    plain = every.step(rect.buffer, SCHEDULE)
    skipped = skipping.step(rect.buffer, SCHEDULE)

    assert int(((rect.buffer.n_legal > 1) & rect.buffer.trainable()).sum()) == 0
    assert seen == [], "the actor was run on a batch with nothing that could move"
    assert skipped.n_optimizer_steps == plain.n_optimizer_steps == 4
    assert skipping.model_updates == every.model_updates == 4
    for parameter in skipping.actor_params:
        assert parameter.grad is not None, "a missing gradient is a parameter Adam steps over"
        assert bool((parameter.grad == 0).all())
    for mine, theirs in zip(every.actor_params, skipping.actor_params, strict=True):
        assert bool(torch.isfinite(theirs).all())
        assert torch.allclose(mine.detach(), theirs.detach(), atol=1e-9), (
            "the actor's parameters did not land where all puts them for these batches"
        )


#: The benchmark's rectangle and the forced-row rate planted in it. The rate is the one the
#: first real iterations reported, 0.88-0.93, so that the share of the actor's work being
#: skipped is the share a run would skip.
BENCH_CYCLES, BENCH_SLOTS, BENCH_FORCED = 32, 32, 0.9


def bench_rectangle(
    spec: Any, observations: list[dict[str, np.ndarray]], arch: Any
) -> Any:
    """A rectangle large enough to time, with nine rows in ten forced, played by a real policy.

    The policy that plays it is built from the same architecture the update will be timed on,
    so the stored log-probabilities are the ones a fresh forward reproduces and the timing is
    taken on a rectangle an update would accept.
    """
    from royalelearn.learn.nets import DefaultNetworkFactory

    built = Fixture(spec, observations, cycles=BENCH_CYCLES, slots=BENCH_SLOTS)
    built.fill()
    model = DefaultNetworkFactory(SEED).build(built.spec, arch, "cpu")
    rng = np.random.default_rng(SEED)
    cells = [(cycle, slot) for cycle in range(BENCH_CYCLES) for slot in range(BENCH_SLOTS)]
    chosen = rng.permutation(len(cells))[: int(BENCH_FORCED * len(cells))]
    plant_forced(built, [cells[index] for index in chosen])

    buffer = built.buffer
    plan = plan_for(BENCH_SLOTS)
    buffer.begin_iteration(plan, BENCH_CYCLES)
    engine = BatchedInference(buffer, model, master_seed=SEED)
    engine.begin_iteration(plan)
    slots = np.arange(BENCH_SLOTS, dtype=np.int64)
    for cycle in range(BENCH_CYCLES + 1):
        played = round_for(
            cycle,
            slots,
            rows=np.array([buffer.layout.row_index(cycle, int(s)) for s in slots]),
            reward=rng.normal(size=BENCH_SLOTS).astype(np.float32),
        )
        if cycle == BENCH_CYCLES:
            buffer.record_round(played, None, None)
            continue
        answer = engine.act(played)
        buffer.record_round(played, answer.actions, answer.log_probs)
    return built


@pytest.mark.slow
def test_skipping_the_forced_rows_is_the_faster_update(
    env_spec: Any, observations: list[dict[str, np.ndarray]]
) -> None:
    """The saving, measured rather than argued, on MockEngine and the CPU.

    What is timed is the epochs, which is ``seconds`` less the critic's pass over the rectangle
    and the advantage recursion: those two run before the first epoch and are identical under
    both values, so leaving them in would report the saving diluted by a constant that depends
    on the geometry.

    The two arms alternate and the best of the repeats is taken, because a slower run on a
    contended machine says something about the machine. The assertion is only the direction --
    the size of the saving is a number this prints, and it belongs in a commit message and in
    section 18.1 rather than in a threshold that would fail on somebody else's laptop.

    This is not the figure the A/B predicts. That one is for the shipped network on a GPU, where
    the actor is about 53% of the per-row compute; here the network is small enough to run on
    the CPU, so a larger share of each minibatch is the gather and the copy, which neither value
    skips.
    """
    from royalelearn.learn.nets import DefaultNetworkFactory
    from test_inference import ARCH

    arch = msgspec.structs.replace(
        ARCH, channels=32, blocks=2, vector_embed=8, value_hidden=64, card_embed=32
    )
    built = bench_rectangle(env_spec, observations, arch)
    config = msgspec.structs.replace(
        CONFIG,
        n_epochs=1,
        timesteps_per_iteration=BENCH_CYCLES * BENCH_SLOTS,
        batch_size=BENCH_CYCLES * BENCH_SLOTS,
        minibatch_size=256,
        critic_chunk=1024,
        debug_assert_iterations=0,
        check_ratio_invariant_every=0,
    )
    try:
        epochs: dict[str, list[float]] = {"all": [], "critic_only": []}
        whole: dict[str, list[float]] = {"all": [], "critic_only": []}
        forced = 0.0
        for repeat in range(4):
            for arm in epochs:
                model = DefaultNetworkFactory(SEED).build(built.spec, arch, "cpu")
                update = update_for(model, msgspec.structs.replace(config, forced_rows=arm))
                result = update.step(built.buffer, SCHEDULE)
                if repeat == 0:  # a warm-up: torch picks its kernels on the first forward
                    forced = result.forced_frac
                    continue
                whole[arm].append(result.seconds)
                epochs[arm].append(
                    result.seconds - result.critic_pass_seconds - result.gae_seconds
                )
    finally:
        built.close()

    best = {arm: min(values) for arm, values in epochs.items()}
    total = {arm: min(values) for arm, values in whole.items()}
    print(
        f"\nforced_frac {forced:.3f} over {BENCH_CYCLES * BENCH_SLOTS} rows; epochs "
        f"{best['all']:.3f}s -> {best['critic_only']:.3f}s "
        f"({best['critic_only'] / best['all']:.3f}x); whole update "
        f"{total['all']:.3f}s -> {total['critic_only']:.3f}s "
        f"({total['critic_only'] / total['all']:.3f}x)"
    )
    assert forced == pytest.approx(BENCH_FORCED, abs=0.01)
    assert best["critic_only"] < best["all"], (
        f"the epochs took {best['critic_only']:.3f}s skipping {forced:.0%} of the rows against "
        f"{best['all']:.3f}s training the actor on all of them"
    )


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


def test_a_cell_with_one_legal_action_that_did_not_take_it_stops_the_update(
    rect: Fixture,
) -> None:
    """The whole arithmetic of a one-action cell rests on two stored numbers, so they are checked.

    Where a mask leaves a single action, ``log_softmax`` over it is exactly zero in float32, so
    the stored log-probability of such a cell is the literal 0.0 and its importance ratio is the
    literal 1.0. Everything that treats these rows as carrying no policy gradient -- the
    conditioned diagnostics today, the actor's skip under ``ppo.forced_rows`` -- is true because
    of that and false without it. A stored log-probability of anything else means the mask the
    update reads is not the mask the policy acted under, which is the failure the ratio invariant
    catches at the first minibatch of an iteration and this one catches on every row of every
    iteration.

    Two cells are planted so that the message has to name the right one: the check runs on the
    whole rectangle and a message that named the first cell it found would look the same.
    """
    model = build_model(rect.spec)
    plant_forced(rect, [(1, 2), (3, 5)])
    buffer = collect(rect, model)
    assert float(buffer.log_prob[3, 5]) == 0.0, "a one-action cell's log-probability is zero"
    buffer.log_prob[3, 5] = -1.0

    with pytest.raises(ValueError, match=r"cycle 3 slot 5") as raised:
        update_for(model).step(buffer, SCHEDULE)

    assert "cycle 1 slot 2" not in str(raised.value), "the untouched cell is not an offender"


def test_a_cell_with_one_legal_action_that_took_another_stops_the_update(
    rect: Fixture,
) -> None:
    """The other half of the same invariant: the action stored beside the zero was that action.

    A log-probability of zero against an action the mask forbids is a rollout that sampled
    outside its own mask, and it would leave the ratio at one while the policy was scored on a
    transition it never made.
    """
    model = build_model(rect.spec)
    plant_forced(rect, [(2, 4)])
    buffer = collect(rect, model)
    assert int(buffer.action[2, 4]) == NOOP
    buffer.action[2, 4] = NOOP + 1

    with pytest.raises(ValueError, match=r"cycle 2 slot 4"):
        update_for(model).step(buffer, SCHEDULE)


def test_the_stored_legal_count_is_checked_against_the_mask_a_minibatch_carries(
    rect: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The column is written once per iteration and read per row, so the two are held together.

    It is filled from the critic's pass over the rectangle and read back through the minibatch
    gather, which sorts its cells; a mapping that drifted between the two would hand the update
    one row's legal count beside another row's observation, and nothing downstream would look
    wrong. The plant is on the column itself, one cell of it, so the assertion that fires can
    only be this one.
    """
    from royalelearn.learn.buffer import RectBuffer

    model = build_model(rect.spec)
    buffer = collect(rect, model)
    stored = RectBuffer.set_n_legal

    def one_cell_too_many(self: RectBuffer, counts: Any) -> None:
        stored(self, counts)
        self.n_legal[0, 0] += 1

    monkeypatch.setattr(RectBuffer, "set_n_legal", one_cell_too_many)

    with pytest.raises(AssertionError, match="legal actions"):
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

    values, n_legal = chunked_critic_pass(buffer, model, gather, chunk=SLOTS * 2)

    assert values.shape == (CYCLES + 1, SLOTS)
    assert bool(torch.isfinite(values).all())
    assert float(values.abs().sum()) > 0.0
    # The legal count covers the same cells: it comes off the mask of the same unpacked row.
    assert n_legal.shape == (CYCLES + 1, SLOTS)
    assert int(n_legal.min()) >= 1


def test_a_cell_no_worker_wrote_has_no_value(rect: Fixture) -> None:
    """A dead worker's rows fall out of the recursion without a special case: no reward, no
    value and the episode flagged ended, so the loop runs over them inertly."""
    from royalelearn.learn.inference import RectGather

    model = build_model(rect.spec)
    buffer = collect(rect, model)
    buffer.valid[1, 3] = False
    gather = RectGather(buffer, rows=SLOTS * 2)

    values, _n_legal = chunked_critic_pass(buffer, model, gather, chunk=SLOTS * 2)

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
        diagnostics.minibatch(
            epoch=0,
            n=count,
            ratio=torch.ones(count),
            advantages=torch.zeros(count),
            surr=torch.zeros(count),
            dual=torch.zeros(count),
            entropy=torch.zeros(count),
            noop_entropy=torch.tensor(noop),
            logit_std=torch.zeros(len(noop)),
            n_legal=torch.tensor(legal),
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
    diagnostics.minibatch(
        epoch=0,
        n=count,
        ratio=ratio,
        advantages=torch.zeros(count),
        surr=torch.zeros(count),
        dual=torch.zeros(count),
        entropy=torch.zeros(count),
        noop_entropy=torch.zeros(count),
        logit_std=torch.zeros(count),
        n_legal=torch.tensor([40 if c else 1 for c in chose]),
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
