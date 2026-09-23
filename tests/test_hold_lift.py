"""What a policy's hold rate is a hold rate OF, and why entropy could not answer it.

On 2026-09-22 the train session read ``ppo/entropy_normalised`` in [0.980, 1.000] across 134
iterations of hog26-2 and concluded the actor had not moved at all. The conclusion was wrong and
the reading was right: at iteration 124 that policy was putting **15.3 times uniform mass** on
the no-op, and that excursion costs 2% of normalised entropy. Over 250 legal actions, moving one
of them from 0.4% to 6.1% leaves the other 249 sharing almost everything, and entropy is a sum
over all 250.

So the harness had no key for the quantity the run turns on. It measured the entropy of the whole
distribution, which is nearly blind to the only action with a distinct meaning, and it threw away
the rollout's own p(no-op) entirely: ``_StatAccumulator`` had summed it since the file was written
and no reader ever read it.

These tests fix the instrument against the case that defeated the old one. The first reproduces
both readings from the same distribution, so the pair (0.98, 15.3) is a computed consequence here
rather than two numbers somebody wrote down.
"""

from __future__ import annotations

import math

import pytest
import torch

from royalelearn.learn.distribution import MaskedCategorical
from royalelearn.learn.inference import _StatAccumulator
from royalelearn.metrics.records import rollout_policy_fields

#: The legal-set size the train session measured on hog26-2: about 250 all run, never collapsing.
N_LEGAL = 250

#: What the no-op's mass was a multiple of uniform at iteration 124.
OBSERVED_LIFT = 15.3

#: And what ``ppo/entropy_normalised`` read on that same iteration.
OBSERVED_ENTROPY = 0.980


def distribution(lift: float, *, n_legal: int = N_LEGAL, rows: int = 1) -> MaskedCategorical:
    """A policy that holds at ``lift`` times uniform and is uniform over everything else.

    Built from probabilities rather than from trained weights, because the claim under test is
    about what the two instruments READ, and a distribution whose hold mass is known exactly is
    the only thing that can settle it.
    """
    p_noop = lift / n_legal
    assert 0.0 < p_noop < 1.0, f"a lift of {lift} over {n_legal} actions is not a probability"
    rest = (1.0 - p_noop) / (n_legal - 1)
    probs = torch.full((rows, n_legal), rest, dtype=torch.float32)
    probs[:, 0] = p_noop
    return MaskedCategorical(probs.log(), torch.ones((rows, n_legal), dtype=torch.bool))


def normalised_entropy(dist: MaskedCategorical) -> float:
    """``ppo/entropy_normalised`` for one row, by its own definition in ``ppo.py``."""
    legal = dist.n_legal().to(torch.float32).log()
    return float((dist.entropy() / legal).mean())


def fields_for(dist: MaskedCategorical) -> dict[str, float]:
    stats = _StatAccumulator()
    stats.policy(dist, rows=int(dist.n_legal().shape[0]))
    return dict(rollout_policy_fields(stats.drain()))  # type: ignore[arg-type]


def test_the_excursion_entropy_could_not_see(tmp_path) -> None:
    """One distribution, two readings: 0.98 of maximum entropy, and 15.3x on the hold.

    This is the test the old instrument fails. If ``policy/rollout_hold_lift`` did not exist,
    the only number in the row describing this policy would be the one that says it is within
    2% of uniform -- which is true, and is not what anybody wanted to know.
    """
    dist = distribution(OBSERVED_LIFT)

    assert normalised_entropy(dist) == pytest.approx(OBSERVED_ENTROPY, abs=0.002), (
        "the train session's entropy reading is not reproduced by the lift it went with, so "
        "one of the two numbers in this file's premise is wrong"
    )

    fields = fields_for(dist)
    assert fields["policy/rollout_hold_lift"] == pytest.approx(OBSERVED_LIFT, rel=1e-4)
    assert fields["policy/rollout_hold_rate"] == pytest.approx(OBSERVED_LIFT / N_LEGAL, rel=1e-4)
    assert fields["policy/rollout_legal_actions"] == pytest.approx(float(N_LEGAL))


def test_a_uniform_policy_reads_as_exactly_one(tmp_path) -> None:
    """The baseline the lift is against, so that 1.0 means uniform and nothing else does."""
    fields = fields_for(distribution(1.0))
    assert fields["policy/rollout_hold_lift"] == pytest.approx(1.0, rel=1e-5)
    assert normalised_entropy(distribution(1.0)) == pytest.approx(1.0, abs=1e-5)


def test_forced_rows_do_not_dilute_the_lift(tmp_path) -> None:
    """Nine decisions in ten can afford nothing, and on those the policy holds by arithmetic.

    A forced row has one legal action, holds with probability 1, and has a lift of exactly 1.
    Averaging over every row would drag a 15x policy down towards 1 in proportion to the elixir
    curve, which is the defect ``ppo/noop_entropy`` and ``entropy_normalised`` were both
    conditioned on choice rows to avoid. The unconditioned mean of this batch is 2.43.
    """
    stats = _StatAccumulator()
    stats.policy(distribution(OBSERVED_LIFT, rows=10), rows=10)
    forced = MaskedCategorical(
        torch.zeros((90, N_LEGAL), dtype=torch.float32),
        torch.tensor([[True] + [False] * (N_LEGAL - 1)] * 90),
    )
    stats.policy(forced, rows=90)
    fields = dict(rollout_policy_fields(stats.drain()))  # type: ignore[arg-type]

    assert fields["policy/rollout_hold_lift"] == pytest.approx(OBSERVED_LIFT, rel=1e-4)
    assert fields["policy/rollout_choice_frac"] == pytest.approx(0.1)
    diluted = (10 * OBSERVED_LIFT + 90 * 1.0) / 100
    assert diluted == pytest.approx(2.43, abs=0.01)
    assert fields["policy/rollout_hold_lift"] != pytest.approx(diluted, rel=0.1)


def test_no_choice_rows_report_no_lift_rather_than_one(tmp_path) -> None:
    """A 1.0 here reads as "the policy is exactly uniform", which is a measurement.

    An iteration in which the bar could never afford anything measured nothing about the policy,
    and the row says so by not carrying the key.
    """
    stats = _StatAccumulator()
    stats.policy(
        MaskedCategorical(
            torch.zeros((8, N_LEGAL), dtype=torch.float32),
            torch.tensor([[True] + [False] * (N_LEGAL - 1)] * 8),
        ),
        rows=8,
    )
    fields = dict(rollout_policy_fields(stats.drain()))  # type: ignore[arg-type]

    assert "policy/rollout_hold_lift" not in fields
    assert "policy/rollout_hold_rate" not in fields
    assert fields["policy/rollout_choice_frac"] == pytest.approx(0.0)


def test_an_iteration_with_no_learner_rows_reports_nothing(tmp_path) -> None:
    """Not even the choice fraction: no rows is no denominator."""
    assert dict(rollout_policy_fields(_StatAccumulator().drain())) == {}  # type: ignore[arg-type]


def test_the_lift_is_the_hold_rate_over_its_own_uniform_baseline(tmp_path) -> None:
    """Rows of different widths, so the per-row baseline is the thing doing the work.

    A 4-legal row that is exactly uniform beside a 250-legal row at 20x uniform. Each row is
    compared with the baseline of its OWN width and the two ratios are averaged, which is 10.5.
    Dividing the pooled hold rate by the pooled baseline instead answers 1.3, because the narrow
    row contributes a quarter of the numerator while the wide one contributes a quarter of a
    percent -- an aggregate dominated by how many actions were legal rather than by the policy.
    """
    stats = _StatAccumulator()
    stats.policy(distribution(1.0, n_legal=4), rows=1)
    stats.policy(distribution(20.0, n_legal=250), rows=1)
    fields = dict(rollout_policy_fields(stats.drain()))  # type: ignore[arg-type]

    assert fields["policy/rollout_hold_lift"] == pytest.approx(10.5, rel=1e-4)
    pooled = ((1 / 4 + 20 / 250) / 2) / ((1 / 4 + 1 / 250) / 2)
    assert pooled == pytest.approx(1.30, abs=0.01), "the two aggregates must disagree here"
    assert fields["policy/rollout_legal_actions"] == pytest.approx(127.0)
    assert math.isclose(fields["policy/rollout_hold_rate"], (0.25 + 0.08) / 2, rel_tol=1e-4)
