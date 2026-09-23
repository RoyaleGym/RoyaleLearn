"""Whether a run's precision can carry its own importance ratio, asked before the run starts.

PPO's update recomputes the log-probability of each stored action and asserts the ratio is one,
because nothing has changed since the rollout. On 2026-09-23 a run died at iteration 6 on that
assertion with a deviation of 0.0261 against a tolerance of 0.02, and the cause was neither a bug
nor a coincidence:

    log p_i = z_i - logsumexp(z)        so    d(log p_i) / d(z_noop) = -p_noop

An error in the LARGEST logit is multiplied into every other action's log-probability by the
probability that logit holds. At `noop_bias` 0 the no-op holds about 1/450 of the mass and a logit
error is invisible. At `noop_bias` 8.0 it holds 0.87, so the SAME arithmetic error is about 390
times larger in the ratio. bf16's spacing at magnitude 8 is 0.031, and 0.87 x 0.031 is 0.027 --
which is what killed the run.

The prediction is checkable against both measurements taken that night:

    bias 8, bfloat16    0.87 x 0.031      = 0.027     observed 0.0261 (died)
    bias 8, float32     0.87 x 9.5e-7     = 8.3e-7    observed 9.5e-7 (8 iterations, quiet)

So it is knowable from the config and the action space alone, before anything runs. That is what
these tests hold: the harness predicts it, and refuses a run whose own arithmetic cannot carry its
own guard.
"""

from __future__ import annotations

import pytest

from royalelearn.rollout.preflight import ratio_precision

#: What the two runs of 2026-09-23 actually measured, on 451 actions at noop_bias 8.0.
DIED_AT_BF16 = 0.0261
QUIET_AT_FLOAT32 = 9.5e-7


def test_it_predicts_the_run_that_died() -> None:
    """bfloat16 at noop_bias 8.0, which is where this came from.

    An UPPER bound: 0.054 against the 0.0261 a minibatch happened to sample. Erring high is the
    right direction for a gate, and what matters is that it lands far above the 0.02 tolerance
    rather than that it lands on the observation.
    """
    predicted = ratio_precision(noop_bias=8.0, n_actions=451, dtype_name="bfloat16")
    assert predicted > 0.02, "the prediction no longer exceeds the tolerance that stopped the run"
    assert DIED_AT_BF16 <= predicted <= 3 * DIED_AT_BF16, predicted


def test_it_predicts_the_float32_measurement() -> None:
    """The same run in float32 measured 9.5e-7 over eight iterations on the real engine."""
    predicted = ratio_precision(noop_bias=8.0, n_actions=451, dtype_name="float32")
    assert predicted == pytest.approx(QUIET_AT_FLOAT32, rel=0.3)
    assert predicted < 1e-4, "float32's own tolerance would have been met with room to spare"


def test_an_unbiased_policy_is_nowhere_near_its_guard() -> None:
    """383 iterations at noop_bias 0 never fired it, and this says why rather than asserting it."""
    predicted = ratio_precision(noop_bias=0.0, n_actions=451, dtype_name="bfloat16")
    assert predicted < 1e-4


def test_the_amplification_is_the_probability_and_not_the_count() -> None:
    """It is p_max, so it SATURATES: past the point where the no-op holds everything, raising
    the bias further stops making the ratio worse and only the logit spacing grows."""
    at_8 = ratio_precision(noop_bias=8.0, n_actions=451, dtype_name="bfloat16")
    at_4 = ratio_precision(noop_bias=4.0, n_actions=451, dtype_name="bfloat16")
    at_0 = ratio_precision(noop_bias=0.0, n_actions=451, dtype_name="bfloat16")
    assert at_0 < at_4 < at_8


def test_a_wider_action_space_dilutes_it() -> None:
    """The same bias over more actions holds less of the mass, so it amplifies less."""
    narrow = ratio_precision(noop_bias=6.0, n_actions=100, dtype_name="bfloat16")
    wide = ratio_precision(noop_bias=6.0, n_actions=4000, dtype_name="bfloat16")
    assert wide < narrow
