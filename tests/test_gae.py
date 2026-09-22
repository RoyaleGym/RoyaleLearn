"""Advantage estimation: the vectorised recursion against the readable one, case by case.

The pair of implementations is only worth having if something holds them to each other, so this
file is the point of the pair. Every boundary the recursion has is named: an episode one cycle
long, a truncation, the rows of a worker that died, an iteration in which everything ended and one
in which nothing did.

The two bugs being guarded against are the ones the older reference learners ship. One bootstraps
a truncation off an unrelated episode's first state; the other self-bootstraps from ``V(s_T)``.
Both are silent, both cost the value of every position a step limit ended, and both are a line.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalelearn.seeding import derive_generator

torch = pytest.importorskip("torch")

from royalelearn.learn.gae import GAE, gae_recursion, reference_gae  # noqa: E402
from royalelearn.learn.returns import WelfordReturnScaler, discounted_returns  # noqa: E402

SEED = 20260921
GAMMA = 0.997
LAMBDA = 0.99
TOLERANCE = 1e-6


def as_tensors(**columns: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, array in columns.items():
        dtype = torch.bool if array.dtype == bool else torch.float64
        out[name] = torch.as_tensor(array, dtype=dtype)
    return out


def random_iteration(
    cycles: int, slots: int, *, path: str, terminate: float = 0.05, truncate: float = 0.02
) -> dict[str, np.ndarray]:
    """One iteration's worth of synthetic columns, from a named stream."""
    rng = derive_generator(SEED, path)
    draw = rng.random((cycles, slots))
    terminated = draw < terminate
    truncated = (draw >= terminate) & (draw < terminate + truncate)
    return {
        "rewards": rng.normal(size=(cycles, slots)),
        "values": rng.normal(size=(cycles + 1, slots)),
        "final_values": rng.normal(size=(cycles, slots)),
        "terminated": terminated,
        "truncated": truncated,
    }


def both(columns: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The vectorised answer and the readable one, for the same inputs."""
    fast_adv, fast_ret = gae_recursion(
        **as_tensors(**columns),  # type: ignore[arg-type]
        gamma=GAMMA,
        lam=LAMBDA,
    )
    slow_adv, slow_ret = reference_gae(
        columns["rewards"],
        columns["values"],
        columns["final_values"],
        columns["terminated"],
        columns["truncated"],
        GAMMA,
        LAMBDA,
    )
    return fast_adv.numpy(), fast_ret.numpy(), slow_adv, slow_ret


def assert_agree(columns: dict[str, np.ndarray]) -> None:
    fast_adv, fast_ret, slow_adv, slow_ret = both(columns)
    assert np.max(np.abs(fast_adv - slow_adv)) < TOLERANCE
    assert np.max(np.abs(fast_ret - slow_ret)) < TOLERANCE


# --------------------------------------------------------------------------
# The two implementations agree
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(4))
def test_the_vectorised_recursion_equals_the_readable_one(trial: int) -> None:
    assert_agree(random_iteration(37, 11, path=f"test/gae/random/{trial}"))


def test_they_agree_on_an_iteration_in_which_nothing_ended(  ) -> None:
    columns = random_iteration(20, 5, path="test/gae/none", terminate=0.0, truncate=0.0)
    assert not columns["terminated"].any()
    assert not columns["truncated"].any()
    assert_agree(columns)


def test_they_agree_on_an_iteration_in_which_everything_terminated() -> None:
    columns = random_iteration(20, 5, path="test/gae/all", terminate=1.0, truncate=0.0)
    assert columns["terminated"].all()
    assert_agree(columns)


def test_they_agree_on_an_iteration_in_which_everything_truncated() -> None:
    columns = random_iteration(20, 5, path="test/gae/all-trunc", terminate=0.0, truncate=1.0)
    assert columns["truncated"].all()
    assert_agree(columns)


def test_they_agree_on_an_episode_one_cycle_long() -> None:
    """A battle that ends on the cycle it started: the recursion has no history to carry."""
    columns = random_iteration(1, 3, path="test/gae/one-cycle", terminate=0.5, truncate=0.5)
    assert_agree(columns)


def test_they_agree_when_every_cycle_ends_an_episode() -> None:
    columns = random_iteration(8, 4, path="test/gae/every", terminate=0.5, truncate=0.5)
    assert (columns["terminated"] | columns["truncated"]).all()
    assert_agree(columns)


# --------------------------------------------------------------------------
# What a cell bootstraps from
# --------------------------------------------------------------------------


def one_slot(
    rewards: list[float],
    values: list[float],
    *,
    terminated: list[bool] | None = None,
    truncated: list[bool] | None = None,
    final_values: list[float] | None = None,
) -> dict[str, np.ndarray]:
    cycles = len(rewards)
    zeros = [False] * cycles
    return {
        "rewards": np.array(rewards, dtype=np.float64).reshape(cycles, 1),
        "values": np.array(values, dtype=np.float64).reshape(cycles + 1, 1),
        "final_values": np.array(
            final_values if final_values is not None else [0.0] * cycles, dtype=np.float64
        ).reshape(cycles, 1),
        "terminated": np.array(terminated or zeros, dtype=bool).reshape(cycles, 1),
        "truncated": np.array(truncated or zeros, dtype=bool).reshape(cycles, 1),
    }


def test_a_terminated_cell_bootstraps_from_zero() -> None:
    """A finished battle has no state after it, so the next value is not a state's value."""
    columns = one_slot([1.0], [0.25, 7.0], terminated=[True])
    advantages, _ = gae_recursion(
        **as_tensors(**columns),  # type: ignore[arg-type]
        gamma=GAMMA,
        lam=LAMBDA,
    )
    assert advantages[0, 0].item() == pytest.approx(1.0 - 0.25)
    assert_agree(columns)


def test_a_truncated_cell_bootstraps_from_its_final_value() -> None:
    """The regression the references fail: a truncation is an episode that was cut, not one
    that was decided, and the value of the position it was cut at is the whole of its tail."""
    columns = one_slot([1.0], [0.25, 7.0], truncated=[True], final_values=[3.0])
    advantages, _ = gae_recursion(
        **as_tensors(**columns),  # type: ignore[arg-type]
        gamma=GAMMA,
        lam=LAMBDA,
    )
    assert advantages[0, 0].item() == pytest.approx(1.0 + GAMMA * 3.0 - 0.25)
    assert_agree(columns)


def test_a_cell_mid_episode_bootstraps_from_the_next_cycle() -> None:
    """Including the last cycle of an iteration for a row still in its battle."""
    columns = one_slot([1.0], [0.25, 7.0])
    advantages, _ = gae_recursion(
        **as_tensors(**columns),  # type: ignore[arg-type]
        gamma=GAMMA,
        lam=LAMBDA,
    )
    assert advantages[0, 0].item() == pytest.approx(1.0 + GAMMA * 7.0 - 0.25)


def test_the_recursion_breaks_at_a_truncation_as_well_as_a_termination() -> None:
    """Breaking only on termination leaks the next episode's advantage backwards."""
    for flag in ("terminated", "truncated"):
        columns = one_slot(
            [0.0, 0.0, 100.0],
            [0.0, 0.0, 0.0, 0.0],
            **{flag: [False, True, False]},  # type: ignore[arg-type]
            final_values=[0.0, 0.0, 0.0],
        )
        advantages, _ = gae_recursion(
            **as_tensors(**columns),  # type: ignore[arg-type]
            gamma=GAMMA,
            lam=LAMBDA,
        )
        # The reward at cycle 2 belongs to the episode that started at cycle 2. Neither the cell
        # that ended the previous episode nor anything before it may see it.
        assert advantages[1, 0].item() == pytest.approx(0.0)
        assert advantages[0, 0].item() == pytest.approx(0.0)
        assert advantages[2, 0].item() == pytest.approx(100.0)


def test_a_dead_worker_s_rows_fall_out_without_a_special_case() -> None:
    """The farm marks a dead worker's cells invalid, truncates the last live one against the
    value it had, and zeroes what follows. The recursion then runs over them inertly."""
    cycles, died = 10, 6
    last = died - 2
    columns = random_iteration(cycles, 1, path="test/gae/dead")
    columns["terminated"][:] = False
    columns["truncated"][:] = False
    columns["truncated"][last, 0] = True
    columns["final_values"][last, 0] = columns["values"][died - 1, 0]
    columns["rewards"][last + 1 :, 0] = 0.0
    columns["values"][last + 1 :, 0] = 0.0
    columns["terminated"][last + 1 :, 0] = True
    advantages, _, _, _ = both(columns)
    assert np.allclose(advantages[last + 1 :, 0], 0.0)
    assert_agree(columns)


# --------------------------------------------------------------------------
# The estimator around the recursion
# --------------------------------------------------------------------------


def test_compute_agrees_with_the_reference_once_the_scaling_is_known() -> None:
    columns = random_iteration(24, 6, path="test/gae/compute")
    trainable = np.ones(columns["rewards"].shape, dtype=bool)
    estimator = GAE(standardize_rewards=True, reward_clip=10.0)
    tensors = as_tensors(**columns)
    advantages, returns, stats = estimator.compute(
        **tensors,  # type: ignore[arg-type]
        trainable=torch.as_tensor(trainable),
        gamma=GAMMA,
        lam=LAMBDA,
    )
    scaled = np.clip(columns["rewards"] / stats.reward_scale, -10.0, 10.0)
    slow_adv, slow_ret = reference_gae(
        scaled,
        columns["values"],
        columns["final_values"],
        columns["terminated"],
        columns["truncated"],
        GAMMA,
        LAMBDA,
    )
    assert np.max(np.abs(advantages.numpy() - slow_adv)) < TOLERANCE
    assert np.max(np.abs(returns.numpy() - slow_ret)) < TOLERANCE
    assert stats.reward_scale > 0.0


def test_the_estimator_scales_by_the_running_deviation_and_never_the_mean() -> None:
    columns = random_iteration(16, 4, path="test/gae/scale")
    shifted = {**columns, "rewards": columns["rewards"] + 5.0}
    estimator = GAE(standardize_rewards=True, reward_clip=1e9)
    tensors = as_tensors(**shifted)
    _, _, stats = estimator.compute(
        **tensors,  # type: ignore[arg-type]
        trainable=torch.ones(shifted["rewards"].shape, dtype=torch.bool),
        gamma=GAMMA,
        lam=LAMBDA,
    )
    ended = torch.as_tensor(columns["terminated"] | columns["truncated"])
    raw = discounted_returns(tensors["rewards"], ended, GAMMA)  # type: ignore[arg-type]
    assert stats.raw_return_mean == pytest.approx(float(raw.mean()), rel=1e-9)
    assert stats.reward_scale == pytest.approx(float(raw.std(unbiased=True)), rel=1e-9)
    assert stats.reward_scale != pytest.approx(0.0)


def test_the_scale_is_one_until_there_is_a_variance_to_divide_by() -> None:
    scaler = WelfordReturnScaler(standardize=True)
    assert scaler.divisor == 1.0
    scaler.update(torch.tensor([3.0]))
    assert scaler.divisor == 1.0


def test_the_estimator_can_be_asked_not_to_standardise() -> None:
    columns = random_iteration(12, 3, path="test/gae/raw")
    estimator = GAE(standardize_rewards=False, reward_clip=1e9)
    advantages, _, stats = estimator.compute(
        **as_tensors(**columns),  # type: ignore[arg-type]
        trainable=torch.ones(columns["rewards"].shape, dtype=torch.bool),
        gamma=GAMMA,
        lam=LAMBDA,
    )
    assert stats.reward_scale == 1.0
    slow_adv, _ = reference_gae(
        columns["rewards"],
        columns["values"],
        columns["final_values"],
        columns["terminated"],
        columns["truncated"],
        GAMMA,
        LAMBDA,
    )
    assert np.max(np.abs(advantages.numpy() - slow_adv)) < TOLERANCE


def test_the_credit_horizon_is_the_one_over_one_minus_gamma_lambda_of_the_schedule() -> None:
    """The single highest-leverage number in the configuration, which is why it is logged."""
    from royalelearn.api.schedule import ScheduleState

    decision_ms = 500
    state = ScheduleState(
        iteration=0,
        cumulative_env_steps=0,
        cumulative_timesteps=0,
        gamma=GAMMA,
        gae_lambda=LAMBDA,
        ent_coef=0.0,
        ent_coef_noop=0.0,
        lr_actor=0.0,
        lr_critic=0.0,
    )
    expected = (1.0 / (1.0 - GAMMA * LAMBDA)) * decision_ms / 1000.0
    assert state.credit_horizon_seconds(decision_ms) == pytest.approx(expected)
