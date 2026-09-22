"""The quantities that move with the run.

A schedule is a pure function of the step count, so the checks are exact: each anneal is at its
start value at zero steps and at its end value at the step count the config names, to the last
digit rather than to a tolerance. The discount's anneal is geometric in the distance to one,
which is the property that makes the credit horizon move smoothly rather than crawling and then
leaping, so that is checked as a constant ratio and not as a shape.
"""

from __future__ import annotations

import itertools

import msgspec
import pytest

from royalelearn.api.schedule import ScheduleState
from royalelearn.config import (
    ConstantSpec,
    GeometricSpec,
    LinearSpec,
    LrBackoffConfig,
    PiecewiseConstantSpec,
    RunConfig,
)
from royalelearn.learn.schedules import (
    Constant,
    Geometric,
    Linear,
    LrBackoff,
    PiecewiseConstant,
    ScheduleSet,
    build_schedule,
)


def test_a_constant_does_not_move() -> None:
    schedule = build_schedule(ConstantSpec(0.25))
    assert isinstance(schedule, Constant)
    assert schedule.value(0) == 0.25
    assert schedule.value(10**12) == 0.25


def test_a_linear_anneal_hits_both_endpoints_exactly() -> None:
    spec = LinearSpec(0.01, 0.003, 30_000_000)
    schedule = build_schedule(spec)
    assert isinstance(schedule, Linear)
    assert schedule.value(0) == spec.start
    assert schedule.value(spec.over_env_steps) == pytest.approx(spec.end, rel=1e-15)
    assert schedule.value(2 * spec.over_env_steps) == pytest.approx(spec.end, rel=1e-15)
    half = schedule.value(spec.over_env_steps // 2)
    assert half == pytest.approx((spec.start + spec.end) / 2, rel=1e-9)


def test_a_geometric_anneal_hits_both_endpoints_exactly() -> None:
    spec = GeometricSpec(0.997, 0.999, 20_000_000)
    schedule = build_schedule(spec)
    assert isinstance(schedule, Geometric)
    assert schedule.value(0) == pytest.approx(spec.start, rel=1e-15)
    assert schedule.value(spec.over_env_steps) == pytest.approx(spec.end, rel=1e-12)
    assert schedule.value(10 * spec.over_env_steps) == pytest.approx(spec.end, rel=1e-12)


def test_the_discount_s_distance_to_one_decays_by_a_constant_factor() -> None:
    spec = GeometricSpec(0.997, 0.999, 20_000_000)
    schedule = build_schedule(spec)
    steps = [spec.over_env_steps * k // 4 for k in range(5)]
    gaps = [1.0 - schedule.value(step) for step in steps]
    ratios = [later / earlier for earlier, later in itertools.pairwise(gaps)]
    for ratio in ratios[1:]:
        assert ratio == pytest.approx(ratios[0], rel=1e-9)
    assert gaps[-1] < gaps[0]


def test_a_geometric_schedule_refuses_an_endpoint_at_or_above_one() -> None:
    with pytest.raises(ValueError, match="distance to one"):
        Geometric(0.99, 1.0, 10)


def test_a_piecewise_schedule_holds_its_value_between_breakpoints() -> None:
    schedule = build_schedule(PiecewiseConstantSpec([(1000, 0.5), (0, 1.0), (2000, 0.25)]))
    assert isinstance(schedule, PiecewiseConstant)
    assert schedule.value(0) == 1.0
    assert schedule.value(999) == 1.0
    assert schedule.value(1000) == 0.5
    assert schedule.value(1999) == 0.5
    assert schedule.value(10**9) == 0.25


def test_a_piecewise_schedule_holds_its_first_value_before_its_first_breakpoint() -> None:
    schedule = PiecewiseConstant([(5000, 0.75)])
    assert schedule.value(0) == 0.75
    assert schedule.value(5000) == 0.75


def test_a_schedule_with_no_span_is_already_at_its_end() -> None:
    assert Linear(1.0, 0.0, 0).value(0) == 0.0


def test_a_schedule_is_a_pure_function_of_the_step_count() -> None:
    schedule = build_schedule(GeometricSpec(0.99, 0.999, 1_000))
    first = [schedule.value(step) for step in range(0, 1_000, 97)]
    again = [schedule.value(step) for step in range(0, 1_000, 97)]
    assert first == again


# --------------------------------------------------------------------------
# The set, and the state one iteration runs at
# --------------------------------------------------------------------------


def test_the_shipped_config_s_schedules_hit_their_configured_endpoints(
    run_config: RunConfig,
) -> None:
    schedules = ScheduleSet.from_config(run_config)
    gamma = run_config.advantage.gamma
    entropy = run_config.ppo.ent_coef
    noop = run_config.ppo.ent_coef_noop
    assert schedules.gamma.value(0) == pytest.approx(gamma.start, rel=1e-15)
    assert schedules.gamma.value(gamma.over_env_steps) == pytest.approx(gamma.end, rel=1e-12)
    assert schedules.ent_coef.value(entropy.over_env_steps) == pytest.approx(entropy.end)
    assert schedules.ent_coef_noop.value(noop.over_env_steps) == pytest.approx(noop.end)
    assert schedules.ent_coef_noop.value(noop.over_env_steps) == 0.0


def test_the_whole_schedule_state_round_trips(run_config: RunConfig) -> None:
    schedules = ScheduleSet.from_config(run_config)
    state = schedules.state(
        iteration=7, cumulative_env_steps=5_000_000, cumulative_timesteps=9_000_000
    )
    restored = msgspec.json.decode(msgspec.json.encode(state), type=ScheduleState)
    assert restored == state
    assert restored.gamma == schedules.gamma.value(5_000_000)
    assert restored.lr_actor == run_config.ppo.lr_actor
    assert restored.gae_lambda == run_config.advantage.gae_lambda


def test_the_state_carries_the_credit_horizon_the_iteration_ran_at(
    run_config: RunConfig, env_spec: object
) -> None:
    schedules = ScheduleSet.from_config(run_config)
    state = schedules.state(iteration=0, cumulative_env_steps=0, cumulative_timesteps=0)
    decision_ms = env_spec.decision_ms  # type: ignore[attr-defined]
    expected = (1.0 / (1.0 - state.gamma * state.gae_lambda)) * decision_ms / 1000.0
    assert state.credit_horizon_seconds(decision_ms) == pytest.approx(expected)


# --------------------------------------------------------------------------
# The learning-rate backoff
# --------------------------------------------------------------------------


def backoff(**overrides: float) -> LrBackoff:
    config = msgspec.structs.replace(LrBackoffConfig(), **overrides)
    return LrBackoff(config, lr_actor=2e-4, lr_critic=2e-4)


def test_the_backoff_fires_after_exactly_its_patience_of_consecutive_breaches() -> None:
    guard = backoff()
    patience = guard.config.patience
    breach = guard.config.kl_threshold * 2
    for _ in range(patience - 1):
        assert guard.observe(breach) is False
    assert guard.lr_actor == 2e-4
    assert guard.observe(breach) is True
    assert guard.lr_actor == pytest.approx(2e-4 * guard.config.factor)
    assert guard.lr_critic == pytest.approx(2e-4 * guard.config.factor)
    assert guard.events == 1
    assert guard.consecutive_breaches == 0


def test_one_quiet_iteration_resets_the_counter() -> None:
    guard = backoff()
    breach = guard.config.kl_threshold * 2
    for _ in range(guard.config.patience - 1):
        guard.observe(breach)
    assert guard.observe(guard.config.kl_threshold / 2) is False
    assert guard.consecutive_breaches == 0
    for _ in range(guard.config.patience - 1):
        assert guard.observe(breach) is False
    assert guard.observe(breach) is True


def test_the_rates_floor_at_the_configured_minimum() -> None:
    guard = backoff()
    breach = guard.config.kl_threshold * 2
    for _ in range(guard.config.patience * 40):
        guard.observe(breach)
    assert guard.lr_actor == pytest.approx(guard.config.lr_min)
    assert guard.lr_critic == pytest.approx(guard.config.lr_min)


def test_the_backoff_state_round_trips_through_a_checkpoint_folder(tmp_path: object) -> None:
    from pathlib import Path

    folder = Path(str(tmp_path)) / "schedules"
    guard = backoff()
    breach = guard.config.kl_threshold * 2
    for _ in range(guard.config.patience):
        guard.observe(breach)
    guard.save_checkpoint(folder)
    restored = backoff()
    restored.load_checkpoint(folder, strict=True)
    assert restored.state() == guard.state()
    assert restored.lr_actor == guard.lr_actor
    assert restored.events == 1


def test_the_state_the_backoff_reached_is_what_the_next_iteration_runs_at(
    run_config: RunConfig,
) -> None:
    schedules = ScheduleSet.from_config(run_config)
    breach = schedules.backoff.config.kl_threshold * 2
    for _ in range(schedules.backoff.config.patience):
        schedules.backoff.observe(breach)
    state = schedules.state(iteration=1, cumulative_env_steps=0, cumulative_timesteps=0)
    assert state.lr_actor == pytest.approx(run_config.ppo.lr_actor * 0.5)
    assert state.lr_backoff_events == 1
