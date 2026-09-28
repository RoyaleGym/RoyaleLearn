"""``bench`` prints a table a reader will quote, so its arithmetic is graded here.

One of the five test files ``docs/harness-spec.md`` section 15 lists and the tree lacked.
"""

from __future__ import annotations

import pytest

from royalelearn.cli import bench_report


def _row(iteration: int, *, cumulative: float, update: float) -> dict[str, float]:
    """One metrics row, carrying only the keys the report reads."""
    return {
        "run/iteration": iteration,
        "run/cumulative_timesteps": cumulative,
        "time/env": 4.0,
        "time/update": update,
        "throughput/boundary_mb_per_second": 120.0,
        "throughput/inference_ms_per_round": 1.5,
        "throughput/rollout_capacity_ratio": 2.4,
        "health/vram_peak_mb": 2009.0,
        "ppo/ratio_max_abs_dev": 1e-7,
    }


def _report(rows: list[dict[str, float]]) -> dict[str, object]:
    return bench_report(
        rows, env_steps_per_iteration=2000, codec_us=3.25, ratio_atol=0.02
    )


def test_the_update_rate_is_one_iterations_rate_and_not_the_whole_run_divided_by_one() -> None:
    """It divided every transition collected since start-up by ONE iteration's update seconds.

    Two iterations gave about twice the truth and three about three times, while the
    single-iteration case was right, which is what kept it hidden: the number a reader quotes is
    usually the one they measured in a hurry with one iteration.
    """
    one = _report([_row(1, cumulative=8000.0, update=4.0)])
    assert one["update_timesteps_per_second"] == pytest.approx(2000.0)

    three = _report(
        [
            _row(1, cumulative=8000.0, update=4.0),
            _row(2, cumulative=16000.0, update=4.0),
            _row(3, cumulative=24000.0, update=4.0),
        ]
    )
    assert three["iterations"] == 3
    # The third iteration collected 8,000 of its own and spent 4 s updating them.
    assert three["update_timesteps_per_second"] == pytest.approx(2000.0)


def test_an_iteration_that_collected_less_reports_less() -> None:
    """A short iteration is a slower rate, not the same rate: the row is its own measurement."""
    rows = [_row(1, cumulative=8000.0, update=4.0), _row(2, cumulative=10000.0, update=4.0)]
    assert _report(rows)["update_timesteps_per_second"] == pytest.approx(500.0)


def test_the_environment_cost_is_per_game_step_of_one_iteration() -> None:
    report = _report([_row(1, cumulative=8000.0, update=4.0)])
    assert report["env_ms_per_game_step"] == pytest.approx(1000.0 * 4.0 / 2000)


def test_the_report_carries_the_tolerance_its_deviation_is_judged_against() -> None:
    """``ratio_max_abs_dev`` alone says nothing; it is read beside the configured atol."""
    report = _report([_row(1, cumulative=8000.0, update=4.0)])
    assert report["ratio_atol"] == pytest.approx(0.02)
    assert report["ratio_max_abs_dev"] == pytest.approx(1e-7)
    assert set(report) == {
        "iterations",
        "env_ms_per_game_step",
        "codec_us_per_row",
        "boundary_mb_per_second",
        "inference_ms_per_round",
        "update_timesteps_per_second",
        "rollout_capacity_ratio",
        "vram_peak_mb",
        "ratio_max_abs_dev",
        "ratio_atol",
    }


@pytest.mark.parametrize(
    ("debug", "every", "iterations", "flags"),
    [
        (0, 0, 1, (False, True)),  # the first iteration of the process: filled, not checked
        (0, 0, 3, (False, False)),
        (10, 50, 1, (True, True)),  # the shipped settings: iteration 0 is a checked one
        (1, 2, 3, (True, True)),  # iteration 2 is due a ratio check
    ],
)
def test_filled_is_checked_or_the_first_iteration_of_the_process(
    debug: int, every: int, iterations: int, flags: tuple[bool, bool]
) -> None:
    """``checked_iteration`` alone printed False for a one-iteration bench with no checks due,
    whose one iteration was filled because it was the first the process ran."""
    import msgspec

    from royalelearn import config as cfg
    from royalelearn.cli import bench_fill_flags

    ppo = msgspec.structs.replace(
        cfg.laptop().ppo, debug_assert_iterations=debug, check_ratio_invariant_every=every
    )
    assert bench_fill_flags(ppo, iterations) == flags
