"""``rollout.overlap``: the next iteration is collected while the update trains on this one.

The workers write one rectangle; when a collection ends, the update's rectangle takes a copy of
it, and the next collection runs on a second thread beside the update, sampling the learner's
seats from a snapshot of the actor taken before the update. So the first iteration's batch is
the learner's own (lag 0) and every later one is one update behind (lag 1). The ratio invariant
is asserted against the snapshot that sampled the batch, which is what makes it still a check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import config as cfg
from royalelearn.testing import coordinator, tiny_config


def _overlapped(tmp_path: Path) -> cfg.RunConfig:
    config = tiny_config(tmp_path)
    return msgspec.structs.replace(
        config, rollout=msgspec.structs.replace(config.rollout, overlap=True)
    )


def test_overlap_is_accepted_and_refused_only_under_run_exact(tmp_path: Path) -> None:
    config = _overlapped(tmp_path)
    assert cfg.check_consistency(config) == []
    exact = msgspec.structs.replace(
        config, determinism=msgspec.structs.replace(config.determinism, tier="run_exact")
    )
    assert any("rollout.overlap" in p for p in cfg.check_consistency(exact))


def test_three_iterations_train_one_behind_with_the_invariant_asserted(tmp_path: Path) -> None:
    """Lag 0, then 1, then 1; the ratio invariant runs on every one of them (the debug
    iterations), so a batch checked against the live learner instead of its snapshot fails."""
    config = _overlapped(tmp_path)
    per_iteration = config.ppo.timesteps_per_iteration
    with coordinator(config) as run:
        assert run.buffer is not run.collect_buffer
        assert config.ppo.debug_assert_iterations >= 3
        run.learn(until_timesteps=3 * per_iteration)
        rows = list(run.rows)
        assert run._prefetch is None and run._prefetched is None
    assert [row["ppo/behaviour_lag_iterations"] for row in rows] == [0, 1, 1]
    assert all(row["time/overlap_saved"] >= 0.0 for row in rows)
    assert rows[-1]["run/cumulative_timesteps"] >= 3 * per_iteration
    assert all(row["ppo/ratio_max_abs_dev"] <= 1e-4 for row in rows)


def test_no_batch_is_collected_ahead_that_no_iteration_will_train(tmp_path: Path) -> None:
    config = _overlapped(tmp_path)
    with coordinator(config) as run:
        started = []
        real = run._start_prefetch
        run._start_prefetch = lambda trainable: (started.append(1), real(trainable))[1]
        run.learn(until_timesteps=2 * config.ppo.timesteps_per_iteration)
    assert started == [1], "one prefetch for the second iteration, none after the last"


def test_a_failure_beside_the_update_is_raised_by_the_iteration(tmp_path: Path) -> None:
    config = _overlapped(tmp_path)
    with coordinator(config) as run:
        real = run._collect

        def failing(*args: Any, overlapped: bool = False, **kwargs: Any) -> Any:
            if overlapped:
                raise RuntimeError("collection failed beside the update")
            return real(*args, overlapped=overlapped, **kwargs)

        run._collect = failing
        with pytest.raises(RuntimeError, match="beside the update"):
            run.learn(until_timesteps=3 * config.ppo.timesteps_per_iteration)


def test_the_update_trains_on_an_exact_copy_of_what_was_collected(tmp_path: Path) -> None:
    config = _overlapped(tmp_path)
    with coordinator(config) as run:
        run.iterate()
        copy, source = run.buffer, run.collect_buffer
        used = copy.layout.row_index(copy.cycles, 0) + copy.n_slots
        assert np.array_equal(copy.obs_view[:used], source.obs_view[:used])
        for name in ("action", "log_prob", "reward", "valid", "group", "terminated"):
            assert np.array_equal(getattr(copy, name), getattr(source, name)), name
        assert copy.plan is source.plan and copy.iteration == source.iteration


def test_the_batch_beside_the_update_is_sampled_from_the_snapshot_not_the_moving_learner(
    tmp_path: Path,
) -> None:
    """Held until the learner has moved, the collection still samples the actor as it stood
    before the update: the next iteration's invariant, asserted against that snapshot, passes.
    Sampled from the live learner, its stored log-probabilities would be the moved weights'."""
    import threading

    import torch

    config = _overlapped(tmp_path)
    moved = threading.Event()
    with coordinator(config) as run:
        real_begin = run.inference.begin_iteration
        real_step = run.update.step

        def begin(plan: Any, **kwargs: Any) -> None:
            if threading.current_thread().name == "royalelearn-collect":
                assert moved.wait(timeout=60)
            real_begin(plan, **kwargs)

        def step(buffer: Any, sched: Any, **kwargs: Any) -> Any:
            result = real_step(buffer, sched, **kwargs)
            if run._prefetch is not None:
                # The update is done and has moved the learner; move it further, then let the
                # collection beside it start sampling.
                with torch.no_grad():
                    for parameter in run.model.actor.parameters():
                        parameter.add_(0.5 * torch.randn_like(parameter))
                moved.set()
            return result

        run.inference.begin_iteration = begin
        run.update.step = step
        run.learn(until_timesteps=2 * config.ppo.timesteps_per_iteration)
        rows = list(run.rows)
    assert [row["ppo/behaviour_lag_iterations"] for row in rows] == [0, 1]
    assert rows[-1]["ppo/ratio_max_abs_dev"] <= 1e-4
