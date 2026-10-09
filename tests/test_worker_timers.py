"""Where a round's time goes: each worker's busy and idle time, the straggle, the parent's wait.

A round is lock-step. The parent waits until every worker has published a shard, runs inference
on it and answers, while each worker steps its other shard. ``time/env`` sums the workers'
environment time: it says how much processor time the environments took, and not where the
round's wall clock went. These say that:

- each worker reports, in the control word it publishes, how long it worked on the command it
  answers (``t_busy_ns``: the step, packing the rows, publishing) and how long it has waited for
  commands in all (``t_idle_ns``, a running total);
- the parent turns them into per-worker means for each round, and the gap between its slowest
  worker and that mean, the straggle, which every other worker waits out;
- and it times its own wait for the publications.

A worker's first publication of an iteration only sets where its idle total stands: before it,
the worker was waiting out the update, which is not a cost of collection.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from rollout_support import SharedRectangle, drive, preflight, rollout_config
from royalelearn.rollout.inline import round_timings

MS = 1_000_000


def _word(env_ms: float, busy_ms: float, idle_ms: float) -> dict[str, int]:
    return {
        "t_env_ns": int(env_ms * MS),
        "t_busy_ns": int(busy_ms * MS),
        "t_idle_ns": int(idle_ms * MS),
    }


def test_a_round_is_described_per_worker_and_not_only_summed() -> None:
    """Plant: report the summed environment time as the per-worker one, and two workers each
    stepping for 4 and 8 ms read as one that stepped for 12."""
    seen: dict[int, int] = {}
    first = round_timings({0: _word(4, 5, 100), 1: _word(8, 9, 50)}, seen)
    assert first["env_ms"] == pytest.approx(12.0)
    assert first["env_mean_ms"] == pytest.approx(6.0)
    assert first["busy_mean_ms"] == pytest.approx(7.0)
    assert first["busy_max_ms"] == pytest.approx(9.0)


def test_idle_counts_from_where_each_worker_stood_when_first_seen() -> None:
    """Plant: count a worker's whole running total on first sight, and the update it waited
    out before this iteration's first round lands in the first round's idle time."""
    seen: dict[int, int] = {}
    first = round_timings({0: _word(1, 1, 100), 1: _word(1, 1, 50)}, seen)
    assert first["idle_ms"] == 0.0
    second = round_timings({0: _word(1, 1, 130), 1: _word(1, 1, 60)}, seen)
    assert second["idle_ms"] == pytest.approx(20.0)  # the mean of 30 and 10


def test_a_restarted_workers_total_starts_again_and_is_not_counted_backwards() -> None:
    """A new process counts from zero. Plant: take the difference anyway, and a restart reads
    as a large negative idle time."""
    seen: dict[int, int] = {}
    round_timings({0: _word(1, 1, 100), 1: _word(1, 1, 500)}, seen)
    after = round_timings({0: _word(1, 1, 140), 1: _word(1, 1, 5)}, seen)
    assert after["idle_ms"] == pytest.approx(20.0)  # 40 for worker 0, nothing for worker 1
    later = round_timings({0: _word(1, 1, 140), 1: _word(1, 1, 25)}, seen)
    assert later["idle_ms"] == pytest.approx(10.0)


def test_an_inline_source_has_no_wait_and_no_idle_and_still_times_its_work() -> None:
    from royalelearn.rollout.inline import InlineRolloutSource
    from royalelearn.rollout.preflight import DEFAULT_CODEC

    config = rollout_config(workers=2, games_per_worker=2, shards_per_worker=2, max_steps=7)
    report = preflight(config, codec=DEFAULT_CODEC)
    timings = _collect(InlineRolloutSource, "timers-inline", config, report, cycles=6)
    # The first round of each shard is the plan's republication: no step, so no environment time,
    # where it used to carry the last step's a second time.
    assert [t["env_mean_ms"] for t in timings[:2]] == [0.0, 0.0]
    assert all(t["busy_mean_ms"] >= t["env_mean_ms"] > 0.0 for t in timings[2:])
    assert all(t["idle_ms"] == 0.0 and t["parent_wait_ms"] == 0.0 for t in timings)


def _collect(
    source_cls: Any,
    run_id: str,
    config: Any,
    report: Any,
    *,
    cycles: int,
    think_s: float = 0.0,
) -> list[dict[str, float]]:
    """Every round's timings over one iteration, with the parent taking ``think_s`` a round."""
    from royalelearn.rollout.plan import SlotPlanner

    planner = SlotPlanner(report.geometry, config.master_seed)
    buffer = SharedRectangle(
        report.spec,
        run_id=run_id,
        cycles=cycles,
        n_slots=report.geometry.n_slots,
        row_bytes=report.row_bytes,
    )
    source = source_cls(config, report.spec, report.table, run_id=run_id, codec=_codec())
    seen: list[dict[str, float]] = []

    def policy(round_: Any, _rect: Any) -> np.ndarray:
        seen.append(dict(round_.timings))
        if think_s:
            time.sleep(think_s)
        return np.zeros(round_.slots.size, dtype=np.int16)

    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        drive(source, buffer, report, cycles=cycles, policy=policy)
    finally:
        source.close()
        buffer.close()
    return seen


def _codec() -> str:
    from royalelearn.rollout.preflight import DEFAULT_CODEC

    return DEFAULT_CODEC


@pytest.mark.slow
def test_a_slow_parent_shows_as_idle_workers_and_not_as_a_wait() -> None:
    """The parent takes 30 ms a round, and the workers about 1 ms a step: they wait on it.

    Plant: a worker that never adds its waits to its total, and this reads no idle time at all.
    """
    from royalelearn.rollout.farm import ProcessRolloutSource
    from royalelearn.rollout.preflight import DEFAULT_CODEC

    config = rollout_config(
        workers=2,
        games_per_worker=2,
        shards_per_worker=2,
        max_steps=9,
        source="process",
        round_timeout_s=60.0,
    )
    report = preflight(config, codec=DEFAULT_CODEC)
    timings = _collect(
        ProcessRolloutSource, "timers-farm", config, report, cycles=12, think_s=0.03
    )
    steady = timings[4:]
    idle = float(np.mean([t["idle_ms"] for t in steady]))
    wait = float(np.mean([t["parent_wait_ms"] for t in steady]))
    busy = float(np.mean([t["busy_mean_ms"] for t in steady]))
    assert all(t["busy_mean_ms"] >= t["env_mean_ms"] > 0.0 for t in steady)
    assert all(t["busy_max_ms"] >= t["busy_mean_ms"] for t in steady)
    assert idle > 15.0, (idle, busy, wait)
    assert idle > 3 * wait, (idle, busy, wait)


@pytest.mark.slow
def test_a_slow_environment_shows_as_the_parents_wait_and_busy_workers() -> None:
    """Each step takes at least 4 ms in the reward alone -- one battle a shard, two seats, 2 ms a
    call, which a sleep never cuts short -- and the parent answers at once: it waits.

    Plant: a parent that stops timing its wait for the publications, and this reads none.
    """
    from rollout_support import SUPPORT_MODULE
    from royalelearn.rollout.envspec import ComponentSpec
    from royalelearn.rollout.farm import ProcessRolloutSource
    from royalelearn.rollout.preflight import DEFAULT_CODEC

    config = rollout_config(
        workers=2,
        games_per_worker=2,
        shards_per_worker=2,
        max_steps=9,
        source="process",
        round_timeout_s=60.0,
        reward=ComponentSpec(f"{SUPPORT_MODULE}.SleepyReward", {"seconds": 0.002}),
    )
    report = preflight(config, codec=DEFAULT_CODEC)
    timings = _collect(ProcessRolloutSource, "timers-slow-env", config, report, cycles=12)
    steady = timings[4:]
    idle = float(np.mean([t["idle_ms"] for t in steady]))
    wait = float(np.mean([t["parent_wait_ms"] for t in steady]))
    busy = float(np.mean([t["busy_mean_ms"] for t in steady]))
    assert busy > 3.5, (idle, busy, wait)
    assert wait > 2.0, (idle, busy, wait)
    assert wait > idle, (idle, busy, wait)


@pytest.mark.slow
def test_a_process_runs_metrics_row_splits_its_collection(tmp_path: Any) -> None:
    """The coordinator adds the rounds up into the row. Plant: leave one out of the sum, and it
    reads zero beside a collection that plainly did the work."""
    pytest.importorskip("torch")
    import json

    import msgspec

    from royalelearn.testing import coordinator, tiny_config

    base = tiny_config(tmp_path)
    config = tiny_config(
        tmp_path,
        rollout=msgspec.structs.replace(
            base.rollout, source="process", workers=2, shards_per_worker=2, round_timeout_s=60.0
        ),
        ppo=msgspec.structs.replace(base.ppo, timesteps_per_iteration=64),
    )
    with coordinator(config) as run:
        run.iterate()
        run.iterate()
        row = json.loads((run.run_dir / "metrics.jsonl").read_bytes().splitlines()[-1])
    parts = row["time/parent_wait"] + row["time/inference"] + row["time/parent_other"]
    assert parts == pytest.approx(row["time/collection"], rel=1e-6)
    assert row["time/worker_busy"] >= row["time/worker_env"] > 0.0
    # One worker's environment time, against the two workers' summed.
    assert row["time/worker_env"] < row["time/env"]
    assert row["time/worker_idle"] > 0.0
    assert row["time/worker_straggle"] >= 0.0
    assert 0.0 < row["throughput/worker_busy_frac"] < 1.0
