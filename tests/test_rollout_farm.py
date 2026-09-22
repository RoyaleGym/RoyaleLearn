"""The differential test: the process farm must be the inline source, byte for byte.

This is the acceptance criterion for any implementation of the worker side -- the process farm
today, a Rust worker the day it exists. Same seed, same geometry, same plan, same actions, and
then the three things the learner reads have to be identical: the rectangle of packed
observations, the per-round scalars, and the episode records.

It is marked ``slow`` because it spawns processes and builds environments in each of them. It
is the only test in the suite that does, which is the point: everything else runs against the
inline source because the inline source is what this test says the farm is.

It runs against the codec a real run uses, not the one the other rollout tests read rows back
with: what crosses the boundary here is the bytes, and the bytes are the shipped codec's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rollout_support import SharedRectangle, drive, preflight, rollout_config
from royalelearn.api.rollout import GROUP_LEARNER, GROUP_SCRIPTED
from royalelearn.rollout.farm import ProcessRolloutSource
from royalelearn.rollout.inline import InlineRolloutSource
from royalelearn.rollout.plan import SlotPlanner
from royalelearn.rollout.preflight import DEFAULT_CODEC
from royalelearn.rollout.scripted import SCRIPTED_NAMES
from royalelearn.seeding import derive_generator

pytestmark = pytest.mark.slow

CYCLES = 30
MAX_STEPS = 11


def _run(source_cls: Any, run_id: str, config: Any, report: Any) -> Any:
    """One iteration through one source, with everything that could differ held fixed."""
    planner = SlotPlanner(report.geometry, config.master_seed)
    buffer = SharedRectangle(
        report.spec,
        run_id=run_id,
        cycles=CYCLES,
        n_slots=report.geometry.n_slots,
        row_bytes=report.row_bytes,
    )
    source = source_cls(
        config, report.spec, report.table, run_id=run_id, codec=DEFAULT_CODEC
    )
    generator = derive_generator(config.master_seed, "test/farm/policy")

    def policy(round_: Any, _rect: SharedRectangle) -> np.ndarray:
        """Actions that depend on the cycle and the slot and on nothing else.

        Not a mask-aware policy: what is under test is the boundary, and an action the engine
        refuses is reported in ``deploy_status``, which is one of the columns being compared.
        """
        draw = generator.integers(0, 2, size=round_.slots.size)
        return (draw * (round_.slots + round_.cycle + 1)).astype(np.int16)

    def matchmaker(battle: int, ordinal: int) -> tuple[tuple[int, int], tuple[int, int], int]:
        draw = derive_generator(config.master_seed, planner.match_path(battle, ordinal))
        if draw.random() < 0.5:
            return (GROUP_LEARNER, GROUP_LEARNER), (-1, -1), -1
        seat = int(draw.integers(2))
        groups = [GROUP_LEARNER, GROUP_LEARNER]
        groups[1 - seat] = GROUP_SCRIPTED
        index = int(draw.integers(len(SCRIPTED_NAMES)))
        return (groups[0], groups[1]), (index, index), seat

    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        return drive(
            source, buffer, report, cycles=CYCLES, policy=policy, matchmaker=matchmaker
        )
    finally:
        source.close()
        buffer.close()


@pytest.fixture(scope="module")
def both() -> tuple[Any, Any, Any]:
    """The same iteration, collected twice: once in this process and once in three others."""
    config = rollout_config(
        workers=3,
        games_per_worker=2,
        shards_per_worker=2,
        max_steps=MAX_STEPS,
        stagger_first_reset=True,
        round_timeout_s=60.0,
    )
    report = preflight(config, codec=DEFAULT_CODEC)
    inline = _run(InlineRolloutSource, "diff-inline", config, report)
    farm = _run(ProcessRolloutSource, "diff-farm", config, report)
    return inline, farm, report


def test_the_rectangles_are_identical(both: tuple[Any, Any, Any]) -> None:
    """Every packed observation of every slot of every cycle, to the byte."""
    inline, farm, report = both
    assert inline.rows.shape == farm.rows.shape
    if not np.array_equal(inline.rows, farm.rows):
        differing = np.flatnonzero((inline.rows != farm.rows).any(axis=1))
        cycles = differing // report.geometry.n_slots
        raise AssertionError(
            f"{differing.size} of {inline.rows.shape[0]} rows differ, first at row "
            f"{int(differing[0])} (cycle {int(cycles[0]) - report.spec.frame_stack + 1})"
        )


def test_the_scalars_are_identical(both: tuple[Any, Any, Any]) -> None:
    """The columns a round carries: rewards, flags, ticks, groups, statuses, ends."""
    inline, farm, _ = both
    for column in (
        "reward",
        "tick",
        "terminated",
        "truncated",
        "valid",
        "deploy_status",
        "episode_end",
        "group",
        "published_group",
        "obs_rows",
        "action",
        "shard",
    ):
        left = getattr(inline, column)
        right = getattr(farm, column)
        assert np.array_equal(left, right), f"the {column} column differs"


def test_the_episode_records_are_identical(both: tuple[Any, Any, Any]) -> None:
    """Including the reward breakdown, which the worker accumulates as the episode runs."""
    inline, farm, _ = both
    assert len(inline.episodes) == len(farm.episodes)
    assert inline.episodes, "no episode finished, so the records proved nothing"
    for left, right in zip(inline.episodes, farm.episodes, strict=True):
        assert left == right


def test_the_final_observations_are_identical(both: tuple[Any, Any, Any]) -> None:
    """The truncation bootstrap rows, which live in the control segment and not the rectangle."""
    inline, farm, _ = both
    assert set(inline.finals) == set(farm.finals)
    assert inline.finals, "nothing truncated, so the bootstrap path was never compared"
    for key, rows in inline.finals.items():
        assert np.array_equal(rows, farm.finals[key])


def test_the_farm_reports_what_its_workers_are(both: tuple[Any, Any, Any]) -> None:
    """A run collected in three processes is a run whose slots came from three processes."""
    _, farm, report = both
    assert farm.valid.all()
    assert len({int(v) for v in farm.shard.reshape(-1)}) == report.geometry.shards_per_worker
