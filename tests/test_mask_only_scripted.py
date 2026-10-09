"""``rollout.mask_only_scripted``: a scripted opponent's seat stops building its observation.

A scripted opponent reads only the masks, so after its battle's assignment arrives the worker
asks RoyaleGym for that seat's observation as masks alone (``set_mask_only``). The run plays
exactly what it played without: the same actions, rewards, ticks and episodes, and the same
learner rows byte for byte. Only the scripted seats' rows after their episode's first step
differ, and those are rows no update trains on. Not on the shard that carries the viewer or the
recorder, which read the battle's state.
"""

from __future__ import annotations

from typing import Any

import msgspec
import numpy as np
import pytest

from rollout_support import SharedRectangle, drive, preflight, rollout_config
from royalelearn.api.rollout import GROUP_LEARNER, GROUP_SCRIPTED
from royalelearn.rollout.inline import InlineRolloutSource
from royalelearn.rollout.plan import SlotPlanner
from royalelearn.rollout.preflight import DEFAULT_CODEC
from royalelearn.rollout.scripted import SCRIPTED_NAMES

royalegym_env = pytest.importorskip("royalegym.env")
if not hasattr(royalegym_env.ClashParallelEnv, "set_mask_only"):
    pytest.skip(
        "this royalegym has no ClashParallelEnv.set_mask_only", allow_module_level=True
    )

CYCLES = 24
#: The scripted opponent every battle meets, on seat 1: one that plays, so its actions matter.
OPPONENT = SCRIPTED_NAMES.index("first-affordable") if "first-affordable" in SCRIPTED_NAMES else 0


def _collect(on: bool, run_id: str, **rollout: Any) -> tuple[Any, Any]:
    config = rollout_config(
        workers=1,
        games_per_worker=4,
        shards_per_worker=2,
        max_steps=9,
        mask_only_scripted=on,
        **rollout,
    )
    report = preflight(config, codec=DEFAULT_CODEC)
    planner = SlotPlanner(report.geometry, config.master_seed)
    buffer = SharedRectangle(
        report.spec,
        run_id=run_id,
        cycles=CYCLES,
        n_slots=report.geometry.n_slots,
        row_bytes=report.row_bytes,
    )
    source = InlineRolloutSource(
        config, report.spec, report.table, run_id=run_id, codec=DEFAULT_CODEC
    )

    def matchmaker(_battle: int, _ordinal: int) -> Any:
        return (GROUP_LEARNER, GROUP_SCRIPTED), (OPPONENT, OPPONENT), 0

    def policy(round_: Any, _rect: Any) -> np.ndarray:
        return ((round_.slots + round_.cycle) % 3).astype(np.int16)

    try:
        source.begin_iteration(planner.mirror_plan(0), buffer, 0)
        runners = dict(source._runners)
        out = drive(source, buffer, report, cycles=CYCLES, policy=policy, matchmaker=matchmaker)
    finally:
        source.close()
        buffer.close()
    return out, runners


def test_a_run_plays_the_same_and_only_scripted_rows_after_their_first_change() -> None:
    """Plant: never switch, and no row differs; switch the learner's seat too, and its rows do."""
    off, _ = _collect(False, "mask-off")
    on, runners = _collect(True, "mask-on")
    assert all(runner.mask_only_scripted for runner in runners.values())
    for column in ("reward", "tick", "terminated", "truncated", "deploy_status", "episode_end"):
        assert np.array_equal(getattr(on, column), getattr(off, column)), column
    assert [msgspec.json.encode(e) for e in on.episodes] == [
        msgspec.json.encode(e) for e in off.episodes
    ]
    assert on.rows is not None and off.rows is not None
    differs = (on.rows[on.obs_rows] != off.rows[off.obs_rows]).any(axis=-1)
    assert differs.any(), "the switch changed no row"
    # A round's done flags belong to the step that produced its observation, so a cell whose
    # step ended an episode holds the next episode's first observation.
    first = on.terminated | on.truncated
    first[0] = True
    cycles, slots = np.nonzero(differs)
    assert np.all(on.group[cycles, slots] == GROUP_SCRIPTED), "a learner's row changed"
    assert not np.any(first[cycles, slots]), "an episode's first observation changed"


def test_not_on_the_shard_that_carries_the_recorder(tmp_path: Any) -> None:
    from royalelearn.rollout.envspec import ComponentSpec

    recorder = ComponentSpec(
        "royalegym.replay.SavingReplayRecorder", {"out_dir": str(tmp_path), "every": 1, "keep": 2}
    )
    _out, runners = _collect(True, "mask-rec", recorder=recorder)
    assert runners[(0, 0)].mask_only_scripted is False
    assert runners[(0, 1)].mask_only_scripted is True


def test_an_environment_without_the_switch_is_refused_by_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before any worker or shared memory exists."""
    from royalelearn.errors import PreflightError

    monkeypatch.delattr(royalegym_env.ClashParallelEnv, "set_mask_only")
    config = rollout_config(workers=1, games_per_worker=2, mask_only_scripted=True)
    with pytest.raises(PreflightError, match="set_mask_only"):
        preflight(config, codec=DEFAULT_CODEC)
