"""Does playing evaluation battles in other processes measure the same thing?

The saving is not in question -- a gate is 2,200 battles at 8.02 s each, measured with the real
engine on 2026-09-23, and they are independent by construction. What is in question is whether a
farm and the parent produce the SAME numbers, and the failure that would make them differ is
silent: an environment whose ``reset`` does not fully reseed gives the same answers serially every
time and different ones when the same battles are dealt to four processes.

So these do not assert that the farm is fast. They assert that it is the same. Every test here
plays a batch twice -- once in the parent, once across workers -- and compares.

Marked ``slow``: each one spawns processes and builds environments inside them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from rollout_support import preflight, rollout_config
from royalelearn.coordinator import EnvBattlePlayer
from royalelearn.ladder.actors import EvalActors
from royalelearn.ladder.evaluate import BattleRequest, EvalRunner, eval_seed_set, play_all
from royalelearn.ladder.farm import EvalFarm, EvalWorkerConfig
from royalelearn.ladder.results import ResultLog

pytestmark = pytest.mark.slow

SEED = 4242
#: Scripted on both sides: no snapshot to load, so what is under test is the dispatch rather than
#: the store. The snapshot path gets its own test below.
A = "scripted:random_legal"
B = "scripted:noop"


def _setup(tmp_path: Path) -> tuple[Any, Any]:
    """A MockEngine spec and the eval env spec to play it with."""
    config = rollout_config(workers=1, games_per_worker=2, shards_per_worker=1, max_steps=24)
    report = preflight(config)
    eval_env = msgspec.structs.replace(config.env, truncation=[])
    return report.spec, eval_env


def _parent_player(spec: Any, eval_env: Any, root: Path) -> EnvBattlePlayer:
    from royalelearn.config import NetConfig
    from royalelearn.ladder.snapshots import DiskSnapshotStore

    actors = EvalActors(spec, NetConfig(), DiskSnapshotStore(root))
    return EnvBattlePlayer(spec, eval_env, actors, master_seed=SEED, device="cpu")


def _farm(spec: Any, eval_env: Any, root: Path, fallback: Any, workers: int) -> EvalFarm:
    from royalelearn.config import NetConfig
    from royalelearn.ladder.snapshots import SnapshotSpec

    config = EvalWorkerConfig(
        spec=spec,
        # A real one: the worker decodes this struct and would refuse a None, and the net is what
        # a snapshot loads into. These battles are scripted on both sides and never build one.
        net=NetConfig(),
        env=eval_env,
        extra_modules=(),
        snapshot_root=str(root),
        template=SnapshotSpec(snapshot_id=""),
        master_seed=SEED,
        release_mode="stochastic",
        max_decisions=2000,
        max_resident=4,
    )
    return EvalFarm(config, workers=workers, fallback=fallback)


def _requests(count: int) -> list[BattleRequest]:
    """A batch whose scores are NOT ALL THE SAME, which is the whole point.

    On MockEngine every battle of one matchup goes the same way: ``random_legal`` beats ``noop``
    1.0 every time and loses 0.0 every time with the sides swapped. A batch of one matchup is
    therefore a list of identical numbers, and reversing it changes nothing -- so an equality
    test built on one would pass against a farm that returned its results in COMPLETION order
    instead of request order. Measured: that exact plant was applied here and all five tests
    passed.

    So the batch alternates the two matchups. Now the expected list is [1, 0, 1, 0, ...] and any
    permutation of it is visible.
    """
    seeds = eval_seed_set(SEED, 64)
    return [
        BattleRequest(
            a=A if index % 2 == 0 else B,
            b=B if index % 2 == 0 else A,
            seed=seeds.seeds[index // 2],
            a_seat=index % 2,
            act_path=f"eval/match/farm/{index}",
        )
        for index in range(count)
    ]


def test_a_farm_plays_the_same_battles_to_the_same_scores(tmp_path: Path) -> None:
    """The property everything else rests on, as an equality rather than an intention."""
    spec, eval_env = _setup(tmp_path)
    parent = _parent_player(spec, eval_env, tmp_path / "snapshots")
    requests = _requests(8)
    here = play_all(parent, requests)

    farm = _farm(spec, eval_env, tmp_path / "snapshots", parent, workers=2)
    try:
        there = farm.play_many(requests)
    finally:
        farm.close()
        parent.close()

    assert there == here, "the same battles played in workers scored differently"
    assert len(set(here)) > 1, (
        "every battle in this batch scored the same, so the order could not have been tested"
    )


def test_the_worker_count_does_not_change_the_answer(tmp_path: Path) -> None:
    """One worker and three deal the batch differently; the scores are a property of the battles.

    This is the test that would catch a battle seeing another's state: with one worker every
    battle runs in one process in order, and with three they are split across processes.
    """
    spec, eval_env = _setup(tmp_path)
    parent = _parent_player(spec, eval_env, tmp_path / "snapshots")
    requests = _requests(6)
    results = []
    for workers in (1, 3):
        farm = _farm(spec, eval_env, tmp_path / "snapshots", parent, workers=workers)
        try:
            results.append(farm.play_many(requests))
        finally:
            farm.close()
    parent.close()

    assert results[0] == results[1]


def test_a_comparison_through_a_farm_is_the_comparison(tmp_path: Path) -> None:
    """End to end: the runner's own object, and the rows it writes, from both players."""
    spec, eval_env = _setup(tmp_path)
    parent = _parent_player(spec, eval_env, tmp_path / "snapshots")
    farm = _farm(spec, eval_env, tmp_path / "snapshots", parent, workers=2)

    def runner(player: Any, name: str) -> EvalRunner:
        return EvalRunner(
            player,
            eval_seed_set(SEED, 64),
            master_seed=SEED,
            run_id="run",
            log=ResultLog(tmp_path / f"{name}.jsonl"),
            bootstrap_resamples=200,
        )

    try:
        here = runner(parent, "parent").compare(A, B, games=8)
        there = runner(farm, "farm").compare(A, B, games=8)
    finally:
        farm.close()
        parent.close()

    assert there == here

    # Everything the row records EXCEPT `wall`, which is when the comparison ran and is the one
    # field that must differ between two runs of it. Comparing it would make this a clock test.
    def rows(name: str) -> list[dict[str, Any]]:
        import json

        text = (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8")
        return [
            {key: value for key, value in json.loads(line).items() if key != "wall"}
            for line in text.splitlines()
            if line.strip()
        ]

    assert rows("farm") == rows("parent")


def test_the_live_learner_never_reaches_a_worker(tmp_path: Path) -> None:
    """A worker has no model, and a snapshot would answer a different question.

    The probe asks for the live weights on purpose. Those requests stay in the parent, which is
    where they were before the farm existed, and the farm does not quietly hand back the most
    recent snapshot instead.
    """
    spec, eval_env = _setup(tmp_path)
    played: list[str] = []

    class _Fallback:
        def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
            played.append(a)
            return 0.5

    farm = _farm(spec, eval_env, tmp_path / "snapshots", _Fallback(), workers=2)
    try:
        requests = [
            BattleRequest(
                a="learner@12500000", b=B, seed=1, a_seat=0, act_path="eval/match/probe/0"
            )
        ]
        assert farm.play_many(requests) == [0.5]
    finally:
        farm.close()

    assert played == ["learner@12500000"]
    assert not farm._procs, "a probe started worker processes it cannot use"


def test_an_empty_batch_starts_nothing(tmp_path: Path) -> None:
    """A gate that short-circuits can ask for nothing, and processes are not free."""
    spec, eval_env = _setup(tmp_path)
    farm = _farm(spec, eval_env, tmp_path / "snapshots", None, workers=4)
    assert farm.play_many([]) == []
    assert not farm._procs


# -- the wiring --------------------------------------------------------------


def test_the_gate_gets_a_farm_and_the_probe_keeps_the_parent(tmp_path: Path) -> None:
    """``rollout.eval_workers`` was accepted, in the run identity, and read by nothing.

    The same defect ``rollout.overlap`` had, found by the same question. The gate's runner uses
    the farm and the probe's keeps the parent's player, because a probe plays the LIVE weights
    and no worker has them.
    """
    import royalelearn.config as cfg
    from royalelearn.ladder.farm import EvalFarm
    from test_coordinator import coordinator, tiny_config

    config = tiny_config(
        tmp_path,
        rollout=msgspec.structs.replace(
            cfg.RunConfig().rollout, source="inline", workers=1, eval_workers=3
        ),
    )
    with coordinator(config) as run:
        assert isinstance(run.eval_player, EvalFarm)
        assert run.eval_player.workers == 3
        assert run.eval_runner.player is run.eval_player
        assert run.probe_runner.player is run.player
        assert not run.eval_player._procs, "a farm started workers before a gate asked for any"


def test_one_worker_is_no_farm_at_all(tmp_path: Path) -> None:
    """Spawning a process to play battles one at a time is strictly worse than not."""
    import royalelearn.config as cfg
    from test_coordinator import coordinator, tiny_config

    config = tiny_config(
        tmp_path,
        rollout=msgspec.structs.replace(
            cfg.RunConfig().rollout, source="inline", workers=1, eval_workers=1
        ),
    )
    with coordinator(config) as run:
        assert run.eval_player is run.player
