"""``rollout.eval_lockstep``: a worker's battles a decision at a time, one forward per policy.

Each battle keeps its own seed, side, generator and frame history, so it plays what ``play``
plays; the forward over several battles' rows is the only difference, and on the processor here
it chooses the same actions. The scripted anchors act a battle at a time, in seat order, so they
draw from their battle's generator exactly as ``play``'s do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn.config import RolloutConfig, eval_lockstep_of
from royalelearn.ladder.evaluate import BattleRequest
from royalelearn.testing import coordinator, tiny_config

torch = pytest.importorskip("torch")


def _two_snapshots(run: Any) -> None:
    run.snapshot_store.put("snap:a", run.model, {"step": 0})
    run.pool.add("snap:a", step=0)
    with torch.no_grad():
        for parameter in run.model.parameters():
            parameter.mul_(1.5)
    run.snapshot_store.put("snap:b", run.model, {"step": 1})
    run.pool.add("snap:b", step=1)


def _requests(run: Any) -> list[BattleRequest]:
    seeds = run.seeds.seeds
    pairs = [("snap:a", "snap:b"), ("snap:a", "scripted:random_legal"), ("snap:b", "snap:a")]
    return [
        BattleRequest(
            a=a, b=b, seed=int(seeds[i % len(seeds)]), a_seat=i % 2, act_path=f"act/test/{i}"
        )
        for i, (a, b) in enumerate(pairs * 2)
    ]


def _config(tmp_path: Path, frames: int = 1, **rollout: Any) -> Any:
    base = tiny_config(tmp_path)
    return tiny_config(
        tmp_path,
        rollout=msgspec.structs.replace(base.rollout, **rollout),
        obs=msgspec.structs.replace(base.obs, frame_stack=frames),
        alarms=msgspec.structs.replace(base.alarms, enabled=False),
    )


def test_unset_nothing_is_encoded_and_it_plays_a_battle_at_a_time() -> None:
    assert b"eval_lockstep" not in msgspec.json.encode(RolloutConfig())
    assert eval_lockstep_of(RolloutConfig()) is False


@pytest.mark.parametrize("width", [4])
def test_battles_in_lockstep_score_as_they_do_one_at_a_time(tmp_path: Path, width: int) -> None:
    """Plant: draw the anchor's moves from another battle's generator, and a score moves. A
    shared frame history moves actions but rarely a score in a battle this short, so the
    history has its own test below."""
    # Two frames, so each seat's history is part of what it sees.
    with coordinator(_config(tmp_path, frames=2)) as run:
        _two_snapshots(run)
        requests = _requests(run)
        player = run.player
        alone = [
            player.play(
                a=r.a, b=r.b, seed=r.seed, a_seat=r.a_seat, act_path=r.act_path
            )
            for r in requests
        ]
        scores: dict[int, float] = {}
        queue = list(enumerate(requests))
        live: list[Any] = []
        steps = 0
        while queue or live:
            while queue and len(live) < width:
                index, request = queue.pop(0)
                live.append(player.begin(index, request))
            ended = player.advance(live)
            steps += 1
            done = {index for index, _ in ended}
            scores.update(ended)
            live = [battle for battle in live if battle.index not in done]
        assert [scores[i] for i in range(len(requests))] == alone
        assert steps < player.max_decisions * len(requests)


def test_a_farm_in_lockstep_returns_the_comparison_a_battle_at_a_time_does(
    tmp_path: Path,
) -> None:
    """Through the workers: the same eight battles, the same scores, in request order."""
    results = {}
    for lockstep in (False, True):
        config = _config(
            tmp_path / str(lockstep),
            eval_workers=2,
            eval_lockstep=lockstep,
            eval_games_per_worker=4,
        )
        lines: list[str] = []
        with coordinator(config, printer=lines.append) as run:
            _two_snapshots(run)
            comparison = run.eval_runner.compare("snap:a", "snap:b", games=8)
            results[lockstep] = [(s.as_blue, s.as_red) for s in comparison.seed_scores]
            assert run.eval_player.config.width == (4 if lockstep else 1)
        assert any(line.startswith("eval lockstep 4 battles") for line in lines) is lockstep
    assert results[True] == results[False]


def test_each_seat_keeps_its_own_frames(tmp_path: Path) -> None:
    """With two frames, a network seat's history after a decision is that seat's own last
    observation, in every battle, never the other seat's or another battle's. Plant: share one
    history between a battle's seats, and seat 0 holds seat 1's frame."""
    import numpy as np

    with coordinator(_config(tmp_path, frames=2)) as run:
        _two_snapshots(run)
        requests = [r for r in _requests(run) if not r.b.startswith("scripted:")][:3]
        player = run.player
        live = [player.begin(index, request) for index, request in enumerate(requests)]
        apart = 0
        for _ in range(60):
            seen = [
                {seat: battle.obs[("blue", "red")[seat]]["spatial"] for seat in (0, 1)}
                for battle in live
            ]
            ended = {index for index, _score in player.advance(live)}
            for battle, before in zip(live, seen, strict=True):
                if battle.index in ended:
                    continue
                for seat in (0, 1):
                    (frame,) = battle.history[seat]
                    assert np.array_equal(frame["spatial"], np.asarray(before[seat]))
                apart += not np.array_equal(np.asarray(before[0]), np.asarray(before[1]))
            live = [battle for battle in live if battle.index not in ended]
            if not live:
                break
        # The two seats' views differed at some decision, or the check above could not tell.
        assert apart > 0
