"""``ladder.pfsp_source`` "games": PFSP weighs the pool by the learner's own learning games.

The fit predicts the learner's score against a member from ratings, through the snapshot that
stands in for the learner, which lags the learner by a snapshot cadence and reads one half for a
member the fit holds no games for. This source reads the learner's record instead: its latest
``ladder.pfsp_window`` learning games against each member, a draw as a half, one half before
any. ``ladder.pfsp_offset`` adds a constant to every shaped weight, so with the hard shape at
power 2 and both floors at zero a member's weight is (1 - score)^2 + 0.05. The scores are read
as each plan is drawn, so every episode of an iteration is drawn by one set of weights, and the
record and the resident set it drew are kept in the matchmaker's checkpoint, because neither is
a function of the pool any longer.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import numpy as np
import pytest

from royalelearn.config import (
    LadderConfig,
    RolloutConfig,
    RunConfig,
    check_consistency,
    geometry,
    pfsp_offset_of,
    pfsp_source_of,
    pfsp_window_of,
)
from royalelearn.ladder.matchmaker import MixMatchmaker
from royalelearn.ladder.pool import LadderPool
from royalelearn.ladder.results import ResultLog
from royalelearn.testing import coordinator, tiny_config
from test_matchmaker import _record

SEED = 20261111
MEMBERS = 6


def _pool(tmp_path: Path, members: int = MEMBERS) -> LadderPool:
    ladder = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    for index in range(members):
        ladder.add(f"snap:{index}", step=index * 1000)
    return ladder


def _ladder(**fields: object) -> LadderConfig:
    base = LadderConfig(
        mix=(0.0, 1.0, 0.0),
        max_resident_opponents=MEMBERS,
        pfsp_source="games",
        pfsp_offset=0.05,
        pfsp_uniform_floor=0.0,
        weight_floor_scale=0.0,
    )
    return msgspec.structs.replace(base, **fields)


def _matchmaker(ladder: LadderConfig) -> tuple[MixMatchmaker, object]:
    config = RunConfig(
        rollout=RolloutConfig(workers=1, games_per_worker=8, shards_per_worker=1), ladder=ladder
    )
    shape = geometry(config)
    return MixMatchmaker(SEED, config.ladder, n_battles=shape.n_battles), shape


def _weights(matchmaker: MixMatchmaker, pool: LadderPool) -> dict[str, float]:
    members = pool.sampler()
    weights = matchmaker.weights(members, None, matchmaker.config.pfsp_weighting)
    return dict(zip(members, weights.tolist(), strict=True))


def test_unset_nothing_is_encoded() -> None:
    encoded = msgspec.json.encode(LadderConfig())
    for name in ("pfsp_source", "pfsp_window", "pfsp_offset"):
        assert name.encode() not in encoded
    assert pfsp_source_of(LadderConfig()) == "fit"
    assert pfsp_window_of(LadderConfig()) == 200
    assert pfsp_offset_of(LadderConfig()) == 0.0


@pytest.mark.parametrize(
    "fields,words",
    [
        ({"pfsp_source": "elo"}, "ladder.pfsp_source is 'elo'"),
        ({"pfsp_source": "games", "pfsp_window": 0}, "pfsp_window must be at least 1"),
        ({"pfsp_window": 50}, "only source that reads it"),
        ({"pfsp_offset": -0.1}, "pfsp_offset is -0.1"),
        ({"pfsp_offset": float("nan")}, "pfsp_offset is nan"),
    ],
)
def test_a_setting_nothing_can_honour_is_named(fields: dict[str, object], words: str) -> None:
    config = RunConfig(ladder=msgspec.structs.replace(LadderConfig(), **fields))
    assert any(words in problem for problem in check_consistency(config))


def test_the_weights_are_the_learners_record_shaped(tmp_path: Path) -> None:
    """(1 - score)^2 + 0.05 over each member's record, one half for a member not yet played.
    Plant: read the fit, and every member reads one half and the weights are uniform."""
    matchmaker, shape = _matchmaker(_ladder())
    pool = _pool(tmp_path)
    matchmaker.record_learning_games(
        [("snap:0", 1.0)] * 4 + [("snap:1", 0.0)] * 4 + [("snap:2", 0.5), ("snap:2", 1.0)]
    )
    matchmaker.plan(0, pool, None, shape)
    scores = {"snap:0": 1.0, "snap:1": 0.0, "snap:2": 0.75}
    assert matchmaker.learning_scores() == scores
    raw = {m: (1.0 - scores.get(m, 0.5)) ** 2 + 0.05 for m in pool.sampler()}
    total = sum(raw.values())
    got = _weights(matchmaker, pool)
    for member, weight in raw.items():
        assert got[member] == pytest.approx(weight / total)
    assert got["snap:1"] > got["snap:3"] > got["snap:2"] > got["snap:0"] > 0.0


def test_the_window_keeps_the_latest_games(tmp_path: Path) -> None:
    matchmaker, shape = _matchmaker(_ladder(pfsp_window=3))
    pool = _pool(tmp_path)
    matchmaker.record_learning_games([("snap:0", 1.0)] * 5 + [("snap:0", 0.0)] * 3)
    matchmaker.plan(0, pool, None, shape)
    assert matchmaker.learning_scores() == {"snap:0": 0.0}


def test_games_reach_the_weights_at_the_next_plan(tmp_path: Path) -> None:
    """Every episode of an iteration is drawn by its plan's weights: games recorded meanwhile
    move nothing until the next plan reads them."""
    matchmaker, shape = _matchmaker(_ladder())
    pool = _pool(tmp_path)
    matchmaker.plan(0, pool, None, shape)
    before = _weights(matchmaker, pool)
    matchmaker.record_learning_games([("snap:4", 0.0)] * 10)
    assert _weights(matchmaker, pool) == before
    matchmaker.plan(1, pool, None, shape)
    assert _weights(matchmaker, pool)["snap:4"] > before["snap:4"]


def test_a_member_no_longer_drawable_leaves_the_record(tmp_path: Path) -> None:
    matchmaker, shape = _matchmaker(_ladder())
    pool = _pool(tmp_path)
    matchmaker.record_learning_games([("snap:2", 1.0), ("scripted:noop", 1.0)])
    pool.evict(["snap:2"])
    matchmaker.plan(0, pool, None, shape)
    assert matchmaker.learning_scores() == {}


def test_off_nothing_is_kept_and_the_checkpoint_is_as_it_was(tmp_path: Path) -> None:
    matchmaker, shape = _matchmaker(LadderConfig(mix=(0.0, 1.0, 0.0)))
    pool = _pool(tmp_path)
    matchmaker.record_learning_games([("snap:0", 1.0)])
    matchmaker.plan(0, pool, None, shape)
    assert matchmaker.learning_scores() == {}
    matchmaker.save_checkpoint(tmp_path / "ckpt")
    state = msgspec.json.decode((tmp_path / "ckpt" / "matchmaker.json").read_bytes())
    assert set(state) == {"ordinal", "n_battles"}


def test_a_resume_draws_what_the_run_would_have(tmp_path: Path) -> None:
    """The record moves every iteration and the resident set is drawn only when the residency
    epoch moves, by the scores of that moment. Both are in the checkpoint, so a resumed run
    holds the residents the run drew and weighs them by the record it had. Plant: leave the
    residents out, and the resumed plan draws them again by later scores."""
    ladder = _ladder(max_resident_opponents=3)
    first, shape = _matchmaker(ladder)
    pool = _pool(tmp_path, members=12)
    first.plan(0, pool, None, shape)
    residents = first.residents(pool, None)
    # Lose to everyone outside the resident set, so a fresh draw would favour them.
    first.record_learning_games(
        [(member, 0.0) for member in pool.sampler() if member not in residents] * 20
        + [(member, 1.0) for member in residents] * 20
    )
    first.save_checkpoint(tmp_path / "ckpt")
    resumed, _ = _matchmaker(ladder)
    resumed.load_checkpoint(tmp_path / "ckpt")
    first.plan(1, pool, None, shape)
    resumed.plan(1, pool, None, shape)
    assert resumed.residents(pool, None) == residents == first.residents(pool, None)
    assert resumed.learning_scores() == first.learning_scores()
    assert np.array_equal(
        resumed.weights(residents, None, ladder.pfsp_weighting),
        first.weights(residents, None, ladder.pfsp_weighting),
    )


def test_the_coordinator_files_each_learning_game_with_the_matchmaker(tmp_path: Path) -> None:
    base = tiny_config(tmp_path)
    config = tiny_config(
        tmp_path, ladder=msgspec.structs.replace(base.ladder, pfsp_source="games")
    )
    with coordinator(config) as run:
        won = msgspec.structs.replace(_record(0, 0), opponent_id="snap:a", outcome=1)
        lost = msgspec.structs.replace(_record(1, 0), opponent_id="snap:a", outcome=-1)
        mirror = msgspec.structs.replace(_record(2, 0), opponent_id="learner", outcome=1)
        run._record_training_results([won, lost, lost, mirror])
        assert run.matchmaker._games == {"snap:a": [1.0, 0.0, 0.0]}
