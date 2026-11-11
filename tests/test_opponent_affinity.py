"""``ladder.opponent_affinity``: the pool battles of one shard on one episode ordinal meet one
opponent.

The inference makes one forward per frozen opponent present in a shard's round. Drawn per
battle, a shard of a dozen pool battles against a pool of two dozen holds about ten of them, so
a round is about ten forwards of a row or two. With affinity, the battles of a shard that are on
the same episode share one draw, by the same PFSP weights, from the shard's own stream; a shard's
battles are an episode or two apart, so a round holds one or two. Each episode's opponent is
still a draw by the weights, a battle's seat is what it was, and an episode keeps its opponent
across an iteration boundary as it always has, because the draw is addressed by the ordinal.
"""

from __future__ import annotations

from collections import Counter

import msgspec
import pytest

from royalelearn.config import LadderConfig, RolloutConfig, RunConfig, geometry
from royalelearn.ladder.matchmaker import MixMatchmaker
from royalelearn.ladder.pool import LadderPool
from royalelearn.ladder.results import ResultLog

SEED = 20261111
MEMBERS = 25


def _pool(tmp_path) -> LadderPool:
    ladder = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    for index in range(MEMBERS):
        ladder.add(f"snap:{index}", step=index * 1000)
    return ladder


def _setup(tmp_path, affinity: bool) -> tuple[MixMatchmaker, LadderPool, object]:
    ladder = LadderConfig(mix=(0.0, 1.0, 0.0), max_resident_opponents=MEMBERS)
    if affinity:
        ladder = msgspec.structs.replace(ladder, opponent_affinity=True)
    config = RunConfig(
        rollout=RolloutConfig(workers=2, games_per_worker=12, shards_per_worker=2),
        ladder=ladder,
    )
    shape = geometry(config)
    matchmaker = MixMatchmaker(SEED, config.ladder, n_battles=shape.n_battles)
    pool = _pool(tmp_path)
    matchmaker.plan(0, pool, None, shape)
    return matchmaker, pool, shape


def _shard(shape: object, battle: int) -> tuple[int, int]:
    return (
        battle // shape.games_per_worker,  # type: ignore[attr-defined]
        (battle % shape.games_per_worker) // shape.games_per_shard,  # type: ignore[attr-defined]
    )


def test_unset_nothing_is_encoded() -> None:
    assert b"opponent_affinity" not in msgspec.json.encode(LadderConfig())


def test_a_shards_battles_on_one_ordinal_meet_one_opponent(tmp_path) -> None:
    """Plant: draw per battle anyway, and a shard on one ordinal meets about ten opponents."""
    matchmaker, pool, shape = _setup(tmp_path, affinity=True)
    plain, plain_pool, _ = _setup(tmp_path / "plain", affinity=False)
    for ordinal in range(3):
        met: dict[tuple[int, int], set[str]] = {}
        plain_met: dict[tuple[int, int], set[str]] = {}
        for battle in range(shape.n_battles):  # type: ignore[attr-defined]
            shard = _shard(shape, battle)
            met.setdefault(shard, set()).add(matchmaker.assign(battle, ordinal, pool).opponent_id)
            plain_met.setdefault(shard, set()).add(
                plain.assign(battle, ordinal, plain_pool).opponent_id
            )
        assert all(len(opponents) == 1 for opponents in met.values()), met
        assert max(len(opponents) for opponents in plain_met.values()) > 3
        # The shards draw from streams of their own.
        assert len({next(iter(opponents)) for opponents in met.values()}) > 1


def test_a_battle_keeps_its_seat_and_another_ordinal_is_another_draw(tmp_path) -> None:
    matchmaker, pool, shape = _setup(tmp_path, affinity=True)
    plain, plain_pool, _ = _setup(tmp_path / "plain", affinity=False)
    for battle in range(shape.n_battles):  # type: ignore[attr-defined]
        for ordinal in range(4):
            shared = matchmaker.assign(battle, ordinal, pool)
            own = plain.assign(battle, ordinal, plain_pool)
            assert shared.learner_seat == own.learner_seat
            assert shared.role == own.role
    opponents = {matchmaker.assign(0, ordinal, pool).opponent_id for ordinal in range(20)}
    assert len(opponents) > 5


def test_each_episodes_opponent_is_still_drawn_by_the_weights(tmp_path) -> None:
    """Over many episodes of one shard, every resident is met, about equally under uniform
    weights. Plant: a draw that ignored the weights' support would leave some never met."""
    matchmaker, pool, _shape = _setup(tmp_path, affinity=True)
    met = Counter(matchmaker.assign(0, ordinal, pool).opponent_id for ordinal in range(2500))
    assert len(met) == MEMBERS
    assert max(met.values()) < 2.0 * min(met.values())


def test_affinity_needs_the_plans_layout(tmp_path) -> None:
    config = LadderConfig(mix=(0.0, 1.0, 0.0), opponent_affinity=True)
    matchmaker = MixMatchmaker(SEED, config, n_battles=4)
    with pytest.raises(RuntimeError, match="plan the iteration"):
        matchmaker.assign(0, 0, _pool(tmp_path))
