"""Seat decks: when the learner's deck meets a field deck, only the learner's deck learns.

A run training one deck (``ladder.seat_decks.deck``) against the decks it will meet (``field``)
wants every row it trains on to be that deck's. RoyaleGym's deck curriculum deals decks per
episode without knowing who sits where, and the matchmaker draws the learner's seat without
knowing the decks, so a mirror battle -- the learner in both seats -- could deal the named deck
against a field deck and train the field deck on the other seat, and a pool battle could hand the
learner the field deck while a frozen player held the named one.

With ``seat_decks`` set:

* a mirror battle deals the named deck to both seats, so both seats train the named deck;
* a pool or scripted battle deals the named deck to the learner's seat and a field deck to the
  other, where a fixed player sits -- a frozen or seeded snapshot, or a scripted opponent -- whose
  rows are not trained on;
* the learner's seat in those battles is fixed per battle (even battles blue, odd red), so the
  seats stay balanced and the deal needs no message per episode; nothing else the matchmaker
  draws moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import config as cfg
from royalelearn.api.rollout import ROLE_MIRROR
from royalelearn.config import LadderConfig, RolloutConfig, RunConfig, geometry
from royalelearn.ladder.matchmaker import MixMatchmaker
from royalelearn.ladder.pool import LadderPool
from royalelearn.ladder.results import ResultLog
from royalelearn.testing import coordinator, tiny_config

SEED = 20260930
DECK = ("Giant", "Musketeer", "MiniPekka", "Archer", "Fireball", "Zap", "Goblins", "Minions")
FIELD = (
    ("Knight", "Valkyrie", "HogRider", "Cannon", "Log", "Skeletons", "Arrows", "GoblinBarrel"),
    ("Knight", "Archer", "Goblins", "Giant", "Musketeer", "Valkyrie", "Arrows", "Zap"),
)
SEATS = cfg.SeatDecks(deck=DECK, field=FIELD)


def _pool(tmp_path: Path) -> LadderPool:
    ladder = LadderPool(ResultLog(tmp_path / "games.jsonl"), context="ctx")
    for index in range(4):
        ladder.add(f"snap:{index}", step=index * 1000)
    return ladder


def _shape(mix: tuple[float, float, float]) -> Any:
    return geometry(
        RunConfig(
            rollout=RolloutConfig(workers=2, games_per_worker=8, shards_per_worker=1),
            ladder=LadderConfig(mix=mix),
        )
    )


def test_the_learner_takes_a_fixed_seat_per_battle_and_nothing_else_moves(tmp_path) -> None:
    mix = (0.25, 0.5, 0.25)
    shape = _shape(mix)
    pool = _pool(tmp_path)
    plain = MixMatchmaker(SEED, LadderConfig(mix=mix), n_battles=shape.n_battles)
    seated = MixMatchmaker(
        SEED, LadderConfig(mix=mix, seat_decks=SEATS), n_battles=shape.n_battles
    )
    seats = set()
    for battle in range(shape.n_battles):
        for ordinal in range(6):
            a = plain.assign(battle, ordinal, pool)
            b = seated.assign(battle, ordinal, pool)
            assert (a.role, a.opponent_id) == (b.role, b.opponent_id)
            if b.role == ROLE_MIRROR:
                assert b == a
                continue
            assert b.learner_seat == battle % 2
            assert b.group[b.learner_seat] == a.group[a.learner_seat]
            seats.add(b.learner_seat)
    assert seats == {0, 1}, "the fixed seats are not balanced"


def _deal(spec: Any, battles: int = 40) -> list[Any]:
    from royalegym.mock_engine import MockEngine

    cards = MockEngine().cards()
    mutator = spec.build()
    rng = np.random.default_rng(3)
    by = {c.name: c.card_id for c in cards}
    named = [by[n] for n in DECK]
    fields = [[by[n] for n in deck] for deck in FIELD]
    out = []
    for _ in range(battles):
        setup = mutator.build(rng, cards)
        out.append((setup.decks, named, fields))
    return out


def test_each_battle_deals_the_named_deck_where_the_learner_sits() -> None:
    from royalelearn.ladder.seat_decks import battle_state_mutators

    mix = (0.25, 0.5, 0.25)
    config = RunConfig(
        rollout=RolloutConfig(workers=2, games_per_worker=8, shards_per_worker=1),
        ladder=LadderConfig(mix=mix, seat_decks=SEATS),
    )
    shape = geometry(config)
    specs = battle_state_mutators(config, shape)
    assert len(specs) == shape.n_battles
    roles = MixMatchmaker(config.master_seed, config.ladder, n_battles=shape.n_battles)
    kinds = set()
    for battle, spec in enumerate(specs):
        mirror = roles.role_of(battle) == ROLE_MIRROR
        kinds.add(mirror)
        for decks, named, fields in _deal(spec):
            if mirror:
                assert decks == [named, named]
            else:
                seat = battle % 2
                assert decks[seat] == named
                assert decks[1 - seat] in fields
    assert kinds == {True, False}
    assert battle_state_mutators(RunConfig(), shape) == ()


def test_a_run_installs_the_deal_on_every_battle(tmp_path: Path) -> None:
    ladder = msgspec.structs.replace(
        tiny_config(tmp_path).ladder, mix=(0.5, 0.0, 0.5), seat_decks=SEATS
    )
    config = tiny_config(tmp_path, ladder=ladder)
    with coordinator(config) as run:
        run.iterate()
        installed = {}
        for runner in run.source._runners.values():
            for game, env in enumerate(runner.vec.envs):
                battle = int(runner.battles[game])
                installed[battle] = getattr(env, "parallel", env).state_mutator.config()
    assert installed, "no battle was inspected"
    for battle, got in installed.items():
        assert got["deck"] == list(DECK)
        if got["mirror_p"] == 1.0:
            continue
        assert got["seat"] == ("blue", "red")[battle % 2]
        assert got["pool"] == [list(d) for d in FIELD]
    assert {got["mirror_p"] for got in installed.values()} == {0.0, 1.0}


@pytest.mark.parametrize(
    ("seats", "message"),
    [
        (cfg.SeatDecks(deck=DECK[:7]), "8 cards"),
        (cfg.SeatDecks(deck=(*DECK[:7], DECK[0])), "twice"),
        (cfg.SeatDecks(deck=DECK, field=(FIELD[0][:7],)), "8 cards"),
    ],
)
def test_a_deck_that_is_not_eight_distinct_cards_is_refused(seats: Any, message: str) -> None:
    from royalelearn.errors import PreflightError

    config = RunConfig(ladder=LadderConfig(seat_decks=seats))
    with pytest.raises(PreflightError, match=message):
        cfg.validate(config)
