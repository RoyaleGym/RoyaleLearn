"""One play of a unit card puts exactly that card's elixir on the board -- measured, not believed.

``CommittedElixirPotential`` priced every entity by the card the engine files it under. An engine
files a produced unit under SOME card that can produce it, not necessarily the one its owner
played, so on RustEngine a Goblin Gang's spear goblins came in under the Goblin Hut at five elixir
each, a dying Golem's golemites under the Golem at eight, a dying Battle Ram's barbarians at four.
Measured by the train session's review on 2026-09-24: +15 elixir of potential per Goblin Gang
play, +8 at a Golem's death, +4 at a Battle Ram's. Under every run to date.

``test_rewards.test_playing_a_card_is_worth_exactly_nothing`` could not see it. It builds a board
by hand on which every unit carries its own card's id, and its fixture leaves spells out. So this
file does what that test could not: it TAPS every card each engine will place, on both seats, and
prices what the tap actually left on the board.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from royalegym.protocol import (
    DeployCommand,
    EntityKind,
    EntityState,
    MatchSetup,
    Placement,
    ShuffleMode,
    to_engine,
)
from royalelearn.rewards import CommittedElixirPotential

TOWERS = (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
#: The catalogue's own ways of saying "this card puts nothing of its own on the board", spelled
#: out here from the engine's placement classes rather than taken from the term under test.
SPELL_PLACEMENTS = (Placement.SPELL, Placement.ROLLING, Placement.SPELL_NOT_ON_WATER)
ENGINES = [
    "mock",
    pytest.param("rust", marks=pytest.mark.engine),
    # The catalogue runs actually train on. Which card an engine files a produced unit under
    # depends on which cards are loaded, so the default catalogue is not the population that
    # matters: on build 1ba01d7d the old rule read a Goblin Gang as 6 and Rascals as 15 on this
    # catalogue, where the review had measured a Goblin Gang at 18 on the build before.
    pytest.param("rust-training", marks=pytest.mark.engine),
]


def _engine(kind: str) -> Any:
    if kind.startswith("rust"):
        from royalegym.rust_engine import RustEngine

        if kind == "rust-training":
            import json
            from pathlib import Path

            config = Path(__file__).parents[1] / "examples" / "configs" / "train-hog26-10.json"
            names = json.loads(config.read_text(encoding="utf-8"))["env"]["engine"]["kwargs"]
            return RustEngine(card_names=names["card_names"])
        return RustEngine()
    from royalegym.mock_engine import MockEngine

    return MockEngine()


def _term(engine: Any) -> CommittedElixirPotential:
    term = CommittedElixirPotential()
    term.bind(engine)
    return term


def _tap_everything(engine: Any):
    """Tap every card the engine will accept, each seat; yield (card, seat, units it put down).

    A TAP, not a hand-built board, because they do not put the same thing down: a tapped Goblin
    Gang is six units of two kinds, and the term has to price what a tap leaves.
    """
    lockout = (
        int(getattr(engine.rules(), "deploy_lockout_ticks", 0)) if hasattr(engine, "rules") else 0
    )
    arena = engine.arena()
    ids = [card.card_id for card in engine.cards()]
    for card in engine.cards():
        deck = [card.card_id, *[i for i in ids if i != card.card_id][:7]]
        for seat in (0, 1):
            engine.reset(
                1,
                MatchSetup(
                    decks=[deck, deck],
                    shuffle=ShuffleMode.NONE,
                    elixir_milli=[10**7] * 2,
                    start_tick=lockout,
                ),
            )
            before = engine.state()
            existing = {entity.uid for entity in before.entities}
            hand = before.players[seat].hand
            if card.card_id not in hand:
                continue
            x, y = to_engine(arena, seat, 9 * arena.subtile, 6 * arena.subtile)
            command = DeployCommand(team=seat, hand_slot=hand.index(card.card_id), x=x, y=y)
            if engine.step([command], 2)[0].status != 0:
                continue
            yield (
                card,
                seat,
                [
                    entity
                    for entity in engine.state().entities
                    if entity.uid not in existing and entity.kind not in TOWERS
                ],
            )


@pytest.mark.parametrize("kind", ENGINES)
def test_one_tap_of_every_card_puts_exactly_its_elixir_on_the_board(kind: str) -> None:
    """Unit cards: exactly the card's elixir. Spells: nothing, because the bar already paid.

    Every card the engine will place, on both seats. A card whose play priced at anything else
    would over- or under-pay every play of it for the rest of training.
    """
    engine = _engine(kind)
    term = _term(engine)
    taps, off = 0, []
    for card, seat, put_down in _tap_everything(engine):
        taps += 1
        on_board = sum((term.unit_value(entity) for entity in put_down), Fraction(0))
        expected = Fraction(0) if card.placement in SPELL_PLACEMENTS else Fraction(card.elixir)
        if on_board != expected:
            off.append((card.name, seat, len(put_down), str(on_board), str(expected)))
    assert off == [], f"plays priced at something other than the card's elixir: {off}"
    assert taps >= 20, f"only {taps} taps landed, so the check would be vacuous"


def _entity(card: Any, **stats: Any) -> EntityState:
    base = {
        "max_hp": card.hitpoints,
        "radius": card.radius,
        "flying": bool(card.flying),
    }
    base.update(stats)
    return EntityState(
        uid=1,
        team=0,
        kind=EntityKind.TROOP,
        card_id=card.card_id,
        tower_slot=-1,
        x=0,
        y=0,
        hp=base["max_hp"],
        max_hp=base["max_hp"],
        radius=base["radius"],
        flying=base["flying"],
        deploy_ticks=0,
    )


def test_a_unit_filed_under_a_card_it_is_not_is_priced_zero() -> None:
    """A spear goblin under the Goblin Hut, a golemite under the Golem: filed there, not theirs.

    Each stat on its own is enough to disqualify, so a rule that compared only one of them fails
    one of these three.
    """
    from royalegym.mock_engine import MockEngine

    engine = MockEngine()
    term = _term(engine)
    card = next(c for c in engine.cards() if c.count >= 1 and c.hitpoints > 0)
    own = Fraction(card.elixir, max(1, card.count))
    assert term.unit_value(_entity(card)) == own
    assert term.unit_value(_entity(card, max_hp=card.hitpoints + 1)) == 0
    assert term.unit_value(_entity(card, radius=card.radius + 1)) == 0
    assert term.unit_value(_entity(card, flying=not card.flying)) == 0


@pytest.mark.engine
def test_a_goblin_gang_tap_reads_three_on_the_real_engine() -> None:
    """The review's Goblin Gang case by name, so a regression names it rather than a total.

    A Goblin Gang play read eighteen elixir; it must read three. The Golem and Battle Ram cases
    happen at a DEATH, which this file does not simulate: what it holds for them is the rule, in
    the test above, that a unit which is not its card's own is priced zero. Whether the engine's
    golemites really fail that match is not checked here.
    """
    engine = _engine("rust")
    term = _term(engine)
    by_name = {card.name: card for card in engine.cards()}
    seen = {}
    for card, seat, put_down in _tap_everything(engine):
        if card.name == "GoblinGang" and seat == 0:
            seen[card.name] = sum((term.unit_value(e) for e in put_down), Fraction(0))
    assert seen.get("GoblinGang") == Fraction(by_name["GoblinGang"].elixir), seen
