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
from typing import Any, NamedTuple

import pytest

from royalegym.protocol import (
    DeployCommand,
    EntityKind,
    EntityState,
    MatchSetup,
    Placement,
    ShuffleMode,
    card_is_spell,
    slot_cost,
    to_engine,
)
from royalelearn.rewards import CommittedElixirPotential

TOWERS = (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
ENGINES = [
    "mock",
    pytest.param("rust", marks=pytest.mark.engine),
    # The catalogue runs actually train on. Which card an engine files a produced unit under
    # depends on which cards are loaded, so the default catalogue is not the population that
    # matters: on build 1ba01d7d the old rule read a Goblin Gang as 6 and Rascals as 15 on this
    # catalogue, where the review had measured a Goblin Gang at 18 on the build before.
    pytest.param("rust-training", marks=pytest.mark.engine),
]

#: Cards the catalogue test does not price, each graded by a strict xfail of its own. The Tri
#: Wizards: from RoyaleSim round 9 a tap puts the three wizards down over several ticks, none
#: of them within the two the catalogue test waits, and the Electro Wizard and Ice Wizard come
#: down under THEIR OWN card ids (42, 23), so once all three are down the term prices the play
#: at 7 + 4 + 3. It is an event-only card, and its fix waits with the other event-only cards.
PRICED_ELSEWHERE = frozenset({"TriWizards"})
#: How long the Tri Wizards test waits after the tap. On round 9 the last wizard is down 8
#: ticks after it.
TRI_WIZARDS_SETTLE_TICKS = 20


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


class Tap(NamedTuple):
    """One card tapped by one seat. ``put_down`` is None when no tap landed, and ``why`` says
    why; ``copied`` is the card a Mirror copied, None for every other card."""

    card: Any
    seat: int
    put_down: list[EntityState] | None
    copied: Any | None = None
    why: str = ""


def _is_mirror(card: Any) -> bool:
    return card.placement == Placement.MIRROR


def _tap_everything(engine: Any, ticks: int = 2):
    """Tap every card in the catalogue, each seat, and yield a ``Tap`` for each, landed or not.

    A TAP, not a hand-built board, because they do not put the same thing down: a tapped Goblin
    Gang is six units of two kinds, and the term has to price what a tap leaves.

    A card that is not in the opening hand is cycled in by playing another card first. The
    engine deals the Mirror and the Elixir Collector as the next card, never in the opening
    hand, and this loop used to skip a card it found outside the hand without saying so: on
    RoyaleSim 6ad6793 it priced 130 of the 132 default cards and passed. A card no tap lands
    for is yielded with ``put_down=None``, so a caller names it rather than never seeing it.
    The Mirror needs the cycle anyway, because it copies its side's last play; the cycled card
    is the cheapest unit card in the hand, so the copy puts something on the board.

    ``ticks`` is how long after the tap the board is read.
    """
    lockout = (
        int(getattr(engine.rules(), "deploy_lockout_ticks", 0)) if hasattr(engine, "rules") else 0
    )
    arena = engine.arena()
    by_id = {card.card_id: card for card in engine.cards()}
    ids = list(by_id)
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
            hand = list(engine.state().players[seat].hand)
            if card.card_id not in hand:
                others = [slot for slot, held in enumerate(hand) if held != card.card_id]
                units = [slot for slot in others if not card_is_spell(by_id[hand[slot]])]
                slot = min(units or others, key=lambda slot: by_id[hand[slot]].elixir)
                x, y = to_engine(arena, seat, 3 * arena.subtile, 6 * arena.subtile)
                status = engine.step([DeployCommand(team=seat, hand_slot=slot, x=x, y=y)], 2)
                if status[0].status != 0:
                    yield Tap(card, seat, None, why=f"cycle play refused: {status[0].status}")
                    continue
                hand = list(engine.state().players[seat].hand)
                if card.card_id not in hand:
                    yield Tap(card, seat, None, why="not in the hand after one cycle play")
                    continue
                # The cycle play spent from a bar that holds at most ten, and a Mirror of it
                # or an Elixir Collector can cost more than what is left. Wait for the bar to
                # hold what the engine says this slot costs.
                slot = hand.index(card.card_id)
                for _ in range(100):
                    player = engine.state().players[seat]
                    if player.elixir_milli >= 1000 * slot_cost(player, slot, card):
                        break
                    engine.step([], 20)
            before = engine.state()
            hand = list(before.players[seat].hand)
            copied = None
            if _is_mirror(card):
                copied = by_id.get(before.players[seat].mirror_target)
            existing = {entity.uid for entity in before.entities}
            x, y = to_engine(arena, seat, 9 * arena.subtile, 6 * arena.subtile)
            command = DeployCommand(team=seat, hand_slot=hand.index(card.card_id), x=x, y=y)
            status = engine.step([command], ticks)[0].status
            if status != 0:
                yield Tap(card, seat, None, copied, why=f"tap refused: {status}")
                continue
            put_down = [
                entity
                for entity in engine.state().entities
                if entity.uid not in existing and entity.kind not in TOWERS
            ]
            yield Tap(card, seat, put_down, copied)


def _price(card: Any) -> Fraction:
    # A spell puts nothing of its own on the board. Decided by WHAT the card is, its kind, and
    # not by where it may be played: since RoyaleSim 95698c5 Heal, a spell placed by a troop's
    # rule, has a troop's placement and is still a spell. Taken from RoyaleGym rather than from
    # the term under test.
    return Fraction(0) if card_is_spell(card) else Fraction(card.elixir)


@pytest.mark.parametrize("kind", ENGINES)
def test_one_tap_of_every_card_puts_exactly_its_elixir_on_the_board(kind: str) -> None:
    """Unit cards: exactly the card's elixir. Spells: nothing, because the bar already paid.

    Every card in the catalogue, on both seats, and a card no tap landed for fails by name. A
    card whose play priced at anything else would over- or under-pay every play of it for the
    rest of training. The Mirror is priced by the test after this one, because what it puts
    down is another card's, and the cards in ``PRICED_ELSEWHERE`` by their own tests.
    """
    engine = _engine(kind)
    term = _term(engine)
    taps = list(_tap_everything(engine))
    missed = [(t.card.name, t.seat, t.why) for t in taps if t.put_down is None]
    assert missed == [], f"cards whose price was never measured, because no tap landed: {missed}"
    off = []
    for t in taps:
        if _is_mirror(t.card) or t.card.name in PRICED_ELSEWHERE:
            continue
        on_board = sum((term.unit_value(entity) for entity in t.put_down), Fraction(0))
        if on_board != _price(t.card):
            off.append((t.card.name, t.seat, len(t.put_down), str(on_board), str(_price(t.card))))
    assert off == [], f"plays priced at something other than the card's elixir: {off}"
    assert len(taps) >= 20, f"only {len(taps)} taps landed, so the check would be vacuous"


class MirrorCopyMispriced(AssertionError):
    """The one failure the Mirror test below is expected to raise, and no other."""


@pytest.mark.engine
@pytest.mark.xfail(
    strict=True,
    raises=MirrorCopyMispriced,
    reason=(
        "unit_value prices a Mirror's copy at 0: the copy is one level above its card, so its "
        "hitpoints miss the card's row, and the engine does not report a unit's level yet"
    ),
)
def test_a_mirror_play_puts_the_copied_cards_elixir_on_the_board() -> None:
    """A Mirror play puts down a copy of its side's last play, worth what that card is worth.

    The bar pays the copied card's elixir plus the Mirror's own one. That extra elixir buys the
    copy a level, which this term does not price, so a Mirror play loses exactly one elixir of
    potential. On RoyaleSim 6ad6793 it loses all of it instead: a mirrored Knight has 1938
    hitpoints against the Knight row's 1766, fails the row match, and is priced at 0.

    Expected to fail, and only with ``MirrorCopyMispriced``: a catalogue with no Mirror, a tap
    that did not land or a copy of a spell fails as an ordinary error. When the pricing is
    fixed this passes, and ``strict`` turns that into a failure until the mark comes off.
    """
    engine = _engine("rust")
    term = _term(engine)
    taps = [t for t in _tap_everything(engine) if _is_mirror(t.card)]
    assert taps, "no Mirror in the default catalogue, so there is nothing to price"
    assert all(t.put_down for t in taps), [(t.seat, t.why) for t in taps]
    assert all(t.copied is not None and not card_is_spell(t.copied) for t in taps), [
        (t.seat, t.copied) for t in taps
    ]
    off = []
    for t in taps:
        on_board = sum((term.unit_value(entity) for entity in t.put_down), Fraction(0))
        expected = _price(t.copied)
        if on_board != expected:
            off.append((t.seat, t.copied.name, len(t.put_down), str(on_board), str(expected)))
    if off:
        raise MirrorCopyMispriced(f"Mirror copies priced at something other than their card: {off}")


class TriWizardsMispriced(AssertionError):
    """The one failure the Tri Wizards test below is expected to raise, and no other."""


@pytest.mark.parametrize("kind", ENGINES[1:])
@pytest.mark.xfail(
    strict=True,
    raises=TriWizardsMispriced,
    reason=(
        "RoyaleSim puts the Tri Wizards' Electro and Ice Wizards down under their own card ids "
        "(42, 23), so the term prices the play 14; stamping the played card's id waits with the "
        "other event-only cards"
    ),
)
def test_a_tri_wizards_play_puts_exactly_its_elixir_on_the_board(kind: str) -> None:
    """The Tri Wizards, priced once all three wizards are on the board.

    The catalogue test reads the board two ticks after a tap and the Tri Wizards arrive later,
    so this test waits ``TRI_WIZARDS_SETTLE_TICKS``. It also requires all three: with only the
    first wizard down the board prices at exactly 7 and would pass for the wrong reason.

    Expected to fail, and only with ``TriWizardsMispriced``: a catalogue with no Tri Wizards, a
    tap that did not land or a board without three wizards fails as an ordinary error. When
    the engine stamps the card's id on all three this passes, ``strict`` turns that into a
    failure, and the card comes out of ``PRICED_ELSEWHERE`` with the mark.
    """
    engine = _engine(kind)
    term = _term(engine)
    taps = [
        t
        for t in _tap_everything(engine, ticks=TRI_WIZARDS_SETTLE_TICKS)
        if t.card.name == "TriWizards"
    ]
    assert taps, "no Tri Wizards in this catalogue, so there is nothing to price"
    assert all(t.put_down is not None and len(t.put_down) == 3 for t in taps), [
        (t.seat, t.why, None if t.put_down is None else len(t.put_down)) for t in taps
    ]
    off = []
    for t in taps:
        on_board = sum((term.unit_value(entity) for entity in t.put_down), Fraction(0))
        if on_board != _price(t.card):
            off.append((t.seat, str(on_board), str(_price(t.card))))
    if off:
        raise TriWizardsMispriced(f"Tri Wizards plays priced at something other than 7: {off}")


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
    for t in _tap_everything(engine):
        if t.card.name == "GoblinGang" and t.seat == 0 and t.put_down is not None:
            seen[t.card.name] = sum((term.unit_value(e) for e in t.put_down), Fraction(0))
    assert seen.get("GoblinGang") == Fraction(by_name["GoblinGang"].elixir), seen
