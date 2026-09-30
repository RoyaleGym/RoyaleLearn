"""The deal of ``ladder.seat_decks``: the learner's deck where the learner sits, and nowhere else.

A run that trains one deck against the decks it will meet wants every row it trains on to be
that deck's. RoyaleGym's deck curriculum deals per episode without knowing the seats, and the
matchmaker draws seats without knowing the decks, so on their own the two let a mirror battle
train a field deck on one seat and let a pool battle hand the learner the field deck.

A battle's role is the slot's for the whole run (``MixMatchmaker.role_of``), and with
``seat_decks`` set the learner's seat in a pool or scripted battle is fixed per battle
(``learner_seat_of``). So each battle's deal is one curriculum decided before the first reset:

* a mirror battle: the named deck on both seats, both of which the learner plays;
* a pool or scripted battle: the named deck on the learner's seat, and a field deck (drawn
  uniformly from ``field``, or a random deck when it is None) on the other, where a frozen or
  seeded snapshot or a scripted opponent sits and whose rows are not trained on.

The workers install one RoyaleGym ``DeckCurriculumStateMutator`` per battle; the env's own
state mutator is not used for those battles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..api.rollout import ROLE_MIRROR

if TYPE_CHECKING:
    from ..config import Geometry, RunConfig
    from ..rollout.envspec import ComponentSpec

#: The RoyaleGym class each battle's deal is.
CURRICULUM = "royalegym.state_mutator.DeckCurriculumStateMutator"
_SEAT_NAMES = ("blue", "red")


def learner_seat_of(battle: int) -> int:
    """The learner's seat in a pool or scripted battle under ``seat_decks``: blue in even
    battles, red in odd ones, so the seats are balanced across the rectangle."""
    return int(battle) % 2


def battle_state_mutators(config: RunConfig, geometry: Geometry) -> tuple[ComponentSpec, ...]:
    """One deal per battle of the rectangle, in battle order; empty without ``seat_decks``."""
    from ..rollout.envspec import ComponentSpec
    from .matchmaker import MixMatchmaker

    seats = config.ladder.seat_decks
    if seats is None:
        return ()
    roles = MixMatchmaker(
        config.master_seed,
        config.ladder,
        n_battles=geometry.n_battles,
        scripted_ids=config.ladder.scripted_opponents,
    )
    field = [list(deck) for deck in seats.field] if seats.field is not None else None
    deals = []
    for battle in range(geometry.n_battles):
        if roles.role_of(battle) == ROLE_MIRROR:
            kwargs = {"p": 1.0, "mirror_p": 1.0, "seat": "either", "pool": None}
        else:
            seat = _SEAT_NAMES[learner_seat_of(battle)]
            kwargs = {"p": 1.0, "mirror_p": 0.0, "seat": seat, "pool": field}
        deals.append(
            ComponentSpec(
                cls=CURRICULUM,
                kwargs={"deck": list(seats.deck), **kwargs, "shuffle": int(seats.shuffle)},
            )
        )
    return tuple(deals)
