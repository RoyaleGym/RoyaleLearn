"""The vector fields the network needs, resolved by name.

The pointer policy head reads three runs of the flat observation vector: the one-hot of the card
in each hand slot, each slot's cost, and whether each slot is affordable now. Their offsets are
not written down here and are not computed from the vector's width. They are looked up by name
in the layout the observation builder itself declares, so that a field added, removed or
reordered upstream moves them, a field the head needs and the builder no longer emits is a
refusal at start-up naming it, and the same code runs against a sixteen-card catalogue and a
sixty-five-card one without knowing which it has.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, NamedTuple

from .errors import PreflightError

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .api.rollout import EnvSpec

__all__ = [
    "HAND_AFFORDABLE",
    "HAND_CARD_ONEHOT",
    "HAND_COST",
    "REQUIRED_FIELDS",
    "HandFields",
    "field_slice",
    "hand_fields",
    "resolve_fields",
]

#: The three fields the pointer head needs, under the names ``royalegym.obs.vector_layout``
#: gives them.
HAND_CARD_ONEHOT = "own_hand_cards"
HAND_COST = "own_hand_cost"
HAND_AFFORDABLE = "own_hand_affordable"

REQUIRED_FIELDS: tuple[str, ...] = (HAND_CARD_ONEHOT, HAND_COST, HAND_AFFORDABLE)


def field_slice(spec: EnvSpec, name: str) -> slice:
    """The slice of the observation vector holding ``name``.

    A missing name is a ``PreflightError`` that lists what the layout does declare, because the
    failure it catches -- a renamed field -- is otherwise a silently wrong slice of somebody
    else's numbers.
    """
    for field_name, offset, size in spec.vector_layout:
        if field_name == name:
            return slice(offset, offset + size)
    known = ", ".join(field_name for field_name, _, _ in spec.vector_layout)
    raise PreflightError(
        f"the observation's vector layout has no field named {name!r}. It declares: {known}"
    )


def resolve_fields(spec: EnvSpec, names: Iterable[str] = REQUIRED_FIELDS) -> Mapping[str, slice]:
    """Every named field's slice, or a ``PreflightError`` naming the first one that is missing."""
    return {name: field_slice(spec, name) for name in names}


class HandFields(NamedTuple):
    """Where the hand lives in the observation vector, and the shape its one-hot unfolds to.

    ``card_onehot`` is ``hand_size`` consecutive one-hot blocks of ``onehot_width`` -- the
    catalogue plus one for an empty slot. The width is read off the field's own size divided by
    the hand size the action space declares, never from the vector's width or the card count, so
    a builder that widens the block says so by widening the field.
    """

    card_onehot: slice
    cost: slice
    affordable: slice
    hand_size: int
    onehot_width: int


def hand_fields(spec: EnvSpec) -> HandFields:
    """The three hand fields, checked against each other and against the action space."""
    resolved = resolve_fields(spec, REQUIRED_FIELDS)
    onehot = resolved[HAND_CARD_ONEHOT]
    cost = resolved[HAND_COST]
    affordable = resolved[HAND_AFFORDABLE]
    hand_size = spec.hand_size
    for name, span in ((HAND_COST, cost), (HAND_AFFORDABLE, affordable)):
        if span.stop - span.start != hand_size:
            raise PreflightError(
                f"vector field {name!r} is {span.stop - span.start} wide, and the action space "
                f"declares a hand of {hand_size}"
            )
    block = onehot.stop - onehot.start
    if hand_size < 1 or block % hand_size:
        raise PreflightError(
            f"vector field {HAND_CARD_ONEHOT!r} is {block} wide, which is not a whole number of "
            f"blocks for a hand of {hand_size}"
        )
    return HandFields(
        card_onehot=onehot,
        cost=cost,
        affordable=affordable,
        hand_size=hand_size,
        onehot_width=block // hand_size,
    )
