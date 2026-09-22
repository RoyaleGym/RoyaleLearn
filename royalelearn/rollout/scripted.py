"""The scripted opponents, played in the worker.

A scripted seat costs about five microseconds of numpy. Routing it through the parent would
cost a boundary crossing and a batched forward pass to compute an action that no network is
involved in, so the worker plays those seats itself and the parent is told which they were
rather than asked what to do about them.

They stay reproducible because each one draws from its own named stream,
``scripted/worker/{w}/slot/{r}/gen/{g}``: a slot's opponent is a pure function of the master
seed, the slot and the worker's respawn generation, so a restarted worker replays its
opponents from a fresh, recorded stream rather than from wherever the old one had got to.

royalegym is imported inside the functions that need it: this module is read by the parent to
resolve a name to an index, which must not cost an environment import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..api.rollout import GROUP_SCRIPTED

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .plan import SlotPlanner

__all__ = [
    "RANDOM_LEGAL_NOOP_PROB",
    "SCRIPTED_NAMES",
    "ScriptedSeats",
    "build_opponent",
    "scripted_id",
]

#: The scripted opponents a run may be assigned, in the order ``opponent_ix`` indexes them.
#: The order is part of the protocol -- the parent sends an index and the worker looks it up --
#: so a new opponent is appended rather than inserted.
SCRIPTED_NAMES: tuple[str, ...] = ("noop", "random_legal")

#: ``RandomLegalOpponent``'s share of no-ops. Uniform over the whole action space takes the
#: no-op essentially never and plays a card the instant one is affordable, which is a strange
#: thing to learn against; at nine in ten it plays at about a human's rate.
RANDOM_LEGAL_NOOP_PROB = 0.9


def scripted_id(name: str) -> str:
    """The opponent id a scripted seat is recorded under, in the ladder and in the results."""
    return f"scripted:{name}"


def build_opponent(name: str) -> Any:
    """One of ``SCRIPTED_NAMES`` as a ``royalegym.selfplay.Opponent``."""
    from royalegym.selfplay import NoopOpponent, RandomLegalOpponent

    if name == "noop":
        return NoopOpponent()
    if name == "random_legal":
        return RandomLegalOpponent(noop_prob=RANDOM_LEGAL_NOOP_PROB)
    raise KeyError(f"scripted opponent {name!r} is not one of {', '.join(SCRIPTED_NAMES)}")


class ScriptedSeats:
    """One shard's scripted opponents: the table of them, and a generator per slot.

    The generators are built for every slot of the shard whether or not it is ever scripted,
    because building them lazily would make a slot's stream depend on when it first drew from
    it, and the whole point of a named stream is that it does not.
    """

    def __init__(
        self,
        planner: SlotPlanner,
        worker: int,
        slots: np.ndarray,
        generation: int,
        names: tuple[str, ...] = SCRIPTED_NAMES,
    ) -> None:
        self.names = names
        self.slots = np.asarray(slots, dtype=np.int32)
        self.opponents = [build_opponent(name) for name in names]
        self.generators = [
            planner.scripted_generator(worker, int(slot), generation) for slot in self.slots
        ]

    def fill(
        self,
        actions: np.ndarray,
        obs: dict[str, np.ndarray],
        group: np.ndarray,
        opponent_ix: np.ndarray,
    ) -> int:
        """Replace the parent's action in every scripted seat, and return how many there were.

        The parent writes something into those entries -- it writes a whole row of actions --
        and what it writes there is not used: a scripted seat is the worker's to play. Rows are
        visited in ascending slot order so that the generators advance in an order that does
        not depend on anything but the shard's own layout.
        """
        rows = np.flatnonzero(group == GROUP_SCRIPTED)
        if rows.size == 0:
            return 0
        masks = obs["action_mask"]
        for row in rows:
            index = int(opponent_ix[row])
            if not 0 <= index < len(self.opponents):
                raise IndexError(
                    f"slot {int(self.slots[row])} was assigned scripted opponent {index}, and "
                    f"this worker knows {len(self.opponents)}: {', '.join(self.names)}"
                )
            one = {key: value[row] for key, value in obs.items()}
            actions[row] = self.opponents[index].act(
                one, masks[row], self.generators[int(row)]
            )
        return int(rows.size)
