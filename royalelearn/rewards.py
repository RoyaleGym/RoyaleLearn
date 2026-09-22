"""The reward the harness trains against: one terminal objective and three potentials.

Every shaping term here is a difference of a potential, ``gamma * Phi(s') - Phi(s)``. That form
is what keeps shaping from changing which policy is optimal (Ng, Harada and Russell, 1999), and
RoyaleGym's own ``reward.py`` states the principle in its header. The terminal win/loss term is
the only term that is the objective; everything else densifies the signal and must leave the
optimum where it was.

The discount in that expression is the run's, not a constant: ``set_gamma`` is called once per
iteration with the value the schedule is at, and the same number rides the ``Step`` command to
every worker, so the discount the reward is computed with and the discount the learner
bootstraps with are one number rather than two that agree at the start of a run. It is
deliberately not in ``config()``: ``config()`` feeds the run identity and the ladder's context,
and a quantity that moves every iteration does not belong in either.

WHY THESE THREE AND NOT ROYALEGYM'S SHIPPED DEFAULTS

- ``TowerHPReward`` computes ``Phi(s') - Phi(s)``. It is one ``gamma`` away from being exactly
  policy-invariant, and with the gamma in place the term never needs annealing.
- ``ElixirTradeReward`` is not a potential and it rewards turtling: a player who never plays a
  card never incurs the negative half, while enemy units still die to its towers and earn the
  positive half. At a weight that makes it visible it is comparable to the terminal reward.
- ``ElixirLeakPenalty`` is not zero-sum -- both players can leak at once -- which is the
  signature of a term standing in for a potential that is missing.

``CommittedElixirPotential`` replaces both. Playing a card moves elixir from the bar to the
board and is net zero; losing a unit costs what the unit cost; killing one gains it; and sitting
at ten elixir is penalised on its own, because the opponent's side of the potential keeps rising
while yours cannot. There is no coefficient to re-tune and no annealing schedule, which is what
RoyaleGym's house rule -- weights should settle, not drift -- asks for.

EXACT ARITHMETIC

Every potential is a ``Fraction`` and the discounting is done on fractions, with one conversion
to float at the end. The seat-mirror property is the reason: ``Phi(s, red)`` is exactly
``-Phi(s, blue)`` as a rational, so the two seats' rewards are exact negatives of each other and
"the reward is zero-sum" is a property a test can assert rather than a tolerance it has to
choose. A running float sum of the same unit values returns plus and minus 1.1e-19 on a perfect
mirror -- harmless to a gradient, and enough to make the property uncheckable.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from fractions import Fraction

from royalegym.protocol import (
    BattleState,
    CardInfo,
    DeployResult,
    Engine,
    EntityKind,
    TowerSlot,
)
from royalegym.reward import CombinedReward, RewardFunction, WinLossReward

from .metrics.records import TERMINAL_REWARD_TERM

__all__ = [
    "CROWNS",
    "CommittedElixirPotential",
    "PotentialCombinedReward",
    "PotentialCrownReward",
    "PotentialReward",
    "PotentialTowerHPReward",
    "default_potential_reward",
    "set_gamma",
]

#: Crowns in a match: one per tower, so a crown difference of this is the whole of the board.
#: Read off the tower enumeration rather than written down, because it is the same three.
CROWNS = len(TowerSlot)

#: Elixir is carried as thousandths.
ELIXIR_MILLI = 1000

#: The entity kinds a player's committed elixir does not count. A crown tower was never played
#: out of a hand, and what happens to one is already the tower-HP potential's subject.
TOWER_KINDS = (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)


class PotentialReward(RewardFunction):
    """``gamma * Phi(s') - Phi(s)`` for a potential a subclass computes from one state.

    The subclass returns a ``Fraction`` from the seat's own point of view, antisymmetric between
    the two seats, and this class does the discounting and the single conversion to float.
    """

    def __init__(self) -> None:
        self.gamma = 1.0
        self._gamma = Fraction(1)

    def set_gamma(self, gamma: float) -> None:
        """Take the discount the schedule is at. Exact: a binary float is a rational."""
        self.gamma = float(gamma)
        self._gamma = Fraction(self.gamma)

    @abstractmethod
    def potential(self, state: BattleState, team: int) -> Fraction:
        """``Phi(s)`` from ``team``'s point of view, exact and antisymmetric in the seat."""

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        return float(self._gamma * self.potential(state, team) - self.potential(prev, team))


class PotentialCrownReward(PotentialReward):
    """``Phi = (own crowns - enemy crowns) / 3``.

    The crown difference is the match's own scoreboard, so the potential is the scoreboard read
    as a fraction of a whole win. It is the coarsest of the three and the one that most directly
    anticipates the terminal term.
    """

    def potential(self, state: BattleState, team: int) -> Fraction:
        own = state.players[team].crowns
        foe = state.players[1 - team].crowns
        return Fraction(own - foe, CROWNS)


class PotentialTowerHPReward(PotentialReward):
    """``Phi = (sum own_hp/own_max - sum foe_hp/foe_max) / 3``.

    The same scoreboard as the crowns, read continuously: a tower at a third of its health is a
    third of the way to the crown it will give up. Dividing by the tower count puts it on the
    crown potential's scale, so the two weights mean comparable things.
    """

    def potential(self, state: BattleState, team: int) -> Fraction:
        return Fraction(self._towers(state, team) - self._towers(state, 1 - team), CROWNS)

    @staticmethod
    def _towers(state: BattleState, team: int) -> Fraction:
        player = state.players[team]
        return sum(
            (
                Fraction(player.tower_hp[slot], max(1, player.tower_max_hp[slot]))
                for slot in TowerSlot
            ),
            Fraction(0),
        )


class CommittedElixirPotential(PotentialReward):
    """``Phi = ((own bar + own board) - (foe bar + foe board)) / scale``.

    ``bar`` is the elixir a player is holding and ``board`` is the elixir value of what that
    player has on the field, a unit at ``Fraction(card.elixir, card.count)`` of the card that
    summoned it. Crown towers are excluded: they were never played, and the tower potential
    already owns what happens to them.

    What the term says, in the four cases that matter: a card played moves elixir from the bar
    to the board and is worth nothing; a unit lost costs what it cost; a unit killed gains it;
    and holding a full bar loses ground, because the opponent's side keeps rising while yours
    cannot. The last is the one that replaces a leak penalty, and unlike a leak penalty it is
    zero-sum.
    """

    def __init__(self, scale: float = 10.0) -> None:
        super().__init__()
        self.scale = scale
        self._scale = Fraction(scale)
        self.value: dict[int, Fraction] = {}

    def config(self) -> dict[str, object]:
        return {"scale": self.scale}

    def bind(self, engine: Engine) -> None:
        cards: Sequence[CardInfo] = engine.cards()
        self.value = {card.card_id: Fraction(card.elixir, max(1, card.count)) for card in cards}

    def potential(self, state: BattleState, team: int) -> Fraction:
        return (self._committed(state, team) - self._committed(state, 1 - team)) / self._scale

    def _committed(self, state: BattleState, team: int) -> Fraction:
        total = Fraction(state.players[team].elixir_milli, ELIXIR_MILLI)
        for entity in state.entities:
            if entity.team != team or entity.kind in TOWER_KINDS:
                continue
            total += self.value.get(entity.card_id, Fraction(0))
        return total


class PotentialCombinedReward(CombinedReward):
    """RoyaleGym's weighted sum, with the run's discount threaded into the terms that take one.

    The worker calls ``set_gamma`` on whatever reward function its environment holds, once per
    iteration, so the composition has to be the thing that forwards it. A term that does not
    take a discount -- the terminal one -- is left alone.

    It also says WHICH of its terms is the objective. ``CombinedReward`` files each term in the
    logged breakdown under its class name, and the metrics group has to tell the objective from
    the shaping to publish either one: ``shaping_dominates`` is the alarm that watches for the
    shaping taking the objective over, and it compares those two sums. Matching on a class name
    would put a rename of a class into the arithmetic of an alarm, so the composition names the
    objective instead, and the breakdown carries it under ``TERMINAL_REWARD_TERM``.
    """

    def __init__(
        self,
        terms: Sequence[tuple[RewardFunction, float]],
        *,
        terminal: int = 0,
    ) -> None:
        super().__init__(terms)
        if terms and not 0 <= terminal < len(terms):
            raise ValueError(
                f"terminal is term {terminal} of a composition with {len(terms)} terms"
            )
        self.terminal = int(terminal)

    @property
    def terminal_class(self) -> str | None:
        """The class name the objective's term would otherwise be filed under."""
        return type(self.terms[self.terminal][0]).__name__ if self.terms else None

    def set_gamma(self, gamma: float) -> None:
        set_gamma(self, gamma)

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        total = super().get_reward(team, prev, state, results)
        breakdown = self.last_terms.get(team)
        name = self.terminal_class
        if breakdown is not None and name is not None and name in breakdown:
            breakdown[TERMINAL_REWARD_TERM] = breakdown.pop(name)
        return total


def set_gamma(reward: RewardFunction, gamma: float) -> None:
    """Give ``gamma`` to every term of ``reward`` that takes one, however it is composed.

    A free function as well as a method because a composition is a tree: a ``CombinedReward``
    may hold another one, and the discount has to reach the leaves of whatever a bot creator
    assembled rather than only the terms this module shipped.
    """
    terms = getattr(reward, "terms", None)
    if terms is not None:
        for term, _weight in terms:
            set_gamma(term, gamma)
        return
    setter = getattr(reward, "set_gamma", None)
    if setter is not None:
        setter(gamma)


def default_potential_reward() -> CombinedReward:
    """The shipped composition: the objective, and three potentials under it.

    The weights are an order of magnitude apart on purpose. The terminal term is the objective
    at 1.0; the crown potential anticipates it; the tower potential is the same scoreboard read
    continuously; and the elixir potential is the fastest-moving of the three and therefore the
    smallest, because its job is to make the first hour of a run legible rather than to be
    optimised. The ``shaping_dominates`` alarm watches the sum of the shaping terms against the
    terminal one for exactly this reason.
    """
    return PotentialCombinedReward(
        [
            (WinLossReward(draw=0.0), 1.0),
            (PotentialCrownReward(), 0.2),
            (PotentialTowerHPReward(), 0.1),
            (CommittedElixirPotential(scale=10.0), 0.05),
        ]
    )
