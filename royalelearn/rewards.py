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
while yours cannot. It replaces the leak penalty's coefficient with a property of the state,
and it needs no annealing schedule, which is what RoyaleGym's house rule -- weights should
settle, not drift -- asks for. Its weight in the composition is configurable (see
``default_potential_reward``); that is a scale, set once for a run, not a schedule.

EXACT ARITHMETIC

Every potential is a ``Fraction`` and the discounting is done on fractions, with one conversion
to float at the end. The seat-mirror property is the reason: ``Phi(s, red)`` is exactly
``-Phi(s, blue)`` as a rational, so the two seats' rewards are exact negatives of each other and
"the reward is zero-sum" is a property a test can assert rather than a tolerance it has to
choose. A running float sum of the same unit values returns plus and minus 1.1e-19 on a perfect
mirror -- harmless to a gradient, and enough to make the property uncheckable.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from fractions import Fraction

from royalegym.protocol import (
    BattleState,
    CardInfo,
    DeployResult,
    Engine,
    EntityKind,
    EntityState,
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
        """``gamma * Phi(s') - Phi(s)``, with ``Phi`` of a finished battle taken to be zero.

        The zero is taken rather than computed, and it is what makes the shaping harmless. Summed
        over an episode the term telescopes to ``gamma^T * Phi(s_T) - Phi(s_0)``; ``Phi(s_0)`` is
        the same whatever the policy does, so if ``Phi(s_T)`` is zero as well the shaping adds a
        constant and cannot move the optimum (Ng, Harada and Russell, 1999). Read off the final
        state instead, ``Phi(s_T)`` is the margin -- a 3-0 win has three times the crown potential
        of a 1-0 win -- and the shaping would quietly pay for the margin beside the objective,
        which is the one thing these terms were chosen not to do.

        A truncation is the other case and it is not this one: ``game_over`` is the engine saying
        the battle was decided, while a step limit cuts a battle that still has a position worth
        something, and the estimator bootstraps from that position's value.
        """
        after = Fraction(0) if state.game_over else self.potential(state, team)
        return float(self._gamma * after - self.potential(prev, team))


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

    WHICH UNITS ARE PRICED, and why this is not every unit on the board. An engine reports each
    entity under SOME card that can produce it, which need not be the card its owner played: a
    Goblin Gang's spear goblins are filed under the Goblin Hut, at five elixir each; a dying
    Golem's golemites are filed under the Golem, at eight; a dying Battle Ram's barbarians under
    the Battle Ram, at four. Priced by the card they are filed under, measured on RustEngine
    2026-09-24 by the train session's review: one Goblin Gang play read +15 elixir of potential,
    a Golem's death +8, a Battle Ram's +4 -- under every run to date, at every weight. So a unit
    is priced only when it IS the unit its card's catalogue row describes: same hitpoints, same
    collision radius, same air or ground. Anything else a card produced scores zero. That is an
    understatement and it is deliberate, the rule RoyaleGym's ``ElixirTradeReward`` adopted in
    4488a2f: a golemite is worth something, and this term says zero rather than eight. What it
    keeps true is the property the term needs. ONE PLAY OF A UNIT CARD PUTS EXACTLY THAT CARD'S
    ELIXIR ON THE BOARD: a card's summon count covers exactly the units its row describes, so a
    Goblin Gang still totals three. ``tests/test_elixir_pricing.py`` taps every card each engine
    will place, on both seats, because the rule rests on that.

    SPELLS ARE CHARGED AT THE TAP, through the bar, by construction: the bar drops by the spell's
    cost and nothing it leaves on the board is priced, so the elixir comes back only through what
    the spell kills. That is the true trade, and it is what ``ElixirTradeReward`` charges too. A
    spell that leaves units behind -- a barrel -- is charged the same way, and its units score
    zero by the rule above. It telescopes like everything else here, so it moves no optimum.
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
        #: The unit each card's row describes. An entity filed under a card is that card's own
        #: unit only if it matches, and only then is it priced.
        self.own_unit: dict[int, tuple[int, int, bool]] = {
            card.card_id: (card.hitpoints, card.radius, bool(card.flying)) for card in cards
        }

    def unit_value(self, entity: EntityState) -> Fraction:
        """What one entity on the board is worth: its card's per-unit elixir, or zero.

        Zero for anything that is not the unit its card's row describes -- a spear goblin filed
        under the Goblin Hut, a golemite under the Golem -- because pricing it by the card it is
        filed under is how a three-elixir play came to read as eighteen.
        """
        row = self.own_unit.get(entity.card_id)
        if row is None or row != (entity.max_hp, entity.radius, bool(entity.flying)):
            return Fraction(0)
        return self.value[entity.card_id]

    def potential(self, state: BattleState, team: int) -> Fraction:
        return (self._committed(state, team) - self._committed(state, 1 - team)) / self._scale

    def _committed(self, state: BattleState, team: int) -> Fraction:
        """The bar plus the board. What it cannot see is a spell in FLIGHT, and that is fine.

        ``BattleState.spells`` is a separate list from ``entities`` -- a spell between its cast
        and its effect has no entity and stands on no tile -- so for those few ticks its elixir
        is in neither the bar nor this sum, and the side that cast it reads as having lost that
        much ground. It telescopes: potential-based shaping pays ``gamma*Phi(s') - Phi(s)``, so
        a dip and its recovery cancel to within one discount factor, and once the spell resolves
        the elixir really is spent and really is worth nothing, which is the steady state this
        term already reports.

        This paragraph used to say that a card which RELEASES units "is not affected at all",
        because what it leaves behind is filed under its own catalogue id. Filed there, yes; its
        own unit, no -- and pricing by the filing is what made a Goblin Gang read as eighteen
        elixir and a dying Golem as a gain. See the class docstring; ``unit_value`` is the fix.
        """
        total = Fraction(state.players[team].elixir_milli, ELIXIR_MILLI)
        for entity in state.entities:
            if entity.team != team or entity.kind in TOWER_KINDS:
                continue
            total += self.unit_value(entity)
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


#: The shaping weights ``default_potential_reward`` takes, by name.
SHAPING_WEIGHTS: tuple[str, ...] = ("crown", "tower_hp", "elixir")


def shaping_weight_problems(weights: Mapping[str, object]) -> list[str]:
    """Everything wrong with a set of shaping weights, every one named, all at once.

    Shared by ``default_potential_reward`` and ``config.check_consistency`` so that a config is
    refused at load for exactly what the reward would refuse at build.
    """
    problems: list[str] = []
    for name, weight in weights.items():
        if name not in SHAPING_WEIGHTS:
            problems.append(
                f"{name!r} is not a shaping weight; the weights are {', '.join(SHAPING_WEIGHTS)}"
            )
        elif isinstance(weight, bool) or not isinstance(weight, int | float):
            problems.append(f"{name}={weight!r} is not a number")
        elif not math.isfinite(weight) or weight < 0:
            problems.append(f"{name}={weight!r} must be finite and zero or more")
    return problems


def default_potential_reward(
    *,
    crown: float = 0.2,
    tower_hp: float = 0.1,
    elixir: float = 0.05,
) -> CombinedReward:
    """The shipped composition: the objective, and three potentials under it.

    The terminal term is the objective at 1.0; the crown potential anticipates it; the tower
    potential is the same scoreboard read continuously; and the elixir potential is the
    fastest-moving of the three.

    The three shaping weights are keyword arguments, so a config can set them::

        "reward_fn": {"cls": "royalelearn.rewards.default_potential_reward",
                      "kwargs": {"crown": 0.2, "tower_hp": 0.1, "elixir": 0.05}}

    Every term is a potential difference, so no weight can change which policy is optimal: a
    weight SCALES a term's per-step magnitude and changes nothing about when it arrives. That holds
    for any real weight, negative ones included, so it neither justifies raising a weight nor
    bounds it. The bound here is a choice, stated below. What a weight does change is how loud
    each term is step to step, and that is what to read before changing one:
    ``env/reward_terms_step_abs/<term>``, the mean of each seat's ``sum |F_t|`` over an episode.
    NOT ``env/reward_terms_abs/<term>``, which is the episode's SUM: for a potential it telescopes
    to ``1 - gamma`` times how far the potential wandered, so it falls as the discount schedule
    rises whatever the weights are.

    How loud the shipped weights are, measured 2026-09-24 on train-hog26-10's environment, ten
    random-legal battles at gamma 0.999, per seat per episode **[M]**: elixir 0.80, tower 0.15,
    crown 0.15, against a terminal of 0.80. The elixir term alone is already about as loud as the
    objective. Each magnitude is linear in its weight.

    A weight must be a real number, finite and not negative: not a bool, which Python would
    quietly take as 1, and not a string, which constructs and then fails inside a worker on the
    first step. A negative potential weight pays a seat for losing ground, and a NaN reaches every
    return it touches; both would train, silently. ``config.check_consistency`` applies the same
    rule when a config is loaded, so a bad weight is named before any environment is built.
    """
    problems = shaping_weight_problems({"crown": crown, "tower_hp": tower_hp, "elixir": elixir})
    if problems:
        raise ValueError("default_potential_reward: " + "; ".join(problems))
    return PotentialCombinedReward(
        [
            (WinLossReward(draw=0.0), 1.0),
            (PotentialCrownReward(), float(crown)),
            (PotentialTowerHPReward(), float(tower_hp)),
            (CommittedElixirPotential(scale=10.0), float(elixir)),
        ]
    )
