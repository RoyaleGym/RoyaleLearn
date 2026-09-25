"""The reward composition: the shaping is a potential, and the potential is exactly zero-sum.

Every assertion here is on hand-built transitions rather than on played ones. A played battle
gives a transition whose potential nobody knows independently, so the test would be comparing
the implementation with itself; a state written out by hand has a potential that can be worked
out on paper, which is the only way "this term is ``gamma*Phi(s') - Phi(s)``" is checkable.

The equalities are exact. That is the point of the ``Fraction`` arithmetic in ``rewards.py``: a
running float sum of the same unit values returns plus and minus a hundredth of an attojoule's
worth of elixir on a perfect mirror, which is harmless to a gradient and enough to turn "the
reward is zero-sum" into a tolerance nobody can choose.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from royalegym.protocol import (
    EMPTY_CARD,
    HAND_SIZE,
    BattleState,
    CardInfo,
    EntityKind,
    EntityState,
    PlayerState,
    TowerSlot,
    Winner,
)
from royalelearn.metrics.records import TERMINAL_REWARD_TERM
from royalelearn.rewards import (
    CROWNS,
    CommittedElixirPotential,
    PotentialCombinedReward,
    PotentialCrownReward,
    PotentialReward,
    PotentialTowerHPReward,
    default_potential_reward,
    set_gamma,
)

GAMMA = 0.997
TOWER_MAX = 1000
MILLI = 1000
#: A uid counter's worth of distinct entities; nothing reads the value but the uniqueness.
FIRST_UID = 100


@pytest.fixture(scope="module")
def engine() -> Any:
    """A MockEngine, for its card catalogue and nothing else."""
    from royalegym.mock_engine import MockEngine

    return MockEngine()


@pytest.fixture(scope="module")
def cards(engine: Any) -> list[CardInfo]:
    """Cards that put units on the board, cheapest first.

    Spells summon nothing and so are never committed elixir; a card that summons several units
    is what makes ``Fraction(elixir, count)`` a fraction rather than an integer, and both kinds
    are wanted here.
    """
    summoning = [card for card in engine.cards() if card.count >= 1 and card.elixir > 0]
    assert summoning, "this catalogue summons no units, so there is nothing to commit"
    return sorted(summoning, key=lambda card: (card.count, card.elixir))


def player(
    team: int,
    *,
    elixir: float = 5.0,
    crowns: int = 0,
    towers: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PlayerState:
    """One seat, with its tower health written as fractions of full."""
    return PlayerState(
        team=team,
        elixir_milli=round(elixir * MILLI),
        hand=[EMPTY_CARD] * HAND_SIZE,
        next_card=EMPTY_CARD,
        crowns=crowns,
        tower_hp=[round(fraction * TOWER_MAX) for fraction in towers],
        tower_max_hp=[TOWER_MAX] * len(TowerSlot),
        king_active=False,
    )


def unit(uid: int, team: int, card: CardInfo) -> EntityState:
    """One live unit of ``team``: the unit ``card``'s own catalogue row describes.

    Its stats are the row's because that is what makes it the card's own unit. This helper used
    to give every unit made-up stats (100 hp, radius 1, ground), which was harmless while the
    elixir potential priced a unit by whatever card it was filed under, and is exactly the
    blindness that let a Goblin Gang read as eighteen elixir: a board built by hand only ever
    held units that were their card's own. Position is irrelevant to every potential here.
    """
    return EntityState(
        uid=uid,
        team=team,
        kind=EntityKind.TROOP,
        card_id=card.card_id,
        tower_slot=-1,
        x=0,
        y=0,
        hp=card.hitpoints,
        max_hp=card.hitpoints,
        radius=card.radius,
        flying=bool(card.flying),
        deploy_ticks=0,
    )


def tower(uid: int, team: int, slot: TowerSlot) -> EntityState:
    """A crown tower, which no committed-elixir potential counts."""
    return EntityState(
        uid=uid,
        team=team,
        kind=EntityKind.KING_TOWER if slot is TowerSlot.KING else EntityKind.PRINCESS_TOWER,
        card_id=EMPTY_CARD,
        tower_slot=int(slot),
        x=0,
        y=0,
        hp=TOWER_MAX,
        max_hp=TOWER_MAX,
        radius=1,
        flying=False,
        deploy_ticks=0,
    )


def state(
    players: list[PlayerState],
    entities: list[EntityState] | None = None,
    *,
    game_over: bool = False,
    winner: int = Winner.NONE,
) -> BattleState:
    """A battle state with the towers of both seats always present."""
    standing = [
        tower(FIRST_UID + index, team, slot)
        for index, (team, slot) in enumerate((t, s) for t in (0, 1) for s in TowerSlot)
    ]
    return BattleState(
        tick=0,
        tick_ms=50,
        regular_ticks=100,
        overtime_ticks=50,
        elixir_rate=1,
        overtime=False,
        players=players,
        entities=standing + list(entities or ()),
        game_over=game_over,
        winner=winner,
    )


def units(team: int, card: CardInfo, *, first: int = 0) -> list[EntityState]:
    """Everything one play of ``card`` puts on the board."""
    return [unit(first + index, team, card) for index in range(card.count)]


# --------------------------------------------------------------------------
# Each term is a potential difference
# --------------------------------------------------------------------------


def test_the_crown_term_is_the_discounted_difference_of_the_crown_potential() -> None:
    previous = state([player(0, crowns=1), player(1, crowns=0)])
    current = state([player(0, crowns=1), player(1, crowns=2)])
    term = PotentialCrownReward()
    term.set_gamma(GAMMA)

    before = Fraction(1 - 0, CROWNS)
    after = Fraction(1 - 2, CROWNS)
    expected = float(Fraction(GAMMA) * after - before)

    assert term.get_reward(0, previous, current, []) == expected


def test_the_tower_term_is_the_discounted_difference_of_the_tower_potential() -> None:
    previous = state([player(0, towers=(1.0, 0.5, 1.0)), player(1, towers=(1.0, 1.0, 1.0))])
    current = state([player(0, towers=(1.0, 0.5, 1.0)), player(1, towers=(1.0, 1.0, 0.25))])
    term = PotentialTowerHPReward()
    term.set_gamma(GAMMA)

    before = Fraction(Fraction(5, 2) - 3, CROWNS)
    after = Fraction(Fraction(5, 2) - Fraction(9, 4), CROWNS)
    expected = float(Fraction(GAMMA) * after - before)

    assert term.get_reward(0, previous, current, []) == expected


def test_the_elixir_term_is_the_discounted_difference_of_the_committed_potential(
    engine: Any, cards: list[CardInfo]
) -> None:
    card = cards[-1]
    previous = state([player(0, elixir=7.0), player(1, elixir=3.0)])
    current = state(
        [player(0, elixir=7.0), player(1, elixir=3.0)], units(1, card, first=1)
    )
    term = CommittedElixirPotential(scale=10.0)
    term.bind(engine)
    term.set_gamma(GAMMA)

    value = Fraction(card.elixir, card.count) * card.count
    before = (Fraction(7) - Fraction(3)) / Fraction(10)
    after = (Fraction(7) - (Fraction(3) + value)) / Fraction(10)
    expected = float(Fraction(GAMMA) * after - before)

    assert term.get_reward(0, previous, current, []) == expected


def test_a_crown_tower_is_not_committed_elixir(engine: Any) -> None:
    """The towers are in ``entities`` and are not something anybody played out of a hand."""
    term = CommittedElixirPotential(scale=10.0)
    term.bind(engine)
    term.set_gamma(1.0)
    balanced = state([player(0, elixir=4.0), player(1, elixir=4.0)])

    assert term.get_reward(0, balanced, balanced, []) == 0.0


# --------------------------------------------------------------------------
# What the committed-elixir potential says
# --------------------------------------------------------------------------


def test_playing_a_card_is_worth_exactly_nothing(engine: Any, cards: list[CardInfo]) -> None:
    """Elixir moves from the bar to the board and the sum does not move.

    Exactly nothing, at the run's own discount and not at one: the potential is level in both
    states, so there is no discounting for the discount to do.
    """
    term = CommittedElixirPotential(scale=10.0)
    term.bind(engine)
    term.set_gamma(GAMMA)
    for card in cards:
        previous = state([player(0, elixir=10.0), player(1, elixir=10.0)])
        current = state(
            [player(0, elixir=10.0 - card.elixir), player(1, elixir=10.0)],
            units(0, card),
        )
        assert term.get_reward(0, previous, current, []) == 0.0, card.name


def test_a_unit_lost_costs_what_it_cost(engine: Any, cards: list[CardInfo]) -> None:
    card = cards[-1]
    previous = state([player(0, elixir=5.0), player(1, elixir=5.0)], units(0, card))
    current = state([player(0, elixir=5.0), player(1, elixir=5.0)])
    term = CommittedElixirPotential(scale=10.0)
    term.bind(engine)
    term.set_gamma(1.0)

    assert term.get_reward(0, previous, current, []) == -float(Fraction(card.elixir, 10))
    assert term.get_reward(1, previous, current, []) == float(Fraction(card.elixir, 10))


def test_hoarding_while_the_opponent_regenerates_is_negative(engine: Any) -> None:
    """A full bar cannot rise, and the opponent's can. That is the whole of the leak penalty,
    without a per-player term that both seats can earn at once."""
    term = CommittedElixirPotential(scale=10.0)
    term.bind(engine)
    term.set_gamma(GAMMA)
    previous = state([player(0, elixir=10.0), player(1, elixir=8.0)])
    current = state([player(0, elixir=10.0), player(1, elixir=9.0)])

    assert term.get_reward(0, previous, current, []) < 0.0
    assert term.get_reward(1, previous, current, []) > 0.0


# --------------------------------------------------------------------------
# The composition
# --------------------------------------------------------------------------


def test_the_composition_is_exactly_antisymmetric_between_the_seats(
    engine: Any, cards: list[CardInfo]
) -> None:
    """Every seat's reward is the other's negated, to the last bit, on a transition in which
    both seats did something different."""
    reward = default_potential_reward()
    reward.bind(engine)
    set_gamma(reward, GAMMA)
    previous = state(
        [player(0, elixir=9.0, crowns=1, towers=(1.0, 0.4, 1.0)), player(1, elixir=2.5)],
        units(0, cards[0]) + units(1, cards[-1], first=FIRST_UID * 2),
    )
    current = state(
        [
            player(0, elixir=3.0, crowns=1, towers=(1.0, 0.4, 0.9)),
            player(1, elixir=4.5, crowns=2, towers=(1.0, 1.0, 1.0)),
        ],
        units(0, cards[-1], first=FIRST_UID * 3),
        game_over=True,
        winner=Winner.RED,
    )

    blue = reward.get_reward(0, previous, current, [])
    red = reward.get_reward(1, previous, current, [])

    assert blue != 0.0
    assert blue == -red


def test_a_mirrored_transition_is_worth_exactly_zero_to_both_seats(
    engine: Any, cards: list[CardInfo]
) -> None:
    """The property the exact arithmetic exists for. A float sum of the same unit values
    returns plus and minus 1e-19 here, which is not zero and cannot be asserted against."""
    reward = default_potential_reward()
    reward.bind(engine)
    set_gamma(reward, GAMMA)
    card = cards[-1]
    chipped = (1.0, 0.3, 1.0)
    standing = units(0, card) + units(1, card, first=FIRST_UID * 2)
    previous = state(
        [player(0, elixir=6.0, towers=chipped), player(1, elixir=6.0, towers=chipped)],
        standing,
    )
    current = state(
        [
            player(0, elixir=6.0 - card.elixir, towers=chipped),
            player(1, elixir=6.0 - card.elixir, towers=chipped),
        ],
        standing
        + units(0, card, first=FIRST_UID * 3)
        + units(1, card, first=FIRST_UID * 4),
    )

    assert reward.get_reward(0, previous, current, []) == 0.0
    assert reward.get_reward(1, previous, current, []) == 0.0


def test_the_terminal_term_is_the_only_one_that_is_not_shaping(engine: Any) -> None:
    """A battle that ends level pays the winner the objective and nothing else."""
    reward = default_potential_reward()
    reward.bind(engine)
    set_gamma(reward, 1.0)
    level = [player(0, elixir=5.0), player(1, elixir=5.0)]
    previous = state(level)
    current = state(level, game_over=True, winner=Winner.BLUE)

    assert reward.get_reward(0, previous, current, []) == 1.0
    assert reward.get_reward(1, previous, current, []) == -1.0


def test_a_potential_is_zero_once_the_battle_is_over(engine: Any) -> None:
    """The shaping over a whole episode has to telescope to nothing, and that needs Phi(end)=0.

    A potential term pays ``gamma * Phi(s') - Phi(s)`` every step, so over an episode the sum
    collapses to ``gamma^T * Phi(s_T) - Phi(s_0)``. Only the first and last terms survive, and
    ``Phi(s_0)`` is the same for every policy. ``Phi(s_T)`` is not: read off the final state, the
    crown potential of a 3-0 win is three times the crown potential of a 1-0 win, so the shaping
    would pay for the margin and the objective would no longer be the objective. Ng, Harada and
    Russell's invariance result takes the terminal state's potential to be zero, and it has to be
    taken rather than computed, because a finished battle has no future to anticipate.

    A truncation is the other case and it is not this one. The battle was cut rather than decided,
    the position still has value, and the estimator bootstraps from it.
    """
    reward = default_potential_reward()
    reward.bind(engine)
    set_gamma(reward, GAMMA)
    ahead = [player(0, crowns=2, towers=(0.0, 0.0, 1.0)), player(1, crowns=0)]
    previous = state(ahead)
    won = state(
        [player(0, crowns=3, towers=(0.0, 0.0, 1.0)), player(1, crowns=0, towers=(0.0, 0.0, 0.0))],
        game_over=True,
        winner=Winner.BLUE,
    )

    blue = reward.terms_for(0)
    reward.get_reward(0, previous, won, [])
    blue = reward.terms_for(0)
    # Only the objective is paid on the closing step, less whatever potential the position held.
    crown = blue["PotentialCrownReward"]
    assert crown == pytest.approx(0.2 * (0.0 - 2 / 3)), blue
    assert blue[TERMINAL_REWARD_TERM] == pytest.approx(1.0)


def test_a_truncated_battle_keeps_the_potential_of_the_position(engine: Any) -> None:
    """A step limit is not an ending. The position still has value and the estimator uses it."""
    reward = default_potential_reward()
    reward.bind(engine)
    set_gamma(reward, GAMMA)
    ahead = [player(0, crowns=2), player(1, crowns=0)]
    reward.get_reward(0, state(ahead), state(ahead), [])

    assert reward.terms_for(0)["PotentialCrownReward"] == pytest.approx(
        0.2 * (GAMMA * 2 / 3 - 2 / 3)
    )


# --------------------------------------------------------------------------
# The discount
# --------------------------------------------------------------------------


def test_set_gamma_reaches_every_term_that_takes_one(engine: Any) -> None:
    reward = default_potential_reward()
    reward.bind(engine)
    reward.set_gamma(0.5)
    potentials = [term for term, _ in reward.terms if hasattr(term, "gamma")]

    assert len(potentials) == 3, "every shaping term of the composition is a potential"
    assert all(term.gamma == 0.5 for term in potentials)


def test_set_gamma_reaches_a_term_nested_in_another_composition(engine: Any) -> None:
    """A bot creator's composition may hold one of these inside another combination; the
    discount has to reach the leaves rather than the first level."""
    from royalegym.reward import CombinedReward

    inner = default_potential_reward()
    outer = CombinedReward([(inner, 0.5)])
    outer.bind(engine)
    set_gamma(outer, 0.25)

    assert all(term.gamma == 0.25 for term, _ in inner.terms if hasattr(term, "gamma"))


def test_the_discount_is_the_one_the_schedule_is_at(run_config: Any) -> None:
    """The reward's gamma and the learner's are one number, read from one place.

    The schedule is evaluated once at the top of an iteration into a ``ScheduleState``; that
    value rides the Step command to every worker and is what GAE bootstraps with, so a test that
    the two agree is a test that nobody read the config a second time.
    """
    pytest.importorskip("torch")
    from royalelearn.learn.schedules import ScheduleSet

    schedules = ScheduleSet.from_config(run_config)
    sched = schedules.state(iteration=3, cumulative_env_steps=1_000_000, cumulative_timesteps=0)
    reward = default_potential_reward()
    set_gamma(reward, sched.gamma)

    assert all(term.gamma == sched.gamma for term, _ in reward.terms if hasattr(term, "gamma"))


def test_an_environment_built_on_this_composition_plays(mock_env_spec: Any) -> None:
    """The composition is reachable from a config, binds through the environment, and returns a
    number for both seats of a real transition.

    The worker sets the discount by calling ``set_gamma`` on whatever reward function its
    environment holds, so the composition has to be the object that carries the method.
    """
    import msgspec

    from royalelearn.rollout.envspec import ComponentSpec

    spec = msgspec.structs.replace(
        mock_env_spec,
        reward_fn=ComponentSpec("royalelearn.rewards.default_potential_reward"),
    )
    env = spec.build_vec(1)
    try:
        for inner in env.envs:
            inner.reward_fn.set_gamma(GAMMA)
        env.reset(seed=7)
        for _ in range(8):
            _obs, rewards, _term, _trunc, _info = env.step([0] * env.num_envs)
            # One battle, two seats, and what one seat is paid the other pays. The cast to
            # float32 on the way out does not disturb it: negating a float is exact.
            assert len(rewards) == env.num_envs
            assert float(rewards[0]) + float(rewards[1]) == 0.0
    finally:
        env.close()


def test_the_discount_is_not_part_of_the_environment_s_configuration() -> None:
    """``config()`` feeds the run identity and the ladder's context, both of which must not move
    when a schedule moves."""
    reward = default_potential_reward()
    reward.set_gamma(0.5)
    before = reward.config()
    reward.set_gamma(0.9)

    assert reward.config() == before


class ConstantPotential(PotentialReward):
    """``Phi = 1`` in every state, so a step pays exactly ``gamma - 1``: the discount the term
    was last given, readable straight off the reward the environment returns."""

    def potential(self, state: BattleState, team: int) -> Fraction:
        return Fraction(1)


def discount_probe() -> PotentialCombinedReward:
    """The shipped composition's class around a term whose reward IS its discount.

    Built by name in a worker process like any reward a config names, so the discount has to
    reach it the way it reaches the shipped terms: through the ``Step`` command and the
    composition's own ``set_gamma``.
    """
    return PotentialCombinedReward([(ConstantPotential(), 1.0)])


#: Dyadic, so ``gamma - 1`` is exact in float32 and a reward can equal it rather than
#: approximate it.
FIRST_GAMMA, SECOND_GAMMA = 0.9921875, 0.984375


@pytest.mark.parametrize("source", ["inline", pytest.param("process", marks=pytest.mark.slow)])
def test_the_schedule_s_discount_reaches_the_reward_during_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """The discount a worker's reward is computed with is the one the learner bootstraps with,
    in a real collection, and it follows the schedule when the schedule moves.

    The tests above call ``set_gamma`` by hand; this one leaves it to the run. The coordinator
    evaluates the schedule once per iteration, the value rides every ``Step`` to the worker, and
    the worker hands it to whatever reward its environment holds. The schedule here steps from
    one discount to another at exactly the second iteration's first environment step, so a
    worker that kept the first value, or never received one, pays a different number.

    It reads each round as the source hands it over, rather than the buffer the rounds are
    filed into, so that it asks only whether the discount arrived. A round at cycle zero is the
    iteration's opening publication, which follows no step; every later one, the trailing round
    included, carries the reward of a step this iteration's command made.
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import msgspec
    import numpy as np

    from royalelearn import config as C
    from royalelearn.rollout.envspec import ComponentSpec
    from test_coordinator import coordinator, tiny_config

    base = tiny_config(tmp_path)
    geometry = C.geometry(base)
    env_steps_per_iteration = geometry.cycles * geometry.n_battles
    config = tiny_config(
        tmp_path,
        env=msgspec.structs.replace(
            base.env, reward_fn=ComponentSpec(f"{__name__}.discount_probe")
        ),
        extra_component_modules=[__name__],
        rollout=msgspec.structs.replace(base.rollout, source=source),
        advantage=msgspec.structs.replace(
            base.advantage,
            gamma=C.PiecewiseConstantSpec(
                [(0, FIRST_GAMMA), (env_steps_per_iteration, SECOND_GAMMA)]
            ),
        ),
    )

    with coordinator(config) as run:
        rounds: list[tuple[int, np.ndarray]] = []

        def keep(round_: Any) -> Any:
            rounds.append((int(round_.cycle), np.array(round_.reward[round_.valid], copy=True)))
            return round_

        next_round, finish_iteration = run.source.next_round, run.source.finish_iteration
        monkeypatch.setattr(run.source, "next_round", lambda *a, **k: keep(next_round(*a, **k)))
        monkeypatch.setattr(
            run.source,
            "finish_iteration",
            lambda *a, **k: [keep(round_) for round_ in finish_iteration(*a, **k)],
        )

        for gamma in (FIRST_GAMMA, SECOND_GAMMA):
            rounds.clear()
            run.iterate()
            assert run.rows[-1]["run/gamma"] == gamma

            stepped = [(cycle, paid) for cycle, paid in rounds if cycle > 0]
            assert sorted({cycle for cycle, _ in stepped}) == list(range(1, geometry.cycles + 1))
            paid = np.concatenate([paid for _, paid in stepped])
            assert paid.size == geometry.cycles * geometry.n_slots
            assert set(paid.tolist()) == {float(np.float32(gamma - 1.0))}


# --------------------------------------------------------------------------
# What ships
# --------------------------------------------------------------------------

#: The composition this module exists to provide, as a config names it.
SHIPPED_REWARD = "royalelearn.rewards.default_potential_reward"
EXAMPLE_CONFIGS = Path(__file__).resolve().parents[1] / "examples" / "configs"
#: Named rather than globbed: the folder also holds configs written for particular runs, which
#: answer to those runs rather than to what ships.
EXAMPLE_FILES = ("laptop.json", "smoke.json", "workstation.json")


def _shipped_configs() -> list[Any]:
    """Every config a user can start a run from without writing one: each profile, and each
    example file, which is what ``train --config`` is shown with."""
    from royalelearn import config as C

    shipped = [
        pytest.param(lambda n=name: C.profile(n), id=f"profile:{name}") for name in C.PROFILES
    ]
    for name in EXAMPLE_FILES:
        path = EXAMPLE_CONFIGS / name
        shipped.append(pytest.param(lambda p=path: C.load_config(p), id=f"file:{name}"))
    return shipped


@pytest.mark.parametrize("build", _shipped_configs())
def test_every_shipped_config_trains_against_the_potential_reward(build: Any) -> None:
    """A run started from anything shipped trains against this composition, not RoyaleGym's
    ``default_reward``.

    The difference is the objective rather than a detail of it: RoyaleGym's elixir-trade term is
    not a potential and pays a player who never commits a card, and its tower term is one
    discount away from policy-invariant. A config that names the other one is a run optimising
    something the harness's own reward was written to stop it optimising.
    """
    config = build()

    assert config.env.reward_fn.cls == SHIPPED_REWARD
    assert config.env.reward_fn.kwargs == {}


def test_the_terminal_term_is_filed_where_the_shaping_alarm_looks_for_it(engine: Any) -> None:
    """The ``shaping_dominates`` alarm compares the shaping terms with the terminal one, and it
    finds the terminal one by name. That name has to be one the shipped reward emits.

    The metrics fixtures file a term called ``terminal`` because they were written against the
    constant rather than against a reward; this asks the reward. A level battle that ends in a
    win pays the objective and nothing else, so the whole of it must arrive under that name.
    """
    from royalelearn.metrics.records import TERMINAL_REWARD_TERM

    reward = default_potential_reward()
    reward.bind(engine)
    set_gamma(reward, GAMMA)
    level = [player(0, elixir=5.0), player(1, elixir=5.0)]
    reward.get_reward(0, state(level), state(level, game_over=True, winner=Winner.BLUE), [])

    assert reward.terms_for(0).get(TERMINAL_REWARD_TERM) == 1.0
