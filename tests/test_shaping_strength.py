"""How loud is the shaping? The row could not say, and a config was tuned on the wrong answer.

``env/reward_shaping_abs`` is the mean, over seats, of each shaping term's per-episode SUM. A
potential term pays ``gamma * Phi(s') - Phi(s)`` every step, so that sum telescopes. Summed without
discounting, which is what the recorder does, what is left is::

    -Phi(s_0) - (1 - gamma) * sum_{t=1}^{T-1} Phi(s_t)

and at a symmetric start ``Phi(s_0)`` is zero. The metric therefore measures how far the potential
wandered, scaled by ``1 - gamma``. That is the right thing for what ``shaping_dominates`` checks,
which is whether a term still telescopes. It says nothing about how strong the shaping is, and it
falls as the discount schedule rises even when nothing else changes.

Both halves were measured before this file was written. On train-hog26-10's own rows, the metric
fell from 0.0346 to 0.0192 between iterations 51 and 201 while ``metric / (1 - gamma)`` held at 14.1
to 14.5. And on identical random-legal battles, moving gamma from 0.997 to 0.999 cut the metric from
0.0462 to 0.0185 while the per-step magnitude, ``sum |F_t|``, went from 1.136 to 1.106. A config
read the first number as "shaping is 1% of the objective" and raised the weights 1.5x to 6x. The
per-step magnitude at the SHIPPED weights was already 1.4x the objective's.

So the second number is now published beside the first: ``env/reward_terms_step_abs/<term>`` and
``env/reward_shaping_step_abs``. These tests pin what each one measures.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import pairwise
from typing import Any

import msgspec
import pytest
from msgspec.structs import replace as dc_replace

from royalelearn.api.rollout import EpisodeRecord
from royalelearn.metrics.records import episode_fields
from royalelearn.rewards import (
    PotentialCombinedReward,
    PotentialCrownReward,
    PotentialTowerHPReward,
    default_potential_reward,
    set_gamma,
)
from test_metrics import LOST, WON, _record
from test_rewards import player, state

# -- the weights ---------------------------------------------------------------------------


def _weights(reward: PotentialCombinedReward) -> dict[str, float]:
    return {type(term).__name__: float(weight) for term, weight in reward.terms}


def test_the_shipped_weights_are_what_a_config_gets_by_default() -> None:
    assert _weights(default_potential_reward()) == {
        "WinLossReward": 1.0,
        "PotentialCrownReward": 0.2,
        "PotentialTowerHPReward": 0.1,
        "CommittedElixirPotential": 0.05,
    }


def test_each_weight_reaches_its_own_term_and_no_other() -> None:
    """Three distinct values, so a weight handed to the wrong term cannot pass."""
    reward = default_potential_reward(crown=0.3, tower_hp=0.7, elixir=0.11)
    assert _weights(reward) == {
        "WinLossReward": 1.0,
        "PotentialCrownReward": 0.3,
        "PotentialTowerHPReward": 0.7,
        "CommittedElixirPotential": 0.11,
    }


def test_a_zero_weight_turns_a_term_off_and_keeps_its_row() -> None:
    """Off is a weight, not a missing term: the row keeps the key and it reads zero."""
    reward = default_potential_reward(elixir=0.0)
    assert _weights(reward)["CommittedElixirPotential"] == 0.0


@pytest.mark.parametrize("name", ["crown", "tower_hp", "elixir"])
@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf])
def test_a_negative_or_non_finite_weight_is_refused_by_name(name: str, value: float) -> None:
    """A negative potential weight pays the seat for losing; NaN poisons every return it touches.

    Either would train. Neither would say so until a reader wondered why the curves looked odd.
    """
    with pytest.raises(ValueError, match=name):
        default_potential_reward(**{name: value})


# -- what the recorded episode sum is -------------------------------------------------------


def _trajectory() -> list[Any]:
    """A battle that starts level, swings both ways, and is decided.

    ``Phi(s_0)`` is zero for every term at a level start, which is the case every real battle is
    in. The middle states move the crown and tower potentials by different amounts, in both
    directions, so a sign error or a skipped step changes the answer.
    """
    level = (1.0, 1.0, 1.0)
    return [
        state([player(0), player(1)]),
        state([player(0), player(1, towers=(0.9, 1.0, 1.0))]),
        state([player(0, crowns=1), player(1, towers=(0.0, 1.0, 1.0))]),
        state([player(0, crowns=1, towers=(0.5, 1.0, 1.0)), player(1, towers=(0.0, 1.0, 1.0))]),
        state([player(0, crowns=1, towers=(0.5, 0.6, 1.0)), player(1, towers=(0.0, 0.8, 1.0))]),
        state(
            [player(0, crowns=3, towers=(0.5, 0.6, 1.0)), player(1, towers=level)],
            game_over=True,
            winner=0,
        ),
    ]


def _per_step(term: Any, states: list[Any], gamma: float) -> list[float]:
    set_gamma(term, gamma)
    return [term.get_reward(0, prev, nxt, []) for prev, nxt in pairwise(states)]


@pytest.mark.parametrize("term_class", [PotentialCrownReward, PotentialTowerHPReward])
@pytest.mark.parametrize("gamma", [0.997, 0.999])
def test_the_recorded_sum_is_the_discount_s_leftover_and_not_the_signal(
    term_class: type, gamma: float
) -> None:
    """The undiscounted episode sum -- what ``env/reward_terms`` records -- has a closed form.

    It is ``-Phi(s_0) - (1 - gamma) * sum of the interior potentials``. The discounted sum is
    ``-Phi(s_0)`` alone, which is Ng, Harada and Russell's result. Holding both here is what makes
    the metric's meaning a checked fact rather than a docstring.
    """
    term = term_class()
    states = _trajectory()
    steps = _per_step(term, states, gamma)
    phi = [term.potential(s, 0) for s in states]
    g = Fraction(gamma)

    expected_undiscounted = -phi[0] - (1 - g) * sum(phi[1:-1], Fraction(0))
    assert sum(steps) == pytest.approx(float(expected_undiscounted), rel=1e-12, abs=1e-15)

    discounted = sum(Fraction(gamma) ** t * Fraction(f) for t, f in enumerate(steps))
    assert float(discounted) == pytest.approx(float(-phi[0]), abs=1e-12)
    assert phi[0] == 0, "the trajectory is meant to start level; the test is weaker if it does not"


def test_the_signal_barely_moves_with_the_discount_and_the_recorded_sum_follows_it() -> None:
    """The controlled version of what train-hog26-10's rows showed.

    Same states, two discounts. With ``Phi(s_0) = 0`` the recorded sum is exactly proportional
    to ``1 - gamma``, so tripling ``1 - gamma`` triples it. The per-step magnitude is what a
    policy gradient is actually handed, and it moves by the size of ``gamma`` itself.
    """
    states = _trajectory()
    net, loud = {}, {}
    for gamma in (0.997, 0.999):
        steps = _per_step(PotentialTowerHPReward(), states, gamma)
        net[gamma] = abs(sum(steps))
        loud[gamma] = sum(abs(f) for f in steps)

    assert net[0.997] / net[0.999] == pytest.approx(3.0, rel=1e-9)
    assert loud[0.997] / loud[0.999] == pytest.approx(1.0, abs=0.01)
    assert loud[0.999] > 20 * net[0.999], "the signal is loud and the recorded sum is not"


# -- the recorder ---------------------------------------------------------------------------


class _Inner:
    """A reward whose breakdown is scripted, so the recorder is tested and nothing else."""

    def __init__(self, script: list[dict[str, float]]) -> None:
        self._script = list(script)
        self._last: dict[str, float] = {}

    def get_reward(self, team: int, prev: Any, state: Any, results: Any) -> float:
        self._last = self._script.pop(0)
        return sum(self._last.values())

    def terms_for(self, team: int) -> dict[str, float]:
        return dict(self._last)

    def reset(self, state: Any) -> None:
        pass


def test_the_recorder_keeps_the_sum_and_the_magnitude_apart() -> None:
    """Plus then minus: the sum is zero and the magnitude is not. That is the whole distinction."""
    from royalelearn.rollout.inline import _term_totals, _TermRecorder

    sink: dict[int, dict[str, list[float]]] = {0: {}}
    recorder = _TermRecorder(
        _Inner([{"terminal": 0.0, "Shape": 0.25}, {"terminal": 1.0, "Shape": -0.25}]),
        sink,
        {0: 0},
    )
    recorder.get_reward(0, None, None, None)
    recorder.get_reward(0, None, None, None)

    net, loud = _term_totals(sink[0])
    assert net == {"terminal": 1.0, "Shape": 0.0}
    assert loud == {"terminal": 1.0, "Shape": 0.5}


# -- the row --------------------------------------------------------------------------------


def _pair(net: dict[str, float], loud: dict[str, float]) -> list[EpisodeRecord]:
    blue = dc_replace(_record(0, 0, WON), reward_terms=net, reward_terms_step_abs=loud)
    red = dc_replace(
        _record(0, 1, LOST),
        reward_terms={k: -v for k, v in net.items()},
        reward_terms_step_abs=dict(loud),
    )
    return [blue, red]


def test_the_row_publishes_how_loud_each_term_was() -> None:
    """Distinct magnitudes per term, and the objective kept out of the shaping total."""
    fields = episode_fields(
        _pair(
            {"terminal": 1.0, "PotentialCrownReward": 0.01, "CommittedElixirPotential": -0.004},
            {"terminal": 1.0, "PotentialCrownReward": 0.15, "CommittedElixirPotential": 0.8},
        )
    ).fields

    assert fields["env/reward_terms_step_abs/PotentialCrownReward"] == pytest.approx(0.15)
    assert fields["env/reward_terms_step_abs/CommittedElixirPotential"] == pytest.approx(0.8)
    assert fields["env/reward_terms_step_abs/terminal"] == pytest.approx(1.0)
    assert fields["env/reward_shaping_step_abs"] == pytest.approx(0.95)
    # the old pair is unchanged, so a row can be read against every row before it
    assert fields["env/reward_shaping_abs"] == pytest.approx(0.014)
    assert fields["env/reward_terminal_abs"] == pytest.approx(1.0)


def test_a_record_written_before_the_magnitude_existed_leaves_it_absent() -> None:
    """Every ``episodes.jsonl`` already on disk decodes, and the row says nothing rather than 0.

    Zero would claim the shaping was silent. The honest answer for an old record is that nobody
    measured it.
    """
    old = msgspec.json.encode(_record(0, 0, WON))
    as_dict = msgspec.json.decode(old)
    as_dict.pop("reward_terms_step_abs", None)
    record = msgspec.convert(as_dict, EpisodeRecord)
    assert record.reward_terms_step_abs == {}

    red = dc_replace(_record(0, 1, LOST), reward_terms_step_abs={})
    fields = episode_fields([record, red]).fields
    assert "env/reward_shaping_step_abs" not in fields
    assert not any(key.startswith("env/reward_terms_step_abs/") for key in fields)
    assert "env/reward_shaping_abs" in fields, "the old metric must not disappear with the new one"
