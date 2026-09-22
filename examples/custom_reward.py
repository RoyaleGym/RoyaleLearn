"""Swapping the reward, and keeping the run identity honest while doing it.

A reward function is the other thing a bot creator changes. What makes it more than a subclass
is the three lines after it: the module it lives in is named in ``extra_component_modules``, so
``EnvFactorySpec`` is allowed to instantiate it in a worker process, and the component is
recorded in ``config.env`` -- which means it is hashed into the run identity, written into
every checkpoint, and carried in the ladder's ``context``. Two policies rated against each
other are then known to have been trained under the same objective, and a run trained under a
different one is a different experiment rather than a confusing line on the same chart.

    python examples/custom_reward.py
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from msgspec.structs import replace

from royalegym.protocol import BattleState, Engine, EntityKind
from royalegym.reward import CombinedReward, WinLossReward
from royalelearn import LearningCoordinator, load_config
from royalelearn.rewards import (
    CommittedElixirPotential,
    PotentialCombinedReward,
    PotentialCrownReward,
    PotentialReward,
    PotentialTowerHPReward,
)


class OffensivePresence(PotentialReward):
    """Reward having units on the opponent's half, as a potential.

    Written as a potential rather than as a bonus per step: ``gamma*Phi(s') - Phi(s)`` sums to
    ``Phi(end) - Phi(start)`` over any trajectory, so it cannot be farmed by standing still and
    cannot change which policy is optimal. Every term the harness ships has the same shape, for
    the same reason, and a term that does not is the one that turns into a turtle.
    """

    def __init__(self, scale: float = 4.0) -> None:
        super().__init__()
        self.scale = scale
        self._scale = Fraction(scale)
        self.half = 0

    def config(self) -> dict[str, object]:
        """What the checkpoint and the ladder's context record about this term."""
        return {"scale": self.scale}

    def bind(self, engine: Engine) -> None:
        """The arena, once. Where the river is, is a fact about the board and not a constant."""
        self.half = int(engine.arena().half)

    def potential(self, state: BattleState, team: int) -> Fraction:
        """Own units past the river, less the opponent's, over ``scale``.

        Antisymmetric in the seat by construction, which is what keeps the composition
        zero-sum and lets the suite check that rather than tolerate it.
        """
        return Fraction(self._past(state, team) - self._past(state, 1 - team), 1) / self._scale

    def _past(self, state: BattleState, team: int) -> int:
        towers = (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
        attacking_up = team == 0
        return sum(
            1
            for entity in state.entities
            if entity.team == team
            and entity.kind not in towers
            and ((entity.y >= self.half) if attacking_up else (entity.y < self.half))
        )


def offensive_reward() -> CombinedReward:
    """The shipped composition with one term added, as a factory a worker can name.

    A factory rather than an instance: ``ComponentSpec`` holds a class path and its keyword
    arguments, a worker imports the path and calls it, and nothing is ever pickled across the
    process boundary. The weight is small for the same reason the shipped shaping weights are:
    the terminal term is the objective and the rest are there to make the first hour legible.
    """
    return PotentialCombinedReward(
        [
            (WinLossReward(draw=0.0), 1.0),
            (PotentialCrownReward(), 0.2),
            (PotentialTowerHPReward(), 0.1),
            (CommittedElixirPotential(scale=10.0), 0.05),
            (OffensivePresence(scale=4.0), 0.02),
        ]
    )


def main() -> None:
    config = load_config(Path(__file__).parent / "configs" / "laptop.json")
    config.run_name = "offensive-presence"
    # The module this file is, so a spawned worker may import the factory by name. Without it
    # the component allow-list refuses the path, which is the point: a worker that imports
    # whatever a config file names is a config file that runs code.
    config.extra_component_modules = ["custom_reward"]
    config.env = replace(
        config.env,
        reward_fn=replace(config.env.reward_fn, cls="custom_reward.offensive_reward"),
    )
    with LearningCoordinator(config) as run:
        run.learn(until_timesteps=10_000_000)


if __name__ == "__main__":
    main()
