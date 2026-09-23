"""Whose frames are in a stacked evaluation observation.

Until 2026-09-23 the stacked-frame history for evaluation battles was ONE LIST on the coordinator.
Both seats act through the same object and alternately, so a policy's "previous frame" was usually
its opponent's; and nothing reset it between battles, so the first decisions of a battle carried
the tail of the one before.

It showed nowhere because ``obs.frame_stack`` is 1 by default and 1 in every run this project has
done, and at 1 nothing stacks. The moment anybody sets it to 2 -- which is a supported option with
a whole rectangle implementation behind it -- every evaluation battle would have been played on
observations no policy was trained on, and the scores would still have come out in [0, 1], and a
gate would still have decided.

The history belongs to the POLICY now, and ``EnvBattlePlayer`` builds its policies once per
battle, so per-battle and per-seat is a property of the construction rather than of somebody
remembering to clear it. These tests grade that by giving two policies different frames and
checking neither sees the other's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from royalelearn.ladder.actors import EvalActors

torch = pytest.importorskip("torch")


class _Spec:
    """The three fields ``obs_batch`` reads, and a frame stack the test chooses."""

    hand_size = 2
    tiles = (2, 3)
    n_actions = 1 + hand_size * tiles[0] * tiles[1]
    vector_size = 4

    def __init__(self, frame_stack: int) -> None:
        self.frame_stack = frame_stack


def actors(frame_stack: int) -> EvalActors:
    return EvalActors(_Spec(frame_stack), net=None, snapshots=None)  # type: ignore[arg-type]


def observation(spec: _Spec, fill: float) -> dict[str, Any]:
    mask = np.zeros(spec.n_actions, dtype=bool)
    mask[0] = True
    return {
        "spatial": np.full((2, *spec.tiles), fill, dtype=np.float32),
        "vector": np.zeros(spec.vector_size, dtype=np.float32),
        "action_mask": mask,
    }


def channels(batch: Any) -> list[float]:
    """The value of each stacked channel, which is the frame it came from."""
    return [float(plane.reshape(-1)[0]) for plane in batch.spatial[0]]


def test_two_policies_acting_alternately_do_not_share_frames() -> None:
    """The defect, stated as the battle it would have happened in.

    Blue sees 1 then 3, red sees 2. Blue's second observation must stack 3 on its own 1, not on
    red's 2 -- and a shared list gives it 2, because red acted in between.
    """
    resolver = actors(frame_stack=2)
    spec = resolver.spec
    blue: list[np.ndarray] = []
    red: list[np.ndarray] = []

    resolver.obs_batch(observation(spec, 1.0), blue)
    resolver.obs_batch(observation(spec, 2.0), red)
    stacked = resolver.obs_batch(observation(spec, 3.0), blue)

    assert channels(stacked) == [3.0, 3.0, 1.0, 1.0], (
        "blue's stacked frame is not its own previous observation"
    )


def test_a_battle_does_not_start_inside_the_previous_one() -> None:
    """A fresh policy carries no history, so its first decision stacks zeros.

    ``EnvBattlePlayer`` builds its policies once per battle, so "fresh list" is what the next
    battle gets. The old shape kept one list for the life of the process.
    """
    resolver = actors(frame_stack=2)
    spec = resolver.spec
    first: list[np.ndarray] = []
    resolver.obs_batch(observation(spec, 1.0), first)
    resolver.obs_batch(observation(spec, 2.0), first)

    second: list[np.ndarray] = []
    stacked = resolver.obs_batch(observation(spec, 7.0), second)
    assert channels(stacked) == [7.0, 7.0, 0.0, 0.0], (
        "a new battle's first decision carried the previous battle's frames"
    )


def test_one_frame_stacks_nothing_and_keeps_no_history() -> None:
    """The shipped default, and the reason none of this has ever shown."""
    resolver = actors(frame_stack=1)
    spec = resolver.spec
    history: list[np.ndarray] = []
    stacked = resolver.obs_batch(observation(spec, 5.0), history)
    assert channels(stacked) == [5.0, 5.0]
    assert history == []


def test_the_live_learner_is_refused_when_there_is_no_model() -> None:
    """A worker process has no live model, and a snapshot would answer a different question.

    Substituting the most recent snapshot would look identical and be wrong: the probe exists
    precisely because that snapshot lags the live weights by the candidate cadence.
    """
    with pytest.raises(LookupError, match="learner"):
        actors(frame_stack=1)("learner@12500000")


def test_a_scripted_member_needs_no_model_and_no_snapshots() -> None:
    """Which is what makes a farm worker possible at all."""
    policy = actors(frame_stack=1)("scripted:noop")
    spec = actors(frame_stack=1).spec
    obs = observation(spec, 0.0)
    rng = np.random.default_rng(1)
    assert policy(obs, 0.5, rng) == 0
