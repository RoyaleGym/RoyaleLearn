"""One episode, re-simulated from its seed and checked against the trace it wrote.

``docs/harness-spec.md`` section 15 lists this file and the tree did not have it, and
``docs/design.md`` listed replay against ``verify_trace`` among the behaviours that work and are
pinned by nothing. It is the claim the whole determinism argument rests on: the battle in the run
folder is the battle that happened, and a resume or a bug report that replays one is reading the
same episode the learner learned from.

The check has to be able to fail. A verifier that walks a trace against an engine fed the trace's
own decisions will agree with itself, so the test also plants a divergence -- one tick of one
frame changed -- and asserts the verifier finds it and names where.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn.metrics.bundle import replay_episode
from royalelearn.rollout.plan import SlotPlanner
from test_coordinator import tiny_config

pytestmark = pytest.mark.slow


def _record(config: Any, battle: int = 0, ordinal: int = 0) -> Any:
    """The episode record the replay reads: which battle, which reset, and whose seed."""
    from royalelearn.api.rollout import EpisodeRecord
    from royalelearn.config import geometry

    planner = SlotPlanner(geometry(config), config.master_seed)
    return msgspec.convert(
        {
            "slot": planner.slot_of(battle, 0),
            "worker": 0,
            "shard": 0,
            "battle": battle,
            "seat": 0,
            "ordinal": ordinal,
            "episode_seed_path": planner.env_seed_path(0, 0, 0),
            "policy_id": "learner",
            "opponent_id": "learner",
            "bucket": "mirror",
            "episode_steps": 0,
            "episode_ticks": 0,
            "own_crowns": 0,
            "enemy_crowns": 0,
            "own_tower_hp_frac": 0.0,
            "enemy_tower_hp_frac": 0.0,
            "elixir_leak_steps": 0,
            "elixir_count_exact": True,
            "winner": -1,
            "outcome": 0,
            "cards_played": 0,
            "illegal_commands": 0,
            "undiscounted_return": 0.0,
            "reward_terms": {},
        },
        type=EpisodeRecord,
    )


def _replayed(tmp_path: Path, **overrides: Any) -> tuple[Any, list[str]]:
    from royalelearn.config import geometry

    config = tiny_config(tmp_path, **overrides)
    planner = SlotPlanner(geometry(config), config.master_seed)
    return replay_episode(config, _record(config), seed=planner.env_seed(0, 0, 0))


def test_a_replayed_episode_reproduces_its_own_trace(tmp_path: Path) -> None:
    """The promise an empty divergence list is: the battle in the folder is the battle that
    happened, tick for tick, on an engine that was not the one that played it."""
    trace, divergences = _replayed(tmp_path)

    assert divergences == [], divergences
    assert trace is not None
    assert len(trace.frames) > 1, "a trace of one frame proves nothing about a re-simulation"


def test_the_same_episode_replays_the_same_way_twice(tmp_path: Path) -> None:
    """Determinism of the replay itself, which is a different claim from the trace matching."""
    first, _ = _replayed(tmp_path / "one")
    second, _ = _replayed(tmp_path / "two")

    assert msgspec.json.encode(first) == msgspec.json.encode(second)


def test_the_verifier_finds_a_planted_divergence(tmp_path: Path) -> None:
    """A verifier that cannot fail is not evidence, so this plants one it has to catch.

    WHICH FIELD, because the first plant I tried was green and the code was fine. ``verify_trace``
    compares each recorded frame's ``state_hash`` against the engine's own, each step's tick
    against the engine's clock, and the deploy statuses. A frame's ``tick`` field is NOT compared:
    it is read only to name the tick in the message. So moving a frame's tick, which looks like
    the obvious corruption, alters nothing the verifier reads and it agrees -- correctly. The hash
    is the field that carries the state.
    """
    from royalegym.replay import verify_trace

    trace, divergences = _replayed(tmp_path)
    assert divergences == []

    frames = list(trace.frames)
    middle = len(frames) // 2
    zeroed = "0" * len(frames[middle].state_hash)
    altered = msgspec.structs.replace(frames[middle], state_hash=zeroed)
    planted = msgspec.structs.replace(
        trace, frames=(*frames[:middle], altered, *frames[middle + 1 :])
    )

    config = tiny_config(tmp_path / "engine")
    engine = config.env.factory(tuple(config.extra_component_modules))().engine
    found = list(verify_trace(planted, engine))

    assert found, "the verifier agreed with a trace whose state hash had been altered"
    assert any("hash" in problem for problem in found), found


def test_a_frames_tick_field_is_not_what_the_verifier_compares(tmp_path: Path) -> None:
    """Written down because it cost me a green plant, and it is a true thing about the check.

    The tick on a frame is a label for the message. The state hash is the claim. A reader who
    assumes otherwise will write a corruption test that passes for the wrong reason, which is the
    same failure as a test that has never been seen fail.
    """
    from royalegym.replay import verify_trace

    trace, _ = _replayed(tmp_path)
    frames = list(trace.frames)
    middle = len(frames) // 2
    moved = msgspec.structs.replace(frames[middle], tick=frames[middle].tick + 1)
    planted = msgspec.structs.replace(
        trace, frames=(*frames[:middle], moved, *frames[middle + 1 :])
    )

    config = tiny_config(tmp_path / "engine")
    engine = config.env.factory(tuple(config.extra_component_modules))().engine

    assert list(verify_trace(planted, engine)) == []
