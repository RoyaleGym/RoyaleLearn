"""A saved bot's folder says what ``build_env`` built, and the environment rebuilds from it.

``environment.json`` is ``env.config()`` of the env ``build_env`` returns: every component's class
and settings, the decision rate, the engine's cards. ``learn`` writes it into the run's folder,
``save`` beside the bot, and an add-on that saves a bot of its own (RoyaleImitate's ``clone``)
writes it through ``royalelearn.extensions.write_environment_record``.

``Learner.load_env(folder)`` builds the env back from that record alone, with no script run: only
RoyaleGym's and RoyaleLearn's own classes, each of the kind its place takes, and only when the
rebuilt env describes itself as the record does. A part it cannot rebuild is named, and can be
handed in instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import Learner
from royalelearn.errors import PreflightError
from royalelearn.testing import PREFLIGHT

pytest.importorskip("torch")

TINY = {"steps_per_update": 16, "_coordinator_kwargs": {"preflight_kwargs": PREFLIGHT}}
HOG = ["HogRider", "Musketeer", "Cannon", "Fireball", "Log", "Skeletons", "Knight", "Archer"]


def build_env():
    from royalegym import ClashParallelEnv, MockEngine
    from royalegym.done_condition import GameOverCondition, StepLimitCondition

    return ClashParallelEnv(
        MockEngine(), termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(6)
    )


def build_env_of_my_own():
    """Every part set to something other than its default."""
    from royalegym import ClashParallelEnv, CrownReward, MockEngine, Reveal
    from royalegym.action import TileActionParser
    from royalegym.done_condition import GameOverCondition, StepLimitCondition
    from royalegym.obs import SpatialObsBuilder
    from royalegym.state_mutator import DefaultStateMutator

    engine = MockEngine()
    return ClashParallelEnv(
        engine,
        obs_builder=SpatialObsBuilder(reveal=Reveal(enemy_elixir=True)),
        action_parser=TileActionParser(hold_while_pending=True),
        reward_fn=CrownReward(),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(9),
        state_mutator=DefaultStateMutator(decks=[HOG, HOG]),
        decision_ms=250,
    )


def _described(env: Any) -> dict[str, Any]:
    """``env.config()`` as it reads back from JSON, then the env closed."""
    try:
        return msgspec.json.decode(msgspec.json.encode(env.config()))
    finally:
        env.close()


def _saved(folder: Path, env_fn: Any) -> Path:
    """A folder holding only the environment record, as an add-on would write it."""
    from royalelearn.extensions import write_environment_record

    learner = Learner(env_fn, n_envs=2, device="cpu", save_dir=folder / "unused", **TINY)
    write_environment_record(folder, learner.environment)
    return folder


def test_save_writes_what_build_env_built_and_it_rebuilds(tmp_path: Path) -> None:
    """``save`` writes the record beside the bot; the run's folder and its checkpoints hold it
    too; each rebuilds an env that the bot plays in."""
    from royalelearn.checkpoint import DirCheckpointStore

    run = tmp_path / "run"
    learner = Learner(build_env, n_envs=2, device="cpu", save_dir=run, **TINY)
    learner.learn(total_steps=16)
    bot_folder = learner.save(tmp_path / "bot")
    record = msgspec.json.decode((bot_folder / "environment.json").read_bytes())
    assert record == _described(build_env())
    assert record == learner.environment

    latest = DirCheckpointStore(run).latest()
    assert latest is not None
    bot = Learner.load_policy(bot_folder, greedy=True)
    for where in (bot_folder, run, latest):
        env = Learner.load_env(where)
        try:
            assert _described_open(env) == record
            obs, _ = env.reset(seed=3)
            assert bool(np.asarray(obs["blue"]["action_mask"])[bot(obs["blue"])])
        finally:
            env.close()


def _described_open(env: Any) -> dict[str, Any]:
    return msgspec.json.decode(msgspec.json.encode(env.config()))


def test_every_part_a_user_sets_rebuilds_as_it_was(tmp_path: Path) -> None:
    folder = _saved(tmp_path / "bot", build_env_of_my_own)
    record = msgspec.json.decode((folder / "environment.json").read_bytes())
    assert record == _described(build_env_of_my_own())
    assert record["decision_ms"] == 250 and record["reveal"]["enemy_elixir"] is True
    env = Learner.load_env(folder)
    try:
        assert _described_open(env) == record
    finally:
        env.close()


def _rewrite(folder: Path, change: Any) -> None:
    path = folder / "environment.json"
    record = msgspec.json.decode(path.read_bytes())
    change(record)
    path.write_bytes(msgspec.json.encode(record))


def test_a_class_from_outside_royalegym_is_never_imported(tmp_path: Path) -> None:
    """The record is a member's data: a class path in it is not code to run. Plant: resolve the
    path before checking where it is from and the module below is imported."""
    from royalegym import CrownReward

    folder = _saved(tmp_path / "bot", build_env)
    # A real module that nothing here imports: a path to one that did not exist could never be
    # imported, and the check below would hold whatever load_env did.
    if "tabnanny" in sys.modules:
        pytest.skip("tabnanny is already imported in this process")

    def theirs(record: dict[str, Any]) -> None:
        record["reward_fn"] = {"class": "tabnanny.NannyNag", "params": {}}

    _rewrite(folder, theirs)
    with pytest.raises(PreflightError, match=r"reward_fn=") as refused:
        Learner.load_env(folder)
    assert "tabnanny" not in sys.modules
    assert "the reward" in str(refused.value)
    # Handed in, it is used as it is, and the rest is still rebuilt and checked.
    env = Learner.load_env(folder, reward_fn=CrownReward())
    try:
        assert type(env.reward_fn) is CrownReward
    finally:
        env.close()


def test_only_a_class_of_the_kind_its_place_takes_is_built(tmp_path: Path) -> None:
    """A RoyaleGym callable that is not a reward is not called as one. Plant: drop the kind check
    and the record below has load_env call ``ReplayRecorder`` with the record's arguments."""
    folder = _saved(tmp_path / "bot", build_env)

    def elsewhere(record: dict[str, Any]) -> None:
        record["reward_fn"] = {"class": "royalegym.replay.ReplayRecorder", "params": {}}

    _rewrite(folder, elsewhere)
    with pytest.raises(PreflightError, match="not a reward"):
        Learner.load_env(folder)


def test_a_part_that_rebuilds_differently_is_refused(tmp_path: Path) -> None:
    """Plant: skip the comparison and a setting the constructor does not take is dropped
    without a word."""
    folder = _saved(tmp_path / "bot", build_env)

    def unknown(record: dict[str, Any]) -> None:
        record["obs_builder"]["params"]["a_setting_from_a_newer_royalegym"] = 3

    _rewrite(folder, unknown)
    with pytest.raises(PreflightError, match="the observation"):
        Learner.load_env(folder)


def test_another_engine_build_is_noted_and_the_env_still_built(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The installed engine is the checker's own: its data stamps may differ from the saver's.
    The cards may not, since a card id is a position in them."""
    folder = _saved(tmp_path / "bot", build_env)

    def older(record: dict[str, Any]) -> None:
        record["engine"]["params"]["calibration_digest"] = "an older calibration"
        record["calibration_digest"] = "an older calibration"

    _rewrite(folder, older)
    env = Learner.load_env(folder)
    env.close()
    out = capsys.readouterr().out
    assert "calibration_digest" in out and "engine" in out, out

    def fewer_cards(record: dict[str, Any]) -> None:
        record["engine"]["params"]["cards"] = record["engine"]["params"]["cards"][:-1]

    _rewrite(folder, fewer_cards)
    env = Learner.load_env(folder)
    try:
        assert len(_described_open(env)["engine"]["params"]["cards"]) == 15
    finally:
        env.close()


def test_a_folder_without_a_record_says_how_to_get_one(tmp_path: Path) -> None:
    (tmp_path / "bot").mkdir()
    with pytest.raises(PreflightError, match=r"environment\.json.*learner\.save"):
        Learner.load_env(tmp_path / "bot")


def test_a_record_written_before_the_engine_was_in_it_still_rebuilds(tmp_path: Path) -> None:
    """Runs since 0.4.3 hold a record of the parts a user edits, without the engine: the engine
    comes from ``policy.json``, with its default cards, and a line says so."""
    from royalelearn.learner import ENVIRONMENT_PARTS

    run = tmp_path / "run"
    learner = Learner(build_env, n_envs=2, device="cpu", save_dir=run, **TINY)
    learner.learn(total_steps=16)
    full = learner.environment
    (run / "environment.json").write_bytes(
        msgspec.json.encode({key: full[key] for key in ENVIRONMENT_PARTS})
    )
    env = Learner.load_env(run)
    try:
        assert _described_open(env)["obs_builder"] == full["obs_builder"]
        assert type(env.engine).__name__ == "MockEngine"
    finally:
        env.close()
