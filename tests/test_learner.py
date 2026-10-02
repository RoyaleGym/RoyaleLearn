"""The one-call trainer: ``Learner(build_env, ...).learn(total_steps=...)``.

The shape the quickstart uses, on MockEngine and a step budget a test can afford: it trains,
checkpoints into ``save_dir``, carries on from there when made again, saves a bot that plays an
environment's own observation, and refuses what it cannot run, saying what to do instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from royalelearn import Learner
from royalelearn.errors import PreflightError
from royalelearn.testing import PREFLIGHT

TINY = {"steps_per_update": 16, "_coordinator_kwargs": {"preflight_kwargs": PREFLIGHT}}


def build_env():
    """A battle on MockEngine, the env's own reward: what a user's ``make_env`` returns."""
    from royalegym import ClashParallelEnv, MockEngine
    from royalegym.done_condition import GameOverCondition, StepLimitCondition

    return ClashParallelEnv(
        MockEngine(),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(6),
    )


def test_the_quickstart_shape_trains_checkpoints_and_saves_a_bot(tmp_path: Path) -> None:
    save = tmp_path / "quickstart"
    learner = Learner(build_env, n_envs=2, device="cpu", save_dir=save, **TINY)
    learner.learn(total_steps=48)
    assert learner.steps >= 48
    assert any((save / "checkpoints").iterdir()), "no checkpoint was written into save_dir"

    learner.save(tmp_path / "bot")
    bot = Learner.load_policy(tmp_path / "bot")
    env = build_env()
    obs, _ = env.reset(seed=3)
    for _ in range(5):
        action = bot(obs["blue"])
        assert isinstance(action, int)
        legal = np.asarray(obs["blue"]["action_mask"])
        assert bool(legal[action]), "the bot played an illegal move"
        obs, _, terminated, truncated, _ = env.step({"blue": action, "red": 0})
        if terminated["blue"] or truncated["blue"]:
            break


def test_the_same_save_dir_carries_the_run_on(tmp_path: Path) -> None:
    save = tmp_path / "run"
    first = Learner(build_env, n_envs=2, device="cpu", save_dir=save, **TINY)
    first.learn(total_steps=32)
    again = Learner(build_env, n_envs=2, device="cpu", save_dir=save, **TINY)
    again.learn(total_steps=64)
    assert again.steps >= 64 and again.resumed_from is not None


def test_a_fresh_start_on_a_used_save_dir_is_refused_with_the_way_out(tmp_path: Path) -> None:
    save = tmp_path / "run"
    Learner(build_env, n_envs=2, device="cpu", save_dir=save, **TINY).learn(total_steps=32)
    with pytest.raises(PreflightError, match=r"resume=True|another save_dir"):
        Learner(build_env, n_envs=2, save_dir=save, resume=False, **TINY).learn(total_steps=64)


def test_an_env_function_the_workers_cannot_import_is_refused(tmp_path: Path) -> None:
    def local_env():  # pragma: no cover - never called
        return build_env()

    for fn in (local_env, lambda: build_env()):
        with pytest.raises(PreflightError, match="top level"):
            Learner(fn, save_dir=tmp_path / "x")


def test_the_opponent_names(tmp_path: Path) -> None:
    random = Learner(build_env, save_dir=tmp_path / "a", opponent="random").config.ladder
    assert random.mix == (0.0, 0.0, 1.0)
    assert random.scripted_opponents == ("scripted:random_legal",)
    assert Learner(build_env, save_dir=tmp_path / "b", opponent="noop").config.ladder.mix == (
        0.0,
        0.0,
        1.0,
    )
    selfplay = Learner(build_env, save_dir=tmp_path / "c", opponent="self").config.ladder
    assert selfplay.mix[0] > 0.0 and selfplay.mix[1] > 0.0
    with pytest.raises(PreflightError, match="opponent"):
        Learner(build_env, save_dir=tmp_path / "d", opponent="nobody")


SCRIPT = """
from royalegym import ClashParallelEnv, MockEngine
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalelearn import Learner
from royalelearn.testing import PREFLIGHT


def build_env():
    return ClashParallelEnv(
        MockEngine(), termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(6)
    )


if __name__ == "__main__":
    learner = Learner(build_env, n_envs=2, device="cpu", save_dir="runs/script",
                      steps_per_update=16, _coordinator_kwargs={"preflight_kwargs": PREFLIGHT})
    learner.learn(total_steps=32)
    learner.save("runs/script/bot")
    print("DONE", learner.steps)
"""


def test_a_build_env_in_the_users_own_script_trains(tmp_path: Path) -> None:
    """The quickstart's shape: ``build_env`` lives in the script that is run, ``__main__``."""
    import os
    import subprocess
    import sys

    script = tmp_path / "my_bot.py"
    script.write_text(SCRIPT, encoding="utf-8")
    import royalelearn

    here = str(Path(royalelearn.__file__).resolve().parents[1])
    path = os.pathsep.join(p for p in (here, os.environ.get("PYTHONPATH", "")) if p)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": path}
    done = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, env=env
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "DONE" in done.stdout, done.stdout[-2000:]
    progress = [line.split() for line in done.stdout.splitlines() if line.startswith("update ")]
    assert progress and progress[0][:3] == ["update", "1", "steps"], done.stdout[-2000:]
    assert len(done.stdout.splitlines()) < 20, "a first run should print a line per update"
    assert (tmp_path / "runs" / "script" / "bot" / "policy.json").is_file()


def test_the_brief_console_line() -> None:
    from royalelearn.metrics.sinks import brief_line

    line = brief_line(
        {
            "run/iteration": 3,
            "run/cumulative_timesteps": 3072,
            "env/episodes_completed": 7,
            "env/crown_diff": 0.5,
            "time/iteration": 21.4,
        }
    )
    assert line.split() == [
        "update",
        "3",
        "steps",
        "3,072",
        "battles",
        "7",
        "crowns",
        "+0.50",
        "a",
        "battle",
        "21",
        "s",
    ]
    assert "-" in brief_line({"run/iteration": 1})


def test_the_env_function_is_in_the_identity_and_nothing_else_moves(tmp_path: Path) -> None:
    import msgspec

    from royalelearn import config as cfg
    from royalelearn.rollout.envspec import env_value_digest

    plain = cfg.default_env_spec(cfg.MOCK_ENGINE)
    assert env_value_digest(msgspec.structs.replace(plain, env_fn=None)) == env_value_digest(plain)
    named = msgspec.structs.replace(plain, env_fn="test_learner.build_env")
    other = msgspec.structs.replace(plain, env_fn="test_learner.other_env")
    assert len({env_value_digest(plain), env_value_digest(named), env_value_digest(other)}) == 3


def test_an_add_ons_section_is_passed_through_and_an_unknown_one_named(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="no_such_section"):
        _ = Learner(
            build_env, save_dir=tmp_path / "x", extensions={"no_such_section": {}}
        ).config


def test_the_finished_run_is_kept_for_add_ons(tmp_path: Path) -> None:
    learner = Learner(build_env, n_envs=2, device="cpu", save_dir=tmp_path / "run", **TINY)
    assert learner.run is None
    learner.learn(total_steps=32)
    run = learner.run
    assert run is not None and run.model is not None
    codec = run.row_codec()
    assert codec.spec.n_actions == run.spec.n_actions
    assert run.snapshot_template.arch_digest == run.arch_digest


def test_a_cpu_run_uses_several_cores_for_the_network(tmp_path: Path) -> None:
    import os

    from royalelearn.learner import default_threads

    cores = os.cpu_count() or 1
    assert 1 <= default_threads() <= max(1, cores // 2)
    config = Learner(build_env, save_dir=tmp_path / "a").config
    assert config.determinism.torch_threads == default_threads()
    three = Learner(build_env, save_dir=tmp_path / "b", threads=3)
    assert three.config.determinism.torch_threads == 3
    with pytest.raises(PreflightError, match="threads"):
        Learner(build_env, save_dir=tmp_path / "c", threads=0)


def test_auto_is_the_gpu_when_torch_sees_one_and_else_the_cpu_saying_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import torch

    from royalelearn.learner import NO_GPU_LINE, resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert capsys.readouterr().out.strip() == NO_GPU_LINE
    assert resolve_device("cpu") == "cpu" and resolve_device("cuda") == "cuda"
    assert Learner.__init__.__kwdefaults__["device"] == "auto"
