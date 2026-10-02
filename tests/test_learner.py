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


def test_a_build_env_in_code_run_by_exec_trains(tmp_path: Path) -> None:
    """A docs checker or an IDE runner executes a page's code in a namespace named ``__main__``
    that is not the ``__main__`` module, where a run looking ``build_env`` up by its path would
    not find it."""
    import os
    import subprocess
    import sys

    runner = tmp_path / "runner.py"
    runner.write_text(
        "import sys\n"
        "code = open(sys.argv[1], encoding='utf-8').read()\n"
        "exec(compile(code, '<block>', 'exec'), {'__name__': '__main__'})\n",
        encoding="utf-8",
    )
    (tmp_path / "block.py").write_text(SCRIPT, encoding="utf-8")
    import royalelearn

    here = str(Path(royalelearn.__file__).resolve().parents[1])
    path = os.pathsep.join(p for p in (here, os.environ.get("PYTHONPATH", "")) if p)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": path}
    done = subprocess.run(
        [sys.executable, str(runner), "block.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "DONE" in done.stdout, done.stdout[-2000:]


# -- the named settings, rlgym-ppo style -------------------------------------------------------


def test_the_defaults_build_the_config_they_always_built(tmp_path: Path) -> None:
    from royalelearn import config as cfg

    plain = Learner(build_env, save_dir="runs/pin", device="cpu", threads=2)
    assert cfg.config_hash(plain.config).startswith("8bcb1242ae4f")
    other = Learner(
        build_env,
        save_dir="runs/pin",
        device="cpu",
        threads=2,
        opponent="self",
        steps_per_update=2048,
        checkpoint_every=10000,
    )
    assert cfg.config_hash(other.config).startswith("77db39c799d0")


def test_each_named_setting_reaches_the_run(tmp_path: Path) -> None:
    from royalelearn import config as cfg

    config = Learner(
        build_env,
        save_dir=tmp_path / "a",
        device="cpu",
        trunk_channels=48,
        trunk_blocks=3,
        critic_hidden=96,
        policy_lr=1e-4,
        critic_lr=3e-4,
        ppo_epochs=5,
        ppo_batch_size=256,
        ppo_minibatch_size=64,
        ppo_ent_coef=0.02,
        ppo_clip_range=0.1,
        gae_lambda=0.95,
        gae_gamma=0.995,
        standardize_returns=False,
        n_checkpoints_to_keep=7,
    ).config
    net, ppo, advantage = config.net, config.ppo, config.advantage
    assert (net.channels, net.card_embed, net.blocks, net.value_hidden) == (48, 48, 3, 96)
    assert (ppo.lr_actor, ppo.lr_critic, ppo.n_epochs) == (1e-4, 3e-4, 5)
    assert (ppo.batch_size, ppo.minibatch_size, ppo.clip_range) == (256, 64, 0.1)
    assert ppo.ent_coef == cfg.ConstantSpec(0.02)
    assert advantage.gae_lambda == 0.95 and advantage.gamma == cfg.ConstantSpec(0.995)
    assert advantage.standardize_rewards is False
    assert config.checkpoint.keep == 7
    with pytest.raises(PreflightError):
        _ = Learner(
            build_env, save_dir=tmp_path / "b", ppo_batch_size=100, ppo_minibatch_size=64
        ).config


def test_log_to_wandb_names_the_project_group_and_run(tmp_path: Path) -> None:
    sinks = Learner(
        build_env,
        save_dir=tmp_path / "a",
        log_to_wandb=True,
        wandb_project_name="my-bots",
        wandb_group_name="hog",
        wandb_run_name="first",
    ).config.metrics.sinks
    (wandb,) = [s for s in sinks if s.kind == "wandb"]
    assert wandb.enabled
    assert wandb.options == {"project": "my-bots", "group": "hog", "name": "first"}
    plain = Learner(build_env, save_dir=tmp_path / "b").config.metrics.sinks
    assert not [s for s in plain if s.kind == "wandb"]


def test_the_wandb_sink_hands_group_and_name_to_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    from royalelearn.metrics.sinks import CompositeSink
    from royalelearn.metrics.wandb_sink import WandbSink

    seen: dict[str, object] = {}
    fake = types.ModuleType("wandb")
    fake.init = lambda **kwargs: seen.update(kwargs) or types.SimpleNamespace(id="x")
    monkeypatch.setitem(sys.modules, "wandb", fake)
    sink = WandbSink(CompositeSink([]), enable=True, project="p", group="g", name="n")
    sink.open(identity={"run_id": "r"}, config_json="{}", run_dir=tmp_path)  # type: ignore[arg-type]
    assert (seen["project"], seen["group"], seen["name"]) == ("p", "g", "n")


def test_timestep_limit_is_where_learn_stops_when_it_is_given_no_total(tmp_path: Path) -> None:
    learner = Learner(
        build_env, n_envs=2, device="cpu", save_dir=tmp_path / "a", timestep_limit=32, **TINY
    )
    learner.learn()
    assert learner.steps >= 32
    with pytest.raises(PreflightError, match="timestep_limit"):
        Learner(build_env, device="cpu", save_dir=tmp_path / "b", **TINY).learn()


def test_a_run_carries_on_with_changed_training_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A learning rate or an epoch count changed between two runs of one ``save_dir`` is the
    same bot trained on; the change is printed."""
    save = tmp_path / "run"
    Learner(build_env, n_envs=2, device="cpu", save_dir=save, **TINY).learn(total_steps=32)
    capsys.readouterr()
    again = Learner(
        build_env, n_envs=2, device="cpu", save_dir=save, policy_lr=1e-4, ppo_epochs=2, **TINY
    )
    again.learn(total_steps=64)
    assert again.steps >= 64 and again.resumed_from is not None
    out = capsys.readouterr().out
    assert "training settings changed" in out and "lr_actor" in out


def test_a_run_with_another_network_is_refused_in_words(tmp_path: Path) -> None:
    save = tmp_path / "run"
    Learner(build_env, n_envs=2, device="cpu", save_dir=save, **TINY).learn(total_steps=32)
    wider = Learner(build_env, n_envs=2, device="cpu", save_dir=save, trunk_channels=16, **TINY)
    with pytest.raises(PreflightError, match="another save_dir") as refused:
        wider.learn(total_steps=64)
    assert "arch_digest" in str(refused.value)
