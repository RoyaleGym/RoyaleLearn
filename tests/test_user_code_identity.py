"""A user who edits their own reward between two runs used to get two runs with one identity.

The identity names every component by its dotted path and hashes the path as a string. For code
in royalelearn, royalegym and royaleviser that is backed by each package's commit. For a user's
own module -- the reason ``extra_component_modules`` exists -- nothing backed it: a different body
of ``mybot.rewards.MyReward`` gave the same ``env_spec_digest``, the same run id, and a ladder
that pooled two objectives as one. ``dirty_sources`` had the same blind spot one folder over: it
watched the three packages, the engine's data and the config, and not the user's code.

So the identity now records the SOURCE of every user package a component comes from, by content,
and ``train`` and ``resume`` refuse an uncommitted edit to it the way they refuse one to a
sibling repo. Content rather than commit, because a user's module is often in no repository at
all, and a hash of what is on disk is the one answer that exists either way.
"""

from __future__ import annotations

import importlib
import itertools
from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn import config as C
from royalelearn import identity as I
from royalelearn.api.rollout import EnvSpec
from royalelearn.checkpoint import check_resume
from royalelearn.errors import IdentityMismatch
from royalelearn.rollout.envspec import ComponentSpec

_names = itertools.count()


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    """A user package on the path, with a reward and a helper it imports.

    A fresh name every time, because the import system caches by name and a test that reuses
    one would read the previous test's files.
    """
    name = f"userbot{next(_names)}"
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "util.py").write_text("SCALE = 1.0\n", encoding="utf-8")
    (root / "rewards.py").write_text(
        "from .util import SCALE\n\nclass MyReward:\n    scale = SCALE\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return name, root


def _config(name: str, *, via_kwargs: bool = False, listed: bool = True) -> C.RunConfig:
    base = C.default_env_spec(C.MOCK_ENGINE)
    if via_kwargs:
        # A royalelearn component whose kwargs name the user's class: the user's code is only
        # reachable through a string, which is how a composition refers to its parts.
        reward = ComponentSpec(
            "royalelearn.rewards.default_potential_reward",
            {"note": f"{name}.rewards.MyReward"},
        )
    else:
        reward = ComponentSpec(f"{name}.rewards.MyReward")
    env = msgspec.structs.replace(base, reward_fn=reward)
    return C.RunConfig(env=env, extra_component_modules=[name] if listed else [])


@pytest.fixture
def facts(env_spec: EnvSpec) -> dict[str, Any]:
    return {
        "env_spec": env_spec,
        "build": I.EngineBuild(
            engine_class="royalegym.mock_engine.MockEngine",
            calibration_digest="0" * 16,
            build_digest="0" * 16,
            catalogue_sha256="c" * 64,
            path_search=None,
            stale_build_differences=[],
            binary_sha256=I.NOT_STATED,
        ),
        "arch_digest": "a" * 64,
        "codec_version": 1,
        "codec_table_digest": "b" * 64,
        "torch_version_string": "2.11.0+cu128",
        "device_kind": "cpu:x86_64",
    }


def _identity(config: C.RunConfig, facts: dict[str, Any]) -> I.RunIdentity:
    return I.compute_identity(config, **facts)


# -- what counts as the user's code ------------------------------------------------------------


def test_a_user_component_s_package_is_recorded_by_content(package: Any, facts: Any) -> None:
    name, _ = package
    recorded = _identity(_config(name), facts).user_code
    assert set(recorded) == {name}
    assert len(recorded[name]) == 64


def test_the_packages_own_components_are_not_user_code(facts: Any) -> None:
    """royalelearn, royalegym and royaleviser are recorded by commit already."""
    assert _identity(C.RunConfig(env=C.default_env_spec(C.MOCK_ENGINE)), facts).user_code == {}


def test_listing_a_module_that_no_component_uses_changes_nothing(package: Any, facts: Any) -> None:
    """The identity table excludes ``extra_component_modules``: a list cannot move a number.

    The code a component actually comes from can, and that is what is recorded. A module that is
    only listed is not.
    """
    name, _ = package
    config = C.RunConfig(env=C.default_env_spec(C.MOCK_ENGINE), extra_component_modules=[name])
    assert _identity(config, facts).user_code == {}


def test_a_user_class_named_in_a_component_s_kwargs_is_found(package: Any, facts: Any) -> None:
    name, _ = package
    assert set(_identity(_config(name, via_kwargs=True), facts).user_code) == {name}


# -- an edit is a different run ----------------------------------------------------------------


def test_editing_the_reward_is_a_different_run(package: Any, facts: Any) -> None:
    name, root = package
    before = _identity(_config(name), facts)
    (root / "rewards.py").write_text(
        "from .util import SCALE\n\nclass MyReward:\n    scale = 2 * SCALE\n", encoding="utf-8"
    )
    after = _identity(_config(name), facts)

    assert I.run_id(before) != I.run_id(after)
    assert set(I.identity_differences(before, after)) == {"user_code"}


def test_editing_a_helper_the_reward_imports_is_a_different_run(package: Any, facts: Any) -> None:
    """The reward file is unchanged; the number it reads is not."""
    name, root = package
    before = _identity(_config(name), facts)
    (root / "util.py").write_text("SCALE = 3.0\n", encoding="utf-8")
    assert I.run_id(_identity(_config(name), facts)) != I.run_id(before)


def test_a_cache_file_or_a_note_does_not_move_it(package: Any, facts: Any) -> None:
    """Python source only, as stated: bytecode and prose cannot change what runs."""
    name, root = package
    before = _identity(_config(name), facts).user_code
    # Computing the identity imports the package, so Python may already have made this.
    (root / "__pycache__").mkdir(exist_ok=True)
    (root / "__pycache__" / "rewards.cpython-312.pyc").write_bytes(b"\x00\x01")
    (root / "NOTES.md").write_text("tried 0.3\n", encoding="utf-8")
    assert _identity(_config(name), facts).user_code == before


def test_line_endings_do_not_make_two_runs(package: Any, facts: Any) -> None:
    """The same code checked out on Windows and on Linux is one run."""
    name, root = package
    # Bytes, both ways: ``write_text`` already translates newlines on Windows, so a test that
    # edited its output would compare CRLF with CR-CR-LF and measure itself.
    (root / "util.py").write_bytes(b"SCALE = 1.0\nOTHER = 2\n")
    unix = _identity(_config(name), facts).user_code
    (root / "util.py").write_bytes(b"SCALE = 1.0\r\nOTHER = 2\r\n")
    assert _identity(_config(name), facts).user_code == unix


# -- resuming across the change ----------------------------------------------------------------


def test_an_identity_from_before_the_field_resumes_and_says_it_could_not_check(
    package: Any, facts: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    name, _ = package
    now = _identity(_config(name), facts)
    raw = msgspec.json.decode(msgspec.json.encode(now))
    del raw["user_code"]
    old = msgspec.convert(raw, I.RunIdentity)
    assert old.user_code is None

    from types import SimpleNamespace

    check_resume(SimpleNamespace(identity=old, config={}), now, {})
    said = capsys.readouterr().out
    assert "your code" in said and "cannot" in said, said


def test_a_resume_after_the_user_edited_their_reward_is_refused(package: Any, facts: Any) -> None:
    from types import SimpleNamespace

    name, root = package
    before = _identity(_config(name), facts)
    (root / "rewards.py").write_text("class MyReward:\n    scale = 9.0\n", encoding="utf-8")
    with pytest.raises(IdentityMismatch) as refused:
        check_resume(
            SimpleNamespace(identity=before, config={}), _identity(_config(name), facts), {}
        )
    assert "user_code" in refused.value.differences


# -- uncommitted edits ---------------------------------------------------------------------


def _git(root: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)


def test_an_uncommitted_edit_to_user_code_is_named(package: Any) -> None:
    """The same refusal a sibling repo gets, one folder over."""
    name, root = package
    repo = root.parent
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "one")
    config = _config(name)
    assert not [line for line in I.dirty_sources(config=config) if name in line]

    (root / "rewards.py").write_text("class MyReward:\n    scale = 5.0\n", encoding="utf-8")

    named = [line for line in I.dirty_sources(config=config) if name in line]
    assert named and "rewards.py" in named[0], I.dirty_sources(config=config)


def test_resume_can_be_told_to_run_dirty_the_way_train_can() -> None:
    """The refusal's docstring said it lived on train AND resume. Resume had neither the check
    nor the flag."""
    from royalelearn.cli import build_parser

    args = build_parser().parse_args(["resume", "--run", "runs/x", "--allow-dirty"])
    assert args.allow_dirty is True


def test_resume_refuses_uncommitted_user_code(
    package: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checked through the command itself, so the wiring is what is tested."""
    from royalelearn import cli
    from royalelearn.errors import PreflightError

    name, _ = package
    config = _config(name)
    monkeypatch.setattr(cli, "_run_config", lambda run: config)
    # The command imports it from the identity module when it runs, so that is where to stand in.
    monkeypatch.setattr(
        I, "dirty_sources", lambda config_path=None, config=None: (f"your code ({name}): x",)
    )
    args = cli.build_parser().parse_args(["resume", "--run", "runs/x"])
    with pytest.raises(PreflightError, match=name):
        args.handler(args)
