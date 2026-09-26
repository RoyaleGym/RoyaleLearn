"""A run refuses to start from a checkout that cannot be reproduced from its own identity.

The identity records each repository by commit and everything else about the environment by NAME.
The reward, the observation builder and the action parser are dotted paths, hashed as strings, so
two different bodies of one function give the same ``env_spec_digest`` and the ladder pools their
games into one context. On 2026-09-22 an uncommitted change to a sibling repo sat under a live run
for twenty minutes and nothing in the identity could have said so.
"""

from __future__ import annotations

import pytest

from royalelearn.cli import build_parser, refuse_dirty_sources
from royalelearn.errors import PreflightError
from royalelearn.identity import _uncommitted, dirty_sources


def _repo(tmp_path, name: str = "repo"):
    """A real git repository with one committed file, because this reads real git output."""
    import subprocess

    root = tmp_path / name
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "code.py").write_text("x = 1\n", encoding="utf-8")
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    run("init", "-q")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "test")
    run("add", "-A")
    run("commit", "-qm", "one")
    return root


def test_a_committed_tree_has_nothing_uncommitted(tmp_path) -> None:
    assert _uncommitted(_repo(tmp_path) / "pkg") == ()


def test_a_modified_file_is_found(tmp_path) -> None:
    root = _repo(tmp_path)
    (root / "pkg" / "code.py").write_text("x = 2\n", encoding="utf-8")
    assert _uncommitted(root / "pkg") == ("pkg/code.py",)


def test_an_untracked_file_is_found(tmp_path) -> None:
    """The case ``git describe --dirty`` cannot see, and the one that bit us.

    The config a run is started from is routinely untracked, and it is the document that says
    what the run IS. The first version of this guard used describe and was blind to it.
    """
    root = _repo(tmp_path)
    (root / "pkg" / "new.py").write_text("y = 1\n", encoding="utf-8")
    assert _uncommitted(root / "pkg") == ("pkg/new.py",)


def test_a_change_elsewhere_in_the_repo_is_not_this_paths_business(tmp_path) -> None:
    """A docs edit cannot change what a run does, and a guard that fires on one gets silenced."""
    root = _repo(tmp_path)
    (root / "README.md").write_text("notes\n", encoding="utf-8")
    assert _uncommitted(root / "pkg") == ()


def test_a_path_in_no_checkout_reports_nothing(tmp_path) -> None:
    """A package installed from a wheel has no commit to be dirty against."""
    (tmp_path / "loose").mkdir()
    assert _uncommitted(tmp_path / "loose") == ()


def test_the_runs_own_config_is_watched(tmp_path) -> None:
    root = _repo(tmp_path)
    config = root / "configs" / "run.json"
    config.parent.mkdir()
    config.write_text("{}\n", encoding="utf-8")
    named = dirty_sources(config)
    assert any("run.json" in line for line in named), named


def test_a_clean_checkout_runs() -> None:
    refuse_dirty_sources(False, ())


def test_a_dirty_checkout_is_refused_and_the_message_names_it() -> None:
    with pytest.raises(PreflightError) as raised:
        refuse_dirty_sources(False, ("royalegym (2381149-dirty)",))
    message = str(raised.value)
    assert "royalegym (2381149-dirty)" in message
    assert "--allow-dirty" in message, "a refusal has to say how to proceed deliberately"


def test_every_dirty_checkout_is_named_rather_than_the_first() -> None:
    """Naming one sends the reader to commit it and hit the same wall again."""
    with pytest.raises(PreflightError) as raised:
        refuse_dirty_sources(False, ("royalelearn (a-dirty)", "royalegym (b-dirty)"))
    assert "royalelearn (a-dirty)" in str(raised.value)
    assert "royalegym (b-dirty)" in str(raised.value)


def test_allow_dirty_runs_anyway() -> None:
    """The escape hatch exists because a tree is dirty for good reasons too."""
    refuse_dirty_sources(True, ("royalegym (2381149-dirty)",))


def test_train_takes_the_flag_and_defaults_to_refusing() -> None:
    args = build_parser().parse_args(["train", "--config", "x.json"])
    assert args.allow_dirty is False
    assert build_parser().parse_args(["train", "--config", "x.json", "--allow-dirty"]).allow_dirty


def test_only_a_real_run_is_stopped() -> None:
    """The suite builds coordinators constantly and a working tree is dirty while it is worked
    on, so this guard belongs on the command that starts a run and nowhere deeper."""
    parser = build_parser()
    for command in (["doctor"], ["bench"], ["config"]):
        assert not hasattr(parser.parse_args(command), "allow_dirty")


def test_verify_resume_refuses_dirty_code_before_its_first_half(monkeypatch) -> None:
    """Its second half is a resume, which refuses uncommitted code. Asked only there, the proof
    ran for minutes and then failed as a subprocess with nothing saying why."""
    from royalelearn import cli, identity

    monkeypatch.setattr(identity, "dirty_sources", lambda *_a, **_k: ("royalegym (abc-dirty)",))
    monkeypatch.setattr(
        cli, "_coordinator", lambda *_a, **_k: pytest.fail("the first half ran on dirty code")
    )
    monkeypatch.setattr(cli, "_config_of", lambda _args: __import__("royalelearn").config.laptop())
    assert cli.main(["verify-resume", "--config", "x.json"]) == 2
    args = build_parser().parse_args(["verify-resume", "--config", "x.json", "--allow-dirty"])
    assert args.allow_dirty


def test_bench_and_verify_resume_leave_folders_of_their_own(monkeypatch) -> None:
    """Not the config's run folder: a later `train` of the same config would refuse it."""
    from royalelearn import cli

    seen: list[str] = []

    class _Stop(Exception):
        pass

    def capture(config, **_kwargs):
        seen.append(config.run_name)
        raise _Stop

    monkeypatch.setattr(cli, "_coordinator", capture)
    monkeypatch.setattr(cli, "_config_of", lambda _args: __import__("royalelearn").config.laptop())
    monkeypatch.setattr("royalelearn.identity.dirty_sources", lambda *_a, **_k: ())
    for argv in (["bench"], ["verify-resume", "--config", "x.json"]):
        with pytest.raises(_Stop):
            cli.main(argv)
    assert seen[0].startswith("bench-") and seen[1].startswith("verify-resume-")
