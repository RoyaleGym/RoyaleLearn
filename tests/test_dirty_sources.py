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


def test_a_clean_checkout_runs() -> None:
    refuse_dirty_sources(False, ())


def test_a_dirty_checkout_is_refused_and_the_message_names_it() -> None:
    with pytest.raises(PreflightError) as raised:
        refuse_dirty_sources(False, ("royalegym (4488a2f-dirty)",))
    message = str(raised.value)
    assert "royalegym (4488a2f-dirty)" in message
    assert "--allow-dirty" in message, "a refusal has to say how to proceed deliberately"


def test_every_dirty_checkout_is_named_rather_than_the_first() -> None:
    """Naming one sends the reader to commit it and hit the same wall again."""
    with pytest.raises(PreflightError) as raised:
        refuse_dirty_sources(False, ("royalelearn (a-dirty)", "royalegym (b-dirty)"))
    assert "royalelearn (a-dirty)" in str(raised.value)
    assert "royalegym (b-dirty)" in str(raised.value)


def test_allow_dirty_runs_anyway() -> None:
    """The escape hatch exists because a tree is dirty for good reasons too."""
    refuse_dirty_sources(True, ("royalegym (4488a2f-dirty)",))


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
