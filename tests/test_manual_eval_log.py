"""A command that asks a question does not edit the thing it is asking about.

``eval`` and ``gate`` play real battles, and the runner writes every one into a result log. Pointed
at a run, that is the run's OWN ladder: the games would join the fit, move the ratings and count
towards ``ladder/eval_games_total`` in a run nobody asked to change. The command reads as a
question, so nothing about running it says an edit happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from royalelearn.cli import _side_log, build_parser


class _Runner:
    """Stands in for the coordinator's evaluation runner, which is all _side_log touches."""

    def __init__(self, log: Any) -> None:
        self.log = log


class _Run:
    def __init__(self, log: Any) -> None:
        self.eval_runner = _Runner(log)


def test_by_default_the_games_go_beside_the_ladder_and_not_into_it(tmp_path: Path) -> None:
    ladder = object()
    run = _Run(ladder)

    side = _side_log(run, tmp_path, record=False)

    assert side == tmp_path / "ladder" / "manual-eval.jsonl"
    assert run.eval_runner.log is not ladder
    assert Path(run.eval_runner.log.path).name == "manual-eval.jsonl"


def test_record_puts_them_in_the_runs_own_ladder(tmp_path: Path) -> None:
    """The dangerous case is the one that has to be visible in the command line.

    ``--record`` is a thing a reviewer can see; "I ran eval against the live run" is not.
    """
    ladder = object()
    run = _Run(ladder)

    assert _side_log(run, tmp_path, record=True) is None
    assert run.eval_runner.log is ladder


def test_both_commands_take_the_flag_and_default_to_not_recording() -> None:
    parser = build_parser()
    evaluated = parser.parse_args(["eval", "--run", "r", "--a", "one", "--b", "two"])
    gated = parser.parse_args(["gate", "--run", "r", "--candidate", "snap:v1"])
    assert evaluated.record is False
    assert gated.record is False
    assert parser.parse_args(
        ["eval", "--run", "r", "--a", "one", "--b", "two", "--record"]
    ).record
