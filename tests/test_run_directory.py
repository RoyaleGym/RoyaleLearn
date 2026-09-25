"""A fresh run into a directory that already holds a run is refused, not written over.

A run's directory is ``<runs_dir>/<run_name>-<run_id>``, and the run id follows the config. So
launching the same config fresh a second time lands in the SAME directory, and it used to start
at iteration 1 and write over the checkpoints already there. train-hog26-10's directory shows it:
metric rows for iterations 1-610, then 1-45, then 1-21, then 611-621, and a checkpoint index that
mixes launches. Found by the train session's evaluation of that run, 2026-09-24.

There is no flag to allow it. Starting fresh into an occupied directory cannot be made safe, only
destructive, and the two things a person could want are both still one command away: resume the
run that is there, or start a separate one under another name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from royalelearn.errors import PreflightError


def _config(tmp_path: Path) -> Any:
    from test_coordinator import tiny_config

    return tiny_config(tmp_path)


def test_a_fresh_run_into_a_directory_that_holds_a_run_is_refused(tmp_path: Path) -> None:
    from test_coordinator import coordinator

    config = _config(tmp_path)
    with coordinator(config) as first:
        first.iterate()
        first.checkpoint()
        run_dir = first.run_dir
    index = run_dir / "checkpoints" / "index.json"
    before = index.read_bytes()
    rows_before = (run_dir / "metrics.jsonl").read_bytes()

    with pytest.raises(PreflightError) as refused, coordinator(config):
        pass
    message = str(refused.value)
    assert "resume" in message and str(run_dir.name) in message, message

    assert index.read_bytes() == before, "the refused start still touched the checkpoints"
    assert (run_dir / "metrics.jsonl").read_bytes() == rows_before


def test_resuming_the_run_that_is_there_still_works(tmp_path: Path) -> None:
    """The control: the refusal is about a FRESH start, not about opening the directory."""
    from test_coordinator import coordinator

    config = _config(tmp_path)
    with coordinator(config) as first:
        first.iterate()
        saved = first.checkpoint()
        run_dir = first.run_dir
    with coordinator(config, resume=saved, run_dir=run_dir) as again:
        assert again.iteration == 1


def test_an_empty_directory_is_not_a_run(tmp_path: Path) -> None:
    """A directory made ahead of time, with nothing in it, is somewhere to start."""
    from test_coordinator import coordinator

    target = tmp_path / "prepared"
    target.mkdir()
    with coordinator(_config(tmp_path), run_dir=target) as run:
        run.iterate()
