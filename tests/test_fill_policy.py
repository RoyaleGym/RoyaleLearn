"""run_exact's fill of unwritten memory: on where the harness checks, off elsewhere, no value moved.

docs/harness-spec.md section 5.1. The fill is torch's guard against reading memory nothing wrote:
under deterministic algorithms every fresh buffer is first written with NaN. It changes no value
of a program that reads only what it wrote, and it costs a full write of every such buffer, so
the harness keeps it only in the iterations it checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn import config as cfg
from royalelearn.testing import coordinator, tiny_config

torch = pytest.importorskip("torch")
torch_deterministic = pytest.importorskip("torch.utils.deterministic")

from royalelearn import determinism  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_the_fill() -> Any:
    """The flag is process-wide; every test here leaves it as it found it."""
    before = torch_deterministic.fill_uninitialized_memory
    yield
    torch_deterministic.fill_uninitialized_memory = before


def exact_config(
    tmp_path: Path, *, debug: int, every: int, forced_rows: str = "all"
) -> cfg.RunConfig:
    config = tiny_config(
        tmp_path,
        determinism=cfg.DeterminismConfig(tier="run_exact"),
        alarms=cfg.AlarmConfig(enabled=False),
    )
    return msgspec.structs.replace(
        config,
        ppo=msgspec.structs.replace(
            config.ppo,
            debug_assert_iterations=debug,
            check_ratio_invariant_every=every,
            forced_rows=forced_rows,
        ),
    )


def test_the_switch_sets_torch_s_own_flag_and_run_exact_starts_with_it_on() -> None:
    determinism.set_fill_uninitialized(False)
    assert torch_deterministic.fill_uninitialized_memory is False
    determinism.set_fill_uninitialized(True)
    assert torch_deterministic.fill_uninitialized_memory is True
    determinism.set_fill_uninitialized(False)
    determinism.apply_cublas_workspace_config()
    was = torch.are_deterministic_algorithms_enabled()
    try:
        applied = determinism.apply("run_exact")
        assert torch_deterministic.fill_uninitialized_memory is True, "start-up is checked"
        assert applied["deterministic_algorithms"] is True
    finally:
        torch.use_deterministic_algorithms(was)


def test_the_fill_is_on_exactly_in_the_iterations_the_harness_checks(tmp_path: Path) -> None:
    """One debug iteration, then a ratio check every second iteration: on, off, on, off."""
    config = exact_config(tmp_path, debug=1, every=2)
    seen: list[bool] = []
    with coordinator(config) as run:
        step = run.update.step

        def recording(buffer: Any, sched: Any) -> Any:
            seen.append(bool(torch_deterministic.fill_uninitialized_memory))
            return step(buffer, sched)

        run.update.step = recording
        for _ in range(4):
            run.iterate()
    assert seen == [True, False, True, False]


def test_a_resumed_process_fills_its_first_iteration_past_the_debug_window(
    tmp_path: Path,
) -> None:
    """The first iteration a process runs is filled wherever the run stands, because that is
    where a resumed run's leftovers first differ from a straight run's. Resumed at iteration
    three, with the one-iteration debug window behind it and no ratio check ever due, the first
    is filled and the second is not. A clause that asked for iteration 0 instead of the first
    in the process leaves both unfilled, and nothing tested past iteration 0 until this.
    """
    config = exact_config(tmp_path, debug=1, every=0)
    with coordinator(config) as first:
        for _ in range(3):
            first.iterate()
        saved = first.checkpoint()
        run_dir = first.run_dir
    seen: list[bool] = []
    with coordinator(config, resume=saved, run_dir=run_dir) as again:
        assert again.iteration == 3
        step = again.update.step

        def recording(buffer: Any, sched: Any) -> Any:
            seen.append(bool(torch_deterministic.fill_uninitialized_memory))
            return step(buffer, sched)

        again.update.step = recording
        again.iterate()
        again.iterate()
    assert seen == [True, False], seen


def _run(
    tmp_path: Path, name: str, *, fill_always: bool, monkeypatch: Any
) -> tuple[list[str], list[bool], list[bool], list[float]]:
    """Four iterations with no checks due; each iteration's digest, the flag as collection and
    the update each saw it, and each iteration's share of forced rows. ``fill_always`` keeps the
    real switch and forces it on, so the arm measures a fill that is ON rather than one nobody
    turned off."""
    real = determinism.set_fill_uninitialized
    if fill_always:
        monkeypatch.setattr(determinism, "set_fill_uninitialized", lambda on: real(True))
    config = exact_config(tmp_path / name, debug=0, every=0, forced_rows="critic_only")
    digests: list[str] = []
    at_update: list[bool] = []
    at_collection: list[bool] = []
    forced: list[float] = []
    with coordinator(config) as run:
        step, act = run.update.step, run.inference.act

        def recording_step(buffer: Any, sched: Any) -> Any:
            at_update.append(bool(torch_deterministic.fill_uninitialized_memory))
            return step(buffer, sched)

        def recording_act(*args: Any, **kwargs: Any) -> Any:
            at_collection.append(bool(torch_deterministic.fill_uninitialized_memory))
            return act(*args, **kwargs)

        run.update.step = recording_step
        run.inference.act = recording_act
        for _ in range(4):
            run.iterate()
            digests.append(run.state_digest())
            forced.append(float(run.rows[-1]["ppo/forced_frac"]))
    monkeypatch.undo()
    return digests, at_update, at_collection, forced


def test_turning_the_fill_off_moves_no_bit_of_the_learner(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The claim the policy rests on, on the path this suite can run: past the first iteration
    of the process, with no checks due, the fill is off in collection and update alike, and the
    learner's state must be the one a run with the fill always on reaches, digest for digest. A
    read of unwritten float memory on the path would be NaN in one run and leftovers in the other.

    What it is evidence for: that this repository's own allocations on the CPU path, the update's
    critic_only path included, are fully written. It is not evidence about torch's CUDA kernels;
    that is a two-arm digest comparison on the GPU past the debug window, which is the A/B the
    training runs are measured with."""
    off, off_update, off_collection, off_forced = _run(
        tmp_path, "off", fill_always=False, monkeypatch=monkeypatch
    )
    on, on_update, on_collection, on_forced = _run(
        tmp_path, "on", fill_always=True, monkeypatch=monkeypatch
    )
    # critic_only is the path under test only where a batch holds both kinds of row: all forced
    # and the actor never runs, none forced and it is the ``all`` path.
    assert all(0.0 < share < 1.0 for share in off_forced + on_forced), (off_forced, on_forced)
    assert on_update == [True] * 4 and on_collection and all(on_collection)
    assert off_update == [True, False, False, False], "the first iteration of a process is checked"
    assert off_collection[0] and not off_collection[-1]
    assert off == on
