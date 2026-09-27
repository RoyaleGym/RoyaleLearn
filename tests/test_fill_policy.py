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


def exact_config(tmp_path: Path, *, debug: int, every: int) -> cfg.RunConfig:
    config = tiny_config(
        tmp_path,
        determinism=cfg.DeterminismConfig(tier="run_exact"),
        alarms=cfg.AlarmConfig(enabled=False),
    )
    return msgspec.structs.replace(
        config,
        ppo=msgspec.structs.replace(
            config.ppo, debug_assert_iterations=debug, check_ratio_invariant_every=every
        ),
    )


def test_the_switch_sets_torch_s_own_flag_and_run_exact_records_the_policy() -> None:
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
        assert applied["fill_uninitialized_memory"] == determinism.FILL_POLICY
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


def _digests(tmp_path: Path, name: str, *, fill_always: bool, monkeypatch: Any) -> list[str]:
    if fill_always:
        monkeypatch.setattr(determinism, "set_fill_uninitialized", lambda on: None)
    config = exact_config(tmp_path / name, debug=0, every=0)
    digests = []
    with coordinator(config) as run:
        for _ in range(3):
            run.iterate()
            digests.append(run.state_digest())
    monkeypatch.undo()
    return digests


def test_turning_the_fill_off_moves_no_bit_of_the_learner(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The claim the policy rests on, on the path this suite can run: with no checks due the fill
    is off from the first iteration, and the learner's state must be the one a run with the fill
    always on reaches, digest for digest. A read of unwritten memory anywhere on the path would
    be NaN in one run and leftovers in the other."""
    off = _digests(tmp_path, "off", fill_always=False, monkeypatch=monkeypatch)
    on = _digests(tmp_path, "on", fill_always=True, monkeypatch=monkeypatch)
    assert off == on
