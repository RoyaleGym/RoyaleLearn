"""The cap on a CUDA run's allocator, which keeps torch's cache on the card.

On Windows a card that runs out is backed with system RAM rather than refused, so a cache that
only grows can take gigabytes of it over a long run. "auto" caps the allocator there and nowhere
else; a number caps it anywhere; None leaves it uncapped. The suite has no GPU, so the call into
torch is checked against a stand-in that records it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from royalelearn.coordinator import WINDOWS_VRAM_FRACTION, LearningCoordinator, vram_cap
from royalelearn.errors import PreflightError


def test_auto_caps_on_windows_only() -> None:
    assert vram_cap("auto", platform="win32") == WINDOWS_VRAM_FRACTION
    assert vram_cap("auto", platform="linux") is None
    assert vram_cap("auto", platform="darwin") is None
    assert 0 < WINDOWS_VRAM_FRACTION < 1


def test_a_number_caps_anywhere_and_none_does_not() -> None:
    assert vram_cap(0.5, platform="linux") == 0.5
    assert vram_cap(1, platform="win32") == 1.0
    assert vram_cap(None, platform="win32") is None


@pytest.mark.parametrize("bad", [0, -0.1, 1.5, "half"])
def test_a_share_outside_the_card_is_refused(bad: Any) -> None:
    with pytest.raises(PreflightError, match="vram_fraction"):
        vram_cap(bad)


def build_env() -> Any:
    """Never called: nothing here builds an environment."""
    raise AssertionError


class _Torch:
    def __init__(self) -> None:
        self.calls: list[tuple[float, Any]] = []
        self.cuda = SimpleNamespace(set_per_process_memory_fraction=self._set)

    def _set(self, fraction: float, device: Any) -> None:
        self.calls.append((fraction, device))


def _run(fraction: float | None, device: str) -> tuple[_Torch, list[str]]:
    import torch

    lines: list[str] = []
    run = SimpleNamespace(
        vram_fraction=fraction, device=torch.device(device), printer=lines.append
    )
    fake = _Torch()
    LearningCoordinator._cap_vram(run, fake)  # type: ignore[arg-type]
    return fake, lines


def test_a_cuda_run_caps_its_own_device_and_says_so() -> None:
    fake, lines = _run(0.8, "cuda:0")
    assert len(fake.calls) == 1
    fraction, device = fake.calls[0]
    assert fraction == 0.8 and str(device) == "cuda:0"
    assert "80% of the card" in lines[0]


def test_a_cpu_run_or_no_cap_touches_nothing() -> None:
    assert _run(0.8, "cpu")[0].calls == []
    assert _run(None, "cuda:0")[0].calls == []


def test_the_learner_passes_its_setting_on_and_refuses_a_bad_one(tmp_path: Any) -> None:
    from royalelearn import Learner

    learner = Learner(build_env, device="cpu", save_dir=tmp_path / "a", vram_fraction=0.6)
    assert learner._kwargs["vram_fraction"] == 0.6
    assert Learner(build_env, device="cpu", save_dir=tmp_path / "b")._kwargs["vram_fraction"] == (
        "auto"
    )
    with pytest.raises(PreflightError, match="vram_fraction"):
        Learner(build_env, device="cpu", save_dir=tmp_path / "c", vram_fraction=2)
