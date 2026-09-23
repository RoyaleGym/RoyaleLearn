"""Which device a run's memory numbers are about.

Every VRAM reading in the harness used to key off ``torch.cuda.is_available()``, which answers a
question about the MACHINE. Seven sessions share this one, so a CPU run here would measure a card
it never touches: the gate's peak reads near zero and passes whatever the run needs, and every
``health/vram_*`` key in the row describes another process's memory under this run's name. It was
already visible as a flake -- ``test_resume`` compared two runs row for row and the vram pair
differed while another session held the GPU.

The gate takes ``torch`` as an argument, which is the seam a CPU test needs to watch it refuse.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from royalelearn.coordinator import LearningCoordinator, _vram_regime


class _Torch:
    """A machine with a card, so that availability and the run's device disagree."""

    def __init__(self) -> None:
        self.probed = False

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def mem_get_info() -> tuple[int, int]:
                raise AssertionError("the gate read a device this run is not on")

            @staticmethod
            def reset_peak_memory_stats() -> None:
                raise AssertionError("the gate touched a device this run is not on")

        self.cuda = _Cuda()


class _Gate:
    """A coordinator reduced to what ``_vram_gate`` reads: a device, a config and a printer."""

    _vram_gate = LearningCoordinator._vram_gate
    _probe_backward = LearningCoordinator._probe_backward

    def __init__(self, device: str, headroom: float = 512.0) -> None:
        self.device = device
        self.said: list[str] = []
        self.printer = self.said.append
        self.config = type(
            "_Config", (), {"doctor": type("_Doctor", (), {"vram_headroom_mb": headroom})()}
        )()


def test_a_cpu_run_does_not_gate_on_a_card_it_never_touches() -> None:
    """The refusal a CPU test could not see before: availability said yes, the device said no."""
    gate = _Gate("cpu")
    gate._vram_gate(_Torch())  # every cuda call in the fake raises if it is reached
    assert not gate.said


def test_a_cuda_run_still_reaches_the_card() -> None:
    """The other half. Without this, "return early" would pass by never gating anything."""
    gate = _Gate("cuda:0")
    with pytest.raises(AssertionError, match="this run is not on"):
        gate._vram_gate(_Torch())


def test_no_headroom_configured_is_still_no_gate() -> None:
    gate = _Gate("cuda:0", headroom=0.0)
    gate._vram_gate(_Torch())
    assert not gate.said


class _Explodes:
    """Any attribute is a failure, and BaseException so no ``except Exception`` hides it."""

    def __getattr__(self, name: str) -> Any:
        raise KeyboardInterrupt(f"the device check must come before torch.{name}")


def test_the_regime_fields_ask_the_device_before_they_ask_torch(monkeypatch: Any) -> None:
    """A CPU run publishes no vram keys, and does not reach torch to find that out.

    KeyboardInterrupt rather than an ordinary error on purpose: ``_vram_regime`` wraps its body
    in ``except Exception``, so a plant that raised one would be swallowed and this test would
    pass against the old code as well.
    """
    monkeypatch.setitem(sys.modules, "torch", _Explodes())
    assert _vram_regime(123.0, device="cpu") == {}


def test_a_cuda_run_still_publishes_its_regime() -> None:
    """The other half again, and it is not hypothetical on this machine.

    This machine HAS a card and the suite runs on the CPU, which is exactly the configuration
    the defect needed: before this, every CPU test run published a real card's driver-free
    figure under health/vram_driver_free_mb. That is what made test_resume's row-for-row
    comparison flake while another session held the GPU.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine, so there is no reading to take")
    fields = _vram_regime(123.0, device="cuda:0")
    assert "health/vram_driver_free_mb" in fields
    assert fields["health/vram_needed_mb"] == 123.0
    assert _vram_regime(123.0, device="cpu") == {}, (
        "the same machine, the same instant, and a CPU run takes no reading from the card"
    )
