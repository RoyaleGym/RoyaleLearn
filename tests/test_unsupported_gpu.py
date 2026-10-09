"""A graphics card this PyTorch has no kernels for: the CPU when the device is "auto", a plain
refusal when it is "cuda".

``torch.cuda.is_available()`` is true for any NVIDIA card the driver sees, whether or not this
torch build was compiled for its architecture. A build for CUDA 12.8 carries kernels for sm_75
and newer, so on a GTX 970 (sm_52) the first kernel a run launches fails with "no kernel image is
available for execution on the device", from the middle of the first collection (reported by a
user, 2026-10-09). The card's capability and the build's architecture list say so before then:
a card runs a build that has machine code for its major version at its minor or below, or PTX
(``compute_``) at or below its capability. No card is needed for these: torch's answers are
stood in for.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from royalelearn.gpu import cuda_arch_supported, unsupported_gpu  # noqa: E402

#: ``torch.cuda.get_arch_list()`` of torch 2.11.0+cu128, read 2026-10-06.
CU128 = ("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120")


def _card(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    capability: tuple[int, int],
    arches: tuple[str, ...] = CU128,
) -> None:
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device=0: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=0: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(arches))
    props = SimpleNamespace(name=name, major=capability[0], minor=capability[1])
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=0: props)


@pytest.mark.parametrize(
    ("capability", "arches", "runs"),
    [
        ((5, 2), CU128, False),  # GTX 970
        ((6, 1), CU128, False),  # GTX 1080
        ((7, 5), CU128, True),  # RTX 2080
        ((8, 9), CU128, True),  # RTX 4070: sm_86 machine code runs on 8.9
        ((8, 6), ("sm_80", "sm_90"), True),  # sm_80 runs on 8.6
        ((8, 6), ("sm_90",), False),  # nothing for major 8, no PTX
        ((8, 6), ("sm_90", "compute_75"), True),  # PTX at or below the card is compiled for it
        ((6, 1), ("sm_75", "compute_75"), False),  # PTX above the card is not
        ((12, 0), CU128, True),
    ],
)
def test_a_card_runs_a_build_with_its_major_at_or_below_its_minor_or_ptx_below_it(
    capability: tuple[int, int], arches: tuple[str, ...], runs: bool
) -> None:
    assert cuda_arch_supported(capability, arches) is runs


def test_a_build_that_does_not_say_is_not_held_against_the_card() -> None:
    assert cuda_arch_supported((5, 2), ()) is None


def test_auto_takes_the_cpu_for_a_card_too_old_for_this_pytorch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plant: return "cuda" whenever a GPU is visible, and a GTX 970 is handed every kernel."""
    from royalelearn.learner import resolve_device

    _card(monkeypatch, "NVIDIA GeForce GTX 970", (5, 2))
    assert resolve_device("auto") == "cpu"
    said = capsys.readouterr().out
    assert "NVIDIA GeForce GTX 970" in said and "sm_52" in said and "sm_75" in said
    assert "CPU" in said
    _card(monkeypatch, "NVIDIA GeForce RTX 4070 Ti", (8, 9))
    assert resolve_device("auto") == "cuda"
    assert capsys.readouterr().out == ""


def test_an_amd_card_is_not_judged_by_nvidia_architectures(monkeypatch: pytest.MonkeyPatch) -> None:
    _card(monkeypatch, "AMD Radeon RX 7900 XTX", (11, 0), arches=())
    monkeypatch.setattr(torch.version, "hip", "7.2.1", raising=False)
    assert unsupported_gpu() is None


def test_an_explicit_cuda_run_on_such_a_card_is_refused_with_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rather than "no kernel image is available" from the middle of the first collection."""
    from royalelearn.coordinator import LearningCoordinator
    from royalelearn.errors import PreflightError

    _card(monkeypatch, "NVIDIA GeForce GTX 970", (5, 2))
    run: Any = SimpleNamespace(
        device=torch.device("cuda"),
        printer=lambda _line: None,
        config=SimpleNamespace(
            determinism=SimpleNamespace(tier="throughput"),
            net=SimpleNamespace(autocast_dtype="float32"),
        ),
    )
    with pytest.raises(PreflightError, match=r"GTX 970.*sm_52.*cpu"):
        LearningCoordinator._check_gpu_stack(run, torch)
