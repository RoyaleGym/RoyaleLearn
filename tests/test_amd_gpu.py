"""AMD cards: torch's ROCm build shows them as "cuda" devices; what differs is asked in one place.

No AMD card is needed for these. Each test stands in for torch's answers about the device --
``torch.version.hip``, the properties, the capability -- the way a ROCm build or a ZLUDA
translation layer gives them, and checks what the run makes of them:

- ROCm: the identity names the card by its AMD architecture, the float32 ratio check keeps
  float32's tolerance (consumer Radeon cards have no TF32), and the throughput tier does not
  switch on cuDNN's benchmark search, which MIOpen answers with an exhaustive search per shape.
- ZLUDA (torch's CUDA build on an AMD card through a translation layer): named as such in the
  identity and noted at start-up. Refused on a torch build without PTX, which is every build
  for CUDA 12 and later, under run_exact, which nobody has shown it keeps, and with a reduced
  precision autocast. cuDNN, which it only partly translates, is switched off.
- NVIDIA: every answer exactly what it was, except the TF32 tolerance, which now follows whether
  the card has TF32 at all (Ampere, compute capability 8.0, and newer).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from royalelearn.config import PPOConfig  # noqa: E402
from royalelearn.gpu import ROCM, ZLUDA, gpu_backend, tf32_capable  # noqa: E402
from royalelearn.learn.ppo import TF32_RATIO_ATOL, ratio_tolerance  # noqa: E402


def _fake_gpu(
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    hip: str | None,
    capability: tuple[int, int] = (8, 6),
    arch: str = "",
    arches: tuple[str, ...] = ("sm_75", "sm_86", "compute_75"),
) -> None:
    props = SimpleNamespace(
        name=name,
        major=capability[0],
        minor=capability[1],
        gcnArchName=arch,
        total_memory=16 * 2**30,
    )
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=0: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device=0: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=0: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(arches))


@pytest.fixture
def rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_gpu(
        monkeypatch,
        name="AMD Radeon RX 7900 XTX",
        hip="7.2.1",
        capability=(11, 0),
        arch="gfx1100:sramecc-:xnack-",
    )


@pytest.fixture
def zluda(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_gpu(monkeypatch, name="AMD Radeon RX 7900 XTX [ZLUDA]", hip=None)


@pytest.fixture
def tf32_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)


def test_the_backend_is_named_from_what_torch_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_gpu(monkeypatch, name="AMD Radeon RX 7900 XTX", hip="7.2.1")
    assert gpu_backend() == ROCM
    _fake_gpu(monkeypatch, name="AMD Radeon RX 7900 XTX [ZLUDA]", hip=None)
    assert gpu_backend() == ZLUDA
    _fake_gpu(monkeypatch, name="NVIDIA GeForce RTX 4070 Ti", hip=None)
    assert gpu_backend() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_backend() is None


@pytest.mark.parametrize(
    ("hip", "capability", "arch", "capable"),
    [
        ("7.2.1", (11, 0), "gfx1100", False),  # Radeon RX 7900: no TF32
        ("7.2.1", (12, 0), "gfx1201", False),  # Radeon RX 9070
        ("7.2.1", (9, 4), "gfx942:sramecc+:xnack-", True),  # Instinct MI300
        (None, (7, 5), "", False),  # NVIDIA Turing
        (None, (8, 6), "", True),  # NVIDIA Ampere
        (None, (8, 9), "", True),  # NVIDIA Ada
    ],
)
def test_tf32_follows_the_card(
    monkeypatch: pytest.MonkeyPatch, hip: Any, capability: Any, arch: str, capable: bool
) -> None:
    _fake_gpu(monkeypatch, name="card", hip=hip, capability=capability, arch=arch)
    assert tf32_capable("cuda") is capable
    assert tf32_capable("cpu") is False


def test_a_radeon_card_keeps_the_float32_ratio_tolerance(rocm: None, tf32_on: None) -> None:
    """Plant: drop the card check and a real 2e-3 drift on a Radeon card passes unnoticed."""
    assert ratio_tolerance(PPOConfig(), torch.float32, "cuda") == 1e-4


def test_an_ampere_card_with_tf32_keeps_the_tf32_tolerance(
    monkeypatch: pytest.MonkeyPatch, tf32_on: None
) -> None:
    _fake_gpu(monkeypatch, name="NVIDIA GeForce RTX 4070 Ti", hip=None, capability=(8, 9))
    assert ratio_tolerance(PPOConfig(), torch.float32, "cuda") == TF32_RATIO_ATOL
    _fake_gpu(monkeypatch, name="NVIDIA GeForce RTX 2080", hip=None, capability=(7, 5))
    assert ratio_tolerance(PPOConfig(), torch.float32, "cuda") == 1e-4


def test_the_identity_names_an_amd_card_by_its_architecture(rocm: None) -> None:
    from royalelearn.identity import describe_device

    assert describe_device("cuda") == "rocm:AMD Radeon RX 7900 XTX:gfx1100"


def test_the_identity_names_zluda_and_leaves_nvidia_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from royalelearn.identity import describe_device

    _fake_gpu(monkeypatch, name="AMD Radeon RX 7900 XTX [ZLUDA]", hip=None, capability=(8, 6))
    assert describe_device("cuda") == "zluda:AMD Radeon RX 7900 XTX [ZLUDA]:sm_86"
    _fake_gpu(monkeypatch, name="NVIDIA GeForce RTX 4070 Ti", hip=None, capability=(8, 9))
    assert describe_device("cuda") == "cuda:NVIDIA GeForce RTX 4070 Ti:sm_89"


def test_the_throughput_tier_leaves_the_benchmark_search_off_on_rocm(
    rocm: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from royalelearn.determinism import apply

    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    applied = apply("throughput")
    assert torch.backends.cudnn.benchmark is False
    assert applied["cudnn_benchmark"] is False


def test_the_throughput_tier_on_nvidia_is_what_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    from royalelearn.determinism import apply

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    applied = apply("throughput")
    assert torch.backends.cudnn.benchmark is True
    assert applied == {
        "tier": "throughput",
        "torch_threads": 1,
        "deterministic_algorithms": False,
        "cudnn_benchmark": True,
    }


def test_the_memory_fields_survive_a_driver_that_cannot_say_what_is_free(
    rocm: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plant: one try around both reads and every health/vram_* key goes when one fails."""
    from royalelearn.coordinator import _vram_regime

    stats = {
        "reserved_bytes.all.current": 2e9,
        "inactive_split_bytes.all.current": 1e8,
        "num_alloc_retries": 3,
    }
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda *_a, **_k: stats)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("HIP error: invalid argument")

    monkeypatch.setattr(torch.cuda, "mem_get_info", refuse)
    fields = _vram_regime(device="cuda")
    assert fields["health/vram_reserved_mb"] == pytest.approx(2000.0)
    assert fields["health/vram_alloc_retries"] == 3
    assert "health/vram_driver_free_mb" not in fields
    assert "health/vram_available_mb" not in fields


def test_the_utilisation_hint_names_the_amd_package(
    rocm: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import royalelearn.coordinator as coordinator

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("no utilisation reading")

    monkeypatch.setattr(torch.cuda, "utilization", refuse)
    monkeypatch.setattr(coordinator, "_GPU_UTIL_COMPLAINED", False)
    assert coordinator._gpu_util("cuda") is None
    out = capsys.readouterr().out
    assert "amdsmi" in out and "pynvml" not in out


def _run(lines: list[str], tier: str = "throughput", autocast: str = "float32") -> Any:
    """What ``_check_gpu_stack`` reads of a coordinator."""
    return SimpleNamespace(
        device=torch.device("cuda"),
        printer=lines.append,
        config=SimpleNamespace(
            determinism=SimpleNamespace(tier=tier),
            net=SimpleNamespace(autocast_dtype=autocast),
        ),
    )


@pytest.fixture
def cudnn_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)


def test_zluda_is_warned_about_and_refused_under_run_exact(zluda: None, cudnn_on: None) -> None:
    from royalelearn.coordinator import LearningCoordinator
    from royalelearn.errors import PreflightError

    lines: list[str] = []

    def run(tier: str) -> Any:
        return _run(lines, tier)

    LearningCoordinator._check_gpu_stack(run("throughput"), torch)  # type: ignore[arg-type]
    assert any("ZLUDA" in line for line in lines)
    with pytest.raises(PreflightError, match="ZLUDA"):
        LearningCoordinator._check_gpu_stack(run("run_exact"), torch)  # type: ignore[arg-type]


def test_rocm_and_nvidia_pass_the_stack_check_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from royalelearn.coordinator import LearningCoordinator

    lines: list[str] = []
    run = SimpleNamespace(
        device=torch.device("cuda"),
        printer=lines.append,
        config=SimpleNamespace(determinism=SimpleNamespace(tier="run_exact")),
    )
    for hip, name in (("7.2.1", "AMD Radeon RX 7900 XTX"), (None, "NVIDIA GeForce RTX 4070 Ti")):
        _fake_gpu(monkeypatch, name=name, hip=hip, arches=("sm_86",))
        LearningCoordinator._check_gpu_stack(run, torch)  # type: ignore[arg-type]
    assert lines == []


#: ``torch.cuda.get_arch_list()`` of torch 2.11.0+cu128 for Windows, read 2026-10-06: machine
#: code for each architecture and no ``compute_`` entry, so no PTX.
CU128_ARCHES = ("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120")


def test_zluda_on_a_torch_build_without_ptx_is_refused(
    monkeypatch: pytest.MonkeyPatch, cudnn_on: None
) -> None:
    """ZLUDA translates PTX and nothing else: on such a build no kernel would ever load."""
    from royalelearn.coordinator import LearningCoordinator
    from royalelearn.errors import PreflightError

    lines: list[str] = []
    _fake_gpu(monkeypatch, name="AMD Radeon RX 7900 XTX [ZLUDA]", hip=None, arches=CU128_ARCHES)
    with pytest.raises(PreflightError, match=r"PTX.*ROCm build of torch"):
        LearningCoordinator._check_gpu_stack(_run(lines), torch)  # type: ignore[arg-type]
    # An older build that carries PTX goes on, with the notice.
    _fake_gpu(monkeypatch, name="AMD Radeon RX 7900 XTX [ZLUDA]", hip=None)
    LearningCoordinator._check_gpu_stack(_run(lines), torch)  # type: ignore[arg-type]
    assert any("ZLUDA" in line for line in lines)


def test_zluda_switches_cudnn_off_and_says_so(zluda: None, cudnn_on: None) -> None:
    from royalelearn.coordinator import LearningCoordinator

    lines: list[str] = []
    LearningCoordinator._check_gpu_stack(_run(lines), torch)  # type: ignore[arg-type]
    assert torch.backends.cudnn.enabled is False
    assert any("cuDNN" in line for line in lines)


def test_zluda_refuses_a_reduced_precision_autocast(zluda: None, cudnn_on: None) -> None:
    from royalelearn.coordinator import LearningCoordinator
    from royalelearn.errors import PreflightError

    with pytest.raises(PreflightError, match=r"net.autocast_dtype"):
        LearningCoordinator._check_gpu_stack(
            _run([], autocast="bfloat16"), torch  # type: ignore[arg-type]
        )


def test_ptx_is_unknown_when_torch_cannot_say(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown is not no: only a list torch gives, without a compute_ entry, refuses."""
    from royalelearn.gpu import carries_ptx

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: [])
    assert carries_ptx() is None

    def refuse() -> Any:
        raise RuntimeError("not compiled")

    monkeypatch.setattr(torch.cuda, "get_arch_list", refuse)
    assert carries_ptx() is None
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(CU128_ARCHES))
    assert carries_ptx() is False
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_86", "compute_86"])
    assert carries_ptx() is True


def _cap(lines: list[str]) -> Any:
    """What ``_cap_vram`` reads of a coordinator."""
    return SimpleNamespace(
        vram_fraction=0.9, device=torch.device("cuda", 0), printer=lines.append
    )


def test_the_vram_cap_refused_on_rocm_leaves_the_run_going(
    rocm: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reported for AMD APUs, whose memory is the system's. Plant: no guard and this raises."""
    from royalelearn.coordinator import LearningCoordinator

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("HIP error: invalid argument")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", refuse)
    lines: list[str] = []
    LearningCoordinator._cap_vram(_cap(lines), torch)  # type: ignore[arg-type]
    assert any("not set" in line for line in lines)


def test_the_vram_cap_refused_on_nvidia_still_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plant: guard every backend and an NVIDIA error, a bug of ours so far, prints and passes."""
    from royalelearn.coordinator import LearningCoordinator

    _fake_gpu(monkeypatch, name="NVIDIA GeForce RTX 4070 Ti", hip=None)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Invalid device argument")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", refuse)
    with pytest.raises(RuntimeError, match="Invalid device"):
        LearningCoordinator._cap_vram(_cap([]), torch)  # type: ignore[arg-type]
