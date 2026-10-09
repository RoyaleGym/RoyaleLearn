"""Which GPU stack torch runs on: NVIDIA's CUDA, AMD's ROCm, or a translation layer.

AMD's ROCm builds of torch show an AMD card through the same ``torch.cuda`` API, under the same
device name, "cuda", and set ``torch.version.hip``. So almost everything a run does on an NVIDIA
card it does unchanged on an AMD one; what differs is asked here, in one place: which precisions
the card has, and what to call it in a run's identity.

ZLUDA runs torch's CUDA build on an AMD card by translating the CUDA calls. It is not a supported
way to train: ``torch.version.hip`` is not set, so torch believes it is on NVIDIA hardware, and
every device name it reports ends in " [ZLUDA]" (``zluda_common::PROJECT_SUFFIX`` upstream), which
is how it is told apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "CUDA",
    "ROCM",
    "ZLUDA",
    "ZLUDA_SUFFIX",
    "carries_ptx",
    "cuda_arch_supported",
    "gpu_backend",
    "is_rocm",
    "tf32_capable",
    "unsupported_gpu",
]

CUDA = "cuda"
ROCM = "rocm"
ZLUDA = "zluda"
#: What ZLUDA appends to every device name it reports.
ZLUDA_SUFFIX = " [ZLUDA]"
#: AMD architectures whose matrix units take float32 at reduced precision when torch allows TF32
#: (the CDNA3 Instinct cards). Consumer Radeon cards have no such mode.
_HIP_XF32_ARCHES = ("gfx94", "gfx95")


def is_rocm() -> bool:
    """Whether this torch is a ROCm build, which is what an AMD card needs."""
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.version, "hip", None))


def gpu_backend(index: int = 0) -> str | None:
    """"cuda", "rocm" or "zluda" for the GPU torch would train on, or None without one."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if is_rocm():
        return ROCM
    if str(torch.cuda.get_device_name(index)).endswith(ZLUDA_SUFFIX):
        return ZLUDA
    return CUDA


def carries_ptx() -> bool | None:
    """Whether this torch build carries PTX, the portable form of its kernels; None if unknown.

    ZLUDA translates PTX and nothing else. torch's builds for CUDA 12 and later carry machine
    code for each NVIDIA architecture and no PTX, so on ZLUDA no kernel of theirs would load.
    ``torch.cuda.get_arch_list()`` names a build's PTX as a ``compute_`` entry.
    """
    try:
        import torch

        arches = torch.cuda.get_arch_list()
    except Exception:
        return None
    if not arches:
        return None
    return any(str(arch).startswith("compute_") for arch in arches)


def _arch(entry: str) -> tuple[str, int, int] | None:
    """``"sm_86"`` -> ("sm", 8, 6); ``"compute_120"`` -> ("compute", 12, 0); None otherwise."""
    kind, _, number = str(entry).partition("_")
    if kind not in ("sm", "compute") or len(number) < 2 or not number.isdigit():
        return None
    return kind, int(number[:-1]), int(number[-1])


def cuda_arch_supported(capability: tuple[int, int], arches: Sequence[str]) -> bool | None:
    """Whether a card of ``capability`` runs a build compiled for ``arches``; None if the
    build does not say.

    Machine code (``sm_XY``) runs on cards of the same major version at the same minor or
    higher. PTX (``compute_XY``) is compiled by the driver for any card at that capability or
    above.
    """
    parsed = [arch for arch in (_arch(entry) for entry in arches) if arch is not None]
    if not parsed:
        return None
    major, minor = int(capability[0]), int(capability[1])
    for kind, arch_major, arch_minor in parsed:
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True
    return False


def unsupported_gpu(index: int = 0) -> str | None:
    """Why this torch cannot run on the NVIDIA card it sees, or None when it can or cannot tell.

    ``torch.cuda.is_available()`` is true for any card the driver sees, whether or not this torch
    was compiled for its architecture; the first kernel it launches on one it was not compiled
    for fails with "no kernel image is available". AMD cards are not judged this way.
    """
    import torch

    try:
        if gpu_backend(index) != CUDA:
            return None
        major, minor = torch.cuda.get_device_capability(index)
        arches = list(torch.cuda.get_arch_list())
        name = str(torch.cuda.get_device_name(index))
    except Exception:
        return None
    if cuda_arch_supported((major, minor), arches) is not False:
        return None
    built = sorted(
        (arch_major, arch_minor)
        for kind, arch_major, arch_minor in (a for a in map(_arch, arches) if a is not None)
        if kind == "sm"
    )
    if built and (major, minor) < built[0]:
        span = f"which runs on sm_{built[0][0]}{built[0][1]} and newer"
        return f"Your graphics card ({name}, sm_{major}{minor}) is too old for this PyTorch, {span}"
    listed = ", ".join(arches)
    card = f"Your graphics card ({name}, sm_{major}{minor})"
    return f"{card} is not one this PyTorch runs on ({listed})"


def tf32_capable(device: Any) -> bool:
    """Whether float32 arithmetic on ``device`` can run at TF32 precision when torch allows it.

    NVIDIA cards from Ampere (compute capability 8.0) on can; Turing and older cannot, and nor
    can AMD's consumer cards. A device that cannot be asked keeps the answer yes, the looser one,
    so that nothing that ran before stops on a question it could not put.
    """
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return False
    try:
        if is_rocm():
            arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
            return arch.startswith(_HIP_XF32_ARCHES)
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:
        return True
    return int(major) >= 8
