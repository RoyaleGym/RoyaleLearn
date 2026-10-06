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

from typing import Any

__all__ = ["CUDA", "ROCM", "ZLUDA", "ZLUDA_SUFFIX", "gpu_backend", "is_rocm", "tf32_capable"]

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
