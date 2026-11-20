"""Training on Apple's GPU (MPS): no float64 on the device, and no unfenced asynchronous copy."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from royalelearn.learn.inference import BatchedInference  # noqa: E402
from royalelearn.learn.returns import _as_float64  # noqa: E402

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")


def _up(device: str, values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """``BatchedInference._up`` on ``device``, and the staging tensor it read."""
    staging = torch.empty(values.size, dtype=torch.int64)
    holder = SimpleNamespace(device=torch.device(device))
    return BatchedInference._up(holder, staging, values), staging


def test_a_copy_off_cuda_does_not_ask_to_be_asynchronous(monkeypatch):
    # Only CUDA's copies are fenced by events, so every other device must get a blocking one.
    seen = []
    real = torch.Tensor.copy_

    def spy(self, src, non_blocking=False):
        seen.append(non_blocking)
        return real(self, src, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", spy)
    _up("cpu", np.arange(8))
    assert seen == [False]


@needs_mps
def test_an_upload_to_mps_is_complete_when_it_returns():
    values = np.arange(1 << 20)
    out, staging = _up("mps", values)
    staging.numpy()[...] = -1  # the next call refills the same staging tensor
    assert torch.equal(out.cpu(), torch.from_numpy(values))


@needs_mps
def test_returns_read_an_mps_tensor_as_float64_on_the_host():
    values = torch.arange(4, dtype=torch.float32, device="mps")
    out = _as_float64(values)
    assert out.dtype == np.float64
    assert out.tolist() == [0.0, 1.0, 2.0, 3.0]
