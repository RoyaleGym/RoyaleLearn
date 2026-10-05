"""The ratio invariant's tolerance follows the precision the forwards actually ran at.

Float32 on a GPU allowed TF32 -- cuDNN's default outside run_exact -- is its own precision: the
rollout's forward and the update's pick different kernels for their different batch shapes, and
their log-probabilities differ by more than float32's 1e-4. The update's assertion and the run's
alarm take the number from one function, so they cannot disagree.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from royalelearn.config import PPOConfig
from royalelearn.learn.ppo import TF32_RATIO_ATOL, ratio_tolerance


@pytest.fixture
def tf32(monkeypatch: pytest.MonkeyPatch) -> Any:
    def allow(on: bool) -> None:
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", on)
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    return allow


def test_float32_on_a_gpu_allowed_tf32_has_the_tf32_tolerance(tf32: Any) -> None:
    tf32(True)
    assert ratio_tolerance(PPOConfig(), torch.float32, "cuda") == TF32_RATIO_ATOL
    assert ratio_tolerance(PPOConfig(), torch.float32, "cpu") == 1e-4, "no TF32 on the CPU"
    named = PPOConfig(ratio_atol={"float32": 1e-4, "bfloat16": 2e-2, "tf32": 1e-3})
    assert ratio_tolerance(named, torch.float32, "cuda") == 1e-3, "a config's tf32 entry wins"
    assert ratio_tolerance(PPOConfig(), torch.bfloat16, "cuda") == 2e-2


def test_run_exact_turns_tf32_off_and_float32_keeps_its_own(tf32: Any) -> None:
    tf32(False)
    assert ratio_tolerance(PPOConfig(), torch.float32, "cuda") == 1e-4


def test_a_precision_the_config_does_not_list_is_refused_by_name(tf32: Any) -> None:
    tf32(False)
    with pytest.raises(KeyError, match="float16"):
        ratio_tolerance(PPOConfig(), torch.float16, "cuda")
