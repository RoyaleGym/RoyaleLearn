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


@pytest.mark.parametrize(("drift", "trips"), [(2e-2, True), (2e-3, False)])
def test_a_real_drift_still_trips_the_check_under_tf32(
    tf32: Any, tmp_path: Any, drift: float, trips: bool
) -> None:
    """The looser tolerance hides the precision and nothing else: a ratio 2e-2 off one -- a mask,
    codec or weight-version mismatch moves it by far more -- is still a refusal on a TF32 GPU,
    and 2e-3, which TF32 can produce, is not."""
    from royalelearn.learn.ppo import _Diagnostics
    from royalelearn.testing import coordinator, tiny_config

    tf32(True)
    with coordinator(tiny_config(tmp_path)) as run:
        update = run.update
        update.device = torch.device("cuda")  # the tolerance is asked about a GPU run
        assert update._ratio_atol() == TF32_RATIO_ATOL
        ratio = torch.ones(4)
        ratio[2] += drift
        cells = torch.arange(4, dtype=torch.int64)
        actions = torch.zeros(4, dtype=torch.int64)
        diagnostics = _Diagnostics(1, torch.device("cpu"))
        update._n_slots = 4
        if trips:
            with pytest.raises(AssertionError, match="importance ratio deviates"):
                update._ratio_invariant(ratio, cells, actions, diagnostics, 0)
        else:
            update._ratio_invariant(ratio, cells, actions, diagnostics, 0)
