"""``rollout.eval_device``: the gate's workers run their actors on the GPU.

On the processor a worker plays one battle at a time with a batch of one per decision, which
for a network the size of a real run's is tens of milliseconds a forward: a gate of a thousand
battles held a run's update for hours while its GPU sat idle. The suite has no GPU, so what runs
here is the setting's path: the config, the identity, and the line a CPU run prints instead.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from royalelearn.config import RolloutConfig, RunConfig, check_consistency, eval_device_of
from royalelearn.identity import EXCLUDED_FROM_IDENTITY
from royalelearn.testing import coordinator, tiny_config


def test_unset_nothing_is_encoded_and_it_is_the_processor() -> None:
    assert b"eval_device" not in msgspec.json.encode(RolloutConfig())
    assert eval_device_of(RolloutConfig()) == "cpu"


def test_a_device_nothing_can_run_on_is_named() -> None:
    config = RunConfig(rollout=RolloutConfig(eval_device="gpu"))
    assert any("rollout.eval_device is 'gpu'" in p for p in check_consistency(config))


def test_it_is_a_performance_setting_and_out_of_the_identity() -> None:
    assert "rollout.eval_device" in EXCLUDED_FROM_IDENTITY


def test_a_run_on_the_processor_says_its_workers_stay_there(tmp_path: Path) -> None:
    base = tiny_config(tmp_path)
    config = tiny_config(
        tmp_path,
        rollout=msgspec.structs.replace(base.rollout, eval_workers=2, eval_device="cuda"),
    )
    lines: list[str] = []
    with coordinator(config, printer=lines.append) as run:
        farm = run.eval_player
        assert farm.config.device == "cpu"
    assert any(line.startswith("eval device   cpu: rollout.eval_device cuda") for line in lines)


@pytest.mark.gpu
def test_on_a_gpu_the_workers_config_names_the_runs_device(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    base = tiny_config(tmp_path)
    config = tiny_config(
        tmp_path,
        net=msgspec.structs.replace(base.net, device="cuda"),
        rollout=msgspec.structs.replace(base.rollout, eval_workers=2, eval_device="cuda"),
    )
    with coordinator(config) as run:
        assert run.eval_player.config.device == str(run.device)
