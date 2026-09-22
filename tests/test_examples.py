"""The shipped examples run on a fresh install.

``examples/train_1v1.py`` is the first file a newcomer runs. Weights and Biases is an optional
extra that a fresh install does not have, and its sink raises at construction when enabled and
absent, so an example that turns it on fails before the first iteration for everyone who has
not installed and logged into it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_1v1_leaves_the_optional_wandb_sink_off(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load("train_1v1")
    seen: dict[str, Any] = {}

    class Recorder:
        def __init__(self, config: Any) -> None:
            seen["config"] = config

        def __enter__(self) -> Recorder:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def learn(self, until_timesteps: int) -> None:
            seen["until"] = until_timesteps

    monkeypatch.setattr(module, "LearningCoordinator", Recorder)
    module.main()
    enabled = [s.kind for s in seen["config"].metrics.sinks if s.enabled]
    assert "wandb" not in enabled, enabled
    assert "jsonl" in enabled
