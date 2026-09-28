"""The file a bot creator runs, and the handful of lines they actually change.

Everything else about a run lives in ``configs/laptop.json``, which ``royalelearn config``
writes out in full. The blocks below are frozen structs, so a number is changed by replacing
the block that holds it: the run identity is computed from these values once, at start-up, and
a field that could be edited after that would be a run whose recorded configuration is not the
one it ran.

    python examples/train_1v1.py
"""

from __future__ import annotations

from pathlib import Path

from msgspec.structs import replace

from royalelearn import LearningCoordinator, load_config
from royalelearn.determinism import apply_cublas_workspace_config

# Weights and Biases is off unless you turn it on here. It is not installed with the package:
# it needs `pip install wandb` and a W&B account first. Metrics always go to the run folder's
# metrics.jsonl and to the console either way.
USE_WANDB = False


def main() -> None:
    config = load_config(Path(__file__).parent / "configs" / "laptop.json")
    config.run_name = "my-first-run"
    # The number to think about: 0.99 lengthens the credit horizon, how far back a reward reaches,
    # toward the deploy-push-tower causal chain. Every run prints its horizon, in seconds, at
    # start-up.
    config.advantage = replace(config.advantage, gae_lambda=0.99)
    if USE_WANDB:
        config.metrics = replace(
            config.metrics,
            sinks=[replace(sink, enabled=True) if sink.kind == "wandb" else sink
                   for sink in config.metrics.sinks],
        )
    # The command line sets this before the coordinator starts; a script has to, even on a CPU.
    apply_cublas_workspace_config()
    with LearningCoordinator(config) as run:
        run.learn(until_timesteps=100_000_000)


if __name__ == "__main__":
    main()
