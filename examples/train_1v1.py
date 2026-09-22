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


def main() -> None:
    config = load_config(Path(__file__).parent / "configs" / "laptop.json")
    config.run_name = "my-first-run"
    # The number to think about: 0.99 gives a credit horizon of about 45 seconds, which is the
    # deploy-push-tower causal chain. Every run prints its horizon at start-up.
    config.advantage = replace(config.advantage, gae_lambda=0.99)
    # Weights and Biases, off by default because it needs an account.
    config.metrics = replace(
        config.metrics,
        sinks=[replace(sink, enabled=True) if sink.kind == "wandb" else sink
               for sink in config.metrics.sinks],
    )
    with LearningCoordinator(config) as run:
        run.learn(until_timesteps=100_000_000)


if __name__ == "__main__":
    main()
