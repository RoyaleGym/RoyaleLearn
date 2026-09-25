"""Test support for RoyaleLearn and for anything built on it. Public, and free of pytest.

What a test of the harness needs over and over -- the smallest run that is still a run, an entered
coordinator, a network at test size, the facts an identity is computed from, rows a run collected
-- is here rather than in ``tests/``, so that a package built on RoyaleLearn tests against the same
definitions instead of importing another repository's test folder. Nothing here imports pytest;
torch is imported where a helper needs it and not before.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import config as cfg
from .determinism import apply_cublas_workspace_config

if TYPE_CHECKING:  # pragma: no cover - annotations only
    import numpy as np

    from .api.rollout import EnvSpec

__all__ = [
    "ARCH",
    "PREFLIGHT",
    "SEED",
    "StubTerm",
    "actor_state",
    "build_model",
    "coordinator",
    "identity_facts",
    "learner_rows",
    "tiny_config",
]

#: What preflight is told for a test: every gate still runs, on a sample a test can afford.
PREFLIGHT = {"table_samples": 32, "mask_samples": 32, "min_table_states": 0}

#: The master seed the unit tests build networks from.
SEED = 20260921

#: The shipped pairing at the smallest size, on the CPU and in float32.
ARCH = cfg.NetConfig(
    channels=8,
    blocks=1,
    norm_groups=4,
    vector_embed=4,
    value_hidden=16,
    card_embed=8,
    device="cpu",
    autocast_dtype="float32",
)


def tiny_config(tmp_path: Path, **overrides: Any) -> cfg.RunConfig:
    """The smallest run that is still a run: MockEngine, one worker, two battles, four cycles.

    The mixture is mirror-only. That is not a simplification for its own sake: the learner-row
    count of a mirror battle is exactly two, so the rectangle collects exactly what the config
    asks for, and the per-iteration invariant that says so can be asserted rather than given a
    tolerance that only means anything over thousands of battles.
    """
    settings: dict[str, Any] = {
        "run_name": "test",
        "runs_dir": str(tmp_path / "runs"),
        "master_seed": 4242,
        "env": cfg.default_env_spec(cfg.MOCK_ENGINE, max_steps=6),
        "rollout": cfg.RolloutConfig(
            source="inline",
            workers=1,
            games_per_worker=2,
            shards_per_worker=1,
            launch_delay_s=0.0,
        ),
        "net": cfg.NetConfig(
            channels=8,
            blocks=1,
            norm_groups=4,
            card_embed=8,
            value_hidden=16,
            autocast_dtype="float32",
            device="cpu",
        ),
        # ``net.autocast_dtype`` is torch's spelling and ``ppo.ratio_atol``'s keys are the
        # update's names for the same precisions, which agree on bfloat16 and not on float32.
        # Naming both is what lets a CPU run assert the ratio invariant at fp32's tolerance.
        "ppo": cfg.PPOConfig(
            n_epochs=2,
            timesteps_per_iteration=16,
            batch_size=8,
            minibatch_size=4,
            ratio_atol={"fp32": 1e-4, "float32": 1e-4, "bfloat16": 2e-2},
        ),
        "ladder": cfg.LadderConfig(
            mix=(1.0, 0.0, 0.0),
            candidate_every_env_steps=1_000_000,
            floor_admit_every_env_steps=1_000_000,
            eval_seed_count=4,
            refit_every_iterations=1_000_000,
        ),
        "checkpoint": cfg.CheckpointConfig(every_env_steps=1_000_000, keep=2),
        "metrics": cfg.MetricsConfig(sinks=[cfg.SinkSpec("jsonl")]),
        "determinism": cfg.DeterminismConfig(tier="throughput"),
        # The exhaustive mask gate is four thousand engine queries and is what
        # ``tests/test_env_contract.py`` and ``royalelearn doctor`` are for; a file that builds
        # twenty coordinators pays for it twenty times and learns nothing new after the first.
        "doctor": cfg.DoctorConfig(run_mask_disagreement_gate=False),
    }
    settings.update(overrides)
    return cfg.RunConfig(**settings)


@contextlib.contextmanager
def coordinator(config: cfg.RunConfig, **kwargs: Any) -> Iterator[Any]:
    """A coordinator, entered, with the entry point's own environment already applied."""
    from royalelearn.coordinator import LearningCoordinator

    apply_cublas_workspace_config()
    kwargs.setdefault("preflight_kwargs", PREFLIGHT)
    kwargs.setdefault("printer", None)
    kwargs.setdefault("install_signal_handler", False)
    run = LearningCoordinator(config, **kwargs)
    with run:
        yield run


def build_model(spec: EnvSpec, *, seed: int = SEED) -> Any:
    """The shipped actor-critic at ``ARCH``'s size, on the CPU and in float32."""
    from .learn.nets import DefaultNetworkFactory

    return DefaultNetworkFactory(seed).build(spec, ARCH, "cpu")


def identity_facts(env_spec: EnvSpec) -> dict[str, Any]:
    """The facts ``compute_identity`` takes beyond the config, fixed, for a MockEngine run."""
    from . import identity as I

    return {
        "env_spec": env_spec,
        "build": I.EngineBuild(
            engine_class="royalegym.mock_engine.MockEngine",
            calibration_digest="0" * 16,
            build_digest="0" * 16,
            catalogue_sha256="c" * 64,
            path_search=None,
            stale_build_differences=[],
            binary_sha256=I.NOT_STATED,
        ),
        "arch_digest": "a" * 64,
        "codec_version": 1,
        "codec_table_digest": "b" * 64,
        "torch_version_string": "2.11.0+cu128",
        "device_kind": "cpu:x86_64",
    }


def learner_rows(run: Any, count: int) -> np.ndarray:
    """Up to ``count`` packed rows a run collected for its own seats, after an iteration."""
    import numpy as np

    buffer = run.buffer
    cycles, slots = np.nonzero(buffer.trainable())
    take = min(count, int(cycles.size))
    rows, _live = buffer._stack_rows(cycles[:take], slots[:take])
    return np.ascontiguousarray(buffer.obs_view[rows[:, 0]])


def actor_state(run: Any) -> dict[str, Any]:
    """A detached copy of a run's actor tensors, as they are right now."""
    return {name: tensor.detach().clone() for name, tensor in run.model.actor.state_dict().items()}


class StubTerm:
    """An actor-loss term that runs the whole path an extension's term runs, at a coefficient the
    test chooses.

    Its raw value is minus the summed log-probability of the legal actions on the minibatch's
    choice rows: a sum over the rows, so ``scaling`` is ``"rows"``, with a gradient wherever the
    policy has one. At coefficient zero the update must be the one without the term, bit for bit;
    that is what shows the host adds nothing of its own around a term. It reports how often it
    was called and, when the update measured one, its gradient ratio, under ``stub/<name>/``.
    ``keep_state`` makes it keep its call count in the checkpoint, for tests of the state layout.
    """

    extension = "stub"
    format_version = 1
    scaling = "rows"
    measure_grad_ratio = True

    def __init__(
        self, coefficient: float = 0.0, *, name: str = "inert", keep_state: bool = False
    ) -> None:
        self.name = name
        self.coefficient = float(coefficient)
        self.keep_state = keep_state
        self.calls = 0

    def begin(self, env_steps: int, device: Any) -> None:
        pass

    def loss(self, inputs: Any, *, epoch: int, measure: bool) -> tuple[float, Any]:
        import torch

        legal = torch.where(inputs.mask, inputs.log_probs, torch.zeros_like(inputs.log_probs))
        self.calls += 1
        return self.coefficient, -legal.sum()

    def finish(
        self,
        *,
        iteration: int,
        actor_trained: bool,
        explained_variance: float,
        grad_ratio: float | None,
    ) -> dict[str, float]:
        fields = {f"stub/{self.name}/calls": float(self.calls)}
        if grad_ratio is not None:
            fields[f"stub/{self.name}/grad_ratio"] = grad_ratio
        return fields

    def state(self) -> dict[str, Any]:
        return {"calls": self.calls} if self.keep_state else {}

    def load_state(self, state: Any) -> None:
        self.calls = int(state["calls"])
