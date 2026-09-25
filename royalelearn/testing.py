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

import msgspec

from . import config as cfg
from .determinism import apply_cublas_workspace_config
from .extensions import ExtensionBase, Provider, with_sections

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Sequence

    import numpy as np

    from .api.rollout import EnvSpec, RolloutRound, SlotPlan
    from .rollout.envspec import EnvFactorySpec

__all__ = [
    "ARCH",
    "CYCLES",
    "FORCED_CELLS",
    "FREEZE_THRESHOLDS",
    "PPO_CONFIG",
    "PPO_SCHEDULE",
    "PREFLIGHT",
    "SAMPLES",
    "SEED",
    "SLOTS",
    "RecordingSGD",
    "RectFixture",
    "StubExtension",
    "StubSection",
    "StubTerm",
    "actor_state",
    "build_model",
    "collect",
    "coordinator",
    "identity_facts",
    "learner_rows",
    "mock_env_spec",
    "observations",
    "plan_for",
    "plant_forced",
    "read_env_spec",
    "round_for",
    "tiny_config",
    "update_for",
    "use_extensions",
    "with_sections",
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

    format_version = 1
    scaling = "rows"
    measure_grad_ratio = True

    def __init__(
        self,
        coefficient: float = 0.0,
        *,
        name: str = "inert",
        keep_state: bool = False,
        extension: str = "stub",
    ) -> None:
        self.extension = extension
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
        fields = {f"{self.extension}/{self.name}/calls": float(self.calls)}
        if grad_ratio is not None:
            fields[f"{self.extension}/{self.name}/grad_ratio"] = grad_ratio
        return fields

    def state(self) -> dict[str, Any]:
        return {"calls": self.calls} if self.keep_state else {}

    def load_state(self, state: Any) -> None:
        self.calls = int(state["calls"])


class StubSection(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """The section of ``StubExtension``: one ``StubTerm`` at this coefficient, or none; and a
    schedule of the actor's learning-rate scale, which turns the core's freeze on."""

    coefficient: float | None = None
    keep_state: bool = False
    actor_lr_scale: cfg.ScheduleSpec | None = None
    #: Thresholds, which belong in no identity, as every section's alarms: the freeze alarms'
    #: ``handoff_window``, ``handoff_kl``, ``handoff_clip`` and ``ev_at_unfreeze`` when the
    #: section schedules the actor's rate.
    alarms: dict[str, float] = {}


class StubExtension(ExtensionBase):
    """An extension for tests: its section adds one ``StubTerm`` and its metric keys.

    ``package`` is the package it claims to be provided by, for the tests of what the identity
    records and what is watched for edits. Without one it is RoyaleLearn's own code, and
    ``use_extensions`` registers it as a built-in: no commit to name, nothing extra to watch, so a
    test using it runs the same from a checkout, a wheel or an archive.
    """

    section_type = StubSection

    def __init__(self, name: str = "stub", *, package: Any = None) -> None:
        import royalelearn

        self.name = name
        self.own_package = package is not None
        self.package = package if package is not None else royalelearn

    def actor_terms(self, section: StubSection, ctx: Any) -> tuple[StubTerm, ...]:
        if section.coefficient is None:
            return ()
        return (
            StubTerm(section.coefficient, keep_state=section.keep_state, extension=self.name),
        )

    def actor_lr_scale(self, section: StubSection) -> Any:
        return section.actor_lr_scale

    def alarms(self, section: StubSection) -> tuple[Any, ...]:
        if section.actor_lr_scale is None:
            return ()
        from .learn.freeze import freeze_alarms

        thresholds = {**FREEZE_THRESHOLDS, **section.alarms}
        return tuple(
            freeze_alarms(
                handoff_window=int(thresholds["handoff_window"]),
                handoff_kl=thresholds["handoff_kl"],
                handoff_clip=thresholds["handoff_clip"],
                ev_at_unfreeze=thresholds["ev_at_unfreeze"],
            )
        )

    def metric_schema(self, section: StubSection) -> Any:
        from .learn.freeze import FREEZE_ALARM_KEYS
        from .metrics.schema import MetricSpec, SchemaContribution, pattern

        spec = MetricSpec(unit="count", description="A stub term's calls or gradient ratio.")
        return SchemaContribution(
            patterns=(
                pattern(f"{self.name}/{{term}}/calls", spec),
                pattern(f"{self.name}/{{term}}/grad_ratio", spec),
            ),
            alarm_metrics=FREEZE_ALARM_KEYS if section.actor_lr_scale is not None else {},
        )


#: The freeze alarms' thresholds a stub section uses unless its ``alarms`` says otherwise.
FREEZE_THRESHOLDS: dict[str, float] = {
    "handoff_window": 20,
    "handoff_kl": 0.05,
    "handoff_clip": 0.3,
    "ev_at_unfreeze": 0.3,
}


def use_extensions(monkeypatch: Any, extensions: dict[str, Any]) -> None:
    """Make ``extensions`` the providers of their names for one test, through ``monkeypatch``.

    Discovery of every other key is unchanged. What a test gets is what an installed package's
    entry point would give it, recorded as distribution ``"test"``, and nothing is written to
    the environment's install metadata. A ``StubExtension`` given no package is registered as a
    built-in, since its code is RoyaleLearn's.
    """
    from . import extensions as registry

    real = registry._discover

    def discover(names: set[str]) -> tuple[dict[str, Provider], list[str]]:
        mine = {
            name: Provider(
                extensions[name],
                "test",
                not getattr(extensions[name], "own_package", True),
            )
            for name in names & set(extensions)
        }
        found, problems = real(names - set(extensions))
        return {**found, **mine}, problems

    real_claimed = registry._claimed

    def claimed(names: set[str]) -> set[str]:
        return (names & set(extensions)) | real_claimed(names - set(extensions))

    monkeypatch.setattr(registry, "_discover", discover)
    monkeypatch.setattr(registry, "_claimed", claimed)


# --------------------------------------------------------------------------
# The rectangle and the update, without a coordinator
# --------------------------------------------------------------------------
#
# What the update's own tests build: a small experience rectangle filled with real MockEngine
# observations, played into the way the coordinator plays an iteration, and a PPO update over
# it. A package whose code runs inside the update -- an actor-loss term -- tests itself against
# the same rectangle rather than a copy of it.

#: The rectangle the update tests fill: six cycles of eight slots, four mirror battles.
CYCLES = 6
SLOTS = 8
#: Every trainable cell of the rectangle in one batch, so that "the one-batch gradient" is a
#: thing the rectangle actually has.
SAMPLES = CYCLES * SLOTS

#: One epoch over the whole rectangle as one minibatch, with the gradient clip out of the way:
#: the update the gradient tests reason about.
PPO_CONFIG = cfg.PPOConfig(
    n_epochs=1,
    timesteps_per_iteration=SAMPLES,
    batch_size=SAMPLES,
    minibatch_size=SAMPLES,
    critic_chunk=SLOTS * 2,
    max_grad_norm=1e9,
    debug_assert_iterations=1,
    check_ratio_invariant_every=0,
)


def _schedule() -> Any:
    from .api.schedule import ScheduleState

    return ScheduleState(
        iteration=0,
        cumulative_env_steps=0,
        cumulative_timesteps=0,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        ent_coef_noop=0.02,
        lr_actor=2e-4,
        lr_critic=2e-4,
    )


#: The schedule values one iteration of those tests runs at.
PPO_SCHEDULE = _schedule()

#: Five cells in every eight, so that the rectangle holds 30 forced rows and 18 choice rows.
#: 18 is not a multiple of ``SAMPLES // 4``, so under the choice-first partition the choice
#: rows run across a minibatch boundary rather than filling minibatches.
FORCED_CELLS = [
    (cycle, slot)
    for cycle in range(CYCLES)
    for slot in range(SLOTS)
    if (cycle * SLOTS + slot) % 8 < 5
]


def mock_env_spec() -> EnvFactorySpec:
    """The environment description the suite runs against: MockEngine, the shipped defaults."""
    return cfg.default_env_spec(cfg.MOCK_ENGINE)


def read_env_spec(factory: EnvFactorySpec | None = None, *, frame_stack: int = 1) -> EnvSpec:
    """The spec read off two built battles of ``factory`` (MockEngine by default)."""
    from .rollout.envspec import read_env_spec as read

    factory = factory if factory is not None else mock_env_spec()
    env = factory.build_vec(2)
    try:
        return read(env, factory, frame_stack=frame_stack)
    finally:
        env.close()


def observations(
    factory: EnvFactorySpec | None = None, *, seed: int = 3, steps: int = SAMPLES
) -> list[dict[str, np.ndarray]]:
    """Real observations, all different, to fill a rectangle with: one seat of two battles."""
    import numpy as np

    factory = factory if factory is not None else mock_env_spec()
    env = factory.build_vec(2)
    try:
        batches = [env.reset(seed=seed)[0]]
        actions = np.zeros(env.num_envs, dtype=np.int64)
        for _ in range(steps):
            batches.append(env.step(actions)[0])
    finally:
        env.close()
    return [{key: value[0] for key, value in batch.items()} for batch in batches]


def plan_for(slots: int, *, iteration: int = 0) -> SlotPlan:
    """A mirror plan: every battle is the learner against itself."""
    from .api.rollout import GROUP_LEARNER, ROLE_MIRROR, Assignment, SlotPlan

    assignments = tuple(
        Assignment(
            battle=slot // 2,
            ordinal=0,
            role=ROLE_MIRROR,
            opponent_id=None,
            group=(GROUP_LEARNER, GROUP_LEARNER),
            learner_seat=slot % 2,
        )
        for slot in range(slots)
    )
    return SlotPlan(
        iteration=iteration,
        n_battles=slots // 2,
        n_slots=slots,
        assignment=assignments,
        resident_snapshots=(),
    )


def round_for(
    cycle: int,
    slots: np.ndarray,
    *,
    rows: np.ndarray,
    group: int | None = None,
    reward: np.ndarray | None = None,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
    valid: bool = True,
    episode_end: np.ndarray | None = None,
    tick: np.ndarray | None = None,
    deploy_status: int = -1,
) -> RolloutRound:
    """One round of the named slots, every field defaulted to a quiet learner round."""
    import numpy as np

    from .api.rollout import GROUP_LEARNER, RolloutRound

    count = slots.size
    zeros = np.zeros(count, dtype=bool)
    return RolloutRound(
        cycle=cycle,
        shard=0,
        slots=slots,
        obs_rows=rows,
        group=np.full(count, GROUP_LEARNER if group is None else group, dtype=np.int8),
        reward=reward if reward is not None else np.zeros(count, dtype=np.float32),
        terminated=terminated if terminated is not None else zeros,
        truncated=truncated if truncated is not None else zeros,
        valid=np.full(count, valid, dtype=bool),
        deploy_status=np.full(count, deploy_status, dtype=np.int8),
        tick=tick if tick is not None else np.full(count, cycle, dtype=np.int32),
        episode_end=episode_end if episode_end is not None else np.zeros(count, dtype=np.int8),
    )


class RectFixture:
    """A rectangle, its codec, and the rows a worker would have written into it."""

    def __init__(
        self,
        spec: EnvSpec,
        observations: list[dict[str, np.ndarray]],
        *,
        cycles: int = CYCLES,
        slots: int = SLOTS,
        frame_stack: int = 1,
    ) -> None:
        import uuid

        from .learn.buffer import RectBuffer
        from .rollout.codec import SpatialObsCodec

        self.spec = msgspec.structs.replace(spec, frame_stack=frame_stack)
        self.codec = SpatialObsCodec()
        # A handful of rows is all a table is needed for here: it is how a row is sized and
        # packed. What a table may be decided from for a run is the codec's own tests' business.
        self.codec.table(self.spec, observations, min_states=0)
        self.observations = observations
        self.buffer = RectBuffer(
            self.spec,
            self.codec,
            run_id=uuid.uuid4().hex[:12],
            cycles=cycles,
            n_slots=slots,
        )
        self.buffer.set_static_planes(self.codec.static_planes(observations[0]))

    def fill(self, cycles: int | None = None) -> None:
        """Pack an observation into every cell of the rectangle, as a worker would."""
        layout = self.buffer.layout
        view = memoryview(self.buffer.shm.buf)[layout.obs_offset :]
        try:
            top = layout.cycles if cycles is None else cycles
            for cycle in range(top + 1):
                for slot in range(layout.n_slots):
                    index = (cycle * layout.n_slots + slot) % len(self.observations)
                    self.codec.pack(
                        self.observations[index], view, layout.row_index(cycle, slot)
                    )
        finally:
            view.release()

    def close(self) -> None:
        self.buffer.close()


def collect(built: RectFixture, model: Any, *, iteration: int = 0) -> Any:
    """One iteration played into the rectangle, the way the coordinator plays one.

    The actions and their log-probabilities come from the real inference path against the real
    stored masks, which is what makes the importance ratio at the first minibatch meaningful
    rather than a comparison against a column of zeros.
    """
    import numpy as np

    from .learn.inference import BatchedInference

    buffer = built.buffer
    plan = plan_for(SLOTS, iteration=iteration)
    buffer.begin_iteration(plan, CYCLES)
    engine = BatchedInference(buffer, model, master_seed=SEED)
    engine.begin_iteration(plan)
    slots = np.arange(SLOTS, dtype=np.int64)
    rewards = np.random.default_rng(SEED).normal(size=(CYCLES + 1, SLOTS)).astype(np.float32)
    # Every round of the iteration, the trailing one included: it carries the bootstrap
    # observation and the last cycle's reward, and without it the last row is never completed.
    for cycle in range(CYCLES + 1):
        trailing = cycle == CYCLES
        terminated = np.zeros(SLOTS, dtype=bool)
        if cycle == CYCLES - 2:
            terminated[0] = True
        played = round_for(
            cycle,
            slots,
            rows=np.array([buffer.layout.row_index(cycle, int(s)) for s in slots]),
            reward=rewards[cycle],
            terminated=terminated,
        )
        if trailing:
            buffer.record_round(played, None, None)
            continue
        answer = engine.act(played)
        buffer.record_round(played, answer.actions, answer.log_probs)
    return buffer


def plant_forced(built: RectFixture, cells: Sequence[tuple[int, int]]) -> None:
    """Repack the named cells with a mask that leaves the no-op and nothing else.

    In the stored bytes rather than in a column, because that is what makes the row forced
    everywhere at once: the rollout forward samples under this mask, the log-probability stored
    beside the action comes out of it, and the update reads the same mask back. Call it before
    the iteration is collected.
    """
    import numpy as np

    from royalegym.action import NOOP

    layout = built.buffer.layout
    view = memoryview(built.buffer.shm.buf)[layout.obs_offset :]
    try:
        for cycle, slot in cells:
            index = (cycle * layout.n_slots + slot) % len(built.observations)
            observation = dict(built.observations[index])
            mask = np.zeros_like(np.asarray(observation["action_mask"]))
            mask[NOOP] = 1
            observation["action_mask"] = mask
            built.codec.pack(observation, view, layout.row_index(cycle, slot))
    finally:
        view.release()


def update_for(model: Any, config: cfg.PPOConfig = PPO_CONFIG, **kwargs: Any) -> Any:
    """A PPO update over a scaler that does not standardise, so a reward is its own number."""
    from .learn.gae import GAE
    from .learn.ppo import PPOUpdate

    return PPOUpdate(model, GAE(standardize_rewards=False), config, master_seed=SEED, **kwargs)


def RecordingSGD(params: Any, **kwargs: Any) -> Any:
    """An optimizer that records the gradient it was handed and changes nothing.

    A real optimizer would answer a different question: Adam divides by the square root of the
    second moment, so two gradients that agree to a part in a million can still produce
    parameters that differ by the whole step size where a coordinate's gradient is near zero.
    What has to be equal is the gradient. Its ``recorded`` list holds one flat gradient a step.
    """
    return _recording_class()(params, **kwargs)


def _recording_class() -> Any:
    import torch

    class _Recording(torch.optim.SGD):
        def __init__(self, params: Any, **kwargs: Any) -> None:
            super().__init__(list(params), lr=float(kwargs.get("lr", 0.0)))
            self.recorded: list[torch.Tensor] = []

        def step(self, closure: Any = None) -> None:  # type: ignore[override]
            flat = [
                parameter.grad.reshape(-1)
                if parameter.grad is not None
                else torch.zeros(parameter.numel())
                for group in self.param_groups
                for parameter in group["params"]
            ]
            self.recorded.append(torch.cat(flat).clone())

    return _Recording
