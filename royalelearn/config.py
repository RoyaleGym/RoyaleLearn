"""The whole configuration of a run, as one tree.

There is one source of truth. The config object is it: no keyword-argument facade duplicates its
leaves, because two sources of truth for one number is the kind of variable this harness exists
to remove. Every struct forbids unknown fields, so a typo is an error at load rather than a
setting that silently does nothing, and every number carries its unit in ``docs/harness-spec.md``
section 6 beside the reason it has the value it has.

msgspec rather than pydantic because it is already a RoyaleGym dependency, it gives forbidden
unknown fields, tagged unions and a canonical JSON round-trip, and the package's dependency list
stays ``royalegym, numpy, msgspec``.

Profiles are functions rather than files: one per machine class, differing only in numbers, and
a JSON config names one and overrides what it likes on top.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import msgspec

from .determinism import TIERS
from .errors import PreflightError
from .rollout.envspec import ComponentSpec, EnvFactorySpec, JsonValue, canonical_json, digest_of

__all__ = [
    "MOCK_ENGINE",
    "PROFILES",
    "RUST_ENGINE",
    "TAG_FIELD",
    "AdvantageConfig",
    "AlarmConfig",
    "ArchSpec",
    "CheckpointConfig",
    "ConstantSpec",
    "DeterminismConfig",
    "DoctorConfig",
    "GateConfig",
    "GeometricSpec",
    "Geometry",
    "LadderConfig",
    "LinearSpec",
    "LrBackoffConfig",
    "MetricsConfig",
    "NetConfig",
    "ObsConfig",
    "PPOConfig",
    "PiecewiseConstantSpec",
    "RaterConfig",
    "RolloutConfig",
    "RunConfig",
    "ScheduleSpec",
    "SinkSpec",
    "check_consistency",
    "config_hash",
    "default_env_spec",
    "dump_config",
    "geometry",
    "laptop",
    "load_config",
    "many_core",
    "profile",
    "validate",
    "workstation",
]

RUST_ENGINE = "royalegym.rust_engine.RustEngine"
MOCK_ENGINE = "royalegym.mock_engine.MockEngine"

#: What ``ppo.forced_rows`` may say, in the order the two-phase A/B compares them. Section 18.1
#: of ``docs/harness-spec.md`` has what each one does and the rule for moving between them.
FORCED_ROW_ARMS = ("all", "critic_only", "critic_only_choice_mean")


# --------------------------------------------------------------------------
# Schedules, as data
# --------------------------------------------------------------------------


class ConstantSpec(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True, tag_field="kind", tag="constant"
):
    """A value that does not move."""

    value: float


class LinearSpec(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True, tag_field="kind", tag="linear"
):
    """``start`` to ``end`` over ``over_env_steps`` game-steps, then ``end`` forever."""

    start: float
    end: float
    over_env_steps: int


class GeometricSpec(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True, tag_field="kind", tag="geometric"
):
    """``start`` to ``end`` geometrically, which is what a discount wants: ``1 - gamma`` decays
    by a constant factor rather than gamma moving by a constant amount."""

    start: float
    end: float
    over_env_steps: int


class PiecewiseConstantSpec(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True, tag_field="kind", tag="piecewise"
):
    """``(env_steps, value)`` breakpoints, held between them."""

    points: list[tuple[int, float]]


ScheduleSpec = ConstantSpec | LinearSpec | GeometricSpec | PiecewiseConstantSpec

#: The field every tagged union in the tree discriminates on, read off one of them rather than
#: retyped, because ``_merge`` has to know a variant from a variant.
TAG_FIELD: str = ConstantSpec.__struct_config__.tag_field


# --------------------------------------------------------------------------
# The tree
# --------------------------------------------------------------------------


class ObsConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """What the policy sees beyond the current frame.

    The observation carries positions and no motion, so a policy that needs to tell an advancing
    unit from a retreating one needs more than one frame. Stacking is a gather of buffer rows
    and costs no extra storage; it is 1 by default because the extra channels cost stem compute
    and the first run measures whether they buy anything.
    """

    frame_stack: int = 1


class RolloutConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The shape and the failure behaviour of the worker farm."""

    source: str = "process"
    workers: int = 3
    games_per_worker: int = 32
    #: A worker steps one shard while the parent infers on the other; worth about 25% of round
    #: time and costs nothing.
    shards_per_worker: int = 2
    #: Microseconds to spin before blocking on a round event. An event round trip measures tens
    #: of microseconds, a spin about two.
    spin_us: int = 100
    #: EVERY wait has a timeout; a timeout is a typed WorkerFailure, never a block.
    round_timeout_s: float = 30.0
    restart_failed_workers: bool = True
    #: Three restarts of one worker in a run raises rather than silently degrading throughput.
    max_restarts_per_worker: int = 3
    #: Seconds between worker starts: several engine constructions at once each decode the
    #: calibration and arena data.
    launch_delay_s: float = 0.5
    #: Desynchronise episode phase across battles at run start, so episode ends spread across
    #: cycles instead of arriving in one spike.
    stagger_first_reset: bool = True
    #: Lag-1 collection under the update. Costs a second rectangle, so it is off on 8 GB.
    #: Spec 14.1's overlapped collection. REFUSED by ``check_consistency`` while true, because
    #: nothing honours it and the preflight charges a second rectangle for it. It stays in the
    #: config rather than being deleted so that a config carrying it gets a refusal naming the
    #: section, instead of ``forbid_unknown_fields`` rejecting the whole file over one line.
    overlap: bool = False
    eval_workers: int = 2
    eval_games_per_worker: int = 24
    #: Saves completed battles to disk so they can be watched afterwards, which is the only form
    #: that works: ``time/collection`` is about 5% of an iteration and those seconds hold tens of
    #: thousands of engine ticks, so a live view is a firehose between silences. NOT part of
    #: ``EnvFactorySpec``, deliberately: a recorder does not change the game, and putting it in
    #: the hashed spec would make a policy trained with one incomparable with every policy
    #: trained without. It reaches exactly one env of one shard of worker 0 -- see
    #: ``EnvFactorySpec.factory`` for why one rather than all of them.
    recorder: ComponentSpec | None = None


class NetConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The architecture. Hashed into ``arch_digest``, so a snapshot built from another one is
    refused rather than shape-errored."""

    channels: int = 64
    blocks: int = 4
    #: GroupNorm, never BatchNorm: BatchNorm computes a different function at rollout than at
    #: update, which breaks the stored-log-prob contract.
    norm_groups: int = 8
    #: The scalar vector is broadcast into the map, so elixir can modulate every tile.
    vector_embed: int = 32
    value_hidden: int = 256
    #: Must equal ``channels``: it is one side of the pointer head's inner product.
    card_embed: int = 64
    #: The arena is not translation-invariant: own half, enemy half, the river, the bridges and
    #: the tower rects are all absolute.
    coord_conv: bool = True
    #: Makes "the mask and the entropy bonus never touch the critic" structural.
    separate_trunks: bool = True
    policy_head: str = "pointer"
    logit_scale: str = "rsqrt_c"
    #: Orthogonal, gain sqrt(2) hidden, 0.01 policy head, 1.0 value head.
    init: str = "orthogonal"
    #: Self-limiting at init; section 8.4 has the formula for when it is needed.
    noop_bias: float = 0.0
    #: bf16 keeps fp32's exponent range, so masking with ``finfo.min`` is safe and no GradScaler
    #: is needed.
    autocast_dtype: str = "bfloat16"
    device: str = "cuda"


#: The spec's name for the same struct, used where it is the network's description rather than a
#: section of the config file.
ArchSpec = NetConfig


class LrBackoffConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Drop the learning rate when the KL stays above a threshold."""

    kl_threshold: float = 0.02
    patience: int = 3
    factor: float = 0.5
    lr_min: float = 2e-5


class PPOConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The update."""

    #: The family disagrees by three orders of magnitude, so none of them is evidence; 3 balances
    #: reuse against the update being the wall-clock bottleneck. Watch clip fraction by epoch.
    n_epochs: int = 3
    timesteps_per_iteration: int = 32_768
    #: Samples per optimizer step.
    batch_size: int = 4_096
    #: Samples per forward. A PURE MEMORY KNOB: gradients accumulate weighted by n/batch_size
    #: with one optimizer step per batch, and a test proves the accumulated gradient equals the
    #: full-batch one.
    #:
    #: 256 is the laptop's, and it is here rather than in ``laptop()`` because every default in
    #: this tree is the laptop column; the larger profiles set their own. It was 512, chosen as
    #: the largest that fits 4 GB, and it does not fit: measured on a 4 GB card, 512 reserved
    #: 99% of it and the update took 180-233 s as the driver backed the overflow with host RAM,
    #: where 256 held 48 s within 2.3%. The cliff belongs to the footprint against the card,
    #: not to the number, which is why the workstation keeps 2048 (spec section 6).
    minibatch_size: int = 256
    clip_range: float = 0.2
    #: For a negative advantage the standard minimum does not bound the loss below, and in a
    #: wide masked space a rarely-sampled action's ratio can be enormous.
    dual_clip_c: float = 3.0
    #: Separate networks and separate optimizers, so this only scales the critic's effective
    #: learning rate.
    vf_coef: float = 1.0
    value_clipping: bool = False
    ent_coef: ScheduleSpec = LinearSpec(0.01, 0.003, 30_000_000)
    #: A warm-start guard against no-op collapse, applied to a binary entropy bounded by 0.693
    #: nats. It anneals to zero because H2(p_noop) is maximised at 0.5 while a healthy policy
    #: sits near 0.94, so a constant coefficient would bias every converged policy toward
    #: overplaying. The alarms outlive the schedule.
    ent_coef_noop: ScheduleSpec = LinearSpec(0.02, 0.0, 10_000_000)
    #: Applied to the actor and the critic parameter sets SEPARATELY.
    max_grad_norm: float = 0.5
    lr_actor: float = 2e-4
    lr_critic: float = 2e-4
    #: The PPO paper's epsilon, not torch's 1e-8.
    adam_eps: float = 1e-5
    lr_backoff: LrBackoffConfig = LrBackoffConfig()
    #: Once per ITERATION, over all trainable cells, before any split.
    advantage_standardization: bool = True
    #: Which rows the ACTOR's forward and backward run on. The critic and the advantage
    #: recursion read every row under all three values; this is about the policy alone.
    #:
    #: On this environment most collected decisions leave exactly one legal action, because the
    #: elixir bar can afford nothing, and such a row contributes exactly zero to each of the
    #: actor's three numerators while occupying a full row of every epoch.
    #:
    #: ``all``          -- the actor runs on every trainable row. The update every run so far
    #:                     has taken, and the default until the A/B of section 18.1 decides.
    #: ``critic_only``  -- the actor runs on the rows that had a choice, with the batch still
    #:                     its denominator. The gradient is ``all``'s to rounding; what changes
    #:                     is how long it takes to compute.
    #: ``critic_only_choice_mean``
    #:                  -- the same rows, with the actor's population changed to them for every
    #:                     statistic its loss uses. That is a change to the effective learning
    #:                     rate rather than to the speed, and it is what stops the actor's step
    #:                     following a fraction the elixir economy sets.
    #:
    #: The two skipping values need ``net.separate_trunks``; ``check_consistency`` says why.
    forced_rows: str = "all"
    #: Rows per chunk of the whole-iteration critic pass; unchunked it is a multi-gigabyte spike.
    critic_chunk: int = 1024
    #: The frozen seat's log-probs came from other weights; V-trace is the wrong tool against a
    #: multi-generation-old snapshot.
    discard_opponent_rows: bool = True
    #: The buffer is cleared every iteration: each timestep is trained on exactly n_epochs times
    #: and discarded. A rolling window is available at a linear memory cost.
    keep_previous_iterations: int = 0
    #: The two mask asserts and the ratio invariant run for this many iterations from a start.
    debug_assert_iterations: int = 10
    check_ratio_invariant_every: int = 50
    #: Selected by ``net.autocast_dtype``. Neither 0.0 nor 1e-4 is claimed under autocast: the
    #: rollout and update forwards are different batch shapes, cuDNN picks a kernel per shape,
    #: and bf16 carries three significant digits.
    ratio_atol: dict[str, float] = msgspec.field(
        default_factory=lambda: {"float32": 1e-4, "bfloat16": 2e-2}
    )


class AdvantageConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The discount and the credit horizon.

    0.99 discounts the win to almost nothing by the end of regulation; the anneal follows
    OpenAI Five's, with an endpoint scaled to a four-minute match. ``gae_lambda`` at 0.99 gives
    a credit horizon of about 45 seconds, which matches the deploy-push-tower causal chain --
    the references' 0.95 gives ten and cannot connect a deploy to the tower it takes.
    """

    gamma: ScheduleSpec = GeometricSpec(0.997, 0.999, 20_000_000)
    gae_lambda: float = 0.99
    #: A discount near one inflates return magnitudes roughly tenfold over 0.99.
    standardize_rewards: bool = True
    #: Standard deviations, independent of the standardisation.
    reward_clip: float = 10.0


class GateConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The three conditions a candidate must pass to become the champion."""

    #: 500 seeds times 2 side assignments.
    champion_games: int = 1000
    #: A score rate whose lower bound must clear this: an observed 55.2% at n=1000, about 35 Elo.
    champion_lower_bound: float = 0.52
    anchor_games: int = 200
    anchor_tolerance_pp: float = 2.0
    stratified_snapshots: int = 8
    stratified_games: int = 100
    bootstrap_resamples: int = 10_000


class RaterConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The Bradley-Terry-Davidson fit."""

    #: Weakly informative; keeps the fit finite for a player with no losses.
    prior_sd: float = 400.0
    #: The gauge exists from the first game, before any snapshot.
    anchor: str = "scripted:noop"
    #: "davidson" or "half_win"; fall back if the measured draw rate is under 2%.
    draws: str = "davidson"


class LadderConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Who the learner plays, and when a snapshot joins the pool."""

    #: mirror / pool / scripted.
    mix: tuple[float, float, float] = (0.50, 0.35, 0.15)
    #: At most four batched forwards per shard-round, and an LRU that never thrashes.
    max_resident_opponents: int = 2
    #: Training wants opponents that still beat the learner; evaluation uses "variance".
    pfsp_weighting: str = "hard"
    pfsp_power: float = 2.0
    #: The anti-forgetting guard AlphaStar's forgotten-players slice does.
    pfsp_uniform_floor: float = 0.2
    weight_floor_scale: float = 0.01
    candidate_every_env_steps: int = 4_000_000
    #: A plateau cannot starve the pool.
    floor_admit_every_env_steps: int = 50_000_000
    #: Sampling is linear in the pool per episode, in python.
    pool_working_size: int = 48
    eval_seed_count: int = 500
    #: Rating both the sampled and the argmax variant doubles the pool and the cost for no
    #: decision.
    release_mode: str = "stochastic"
    refit_every_iterations: int = 10
    #: Iterations between probes of the LIVE policy against the fixed rungs below. Zero is off,
    #: and off is the default: a probe plays real battles in the parent, so turning it on costs
    #: wall clock that no existing run was paying. Every other evaluation in the harness is a
    #: frozen snapshot against something, which is why without this a run publishes no
    #: measurement of the policy it is actually training.
    probe_every_iterations: int = 0
    #: Battles per rung per probe, paired: ``probe_games // 2`` seeds, each played from both
    #: sides. It is a wide instrument and the width is measured, not assumed. At 40 (20 seeds)
    #: the published interval is 20.0 points at a score of 0.750 and 22.5 at 0.600
    #: (``bootstrap_interval([1.0]*10 + [0.5]*10, derive_generator(1, "a/b"), 10000)`` gives
    #: [0.650, 0.850]), and five probes of an UNCHANGED policy against ``scripted:random_legal``
    #: read 0.375, 0.275, 0.225, 0.400, 0.350 -- a spread of 17.5 points with nothing moving,
    #: because each probe is a new comparison id and so a new acting stream **[M]**. So a
    #: probe-to-probe change under about 18 points against that rung is the instrument, not the
    #: policy. Against ``scripted:noop`` the same five probes were identical, because that rung
    #: does not act, which is what makes it the readable one. Measured 2026-09-22 by the
    #: integrator. Raise this to narrow it; the gate, at 1000 battles, is what measures a small
    #: improvement.
    probe_games: int = 40
    #: The rungs. Scripted ids, because the point of a rung is that it never changes: a snapshot
    #: improves with the pool and a score against it says nothing on its own.
    probe_opponents: tuple[str, ...] = ("scripted:noop", "scripted:random_legal")
    #: All three of these are in the run identity, and they are meant to be, although the
    #: integrator measured that two runs differing only in them produce the same weights at every
    #: iteration. That measurement is about the WEIGHTS and the identity's question is wider: a
    #: run that probes plays battles the other does not, writes games into the shared result log
    #: under its own ids, draws on the same frozen seed set and takes materially longer. Two such
    #: runs are two experiments, and saying so is what the identity is for.
    gate: GateConfig = GateConfig()
    rater: RaterConfig = RaterConfig()


class CheckpointConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    every_env_steps: int = 2_000_000
    keep: int = 10
    #: The iteration boundary is the resume point; there is no mid-iteration state to preserve.
    include_buffer: bool = False
    strict_load: bool = True


class SinkSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One metrics sink. ``JsonlSink`` is always installed: it is the file the resume test
    compares."""

    kind: str
    enabled: bool = True
    options: dict[str, JsonValue] = {}


class MetricsConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    sinks: list[SinkSpec] = msgspec.field(
        default_factory=lambda: [
            SinkSpec("jsonl"),
            SinkSpec("console"),
            SinkSpec("viser"),
            SinkSpec("wandb", enabled=False),
        ]
    )
    #: Iterations between per-card tile heatmaps.
    image_every: int = 50
    #: After this many iterations, ``episodes.jsonl`` is gzipped in place.
    keep_episode_log_iterations: int = 200


class AlarmConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Thresholds for the alarm table of section 13.3.

    Recorded and excluded from the run identity: an alarm can stop a run and can never alter a
    value, so two runs that differ only here produce the same numbers for as long as both run.
    The severities and patiences live with the alarms themselves; the overrides below are for an
    operator who wants one of them louder or quieter on their own machine.
    """

    enabled: bool = True
    #: ``ratio_invariant`` fires above this multiple of the configured ``ppo.ratio_atol``, which
    #: is what keeps the alarm meaningful in fp32 and under bf16 alike.
    ratio_invariant_multiple: float = 5.0
    buffer_fill_frac: float = 0.98
    clip_fraction: float = 0.5
    kl_high: float = 0.05
    kl_dead: float = 1e-5
    explained_variance: float = 0.0
    cards_per_match_warn: float = 8.0
    cards_per_match_halt: float = 3.0
    noop_entropy_floor: float = 0.02
    tile_top1_share: float = 0.25
    card_tile_top10_share: float = 0.5
    draw_rate: float = 0.5
    episode_steps_at_cap_frac: float = 0.8
    transitivity_residual: float = 0.10
    capacity_ratio: float = 1.5
    gate_failures: int = 5
    disabled: list[str] = []
    patience_overrides: dict[str, int] = {}
    severity_overrides: dict[str, str] = {}


class DeterminismConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Section 5.1. The tier is part of the run identity; the thread count is part of the tier."""

    tier: str = "run_exact"
    #: One, and deliberately. A BLAS thread count changes the order floats are reduced in, so a
    #: two-threaded update is not bit-reproducible against a one-threaded one and ``run_exact``
    #: would stop meaning what it says. This is the price of the tier rather than an oversight,
    #: and it is the first thing somebody meeting a single-threaded update on a many-core box
    #: will reach for: raise it with ``tier = "throughput"``, which says in the run identity
    #: that the run is no longer reproducible, rather than on its own.
    torch_threads: int = 1


class DoctorConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The start-up gates that can refuse a run."""

    #: Of the machine's total. A run whose projection exceeds this does not start.
    ram_budget_mb: int = 6500
    #: Exhaustive over every non-no-op action, both teams.
    run_mask_disagreement_gate: bool = True
    #: Device memory that must remain free after one minibatch's measured peak, or the run does
    #: not start. Zero disables the gate.
    #:
    #: It is a measurement rather than a projection, and it is compared against what the DRIVER
    #: reports free rather than against the card's capacity, because those differ by whatever
    #: else is resident and that term is the one that decides the outcome. Measured 2026-09-22:
    #: minibatch 512 reserved 4243 MB against a 4294 MB card and ran 4.5x slow rather than
    #: failing, because Windows backs an oversubscribed allocation with host RAM over PCIe
    #: instead of refusing it. A cap on the card's own total would have PASSED that
    #: configuration -- 4243 is under 4294 -- and it spilled anyway, on the other processes'
    #: share. Only free-at-startup sees the term that matters.
    #:
    #: The margin exists because free memory is not a promise: another process can take it
    #: later. That is the argument for failing here rather than for a hard allocation cap, which
    #: would convert this into an out-of-memory error at an unpredictable hour of a long run --
    #: strictly worse than the slow run it replaced, because a legible penalty becomes an
    #: illegible late crash.
    vram_headroom_mb: int = 256


def default_env_spec(engine: str = RUST_ENGINE, *, max_steps: int = 480) -> EnvFactorySpec:
    """The environment a run uses unless it says otherwise.

    The truncation is a step limit rather than a tick limit because it is decisions that the
    buffer counts; 480 of them is a full regulation match plus overtime at the default decision
    granularity, and it is the cap the episode-length histogram spikes at when two policies
    settle into the turtle equilibrium.

    The reward is this package's potential composition rather than RoyaleGym's
    ``default_reward``, whose elixir-trade term pays a player who never commits a card and whose
    tower term is one discount short of a potential (``rewards.py`` says why each matters). The
    reward is part of the environment spec and so of the context digest: runs trained against
    the other one are a different objective and do not share a context with these.
    """
    return EnvFactorySpec(
        engine=ComponentSpec(engine),
        obs_builder=ComponentSpec("royalegym.obs.SpatialObsBuilder"),
        action_parser=ComponentSpec("royalegym.action.TileActionParser"),
        reward_fn=ComponentSpec("royalelearn.rewards.default_potential_reward"),
        state_mutator=ComponentSpec("royalegym.state_mutator.DefaultStateMutator"),
        termination=[ComponentSpec("royalegym.done_condition.GameOverCondition")],
        truncation=[
            ComponentSpec("royalegym.done_condition.StepLimitCondition", {"max_steps": max_steps})
        ],
    )


class RunConfig(msgspec.Struct, forbid_unknown_fields=True):
    """One run, completely: RoyaleLearn's own keys.

    A run that uses an extension (``royalelearn.extensions``) has one more top-level key per
    extension, its section. ``load_config`` returns such a config as a subclass of this one with
    the sections as fields after these, so a config without sections is exactly this type and
    encodes as it always did.
    """

    format_version: int = 1
    #: Cosmetic: recorded, printed, and not in the identity.
    run_name: str = "royalelearn"
    runs_dir: str = "runs"
    #: In the identity: it changes every byte that follows.
    master_seed: int = 20260921
    #: Selects the shipped defaults; all of them are still overridable.
    profile: str = "laptop"
    #: Learner transitions.
    timestep_limit: int = 100_000_000
    env: EnvFactorySpec = msgspec.field(default_factory=default_env_spec)
    #: None means ``env`` with the truncation removed.
    eval_env: EnvFactorySpec | None = None
    #: Modules a ComponentSpec may name beyond royalegym and royalelearn.
    extra_component_modules: list[str] = []
    obs: ObsConfig = ObsConfig()
    rollout: RolloutConfig = RolloutConfig()
    net: NetConfig = NetConfig()
    ppo: PPOConfig = PPOConfig()
    advantage: AdvantageConfig = AdvantageConfig()
    ladder: LadderConfig = LadderConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    metrics: MetricsConfig = MetricsConfig()
    alarms: AlarmConfig = AlarmConfig()
    determinism: DeterminismConfig = DeterminismConfig()
    doctor: DoctorConfig = DoctorConfig()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def role_counts(mix: Sequence[float], n_battles: int) -> tuple[int, int, int]:
    """How many battle slots play mirror, how many pool and how many scripted.

    The mixture is a count of slots, not a probability each slot is drawn from, so this is the
    one place it is turned into whole battles. ``MixMatchmaker`` decides *which* slots get each
    role and ``geometry`` sizes the iteration from the same counts: two callers, one function,
    because the whole point of the exercise is that the number the iteration is sized for is
    the number the matchmaker fills.

    It lives here rather than on the matchmaker only because of the import graph -- the
    matchmaker reaches ``config`` through ``api.ladder`` and ``identity``, so ``config``
    importing the matchmaker would close a cycle. The dependency runs the way it already did.

    Rounding is resolved in two stages, each to the nearest whole battle with a half rounding
    up, and the last share takes whatever is left:

    * ``mirror = round(mix[0] * n_battles)``,
    * the remaining battles split between pool and scripted in the ratio ``mix[1] : mix[2]``,
      ``pool`` rounded and ``scripted`` taking the remainder.

    So the three always sum to ``n_battles``, and each lands within one battle of its exact
    share (the proof is short: each stage rounds by at most a half, and the second stage's
    input is already off by at most a half). Halves round up rather than to even, because
    ``round(0.5)`` is 0 in Python and a single battle under the shipped mixture would then be a
    rectangle with no mirror in it at all.

    Two consequences are worth saying out loud. A share small enough that its quota is under a
    half rounds to zero: three battles cannot hold a tenth of one, so that role is simply not
    in the rectangle, and the mixture the run reports is the count, not the config. And an
    all-mirror mixture keeps every slot, which is what makes ``(1, 0, 0)`` collect twice the
    rows per cycle rather than one and a half.
    """
    if n_battles < 0:
        raise ValueError(f"a rectangle cannot have {n_battles} battles")
    mirror = min(n_battles, _round_half_up(mix[0] * n_battles))
    rest = n_battles - mirror
    weight = mix[1] + mix[2]
    pool = min(rest, _round_half_up(rest * mix[1] / weight)) if weight > 0.0 else 0
    return mirror, pool, rest - pool


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


class Geometry(msgspec.Struct, frozen=True):
    """The iteration's rectangle, derived from the config once and passed around whole.

    ``cycles`` by ``n_slots``, with slot -> (worker, shard, battle, seat) fixed for the run.
    Everything downstream -- the buffer's shape, the slot maps, the matchmaker's plan -- reads
    these numbers rather than recomputing them, so the arithmetic of section 2.2 lives in one
    place (``config.geometry``).
    """

    workers: int
    games_per_worker: int
    shards_per_worker: int
    games_per_shard: int
    n_battles: int
    n_slots: int
    mirror_battles: int
    learner_row_fraction: float
    learner_rows: int
    cycles: int
    timesteps_per_iteration: int

    @property
    def slots_per_shard(self) -> int:
        return 2 * self.games_per_shard

    @property
    def slots_per_worker(self) -> int:
        return 2 * self.games_per_worker


def geometry(config: RunConfig) -> Geometry:
    """The iteration's rectangle, derived from the config.

    Learner rows are the slots whose seat the learner plays: both seats of a mirror battle and
    one of every other. Because ``role_counts`` makes the mirror battles a count rather than a
    share, that is ``n_battles + mirror_battles`` exactly -- the same number on every cycle of
    every iteration of every run, whatever the master seed.

    It used to be ``int(n_slots * (mirror + (1 - mirror) / 2))``, the count's expectation. The
    two agree at the shipped laptop profile (96 + 48 = int(192 * 0.75) = 144), and what the
    expectation could not do was promise it: the realised count was Binomial(96, 0.5) + 96,
    with a standard deviation of 4.9 rows per cycle against an iteration whose slack over the
    trainable-row floor was 0.2%, so about one master seed in four halted the run.

    The cycle count is then what it takes to reach ``timesteps_per_iteration`` learner
    transitions, rounded up -- an iteration collects at least what was asked for, never less,
    and now the "at least" is a fact about the rectangle rather than about its average.
    """
    rollout = config.rollout
    if rollout.shards_per_worker < 1 or rollout.games_per_worker % rollout.shards_per_worker:
        raise ValueError(
            f"games_per_worker {rollout.games_per_worker} is not divisible by "
            f"shards_per_worker {rollout.shards_per_worker}"
        )
    n_battles = rollout.workers * rollout.games_per_worker
    n_slots = 2 * n_battles
    mirror_battles = role_counts(config.ladder.mix, n_battles)[0]
    learner_rows = n_battles + mirror_battles
    if learner_rows < 1:
        raise ValueError(
            f"the mixture {config.ladder.mix} leaves no learner rows in {n_slots} slots"
        )
    cycles = math.ceil(config.ppo.timesteps_per_iteration / learner_rows)
    return Geometry(
        workers=rollout.workers,
        games_per_worker=rollout.games_per_worker,
        shards_per_worker=rollout.shards_per_worker,
        games_per_shard=rollout.games_per_worker // rollout.shards_per_worker,
        n_battles=n_battles,
        n_slots=n_slots,
        mirror_battles=mirror_battles,
        learner_row_fraction=learner_rows / n_slots,
        learner_rows=learner_rows,
        cycles=cycles,
        timesteps_per_iteration=config.ppo.timesteps_per_iteration,
    )


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def laptop() -> RunConfig:
    """8 GB RAM, 4 GB VRAM, 8 threads. The shipped default.

    The learner is the bottleneck on this machine by about 2.5 times, which is why there are
    three workers and not thirty-two, and why the network is four hundred thousand parameters
    and not four million.
    """
    return RunConfig(profile="laptop")


def workstation() -> RunConfig:
    """32 GB RAM, 12 GB VRAM, 16 threads.

    It asked for ``rollout.overlap`` until 2026-09-22, which bought a second rectangle and no
    overlap, because nothing in the package honours the flag. It is refused now.
    """
    return RunConfig(
        profile="workstation",
        rollout=RolloutConfig(workers=12, games_per_worker=64),
        net=NetConfig(channels=96, blocks=8, card_embed=96),
        ppo=PPOConfig(timesteps_per_iteration=131_072, batch_size=16_384, minibatch_size=2_048),
        ladder=LadderConfig(candidate_every_env_steps=2_000_000),
    )


def many_core() -> RunConfig:
    """64 GB RAM, 24 GB VRAM, 64 threads."""
    return RunConfig(
        profile="many_core",
        rollout=RolloutConfig(workers=24, games_per_worker=64),
        net=NetConfig(channels=128, blocks=12, card_embed=128),
        ppo=PPOConfig(timesteps_per_iteration=262_144, batch_size=32_768, minibatch_size=8_192),
        ladder=LadderConfig(candidate_every_env_steps=2_000_000),
    )


PROFILES: dict[str, Any] = {"laptop": laptop, "workstation": workstation, "many_core": many_core}


def profile(name: str) -> RunConfig:
    """The shipped config for one machine class."""
    try:
        build = PROFILES[name]
    except KeyError:
        raise PreflightError(
            f"profile {name!r} is not one of {', '.join(sorted(PROFILES))}"
        ) from None
    return build()


# --------------------------------------------------------------------------
# Consistency
# --------------------------------------------------------------------------


def _probe_problems(ladder: LadderConfig) -> list[str]:
    """Everything wrong with the live-policy probe's settings.

    The opponent names are checked whether or not the probe is on: a name is either one of the
    scripted opponents or a typo, and a typo caught at the first probe is caught an hour into a
    run rather than before it starts. The counts are checked only when the probe is on. They
    describe battles that will be played, and holding a run that never probes to them would
    make these new fields change what an existing config is allowed to say.
    """
    from .rollout.scripted import SCRIPTED_NAMES, scripted_id

    known = tuple(scripted_id(name) for name in SCRIPTED_NAMES)
    problems = [
        f"ladder.probe_opponents names {opponent!r}, which is not one of {', '.join(known)}"
        for opponent in ladder.probe_opponents
        if opponent not in known
    ]
    repeated = sorted(
        {name for name in ladder.probe_opponents if ladder.probe_opponents.count(name) > 1}
    )
    if repeated:
        # The rungs are keyed by name in the row, so a repeat plays every battle twice and
        # publishes one key: the run pays double and the reader cannot see that it did.
        problems.append(
            f"ladder.probe_opponents names {', '.join(repeated)} more than once; each rung is "
            "one key in the row, so a repeat costs its battles twice and publishes once"
        )
    if ladder.probe_every_iterations < 0:
        problems.append(
            f"ladder.probe_every_iterations {ladder.probe_every_iterations} is negative; 0 is "
            "how a run says it does not probe"
        )
    if ladder.probe_every_iterations <= 0:
        return problems
    if not ladder.probe_opponents:
        problems.append(
            "ladder.probe_every_iterations asks for a probe and ladder.probe_opponents is "
            "empty, so every probe would measure nothing"
        )
    if ladder.probe_games < 2:
        problems.append(
            f"ladder.probe_games {ladder.probe_games} is below the two battles a paired "
            "comparison is: one seed, played from both sides"
        )
    elif ladder.probe_games % 2:
        # A comparison takes games // 2 seeds and plays each twice, so an odd count silently
        # plays one fewer battle than it says. A number in a config that is not the number the
        # run uses is the kind of thing somebody later measures against.
        problems.append(
            f"ladder.probe_games {ladder.probe_games} is odd, and a probe plays each seed from "
            f"both sides: it would play {ladder.probe_games - 1}. Ask for an even number"
        )
    elif ladder.probe_games // 2 > ladder.eval_seed_count:
        problems.append(
            f"ladder.probe_games {ladder.probe_games} needs {ladder.probe_games // 2} of the "
            f"{ladder.eval_seed_count} frozen evaluation seeds; raise ladder.eval_seed_count, "
            "or probe fewer battles"
        )
    return problems


def check_consistency(config: RunConfig) -> list[str]:
    """Everything wrong with a config, in one list.

    All of it at once rather than one exception at a time: on a machine where building an
    environment costs a minute, finding four mistakes takes four minutes otherwise.
    """
    problems: list[str] = []
    ppo, rollout, net = config.ppo, config.rollout, config.net

    # The environment's components, checked against their own signatures, so a misspelt kwarg
    # is named here rather than raised from inside a worker when the environment is built. The
    # shaping weights get the reward's own rule on top: a config passing this says something
    # about them now, where it used to inspect none of them.
    from .rollout.envspec import component_kwarg_problems

    modules = tuple(config.extra_component_modules)
    env = config.env
    for spec in (
        env.engine,
        env.obs_builder,
        env.action_parser,
        env.reward_fn,
        env.state_mutator,
        *env.termination,
        *env.truncation,
    ):
        problems.extend(f"env: {problem}" for problem in component_kwarg_problems(spec, modules))
    if env.reward_fn.cls == "royalelearn.rewards.default_potential_reward":
        from .rewards import shaping_weight_problems

        problems.extend(
            f"env.reward_fn: {problem}" for problem in shaping_weight_problems(env.reward_fn.kwargs)
        )

    if rollout.overlap:
        # Refused rather than accepted and ignored. Spec 14.1 says what it would do: collect
        # iteration i on a second thread and a second buffer while the update of i-1 runs,
        # against a BehaviourSnapshot taken at the boundary. HALF of that exists --
        # BatchedInference.begin_iteration takes a snapshot and samples from it -- and the
        # driver does not: nothing passes one, there is no second thread, there is no second
        # buffer, and ppo/behaviour_lag_iterations is in the spec and in no schema.
        #
        # Meanwhile the flag was not free. preflight sizes TWO rectangles when it is set, so the
        # two profiles that shipped with it on reserved twice the buffer memory for a feature
        # that never ran, and that reservation is what the memory gate checks against.
        problems.append(
            "rollout.overlap is not implemented: it is accepted, recorded in the run identity "
            "and charged for (the preflight reserves a second rectangle) while changing "
            "nothing. Spec 14.1 says what it would do. Leave it false until the driver exists"
        )

    if rollout.shards_per_worker < 1:
        problems.append("rollout.shards_per_worker must be at least 1")
    elif rollout.games_per_worker % rollout.shards_per_worker:
        problems.append(
            f"rollout.games_per_worker {rollout.games_per_worker} is not divisible by "
            f"rollout.shards_per_worker {rollout.shards_per_worker}"
        )
    else:
        geo = geometry(config)
        if geo.cycles * geo.learner_rows < ppo.timesteps_per_iteration:
            problems.append(
                f"the rectangle collects {geo.cycles * geo.learner_rows} learner transitions, "
                f"below ppo.timesteps_per_iteration {ppo.timesteps_per_iteration}"
            )
    if rollout.workers < 1:
        problems.append("rollout.workers must be at least 1")
    if rollout.source not in ("process", "inline"):
        problems.append(f"rollout.source {rollout.source!r} is not 'process' or 'inline'")

    if ppo.minibatch_size < 1 or ppo.batch_size % ppo.minibatch_size:
        problems.append(
            f"ppo.batch_size {ppo.batch_size} is not a multiple of ppo.minibatch_size "
            f"{ppo.minibatch_size}, so a minibatch would not be a pure memory knob"
        )
    if ppo.n_epochs < 1:
        problems.append("ppo.n_epochs must be at least 1")
    if ppo.forced_rows not in FORCED_ROW_ARMS:
        problems.append(
            f"ppo.forced_rows {ppo.forced_rows!r} is not one of {', '.join(FORCED_ROW_ARMS)}"
        )
    if ppo.dual_clip_c < 1.0:
        # The skipping arms add the skipped rows' surrogate back analytically, and that add-back
        # is only correct because a forced row's dual clip cannot bind: max(A, cA) is A for a
        # negative advantage when c is at least one. Below one the clip takes the other branch,
        # the constant the arms add back is the wrong constant, and ppo/policy_loss stops being
        # comparable between them. Nothing checked this until the verifier asked what happens at
        # 0.5. The value is a lower bound on a negative surrogate, so under one is not a
        # configuration anybody wants either.
        problems.append(
            f"ppo.dual_clip_c {ppo.dual_clip_c} is below 1.0, which is not a lower bound on a "
            "negative surrogate and breaks the add-back ppo.forced_rows relies on"
        )
    elif ppo.forced_rows != "all" and not net.separate_trunks:
        problems.append(
            f"ppo.forced_rows {ppo.forced_rows!r} needs net.separate_trunks: with one trunk "
            "under both heads the critic's forward IS the actor's, so a row the actor skips is "
            "a row nothing is saved on, and that row's value loss still reaches the parameters "
            "the policy gradient uses. Only the policy head could be left out, and a choice-row "
            "mean would then weigh the value term against the policy term inside the trunk by "
            "a fraction the elixir economy moves every iteration"
        )
    if net.autocast_dtype not in ppo.ratio_atol:
        problems.append(
            f"ppo.ratio_atol has no entry for net.autocast_dtype {net.autocast_dtype!r}; "
            f"it has {sorted(ppo.ratio_atol)}"
        )
    if net.card_embed != net.channels:
        problems.append(
            f"net.card_embed {net.card_embed} must equal net.channels {net.channels}: they are "
            "the two sides of the pointer head's inner product"
        )
    if net.channels % net.norm_groups:
        problems.append(
            f"net.channels {net.channels} is not divisible by net.norm_groups {net.norm_groups}"
        )

    if config.obs.frame_stack < 1:
        problems.append("obs.frame_stack must be at least 1")
    if abs(sum(config.ladder.mix) - 1.0) > 1e-9:
        problems.append(f"ladder.mix {config.ladder.mix} does not sum to 1")
    if any(share < 0 for share in config.ladder.mix):
        problems.append(f"ladder.mix {config.ladder.mix} has a negative share")
    if config.ladder.max_resident_opponents < 1:
        problems.append("ladder.max_resident_opponents must be at least 1")
    if config.ladder.rater.draws not in ("davidson", "half_win"):
        problems.append(f"ladder.rater.draws {config.ladder.rater.draws!r} is not a draw model")
    problems.extend(_probe_problems(config.ladder))
    problems.extend(_extension_problems(config))
    problems.extend(_alarm_override_problems(config))
    if config.checkpoint.keep < 1:
        problems.append("checkpoint.keep must be at least 1")
    if config.determinism.tier not in TIERS:
        problems.append(
            f"determinism.tier {config.determinism.tier!r} is not one of {', '.join(TIERS)}"
        )
    if config.profile not in PROFILES:
        problems.append(f"profile {config.profile!r} is not one of {', '.join(sorted(PROFILES))}")
    if not any(sink.kind == "jsonl" and sink.enabled for sink in config.metrics.sinks):
        problems.append(
            "metrics.sinks must include an enabled 'jsonl' sink: it is the file a resume is "
            "checked against"
        )
    return problems


def _extension_problems(config: RunConfig) -> list[str]:
    from .extensions import extension_problems

    return extension_problems(config)


def _alarm_override_problems(config: RunConfig) -> list[str]:
    """Overrides naming an alarm this run's table does not have (``metrics.alarms``)."""
    from .extensions import extension_alarms
    from .metrics.alarms import default_alarms, names, override_problems

    table = default_alarms(config.alarms, extra=extension_alarms(config))
    return override_problems(config.alarms, names(table))


def validate(config: RunConfig) -> RunConfig:
    """Return the config, or raise ``PreflightError`` naming everything wrong with it."""
    problems = check_consistency(config)
    if problems:
        raise PreflightError(
            "the configuration is inconsistent:\n" + "\n".join(f"  {p}" for p in problems)
        )
    return config


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def _merge(base: Any, overlay: Any) -> Any:
    """Overlay a decoded JSON document onto another, mapping by mapping.

    Lists replace rather than merge: a partially overridden list of sinks or of termination
    conditions would be a third thing that neither document says. A mapping that names a
    different variant of a tagged union replaces for the same reason -- a constant schedule
    carrying the leftover endpoints of the linear one it overrode is not a schedule.
    """
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        tag = overlay.get(TAG_FIELD)
        if tag is not None and tag != base.get(TAG_FIELD):
            return overlay
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = _merge(merged[key], value) if key in merged else value
        return merged
    return overlay


#: Keys that builds between 0ab7a29 and the extension API wrote into EVERY config.json, with
#: the values they had there, and where each lives now. Dropped with a notice when a file still
#: carries them at those values, for one release; a changed value is refused with its new home,
#: because dropping it would silently move a threshold somebody set.
RETIRED_ALARM_KEYS: dict[str, tuple[float, str]] = {
    "imitation_ref_kl_warn": (1.0, "imitation.alarms.ref_kl_warn"),
    "imitation_lambda_saturated_patience": (10, "imitation.alarms.lambda_saturated_patience"),
    "imitation_handoff_window": (20, "warm_start.alarms.handoff_window"),
    "imitation_handoff_kl": (0.05, "warm_start.alarms.handoff_kl"),
    "imitation_handoff_clip": (0.3, "warm_start.alarms.handoff_clip"),
    "imitation_ev_at_unfreeze": (0.3, "warm_start.alarms.ev_at_unfreeze"),
}


def _retire(raw: dict[str, Any], say: Callable[[str], None] | None) -> dict[str, Any]:
    """The one-release shim for configs written between 0ab7a29 and the extension API.

    Every such file carries ``"imitation": null`` -- which any null section now loads as -- and
    the six ``alarms.imitation_*`` thresholds. And an IL config of that time kept the actor's
    starting weights and learning-rate schedule inside ``imitation``, where they are now refused
    as unknown fields: this names the section they moved to instead.
    """
    raw = dict(raw)
    problems: list[str] = []
    dropped: list[str] = []
    alarms = raw.get("alarms")
    if isinstance(alarms, dict) and any(key in alarms for key in RETIRED_ALARM_KEYS):
        alarms = dict(alarms)
        for key, (old_default, moved_to) in RETIRED_ALARM_KEYS.items():
            if key not in alarms:
                continue
            value = alarms.pop(key)
            if value == old_default:
                dropped.append(f"alarms.{key}")
            else:
                problems.append(
                    f"alarms.{key} is {value!r}: that threshold is now {moved_to}, and it applies "
                    "only to a run with that section"
                )
        raw["alarms"] = alarms
    imitation = raw.get("imitation", {})
    if imitation is None:
        dropped.append('"imitation": null')
    if isinstance(imitation, dict):
        for key in ("init", "actor_lr_scale"):
            if key in imitation:
                problems.append(f"imitation.{key} is now warm_start.{key}")
    if problems:
        raise PreflightError("the configuration uses keys that have moved:\n" + "\n".join(
            f"  {p}" for p in problems
        ))
    if dropped and say is not None:
        say(
            "config        dropped keys an older build wrote at their default values: "
            + ", ".join(dropped)
        )
    return raw


def load_config(
    source: str | bytes | Path | Mapping[str, Any],
    *,
    say: Callable[[str], None] | None = print,
) -> RunConfig:
    """Read a config: a JSON string, a path to one, or an already-decoded mapping.

    The document is applied ON TOP of the profile it names, so a file that sets three numbers
    gets the rest of that machine class rather than the laptop's, and a file that sets none is
    exactly the profile. Unknown fields are refused at every level of the tree: a top-level key
    RoyaleLearn does not own is a section, and one that no installed extension provides is
    refused by name (``extensions.providers_for``). A section that is null counts as absent.
    """
    from .extensions import build_config

    if isinstance(source, Path):
        raw: Any = msgspec.json.decode(source.read_bytes())
    elif isinstance(source, (str, bytes)):
        raw = msgspec.json.decode(source if isinstance(source, bytes) else source.encode("utf-8"))
    else:
        raw = dict(source)
    if not isinstance(raw, dict):
        raise PreflightError(f"a config must be a JSON object, not {type(raw).__name__}")
    raw = _retire(raw, say)
    core_keys = set(RunConfig.__struct_fields__)
    sections = {key: raw.pop(key) for key in [k for k in raw if k not in core_keys]}
    base = msgspec.to_builtins(profile(str(raw.get("profile", "laptop"))))
    return build_config(_merge(base, raw), sections)


def dump_config(config: RunConfig, *, indent: int | None = None) -> str:
    """The canonical JSON of a config: keys in a fixed order, so a reformat is not a new run."""
    blob = canonical_json(config)
    if indent is not None:
        blob = msgspec.json.format(blob, indent=indent)
    return blob.decode("utf-8")


def config_hash(config: RunConfig) -> str:
    """sha256 of the canonical JSON. Recorded, printed and diffed on resume."""
    return digest_of(config)
