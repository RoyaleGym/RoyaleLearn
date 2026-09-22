"""Every metric the harness emits: its unit, what it means, and what healthy looks like.

This is the single source of truth. A sink never indexes a fixed key list, so a new metric
cannot crash the console; instead the suite asserts both directions -- every key a run emits is
here, and every key here is emitted -- so the documentation and the code cannot drift apart.

Keys are ``group/name``. A few families are per member, per card or per epoch, and those are
patterns rather than keys: the number of epochs is a config value and the members of the pool
are discovered as a run goes on, so writing them out would be writing down something the run
decides.

``ALARM_METRICS`` names, for each alarm of section 13.3, the keys its predicate reads. The
alarms themselves live in ``metrics/alarms.py``; this table is what keeps one of them from
watching a key nobody emits any more.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import msgspec

__all__ = [
    "ALARM_METRICS",
    "METRICS",
    "PATTERNS",
    "MetricPattern",
    "MetricSpec",
    "groups",
    "is_known",
    "lookup",
]


class MetricSpec(msgspec.Struct, frozen=True):
    """One metric. ``low`` and ``high`` bound the healthy range; None is unbounded, and both
    None means there is no range to give -- a count, a digest, a label."""

    unit: str
    description: str
    dtype: str = "float"
    low: float | None = None
    high: float | None = None


class MetricPattern(NamedTuple):
    """A family of keys, such as one per pool member. ``template`` carries ``{placeholders}``."""

    template: str
    spec: MetricSpec
    regex: re.Pattern[str]


def _pattern(template: str, spec: MetricSpec) -> MetricPattern:
    """A template with ``{placeholders}`` becomes a regex whose holes match one key segment."""
    escaped = re.escape(template).replace(r"\{", "{").replace(r"\}", "}")
    regex = re.compile("^" + re.sub(r"\{[a-z_]+\}", "[^/]+", escaped) + "$")
    return MetricPattern(template, spec, regex)


def _m(unit: str, description: str, **kwargs: object) -> MetricSpec:
    return MetricSpec(unit=unit, description=description, **kwargs)  # type: ignore[arg-type]


METRICS: dict[str, MetricSpec] = {
    # -- run ---------------------------------------------------------------
    "run/iteration": _m(
        "count", "Iterations completed in this run, resumes included.", dtype="int"
    ),
    "run/cumulative_timesteps": _m("count", "Learner transitions collected so far.", dtype="int"),
    "run/cumulative_env_steps": _m(
        "count", "Game-steps stepped so far, both seats of a battle counted once.", dtype="int"
    ),
    "run/cumulative_updates": _m("count", "Optimizer steps taken so far.", dtype="int"),
    "run/wall_seconds": _m("s", "Wall clock since the run started, resumes included."),
    "run/gamma": _m("ratio", "The discount this iteration ran at.", low=0.99, high=1.0),
    "run/gae_lambda": _m("ratio", "The GAE lambda this iteration ran at.", low=0.9, high=1.0),
    "run/credit_horizon_seconds": _m(
        "s",
        "1/(1 - gamma*lambda) decisions in seconds: how far back a reward is still felt. Logged "
        "every run so that it can never be ten seconds by accident.",
        low=40.0,
        high=50.0,
    ),
    "run/ent_coef": _m("coefficient", "The joint entropy bonus this iteration ran at."),
    "run/ent_coef_noop": _m("coefficient", "The no-op binary entropy bonus this iteration ran at."),
    "run/lr_actor": _m("rate", "Actor learning rate after any backoff."),
    "run/lr_critic": _m("rate", "Critic learning rate after any backoff."),
    "run/determinism_tier": _m(
        "label", "run_exact or throughput, stamped on every row.", dtype="str"
    ),
    "run/resumed_with_drift": _m(
        "flag",
        "True on every row after a resume that was allowed to differ in an identity field.",
        dtype="bool",
    ),
    "run/state_digest": _m(
        "digest",
        "sha256 over the weights, the optimizer state, the return scaler and the schedule "
        "positions, so that the iteration at which two runs first differ is a lookup.",
        dtype="str",
    ),
    # -- throughput --------------------------------------------------------
    "throughput/overall_steps_per_second": _m(
        "timesteps/s", "Learner transitions per second of wall clock, everything included."
    ),
    "throughput/collected_steps_per_second": _m(
        "timesteps/s", "Learner transitions per second of collection time alone."
    ),
    "throughput/engine_ticks_per_second": _m(
        "ticks/s", "Engine ticks per second across all workers."
    ),
    "throughput/rollout_capacity_ratio": _m(
        "ratio",
        "Rollout capacity over update capacity. The invariant is that the harness is never the "
        "bottleneck: at least 2 on every shipped profile, a warning below 1.5.",
        low=2.0,
    ),
    "throughput/boundary_mb_per_second": _m(
        "MB/s", "Bytes crossing the worker boundary per second."
    ),
    "throughput/worker_idle_frac": _m(
        "fraction", "Share of a round a worker spends waiting for the parent.", low=0.0, high=0.3
    ),
    "throughput/inference_ms_per_round": _m("ms", "Parent time per shard-round, all policies."),
    "throughput/discarded_rows_frac": _m(
        "fraction", "Collected rows not trained on: the frozen seats, by design."
    ),
    "throughput/gpu_util_frac": _m("fraction", "Device utilisation over the iteration."),
    # -- time --------------------------------------------------------------
    "time/iteration": _m("s", "Seconds for the whole iteration."),
    "time/collection": _m("s", "Seconds collecting, inference included."),
    "time/inference": _m("s", "Seconds in the rollout forwards."),
    "time/env": _m("s", "Seconds inside the environments, as the workers report them."),
    "time/codec": _m("s", "Seconds packing observations in the workers."),
    "time/ipc": _m("s", "Seconds waiting on round events and reading scalars."),
    "time/critic_pass": _m("s", "Seconds in the whole-iteration critic pass."),
    "time/gae": _m("s", "Seconds computing advantages and returns."),
    "time/update": _m("s", "Seconds in the PPO update."),
    "time/checkpoint": _m("s", "Seconds writing checkpoints this iteration."),
    "time/gate": _m("s", "Seconds in the promotion gate this iteration."),
    "time/overlap_saved": _m(
        "s", "Seconds of collection hidden under the update by rollout.overlap."
    ),
    "time/residual": _m(
        "s",
        "Iteration seconds not attributed to any phase above, broken out rather than absorbed "
        "so that an unmeasured cost shows up as itself.",
    ),
    # -- ppo ---------------------------------------------------------------
    "ppo/policy_loss": _m("loss", "The clipped surrogate, averaged over samples."),
    "ppo/value_loss": _m("loss", "The critic's loss on standardised returns."),
    "ppo/entropy": _m("nats", "Mean entropy of the masked action distribution."),
    "ppo/entropy_normalised": _m(
        "fraction",
        "Entropy over log(legal actions). Raw entropy falling is ambiguous -- a confident "
        "policy or a tighter mask -- and only the normalised form separates them.",
        low=0.3,
        high=0.8,
    ),
    "ppo/noop_entropy": _m(
        "nats",
        "Binary entropy of p(no-op) against p(play): the leading indicator of no-op collapse, "
        "before cards_per_match bottoms out.",
        low=0.05,
        high=0.693,
    ),
    "ppo/kl": _m(
        "nats",
        "Mean KL between the behaviour policy and the updated one. Below the band, lower "
        "batch_size; above it, raise batch_size or let the backoff act.",
        low=0.003,
        high=0.02,
    ),
    "ppo/clip_fraction": _m(
        "fraction",
        "Share of samples whose ratio hit the clip. Pinned near 1.0 is the signature of a "
        "rollout/update mask disagreement, or a learning rate far too high.",
        low=0.05,
        high=0.20,
    ),
    "ppo/dual_clip_fraction": _m(
        "fraction", "Share of negative-advantage samples the dual clip bound."
    ),
    "ppo/explained_variance": _m(
        "fraction",
        "1 - Var(returns - values)/Var(returns). The critic's health, and still negative after "
        "fifty iterations is the most likely cause of a plateau.",
        low=0.5,
        high=0.9,
    ),
    "ppo/grad_norm_actor": _m(
        "norm",
        "Actor gradient norm before clipping. Pinned at max_grad_norm every step means the clip "
        "is the binding constraint and the effective learning rate is unknown.",
        high=0.5,
    ),
    "ppo/grad_norm_critic": _m("norm", "Critic gradient norm before clipping.", high=0.5),
    "ppo/update_magnitude_actor": _m(
        "norm", "L2 distance the actor's parameters moved this iteration."
    ),
    "ppo/update_magnitude_critic": _m(
        "norm", "L2 distance the critic's parameters moved this iteration."
    ),
    "ppo/ratio_max_abs_dev": _m(
        "ratio",
        "Largest |ratio - 1| at the first minibatch of the first epoch, where it must be zero "
        "to tolerance: the cheapest detector there is for a mask, codec or weight-version drift.",
    ),
    "ppo/advantage_std_pre_norm": _m(
        "units", "Advantage standard deviation before standardisation."
    ),
    "ppo/return_running_mean": _m(
        "units", "The return scaler's running mean, recorded and never subtracted."
    ),
    "ppo/return_running_std": _m(
        "units", "The return scaler's running standard deviation: the divisor."
    ),
    "ppo/reward_clip_frac": _m(
        "fraction", "Share of rewards the clip bound touched.", low=0.0, high=0.01
    ),
    "ppo/n_minibatches": _m("count", "Minibatch forwards this iteration.", dtype="int"),
    "ppo/n_optimizer_steps": _m("count", "Optimizer steps this iteration.", dtype="int"),
    "ppo/samples_unused_frac": _m(
        "fraction",
        "Trainable samples not trained on. Zero by construction: the remainder of an epoch is a "
        "smaller final batch, weighted by its true sample count.",
        low=0.0,
        high=0.0,
    ),
    "ppo/lr_backoff_events": _m("count", "Learning-rate backoffs so far in this run.", dtype="int"),
    # -- policy ------------------------------------------------------------
    "policy/cards_per_match": _m(
        "cards",
        "Cards played per finished episode. THIS and not the no-op rate is the no-op-collapse "
        "metric: a healthy policy is about 94% no-op, and 99.5% no-op is under two cards a "
        "match and dead. About 22 is healthy, which is elixir arithmetic rather than a guess.",
        low=10.0,
        high=30.0,
    ),
    "policy/noop_rate": _m(
        "fraction", "Share of decisions that were the no-op.", low=0.85, high=0.97
    ),
    "policy/legal_actions_mean": _m("count", "Legal actions per decision, mean."),
    "policy/legal_actions_p05": _m("count", "Legal actions per decision, 5th percentile."),
    "policy/legal_actions_p50": _m("count", "Legal actions per decision, median."),
    "policy/legal_actions_p95": _m("count", "Legal actions per decision, 95th percentile."),
    "policy/forced_noop_frac": _m(
        "fraction",
        "Share of decisions with exactly one legal action, which carry no policy gradient.",
    ),
    "policy/tile_entropy": _m("nats", "Entropy of the play distribution over tiles."),
    "policy/tile_top1_share": _m(
        "fraction", "Share of plays on the single most-played tile.", high=0.25
    ),
    "policy/card_tile_top10_share": _m(
        "fraction",
        "Share of plays in the ten most-played (card, tile) pairs. A real meta is not that "
        "concentrated.",
        high=0.5,
    ),
    # -- env ---------------------------------------------------------------
    "env/episode_steps_mean": _m(
        "decisions", "Mean length of the episodes that finished this iteration."
    ),
    "env/episode_steps_p05": _m("decisions", "Episode length, 5th percentile."),
    "env/episode_steps_p50": _m("decisions", "Episode length, median."),
    "env/episode_steps_p95": _m("decisions", "Episode length, 95th percentile."),
    "env/episode_steps_hist": _m(
        "counts",
        "The episode-length histogram, as JSON: a spike at the truncation cap is the draw and "
        "turtle equilibrium.",
        dtype="str",
    ),
    "env/episode_steps_at_cap_frac": _m(
        "fraction", "Share of episodes that ended at the truncation cap rather than in a result."
    ),
    "env/episodes_completed": _m("count", "Episodes that finished this iteration.", dtype="int"),
    "env/ticks_mean": _m("ticks", "Mean engine ticks per finished episode."),
    "env/crowns_for": _m("crowns", "Mean crowns taken per episode, learner seats."),
    "env/crowns_against": _m("crowns", "Mean crowns conceded per episode, learner seats."),
    "env/crown_diff": _m("crowns", "Crowns taken minus crowns conceded."),
    "env/tower_hp_frac_end_own": _m(
        "fraction", "Mean of the learner's three towers' hp fraction at the end.", low=0.0, high=1.0
    ),
    "env/tower_hp_frac_end_enemy": _m(
        "fraction", "Mean of the opponent's three towers' hp fraction at the end.", low=0.0,
        high=1.0
    ),
    "env/draw_rate": _m("fraction", "Share of episodes that ended in a draw.", high=0.5),
    "env/win_rate_by_seat": _m(
        "fraction",
        "Blue's share of decided episodes. A deviation from a half is a learner-side leak, "
        "because the observation layer guarantees bit-identical mirrored observations.",
        low=0.45,
        high=0.55,
    ),
    "env/win_rate_by_seat_ci95_lo": _m(
        "fraction", "Lower end of the 95% interval around win_rate_by_seat."
    ),
    "env/win_rate_by_seat_ci95_hi": _m(
        "fraction", "Upper end of the 95% interval around win_rate_by_seat."
    ),
    "env/elixir_leak_frac": _m(
        "fraction", "Share of decisions spent at a full elixir bar.", high=0.1
    ),
    "env/elixir_count_exact_frac": _m(
        "fraction",
        "Share of episode-seats whose count of the opponent's elixir stayed exact. Below one, "
        "the observation's enemy-elixir field was an estimate on some episodes.",
        low=0.99,
        high=1.0,
    ),
    "env/mean_elixir_at_decision": _m(
        "elixir", "Mean elixir in the bar when a decision was taken."
    ),
    "env/frac_elixir_above_99": _m(
        "fraction", "Share of decisions taken with a bar above 99% full."
    ),
    "env/illegal_action_rate": _m(
        "fraction",
        "Share of commands the engine refused. Exactly zero under a correct mask.",
        low=0.0,
        high=0.0,
    ),
    "env/reward_shaping_abs": _m(
        "units", "Sum of the absolute shaping terms per episode, for the shaping_dominates alarm."
    ),
    "env/reward_terminal_abs": _m("units", "Absolute terminal reward term per episode."),
    # -- ladder ------------------------------------------------------------
    "ladder/rating_above_v0": _m(
        "Elo", "The learner's fitted rating above the run's first snapshot."
    ),
    "ladder/elo_readout": _m("Elo", "The online Elo dashboard figure, which is never the gate."),
    "ladder/champion_id": _m(
        "label", "The snapshot the candidate is measured against.", dtype="str"
    ),
    "ladder/champion_step": _m(
        "count", "The env step the champion was snapshotted at.", dtype="int"
    ),
    "ladder/pool_size": _m("count", "Snapshots in the archive.", dtype="int"),
    "ladder/sampler_size": _m("count", "Snapshots the matchmaker draws from.", dtype="int"),
    "ladder/gate_attempts": _m("count", "Gates run so far in this run.", dtype="int"),
    "ladder/gate_passes": _m("count", "Gates passed so far in this run.", dtype="int"),
    "ladder/gate_observed_rate": _m(
        "fraction", "The candidate's observed score rate against the champion."
    ),
    "ladder/gate_lower_bound": _m(
        "fraction", "The lower end of the interval that decides the gate."
    ),
    "ladder/gate_failed_condition": _m(
        "label", "Which of the three conditions failed, or none.", dtype="str"
    ),
    "ladder/gate_seconds_frac": _m("fraction", "Share of wall clock spent gating.", high=0.1),
    "ladder/consecutive_gate_failures": _m(
        "count", "Gates failed in a row: the plateau signal, stated as an event.", dtype="int"
    ),
    "ladder/score_vs_noop": _m(
        "fraction", "Score rate against the scripted no-op anchor.", low=0.9, high=1.0
    ),
    "ladder/score_vs_random_legal": _m(
        "fraction", "Score rate against the random-legal anchor.", low=0.7, high=1.0
    ),
    "ladder/transitivity_residual": _m(
        "fraction",
        "How much of the result log one number per player fails to explain. Large means the "
        "scalar rating is lying.",
        high=0.10,
    ),
    "ladder/paired_rho": _m(
        "correlation",
        "Correlation between the two sides of a paired seed, which sets the gate's real "
        "effective sample size.",
    ),
    "ladder/draw_rate_eval": _m(
        "fraction", "Draw rate in evaluation battles; it decides the rater's draw model."
    ),
    "ladder/eval_games_total": _m(
        "count", "Evaluation battles played so far in this run.", dtype="int"
    ),
    "ladder/evictions": _m("count", "Snapshots removed from the sampler so far.", dtype="int"),
    # -- health ------------------------------------------------------------
    "health/illegal_action_rate": _m(
        "fraction",
        "Commands the engine refused, over commands sent. Exactly zero by construction: this is "
        "an alert, not a plot.",
        low=0.0,
        high=0.0,
    ),
    "health/mask_disagreements": _m(
        "count",
        "Actions where the mask and the engine disagreed at the start-up gate; a run does not "
        "start above zero.",
        dtype="int",
        low=0.0,
        high=0.0,
    ),
    "health/worker_restarts": _m("count", "Workers restarted so far in this run.", dtype="int"),
    "health/rows_dropped_dead_worker": _m(
        "count", "Cells marked invalid because their worker was dead.", dtype="int"
    ),
    "health/obs_codec_clipped": _m(
        "count",
        "Values clipped on the way into a uint8 plane. Non-zero is a bug report, not a "
        "tolerance: the plane was admitted on a declared bound, so a clip means the bound was "
        "wrong.",
        dtype="int",
        low=0.0,
        high=0.0,
    ),
    "health/samples_unused_frac": _m(
        "fraction", "Trainable samples not trained on.", low=0.0, high=0.0
    ),
    "health/nan_guard_trips": _m(
        "count",
        "Non-finite losses, gradients or logits caught this iteration.",
        dtype="int",
        low=0.0,
        high=0.0,
    ),
    "health/vram_peak_mb": _m("MB", "Peak device memory this iteration."),
    "health/rss_peak_mb": _m("MB", "Peak resident memory of the parent this iteration."),
    "health/buffer_fill_frac": _m(
        "fraction", "Share of the rectangle's cells written this iteration.", high=0.98
    ),
}

PATTERNS: tuple[MetricPattern, ...] = (
    _pattern(
        "ppo/kl_epoch{epoch}",
        _m("nats", "KL for one epoch. Per epoch rather than averaged, because the rule for "
                   "n_epochs is read off the spread between the first and the last."),
    ),
    _pattern(
        "ppo/clip_fraction_epoch{epoch}",
        _m("fraction", "Clip fraction for one epoch. If the last epoch's is more than twice the "
                       "first's, lower n_epochs."),
    ),
    _pattern(
        "policy/card_play_frac/{card}",
        _m("fraction", "Share of plays that were this card."),
    ),
    _pattern(
        "env/reward_terms/{term}",
        _m("units", "One weighted reward term's mean per episode."),
    ),
    _pattern("ladder/rating/{member}", _m("Elo", "One pool member's fitted rating.")),
    _pattern("ladder/rating_se/{member}", _m("Elo", "The standard error of that rating.")),
    _pattern(
        "ladder/rating_ci95_lo/{member}",
        _m("Elo", "Lower end of that rating's 95% interval."),
    ),
    _pattern(
        "ladder/rating_ci95_hi/{member}",
        _m("Elo", "Upper end of that rating's 95% interval."),
    ),
    _pattern(
        "health/worker_failures/{kind}",
        _m(
            "count",
            "Worker failures of one kind: exception, crash, timeout or protocol.",
            dtype="int",
        ),
    ),
)

#: Which keys each alarm of section 13.3 reads. ``metrics/alarms.py`` implements the predicates;
#: the suite checks that every key named here exists above.
ALARM_METRICS: dict[str, tuple[str, ...]] = {
    "illegal_actions": ("env/illegal_action_rate",),
    "ratio_invariant": ("ppo/ratio_max_abs_dev",),
    "nonfinite": ("health/nan_guard_trips",),
    "buffer_overflow": ("health/buffer_fill_frac",),
    "worker_failures": ("health/worker_restarts",),
    "worker_failures_persistent": ("health/worker_restarts",),
    "clip_pinned": ("ppo/clip_fraction",),
    "kl_high": ("ppo/kl",),
    "kl_dead": ("ppo/kl",),
    "ev_negative": ("ppo/explained_variance",),
    "noop_collapse": ("policy/cards_per_match",),
    "noop_collapse_severe": ("policy/cards_per_match",),
    "noop_entropy_floor": ("ppo/noop_entropy",),
    "tile_spam": ("policy/tile_top1_share",),
    "artefact_exploit": ("policy/card_tile_top10_share",),
    "draw_equilibrium": ("env/draw_rate", "env/episode_steps_at_cap_frac"),
    "seat_bias": ("env/win_rate_by_seat_ci95_lo", "env/win_rate_by_seat_ci95_hi"),
    "elixir_count_inexact": ("env/elixir_count_exact_frac",),
    "shaping_dominates": ("env/reward_shaping_abs", "env/reward_terminal_abs"),
    "transitivity": ("ladder/transitivity_residual",),
    "gate_starved": ("ladder/consecutive_gate_failures",),
    "capacity_ratio": ("throughput/rollout_capacity_ratio",),
}


def lookup(key: str) -> MetricSpec | None:
    """The spec for one key, fixed or patterned, or None if the schema does not know it."""
    spec = METRICS.get(key)
    if spec is not None:
        return spec
    for pattern in PATTERNS:
        if pattern.regex.match(key):
            return pattern.spec
    return None


def is_known(key: str) -> bool:
    return lookup(key) is not None


def groups() -> tuple[str, ...]:
    """The metric groups, in the order the console prints them."""
    seen: list[str] = []
    for key in METRICS:
        group = key.split("/", 1)[0]
        if group not in seen:
            seen.append(group)
    return tuple(seen)
