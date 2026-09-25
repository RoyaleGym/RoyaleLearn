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
    "ACTOR_UPDATE_KEYS",
    "ALARM_METRICS",
    "METRICS",
    "PATTERNS",
    "RENAMED",
    "MetricPattern",
    "MetricSpec",
    "current_name",
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
    "throughput/parent_wait_frac": _m(
        "fraction",
        "Share of a round the parent spends blocked on workers that have not published; it "
        "rises when the workers cannot keep up. Zero for an inline source, which has nobody "
        "to wait for.",
        low=0.0,
        high=0.3,
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
    "time/probe": _m(
        "s", "Seconds playing the live policy against the scripted rungs this iteration."
    ),
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
    "ppo/policy_loss_choice": _m(
        "loss",
        "The clipped surrogate over the rows whose mask offered more than the no-op. The "
        "companion to policy_loss, which is a mean over every row and therefore a reading of "
        "the elixir bar as much as of the policy; compare two arms of ppo.forced_rows on this.",
    ),
    "ppo/value_loss": _m("loss", "The critic's loss on standardised returns."),
    "ppo/entropy": _m("nats", "Mean entropy of the masked action distribution."),
    "ppo/entropy_normalised": _m(
        "fraction",
        "Entropy over log(legal actions). Raw entropy falling is ambiguous -- a confident "
        "policy or a tighter mask -- and only the normalised form separates them.",
        low=0.3,
        high=0.8,
    ),
    "ppo/logit_std": _m(
        "nats",
        "Spread of the logits over each row's legal set, over the rows that had a choice. The "
        "un-saturated reading of how far the policy is from uniform, and the one to plot: "
        "entropy_normalised is a saturating function of this and spans 0.999991 to 0.994957 "
        "across a run whose policy became 560 times less uniform (hog26-3, 147 iterations), so "
        "the whole of a run's learning lives in its fifth decimal place. This grows linearly "
        "with the pointer head's query norm instead. Starts near 0.015 on the shipped "
        "architecture and rises about 5% an iteration; a decisive policy is of order 1. "
        "The checkpoint quantity it corresponds to is the norm of "
        "actor.head.query.weight, which grew 0.22 -> 1.82 monotonically over 147 "
        "iterations of hog26-3 and 0.22 -> 1.18 over 102 of hog26-6. Runs from before "
        "2026-09-23 do not carry this key, and that norm is how to join them to one "
        "that does.",
    ),
    "ppo/noop_entropy": _m(
        "nats",
        "Binary entropy of play against wait, over the rows whose mask offered more than the "
        "no-op. Conditioned on those rows because a decision the elixir bar cannot afford has "
        "an entropy of zero by construction, and on this environment most decisions are that: "
        "an unconditioned mean measures the elixir curve rather than the policy.",
        low=0.02,
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
    "ppo/explained_variance_choice": _m(
        "fraction",
        "Explained variance over the cells that had a choice. The critic trains on every cell "
        "under every value of ppo.forced_rows, and this is the part of its accuracy the policy "
        "actually reads, so it is the first guardrail when the actor's population changes.",
    ),
    "ppo/forced_frac": _m(
        "fraction",
        "Share of trainable cells whose mask left one action, from the stored column. The "
        "companion to policy/forced_noop_frac, which samples the same quantity in the rollout: "
        "the two disagreeing is a mask the update and the rollout do not share.",
    ),
    "ppo/actor_rows": _m(
        "count",
        "Rows the actor's forward was given this iteration, epochs included. Under "
        "ppo.forced_rows 'all' it is the trainable rows times the epochs; under the skipping "
        "values it is the rows that had a choice, and nothing else in this group moves.",
        dtype="int",
    ),
    "ppo/actor_forwards": _m(
        "count", "Forwards the actor ran this iteration, epochs included.", dtype="int"
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
    "ppo/advantage_std_choice_pre_norm": _m(
        "units",
        "Advantage standard deviation before standardisation, over the cells that had a "
        "choice. Its ratio to advantage_std_pre_norm is how much the entropy terms' weight "
        "against the policy term moves between the values of ppo.forced_rows.",
    ),
    "ppo/advantage_mean_choice": _m(
        "units",
        "Mean standardised advantage over the cells that had a choice. Zero by construction "
        "under ppo.forced_rows 'critic_only_choice_mean'; under the other two it is how far "
        "the actor's baseline sits from the population the actor is applied to.",
    ),
    "ppo/return_running_mean": _m(
        "units", "The return scaler's running mean, recorded and never subtracted."
    ),
    "ppo/return_running_std": _m(
        "units", "The return scaler's running standard deviation: the divisor."
    ),
    "ppo/reward_clip_frac": _m(
        "fraction",
        "Share of rewards the clip bound touched. Healthy is exactly zero: a clipped step is one "
        "where the potential terms stop cancelling, so the shaping can move the optimum, and a "
        "clipped terminal step makes a win worth less than a win.",
        low=0.0,
        high=0.0,
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
    "ppo/adam_eps_floor_frac_actor": _m(
        "fraction",
        "Share of the actor's parameters whose Adam second moment sits under adam_eps, where "
        "the step stops being normalised by the gradient and becomes proportional to it again. "
        "No band, and no absolute reading: what means something is this share against the "
        "CRITIC's in the same row. Measured on hog26-6 (adam_eps 1e-08) at its last "
        "checkpoint: 10.1% of actor parameters under the floor against 16.9% of the critic's, "
        "so the actor was LESS floored than the critic while grad_norm_actor was 0.0068 against "
        "a critic at 26.6. The floor is therefore NOT the account of that asymmetry, which is "
        "what this key was added to find out and what it answered.",
    ),
    "ppo/adam_eps_floor_frac_critic": _m(
        "fraction",
        "The same share for the critic, published beside the actor's because the ASYMMETRY is "
        "the reading: one number alone cannot say whether a floor is this network or this side. "
        "Note that the threshold each is measured against is its OWN optimizer's eps. Measuring "
        "the actor against a value the run did not use answers a counterfactual -- at 1e-5 the "
        "same checkpoint reads 94.8% actor against 29.1% critic -- and quoting that about a run "
        "at 1e-08 is a true number attached to the wrong population.",
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
    "policy/cards_per_100_decisions": _m(
        "cards",
        "Cards played per 100 decisions, over every seat-episode that finished. The companion "
        "to cards_per_match, which is a count and therefore rises with episode length: early "
        "iterations finish only the episodes that end early. Compare two iterations on this.",
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
    "policy/rollout_choice_frac": _m(
        "fraction",
        "Share of the learner's rollout decisions whose mask offered more than the no-op. The "
        "denominator of the three rollout keys below, and the elixir curve read directly: on "
        "this environment it is about one decision in ten.",
    ),
    "policy/rollout_hold_gap": _m(
        "nats",
        "Mean of the no-op's logit minus the log-sum-exp of everything else legal, over choice "
        "rows. What net.noop_bias moves by a constant, so it is the quantity a prior and a "
        "learnt policy can be told apart on.",
    ),
    "policy/rollout_hold_gap_std": _m(
        "nats",
        "Spread of that gap WITHIN an iteration. Confounded on its own: the legal-set size "
        "varies across rows and moves the gap even for an untrained policy. Read the residual.",
    ),
    "policy/rollout_hold_gap_residual_std": _m(
        "nats",
        "The gap's spread after its least-squares dependence on log(n_legal) is removed -- how "
        "differently the policy holds in different states, with the elixir bar taken out. THE "
        "QUESTION IT ANSWERS: a constant noop_bias holds the same way everywhere, and so does a "
        "head that learnt one number; measured across iterations the two are indistinguishable "
        "from a state-dependent policy, and within an iteration they are not. Near zero is a "
        "constant. Only near, because 'one number' gives c - logsumexp(others), which is close "
        "to linear in log(n_legal) rather than linear.",
    ),
    "policy/rollout_hold_rate": _m(
        "fraction",
        "Mean p(no-op) at DECISION time over the rows that had a choice. Not comparable across "
        "iterations on its own -- it moves with how many actions were legal -- which is what "
        "rollout_hold_lift is for.",
    ),
    "policy/rollout_hold_lift": _m(
        "ratio",
        "The same hold mass against the uniform baseline of each row's OWN width, averaged over "
        "choice rows. 1.0 is a policy that has learnt nothing about when to wait; hog26-2 reached "
        "15.3 at iteration 124 while entropy_normalised read 0.980, which is why this key exists: "
        "entropy over 250 actions is nearly blind to the one action with a distinct meaning. No "
        "healthy band, because nobody has yet trained a policy far enough to know what one is.",
    ),
    "policy/rollout_legal_actions": _m(
        "count",
        "Legal actions per CHOICE decision, from the rollout forwards. The baseline "
        "rollout_hold_lift divides by, published so that a lift can be checked against the width "
        "it was computed over rather than assumed.",
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
        "fraction",
        "Mean of the opponent's three towers' hp fraction at the end.",
        low=0.0,
        high=1.0,
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
        "units",
        "Sum over the shaping terms of the mean magnitude of each one's per-episode SUM, for the "
        "shaping_dominates alarm. For a potential term that sum telescopes to 1 - gamma times how "
        "far the potential wandered, so this checks that a term still telescopes and says nothing "
        "about how loud the shaping is: read env/reward_shaping_step_abs for that.",
    ),
    "env/reward_shaping_step_abs": _m(
        "units",
        "Sum over the shaping terms of each one's mean per-episode sum of |F_t|: how loud the "
        "shaping was, step by step, which is what a policy gradient is handed. Compare it with "
        "env/reward_terms_step_abs/terminal. Measured at the shipped weights 2026-09-24: already "
        "about 1.4x the terminal term under random-legal play.",
    ),
    "env/reward_terminal_abs": _m(
        "units", "The terminal reward term's mean magnitude per episode."
    ),
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
        "fraction",
        "The live policy's score rate against the scripted no-op anchor, from the last probe.",
        low=0.9,
        high=1.0,
    ),
    "ladder/score_vs_random_legal": _m(
        "fraction",
        "The live policy's score rate against the random-legal anchor, from the last probe.",
        low=0.7,
        high=1.0,
    ),
    "ladder/probe_seconds_frac": _m(
        "fraction",
        "Share of wall clock spent probing the live policy against the scripted rungs.",
        high=0.1,
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
    "health/housekeeping_failures": _m(
        "count",
        "Retries that did not take, across pruning, metric compaction and evaluation-worker "
        "shutdown. Each of those actions swallows its own error to keep the run alive, so this "
        "is the only place their failure is visible; a non-zero value means the run is carrying "
        "a directory, a file or a process it meant to be rid of.",
        dtype="int",
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
    # The three that separate the memory regimes, plus the retry counter. An update that appears
    # to slow down across iterations has four candidate explanations and these tell them apart
    # without an argument: reservation growth squeezing the workspace (reserved grows,
    # inactive_split flat, driver free shrinks), fragmentation inside a fixed reservation
    # (reserved flat, inactive_split grows), something outside the caching allocator holding
    # memory (all flat but driver free shrinking), or no memory mechanism at all (all three
    # flat). The last is the one worth being able to rule out: without it a plausible story
    # about fragmentation can be refined indefinitely against data that never supported it.
    #
    # That is not hypothetical. These four keys were added to explain an apparent 16% slowdown
    # between the two iterations of a fixed configuration, seen twice an hour apart. Their first
    # run returned reserved and inactive_split identical to a tenth of a megabyte across both
    # iterations, with zero retries -- and the same run did not slow down at all. The two
    # "replications" had been taken under the same machine load, so they agreed with each other
    # and not with the truth. Four lines of measurement ended a mechanism two people had spent
    # an hour refining.
    "health/vram_reserved_mb": _m("MB", "Device memory the caching allocator holds."),
    "health/vram_inactive_split_mb": _m(
        "MB", "Non-releasable memory inside the allocator's blocks: fragmentation."
    ),
    "health/vram_driver_free_mb": _m(
        "MB", "Free device memory the driver reports, which is what bounds a cuDNN workspace."
    ),
    "health/vram_available_mb": _m(
        "MB",
        "Free device memory plus what this process holds: what it could occupy if it asked.",
    ),
    "health/vram_needed_mb": _m(
        "MB",
        "One minibatch's device peak measured at startup, plus doctor.vram_headroom_mb.",
    ),
    "health/vram_alloc_retries": _m(
        "count",
        "Times the allocator freed its cache and retried. A retry is a synchronising stall; "
        "zero does not mean the allocator is innocent, only that this path was not taken.",
        dtype="int",
    ),
    "health/rss_peak_mb": _m("MB", "Peak resident memory of the parent this iteration."),
    "health/buffer_fill_frac": _m(
        "fraction", "Share of the rectangle's cells written this iteration.", high=0.98
    ),
    # -- imitation (section 19; only with the imitation block) -------------------------
    "imitation/actor_lr_scale": _m(
        "multiplier",
        "imitation.actor_lr_scale at this iteration's clock. It multiplies the backoff's actor "
        "learning rate, and zero freezes the actor.",
        low=0.0,
    ),
    "imitation/actor_frozen": _m(
        "flag",
        "1 on an iteration the actor was frozen: no actor loss, no actor step, and the ppo/ keys "
        "computed from the actor's forward are left out of the row.",
    ),
    "imitation/ev_at_unfreeze": _m(
        "fraction",
        "ppo/explained_variance on the first iteration after a frozen stretch: how ready the "
        "critic was when the actor started to move. Published on that row only.",
        low=0.3,
    ),
    "imitation/iterations_since_unfreeze": _m(
        "count",
        "Unfrozen iterations since the last frozen stretch, this one included. The handoff "
        "alarm watches the first imitation_handoff_window of them.",
    ),
}

PATTERNS: tuple[MetricPattern, ...] = (
    _pattern(
        "ppo/kl_epoch{epoch}",
        _m(
            "nats",
            "KL for one epoch. Per epoch rather than averaged, because the rule for "
            "n_epochs is read off the spread between the first and the last.",
        ),
    ),
    _pattern(
        "ppo/clip_fraction_epoch{epoch}",
        _m(
            "fraction",
            "Clip fraction for one epoch. If the last epoch's is more than twice the "
            "first's, lower n_epochs.",
        ),
    ),
    _pattern(
        "policy/card_play_frac/{card}",
        _m("fraction", "Share of plays that were this card."),
    ),
    _pattern(
        "policy/card_in_hand_frac/{card}",
        _m("fraction", "Share of sampled decisions where this card was in the hand."),
    ),
    _pattern(
        "policy/card_legal_frac/{card}",
        _m(
            "fraction",
            "Share of sampled decisions where this card was in the hand AND affordable.",
        ),
    ),
    _pattern(
        "policy/card_play_rate/{card}",
        _m(
            "fraction",
            "Plays of this card over the decisions where it was affordable: the policy's "
            "preference with the elixir economy divided out. A cheap card is legal far more "
            "often than an expensive one, so the play SHARE reflects both and this reflects "
            "one. Absent for a card that was never affordable in the sample.",
        ),
    ),
    _pattern(
        "env/reward_terms/{term}",
        _m("units", "One weighted reward term's mean per episode, signed."),
    ),
    _pattern(
        "env/reward_terms_abs/{term}",
        _m(
            "units",
            "One weighted reward term's per-episode SUM, as a mean magnitude over seats. A "
            "potential term's sum telescopes; see env/reward_terms_step_abs/{term}.",
        ),
    ),
    _pattern(
        "env/reward_terms_step_abs/{term}",
        _m(
            "units",
            "One weighted reward term's per-episode sum of |F_t|, as a mean over seats: how loud "
            "the term was. Linear in the term's weight. Absent for episodes recorded before "
            "2026-09-24.",
        ),
    ),
    _pattern(
        "env/play_rate_by_elixir/{elixir}",
        _m(
            "fraction",
            "Of the learner's sampled choice rows at this whole elixir (the bar floored, with "
            "a 0.005 tolerance for the vector's half precision), the share where it played. "
            "Absent for an elixir level with no choice row in the sample.",
        ),
    ),
    _pattern(
        "imitation/{name}/kl",
        _m(
            "nats",
            "KL(reference || policy), the mean over the choice rows the regulariser covered "
            "in epoch 1. The number lambda is moved by. Absent when no row was covered.",
        ),
    ),
    _pattern(
        "imitation/{name}/kl_noop",
        _m("nats", "The play/wait part of kl, by the chain rule."),
    ),
    _pattern(
        "imitation/{name}/kl_card",
        _m("nats", "p_ref(play) times the KL of the card given a play. Joint factor only."),
    ),
    _pattern(
        "imitation/{name}/kl_tile",
        _m("nats", "The reference-weighted KL of the tile given the card. Joint factor only."),
    ),
    _pattern(
        "imitation/{name}/lambda",
        _m("coefficient", "The coefficient this iteration's loss used.", low=0.0),
    ),
    _pattern(
        "imitation/{name}/lambda_at_max",
        _m(
            "flag",
            "1 when the coefficient this iteration used was coef.max: the anchor is pulling as "
            "hard as it is allowed to.",
        ),
    ),
    _pattern(
        "imitation/{name}/budget",
        _m("nats", "The budget kl is held to this iteration.", low=0.0),
    ),
    _pattern(
        "imitation/{name}/grad_ratio",
        _m(
            "ratio",
            "||grad of the unscaled KL|| / ||grad of the policy term|| on the first minibatch "
            "with a choice row. Zero while the policy equals the reference. For setting "
            "coef.start.",
        ),
    ),
    _pattern(
        "imitation/{name}/rows_frac",
        _m("fraction", "Share of epoch-1 choice rows the regulariser covered (exclude_when)."),
    ),
    _pattern(
        "imitation/{name}/top1_agree",
        _m(
            "fraction",
            "Share of covered rows where the reference's and the policy's most likely actions "
            "agree; under noop_marginal, whether both would play.",
        ),
    ),
    _pattern(
        "imitation/{name}/ref_p_noop",
        _m("probability", "The reference's mean p(no-op) over the covered rows."),
    ),
    _pattern(
        "ladder/score_vs/{opponent}",
        _m("fraction", "The live policy's score rate against one scripted rung."),
    ),
    _pattern(
        "ladder/score_vs_n/{opponent}",
        _m(
            "count",
            "Seeds behind that score rate. Each seed is two battles, one from each side, and "
            "the seed is the unit: the two are correlated and counting them separately would "
            "overstate the sample.",
            dtype="int",
        ),
    ),
    _pattern(
        "ladder/score_vs_ci95_lo/{opponent}",
        _m("fraction", "Lower end of that score rate's 95% bootstrap interval over seeds."),
    ),
    _pattern(
        "ladder/score_vs_ci95_hi/{opponent}",
        _m("fraction", "Upper end of that score rate's 95% bootstrap interval over seeds."),
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
        "health/housekeeping/{kind}",
        _m(
            "count",
            "Housekeeping failures of one kind: prune, compaction or eval_shutdown. Present "
            "only when that kind has failed.",
            dtype="int",
        ),
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
    # Section 19.9. A template names a family: the alarm reads every key of it the row carries.
    "imitation_ref_kl_high": ("imitation/{name}/kl",),
    "imitation_lambda_saturated": ("imitation/{name}/lambda_at_max",),
    "imitation_handoff": (
        "ppo/kl",
        "ppo/clip_fraction",
        "imitation/iterations_since_unfreeze",
    ),
    "imitation_critic_unready": ("imitation/ev_at_unfreeze",),
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
    "reward_clipped": ("ppo/reward_clip_frac",),
    "vram_spilling": ("time/update", "health/vram_driver_free_mb"),
    "transitivity": ("ladder/transitivity_residual",),
    "gate_starved": ("ladder/consecutive_gate_failures",),
    "capacity_ratio": ("throughput/rollout_capacity_ratio",),
}


#: Keys this schema used to publish, and what they are called now.
#:
#: A metric name is written into every row of every run that was made while it was live, and
#: those rows outlive the name: two iterations of this project's first real training data are
#: addressed by ``throughput/worker_idle_frac``, which the schema no longer has. Nothing can
#: rewrite a file that is already on disk, so what is offered instead is the mapping -- one
#: place that says what an old name became, so a reader plotting several runs together does not
#: have to know the history to line them up.
#:
#: An entry is added in the same commit as the rename. ``tests/test_metric_names.py`` holds
#: every value here to being a key the schema currently publishes.
RENAMED: dict[str, str] = {
    # It measured the parent blocked on workers that had not published, and rose when the
    # workers could not keep up -- the opposite of what a reader would do about a worker idling.
    "throughput/worker_idle_frac": "throughput/parent_wait_frac",
}


#: Keys a healthy row may legitimately omit, with the condition that governs each.
#:
#: Absence is already how this harness says "there is nothing to report": ``episode_fields([])``
#: returns ``env/episodes_completed`` and nothing else, and ``alarms.py`` guarantees that a
#: missing key never fires. What was missing was a way to say WHICH keys that applies to, so the
#: distinction lived in whichever function happened to fill a default. Declaring it here makes
#: the row contract checkable: every other key must be present in every row.
#: The ``ppo/`` keys computed from the actor's forward in the update. A frozen actor (section
#: 19.5) ran none, and the diagnostics' 0.0 for them is the healthiest reading each has, so a
#: frozen row leaves them out rather than publish it.
ACTOR_UPDATE_KEYS: tuple[str, ...] = (
    "ppo/policy_loss",
    "ppo/policy_loss_choice",
    "ppo/entropy",
    "ppo/noop_entropy",
    "ppo/entropy_normalised",
    "ppo/logit_std",
    "ppo/kl",
    "ppo/clip_fraction",
    "ppo/dual_clip_fraction",
    "ppo/grad_norm_actor",
    "ppo/update_magnitude_actor",
)

CONDITIONAL: dict[str, str] = {
    **dict.fromkeys(
        ACTOR_UPDATE_KEYS,
        "the actor trained this iteration: imitation.actor_lr_scale was above zero, which it "
        "always is without the imitation block",
    ),
    "imitation/actor_lr_scale": "the run has an imitation block that sets actor_lr_scale",
    "imitation/actor_frozen": "the run has an imitation block that sets actor_lr_scale",
    "imitation/ev_at_unfreeze": "this iteration is the first after a frozen stretch",
    "imitation/iterations_since_unfreeze": "the actor has been unfrozen after a frozen stretch",
    # Written only on the iterations that probed the live policy, which is none of them unless
    # ladder.probe_every_iterations is set. Carrying the last probe's number forward would
    # publish a score for weights that have moved since, and a 0.5 for a rung nobody played is
    # indistinguishable from an even contest, which is what these two used to do.
    "ladder/score_vs_noop": "a probe measured the live policy against this anchor this iteration",
    "ladder/score_vs_random_legal": (
        "a probe measured the live policy against this anchor this iteration"
    ),
    "ladder/probe_seconds_frac": "a probe has run in this run",
    # An episode written before the recorder kept per-step magnitudes was never measured, and a
    # zero would say its shaping was silent.
    "env/reward_shaping_step_abs": (
        "a learner episode finished this iteration and its record carries per-step magnitudes"
    ),
    # The same rule for the rest of the ladder group. Every one of these had a neutral value that
    # reads as a measurement: an Elo of zero, uncorrelated seats, a gate that cost nothing, a
    # perfectly transitive rating, and a learner exactly as good as the first snapshot.
    "ladder/elo_readout": "a training game has been scored this run",
    "ladder/paired_rho": (
        "a gate has played a champion comparison and its paired-seed correlation is defined"
    ),
    "ladder/gate_seconds_frac": (
        "the run's gate total is known: absent only after resuming a checkpoint written before "
        "it was recorded"
    ),
    "ladder/transitivity_residual": "a rating fit exists",
    "ladder/rating_above_v0": "the fit holds both the learner and the first snapshot",
    "ladder/gate_observed_rate": "a gate decision carries its champion condition",
    "ladder/gate_lower_bound": "a gate decision carries its champion condition",
    # Absent without a device rather than zero. A zero here would read as "no fragmentation"
    # and "no free memory" on a machine that simply has no GPU to report either about, which is
    # the same sentinel this file exists to stop. ``vram_peak_mb`` predates the rule and still
    # returns 0.0.
    #
    # "The RUN is on a CUDA device", not "the machine has one". They came apart on this machine,
    # which holds a card that the suite's CPU runs never touch: every CPU run was publishing
    # that card's driver-free figure as its own, and test_resume's row-for-row comparison flaked
    # on it whenever a sibling session was using the GPU.
    "throughput/gpu_util_frac": "a CUDA device is present and its utilisation can be read",
    "health/rss_peak_mb": "the platform reports a peak working set",
    "health/vram_reserved_mb": "the RUN is on a CUDA device",
    "health/vram_inactive_split_mb": "the RUN is on a CUDA device",
    "health/vram_driver_free_mb": "the RUN is on a CUDA device",
    "health/vram_available_mb": "the RUN is on a CUDA device",
    "health/vram_needed_mb": (
        "the RUN is on a CUDA device, the preflight gate is enabled, and its probe did not raise"
    ),
    "health/vram_alloc_retries": "the RUN is on a CUDA device",
    # An iteration whose every decision was forced measured nothing about the policy. A lift of
    # 1.0 would read as "exactly uniform", a hold rate of 1.0 as "it never plays", and both are
    # statements about the elixir bar.
    "time/overlap_saved": "rollout.overlap is honoured, which it is not yet (spec 14.1)",
    "ppo/adam_eps_floor_frac_actor": "the actor's optimizer has taken a step",
    "ppo/adam_eps_floor_frac_critic": "the critic's optimizer has taken a step",
    "policy/rollout_hold_gap": "a rollout decision this iteration had more than one legal action",
    "policy/rollout_hold_gap_std": (
        "a rollout decision this iteration had more than one legal action"
    ),
    "policy/rollout_hold_gap_residual_std": (
        "a rollout decision this iteration had more than one legal action"
    ),
    "policy/rollout_hold_rate": "a rollout decision this iteration had more than one legal action",
    "policy/rollout_hold_lift": "a rollout decision this iteration had more than one legal action",
    "policy/rollout_legal_actions": (
        "a rollout decision this iteration had more than one legal action"
    ),
}


def current_name(key: str) -> str:
    """What a key from an older run is called now, or the key itself if it has not moved."""
    return RENAMED.get(key, key)


def lookup(key: str) -> MetricSpec | None:
    """The spec for one key, fixed or patterned, or None if the schema does not know it."""
    key = current_name(key)
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
