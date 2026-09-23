"""One iteration as one flat row, assembled from the three places numbers come from.

The environment reports through the round scalars and the episode records a worker emits; the
learner through ``UpdateResult`` and the iteration's ``ScheduleState``; the ladder through the
rating table and the last gate decision. Merging them here rather than in the loop means the
row is built the same way whether it is written by a training run, a smoke test or a resume,
and that ``metrics/schema.py`` can be checked against one function instead of against a loop.

Aggregation of per-step quantities happens in the worker, not here: a metric that is always
reduced to a mean should be reduced where it is produced. What crosses the boundary is one
fixed record per finished episode, and what this module does with those is arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np

from ..api.ladder import GateDecision, RatingTable
from ..api.metrics import MetricValue
from ..api.rollout import EpisodeRecord
from ..ladder.pool import LEARNER_ID
from ..ladder.rating import Z95, wilson_interval
from . import schema

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.schedule import ScheduleState
    from ..api.update import UpdateResult
    from ..ladder.evaluate import Comparison
    from ..ladder.pool import LadderPool
    from ..learn.inference import RoundStats

__all__ = [
    "EpisodeAggregate",
    "IterationMetrics",
    "episode_fields",
    "flatten",
    "ladder_fields",
    "rollout_policy_fields",
    "schedule_fields",
    "unknown_keys",
    "update_fields",
]

#: The reward term the terminal reward is filed under. Everything else in ``reward_terms`` is
#: shaping, which is what the ``shaping_dominates`` alarm compares against it.
TERMINAL_REWARD_TERM = "terminal"

#: The other names an objective arrives under. A composition names its own terminal term, but a
#: reward assembled from RoyaleGym's ``CombinedReward`` names every term after its class, and a
#: config is free to use one: the shipped configs did until 23d971c. Matching on a class name is
#: fragile, which is why it is a fallback and why the list is here rather than inside the loop.
TERMINAL_REWARD_CLASSES = ("WinLossReward",)


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, MetricValue]:
    """A nested mapping as ``group/name`` keys.

    One level or ten: a sink writes what it is given and never indexes a fixed key list, so the
    shape of a row is decided here and nowhere else.
    """
    flat: dict[str, MetricValue] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}/"))
        else:
            flat[name] = value
    return flat


def unknown_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Keys the schema does not know, sorted. The suite turns this into a failure."""
    return tuple(sorted(key for key in row if not schema.is_known(key)))


class EpisodeAggregate(msgspec.Struct, frozen=True):
    """One iteration's finished episodes, reduced.

    Battles are counted once and seats are counted once each: a mirror battle produces two
    learner seats of the same episode, and a draw rate that counted both would be a draw rate
    over seats, which is not what the word means.
    """

    battles: int
    seats: int
    fields: dict[str, MetricValue]


def _battle_records(records: Sequence[EpisodeRecord]) -> list[EpisodeRecord]:
    """One record per finished battle, preferring the blue seat's view of it."""
    chosen: dict[tuple[int, int, int, int], EpisodeRecord] = {}
    for record in records:
        key = (record.worker, record.shard, record.battle, record.ordinal)
        held = chosen.get(key)
        if held is None or (held.seat != 0 and record.seat == 0):
            chosen[key] = record
    return [chosen[key] for key in sorted(chosen)]


def _blue_outcome(record: EpisodeRecord) -> int:
    """A battle's outcome as a sign from blue's side, whichever seat reported it.

    ``EpisodeRecord.outcome`` is the reporting seat's own sign: ``+1`` it won, ``-1`` it lost,
    ``0`` a draw or a truncation. Red's view of a battle is blue's negated, which is what makes
    one record per battle enough to score it.
    """
    return record.outcome if record.seat == 0 else -record.outcome


def _terminal_name(terms: Mapping[str, Any], declared: str) -> str | None:
    """Which of these term names is the objective, or None when none of them is.

    The declared name first, because a composition that names its own terminal term is saying so
    on purpose. Then the classes a reward assembled elsewhere files its objective under. A reward
    that carries neither leaves the objective unidentified, which is a different answer from zero.
    """
    if declared in terms:
        return declared
    for name in TERMINAL_REWARD_CLASSES:
        if name in terms:
            return name
    return None


def episode_fields(
    records: Sequence[EpisodeRecord],
    *,
    truncation_steps: int | None = None,
    terminal_term: str = TERMINAL_REWARD_TERM,
    histogram_bins: int = 12,
) -> EpisodeAggregate:
    """The ``env/`` group and ``policy/cards_per_match``, from the episodes that finished."""
    battles = _battle_records(records)
    fields: dict[str, MetricValue] = {}
    if not records or not any(record.policy_id == LEARNER_ID for record in records):
        # No seat of the learner's finished, so there is nothing to say about how it played. An
        # iteration of opponents' episodes alone is not a zero.
        return EpisodeAggregate(battles=0, seats=0, fields={"env/episodes_completed": 0})

    steps = np.array([record.episode_steps for record in battles], dtype=np.float64)
    fields["env/episodes_completed"] = len(battles)
    fields["env/episode_steps_mean"] = float(steps.mean())
    for percentile, name in ((5, "p05"), (50, "p50"), (95, "p95")):
        fields[f"env/episode_steps_{name}"] = float(np.percentile(steps, percentile))
    counts, edges = np.histogram(steps, bins=histogram_bins)
    fields["env/episode_steps_hist"] = msgspec.json.encode(
        {"edges": [float(edge) for edge in edges], "counts": [int(count) for count in counts]}
    ).decode("utf-8")
    fields["env/episode_steps_at_cap_frac"] = (
        float(np.mean(steps >= truncation_steps)) if truncation_steps else 0.0
    )
    fields["env/ticks_mean"] = float(
        np.mean([record.episode_ticks for record in battles])
    )

    # A draw and a truncation are both a zero: neither side took the match, which is what the
    # draw rate counts and what has to be kept out of the decided games the seat share is over.
    outcomes = [_blue_outcome(record) for record in battles]
    draws = sum(1 for outcome in outcomes if outcome == 0)
    decided = [outcome for outcome in outcomes if outcome != 0]
    fields["env/draw_rate"] = draws / len(battles)
    blue_share = (
        sum(1 for outcome in decided if outcome > 0) / len(decided) if decided else 0.5
    )
    fields["env/win_rate_by_seat"] = blue_share
    lo, hi = wilson_interval(blue_share, len(decided))
    fields["env/win_rate_by_seat_ci95_lo"] = lo
    fields["env/win_rate_by_seat_ci95_hi"] = hi

    # The seat-level numbers are the LEARNER's, over the seats it actually played. A battle that
    # is not a mirror has an opponent in its other seat, and that opponent's crowns, towers,
    # elixir and card play are not the policy's behaviour: averaging them in means every one of
    # these reads as a blend of the learner and whatever it happened to be drawn against, and the
    # blend moves as the mixture and the pool move. `random_legal` alone kept `cards_per_match`
    # at about 3.9, which is above the 3.0 the severe no-op-collapse alarm halts on, so a learner
    # collapsing to nothing could be held over the line by its opponent. A mirror battle has the
    # learner in both seats and both count.
    #
    # It is NOT that random_legal plays a card whenever it can; this comment said so until
    # 2026-09-22 and the train session built an argument on it. `RANDOM_LEGAL_NOOP_PROB` is 0.9
    # and it takes the no-op with that probability regardless of affordability
    # (`rollout/scripted.py:52`). The uniform-over-the-action-space opponent that really does
    # play the instant it can afford to was deliberately replaced by this one, and the constant's
    # own docstring says why. 3.9 a match is what nine-in-ten no-ops looks like.
    own = [record for record in records if record.policy_id == LEARNER_ID]
    seat_steps = float(sum(record.episode_steps for record in own))
    fields["env/crowns_for"] = float(np.mean([record.own_crowns for record in own]))
    fields["env/crowns_against"] = float(np.mean([record.enemy_crowns for record in own]))
    fields["env/crown_diff"] = fields["env/crowns_for"] - fields["env/crowns_against"]
    fields["env/tower_hp_frac_end_own"] = float(
        np.mean([record.own_tower_hp_frac for record in own])
    )
    fields["env/tower_hp_frac_end_enemy"] = float(
        np.mean([record.enemy_tower_hp_frac for record in own])
    )
    fields["env/elixir_leak_frac"] = (
        sum(record.elixir_leak_steps for record in own) / seat_steps if seat_steps else 0.0
    )
    fields["env/elixir_count_exact_frac"] = float(
        np.mean([1.0 if record.elixir_count_exact else 0.0 for record in records])
    )
    # Every seat, because a refused command is the mask's fault whoever sent it, and a scripted
    # opponent draws from the same mask.
    all_steps = float(sum(record.episode_steps for record in records))
    fields["env/illegal_action_rate"] = (
        sum(record.illegal_commands for record in records) / all_steps if all_steps else 0.0
    )
    fields["policy/cards_per_match"] = float(
        np.mean([record.cards_played for record in own])
    )
    # The same count over the decisions it had to spend, because the count alone is a length
    # measurement wearing a policy's name. Iteration 1 can only finish the episodes that end
    # early, so its mean episode ran 171 steps and scored 6.98 cards; iteration 2 reached 328
    # steps and 22.08, and the integration log read the jump as the policy learning to play.
    # It was the episode length, and a near-uniform policy scores about 22 as soon as episodes
    # run their length. Per decision the two iterations are 0.041 and 0.068, which is a real
    # move and a much smaller one. Both are published: the count is what a reader of a battle
    # recognises, and the rate is what two iterations can be compared on.
    fields["policy/cards_per_100_decisions"] = (
        100.0 * float(sum(record.cards_played for record in own)) / seat_steps
        if seat_steps
        else 0.0
    )

    # Two numbers per term, because one of them is zero by construction. Every shipped reward is
    # zero sum, and both seats of a battle are in this batch, so a term's signed mean is zero
    # whatever the term did: red's tower damage is blue's negated. That mean is still worth
    # publishing -- it is how a term that is NOT antisymmetric shows itself -- but the two shares
    # the shaping_dominates alarm reads are magnitudes, so they take the absolute value on each
    # seat's own number rather than on the mean. Taking it on the mean, as this did until
    # 2026-09-22, made both shares exactly 0.0 in every row ever recorded, and an alarm that
    # compares two structural zeroes cannot fire.
    terms: dict[str, list[float]] = {}
    for record in records:
        for name, value in record.reward_terms.items():
            terms.setdefault(name, []).append(float(value))
    # Which term is the objective, before anything is added up. If none of the names is one this
    # module recognises, the objective is NOT zero: it is unknown, and the two shares are
    # published without it, so shaping_dominates stays silent under the missing-key rule instead
    # of comparing the shaping against a zero it invented. That is what it did until 2026-09-22:
    # the name never matched, the objective was added into the shaping, and the comparison was
    # 1.379 > 0.0 on every row of every run.
    objective = _terminal_name(terms, terminal_term)
    shaping = 0.0
    terminal = 0.0
    for name, values in sorted(terms.items()):
        fields[f"env/reward_terms/{name}"] = float(np.mean(values))
        magnitude = float(np.mean(np.abs(values)))
        fields[f"env/reward_terms_abs/{name}"] = magnitude
        if name == objective:
            terminal += magnitude
        else:
            shaping += magnitude
    fields["env/reward_shaping_abs"] = shaping
    if objective is not None:
        fields["env/reward_terminal_abs"] = terminal
    return EpisodeAggregate(battles=len(battles), seats=len(records), fields=fields)


def update_fields(result: UpdateResult) -> dict[str, MetricValue]:
    """The ``ppo/`` group, straight off what the update reported about itself."""
    fields: dict[str, MetricValue] = {
        "ppo/policy_loss": result.policy_loss,
        "ppo/value_loss": result.value_loss,
        "ppo/entropy": result.entropy,
        "ppo/noop_entropy": result.noop_entropy,
        "ppo/entropy_normalised": result.entropy_normalised,
        "ppo/logit_std": result.logit_std,
        "ppo/kl": result.kl,
        "ppo/clip_fraction": result.clip_fraction,
        "ppo/dual_clip_fraction": result.dual_clip_fraction,
        "ppo/explained_variance": result.explained_variance,
        "ppo/explained_variance_choice": result.explained_variance_choice,
        "ppo/policy_loss_choice": result.policy_loss_choice,
        "ppo/forced_frac": result.forced_frac,
        "ppo/actor_rows": result.actor_rows,
        "ppo/actor_forwards": result.actor_forwards,
        "ppo/ratio_max_abs_dev": result.ratio_max_abs_dev,
        "ppo/grad_norm_actor": result.grad_norm_actor,
        "ppo/grad_norm_critic": result.grad_norm_critic,
        "ppo/update_magnitude_actor": result.update_magnitude_actor,
        "ppo/update_magnitude_critic": result.update_magnitude_critic,
        "ppo/n_minibatches": result.n_minibatches,
        "ppo/n_optimizer_steps": result.n_optimizer_steps,
        "ppo/samples_unused_frac": result.samples_unused_frac,
    }
    # Per epoch rather than averaged: the rule for n_epochs is read off the spread between the
    # first and the last, and an average cannot answer that.
    for epoch, value in enumerate(result.kl_by_epoch):
        fields[f"ppo/kl_epoch{epoch}"] = value
    for epoch, value in enumerate(result.clip_fraction_by_epoch):
        fields[f"ppo/clip_fraction_epoch{epoch}"] = value
    # Absent before an optimizer has stepped. A 0.0 there would say "fully adaptive", which is
    # the reading the key exists to correct.
    for side in ("actor", "critic"):
        floor = getattr(result, f"adam_eps_floor_frac_{side}", None)
        if floor is not None:
            fields[f"ppo/adam_eps_floor_frac_{side}"] = floor
    return fields


def rollout_policy_fields(stats: RoundStats) -> dict[str, MetricValue]:
    """What the policy did at DECISION time, from the rollout's own forwards.

    ``update_fields`` describes the policy the optimizer saw, over the rows it trained on and
    after the first epoch has already moved them. This describes the policy that actually chose
    the actions, and it costs nothing: ``_StatAccumulator`` has summed these quantities since the
    file was written and no reader ever read them.

    ``rollout_hold_lift`` is the key the harness was missing. A policy's hold rate on its own is
    uninterpretable -- it is high because the elixir bar is usually empty -- and the entropy is
    nearly blind to it: over 250 legal actions, a policy putting **15.3x uniform mass** on the
    no-op still reads 0.980 of maximum normalised entropy, because the other 249 actions carry
    almost all of the sum. That pair was measured on hog26-2 at iteration 124 and reproduced from
    first principles in ``tests/test_hold_lift.py``. The lift divides each row's hold mass by the
    uniform baseline of its own width, so 1.0 is a policy that has learnt nothing about when to
    wait and the distance from 1.0 is the only part of it that is about the policy.
    """
    if stats.rows <= 0:
        return {}
    fields: dict[str, MetricValue] = {
        "policy/rollout_choice_frac": stats.choice_rows / stats.rows,
    }
    if not stats.choice_rows:
        # A lift of 1.0 reads as "the policy is exactly uniform on its hold", which is a
        # measurement. An iteration whose every decision was forced made no measurement.
        return fields
    fields["policy/rollout_hold_rate"] = stats.hold / stats.choice_rows
    fields["policy/rollout_hold_lift"] = stats.hold_lift / stats.choice_rows
    fields["policy/rollout_legal_actions"] = stats.choice_n_legal / stats.choice_rows
    return fields


def schedule_fields(state: ScheduleState, *, decision_ms: int) -> dict[str, MetricValue]:
    """The ``run/`` group that moves with the run."""
    return {
        "run/iteration": state.iteration,
        "run/cumulative_timesteps": state.cumulative_timesteps,
        "run/cumulative_env_steps": state.cumulative_env_steps,
        "run/gamma": state.gamma,
        "run/gae_lambda": state.gae_lambda,
        "run/credit_horizon_seconds": state.credit_horizon_seconds(decision_ms),
        "run/ent_coef": state.ent_coef,
        "run/ent_coef_noop": state.ent_coef_noop,
        "run/lr_actor": state.lr_actor,
        "run/lr_critic": state.lr_critic,
        "ppo/lr_backoff_events": state.lr_backoff_events,
    }


def ladder_fields(
    pool: LadderPool,
    *,
    ratings: RatingTable | None = None,
    decision: GateDecision | None = None,
    elo: float | None = None,
    paired_rho: float | None = None,
    draw_rate_eval: float | None = None,
    gate_seconds_frac: float | None = None,
    learner_id: str = "learner",
    members: Iterable[str] | None = None,
    rungs: Mapping[str, Comparison] | None = None,
    probe_seconds_frac: float | None = None,
) -> dict[str, MetricValue]:
    """The ``ladder/`` group: the fit, the pool's shape, the last gate and the last probe.

    The per-member ratings are written for the members named, which is the sampler rather than
    the archive: an archive of five hundred would be five hundred columns a run after run, and
    every one of them is still in the result log where the fit can be redone.

    ``rungs`` is the live policy's last probe, one comparison per scripted opponent, and it is
    passed only on the iterations that ran one. Carrying the previous probe's numbers forward
    would publish a score for weights that have since moved, and the whole reason these keys
    were empty before the probe existed is that a number which lags the policy is worse than a
    missing one: the absence is visible.
    """
    from ..ladder.gate import failed_condition
    from ..ladder.pool import SCRIPTED_NOOP, SCRIPTED_RANDOM_LEGAL

    fields: dict[str, MetricValue] = {
        "ladder/champion_id": pool.champion or "",
        "ladder/champion_step": pool.step_of(pool.champion) if pool.champion else 0,
        "ladder/pool_size": len(pool.members()),
        "ladder/sampler_size": len(pool.sampler()),
        "ladder/gate_attempts": pool.state.gate_attempts,
        "ladder/gate_passes": pool.state.gate_passes,
        "ladder/consecutive_gate_failures": pool.state.consecutive_gate_failures,
        "ladder/evictions": pool.state.evictions,
        # Counted from the log rather than from ``pool.state.eval_games``. That counter lives in
        # ``LadderPool.record`` (pool.py:226) and is keyed on KIND_EVAL correctly, but the only
        # caller of ``record`` in the package is ``record_training_results``, which hardcodes
        # kind=KIND_TRAIN -- so the sum it takes is always zero. The evaluation runner writes
        # around the pool, appending straight to the shared ResultLog (evaluate.py:250), which
        # is why a run could fit ``ladder/rating/snap:v0`` off 24 real eval battles while this
        # key read 0. Measured 2026-09-22: 0 in all 124 metric rows on disk, at every run
        # length, which is what "structurally unreachable" looks like from the outside.
        "ladder/eval_games_total": len(pool.eval_view()),
    }
    # Each of these is absent until it has been measured. The neutral value of every one of them
    # is a statement: an Elo of zero, seats that are uncorrelated, a gate that cost nothing. A
    # reader cannot tell such a number from a measurement, and a plot of it draws a flat line
    # through the part of the run where nothing was computed.
    if elo is not None:
        fields["ladder/elo_readout"] = float(elo)
    if paired_rho is not None:
        fields["ladder/paired_rho"] = float(paired_rho)
    if gate_seconds_frac is not None:
        fields["ladder/gate_seconds_frac"] = float(gate_seconds_frac)
    if probe_seconds_frac is not None:
        fields["ladder/probe_seconds_frac"] = float(probe_seconds_frac)
    view = pool.eval_view()
    # The two anchor keys, and the same score with its n and its interval for every rung the
    # probe played. Both exist because the anchors are the two rungs a reader already knows by
    # name and the older rows address them that way, while the family is what lets a run add a
    # rung without adding a key nobody declared.
    #
    # These used to be read out of the result log, off the ("learner", anchor) pair, which is a
    # pair no evaluator could write: every game in that log is a frozen snapshot against
    # something, because the gate hands ``snap:v{n}`` to the runner. Measured 2026-09-22:
    # exactly 0.5 in all 124 metric rows on disk -- the value ``PairRecord.score_a`` answers for
    # a pair with no games, and also what a genuine even contest looks like -- including on a
    # run that had played 24 real evaluation battles. Reading the latest SNAPSHOT's record
    # instead would have been tempting and wrong: those weights lag the live policy by the
    # candidate cadence, 4,000,000 env steps on the shipped laptop profile, and a number that
    # silently lags by that much is worse than an absent one. So the score now comes from a
    # measurement of the live policy or from nowhere. ``alarms.py`` guarantees that a missing
    # key never fires.
    for opponent, comparison in sorted((rungs or {}).items()):
        rung = opponent.split(":", 1)[-1]
        fields[f"ladder/score_vs/{rung}"] = float(comparison.score_a)
        fields[f"ladder/score_vs_n/{rung}"] = int(comparison.n_seeds)
        fields[f"ladder/score_vs_ci95_lo/{rung}"] = float(comparison.lo)
        fields[f"ladder/score_vs_ci95_hi/{rung}"] = float(comparison.hi)
    for key, anchor in (
        ("ladder/score_vs_noop", SCRIPTED_NOOP),
        ("ladder/score_vs_random_legal", SCRIPTED_RANDOM_LEGAL),
    ):
        anchor_probe = (rungs or {}).get(anchor)
        if anchor_probe is not None:
            fields[key] = float(anchor_probe.score_a)
    fields["ladder/draw_rate_eval"] = (
        float(draw_rate_eval) if draw_rate_eval is not None else view.draw_rate()
    )

    if ratings is not None:
        # Both sides of the difference have to be in the fit. The learner is not: every eval game
        # is a snapshot against something, so ``rating.get(learner_id, 0.0)`` returned the default
        # and the key published MINUS the first snapshot's rating -- read as -93.9, -146.2,
        # -191.7 and -129.6 on one run, which looks like a learner falling behind its own opening
        # snapshot and is nothing of the kind.
        #
        # The probe does NOT bring it back, and that is a choice rather than an oversight. A
        # probe names the live policy ``learner@{step}``, a player that exists for one moment,
        # and its games are ``kind="probe"`` and outside the fit; putting them in would grow the
        # fit by a column per probe, each too thinly played to place, and would let the rating
        # scale move because the run measured itself. What this key wants instead is a rated
        # player: the difference between the newest snapshot and v0 is that number, one gate
        # cadence stale, and it is not this key. Until it is decided, the row carries the probe's
        # score against the fixed rungs, which is a measurement of the live policy that needs no
        # fit at all.
        learner_rating = ratings.rating.get(learner_id)
        v0_rating = ratings.rating.get(pool.v0 or "")
        if learner_rating is not None and v0_rating is not None:
            fields["ladder/rating_above_v0"] = learner_rating - v0_rating
        fields["ladder/transitivity_residual"] = ratings.transitivity_residual
        named = set(members if members is not None else pool.sampler()) | {learner_id}
        for member in sorted(named):
            rating = ratings.rating.get(member)
            if rating is None:
                continue
            se = ratings.se.get(member, 0.0)
            fields[f"ladder/rating/{member}"] = rating
            fields[f"ladder/rating_se/{member}"] = se
            fields[f"ladder/rating_ci95_lo/{member}"] = rating - Z95 * se
            fields[f"ladder/rating_ci95_hi/{member}"] = rating + Z95 * se

    if decision is not None:
        from ..ladder.gate import CONDITION_CHAMPION

        champion_condition = decision.conditions.get(CONDITION_CHAMPION)
        if champion_condition is not None:
            fields["ladder/gate_observed_rate"] = champion_condition.observed
            fields["ladder/gate_lower_bound"] = champion_condition.bound
        fields["ladder/gate_failed_condition"] = failed_condition(decision)
    else:
        # The only one of the three that keeps a value without a gate, because "no gate has run"
        # is a state of the run rather than a missing measurement, and a reader of the column
        # wants to see where the gates begin.
        fields["ladder/gate_failed_condition"] = "none"
    return fields


class IterationMetrics(msgspec.Struct):
    """The pieces of one row, held until they are all in.

    A struct rather than a dict being filled in place, so that a piece nobody supplied is
    visibly absent instead of silently zero, and so that the loop can hand the row to an alarm
    and to a sink knowing the two saw the same numbers.
    """

    iteration: int
    run: dict[str, MetricValue] = msgspec.field(default_factory=dict)
    throughput: dict[str, MetricValue] = msgspec.field(default_factory=dict)
    time: dict[str, MetricValue] = msgspec.field(default_factory=dict)
    ppo: dict[str, MetricValue] = msgspec.field(default_factory=dict)
    policy: dict[str, MetricValue] = msgspec.field(default_factory=dict)
    env: dict[str, MetricValue] = msgspec.field(default_factory=dict)
    ladder: dict[str, MetricValue] = msgspec.field(default_factory=dict)
    health: dict[str, MetricValue] = msgspec.field(default_factory=dict)

    def row(self) -> dict[str, MetricValue]:
        """The flat row, with ``run/iteration`` always present."""
        row: dict[str, MetricValue] = {"run/iteration": self.iteration}
        for group in (
            self.run,
            self.throughput,
            self.time,
            self.ppo,
            self.policy,
            self.env,
            self.ladder,
            self.health,
        ):
            row.update(group)
        return row
