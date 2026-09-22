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
    from ..ladder.pool import LadderPool

__all__ = [
    "EpisodeAggregate",
    "IterationMetrics",
    "episode_fields",
    "flatten",
    "ladder_fields",
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
    # blend moves as the mixture and the pool move. `random_legal` plays a card whenever it can,
    # so it would hold `cards_per_match` up while the thing the no-op-collapse alarm watches for
    # was happening to the learner. A mirror battle has the learner in both seats and both count.
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
        "ppo/kl": result.kl,
        "ppo/clip_fraction": result.clip_fraction,
        "ppo/dual_clip_fraction": result.dual_clip_fraction,
        "ppo/explained_variance": result.explained_variance,
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
) -> dict[str, MetricValue]:
    """The ``ladder/`` group: the fit, the pool's shape and the last gate.

    The per-member ratings are written for the members named, which is the sampler rather than
    the archive: an archive of five hundred would be five hundred columns a run after run, and
    every one of them is still in the result log where the fit can be redone.
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
    view = pool.eval_view()
    # Emitted only when the pair was actually played. ``PairRecord.score_a`` answers 0.5 for a
    # pair with no games (results.py), and 0.5 is also what a genuine draw looks like -- so an
    # unplayed pair published as a number is indistinguishable from an even contest. Measured
    # 2026-09-22: exactly 0.5 in all 124 metric rows on disk, including one run that had played
    # 24 real evaluation battles, because the learner is never evaluated under its own id. Every
    # eval game is snapshot-against-something (the gate hands ``snap:v{n}`` to the runner), so
    # the ("learner", anchor) pair the reader asks for is one the evaluator cannot write.
    #
    # Keying the lookup to the latest snapshot instead is TEMPTING AND WRONG: it would report
    # the most recently gated candidate rather than the live policy, and those diverge by the
    # gate cadence -- 4,000,000 env steps on the shipped laptop profile. A number that silently
    # lags the policy by that much is worse than an absent one, because the absence is visible.
    # Making these real needs the live learner evaluated against the anchors, which is a new
    # pairing and a cost per iteration rather than a rename; until that is decided, the honest
    # row carries no score at all. ``alarms.py`` already guarantees a missing key never fires.
    for key, anchor in (
        ("ladder/score_vs_noop", SCRIPTED_NOOP),
        ("ladder/score_vs_random_legal", SCRIPTED_RANDOM_LEGAL),
    ):
        pair = view.record(learner_id, anchor)
        if pair.games:
            fields[key] = pair.score_a
    fields["ladder/draw_rate_eval"] = (
        float(draw_rate_eval) if draw_rate_eval is not None else view.draw_rate()
    )

    if ratings is not None:
        # Both sides of the difference have to be in the fit. The learner is not: every eval game
        # is a snapshot against something, so ``rating.get(learner_id, 0.0)`` returned the default
        # and the key published MINUS the first snapshot's rating -- read as -93.9, -146.2,
        # -191.7 and -129.6 on one run, which looks like a learner falling behind its own opening
        # snapshot and is nothing of the kind. It comes back when the live learner is evaluated
        # under its own id.
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
