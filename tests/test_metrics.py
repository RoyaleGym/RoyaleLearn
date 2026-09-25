"""The schema and the code are asserted against each other in both directions.

Every key a row carries has to be in ``metrics/schema.py``, and every key in the schema has to
be produced by something -- either by one of the functions here, or by a part of the harness
this file names. Without the second direction a metric can be documented and never emitted,
which is the failure that makes a dashboard quietly wrong rather than loudly broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

import msgspec
import pytest
from msgspec.structs import replace

from royalelearn.api.ladder import ConditionResult, GateDecision, RatingTable
from royalelearn.api.metrics import MetricsSink
from royalelearn.api.rollout import EpisodeRecord
from royalelearn.api.schedule import ScheduleState
from royalelearn.api.update import UpdateResult
from royalelearn.identity import EngineBuild, RunIdentity
from royalelearn.ladder.evaluate import Comparison
from royalelearn.ladder.pool import SCRIPTED_NOOP, SCRIPTED_RANDOM_LEGAL, LadderPool
from royalelearn.ladder.results import GameResult, ResultLog
from royalelearn.learn.inference import RoundStats
from royalelearn.metrics import schema
from royalelearn.metrics.records import (
    IterationMetrics,
    episode_fields,
    flatten,
    ladder_fields,
    rollout_policy_fields,
    schedule_fields,
    unknown_keys,
    update_fields,
)
from royalelearn.metrics.sinks import (
    ALARMS_NAME,
    EPISODES_NAME,
    METRICS_NAME,
    CompositeSink,
    ConsoleSink,
    JsonlSink,
    build_sinks,
)
from royalelearn.metrics.wandb_sink import WandbSink

#: Keys this file does not produce, with what does. The rollout workers reduce their own
#: per-step counters and the coordinator times its own phases; naming them here is what keeps
#: "every schema key is emitted" an assertion rather than a hope.
PRODUCED_ELSEWHERE: dict[str, str] = {
    # the coordinator's own bookkeeping
    "run/cumulative_updates": "coordinator",
    "run/wall_seconds": "coordinator",
    "run/determinism_tier": "coordinator",
    "run/resumed_with_drift": "coordinator",
    "run/state_digest": "coordinator",
    # the rollout farm's timing and the parent's own clock
    **{key: "coordinator" for key in schema.METRICS if key.startswith("throughput/")},
    **{key: "coordinator" for key in schema.METRICS if key.startswith("time/")},
    **{key: "coordinator" for key in schema.METRICS if key.startswith("health/")},
    # per-step counters, reduced in the worker where they are produced
    "policy/noop_rate": "rollout worker",
    "policy/legal_actions_mean": "rollout worker",
    "policy/legal_actions_p05": "rollout worker",
    "policy/legal_actions_p50": "rollout worker",
    "policy/legal_actions_p95": "rollout worker",
    "policy/forced_noop_frac": "rollout worker",
    "policy/tile_entropy": "rollout worker",
    "policy/tile_top1_share": "rollout worker",
    "policy/card_tile_top10_share": "rollout worker",
    "env/mean_elixir_at_decision": "rollout worker",
    "env/frac_elixir_above_99": "rollout worker",
    # the advantage estimator's own report
    "ppo/advantage_std_pre_norm": "advantage estimator",
    # section 19.5: the freeze's bookkeeping, on a run that schedules the actor's rate
    **dict.fromkeys(
        (
            "ppo/actor_lr_scale",
            "ppo/actor_frozen",
            "ppo/ev_at_unfreeze",
            "ppo/iterations_since_unfreeze",
        ),
        "the freeze",
    ),
    # the update's own statistics of the cells that had a choice, taken where the population is
    "ppo/advantage_std_choice_pre_norm": "the update",
    "ppo/advantage_mean_choice": "the update",
    "ppo/return_running_mean": "advantage estimator",
    "ppo/return_running_std": "advantage estimator",
    "ppo/reward_clip_frac": "advantage estimator",
}

IDENTITY = RunIdentity(
    format_version=1,
    royalelearn_version="0.1.0",
    royalelearn_git="unknown",
    royalegym_version="0.1.0",
    royalegym_git="unknown",
    engine_build=EngineBuild(
        engine_class="royalegym.mock_engine.MockEngine",
        calibration_digest="c",
        build_digest="b",
        catalogue_sha256="a",
        path_search=None,
        stale_build_differences=[],
    ),
    env_spec_digest="e",
    obs_digest="o",
    action_digest="d",
    frame_stack=1,
    arch_digest="r",
    codec_version=1,
    codec_table_digest="t",
    algo_digest="g",
    rollout_digest="l",
    ladder_digest="p",
    master_seed=7,
    determinism_tier="run_exact",
    torch_version="none",
    device_kind="cpu",
)


#: The outcome a seat reports, as the environment writes it: its own sign, not a code. The
#: fixtures below are built on it so that what this file asserts is the contract
#: ``tests/test_rollout_inline.py`` states, rather than whatever the arithmetic happens to read.
WON, LOST, DREW = 1, -1, 0


def _record(battle: int, seat: int, outcome: int, steps: int = 400) -> EpisodeRecord:
    return EpisodeRecord(
        slot=battle * 2 + seat,
        worker=0,
        shard=0,
        battle=battle,
        seat=seat,
        ordinal=0,
        episode_seed_path="env/worker/0/shard/0/gen/0",
        policy_id="learner",
        opponent_id="snap:v0",
        bucket="pool",
        episode_steps=steps,
        episode_ticks=steps * 5,
        own_crowns=1 if outcome > 0 else 0,
        enemy_crowns=1 if outcome < 0 else 0,
        own_tower_hp_frac=0.8,
        enemy_tower_hp_frac=0.6,
        elixir_leak_steps=steps // 10,
        elixir_count_exact=True,
        winner=-1 if outcome == 0 else (seat if outcome > 0 else 1 - seat),
        outcome=outcome,
        cards_played=22,
        illegal_commands=0,
        undiscounted_return=1.0,
        reward_terms={"terminal": 1.0, "tower_damage": 0.25, "elixir": -0.1},
        # louder than the sums, as a potential term is: its steps cancel in the sum
        reward_terms_step_abs={"terminal": 1.0, "tower_damage": 0.9, "elixir": 1.3},
    )


def _episodes() -> list[EpisodeRecord]:
    return [
        _record(0, 0, WON),
        _record(0, 1, LOST),
        _record(1, 0, LOST, steps=600),
        _record(2, 0, DREW, steps=600),
        _record(3, 1, WON, steps=300),
    ]


def _update() -> UpdateResult:
    return UpdateResult(
        policy_loss=-0.02,
        value_loss=0.31,
        entropy=3.4,
        noop_entropy=0.21,
        entropy_normalised=0.55,
        logit_std=0.42,
        kl=0.008,
        clip_fraction=0.12,
        dual_clip_fraction=0.001,
        explained_variance=0.72,
        ratio_max_abs_dev=1e-7,
        adam_eps_floor_frac_actor=0.992,
        adam_eps_floor_frac_critic=0.337,
        grad_norm_actor=0.3,
        grad_norm_critic=0.28,
        update_magnitude_actor=0.004,
        update_magnitude_critic=0.006,
        kl_by_epoch=[0.004, 0.011],
        clip_fraction_by_epoch=[0.09, 0.15],
        n_minibatches=96,
        n_optimizer_steps=24,
        n_samples=50_000,
        samples_unused_frac=0.0,
        seconds=12.5,
    )


def _decision() -> GateDecision:
    """One gate's audition, with the champion condition the row's two gate numbers come from."""
    from royalelearn.ladder.gate import CONDITION_CHAMPION

    return GateDecision(
        candidate="snap:v1",
        champion="snap:v0",
        admit=True,
        promote=True,
        cycle=False,
        conditions={
            CONDITION_CHAMPION: ConditionResult(
                passed=True, n=1000, observed=0.56, bound=0.52, reference=0.5
            )
        },
        eval_seed_set_sha="0" * 8,
        wall_seconds=12.0,
    )


def _pool(tmp_path) -> LadderPool:
    pool = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    pool.add("snap:v0", step=0)
    pool.promote("snap:v0")
    pool.record(
        [
            GameResult(
                a="learner",
                b=opponent,
                score_a=1.0,
                seed_index=index,
                side_a="blue",
                context="ctx",
                kind="eval",
                run_id="run",
                iteration=1,
                wall="",
            )
            for opponent in ("snap:v0", SCRIPTED_NOOP, SCRIPTED_RANDOM_LEGAL)
            for index in range(40)
        ]
    )
    pool.note_refit(
        RatingTable(
            rating={"learner": 180.0, "snap:v0": 60.0, SCRIPTED_NOOP: 0.0},
            se={"learner": 21.7, "snap:v0": 18.0, SCRIPTED_NOOP: 0.0},
            anchor=SCRIPTED_NOOP,
            draw_nu=0.3,
            n_games={"learner": 120, "snap:v0": 40, SCRIPTED_NOOP: 40},
            transitivity_residual=0.02,
            converged=True,
            iterations=5,
        )
    )
    return pool


def _probe() -> dict[str, Comparison]:
    """One probe of the live policy: a comparison per scripted rung.

    The scores in a row come from here and not from the result log. The log holds games between
    frozen snapshots, so the pair a reader asks about -- the live policy against an anchor -- is
    one no evaluator ever wrote, and reading it answered 0.5 for "never played".
    """
    return {
        SCRIPTED_NOOP: Comparison(
            a="learner@4000000",
            b=SCRIPTED_NOOP,
            n_seeds=20,
            n_games=40,
            score_a=0.975,
            lo=0.93,
            hi=1.0,
            rho=0.12,
            draw_rate=0.05,
            seed_scores=(),
        ),
        SCRIPTED_RANDOM_LEGAL: Comparison(
            a="learner@4000000",
            b=SCRIPTED_RANDOM_LEGAL,
            n_seeds=20,
            n_games=40,
            score_a=0.7,
            lo=0.58,
            hi=0.81,
            rho=0.2,
            draw_rate=0.1,
            seed_scores=(),
        ),
    }


def _rollout_stats() -> RoundStats:
    """One iteration of rollout forwards: 100 learner rows, 10 of which had a choice.

    The numbers are a policy holding at 15.3x uniform over 250 legal actions, which is what
    hog26-2 was doing at iteration 124.
    """
    return RoundStats(
        rows=100,
        forwards=4,
        rounds=4,
        seconds=1.5,
        entropy=540.0,
        p_noop=90.6,
        n_legal=2590.0,
        choice_rows=10,
        hold=0.612,
        hold_lift=153.0,
        choice_n_legal=2500.0,
        gap=21.0,
        gap_sq=45.0,
        legal_log=55.3,
        legal_log_sq=306.0,
        gap_legal_log=116.2,
    )


def _row(tmp_path) -> dict:
    pool = _pool(tmp_path)
    state = ScheduleState(
        iteration=12,
        cumulative_env_steps=4_000_000,
        cumulative_timesteps=3_000_000,
        gamma=0.997,
        gae_lambda=0.95,
        ent_coef=0.01,
        ent_coef_noop=0.004,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_backoff_events=1,
    )
    metrics = IterationMetrics(
        iteration=12,
        run=schedule_fields(state, decision_ms=500),
        ppo=update_fields(_update()),
        env=episode_fields(_episodes(), truncation_steps=600).fields,
        ladder=ladder_fields(
            pool,
            ratings=pool.ratings,
            elo=1240.0,
            paired_rho=0.31,
            gate_seconds_frac=0.04,
            decision=_decision(),
            rungs=_probe(),
            probe_seconds_frac=0.02,
        ),
        policy=rollout_policy_fields(_rollout_stats()),
    )
    return metrics.row()


def test_the_parents_peak_memory_is_a_number_or_nothing_but_never_zero() -> None:
    """It read 0.0 on this machine for the life of the project, and 0.0 is a planner's input.

    The Windows path asks ``GetCurrentProcess`` for a handle, and that function returns the
    pseudo-handle 0xFFFFFFFFFFFFFFFF. ctypes assumes a C ``int`` return for a function nobody has
    declared, so the handle arrived truncated to 32 bits, the call failed, and the failure became
    a zero. The RAM ledger is what ``doctor`` projects a run from.
    """
    from royalelearn.coordinator import _rss_peak_mb

    peak = _rss_peak_mb()
    assert peak is None or peak > 0.0, peak
    if sys.platform == "win32":
        assert peak is not None and peak > 1.0, "a live interpreter holds more than a megabyte"


def test_a_number_nobody_could_read_is_left_out_of_the_row() -> None:
    """``throughput/gpu_util_frac`` read 0.0 on a card at 92%, because the call that reads it
    raises without pynvml and every failure path returned zero. An idle GPU is what a reader
    hunting a bottleneck would have concluded."""
    from royalelearn.coordinator import _optional

    assert _optional("throughput/gpu_util_frac", None) == {}
    assert _optional("throughput/gpu_util_frac", 0.0) == {"throughput/gpu_util_frac": 0.0}
    assert _optional("throughput/gpu_util_frac", 0.92) == {"throughput/gpu_util_frac": 0.92}


def test_the_update_publishes_what_its_two_long_phases_cost() -> None:
    """``time/critic_pass`` and ``time/gae`` were the literal 0.0, not a measurement.

    The update is most of an iteration on this machine, and those are the two phases inside it
    that run before a single epoch does. Published as zero they said the update was entirely
    epochs, so a reader deciding where to spend an optimisation had the one number that would
    have redirected them reading as nothing.
    """
    result = _update()
    assert result.critic_pass_seconds >= 0.0
    assert result.gae_seconds >= 0.0
    fields = update_fields(
        msgspec.structs.replace(result, critic_pass_seconds=1.25, gae_seconds=0.5)
    )
    assert "ppo/critic_pass_seconds" not in fields, "it belongs in the time group"


def test_the_ladder_group_leaves_out_what_it_has_not_measured(tmp_path) -> None:
    """Every one of these had a neutral value, and every neutral value is a claim.

    An Elo of zero, seats that are uncorrelated, a gate that cost nothing, a rating that is
    perfectly transitive, a learner exactly as good as the first snapshot. A reader cannot tell
    one of those from a measurement, and a plot draws a flat line through the part of the run
    where nothing was computed. One of them was worse than flat: the learner is never in the fit,
    so ``rating_above_v0`` published minus the first snapshot's rating, which one run read as
    -93.9 falling to -191.7 and which looks exactly like a policy losing to its own opening
    snapshot.
    """
    pool = _pool(tmp_path)
    bare = ladder_fields(pool)
    for key in (
        "ladder/elo_readout",
        "ladder/paired_rho",
        "ladder/gate_seconds_frac",
        "ladder/transitivity_residual",
        "ladder/rating_above_v0",
        "ladder/gate_observed_rate",
        "ladder/gate_lower_bound",
        "ladder/score_vs_noop",
        "ladder/score_vs_random_legal",
        "ladder/probe_seconds_frac",
    ):
        assert key not in bare, key
    # The pool's own shape is not conditional: it is known from the moment the run starts.
    assert bare["ladder/pool_size"] == len(pool.members())

    fitted = ladder_fields(pool, ratings=pool.ratings, elo=1240.0, paired_rho=0.31)
    assert fitted["ladder/elo_readout"] == pytest.approx(1240.0)
    assert fitted["ladder/paired_rho"] == pytest.approx(0.31)
    assert "ladder/transitivity_residual" in fitted
    # This fixture's log holds games the learner played, so the fit knows it and the difference
    # is real. A run's does not: the gate hands snap:v{n} to the evaluator, so no game is ever
    # keyed to the learner.
    assert "ladder/rating_above_v0" in fitted

    without_learner = msgspec.structs.replace(
        pool.ratings,
        rating={member: value for member, value in pool.ratings.rating.items()
                if member != "learner"},
    )
    thin = ladder_fields(pool, ratings=without_learner)
    assert "ladder/rating_above_v0" not in thin
    assert "ladder/transitivity_residual" in thin


# -- the schema --------------------------------------------------------------


def test_every_key_a_row_carries_is_in_the_schema(tmp_path) -> None:
    assert unknown_keys(_row(tmp_path)) == ()


def test_every_schema_key_has_something_that_emits_it(tmp_path) -> None:
    produced = set(_row(tmp_path))
    documented = set(schema.METRICS)
    assert documented - produced - set(PRODUCED_ELSEWHERE) == set()
    assert set(PRODUCED_ELSEWHERE) <= documented
    # And nothing is claimed by both sides.
    assert produced & set(PRODUCED_ELSEWHERE) == set()


def test_the_patterned_families_are_matched(tmp_path) -> None:
    row = _row(tmp_path)
    assert schema.is_known("ppo/kl_epoch0")
    assert "ppo/kl_epoch1" in row
    assert "env/reward_terms/tower_damage" in row
    assert "ladder/rating/snap:v0" in row
    assert "ladder/rating_ci95_lo/learner" in row
    assert "ladder/score_vs/random_legal" in row
    assert "ladder/score_vs_ci95_hi/noop" in row


# -- the arithmetic ----------------------------------------------------------


def test_episode_statistics_count_battles_once_and_seats_once(tmp_path) -> None:
    aggregate = episode_fields(_episodes(), truncation_steps=600)
    assert aggregate.seats == 5
    assert aggregate.battles == 4  # battle 0 reported by both of its seats
    fields = aggregate.fields
    assert fields["env/episodes_completed"] == 4
    assert fields["env/draw_rate"] == pytest.approx(0.25)
    # Blue won battle 0; battle 1 was a loss for blue and battle 3 a win for red.
    assert fields["env/win_rate_by_seat"] == pytest.approx(1 / 3)
    assert fields["env/episode_steps_at_cap_frac"] == pytest.approx(0.5)
    assert fields["policy/cards_per_match"] == pytest.approx(22.0)
    assert fields["env/elixir_leak_frac"] == pytest.approx(0.1)
    assert fields["env/illegal_action_rate"] == 0.0
    assert fields["env/reward_terminal_abs"] == pytest.approx(1.0)
    assert fields["env/reward_shaping_abs"] == pytest.approx(0.35)
    assert msgspec.json.decode(fields["env/episode_steps_hist"])["counts"]


def test_the_reward_shares_survive_a_zero_sum_reward(tmp_path) -> None:
    """Both seats of a battle report the same reward negated, and the shares must survive it.

    Every shipped reward is zero sum: what one seat gains the other loses, so a term's mean over
    a batch that holds both seats is zero whatever the term did. A share computed as the absolute
    value of that mean is therefore zero for every run, and ``shaping_dominates`` -- the alarm
    that watches for the shaping terms taking over the objective -- can never fire. The shares
    are magnitudes, so the absolute value belongs on each seat's own number.
    """
    terms = {"terminal": 1.0, "tower_damage": 0.25, "elixir": -0.1}
    blue = replace(_record(0, 0, WON), reward_terms=terms)
    red = replace(_record(0, 1, LOST), reward_terms={k: -v for k, v in terms.items()})
    fields = episode_fields([blue, red]).fields
    assert fields["env/reward_terminal_abs"] == pytest.approx(1.0)
    assert fields["env/reward_shaping_abs"] == pytest.approx(0.35)
    # The signed mean is what it is -- zero, by antisymmetry -- and it stays published as that.
    assert fields["env/reward_terms/tower_damage"] == pytest.approx(0.0)
    assert fields["env/reward_terms_abs/tower_damage"] == pytest.approx(0.25)


def test_the_objective_is_found_when_a_reward_names_its_terms_after_their_classes() -> None:
    """A composition assembled from RoyaleGym names every term after its class.

    The shipped configs did exactly that until 23d971c, and a bot creator's own composition may.
    The objective then arrives as ``WinLossReward`` rather than under the declared name, and until
    2026-09-22 it was counted as shaping: a real run read shaping 1.475 against terminal 0.0, and
    the difference was exactly the 1.0 the objective contributed. ``shaping_dominates`` then holds
    on every row of every run.
    """
    terms = {"WinLossReward": 1.0, "TowerHPReward": 0.1, "ElixirTradeReward": -0.02}
    blue = replace(_record(0, 0, WON), reward_terms=terms)
    red = replace(_record(0, 1, LOST), reward_terms={k: -v for k, v in terms.items()})
    fields = episode_fields([blue, red]).fields
    assert fields["env/reward_terminal_abs"] == pytest.approx(1.0)
    assert fields["env/reward_shaping_abs"] == pytest.approx(0.12)


def test_an_unrecognised_objective_is_absent_rather_than_zero() -> None:
    """A reward whose objective this module cannot name leaves the share out of the row.

    Zero would be a claim -- that the objective contributed nothing -- and ``shaping_dominates``
    compares the two, so a fabricated zero makes it hold forever. Absent is the honest answer, and
    the alarm's missing-key rule then keeps it silent until somebody teaches the module the name.
    """
    terms = {"SomeonesOwnObjective": 1.0, "TheirShaping": 0.3}
    blue = replace(_record(0, 0, WON), reward_terms=terms)
    red = replace(_record(0, 1, LOST), reward_terms={k: -v for k, v in terms.items()})
    fields = episode_fields([blue, red]).fields
    assert "env/reward_terminal_abs" not in fields
    assert fields["env/reward_shaping_abs"] == pytest.approx(1.3)


def test_the_seat_numbers_are_the_learners_and_not_its_opponents() -> None:
    """A battle that is not a mirror has somebody else in its other seat.

    Averaging both seats makes every one of these a blend of the learner and whatever it was drawn
    against, and the blend moves with the mixture and the pool rather than with the policy.
    ``random_legal`` plays a card whenever it can afford one, so it would hold ``cards_per_match``
    up while the collapse its alarm watches for was happening to the learner.
    """
    learner = replace(
        _record(0, 0, WON, steps=200),
        cards_played=4,
        own_crowns=1,
        enemy_crowns=0,
        elixir_leak_steps=0,
    )
    opponent = replace(
        _record(0, 1, LOST, steps=200),
        policy_id="scripted:random_legal",
        opponent_id="learner",
        cards_played=40,
        own_crowns=0,
        enemy_crowns=1,
        elixir_leak_steps=100,
    )
    fields = episode_fields([learner, opponent]).fields
    assert fields["policy/cards_per_match"] == pytest.approx(4.0)
    assert fields["policy/cards_per_100_decisions"] == pytest.approx(2.0)
    assert fields["env/crowns_for"] == pytest.approx(1.0)
    assert fields["env/crowns_against"] == pytest.approx(0.0)
    assert fields["env/elixir_leak_frac"] == pytest.approx(0.0)
    # A mirror battle is the other case: both seats are the learner and both count.
    mirror = [
        replace(_record(1, 0, WON, steps=200), cards_played=4),
        replace(_record(1, 1, LOST, steps=200), cards_played=10),
    ]
    assert episode_fields(mirror).fields["policy/cards_per_match"] == pytest.approx(7.0)


def test_an_iteration_of_only_opponent_episodes_says_nothing_about_the_learner() -> None:
    opponent = replace(_record(0, 1, LOST), policy_id="scripted:noop", opponent_id="snap:v0")
    fields = episode_fields([opponent]).fields
    assert fields == {"env/episodes_completed": 0}


def test_cards_per_match_is_a_count_and_its_companion_is_a_rate() -> None:
    """The count rises with episode length; the rate is what two iterations compare on.

    Measured on a real run: iteration 1 finished 28 episodes averaging 171 steps and scored 6.98
    cards a match, iteration 2 finished 82 averaging 328 steps and scored 22.08. Read as a policy
    improving, it was the episodes getting longer. Per decision the same two rows are 4.1 and 6.7
    per hundred.
    """
    short = replace(_record(0, 0, WON, steps=100), cards_played=5)
    long_one = replace(_record(1, 0, WON, steps=400), cards_played=20)
    fields = episode_fields([short, long_one]).fields
    assert fields["policy/cards_per_match"] == pytest.approx(12.5)
    assert fields["policy/cards_per_100_decisions"] == pytest.approx(100 * 25 / 500)


def test_a_truncated_episode_is_a_draw_and_reds_view_is_blues_negated() -> None:
    """Both are the default case rather than an edge one.

    Training truncates on the step limit, and a truncated episode has no winner, so under the
    shipped environment most episodes carry a zero. And a battle whose blue seat an opponent
    played is reported only by red, whose sign is the opposite of the one the metric is over.
    """
    drawn = episode_fields([_record(battle, 0, DREW) for battle in range(4)])
    assert drawn.fields["env/draw_rate"] == pytest.approx(1.0)

    red_only = episode_fields([_record(battle, 1, LOST) for battle in range(4)])
    assert red_only.fields["env/draw_rate"] == 0.0
    assert red_only.fields["env/win_rate_by_seat"] == pytest.approx(1.0)


def test_an_iteration_with_no_episodes_still_says_so() -> None:
    aggregate = episode_fields([])
    assert aggregate.fields == {"env/episodes_completed": 0}


def test_the_ladder_fields_read_the_fit_and_the_log(tmp_path) -> None:
    pool = _pool(tmp_path)
    fields = ladder_fields(pool, ratings=pool.ratings, elo=1240.0)
    assert fields["ladder/rating_above_v0"] == pytest.approx(120.0)
    assert fields["ladder/champion_id"] == "snap:v0"
    assert "ladder/score_vs_noop" not in fields, (
        "the anchor scores come from a probe of the live policy, never from the log: the log's "
        "games are all snapshot-against-something"
    )
    probed = ladder_fields(pool, ratings=pool.ratings, rungs=_probe(), probe_seconds_frac=0.02)
    assert probed["ladder/score_vs_noop"] == pytest.approx(0.975)
    assert probed["ladder/score_vs_random_legal"] == pytest.approx(0.7)
    assert probed["ladder/score_vs_n/random_legal"] == 20
    assert probed["ladder/score_vs_ci95_lo/random_legal"] == pytest.approx(0.58)
    assert probed["ladder/probe_seconds_frac"] == pytest.approx(0.02)
    assert fields["ladder/transitivity_residual"] == pytest.approx(0.02)
    assert fields["ladder/rating_ci95_hi/learner"] > fields["ladder/rating/learner"]
    assert fields["ladder/gate_failed_condition"] == "none"


def test_nested_keys_flatten_to_one_level() -> None:
    assert flatten({"ppo": {"kl": 0.01, "deep": {"er": 1}}, "run/iteration": 3}) == {
        "ppo/kl": 0.01,
        "ppo/deep/er": 1,
        "run/iteration": 3,
    }


# -- the sinks ---------------------------------------------------------------


def test_settings_beside_the_sink_list_reach_the_sink_the_list_names() -> None:
    """Some of a sink's settings live on the config beside the list rather than inside a spec.

    ``keep_episode_log_iterations`` is one: it is a field of ``MetricsConfig``, and the shipped
    config names a jsonl sink with no options at all, so a default run would compact nothing if
    the two were only joined on the path where the list does not mention jsonl.
    """
    from royalelearn.config import SinkSpec

    composite = build_sinks(
        [SinkSpec("jsonl")], jsonl={"keep_episode_log_iterations": 7}
    )
    jsonl = [sink for sink in composite.sinks if isinstance(sink, JsonlSink)]
    assert len(jsonl) == 1
    assert jsonl[0].keep_episode_log_iterations == 7

    # And a spec that names the key itself wins over the default the caller passed.
    named = build_sinks(
        [SinkSpec("jsonl", options={"keep_episode_log_iterations": 3})],
        jsonl={"keep_episode_log_iterations": 7},
    )
    assert next(iter(named.sinks)).keep_episode_log_iterations == 3


def test_the_jsonl_sink_writes_the_run_and_its_rows(tmp_path) -> None:
    run_dir = tmp_path / "run"
    sink = JsonlSink()
    sink.open(identity=IDENTITY, config_json='{"run_name": "test"}', run_dir=run_dir)
    row = _row(tmp_path)
    sink.write(row)
    sink.write_episodes(_episodes())
    sink.close()
    assert msgspec.json.decode((run_dir / "identity.json").read_bytes())["master_seed"] == 7
    assert (run_dir / "config.json").read_text(encoding="utf-8") == '{"run_name": "test"}'
    lines = (run_dir / METRICS_NAME).read_bytes().splitlines()
    assert len(lines) == 1
    assert msgspec.json.decode(lines[0]) == row
    assert len((run_dir / EPISODES_NAME).read_bytes().splitlines()) == len(_episodes())
    assert not (run_dir / ALARMS_NAME).exists()


def test_the_console_sink_prints_a_key_nobody_documented(tmp_path, capsys) -> None:
    sink = ConsoleSink()
    sink.open(identity=IDENTITY, config_json="{}", run_dir=tmp_path)
    sink.write({"run/iteration": 4, "brand/new": 1.5, "ppo/kl": 0.004})
    printed = capsys.readouterr().out
    assert "iteration 4" in printed
    assert "new" in printed and "1.5" in printed
    assert "  ppo" in printed
    sink.close()


def test_the_wandb_sink_is_a_pure_pass_through_when_disabled(tmp_path) -> None:
    class Recording(MetricsSink):
        FORMAT_VERSION = 1

        def __init__(self) -> None:
            self.rows: list[dict] = []
            self.opened = False
            self.closed = False
            self.saved: Path | None = None

        def open(self, *, identity, config_json, run_dir) -> None:
            self.opened = True

        def write(self, row) -> None:
            self.rows.append(dict(row))

        def close(self) -> None:
            self.closed = True

        def save_checkpoint(self, folder: Path) -> None:
            self.saved = folder
            folder.mkdir(parents=True, exist_ok=True)

        def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
            self.saved = folder

    inner = Recording()
    sink = WandbSink(inner, enable=False)
    sink.open(identity=IDENTITY, config_json="{}", run_dir=tmp_path)
    row = _row(tmp_path)
    sink.write(row)
    sink.close()
    assert inner.opened and inner.closed
    assert inner.rows == [row]
    assert sink.run_id is None

    folder = tmp_path / "metrics"
    sink.save_checkpoint(folder)
    assert msgspec.json.decode((folder / "wandb.json").read_bytes()) == {"run_id": None}
    assert inner.saved == folder / "inner"
    sink.load_checkpoint(folder, strict=True)


def test_the_composite_fans_out_and_nests_its_checkpoints(tmp_path) -> None:
    composite = CompositeSink([JsonlSink(), ConsoleSink()])
    run_dir = tmp_path / "run"
    composite.open(identity=IDENTITY, config_json="{}", run_dir=run_dir)
    composite.write({"run/iteration": 1})
    composite.close()
    assert (run_dir / METRICS_NAME).exists()
    folder = tmp_path / "ck"
    composite.save_checkpoint(folder)
    assert (folder / "0-JsonlSink" / "jsonl.json").exists()
    composite.load_checkpoint(folder, strict=True)
