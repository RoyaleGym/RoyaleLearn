"""The published metric names, pinned.

A metric key is a consumer contract the moment anything outside this repository reads it: a
plot, a spreadsheet, a watcher beside a run, a panel that names a field. Renaming one is
sometimes right -- `worker_idle_frac` became `parent_wait_frac` because the old name prescribed
the opposite action to the one its number called for -- but it must never be something a change
does quietly. Every other test here passed through that rename without noticing.

So the set is written down, and a change to it fails with the difference named. Update the
lists below in the same commit that changes the schema, which is the point: it makes the
rename a decision rather than a side effect.
"""

from __future__ import annotations

from royalelearn.metrics import schema

PUBLISHED_KEYS = (
    'env/crown_diff',
    'env/crowns_against',
    'env/crowns_for',
    'env/draw_rate',
    'env/elixir_count_exact_frac',
    'env/elixir_leak_frac',
    'env/episode_steps_at_cap_frac',
    'env/episode_steps_hist',
    'env/episode_steps_mean',
    'env/episode_steps_p05',
    'env/episode_steps_p50',
    'env/episode_steps_p95',
    'env/episodes_completed',
    'env/frac_elixir_above_99',
    'env/illegal_action_rate',
    'env/mean_elixir_at_decision',
    'env/reward_shaping_abs',
    'env/reward_terminal_abs',
    'env/ticks_mean',
    'env/tower_hp_frac_end_enemy',
    'env/tower_hp_frac_end_own',
    'env/win_rate_by_seat',
    'env/win_rate_by_seat_ci95_hi',
    'env/win_rate_by_seat_ci95_lo',
    'health/buffer_fill_frac',
    'health/illegal_action_rate',
    'health/mask_disagreements',
    'health/nan_guard_trips',
    'health/obs_codec_clipped',
    'health/rows_dropped_dead_worker',
    'health/rss_peak_mb',
    'health/samples_unused_frac',
    'health/vram_alloc_retries',
    'health/vram_available_mb',
    'health/vram_driver_free_mb',
    'health/vram_inactive_split_mb',
    'health/vram_needed_mb',
    'health/vram_peak_mb',
    'health/vram_reserved_mb',
    'health/worker_restarts',
    'ladder/champion_id',
    'ladder/champion_step',
    'ladder/consecutive_gate_failures',
    'ladder/draw_rate_eval',
    'ladder/elo_readout',
    'ladder/eval_games_total',
    'ladder/evictions',
    'ladder/gate_attempts',
    'ladder/gate_failed_condition',
    'ladder/gate_lower_bound',
    'ladder/gate_observed_rate',
    'ladder/gate_passes',
    'ladder/gate_seconds_frac',
    'ladder/paired_rho',
    'ladder/pool_size',
    'ladder/probe_seconds_frac',
    'ladder/rating_above_v0',
    'ladder/sampler_size',
    'ladder/score_vs_noop',
    'ladder/score_vs_random_legal',
    'ladder/transitivity_residual',
    'policy/card_tile_top10_share',
    'policy/cards_per_100_decisions',
    'policy/cards_per_match',
    'policy/forced_noop_frac',
    'policy/legal_actions_mean',
    'policy/legal_actions_p05',
    'policy/legal_actions_p50',
    'policy/legal_actions_p95',
    'policy/noop_rate',
    'policy/rollout_choice_frac',
    'policy/rollout_hold_gap',
    'policy/rollout_hold_gap_residual_std',
    'policy/rollout_hold_gap_std',
    'policy/rollout_hold_lift',
    'policy/rollout_hold_rate',
    'policy/rollout_legal_actions',
    'policy/tile_entropy',
    'policy/tile_top1_share',
    'ppo/adam_eps_floor_frac_actor',
    'ppo/adam_eps_floor_frac_critic',
    'ppo/actor_forwards',
    'ppo/actor_rows',
    'ppo/advantage_mean_choice',
    'ppo/advantage_std_choice_pre_norm',
    'ppo/advantage_std_pre_norm',
    'ppo/clip_fraction',
    'ppo/dual_clip_fraction',
    'ppo/entropy',
    'ppo/entropy_normalised',
    'ppo/explained_variance',
    'ppo/explained_variance_choice',
    'ppo/forced_frac',
    'ppo/grad_norm_actor',
    'ppo/grad_norm_critic',
    'ppo/kl',
    'ppo/logit_std',
    'ppo/lr_backoff_events',
    'ppo/n_minibatches',
    'ppo/n_optimizer_steps',
    'ppo/noop_entropy',
    'ppo/policy_loss',
    'ppo/policy_loss_choice',
    'ppo/ratio_max_abs_dev',
    'ppo/return_running_mean',
    'ppo/return_running_std',
    'ppo/reward_clip_frac',
    'ppo/samples_unused_frac',
    'ppo/update_magnitude_actor',
    'ppo/update_magnitude_critic',
    'ppo/value_loss',
    'run/credit_horizon_seconds',
    'run/cumulative_env_steps',
    'run/cumulative_timesteps',
    'run/cumulative_updates',
    'run/determinism_tier',
    'run/ent_coef',
    'run/ent_coef_noop',
    'run/gae_lambda',
    'run/gamma',
    'run/iteration',
    'run/lr_actor',
    'run/lr_critic',
    'run/resumed_with_drift',
    'run/state_digest',
    'run/wall_seconds',
    'throughput/boundary_mb_per_second',
    'throughput/collected_steps_per_second',
    'throughput/discarded_rows_frac',
    'throughput/engine_ticks_per_second',
    'throughput/gpu_util_frac',
    'throughput/inference_ms_per_round',
    'throughput/overall_steps_per_second',
    'throughput/parent_wait_frac',
    'throughput/rollout_capacity_ratio',
    'time/checkpoint',
    'time/codec',
    'time/collection',
    'time/critic_pass',
    'time/env',
    'time/gae',
    'time/gate',
    'time/inference',
    'time/ipc',
    'time/iteration',
    'time/overlap_saved',
    'time/probe',
    'time/residual',
    'time/update',
)

PUBLISHED_PATTERNS = (
    'env/reward_terms/{term}',
    'env/reward_terms_abs/{term}',
    'health/worker_failures/{kind}',
    'ladder/rating/{member}',
    'ladder/rating_ci95_hi/{member}',
    'ladder/rating_ci95_lo/{member}',
    'ladder/rating_se/{member}',
    'ladder/score_vs/{opponent}',
    'ladder/score_vs_ci95_hi/{opponent}',
    'ladder/score_vs_ci95_lo/{opponent}',
    'ladder/score_vs_n/{opponent}',
    'policy/card_in_hand_frac/{card}',
    'policy/card_legal_frac/{card}',
    'policy/card_play_frac/{card}',
    'policy/card_play_rate/{card}',
    'ppo/clip_fraction_epoch{epoch}',
    'ppo/kl_epoch{epoch}',
)

def test_the_published_metric_keys_are_the_ones_written_down() -> None:
    live = set(schema.METRICS)
    pinned = set(PUBLISHED_KEYS)
    added, removed = sorted(live - pinned), sorted(pinned - live)
    assert not (added or removed), (
        "the published metric keys changed.\n"
        f"  added:   {added}\n"
        f"  removed: {removed}\n"
        "If that is intended, update PUBLISHED_KEYS in this file in the same commit, and if it "
        "is a rename rather than a new key, add it to schema.RENAMED so that rows already on "
        "disk still resolve. Anything reading a run's rows -- a plot, a watcher, the viewer's "
        "panel -- reads these names."
    )


def test_the_published_metric_families_are_the_ones_written_down() -> None:
    live = {pattern.template for pattern in schema.PATTERNS}
    pinned = set(PUBLISHED_PATTERNS)
    added, removed = sorted(live - pinned), sorted(pinned - live)
    assert not (added or removed), (
        f"the published metric families changed.\n  added:   {added}\n  removed: {removed}"
    )


def test_every_pinned_key_has_a_unit_and_a_description() -> None:
    """A name nobody can read is a name nobody can act on."""
    for key in PUBLISHED_KEYS:
        spec = schema.METRICS[key]
        assert spec.unit, f"{key} has no unit"
        assert spec.description.strip().endswith("."), f"{key}'s description is not a sentence"


def test_every_renamed_key_points_at_one_that_exists() -> None:
    """A rename map that names a key nobody publishes is worse than none at all."""
    for old, new in schema.RENAMED.items():
        assert new in schema.METRICS, f"{old} is mapped to {new}, which the schema does not have"
        assert old not in schema.METRICS, f"{old} is both published and marked as renamed"


def test_a_key_from_an_older_run_still_resolves() -> None:
    """The point of the map: rows written under the old name are still readable.

    A run's file carries whatever the name was when it ran, and nothing rewrites a file that is
    already on disk. So the schema answers for both, and a reader lining several runs up does
    not have to know the history.
    """
    for old, new in schema.RENAMED.items():
        assert schema.current_name(old) == new
        assert schema.is_known(old), f"{old} was published once and no longer resolves"
        assert schema.lookup(old) is schema.METRICS[new]
    assert schema.current_name("ppo/kl") == "ppo/kl", "a key that never moved must not move"


def test_every_key_the_viewer_panel_reads_is_a_key_the_schema_publishes() -> None:
    """The sink is the one consumer of these names that lives inside this repository.

    It maps a metric key to a row of the viewer's panel, and a key that no longer exists maps
    to nothing at all: the row renders as an em dash, which is what the panel also shows when
    no learner is attached. So a rename would take a number off the dashboard and look exactly
    like a learner that was never there.
    """
    from royalelearn.metrics import viser_sink

    sources = [key for _field, key in viser_sink.FIELD_SOURCES.items()]
    sources += [key for _name, key in viser_sink.EXTRA_SOURCES if key]
    unknown = sorted({key for key in sources if not schema.is_known(key)})
    assert not unknown, (
        f"the viewer panel reads {unknown}, which the schema does not publish. A panel row fed "
        f"by a key that does not exist is an em dash, and an em dash is what the panel shows "
        f"when no learner is attached at all."
    )
