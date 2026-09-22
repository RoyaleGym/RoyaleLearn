"""Every alarm, on a row that should fire it and on a row that should not.

The healthy row below is the important half. An alarm table is easy to write so that something
is always firing, and a run whose console prints a warning every iteration has no alarms at
all -- so the first test here asserts that a row in the middle of the healthy band fires
nothing, and every other test moves exactly one key out of that row.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from royalelearn.api.metrics import Alarm, MetricRow
from royalelearn.config import AlarmConfig
from royalelearn.errors import AlarmHalt
from royalelearn.metrics import schema
from royalelearn.metrics.alarms import HALT, WARN, AlarmSet, default_alarms
from royalelearn.metrics.bundle import write_bundle

#: A row in the middle of every healthy band the schema declares. Every test below starts from
#: this and moves one key.
HEALTHY: dict[str, float | int | str] = {
    "run/iteration": 1,
    "env/illegal_action_rate": 0.0,
    "ppo/ratio_max_abs_dev": 1e-7,
    "health/nan_guard_trips": 0,
    "health/buffer_fill_frac": 1.0,
    "health/worker_restarts": 0,
    "ppo/clip_fraction": 0.12,
    "ppo/kl": 0.008,
    "ppo/explained_variance": 0.6,
    "policy/cards_per_match": 22.0,
    "ppo/noop_entropy": 0.2,
    "policy/tile_top1_share": 0.05,
    "policy/card_tile_top10_share": 0.2,
    "env/draw_rate": 0.1,
    "env/episode_steps_at_cap_frac": 0.1,
    "env/win_rate_by_seat_ci95_lo": 0.48,
    "env/win_rate_by_seat_ci95_hi": 0.52,
    "env/elixir_count_exact_frac": 1.0,
    "env/reward_shaping_abs": 0.2,
    "env/reward_terminal_abs": 1.0,
    # What laptop runs at minibatch 256 on a 4 GB card report: about 3372 MB available against
    # a measured need of 2349.
    "health/vram_available_mb": 3400.0,
    "health/vram_needed_mb": 2350.0,
    "ladder/transitivity_residual": 0.02,
    "ladder/consecutive_gate_failures": 0,
    "throughput/rollout_capacity_ratio": 2.4,
}

#: What one key has to become for each alarm to hold. ``worker_failures`` is a rise rather than
#: a level, so it is two rows and is tested on its own below.
TRIPS: dict[str, dict[str, float]] = {
    "illegal_actions": {"env/illegal_action_rate": 1e-6},
    "ratio_invariant": {"ppo/ratio_max_abs_dev": 0.9},
    "nonfinite": {"health/nan_guard_trips": 1},
    "buffer_overflow": {"health/buffer_fill_frac": 1.5},
    "clip_pinned": {"ppo/clip_fraction": 0.8},
    "kl_high": {"ppo/kl": 0.2},
    "kl_dead": {"ppo/kl": 1e-9},
    "ev_negative": {"ppo/explained_variance": -0.4},
    "noop_collapse": {"policy/cards_per_match": 6.0},
    "noop_collapse_severe": {"policy/cards_per_match": 1.0},
    "noop_entropy_floor": {"ppo/noop_entropy": 0.001},
    "tile_spam": {"policy/tile_top1_share": 0.6},
    "artefact_exploit": {"policy/card_tile_top10_share": 0.8},
    "draw_equilibrium": {"env/draw_rate": 0.9, "env/episode_steps_at_cap_frac": 0.95},
    "seat_bias": {
        "env/win_rate_by_seat_ci95_lo": 0.62,
        "env/win_rate_by_seat_ci95_hi": 0.71,
    },
    "elixir_count_inexact": {"env/elixir_count_exact_frac": 0.4},
    "shaping_dominates": {"env/reward_shaping_abs": 4.0},
    # A neighbour took about 1.4 GB of the card after startup.
    "vram_spilling": {"health/vram_available_mb": 2000.0},
    "transitivity": {"ladder/transitivity_residual": 0.4},
    "gate_starved": {"ladder/consecutive_gate_failures": 6},
    "capacity_ratio": {"throughput/rollout_capacity_ratio": 0.8},
}

#: Alarms about a counter rising across rows rather than about one row's level. No single row
#: can trip them, so they are tested on their own below and have no TRIPS entry.
RISES = frozenset({"worker_failures", "worker_failures_persistent"})


def _row(**moved: float) -> dict[str, float | int | str]:
    row = dict(HEALTHY)
    row.update(moved)
    return row


def _alarm(name: str) -> Alarm:
    return next(alarm for alarm in default_alarms(AlarmConfig()) if alarm.name == name)


def _set(**overrides: object) -> AlarmSet:
    """An alarm set that says nothing on the console: a test reads the results, not the log."""
    return AlarmSet(AlarmConfig(**overrides), printer=None)  # type: ignore[arg-type]


# -- the table ---------------------------------------------------------------


def test_the_table_covers_every_alarm_the_schema_names() -> None:
    names = {alarm.name for alarm in default_alarms(AlarmConfig())}
    assert names == set(schema.ALARM_METRICS)


def test_every_alarm_is_tested_firing_and_staying_silent() -> None:
    """The healthy-row and trip tests below see only the alarms this file has rows for.

    An alarm added to the table with no entry here passes both of them anyway. The healthy row
    does not carry its keys, so the missing-key rule keeps it silent for a reason that has
    nothing to do with its threshold, and nothing ever asks it to fire. The suite stays green
    around an alarm nobody has seen work, which is the one kind of alarm section 13.3 says is
    unvalidated.
    """
    built = {alarm.name: alarm for alarm in default_alarms(AlarmConfig())}
    assert sorted(set(built) - set(TRIPS) - RISES) == [], "an alarm with no row that trips it"
    for alarm in built.values():
        absent = [key for key in alarm.keys if key not in HEALTHY]
        assert absent == [], f"{alarm.name} is silent on the healthy row: it lacks {absent}"


def test_every_alarm_reads_keys_the_schema_knows() -> None:
    for alarm in default_alarms(AlarmConfig()):
        for key in alarm.keys:
            assert schema.is_known(key), f"{alarm.name} reads {key}, which is not in the schema"


def test_a_healthy_row_fires_nothing() -> None:
    alarms = _set()
    for iteration in range(60):
        assert alarms.evaluate(_row(**{"run/iteration": iteration})) == []


@pytest.mark.parametrize("name", sorted(TRIPS))
def test_each_alarm_fires_on_its_own_row(name: str) -> None:
    alarm = _alarm(name)
    row = _row(**TRIPS[name])
    assert alarm.holds(row)
    assert not alarm.holds(_row())


def test_a_missing_key_is_never_a_firing() -> None:
    """A row from an iteration in which nothing finished carries no episode statistics."""
    sparse = {"run/iteration": 4}
    for alarm in default_alarms(AlarmConfig()):
        assert not alarm.holds(sparse)


# -- patience ----------------------------------------------------------------


def test_patience_is_honoured_and_resets() -> None:
    alarms = _set()
    tripped = _row(**TRIPS["kl_high"])  # patience 3
    assert alarms.evaluate(tripped) == []
    assert alarms.evaluate(tripped) == []
    fired = alarms.evaluate(tripped)
    assert [result.name for result in fired] == ["kl_high"]
    assert fired[0].consecutive == 3
    assert fired[0].severity == WARN
    # One quiet iteration resets the counter: what this guards against is a policy walking
    # away from its behaviour distribution, not one noisy update.
    assert alarms.evaluate(_row()) == []
    assert alarms.evaluate(tripped) == []


def test_a_rise_in_the_worker_restart_counter_warns_once() -> None:
    alarms = _set()
    assert alarms.evaluate(_row()) == []
    fired = alarms.evaluate(_row(**{"health/worker_restarts": 1}))
    assert [result.name for result in fired] == ["worker_failures"]
    # It is a rise and not a level: the same count again is not another failure.
    assert alarms.evaluate(_row(**{"health/worker_restarts": 1})) == []


def test_three_consecutive_rises_halt() -> None:
    alarms = _set()
    alarms.evaluate(_row())
    with pytest.raises(AlarmHalt) as excinfo:
        for restarts in (1, 2, 3):
            alarms.evaluate(_row(**{"health/worker_restarts": restarts}))
    assert excinfo.value.alarm == "worker_failures_persistent"


def test_the_config_can_disable_an_alarm_and_move_its_patience() -> None:
    alarms = _set(disabled=["kl_high"], patience_overrides={"tile_spam": 1})
    assert "kl_high" not in {alarm.name for alarm in alarms.alarms}
    fired = alarms.evaluate(_row(**TRIPS["tile_spam"]))
    assert [result.name for result in fired] == ["tile_spam"]


def test_the_ratio_alarm_is_a_multiple_of_the_precision_it_runs_at() -> None:
    """Under bf16 a deviation of a percent is the precision, not a defect."""
    strict = AlarmSet(AlarmConfig(), ratio_atol=1e-4, printer=None)
    loose = AlarmSet(AlarmConfig(), ratio_atol=2e-2, printer=None)
    row = _row(**{"ppo/ratio_max_abs_dev": 1e-2})
    with pytest.raises(AlarmHalt):
        strict.evaluate(row)
    assert loose.evaluate(row) == []


# -- what a halt does --------------------------------------------------------


def test_a_halt_writes_a_bundle_and_raises(tmp_path: Path) -> None:
    written: list[str] = []

    def on_halt(result: object) -> str:
        folder = write_bundle(
            tmp_path,
            7,
            rows=[dict(HEALTHY)],
            alarms=[result],  # type: ignore[list-item]
            state_digest="abc",
            note="test",
        )
        written.append(str(folder))
        return str(folder)

    alarms = _set()
    with pytest.raises(AlarmHalt) as excinfo:
        alarms.evaluate(_row(**TRIPS["illegal_actions"]), on_halt=on_halt)
    assert excinfo.value.alarm == "illegal_actions"
    assert excinfo.value.bundle == written[0]
    folder = Path(written[0])
    assert msgspec.json.decode((folder / "bundle.json").read_bytes())["state_digest"] == "abc"
    assert (folder / "metrics.jsonl").read_bytes().count(b"\n") == 1
    assert (folder / "alarms.jsonl").exists()


def test_the_halting_row_records_the_warnings_beside_it() -> None:
    """A halt on the clip fraction beside a warning on the KL is a different story."""
    alarms = _set(patience_overrides={"kl_high": 1, "clip_pinned": 1})
    captured: list[MetricRow] = []
    with pytest.raises(AlarmHalt):
        alarms.evaluate(
            _row(**TRIPS["clip_pinned"], **TRIPS["kl_high"]),
            on_halt=lambda result: captured.append(result.values) or None,  # type: ignore[arg-type,func-returns-value]
        )
    assert alarms.fired["kl_high"] == 1
    assert alarms.fired["clip_pinned"] == 1


def test_a_severity_override_turns_a_warning_into_a_halt() -> None:
    alarms = _set(severity_overrides={"tile_spam": HALT}, patience_overrides={"tile_spam": 1})
    with pytest.raises(AlarmHalt):
        alarms.evaluate(_row(**TRIPS["tile_spam"]))


def test_alarms_can_be_turned_off_wholesale() -> None:
    alarms = _set(enabled=False)
    assert alarms.evaluate(_row(**TRIPS["illegal_actions"])) == []
