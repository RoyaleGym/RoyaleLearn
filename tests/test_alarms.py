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

from contributed import contributed_alarms, full_alarms, full_config, full_schema
from royalelearn.api.metrics import Alarm, AlarmResult, MetricRow
from royalelearn.config import AlarmConfig, RunConfig, check_consistency
from royalelearn.errors import AlarmHalt, PreflightError
from royalelearn.metrics import schema
from royalelearn.metrics.alarms import HALT, WARN, AlarmSet, MetricAlarm, default_alarms, names
from royalelearn.metrics.bundle import write_bundle

#: A row in the middle of every healthy band the schema declares. Every test below starts from
#: this and moves one key.
HEALTHY: dict[str, float | int | str] = {
    "run/iteration": 1,
    "env/illegal_action_rate": 0.0,
    "ppo/ratio_max_abs_dev": 1e-7,
    "ppo/reward_clip_frac": 0.0,
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
    # What laptop runs at minibatch 256 on a 4 GB card report: a 48 s update with about 1.2 GB
    # of the card still free.
    "time/update": 48.0,
    "health/vram_driver_free_mb": 1176.0,
    "ladder/transitivity_residual": 0.02,
    "ladder/consecutive_gate_failures": 0,
    "throughput/rollout_capacity_ratio": 2.4,
    # One imitation regulariser named "bc", well inside its budget, long after the unfreeze.
    "imitation/bc/kl": 0.1,
    "imitation/bc/lambda_at_max": 0.0,
    "ppo/iterations_since_unfreeze": 50.0,
    "ppo/ev_at_unfreeze": 0.6,
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
    "reward_clipped": {"ppo/reward_clip_frac": 0.001},
    "transitivity": {"ladder/transitivity_residual": 0.4},
    "gate_starved": {"ladder/consecutive_gate_failures": 6},
    "capacity_ratio": {"throughput/rollout_capacity_ratio": 0.8},
    "imitation_ref_kl_high": {"imitation/bc/kl": 1.5},
    "imitation_lambda_saturated": {"imitation/bc/lambda_at_max": 1.0},
    # Inside the handoff window with a clip fraction the handoff bound catches and clip_pinned
    # does not: it is the early-unfreeze reading this alarm exists for.
    "actor_handoff": {"ppo/iterations_since_unfreeze": 3.0, "ppo/clip_fraction": 0.35},
    "critic_unready": {"ppo/ev_at_unfreeze": 0.1},
}

#: Alarms about a counter rising across rows rather than about one row's level. No single row
#: can trip them, so they are tested on their own below and have no TRIPS entry.
RISES = frozenset({"worker_failures", "worker_failures_persistent"})

#: Alarms that compare a row against the rows before it rather than against a threshold. Like the
#: rises, no single row can trip one, so they are tested on their own below.
HISTORY = frozenset({"vram_spilling"})


def _row(**moved: float) -> dict[str, float | int | str]:
    row = dict(HEALTHY)
    row.update(moved)
    return row


def _alarm(name: str) -> Alarm:
    return next(alarm for alarm in full_alarms() if alarm.name == name)


def _set(**overrides: object) -> AlarmSet:
    """The alarm set of a run with every optional part, saying nothing on the console: a test
    reads the results, not the log."""
    config = AlarmConfig(**overrides)  # type: ignore[arg-type]
    return AlarmSet(config, extra=contributed_alarms(config), printer=None)


# -- the table ---------------------------------------------------------------


def test_the_table_covers_every_alarm_the_schema_names() -> None:
    """The core table against the core schema, and a run's whole table against its own."""
    core = {alarm.name for alarm in default_alarms(AlarmConfig())}
    assert core == set(schema.ALARM_METRICS)
    assert set(names(full_alarms())) == set(full_schema().alarm_metrics)


def test_a_run_without_the_optional_parts_has_none_of_their_alarms() -> None:
    """The regularisers' and the freeze's alarms belong to the runs that have them."""
    plain = set(names(default_alarms(AlarmConfig())))
    contributed = set(names(contributed_alarms()))
    assert contributed == {
        "imitation_ref_kl_high",
        "imitation_lambda_saturated",
        "actor_handoff",
        "critic_unready",
    }
    assert not plain & contributed
    assert len(plain) == 24


def test_an_override_naming_no_alarm_of_the_run_is_refused() -> None:
    """Section 13.3 after the freeze's alarms were renamed: an old name would otherwise switch
    the renamed alarm back on without a word. The refusal names the new name."""
    with pytest.raises(PreflightError, match="renamed to 'actor_handoff'"):
        AlarmSet(AlarmConfig(disabled=["imitation_handoff"]), printer=None)
    with pytest.raises(PreflightError, match="patience_overrides names 'critic_unreadyy'"):
        _set(patience_overrides={"critic_unreadyy": 3})
    # A contributed alarm is a name only on a run that has its part.
    with pytest.raises(PreflightError, match="imitation_ref_kl_high"):
        AlarmSet(AlarmConfig(severity_overrides={"imitation_ref_kl_high": HALT}), printer=None)
    kept = _set(disabled=["imitation_ref_kl_high"]).alarms
    assert "imitation_ref_kl_high" not in {alarm.name for alarm in kept}


def test_the_config_check_refuses_the_same_names_before_anything_is_built() -> None:
    """The coordinator builds its alarm set after the rollout buffer's shared memory exists, so
    the refusal that matters is the config check's, which runs first."""
    renamed = msgspec.structs.replace(
        RunConfig(), alarms=AlarmConfig(severity_overrides={"imitation_critic_unready": HALT})
    )
    assert any("renamed to 'critic_unready'" in p for p in check_consistency(renamed))
    assert not any(
        "alarm table" in p
        for p in check_consistency(full_config(AlarmConfig(disabled=["critic_unready"])))
    )


def test_a_name_twice_in_the_table_is_refused() -> None:
    """The patience counters are keyed by name: a doubled alarm would count twice a row."""
    twice = MetricAlarm("kl_high", lambda kl: kl > 1.0)
    with pytest.raises(ValueError, match="kl_high"):
        AlarmSet(AlarmConfig(), extra=[twice], printer=None)


def test_every_alarm_is_tested_firing_and_staying_silent() -> None:
    """The healthy-row and trip tests below see only the alarms this file has rows for.

    An alarm added to the table with no entry here passes both of them anyway. The healthy row
    does not carry its keys, so the missing-key rule keeps it silent for a reason that has
    nothing to do with its threshold, and nothing ever asks it to fire. The suite stays green
    around an alarm nobody has seen work, which is the one kind of alarm section 13.3 says is
    unvalidated.
    """
    built = {alarm.name: alarm for alarm in full_alarms()}
    assert sorted(set(built) - set(TRIPS) - RISES - HISTORY) == [], (
        "an alarm with no row that trips it"
    )
    for alarm in built.values():
        absent = [key for key in alarm.keys if not any(_carries(key, k) for k in HEALTHY)]
        assert absent == [], f"{alarm.name} is silent on the healthy row: it lacks {absent}"


def _carries(template: str, key: str) -> bool:
    """Whether a row key is the template itself or, for a family, one of its members."""
    import re

    pattern = re.escape(template).replace(r"\{name\}", "[^/]+")
    return re.fullmatch(pattern, key) is not None


def test_a_family_alarm_names_only_the_members_that_hold() -> None:
    """Two regularisers, one over its bound: the message names that one and not the other, and
    a row carrying no member at all is not a firing."""
    alarm = _alarm("imitation_ref_kl_high")
    row = _row(**{"imitation/bc/kl": 1.5, "imitation/timing/kl": 0.2})
    assert alarm.holds(row)
    assert "imitation/bc/kl" in alarm.message(row)
    assert "imitation/timing/kl" not in alarm.message(row)
    bare = {key: value for key, value in HEALTHY.items() if not key.startswith("imitation/")}
    assert not alarm.holds(bare)


def _spill_rows(alarms, seconds, free_mb, *, count=1, start=1):
    """Feed ``count`` rows at one update time and one free-memory reading."""
    fired = []
    for offset in range(count):
        fired = alarms.evaluate(
            _row(
                **{
                    "run/iteration": start + offset,
                    "time/update": seconds,
                    "health/vram_driver_free_mb": free_mb,
                }
            )
        )
    return [result.name for result in fired]


def test_the_spill_alarm_fires_when_a_full_card_comes_with_a_slow_update() -> None:
    """The measured case: 48 s with room, then 200 s with none. Patience is three rows."""
    alarms = _set()
    _spill_rows(alarms, 48.0, 1176.0, count=3)
    assert _spill_rows(alarms, 200.0, 0.0, start=4) == []
    assert _spill_rows(alarms, 200.0, 0.0, start=5) == []
    assert _spill_rows(alarms, 200.0, 0.0, start=6) == ["vram_spilling"]


def test_the_spill_alarm_is_silent_when_the_machine_is_merely_busy() -> None:
    """Contention slows an update too, and it is not this alarm's subject.

    Measured on this machine: the same update took 435-531 s quiet and 758 s under load, with the
    card unchanged. Without the memory reading beside it, an alarm on time alone would call a
    loaded laptop a spill every time somebody else built something.
    """
    alarms = _set()
    _spill_rows(alarms, 48.0, 1176.0, count=3)
    for iteration in range(4, 12):
        assert _spill_rows(alarms, 400.0, 1176.0, start=iteration) == []


def test_the_spill_alarm_is_silent_on_a_full_card_that_is_keeping_up() -> None:
    """A caching allocator that has finished growing sits at or near zero free as a matter of
    course, so a full card on its own says nothing. Measured: a healthy run read exactly 0.0 free
    on every iteration."""
    alarms = _set()
    for iteration in range(1, 12):
        assert _spill_rows(alarms, 48.0, 0.0, start=iteration) == []


def test_the_spill_alarms_bar_is_the_best_of_the_run_not_its_first_iteration() -> None:
    """The first update pays for cuDNN's first look at each shape, so it is the slowest of a
    healthy run. Taken as the baseline it would hide a spill that is only twice as slow as it."""
    alarms = _set()
    _spill_rows(alarms, 90.0, 1176.0)
    _spill_rows(alarms, 45.0, 1176.0, start=2)
    assert _spill_rows(alarms, 95.0, 0.0, start=3) == []
    assert _spill_rows(alarms, 95.0, 0.0, start=4) == []
    assert _spill_rows(alarms, 95.0, 0.0, start=5) == ["vram_spilling"]


def test_an_alarm_that_names_a_metric_key_names_one_that_exists() -> None:
    """An alarm's message is read by somebody in a hurry, and it should not send them hunting.

    ``elixir_count_inexact`` told the reader to read ``run/engine_build_digest`` first. No such key
    is in the schema or in any row: the engine build is in the run's identity.json, and the only
    digest a row carries is the learner's weights. Found by the docs session while writing the page
    that lists one section per alarm.
    """
    run_schema = full_schema()
    for alarm in full_alarms():
        for word in alarm.meaning.replace(",", " ").replace(";", " ").split():
            token = word.strip("`'\".()").rstrip(".")
            if "/" in token and token.split("/")[0] in {
                "run", "env", "ppo", "policy", "ladder", "health", "time", "throughput"
            }:
                assert run_schema.is_known(token), (
                    f"{alarm.name} names {token}, which no row carries"
                )


def test_every_alarm_reads_keys_the_schema_knows() -> None:
    """Each alarm against the schema of a run that holds it, and each declares its keys where
    that schema says it does."""
    run_schema = full_schema()
    for alarm in full_alarms():
        assert tuple(alarm.keys) == run_schema.alarm_metrics[alarm.name], alarm.name
        for key in alarm.keys:
            assert run_schema.is_known(key), (
                f"{alarm.name} reads {key}, which is not in the schema"
            )
    for alarm in default_alarms(AlarmConfig()):
        for key in alarm.keys:
            assert schema.is_known(key), f"{alarm.name} reads {key}, which the core lacks"


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
    for alarm in full_alarms():
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


# -- what a halting iteration leaves behind ----------------------------------


def test_a_halting_iteration_hands_over_its_alarms_before_it_raises() -> None:
    """The one iteration a reader most wants is the one that used to be missing.

    ``evaluate`` raised ``AlarmHalt`` before RETURNING the fired list, and the coordinator wrote
    ``alarms.jsonl`` from that return value. So a halting iteration's alarms -- the halting one
    and every warning beside it -- never reached the file. This module's own docstring promised
    the opposite ("the alarms.jsonl row of a halting iteration carries the other alarms that
    were firing beside it"), and the promise was broken one step downstream of where it is made.

    The order matters as well as the fact. The alarms go to the writer BEFORE ``on_halt`` runs,
    because ``on_halt`` writes the checkpoint and the diagnostic bundle, and a bundle assembled
    from a run directory ought to find the alarms in it.
    """
    alarms = _set(patience_overrides={"kl_high": 1, "clip_pinned": 1})
    order: list[str] = []
    written: list[AlarmResult] = []

    def on_fired(results: list[AlarmResult]) -> None:
        order.append("fired")
        written.extend(results)

    def on_halt(result: AlarmResult) -> str | None:
        order.append("halt")
        return None

    with pytest.raises(AlarmHalt):
        alarms.evaluate(
            _row(**TRIPS["clip_pinned"], **TRIPS["kl_high"]),
            on_halt=on_halt,
            on_fired=on_fired,
        )

    assert order == ["fired", "halt"], (
        "the alarms must reach the writer before the bundle is assembled from the run directory"
    )
    assert {result.name for result in written} == {"clip_pinned", "kl_high"}
    assert [result.severity for result in written if result.name == "clip_pinned"] == [HALT]


def test_a_quiet_iteration_hands_over_nothing() -> None:
    """No firing is not an empty firing: a writer called with [] would append an empty batch."""
    alarms = _set()
    calls: list[list[AlarmResult]] = []
    assert alarms.evaluate(_row(), on_fired=calls.append) == []
    assert calls == []


def test_a_plain_run_builds_the_core_table_and_nothing_else() -> None:
    """What a run without sections holds, from its config the way the coordinator gets it."""
    from royalelearn.extensions import extension_alarms, schema_contributions, with_sections
    from royalelearn.metrics.schema import for_run

    plain = RunConfig()
    assert extension_alarms(plain) == ()
    assert schema_contributions(plain) == ()
    built = AlarmSet(plain.alarms, extra=extension_alarms(plain), printer=None)
    assert {alarm.name for alarm in built.alarms} == set(schema.ALARM_METRICS)
    assert not for_run(schema_contributions(plain)).is_known("imitation/bc/kl")
    refused = msgspec.structs.replace(plain, alarms=AlarmConfig(disabled=["critic_unready"]))
    assert any("critic_unready" in problem for problem in check_consistency(refused))
    # a section that only loads weights schedules no freeze and anchors nothing
    init_only = with_sections(plain, warm_start={"init": {"path": "x", "sha256": "y"}})
    assert extension_alarms(init_only) == ()


@pytest.mark.parametrize(
    ("alarm", "keys", "over", "under"),
    [
        (
            "actor_handoff",
            ("ppo/kl", "ppo/clip_fraction", "ppo/iterations_since_unfreeze"),
            (0.071, 0.0, 5.0),
            (0.069, 0.0, 5.0),
        ),
        (
            "actor_handoff",
            ("ppo/kl", "ppo/clip_fraction", "ppo/iterations_since_unfreeze"),
            (0.0, 0.41, 5.0),
            (0.0, 0.39, 5.0),
        ),
        (
            "actor_handoff",
            ("ppo/kl", "ppo/clip_fraction", "ppo/iterations_since_unfreeze"),
            (0.2, 0.0, 5.0),
            (0.2, 0.0, 6.0),
        ),
        ("critic_unready", ("ppo/ev_at_unfreeze",), (0.19,), (0.21,)),
    ],
)
def test_each_freeze_threshold_is_the_one_its_section_names(
    alarm: str, keys: tuple[str, ...], over: tuple[float, ...], under: tuple[float, ...]
) -> None:
    """Four thresholds set to four distinct values, each probed either side of its own number:
    swapping two of them in the wiring would move a boundary here."""
    from royalelearn.extensions import extension_alarms, with_sections

    config = with_sections(
        RunConfig(),
        warm_start={
            "actor_lr_scale": {"kind": "constant", "value": 1.0},
            "alarms": {
                "handoff_window": 5,
                "handoff_kl": 0.07,
                "handoff_clip": 0.4,
                "ev_at_unfreeze": 0.2,
            },
        },
    )
    (built,) = [a for a in extension_alarms(config) if a.name == alarm]
    assert built.holds(dict(zip(keys, over, strict=True)))
    assert not built.holds(dict(zip(keys, under, strict=True)))


def test_a_doubled_alarm_name_is_refused_by_the_config_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before anything is built: the config check sees the run's whole table, sections and all."""
    from royalelearn.extensions import with_sections
    from royalelearn.testing import StubExtension, use_extensions

    class _Doubler(StubExtension):
        def alarms(self, section: object) -> tuple[Alarm, ...]:
            return (MetricAlarm("kl_high", lambda kl: kl > 1.0),)

    use_extensions(monkeypatch, {"doubler": _Doubler("doubler")})
    config = with_sections(RunConfig(), doubler={})
    assert any("more than once: ['kl_high']" in p for p in check_consistency(config))
