"""Make each alarm's condition true and see whether it notices.

Nineteen of these alarms had never fired in any recorded run, and four turned out to be unable to
fire at all: `vram_spilling` watched a quantity that floors at what the process already holds,
`shaping_dominates` read two shares that were zero in every row ever written, `noop_collapse` and
its severe twin read a mean over the learner AND its opponent. From a run log "never fired" and
"cannot fire" are the same picture, and a guard nobody has seen trip is not a guard.

So this is the standing version of the check that found them. It began as a one-off script run
by hand against a run directory; here it runs in the suite, against a REAL metric row rather than
a synthetic one, so an alarm gated on a key that no row carries fails here rather than in six
weeks.

Three properties, and the third is the one a synthetic row cannot give:

* every alarm in the table has a plant, and every plant names an alarm that exists;
* a plant only touches keys the alarm itself DECLARES, so an alarm tripped through a key missing
  from its own ``keys`` tuple is a failure rather than a pass;
* the plant fires it, from a baseline the alarm was quiet on.

The baseline is iteration 4 of `train-hog26-3`, saved as `tests/data/metrics-row.json`. It is not
a healthy row and is not pretending to be: two alarms hold on it, and both are true statements
about that run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from royalelearn.config import AlarmConfig
from royalelearn.errors import AlarmHalt
from royalelearn.metrics.alarms import AlarmSet, default_alarms

#: One iteration of a real run. Real, because the interesting failure is an alarm that reads a
#: key no row carries, and a row written by a test carries whatever the test decided to write.
BASELINE = json.loads(
    (Path(__file__).parent / "data" / "metrics-row.json").read_text(encoding="utf-8")
)

#: What that row really trips, and why each is a true statement rather than a defect here.
#: `ppo/kl` is 9.4e-07, which is a policy that is barely moving, and `explained_variance` is
#: -0.022 at iteration 4, which is a critic that has not started. Both were confirmed findings on
#: that run. If this set has to grow, the question is what changed about the row, not about the
#: alarms.
BASELINE_FIRES = frozenset({"kl_dead", "ev_negative"})

#: Those two values put back inside their documented bands, and nothing else touched. The plants
#: are fed from THIS row so that every alarm's evidence is the same shape: silent first, then
#: fired. Without it, `kl_dead` and `ev_negative` would "pass" their plant by having been firing
#: already, which is the weaker claim of the two and indistinguishable from the stronger one.
QUIET = dict(BASELINE, **{"ppo/kl": 0.008, "ppo/explained_variance": 0.7})

#: How to trip each alarm, in keys that alarm itself reads. A value is a statement about the
#: alarm's own threshold. A LIST is fed in order and then held at the last: some alarms are
#: stateful and a constant row can never trip them, which is a way of testing nothing at all.
#: A ``"+n"`` string is a RISE from the baseline's own value, for the alarms that read a change
#: between rows rather than a level.
PLANTS: dict[str, dict[str, Any] | list[dict[str, Any]]] = {
    "ratio_invariant": {"ppo/ratio_max_abs_dev": 1.0},
    # Below the 1% the schema used to call healthy: under potential shaping any cut is a cut.
    "reward_clipped": {"ppo/reward_clip_frac": 0.001},
    "nonfinite": {"health/nan_guard_trips": 3},
    "buffer_overflow": {"health/buffer_fill_frac": 1.5},
    "illegal_actions": {"env/illegal_action_rate": 0.5},
    "worker_failures": {"health/worker_restarts": "+1"},
    "worker_failures_persistent": {"health/worker_restarts": "+1"},
    "clip_pinned": {"ppo/clip_fraction": 0.9},
    "kl_high": {"ppo/kl": 0.5},
    "kl_dead": {"ppo/kl": 0.0},
    "ev_negative": {"ppo/explained_variance": -1.0},
    "noop_collapse": {"policy/cards_per_match": 1.0},
    "noop_collapse_severe": {"policy/cards_per_match": 0.5},
    "noop_entropy_floor": {"ppo/noop_entropy": 0.0},
    # Only tile_top1_share: card_tile_top10_share belongs to artefact_exploit, and planting
    # both would have let either alarm pass on the other one's key.
    "tile_spam": {"policy/tile_top1_share": 0.99},
    "draw_equilibrium": {"env/draw_rate": 0.99, "env/episode_steps_at_cap_frac": 0.99},
    "seat_bias": {"env/win_rate_by_seat_ci95_lo": 0.8, "env/win_rate_by_seat_ci95_hi": 0.95},
    "elixir_count_inexact": {"env/elixir_count_exact_frac": 0.0},
    "shaping_dominates": {"env/reward_shaping_abs": 10.0, "env/reward_terminal_abs": 0.01},
    # Row one sets the running best at a fast update; row two is three times slower on a card the
    # driver says is full. One row would set the best and then be judged against itself.
    "vram_spilling": [
        {"time/update": 24.0, "health/vram_driver_free_mb": 900.0},
        {"time/update": 72.0, "health/vram_driver_free_mb": 12.0},
    ],
    "transitivity": {"ladder/transitivity_residual": 0.99},
    "gate_starved": {"ladder/consecutive_gate_failures": 99},
    "capacity_ratio": {"throughput/rollout_capacity_ratio": 0.01},
    "artefact_exploit": {"policy/card_tile_top10_share": 0.99},
    # The imitation alarms read families named by the run's own regularisers; "bc" is one. The
    # real baseline row has no imitation keys at all, which is also what makes them quiet on it.
    "imitation_ref_kl_high": {"imitation/bc/kl": 2.0},
    "imitation_lambda_saturated": {"imitation/bc/lambda_at_max": 1.0},
    # Early after an unfreeze, with a clip fraction the handoff bound catches and clip_pinned
    # does not.
    "imitation_handoff": {"imitation/iterations_since_unfreeze": 2.0, "ppo/clip_fraction": 0.4},
    "imitation_critic_unready": {"imitation/ev_at_unfreeze": 0.05},
}

ALARMS = {alarm.name: alarm for alarm in default_alarms(AlarmConfig())}


def one(name: str) -> AlarmSet:
    """An alarm set holding exactly one alarm.

    One per alarm, because a shared set advances every alarm's patience counter on every row and
    a fire could then belong to a neighbour.
    """
    only = [alarm for alarm in default_alarms(AlarmConfig()) if alarm.name == name]
    assert only, f"{name} is not in the alarm table"
    return AlarmSet(AlarmConfig(), alarms=only, printer=None)


def fires(name: str, rows: list[dict[str, Any]]) -> bool:
    """Feed the rows in order and say whether the alarm fired on any of them."""
    alarms = one(name)
    for row in rows:
        try:
            if alarms.evaluate(row):
                return True
        except AlarmHalt:
            return True
    return False


def planted(name: str) -> list[dict[str, Any]]:
    """The rows to feed: the baseline with this alarm's plant applied, one row per iteration.

    Long enough to outlast the patience, with the listed steps in order and held at the last.
    A ``"+n"`` value RISES on every row rather than on every listed step, which is the only way
    a ``RoseAlarm`` can see anything: it reads the change between consecutive rows, so a row
    planted once and then repeated is a constant and rises by nothing. The first version of this
    file did exactly that and reported both worker alarms as deaf.
    """
    plant = PLANTS[name]
    steps = plant if isinstance(plant, list) else [plant]
    count = ALARMS[name].patience + len(steps) + 1
    rows: list[dict[str, Any]] = []
    for index in range(count):
        row = dict(QUIET)
        for key, value in steps[min(index, len(steps) - 1)].items():
            if isinstance(value, str) and value.startswith("+"):
                row[key] = float(QUIET.get(key, 0) or 0) + float(value[1:]) * (index + 1)
            else:
                row[key] = value
        rows.append(row)
    return rows


def test_every_alarm_has_a_plant_and_every_plant_an_alarm() -> None:
    """A gap in either direction is silent otherwise: an unplanted alarm is simply not checked."""
    assert set(PLANTS) == set(ALARMS), (
        f"alarms with no plant: {sorted(set(ALARMS) - set(PLANTS))}; "
        f"plants for alarms that no longer exist: {sorted(set(PLANTS) - set(ALARMS))}"
    )


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_a_plant_only_touches_keys_the_alarm_declares(name: str) -> None:
    """Otherwise the plant proves something about the row rather than about the alarm.

    ``Alarm.keys`` is what ``metrics/schema.py`` checks against the emitted keys, so an alarm
    tripped through a key outside it is an alarm whose declaration is wrong -- and the schema
    check would not notice, because it only looks at what is declared.
    """
    declared = ALARMS[name].keys
    plant = PLANTS[name]
    steps = plant if isinstance(plant, list) else [plant]
    used = {key for step in steps for key in step}
    undeclared = sorted(key for key in used if not any(_member(t, key) for t in declared))
    assert not undeclared, f"{name} was planted through {undeclared}"


def _member(template: str, key: str) -> bool:
    """A declared key, or a member of a declared family such as ``imitation/{name}/kl``."""
    import re

    return re.fullmatch(re.escape(template).replace(r"\{name\}", "[^/]+"), key) is not None


def test_the_baseline_trips_exactly_what_that_run_really_had_wrong() -> None:
    """Pins the fixture. A row that quietly stopped tripping these is a different row."""
    holding = {name for name, alarm in ALARMS.items() if alarm.holds(BASELINE)}
    assert holding == BASELINE_FIRES


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_an_alarm_is_quiet_before_it_is_planted(name: str) -> None:
    """Half of a plant's evidence, and every alarm gets to give it."""
    repeats = ALARMS[name].patience + 2
    assert not fires(name, [dict(QUIET) for _ in range(repeats)])


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_a_planted_condition_fires_its_alarm(name: str) -> None:
    """The check itself. DEAF here means the guard cannot do the job its message claims."""
    assert fires(name, planted(name)), (
        f"{name} did not fire on a row carrying its own tripping condition: "
        f"{PLANTS[name]}. Either the predicate is wrong or something else gates it."
    )
