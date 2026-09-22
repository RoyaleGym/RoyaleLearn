"""What a run watches for, and what it does about it.

Every alarm is a predicate over one metric row, a severity and a patience: it fires when the
predicate has held for ``patience`` consecutive iterations. ``warn`` writes an ``alarms.jsonl``
row and says so on the console; ``halt`` additionally writes a checkpoint and a diagnostic
bundle and then raises ``AlarmHalt``, so the process exits non-zero with the evidence already on
disk.

Two properties shape the table below. The first is that the keys an alarm reads are declared
rather than reached for: ``metrics/schema.py``'s ``ALARM_METRICS`` names them, and the suite
checks the two against each other, so an alarm cannot end up watching a key nobody emits any
more. The second is that a missing key is never a firing: a row from an iteration in which no
episode finished carries no ``env/`` group at all, and an alarm that read that absence as a zero
would halt a healthy run.

Thresholds live in ``config.alarms``: they are recorded in the checkpoint and left out of the
run identity, because an alarm can stop a run and can never alter a number.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from ..api.metrics import Alarm, AlarmResult, MetricRow
from ..errors import AlarmHalt
from .schema import ALARM_METRICS

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..config import AlarmConfig

__all__ = [
    "DEFAULT_RATIO_ATOL",
    "HALT",
    "WARN",
    "AlarmSet",
    "MetricAlarm",
    "RoseAlarm",
    "default_alarms",
    "names",
]

WARN = "warn"
HALT = "halt"


def _value(row: MetricRow, key: str) -> float | None:
    """One key as a float, or None where the row does not carry it.

    A row carries what the iteration produced. An iteration in which nothing finished has no
    episode statistics in it, and reading that as a zero would fire the draw-rate alarm on a
    run that is working.
    """
    raw = row.get(key)
    if raw is None or isinstance(raw, str):
        return None
    return float(raw)


class MetricAlarm(Alarm):
    """An alarm whose predicate is a function of the values of its declared keys.

    The predicate is handed one value per key, in the order ``ALARM_METRICS`` declares them,
    and is not called at all when any of them is missing. That is the whole of the
    missing-key rule, in one place.
    """

    def __init__(
        self,
        name: str,
        predicate: Callable[..., bool],
        *,
        severity: str = WARN,
        patience: int = 1,
        meaning: str = "",
        dump_bundle: bool = False,
    ) -> None:
        self.name = name
        self.severity = severity
        self.patience = int(patience)
        self.keys = ALARM_METRICS[name]
        self.predicate = predicate
        self.meaning = meaning
        #: Whether a firing of this alarm is worth a diagnostic bundle even at ``warn``.
        self.dump_bundle = dump_bundle

    def holds(self, row: MetricRow) -> bool:
        values = [_value(row, key) for key in self.keys]
        if any(value is None for value in values):
            return False
        return bool(self.predicate(*values))

    def message(self, row: MetricRow) -> str:
        parts = ", ".join(f"{key}={row.get(key)!r}" for key in self.keys)
        tail = f" -- {self.meaning}" if self.meaning else ""
        return f"{self.name}: {parts}{tail}"

    def values(self, row: MetricRow) -> dict[str, float]:
        return {key: value for key in self.keys if (value := _value(row, key)) is not None}


class RoseAlarm(MetricAlarm):
    """An alarm about a counter going up rather than about its level.

    Worker restarts are the case: the number itself is cumulative over the run, so the thing
    worth knowing is that it moved this iteration. The previous value is kept here rather than
    derived from the metric stream, because a resumed run's first row is not a step from the
    last row of the run before it.
    """

    def __init__(self, name: str, **kwargs: object) -> None:
        super().__init__(name, self._rose, **kwargs)  # type: ignore[arg-type]
        self._last: float | None = None

    def _rose(self, value: float) -> bool:
        previous, self._last = self._last, value
        return previous is not None and value > previous


#: The tolerance ``ratio_invariant`` is a multiple of when the caller does not say. It belongs
#: to the precision the update runs at -- ``ppo.ratio_atol`` under ``net.autocast_dtype`` -- and
#: the alarm config does not carry it, so the coordinator passes the run's own. The default is
#: the fp32 value, which is what the whole suite runs at.
DEFAULT_RATIO_ATOL = 1e-4


def default_alarms(config: AlarmConfig, *, ratio_atol: float = DEFAULT_RATIO_ATOL) -> tuple[
    Alarm, ...
]:
    """The table of section 13.3, with this run's thresholds in it.

    Built as a function of the config rather than as constants, so that an operator who wants
    one of them louder on their own machine changes a number in the config file and does not
    end up with a second copy of the table.
    """
    alarms: list[Alarm] = [
        MetricAlarm(
            "illegal_actions",
            lambda rate: rate > 0.0,
            severity=HALT,
            meaning=(
                "a mask bug, an unmasked policy or a wrong action encoding; under a correct "
                "mask this is exactly zero"
            ),
        ),
        MetricAlarm(
            "ratio_invariant",
            lambda deviation: deviation > config.ratio_invariant_multiple * ratio_atol,
            severity=HALT,
            meaning=(
                "the stored log-probabilities are not the ones the current weights produce on "
                "the stored bytes: a mask, codec or weight-version mismatch"
            ),
        ),
        MetricAlarm(
            "nonfinite",
            lambda trips: trips > 0.0,
            severity=HALT,
            meaning="a loss, a gradient or a logit was not finite",
        ),
        MetricAlarm(
            "buffer_overflow",
            # A rectangle cannot be more than full: every cell is written exactly once by the
            # worker that owns it. Above one, a cycle was published twice or the geometry moved
            # under the buffer, and the rows of that iteration are not the rows they say they
            # are. The configured threshold is a floor on that bound rather than the bound
            # itself, so lowering it can make the alarm stricter and cannot make a healthy
            # rectangle -- which reads exactly one -- trip it.
            lambda fill: fill > max(1.0, config.buffer_fill_frac),
            severity=HALT,
            meaning="the rectangle holds more cells than it has; an invariant is broken",
        ),
        RoseAlarm("worker_failures", meaning="a rollout worker was restarted"),
        RoseAlarm(
            "worker_failures_persistent",
            severity=HALT,
            patience=3,
            meaning="workers have been failing on three consecutive iterations",
        ),
        MetricAlarm(
            "clip_pinned",
            lambda fraction: fraction > config.clip_fraction,
            severity=HALT,
            patience=3,
            meaning="a mask disagreement, or a learning rate far too high",
        ),
        MetricAlarm(
            "kl_high",
            lambda kl: kl > config.kl_high,
            patience=3,
            meaning="the learning-rate backoff should already be acting",
        ),
        MetricAlarm(
            "kl_dead",
            lambda kl: kl < config.kl_dead,
            patience=10,
            meaning="nothing is moving: dead entropy, a rate too low, or a frozen head",
        ),
        MetricAlarm(
            "ev_negative",
            lambda ev: ev < config.explained_variance,
            patience=50,
            meaning="the critic explains none of the return; the usual cause of a plateau",
        ),
        MetricAlarm(
            "noop_collapse",
            lambda cards: cards < config.cards_per_match_warn,
            patience=5,
            meaning="about twenty-two cards a match is healthy",
        ),
        MetricAlarm(
            "noop_collapse_severe",
            lambda cards: cards < config.cards_per_match_halt,
            severity=HALT,
            patience=5,
            meaning="the policy has stopped playing cards",
        ),
        MetricAlarm(
            "noop_entropy_floor",
            lambda entropy: entropy < config.noop_entropy_floor,
            patience=5,
            meaning=(
                "the leading indicator of no-op collapse, and it watches after ent_coef_noop "
                "has annealed away, which is when it matters most"
            ),
        ),
        MetricAlarm(
            "tile_spam",
            lambda share: share > config.tile_top1_share,
            patience=5,
            meaning="a quarter of every card played goes on one tile",
        ),
        MetricAlarm(
            "artefact_exploit",
            lambda share: share > config.card_tile_top10_share,
            patience=5,
            dump_bundle=True,
            meaning="a real meta is not that concentrated; the bundle carries a trace to read",
        ),
        MetricAlarm(
            "draw_equilibrium",
            lambda draws, at_cap: (
                draws > config.draw_rate and at_cap > config.episode_steps_at_cap_frac
            ),
            patience=5,
            meaning="the turtle equilibrium: nobody attacks and the step limit ends every game",
        ),
        MetricAlarm(
            "seat_bias",
            lambda lo, hi: hi < 0.45 or lo > 0.55,
            patience=3,
            meaning=(
                "an unseeded reset, a reward asymmetry or an observation mirror bug; a few "
                "points can be the shipped engine's own seat asymmetry, which is why this warns"
            ),
        ),
        MetricAlarm(
            "elixir_count_inexact",
            lambda exact: exact < 0.99,
            patience=3,
            meaning=(
                "the observation's opponent-elixir slot is documented exact and was an estimate "
                "on some episodes; near zero rather than just under one, read "
                "run/engine_build_digest first"
            ),
        ),
        MetricAlarm(
            "shaping_dominates",
            lambda shaping, terminal: shaping > terminal,
            patience=5,
            meaning="the shaping terms have taken over the objective",
        ),
        MetricAlarm(
            "transitivity",
            lambda residual: residual > config.transitivity_residual,
            patience=3,
            meaning="one number per player does not explain the result log; the rating is lying",
        ),
        MetricAlarm(
            "gate_starved",
            lambda failures: failures >= config.gate_failures,
            meaning="the plateau signal, stated as an event",
        ),
        MetricAlarm(
            "capacity_ratio",
            lambda ratio: ratio < config.capacity_ratio,
            patience=3,
            meaning="the harness is becoming the bottleneck rather than the learner",
        ),
    ]
    return tuple(alarms)


class AlarmSet:
    """Every alarm, the patience counters behind them, and what a halt does.

    One object per run. It holds the counters rather than the alarms holding them, so that a
    resumed run restarts its patience from zero -- which is the honest reading: the three
    iterations that would have tripped an alarm were not observed by this process.
    """

    def __init__(
        self,
        config: AlarmConfig,
        *,
        ratio_atol: float = DEFAULT_RATIO_ATOL,
        alarms: Iterable[Alarm] | None = None,
        printer: Callable[[str], None] | None = print,
    ) -> None:
        self.config = config
        self.printer = printer
        disabled = set(config.disabled)
        self.alarms: list[Alarm] = []
        table = alarms if alarms is not None else default_alarms(config, ratio_atol=ratio_atol)
        for alarm in table:
            if alarm.name in disabled:
                continue
            alarm.patience = int(config.patience_overrides.get(alarm.name, alarm.patience))
            alarm.severity = str(config.severity_overrides.get(alarm.name, alarm.severity))
            self.alarms.append(alarm)
        self.consecutive: dict[str, int] = {alarm.name: 0 for alarm in self.alarms}
        self.fired: dict[str, int] = {alarm.name: 0 for alarm in self.alarms}

    def evaluate(
        self,
        row: MetricRow,
        *,
        on_halt: Callable[[AlarmResult], str | None] | None = None,
        on_dump: Callable[[AlarmResult], str | None] | None = None,
    ) -> list[AlarmResult]:
        """Every alarm that fired on this row, and a halt if one of them was one.

        The whole table is evaluated before anything is raised, so the ``alarms.jsonl`` row of
        a halting iteration carries the other alarms that were firing beside it. That is
        usually where the cause is: a halt on the clip fraction beside a warning on the KL is a
        different story from a halt on the clip fraction alone.
        """
        if not self.config.enabled:
            return []
        iteration = int(row.get("run/iteration", 0) or 0)
        fired: list[AlarmResult] = []
        halting: AlarmResult | None = None
        for alarm in self.alarms:
            if not alarm.holds(row):
                self.consecutive[alarm.name] = 0
                continue
            count = self.consecutive[alarm.name] + 1
            self.consecutive[alarm.name] = count
            if count < alarm.patience:
                continue
            self.fired[alarm.name] += 1
            result = AlarmResult(
                name=alarm.name,
                severity=alarm.severity,
                iteration=iteration,
                fired=True,
                consecutive=count,
                message=alarm.message(row),
                values=_values_of(alarm, row),
            )
            fired.append(result)
            if self.printer is not None:
                self.printer(f"alarm [{alarm.severity}] {result.message}")
            if getattr(alarm, "dump_bundle", False) and on_dump is not None:
                on_dump(result)
            if alarm.severity == HALT and halting is None:
                halting = result
        if halting is not None:
            bundle = on_halt(halting) if on_halt is not None else None
            raise AlarmHalt(halting.name, halting.message, halting.iteration, bundle)
        return fired


def _values_of(alarm: Alarm, row: MetricRow) -> dict[str, float]:
    reader = getattr(alarm, "values", None)
    if callable(reader):
        return dict(reader(row))
    return {key: value for key in alarm.keys if (value := _value(row, key)) is not None}


def names(alarms: Sequence[Alarm]) -> tuple[str, ...]:
    """The alarms' names, for a printout and for the tests."""
    return tuple(alarm.name for alarm in alarms)
