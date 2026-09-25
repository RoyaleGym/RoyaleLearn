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

The table below is the core's, the same for every run. An optional part of a run brings its own
alarms, each constructed with the keys it reads, and they are appended to this run's table only:
a run without the part neither evaluates them nor accepts an override naming them.
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
    "RENAMED_ALARMS",
    "WARN",
    "AlarmSet",
    "FamilyAlarm",
    "MetricAlarm",
    "RoseAlarm",
    "SpillAlarm",
    "default_alarms",
    "names",
    "override_problems",
]

WARN = "warn"
HALT = "halt"

#: Alarms that used to be called something else, and what they are called now. An override in
#: ``config.alarms`` naming the old name is refused with the new one in the message, because
#: acting on it silently would re-enable an alarm the operator had switched off.
RENAMED_ALARMS: dict[str, str] = {
    "imitation_handoff": "actor_handoff",
    "imitation_critic_unready": "critic_unready",
}


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

    The predicate is handed one value per key, in the order they are declared -- ``keys``, or
    for a core alarm its ``ALARM_METRICS`` entry -- and is not called at all when any of them
    is missing. That is the whole of the missing-key rule, in one place.
    """

    def __init__(
        self,
        name: str,
        predicate: Callable[..., bool],
        *,
        keys: Sequence[str] | None = None,
        severity: str = WARN,
        patience: int = 1,
        meaning: str = "",
        dump_bundle: bool = False,
    ) -> None:
        self.name = name
        self.severity = severity
        self.patience = int(patience)
        self.keys = tuple(keys) if keys is not None else ALARM_METRICS[name]
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


class SpillAlarm(MetricAlarm):
    """The update has slowed against this run's own best while the card has no room left.

    Written this way because the obvious way does not work. The first version compared the memory
    this process could still take -- free memory plus what its allocator already held -- against
    the peak one minibatch measured at startup. The train session tested it on the card: a second
    process took 2 GB and left 35 MB free for five iterations, and the alarm stayed silent by
    645 MB. The reason is arithmetic rather than tuning. An outsider can only consume the FREE
    part, so that sum bottoms out at what this process holds, and the startup gate guarantees that
    what it holds is more than one minibatch needs. It was least sensitive in the case its own
    text named.

    Nothing this process can read about memory says the thing that matters either, because the
    platform does not refuse an oversubscribed allocation: it backs it with host memory over PCIe
    and reports success. The allocation is served, the counters look ordinary and the run is
    several times slower for the rest of its life. So the alarm watches the harm and takes the
    memory reading as the evidence that the harm is this one: the update several times slower than
    the best this run has managed, on a card the driver says is full. Measured 2026-09-22 on the
    4 GB card: the same update took 47-49 s with room to spare and 180-233 s without, while machine
    contention alone moved it by 1.4 to 1.7. A factor of two sits between them.

    The best is a running minimum rather than a mean, so a slow iteration cannot raise the bar it
    is judged against, and the first iteration's cuDNN warm-up cannot lower it.
    """

    def __init__(
        self, name: str, *, factor: float, floor_mb: float, **kwargs: object
    ) -> None:
        super().__init__(name, self._spilling, **kwargs)  # type: ignore[arg-type]
        self.factor = float(factor)
        self.floor_mb = float(floor_mb)
        self._best: float | None = None

    def _spilling(self, seconds: float, driver_free_mb: float) -> bool:
        if seconds <= 0.0:
            return False
        best = self._best
        if best is None or seconds < best:
            self._best = seconds
            return False
        return seconds >= self.factor * best and driver_free_mb < self.floor_mb


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


class FamilyAlarm(Alarm):
    """An alarm over a family of keys, one per member: it holds when any member's value does.

    For keys named by the run's own config -- one per regulariser, say -- which cannot be
    listed ahead of time. ``keys`` are templates such as ``myext/{name}/kl``, and this reads
    every key of the row that a template matches. ``predicate`` gets the member's name and its
    value, so a bound can differ per member. A row without any member is not a firing.
    """

    def __init__(
        self,
        name: str,
        predicate: Callable[[str, float], bool],
        *,
        keys: Sequence[str] | None = None,
        severity: str = WARN,
        patience: int = 1,
        meaning: str = "",
    ) -> None:
        import re

        self.name = name
        self.severity = severity
        self.patience = int(patience)
        self.keys = tuple(keys) if keys is not None else ALARM_METRICS[name]
        self.predicate = predicate
        self.meaning = meaning
        self.dump_bundle = False
        self._patterns = [
            re.compile("^" + re.escape(template).replace(r"\{name\}", "([^/]+)") + "$")
            for template in self.keys
        ]

    def _members(self, row: MetricRow) -> list[tuple[str, str, float]]:
        found: list[tuple[str, str, float]] = []
        for key in sorted(row):
            for pattern in self._patterns:
                match = pattern.match(key)
                value = _value(row, key)
                if match and value is not None:
                    found.append((key, match.group(1), value))
        return found

    def holds(self, row: MetricRow) -> bool:
        return any(self.predicate(member, value) for _, member, value in self._members(row))

    def message(self, row: MetricRow) -> str:
        firing = [
            f"{key}={value!r}"
            for key, member, value in self._members(row)
            if self.predicate(member, value)
        ]
        tail = f" -- {self.meaning}" if self.meaning else ""
        return f"{self.name}: {', '.join(firing)}{tail}"

    def values(self, row: MetricRow) -> dict[str, float]:
        return {key: value for key, _member, value in self._members(row)}


def default_alarms(
    config: AlarmConfig,
    *,
    ratio_atol: float = DEFAULT_RATIO_ATOL,
    extra: Iterable[Alarm] = (),
) -> tuple[Alarm, ...]:
    """The table of section 13.3, with this run's thresholds in it, then ``extra``.

    Built as a function of the config rather than as constants, so that an operator who wants
    one of them louder on their own machine changes a number in the config file and does not
    end up with a second copy of the table. ``extra`` is what the run's optional parts
    contribute; without it this is the core table every run has.
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
            # The fourth cause is worth naming because it is invisible from the KL alone: under
            # the Adam eps floor the step is lr*m/eps rather than normalised, so a small gradient
            # stays a small step. It is a hypothesis to CHECK and not the answer -- measured on
            # this project's own runs it is not what happened. hog26-6 ran at adam_eps 1e-08 and
            # its last checkpoint has 10.1% of actor parameters under that floor against 16.9%
            # of the critic's, so the actor was LESS floored than the critic while its gradient
            # norm was 0.0068 against the critic's 26.6. The comparison that means something is
            # the two shares against each other in one row, not either against a threshold.
            meaning=(
                "nothing is moving: dead entropy, a rate too low, a frozen head, or an actor "
                "under the Adam eps floor -- compare ppo/adam_eps_floor_frac_actor with the "
                "critic's in the same row, and only if the actor's is far higher is ppo.adam_eps "
                "the lever rather than lr_actor"
            ),
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
                "on some episodes; near zero rather than just under one, read the run's "
                "identity.json engine_build first"
            ),
        ),
        # "The optimum is unchanged" was an OBSERVATION until this: potential shaping telescopes
        # only while every step's reward reaches the return intact, and the clip is the one
        # place a step can be cut. It never fired in the first eight runs on this machine, which
        # said the rewards had stayed small, not that they must. Stronger weights, another
        # reward scale or another deck can start it cutting, and nothing else would change
        # visibly. Found by the integrator, reading every run's rows, 2026-09-24.
        MetricAlarm(
            "reward_clipped",
            lambda frac: frac > 0.0,
            patience=1,
            meaning=(
                "a reward was clipped: on that step the potential terms no longer cancel, so the "
                "shaping can move the optimum, and if the terminal step was cut a win is worth "
                "less than a win. Raise advantage.reward_clip or lower the shaping weights"
            ),
        ),
        MetricAlarm(
            "shaping_dominates",
            lambda shaping, terminal: shaping > terminal,
            patience=5,
            meaning=(
                "the shaping terms have taken over the objective; under a potential reward that "
                "means one of them has stopped telescoping rather than that a weight is wrong"
            ),
        ),
        # The only threshold here measured on the machine it is applied to. `needed` is the
        # device peak the preflight measured for THIS minibatch on THIS card, plus the
        # configured headroom; `available` is free memory plus what this process already holds,
        # which is what it could occupy if it asked. The preflight guards the first instant and
        # nothing guarded the rest: another process taking memory at hour three produces the
        # same several-times-slower run, silently, because this platform backs an oversubscribed
        # allocation with host RAM rather than refusing it.
        #
        # It warns rather than halting, deliberately. Stopping a nine-hour run because a
        # neighbour got greedy is worse than the slowdown it would prevent, and the preflight
        # can afford to refuse only because nothing is lost at second five.
        SpillAlarm(
            "vram_spilling",
            factor=2.0,
            floor_mb=128.0,
            patience=3,
            meaning=(
                "the update is at least twice this run's best while the driver reports no free "
                "device memory: the sign of an allocation backed by host memory over PCIe"
            ),
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
    alarms.extend(extra)
    return tuple(alarms)


def override_problems(config: AlarmConfig, table: Iterable[str]) -> list[str]:
    """Every name in ``disabled`` and the two override maps that ``table`` does not have.

    An override naming no alarm of the run does nothing, and doing nothing is exactly wrong when
    the name used to mean something: after a rename, ``disabled: ["imitation_handoff"]`` would
    switch the renamed alarm back on without a word. So it is a problem, and a renamed name says
    what it is called now.
    """
    known = set(table)
    problems: list[str] = []
    for where, named in (
        ("alarms.disabled", list(config.disabled)),
        ("alarms.patience_overrides", list(config.patience_overrides)),
        ("alarms.severity_overrides", list(config.severity_overrides)),
    ):
        for name in named:
            if name in known:
                continue
            renamed = RENAMED_ALARMS.get(name)
            hint = f"; it was renamed to {renamed!r}" if renamed is not None else ""
            problems.append(f"{where} names {name!r}, which is not in this run's alarm table{hint}")
    return problems


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
        extra: Iterable[Alarm] = (),
        printer: Callable[[str], None] | None = print,
    ) -> None:
        """``alarms`` replaces the core table; ``extra`` is appended to whichever table it is."""
        self.config = config
        self.printer = printer
        disabled = set(config.disabled)
        self.alarms: list[Alarm] = []
        base = alarms if alarms is not None else default_alarms(config, ratio_atol=ratio_atol)
        table = [*base, *extra]
        listed = names(table)
        # The counters are keyed by name, so a name twice would count one alarm twice a row:
        # half its patience and every firing doubled.
        doubled = sorted({name for name in listed if listed.count(name) > 1})
        if doubled:
            raise ValueError(f"the alarm table names these alarms more than once: {doubled}")
        problems = override_problems(config, listed)
        if problems:
            from ..errors import PreflightError

            raise PreflightError("\n".join(problems))
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
        on_fired: Callable[[list[AlarmResult]], None] | None = None,
    ) -> list[AlarmResult]:
        """Every alarm that fired on this row, and a halt if one of them was one.

        The whole table is evaluated before anything is raised, so the ``alarms.jsonl`` row of
        a halting iteration carries the other alarms that were firing beside it. That is
        usually where the cause is: a halt on the clip fraction beside a warning on the KL is a
        different story from a halt on the clip fraction alone.

        ``on_fired`` is how that reaches the file, and it is not optional decoration. The halt
        is raised from inside this method, so a caller that wrote the file from the RETURN value
        never wrote the halting iteration at all -- the one iteration anybody reading
        ``alarms.jsonl`` afterwards is looking for. It is called once, with every result, before
        ``on_halt``, so the checkpoint and the bundle that ``on_halt`` writes are assembled from
        a run directory that already has the alarms in it.
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
        if fired and on_fired is not None:
            on_fired(fired)
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
