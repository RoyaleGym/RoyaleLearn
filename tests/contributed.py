"""The alarm table of a run whose section schedules the actor's learning-rate scale.

The core's table, and the freeze's two alarms after it: a section that schedules the actor's rate
turns the core's freeze on and brings them. The core tests check every alarm such a run can hold.
The regularisers' family alarms belong to the package that provides them and are tested there.
"""

from __future__ import annotations

from typing import Any

from royalelearn import config as cfg
from royalelearn.extensions import with_sections
from royalelearn.learn.freeze import FREEZE_ALARM_KEYS, freeze_alarms
from royalelearn.metrics.alarms import default_alarms
from royalelearn.metrics.schema import RunSchema, SchemaContribution, for_run
from royalelearn.testing import FREEZE_THRESHOLDS, StubExtension, use_extensions

#: The stub section a test schedules the actor's rate with.
SCHEDULE: dict[str, Any] = {"actor_lr_scale": {"kind": "constant", "value": 1.0}}


def contributed_alarms(alarms: cfg.AlarmConfig | None = None) -> tuple[Any, ...]:
    """The freeze's alarms, at the thresholds a section uses by default."""
    return tuple(
        freeze_alarms(
            handoff_window=int(FREEZE_THRESHOLDS["handoff_window"]),
            handoff_kl=FREEZE_THRESHOLDS["handoff_kl"],
            handoff_clip=FREEZE_THRESHOLDS["handoff_clip"],
            ev_at_unfreeze=FREEZE_THRESHOLDS["ev_at_unfreeze"],
        )
    )


def full_alarms(alarms: cfg.AlarmConfig | None = None) -> tuple[Any, ...]:
    """The core table at ``alarms``' thresholds, with the freeze's alarms after it."""
    return default_alarms(alarms or cfg.AlarmConfig(), extra=contributed_alarms())


def full_schema() -> RunSchema:
    return for_run([SchemaContribution(alarm_metrics=FREEZE_ALARM_KEYS)])


def full_config(
    monkeypatch: Any, alarm_config: cfg.AlarmConfig | None = None, **section: Any
) -> Any:
    """A config whose ``freezer`` section schedules the actor's rate, the way a real section
    would: registered as an installed extension for this one test. ``section`` adds to the
    section, its own ``alarms`` thresholds included; ``alarm_config`` is the core's."""
    use_extensions(monkeypatch, {"freezer": StubExtension("freezer")})
    core = cfg.RunConfig(alarms=alarm_config if alarm_config is not None else cfg.AlarmConfig())
    return with_sections(core, freezer={**SCHEDULE, **section})
