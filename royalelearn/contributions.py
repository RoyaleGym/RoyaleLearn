"""What a run's optional parts add to its schema and its alarm table.

Two parts exist today: the ``imitation`` block's regularisers, and the freeze that an actor
learning-rate schedule turns on. Each is asked only when the run has it, so a run with neither
has the core schema and the core alarm table and nothing else. This is the one place that knows
which parts there are; the schema, the alarms and the coordinator are handed what it returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .api.metrics import Alarm
    from .config import RunConfig
    from .metrics.schema import SchemaContribution

__all__ = ["RunContributions", "run_alarm_names", "run_contributions"]


class RunContributions(NamedTuple):
    schema: tuple[SchemaContribution, ...] = ()
    alarms: tuple[Alarm, ...] = ()


def run_contributions(config: RunConfig) -> RunContributions:
    """The schema contributions and the alarms of every optional part this run has."""
    imitation = config.imitation
    if imitation is None:
        return RunContributions()
    thresholds = config.alarms
    schema: list[SchemaContribution] = []
    alarms: list[Alarm] = []
    if imitation.regularisers:
        from .imitation.alarms import imitation_alarms
        from .imitation.schema import schema_contribution

        schema.append(schema_contribution())
        alarms.extend(
            imitation_alarms(
                ref_kl_warn=thresholds.imitation_ref_kl_warn,
                lambda_saturated_patience=thresholds.imitation_lambda_saturated_patience,
            )
        )
    if imitation.actor_lr_scale is not None:
        from .learn.freeze import FREEZE_ALARM_KEYS, freeze_alarms
        from .metrics.schema import SchemaContribution

        schema.append(SchemaContribution(alarm_metrics=FREEZE_ALARM_KEYS))
        alarms.extend(
            freeze_alarms(
                handoff_window=thresholds.imitation_handoff_window,
                handoff_kl=thresholds.imitation_handoff_kl,
                handoff_clip=thresholds.imitation_handoff_clip,
                ev_at_unfreeze=thresholds.imitation_ev_at_unfreeze,
            )
        )
    return RunContributions(tuple(schema), tuple(alarms))


def run_alarm_names(config: RunConfig) -> tuple[str, ...]:
    """Every alarm this run's table will hold, core first, in evaluation order."""
    from .metrics.alarms import default_alarms, names

    return names(default_alarms(config.alarms, extra=run_contributions(config).alarms))
