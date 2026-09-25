"""A run with every optional part the core has today, for the tests of the whole alarm table.

One reference-KL regulariser named ``bc`` and a scheduled actor learning-rate scale: the run whose
alarm table is the core's plus the regularisers' family alarms plus the freeze's two. The alarm
tests check every alarm such a run can hold, not only the core's, and they get the table the way
a run does, through ``contributions.run_contributions``.
"""

from __future__ import annotations

from typing import Any

import msgspec

from royalelearn import config as cfg
from royalelearn.contributions import run_contributions
from royalelearn.metrics.alarms import default_alarms
from royalelearn.metrics.schema import RunSchema, for_run

BLOCK: dict[str, Any] = {
    "actor_lr_scale": {"kind": "constant", "value": 1.0},
    "references": {"ref": {"kind": "snapshot", "path": "ref", "sha256": "0" * 64}},
    "regularisers": [
        {
            "kind": "reference_kl",
            "name": "bc",
            "reference": "ref",
            "budget": {"kind": "constant", "value": 0.1},
            "coef": {"start": 1.0},
        }
    ],
}


def full_config(alarms: cfg.AlarmConfig | None = None) -> cfg.RunConfig:
    return msgspec.structs.replace(
        cfg.RunConfig(),
        imitation=msgspec.convert(BLOCK, type=cfg.ImitationConfig),
        alarms=alarms if alarms is not None else cfg.AlarmConfig(),
    )


def full_alarms(alarms: cfg.AlarmConfig | None = None) -> tuple[Any, ...]:
    """The core table with this run's contributed alarms after it, at ``alarms``' thresholds."""
    config = full_config(alarms)
    return default_alarms(config.alarms, extra=run_contributions(config).alarms)


def contributed_alarms(alarms: cfg.AlarmConfig | None = None) -> tuple[Any, ...]:
    return run_contributions(full_config(alarms)).alarms


def full_schema() -> RunSchema:
    return for_run(run_contributions(full_config()).schema)
