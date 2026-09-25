"""A run with every section the core provides today, for the tests of the whole alarm table.

One reference-KL regulariser named ``bc`` under ``imitation`` and a scheduled actor learning-rate
scale under ``warm_start``: the run whose alarm table is the core's plus the regularisers' family
alarms plus the freeze's two. The alarm tests check every alarm such a run can hold, not only the
core's, and they get the table the way a run does, through its active sections.
"""

from __future__ import annotations

from typing import Any

from royalelearn import config as cfg
from royalelearn.extensions import extension_alarms, schema_contributions, with_sections
from royalelearn.metrics.alarms import default_alarms
from royalelearn.metrics.schema import RunSchema, for_run

IMITATION: dict[str, Any] = {
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
WARM_START: dict[str, Any] = {"actor_lr_scale": {"kind": "constant", "value": 1.0}}


def full_config(alarms: cfg.AlarmConfig | None = None) -> cfg.RunConfig:
    core = cfg.RunConfig(alarms=alarms if alarms is not None else cfg.AlarmConfig())
    return with_sections(core, imitation=IMITATION, warm_start=WARM_START)


def full_alarms(alarms: cfg.AlarmConfig | None = None) -> tuple[Any, ...]:
    """The core table at ``alarms``' thresholds, with the sections' alarms after it."""
    config = full_config(alarms)
    return default_alarms(config.alarms, extra=extension_alarms(config))


def contributed_alarms(alarms: cfg.AlarmConfig | None = None) -> tuple[Any, ...]:
    """The sections' alarms, at the sections' default thresholds."""
    return extension_alarms(full_config(alarms))


def full_schema() -> RunSchema:
    return for_run(schema_contributions(full_config()))
