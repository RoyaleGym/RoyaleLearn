"""The regularisers' alarms (section 19.9). Both warn: a halted treatment run is censored out of
its comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..metrics.alarms import FamilyAlarm
from .schema import IMITATION_ALARM_METRICS

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.metrics import Alarm

__all__ = ["imitation_alarms"]


def imitation_alarms(*, ref_kl_warn: float, lambda_saturated_patience: int) -> list[Alarm]:
    return [
        FamilyAlarm(
            "imitation_ref_kl_high",
            lambda _member, kl: kl > ref_kl_warn,
            keys=IMITATION_ALARM_METRICS["imitation_ref_kl_high"],
            meaning="the policy has moved far from a reference it is anchored to",
        ),
        FamilyAlarm(
            "imitation_lambda_saturated",
            lambda _member, at_max: at_max >= 1.0,
            keys=IMITATION_ALARM_METRICS["imitation_lambda_saturated"],
            patience=lambda_saturated_patience,
            meaning=(
                "lambda has sat at coef.max: the reward is pulling harder than the anchor can "
                "hold, which is a statement about the reward"
            ),
        ),
    ]
