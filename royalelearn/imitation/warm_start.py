"""The ``warm_start`` section: start the actor from saved weights, and hold it still at first.

Two things, both optional. ``init`` names an actor artifact: on a fresh start the actor is loaded
from it and made to reproduce the log-probabilities recorded on the artifact's own probe rows,
and the importance-ratio guard of preflight is measured on the loaded actor instead of predicted
from a seeded one. On a resume the checkpoint's weights win, and the measured guard runs on
them. ``actor_lr_scale`` schedules a multiplier on the actor's learning rate, zero freezing it;
the freeze itself is the core's, and this section only supplies its schedule, its two alarms and
their thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..extensions import ExtensionBase, RunContext
from .config import WarmStartSection, warm_start_problems

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.metrics import Alarm
    from ..config import RunConfig, ScheduleSpec
    from ..metrics.schema import SchemaContribution

__all__ = ["EXTENSION", "WarmStart"]


class WarmStart(ExtensionBase):
    name = "warm_start"
    format_version = 1
    section_type = WarmStartSection

    @property
    def package(self) -> Any:
        from .. import imitation

        return imitation

    def problems(self, section: WarmStartSection, config: RunConfig) -> list[str]:
        return warm_start_problems(section)

    def verify(self, section: WarmStartSection) -> list[str]:
        if section.init is None:
            return []
        from ..artifacts import verify_artifact
        from ..errors import PreflightError

        try:
            verify_artifact(section.init.path, section.init.sha256, what="warm_start.init")
        except PreflightError as exc:
            return [str(exc)]
        return []

    def identity_value(self, section: WarmStartSection) -> Any:
        """The init by content digest, not path, and the schedule; the alarms stay out."""
        import msgspec

        value = msgspec.to_builtins(section)
        value.pop("alarms", None)
        if value.get("init") is not None:
            value["init"].pop("path", None)
        return value

    def sets_starting_weights(self, section: WarmStartSection) -> bool:
        return section.init is not None

    def prepare(self, section: WarmStartSection, ctx: RunContext) -> Mapping[str, Any] | None:
        """Section 19.4, on a fresh start only: load, self-test, then the measured guard."""
        if section.init is None or ctx.resuming:
            return None
        from ..artifacts import read_actor_artifact
        from .init import initialise_actor, loaded_ratio_guard

        codec = ctx.row_codec()
        worst = initialise_actor(
            ctx.model,
            section.init,
            current=ctx.artifact_spec(),
            codec=codec,
            printer=ctx.printer,
        )
        probe = read_actor_artifact(section.init.path).probe
        assert probe is not None  # initialise_actor refuses an init without one
        predicted = loaded_ratio_guard(
            ctx.model,
            codec,
            probe.rows,
            atol=ctx.ratio_atol,
            precision=ctx.precision,
            say=ctx.printer,
        )
        return {"self_test": worst, "ratio_precision": predicted}

    def loaded(self, section: WarmStartSection, ctx: RunContext) -> Mapping[str, Any] | None:
        """A resumed warm-started run skipped preflight's predicted guard, so the guard is
        measured on the checkpoint's actor, on the init artifact's probe rows."""
        if section.init is None:
            return None
        from ..artifacts import read_actor_artifact
        from .init import loaded_ratio_guard

        probe = read_actor_artifact(Path(section.init.path)).probe
        assert probe is not None
        predicted = loaded_ratio_guard(
            ctx.model,
            ctx.row_codec(),
            probe.rows,
            atol=ctx.ratio_atol,
            precision=ctx.precision,
            say=ctx.printer,
        )
        return {"ratio_precision": predicted}

    def actor_lr_scale(self, section: WarmStartSection) -> ScheduleSpec | None:
        return section.actor_lr_scale

    def alarms(self, section: WarmStartSection) -> Sequence[Alarm]:
        if section.actor_lr_scale is None:
            return ()
        from ..learn.freeze import freeze_alarms

        thresholds = section.alarms
        return tuple(
            freeze_alarms(
                handoff_window=thresholds.handoff_window,
                handoff_kl=thresholds.handoff_kl,
                handoff_clip=thresholds.handoff_clip,
                ev_at_unfreeze=thresholds.ev_at_unfreeze,
            )
        )

    def metric_schema(self, section: WarmStartSection) -> SchemaContribution:
        from ..learn.freeze import FREEZE_ALARM_KEYS
        from ..metrics.schema import SchemaContribution

        if section.actor_lr_scale is None:
            return SchemaContribution()
        return SchemaContribution(alarm_metrics=FREEZE_ALARM_KEYS)


EXTENSION = WarmStart()
