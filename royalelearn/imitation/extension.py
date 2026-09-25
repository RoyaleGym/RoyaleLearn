"""The ``imitation`` section: anchor the policy to reference policies (sections 19.6-19.9).

References are actor artifacts or field models named by path and content digest. Each
regulariser is an actor-loss term (``regularisers.ReferenceKL``) with its own adaptive
coefficient, and brings its metric family and two alarms into the run's schema and alarm table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..extensions import ExtensionBase, RunContext
from .config import ImitationSection, imitation_problems

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.metrics import Alarm
    from ..api.update import ActorLossTerm
    from ..config import RunConfig
    from ..metrics.schema import SchemaContribution

__all__ = ["EXTENSION", "Imitation", "verify_references"]


def verify_references(section: ImitationSection) -> list[str]:
    """Every reference folder checked against its stated digest, all before any refusal, so a
    config with two stale digests is told about both at once."""
    from ..artifacts import verify_artifact
    from ..errors import PreflightError

    problems: list[str] = []
    for name, reference in sorted(section.references.items()):
        try:
            verify_artifact(reference.path, reference.sha256, what=f"imitation.references.{name}")
        except PreflightError as exc:
            problems.append(str(exc))
    return problems


class Imitation(ExtensionBase):
    name = "imitation"
    format_version = 1
    section_type = ImitationSection

    @property
    def package(self) -> Any:
        from .. import imitation

        return imitation

    def problems(self, section: ImitationSection, config: RunConfig) -> list[str]:
        return imitation_problems(section)

    def verify(self, section: ImitationSection) -> list[str]:
        return verify_references(section)

    def identity_value(self, section: ImitationSection) -> Any:
        """The section by value with every reference named by digest, not path; alarms out."""
        import msgspec

        value = msgspec.to_builtins(section)
        value.pop("alarms", None)
        for reference in value.get("references", {}).values():
            reference.pop("path", None)
        return value

    def actor_terms(self, section: ImitationSection, ctx: RunContext) -> Sequence[ActorLossTerm]:
        if not section.regularisers:
            return ()
        from .references import build_references
        from .regularisers import build_terms

        references = build_references(
            section,
            spec=ctx.spec,
            current=ctx.artifact_spec(),
            build_actor=ctx.build_actor,
            codec=ctx.row_codec(),
            device=ctx.device,
            printer=ctx.printer,
        )
        return build_terms(section, references, ctx.spec)

    def alarms(self, section: ImitationSection) -> Sequence[Alarm]:
        if not section.regularisers:
            return ()
        from .alarms import imitation_alarms

        return tuple(
            imitation_alarms(
                ref_kl_warn=section.alarms.ref_kl_warn,
                lambda_saturated_patience=section.alarms.lambda_saturated_patience,
            )
        )

    def metric_schema(self, section: ImitationSection) -> SchemaContribution:
        from ..metrics.schema import SchemaContribution
        from .schema import schema_contribution

        if not section.regularisers:
            return SchemaContribution()
        return schema_contribution()


EXTENSION = Imitation()
