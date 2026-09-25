"""The metric keys the reference-KL regularisers publish (section 19.9).

Named by the run's own config, one family per regulariser, so they are patterns: ``{name}`` is
the regulariser's name. They are this add-on's contribution to the schema of a run that has the
block, and a run without it does not know them.
"""

from __future__ import annotations

from ..metrics.schema import MetricSpec, SchemaContribution, pattern

__all__ = ["IMITATION_ALARM_METRICS", "IMITATION_PATTERNS", "schema_contribution"]


def _m(unit: str, description: str, **kwargs: object) -> MetricSpec:
    return MetricSpec(unit=unit, description=description, **kwargs)  # type: ignore[arg-type]


IMITATION_PATTERNS = (
    pattern(
        "imitation/{name}/kl",
        _m(
            "nats",
            "KL(reference || policy), the mean over the choice rows the regulariser covered "
            "in epoch 1. The number lambda is moved by. Absent when no row was covered.",
        ),
    ),
    pattern(
        "imitation/{name}/kl_noop",
        _m("nats", "The play/wait part of kl, by the chain rule."),
    ),
    pattern(
        "imitation/{name}/kl_card",
        _m("nats", "p_ref(play) times the KL of the card given a play. Joint factor only."),
    ),
    pattern(
        "imitation/{name}/kl_tile",
        _m("nats", "The reference-weighted KL of the tile given the card. Joint factor only."),
    ),
    pattern(
        "imitation/{name}/lambda",
        _m("coefficient", "The coefficient this iteration's loss used.", low=0.0),
    ),
    pattern(
        "imitation/{name}/lambda_at_max",
        _m(
            "flag",
            "1 when the coefficient this iteration used was coef.max: the anchor is pulling as "
            "hard as it is allowed to.",
        ),
    ),
    pattern(
        "imitation/{name}/budget",
        _m("nats", "The budget kl is held to this iteration.", low=0.0),
    ),
    pattern(
        "imitation/{name}/grad_ratio",
        _m(
            "ratio",
            "||grad of the unscaled KL|| / ||grad of the policy term|| on the first minibatch "
            "with a choice row. Zero while the policy equals the reference. For setting "
            "coef.start.",
        ),
    ),
    pattern(
        "imitation/{name}/rows_frac",
        _m("fraction", "Share of epoch-1 choice rows the regulariser covered (exclude_when)."),
    ),
    pattern(
        "imitation/{name}/top1_agree",
        _m(
            "fraction",
            "Share of covered rows where the reference's and the policy's most likely actions "
            "agree; under noop_marginal, whether both would play.",
        ),
    ),
    pattern(
        "imitation/{name}/ref_p_noop",
        _m("probability", "The reference's mean p(no-op) over the covered rows."),
    ),
)

#: A template names a family: the alarm reads every key of it the row carries.
IMITATION_ALARM_METRICS: dict[str, tuple[str, ...]] = {
    "imitation_ref_kl_high": ("imitation/{name}/kl",),
    "imitation_lambda_saturated": ("imitation/{name}/lambda_at_max",),
}


def schema_contribution() -> SchemaContribution:
    return SchemaContribution(
        patterns=IMITATION_PATTERNS, alarm_metrics=IMITATION_ALARM_METRICS
    )
