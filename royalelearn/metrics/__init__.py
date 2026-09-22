"""Where a run's numbers go: the schema they are checked against, the sinks that write them,
and the alarms that read them.

Names resolve lazily, so that reading the schema -- which is what the documentation build and
the metric tests do -- does not import a sink, a socket or wandb.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "ALARM_METRICS": ".schema",
    "METRICS": ".schema",
    "PATTERNS": ".schema",
    "MetricSpec": ".schema",
    "is_known": ".schema",
    "lookup": ".schema",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'royalelearn.metrics' has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return list(__all__)
