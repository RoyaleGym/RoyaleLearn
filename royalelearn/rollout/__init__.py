"""The rollout side: the env description, the shared-memory byte layout, and the workers.

Names resolve lazily. ``api/rollout.py`` imports ``EnvFactorySpec`` from here, and
``layout.py`` imports ``EnvSpec`` from there, so an eager re-export in this file would close
that loop at import time; resolving on first use keeps both directions working whichever module
a caller imports first, and keeps the cost of importing the env description down to msgspec.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "ComponentSpec": ".envspec",
    "EnvFactorySpec": ".envspec",
    "read_env_spec": ".envspec",
    "BufferHandle": ".layout",
    "BufferLayout": ".layout",
    "ControlHandle": ".layout",
    "ControlLayout": ".layout",
    "LAYOUT_VERSION": ".layout",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'royalelearn.rollout' has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return list(__all__)
