"""The ladder: the rating, who plays whom, what a gate decides, and where the evidence lives.

Names resolve lazily, so that reading a result log -- which is what the rating tests and any
offline analysis do -- does not import an environment, a snapshot store or torch.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "Aggregate": ".results",
    "BradleyTerryDavidsonRater": ".rating",
    "Comparison": ".evaluate",
    "DiskSnapshotStore": ".snapshots",
    "EloReadout": ".rating",
    "EvalRunner": ".evaluate",
    "GameResult": ".results",
    "HallOfFameEviction": ".eviction",
    "LEARNER_ID": ".pool",
    "LadderPool": ".pool",
    "MixMatchmaker": ".matchmaker",
    "PairRecord": ".results",
    "PoolState": ".pool",
    "ResultLog": ".results",
    "ResultView": ".results",
    "SCRIPTED_IDS": ".pool",
    "SCRIPTED_NOOP": ".pool",
    "SCRIPTED_RANDOM_LEGAL": ".pool",
    "SeedSet": ".evaluate",
    "SnapshotSpec": ".snapshots",
    "WilsonGate": ".gate",
    "context_digest": ".results",
    "eval_seed_set": ".evaluate",
    "failed_condition": ".gate",
    "floor_decision": ".gate",
    "pfsp_shape": ".matchmaker",
    "wilson_interval": ".rating",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'royalelearn.ladder' has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return list(__all__)
