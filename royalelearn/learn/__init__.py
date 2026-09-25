"""The learner: the networks, the masked distribution, and everything that touches a device.

Every module under here imports torch. Names resolve lazily so that importing the package, the
config tree or the CLI's read-only commands costs nothing and works in an environment where torch
is not installed; asking for one of these names without torch raises that import's own error,
which already says which package is missing.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "AdvantageInputs": ".buffer",
    "BackoffState": ".schedules",
    "Batch": ".buffer",
    "batch_count": ".buffer",
    "empty_obs": ".buffer",
    "BatchedInference": ".inference",
    "BehaviourSnapshot": ".actor_critic",
    "ClashActor": ".actor_critic",
    "ClashCritic": ".actor_critic",
    "ClashTrunk": ".nets",
    "Constant": ".schedules",
    "DefaultNetworkFactory": ".nets",
    "GAE": ".gae",
    "Geometric": ".schedules",
    "InferenceResult": ".inference",
    "Linear": ".schedules",
    "LrBackoff": ".schedules",
    "MaskedCategorical": ".distribution",
    "Minibatch": ".buffer",
    "PPOUpdate": ".ppo",
    "PiecewiseConstant": ".schedules",
    "PointerPolicyHead": ".nets",
    "RectBuffer": ".buffer",
    "RectGather": ".inference",
    "ResBlock": ".nets",
    "RoundStats": ".inference",
    "ScheduleSet": ".schedules",
    "SeparateActorCritic": ".actor_critic",
    "SharedTrunkActorCritic": ".actor_critic",
    "ValueHead": ".nets",
    "WelfordReturnScaler": ".returns",
    "WelfordState": ".returns",
    "adam_eps_floor_frac": ".ppo",
    "approx_kl": ".ppo",
    "autocast_context": ".actor_critic",
    "bootstrap_values": ".gae",
    "build_schedule": ".schedules",
    "chunked_critic_pass": ".ppo",
    "clipped_fraction": ".ppo",
    "discounted_returns": ".returns",
    "dual_clipped_fraction": ".ppo",
    "explained_variance": ".ppo",
    "frozen_dtype": ".inference",
    "gae_recursion": ".gae",
    "reference_gae": ".gae",
    "resolve_dtype": ".nets",
    "standardise": ".ppo",
    "surrogate": ".ppo",
    "value_error": ".ppo",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'royalelearn.learn' has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return list(__all__)
