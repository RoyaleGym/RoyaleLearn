"""RoyaleLearn: the training harness for RoyaleGym environments (RLGym-PPO's role).

The harness itself is not written yet. The six modules beside this file
(continuous_policy, discrete_policy, multi_discrete_policy, value_estimator,
experience_buffer, ppo_learner) are rlgym_ppo's ``ppo/`` subpackage copied as a seed;
they import ``torch`` and ``rlgym_ppo`` and do not run here. This ``__init__`` imports
none of them, so ``import royalelearn`` works in the shared workspace venv (no torch,
no rlgym_ppo); asking for one of their classes raises an ImportError that names the
missing package.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

# The vendored modules and the class each one defines, in rlgym_ppo's own names.
SEED_CLASSES: dict[str, str] = {
    "ContinuousPolicy": "continuous_policy",
    "DiscreteFF": "discrete_policy",
    "MultiDiscreteFF": "multi_discrete_policy",
    "ValueEstimator": "value_estimator",
    "ExperienceBuffer": "experience_buffer",
    "PPOLearner": "ppo_learner",
}

__all__ = ["SEED_CLASSES", "__version__"]


def __getattr__(name: str) -> object:
    """Lazy access to the seed classes, e.g. ``royalelearn.PPOLearner``.

    The submodule is imported only now, so the package itself never needs torch or
    rlgym_ppo. A missing dependency surfaces as an ImportError that says which one.
    """
    module_name = SEED_CLASSES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'royalelearn' has no attribute {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
    except ImportError as exc:
        missing = exc.name or "a dependency"
        raise ImportError(
            f"royalelearn.{name} lives in the vendored rlgym_ppo seed "
            f"royalelearn/{module_name}.py, which needs {missing!r} (not installed). "
            "The seed is not wired to RoyaleGym; see README.md.",
            name=exc.name,
        ) from exc
    return getattr(module, name)
