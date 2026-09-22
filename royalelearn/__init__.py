"""royalelearn -- the self-play training harness for RoyaleGym environments.

Layers (each user-facing behaviour is an ABC in ``api/`` with a shipped default):

    RolloutSource     api/rollout.py     ProcessRolloutSource, InlineRolloutSource
    ActorCritic       api/policy.py      SeparateActorCritic, SharedTrunkActorCritic
    ObsCodec          api/buffer.py      SpatialObsCodec
    ExperienceBuffer  api/buffer.py      RectBuffer
    Matchmaker/Rater  api/ladder.py      MixMatchmaker, BradleyTerryDavidsonRater
    MetricsSink       api/metrics.py     JsonlSink, ConsoleSink, WandbSink
    CheckpointStore   api/checkpoint.py  DirCheckpointStore

``LearningCoordinator`` (coordinator.py) is the only place the phases of an iteration are
ordered. ``docs/harness-spec.md`` specifies all of it; ``docs/design.md`` says why.

Importing this package pulls in nothing that needs torch: the ABCs use numpy and msgspec,
and the concrete learner lives behind the lazy attributes below. That is what lets
``royalelearn config``, ``royalelearn identity`` and ``royalelearn --help`` work in an
environment where torch is not installed. Asking for a name that does need torch raises an
ImportError naming the package that is missing.
"""

from __future__ import annotations

import importlib
from typing import Any

from .version import __version__, git_describe

# Public name -> (module, attribute). A None attribute exports the module itself.
# Resolution is deferred so that the import cost and the torch dependency of a name are
# paid only by the caller that asks for it.
_EXPORTS: dict[str, tuple[str, str | None]] = {
    "LearningCoordinator": (".coordinator", "LearningCoordinator"),
    "RunConfig": (".config", "RunConfig"),
    "load_config": (".config", "load_config"),
    "api": (".api", None),
}

__all__ = ["__version__", "git_describe", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Resolve a public name on first use.

    An unknown name is an AttributeError. A known name whose module cannot be imported
    raises that import's own ImportError unchanged, because its message already names what
    is missing and rewording it would hide which package to install.
    """
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'royalelearn' has no attribute {name!r}") from None
    module = importlib.import_module(module_name, __name__)
    return module if attribute is None else getattr(module, attribute)


def __dir__() -> list[str]:
    return sorted(__all__)
