"""Extensions: optional parts of a run that another package provides.

A run's config is RoyaleLearn's own keys plus, for each extension the run uses, one top-level
section named after it. The key is the extension's name, and the name is how it is found: an
installed package declares it under the ``royalelearn.extensions`` entry-point group, and
``load_config`` asks for the providers of exactly the keys the core does not own. A key no
provider claims is refused by name, so a misspelt section is still an error, and a section that
is ``null`` counts as absent.

An extension that is installed and not named by the config is never imported. It cannot move a
config, an identity, a schema, an alarm table or a state digest, and its uncommitted edits cannot
stop a run that does not use it.

What an extension can do is the ``Extension`` protocol and nothing else: check its section, name
the files it depends on, set the actor's starting weights, schedule the actor's learning-rate
scale, add terms to the actor's loss, and contribute alarms and metric keys. ``__all__`` is the
whole surface a package built on RoyaleLearn may import, and every name in it outside the
registry resolves lazily, so importing this module imports nothing else of RoyaleLearn -- not
torch, and not the modules that import this one.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from .errors import PreflightError

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .api.metrics import Alarm
    from .api.rollout import EnvSpec
    from .api.update import ActorLossTerm
    from .config import RunConfig, ScheduleSpec
    from .metrics.schema import SchemaContribution

ENTRY_POINT_GROUP = "royalelearn.extensions"
#: Raised whenever the protocol changes in a way an existing extension would notice.
EXTENSION_API_VERSION = 1


class RunContext(NamedTuple):
    """What an extension is handed about the run it is part of, once the model exists."""

    config: RunConfig
    spec: EnvSpec
    device: Any
    printer: Callable[[str], None]
    #: The live actor-critic, for an extension that sets its starting weights.
    model: Any
    #: A bare actor of this run's architecture on a device, for a frozen reference.
    build_actor: Callable[[Any], Any]
    #: What an actor artifact this run can load must agree with.
    artifact_spec: Callable[[], Any]
    #: The codec's row unpacking on the learner's device.
    row_codec: Callable[[], Any]
    #: True when the run resumes a checkpoint, whose weights win over any starting weights.
    resuming: bool
    #: The importance-ratio tolerance at this run's precision, and that precision's name.
    ratio_atol: float
    precision: str


class Extension(Protocol):
    """One optional part of a run, owning one top-level config key.

    ``name`` is that key and the entry-point name. ``package`` is the extension's top-level
    package: its folder is watched for uncommitted edits and its commit and ``__version__`` are
    recorded in the run's identity. ``section_type`` is decoded with unknown fields refused.
    """

    name: str
    api_version: int
    format_version: int
    package: ModuleType
    section_type: type

    def problems(self, section: Any, config: RunConfig) -> list[str]: ...

    def verify(self, section: Any) -> list[str]: ...

    def identity_value(self, section: Any) -> Any: ...

    def sets_starting_weights(self, section: Any) -> bool: ...

    def prepare(self, section: Any, ctx: RunContext) -> Mapping[str, Any] | None: ...

    def loaded(self, section: Any, ctx: RunContext) -> Mapping[str, Any] | None: ...

    def actor_lr_scale(self, section: Any) -> ScheduleSpec | None: ...

    def actor_terms(self, section: Any, ctx: RunContext) -> Sequence[ActorLossTerm]: ...

    def alarms(self, section: Any) -> Sequence[Alarm]: ...

    def metric_schema(self, section: Any) -> SchemaContribution: ...


class ExtensionBase:
    """The protocol with every hook doing nothing, so an extension writes only what it uses."""

    api_version = EXTENSION_API_VERSION
    format_version = 1

    def problems(self, section: Any, config: RunConfig) -> list[str]:
        """What is wrong with the section that can be seen without building anything."""
        return []

    def verify(self, section: Any) -> list[str]:
        """Problems with the files the section names, checked on every start, fresh or resumed,
        before preflight."""
        return []

    def identity_value(self, section: Any) -> Any:
        """What of the section is the experiment: by default all of it but its ``alarms``.

        An extension whose section names files overrides this to leave out each path and keep
        the file's content digest, so a folder moved to another disk is not a new experiment.
        """
        import msgspec

        value = msgspec.to_builtins(section)
        if isinstance(value, dict):
            value.pop("alarms", None)
        return value

    def sets_starting_weights(self, section: Any) -> bool:
        """Whether ``prepare`` replaces the seeded actor's weights on a fresh start."""
        return False

    def prepare(self, section: Any, ctx: RunContext) -> Mapping[str, Any] | None:
        """Once, after the model exists and before the rollout buffer does, fresh or resumed.

        What it returns is kept on the coordinator under the extension's name, for a report."""
        return None

    def loaded(self, section: Any, ctx: RunContext) -> Mapping[str, Any] | None:
        """Once, on a resume, after the checkpoint's weights are in the model."""
        return None

    def actor_lr_scale(self, section: Any) -> ScheduleSpec | None:
        """A schedule the actor's learning rate is multiplied by; zero freezes the actor."""
        return None

    def actor_terms(self, section: Any, ctx: RunContext) -> Sequence[ActorLossTerm]:
        return ()

    def alarms(self, section: Any) -> Sequence[Alarm]:
        return ()

    def metric_schema(self, section: Any) -> SchemaContribution:
        from .metrics.schema import SchemaContribution

        return SchemaContribution()


class Provider(NamedTuple):
    """An extension, and where it came from."""

    extension: Any
    #: The distribution that declared the entry point; None for one provided in code.
    distribution: str | None
    #: Provided by RoyaleLearn itself, and so named by ``royalelearn_git`` in the identity.
    builtin: bool


class Active(NamedTuple):
    """One extension a run uses, with its section."""

    name: str
    extension: Any
    section: Any
    distribution: str | None
    builtin: bool


#: Sections RoyaleLearn provides in code rather than through an entry point, as
#: ``module:attribute``. Install metadata can be stale -- an in-tree egg-info with no entry points
#: at all -- so what the core itself provides does not depend on it.
#:
#: TRANSITIONAL: both live in ``royalelearn.imitation`` until it becomes its own package, when
#: they leave this table for that package's entry points.
_BUILTINS: dict[str, str] = {
    "imitation": "royalelearn.imitation.extension:EXTENSION",
    "warm_start": "royalelearn.imitation.warm_start:EXTENSION",
}


def _load(reference: str) -> Any:
    module_name, _, attribute = reference.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


def _declared(names: set[str]) -> dict[str, dict[str, set[str]]]:
    """``{key: {entry-point value: {distribution, ...}}}`` for every installed declaration.

    Every distribution is walked, duplicates included: ``importlib.metadata.entry_points`` keeps
    only the first copy of a distribution seen twice, and which copy is first depends on the
    working directory, so a stale in-tree egg-info could win from one folder and not another.
    """
    found: dict[str, dict[str, set[str]]] = {}
    for distribution in importlib.metadata.distributions():
        try:
            points = distribution.entry_points
            owner = distribution.metadata["Name"] or "?"
        except Exception:  # pragma: no cover - a broken metadata folder
            continue
        for point in points:
            if point.group == ENTRY_POINT_GROUP and point.name in names:
                found.setdefault(point.name, {}).setdefault(point.value, set()).add(owner)
    return found


def _discover(names: set[str]) -> tuple[dict[str, Provider], list[str]]:
    providers: dict[str, Provider] = {}
    problems: list[str] = []
    declared = _declared(names - set(_BUILTINS))
    for key in sorted(names):
        if key in _BUILTINS:
            providers[key] = Provider(_load(_BUILTINS[key]), None, True)
            continue
        values = declared.get(key, {})
        if not values:
            problems.append(
                f"{key!r} is not a RoyaleLearn config key, and no installed package provides it "
                f"as a section (entry-point group {ENTRY_POINT_GROUP!r})"
            )
            continue
        if len(values) > 1:
            problems.append(
                f"{key!r} is provided by more than one installed package: "
                + "; ".join(f"{value} ({', '.join(sorted(d))})" for value, d in values.items())
            )
            continue
        ((value, owners),) = values.items()
        providers[key] = Provider(_load(value), sorted(owners)[0], False)
    return providers, problems


def providers_for(keys: Iterable[str]) -> dict[str, Provider]:
    """The provider of each of ``keys``, loading nothing else.

    Refused, by key, all together: a key nothing provides, two different providers for one key,
    an extension at another API version, and one whose declared name is not its key.
    """
    wanted = set(keys)
    if not wanted:
        return {}
    providers, problems = _discover(wanted)
    for key, provider in sorted(providers.items()):
        extension = provider.extension
        version = getattr(extension, "api_version", None)
        if version != EXTENSION_API_VERSION:
            problems.append(
                f"{key!r} is provided at extension API version {version}; this RoyaleLearn is at "
                f"{EXTENSION_API_VERSION}"
            )
        elif getattr(extension, "name", None) != key:
            problems.append(
                f"{key!r} is provided by an extension that calls itself "
                f"{getattr(extension, 'name', None)!r}"
            )
    if problems:
        raise PreflightError("\n".join(problems))
    return providers


@cache
def _config_type(table: tuple[tuple[str, Provider], ...]) -> type:
    """RunConfig with one field per section, in name order after the core's. Cached, so every
    config with the same sections has the same type."""
    import msgspec

    from .config import RunConfig

    if not table:
        return RunConfig
    return msgspec.defstruct(
        "RunConfig",
        [(name, provider.extension.section_type) for name, provider in table],
        bases=(RunConfig,),
        kw_only=True,
        forbid_unknown_fields=True,
        namespace={"_extensions": MappingProxyType(dict(table))},
        module=RunConfig.__module__,
    )


def config_type(providers: Mapping[str, Provider]) -> type:
    """The config type of a run with these sections; ``RunConfig`` itself for none."""
    return _config_type(tuple((name, providers[name]) for name in sorted(providers)))


def build_config(core: Mapping[str, Any], sections: Mapping[str, Any]) -> RunConfig:
    """A config from the core's fields as builtins and the sections as builtins; a ``None``
    section is left out."""
    import msgspec

    present = {name: value for name, value in sections.items() if value is not None}
    cls = config_type(providers_for(present))
    return msgspec.convert({**core, **present}, type=cls)


def with_sections(config: RunConfig, **sections: Any) -> RunConfig:
    """``config`` with sections set -- a mapping or the section itself -- or removed with None."""
    import msgspec

    data = msgspec.to_builtins(config)
    current = {active.name: data.pop(active.name) for active in active_extensions(config)}
    current.update({name: msgspec.to_builtins(value) for name, value in sections.items()})
    return build_config(data, current)


def active_extensions(config: Any) -> tuple[Active, ...]:
    """Each extension this run uses with its section, in name order. Empty for a plain config."""
    table: Mapping[str, Provider] = getattr(type(config), "_extensions", {})
    return tuple(
        Active(name, provider.extension, section, provider.distribution, provider.builtin)
        for name, provider in sorted(table.items())
        if (section := getattr(config, name, None)) is not None
    )


def extension_problems(config: Any) -> list[str]:
    """Every active section's own problems, and the ones only the set of them can have."""
    problems: list[str] = []
    schedulers = []
    for active in active_extensions(config):
        problems.extend(active.extension.problems(active.section, config))
        if active.extension.actor_lr_scale(active.section) is not None:
            schedulers.append(active.name)
    if len(schedulers) > 1:
        problems.append(
            f"more than one section schedules the actor's learning-rate scale: {schedulers}"
        )
    return problems


def actor_lr_scale_of(config: Any) -> ScheduleSpec | None:
    """The one schedule of the actor's learning-rate scale this run has, or None."""
    for active in active_extensions(config):
        schedule = active.extension.actor_lr_scale(active.section)
        if schedule is not None:
            return schedule
    return None


def extension_alarms(config: Any) -> tuple[Alarm, ...]:
    return tuple(
        alarm
        for active in active_extensions(config)
        for alarm in active.extension.alarms(active.section)
    )


def schema_contributions(config: Any) -> tuple[SchemaContribution, ...]:
    return tuple(
        active.extension.metric_schema(active.section) for active in active_extensions(config)
    )


# --------------------------------------------------------------------------
# The supported surface: what a package built on RoyaleLearn may import, from here.
# --------------------------------------------------------------------------

_SURFACE: dict[str, str] = {
    # artifacts: actor weights with a digest and probe rows
    "PROBE_CHUNK": "royalelearn.artifacts",
    "PROBE_NAME": "royalelearn.artifacts",
    "ActorArtifact": "royalelearn.artifacts",
    "ProbeSet": "royalelearn.artifacts",
    "artifact_digest": "royalelearn.artifacts",
    "check_actor_artifact": "royalelearn.artifacts",
    "load_actor_state": "royalelearn.artifacts",
    "probe_log_probs": "royalelearn.artifacts",
    "read_actor_artifact": "royalelearn.artifacts",
    "self_test": "royalelearn.artifacts",
    "verify_artifact": "royalelearn.artifacts",
    "write_actor_artifact": "royalelearn.artifacts",
    # the update's hook and the freeze
    "ActorLossTerm": "royalelearn.api.update",
    "ActorTermInputs": "royalelearn.api.update",
    "FREEZE_ALARM_KEYS": "royalelearn.learn.freeze",
    "freeze_alarms": "royalelearn.learn.freeze",
    # observations, rows and the environment
    "ObsBatch": "royalelearn.api.policy",
    "EnvSpec": "royalelearn.api.rollout",
    "RowCodec": "royalelearn.learn.rows",
    "empty_obs": "royalelearn.learn.buffer",
    "field_slice": "royalelearn.obs_layout",
    "build_env": "royalelearn.rollout.envspec",
    "behaviour_fields": "royalelearn.metrics.behaviour",
    # schedules and config
    "RunConfig": "royalelearn.config",
    "ScheduleSpec": "royalelearn.config",
    "ConstantSpec": "royalelearn.config",
    "PiecewiseConstantSpec": "royalelearn.config",
    "build_schedule": "royalelearn.learn.schedules",
    "load_config": "royalelearn.config",
    # identity and provenance
    "action_digest_of": "royalelearn.identity",
    "package_provenance": "royalelearn.identity",
    "section_digest": "royalelearn.identity",
    # precision
    "judge_ratio_precision": "royalelearn.rollout.preflight",
    "precision_bound": "royalelearn.rollout.preflight",
    # snapshots
    "SnapshotSpec": "royalelearn.ladder.snapshots",
    # alarms and metrics
    "Alarm": "royalelearn.api.metrics",
    "FamilyAlarm": "royalelearn.metrics.alarms",
    "MetricAlarm": "royalelearn.metrics.alarms",
    "MetricSpec": "royalelearn.metrics.schema",
    "SchemaContribution": "royalelearn.metrics.schema",
    "pattern": "royalelearn.metrics.schema",
    # errors
    "PreflightError": "royalelearn.errors",
}

__all__ = sorted(
    [
        "ENTRY_POINT_GROUP",
        "EXTENSION_API_VERSION",
        "Active",
        "Extension",
        "ExtensionBase",
        "Provider",
        "RunContext",
        "active_extensions",
        "actor_lr_scale_of",
        "build_config",
        "config_type",
        "extension_alarms",
        "extension_problems",
        "providers_for",
        "schema_contributions",
        "with_sections",
        *_SURFACE,
    ]
)


def __getattr__(name: str) -> Any:
    module_name = _SURFACE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return list(__all__)
