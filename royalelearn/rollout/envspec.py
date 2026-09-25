"""The environment as JSON: what a worker is handed, what a checkpoint records, and what the
ladder's context is hashed from.

An ``EnvFactorySpec`` is a description, not an env. It crosses to a worker as msgspec bytes
rather than as a pickled closure, which is what removes the reference learner's undocumented
4096-byte factory limit and gives a checkpoint its environment description for free. Building
one goes through RoyaleGym's own picklable ``EnvFactory`` recipe: one place knows how a
``ClashParallelEnv`` is assembled and how to keep it picklable under ``spawn``, and a second
assembly here would be a copy that drifts.

Only classes on the allow-list may be instantiated. A spec arrives from a config file and later
from a checkpoint, and ``importlib`` on an arbitrary string is a code-execution surface.

royalegym is imported inside the functions that build things rather than at module scope, so
that reading, hashing and printing a spec -- which is what the CLI's ``config`` and ``identity``
commands do -- costs nothing but msgspec.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from typing import TYPE_CHECKING, Any

import msgspec

from ..errors import PreflightError, StaleEngineBuild

if TYPE_CHECKING:  # pragma: no cover - annotations only; see the module docstring
    from royalegym.env import ClashSelfPlayVecEnv

    from ..api.rollout import EnvSpec

__all__ = [
    "ALLOWED_MODULE_PREFIXES",
    "ComponentSpec",
    "EnvFactorySpec",
    "JsonValue",
    "canonical_json",
    "digest_of",
    "env_value_digest",
    "read_env_spec",
    "resolve_component",
]

#: Anything a ComponentSpec may name, before ``config.extra_component_modules`` is added.
ALLOWED_MODULE_PREFIXES: tuple[str, ...] = ("royalegym.", "royalelearn.")

#: What may appear in a ComponentSpec's kwargs. The top level is typed and the nested containers
#: are not, because a recursive alias is not something msgspec can build a decoder for; what
#: proves the whole value is JSON-able is that every spec is encoded -- for the digest, for the
#: checkpoint and for the worker -- before anything is built from it.
JsonValue = bool | int | float | str | list | dict | None

_ENCODER = msgspec.json.Encoder(order="deterministic")


def canonical_json(value: Any) -> bytes:
    """One encoding of a value, with mapping keys in a fixed order.

    Every digest in the harness is taken over this, so two configs that differ only in the order
    their JSON was written produce the same hash and a run is not re-identified by a reformat.
    """
    return _ENCODER.encode(value)


#: The engine has no compiled file to name. MockEngine is one; a run on it is not missing a fact.
NOT_STATED = "not stated"
#: The identity was written before runs recorded the engine binary. Not a match and not a
#: mismatch: nobody looked, and a resume says so rather than guessing either way.
NOT_RECORDED = "not recorded"


def engine_binary(env_config: Any) -> str:
    """The engine binary's hash, as the environment's own ``config()`` states it.

    One rule, used by the parent for the identity and by every rollout worker for its startup
    report, because the same question answered two ways is two answers. ``RustEngine`` puts the
    hash of the extension FILE it loaded in its config; an engine that states none is
    ``NOT_STATED``. It is the binary THIS process loaded, which is why each worker states its own
    rather than inheriting the parent's: a worker restarted after a rebuild loads the new file.
    """
    engine = env_config.get("engine") if isinstance(env_config, dict) else None
    params = engine.get("params") if isinstance(engine, dict) else None
    stated = params.get("engine_binary_sha256") if isinstance(params, dict) else None
    return str(stated) if stated else NOT_STATED


def digest_of(value: Any) -> str:
    """sha256 of ``canonical_json(value)``, as hex."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _normal(value: Any) -> Any:
    """A value as the identity should see it: numbers as floats, containers recursively.

    ``1`` and ``1.0`` are one weight. A bool stays a bool, because ``True`` is not a weight of
    one. A default that is some object is taken through ``msgspec.to_builtins``; one that will not
    go is recorded by its type's name, which is deterministic across processes where a repr with
    a memory address in it would not be -- at the cost that two such defaults of one type hash
    alike.
    """
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, list | tuple):
        return [_normal(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normal(item) for key, item in value.items()}
    try:
        return _normal(msgspec.to_builtins(value))
    except (TypeError, ValueError):
        kind = type(value)
        return {"__default__": f"{kind.__module__}.{kind.__qualname__}"}


def bound_kwargs(spec: ComponentSpec, extra_modules: tuple[str, ...] = ()) -> dict[str, Any]:
    """A component's kwargs bound to its real signature, defaults applied, numbers normalised.

    What the component will actually be built with, rather than what the config happened to
    write down. A component that cannot be resolved or inspected, or whose kwargs do not bind,
    falls back to its spelling: it will fail when it is built, and this is not the place to
    report that.
    """
    try:
        target = spec.resolve(extra_modules)
        signature = inspect.signature(target)
        bound = signature.bind_partial(**spec.kwargs)
    except (PreflightError, TypeError, ValueError):
        return _normal(dict(spec.kwargs))
    bound.apply_defaults()
    out: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            out.update(value)
        elif kind is not inspect.Parameter.VAR_POSITIONAL:
            out[name] = value
    return _normal(out)


def component_kwarg_problems(spec: ComponentSpec, extra_modules: tuple[str, ...] = ()) -> list[str]:
    """What is wrong with one component's kwargs, found without building it.

    A kwarg the component does not take used to surface as a raw ``TypeError`` inside a worker
    when the environment was built. Binding it to the signature here names it at config load.
    """
    try:
        target = spec.resolve(extra_modules)
    except PreflightError as exc:
        return [str(exc)]
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return []
    try:
        signature.bind_partial(**spec.kwargs)
    except TypeError as exc:
        return [f"{spec.cls}: {exc}"]
    return []


def env_value_digest(spec: EnvFactorySpec, extra_modules: tuple[str, ...] = ()) -> str:
    """sha256 of the environment as it will be BUILT: every component's kwargs bound first.

    ``EnvFactorySpec.digest`` hashes the spelling, so ``{}`` and the same defaults written out
    were two identities, ``1`` and ``1.0`` were two, and a default changed in the code moved the
    objective of every config relying on it without moving any digest. The run identity and the
    ladder's result context read this instead. Found by the train session's review, 2026-09-24.
    """

    def component(item: ComponentSpec) -> dict[str, Any]:
        return {"cls": item.cls, "kwargs": bound_kwargs(item, extra_modules)}

    return digest_of(
        {
            "engine": component(spec.engine),
            "obs_builder": component(spec.obs_builder),
            "action_parser": component(spec.action_parser),
            "reward_fn": component(spec.reward_fn),
            "state_mutator": component(spec.state_mutator),
            "termination": [component(item) for item in spec.termination],
            "truncation": [component(item) for item in spec.truncation],
            "decision_ms": spec.decision_ms,
        }
    )


def resolve_component(path: str, extra_modules: tuple[str, ...] = ()) -> Any:
    """``"royalegym.obs.SpatialObsBuilder"`` -> the class, if the allow-list permits it."""
    allowed = ALLOWED_MODULE_PREFIXES + tuple(extra_modules)
    if not any(path == prefix.rstrip(".") or path.startswith(prefix) for prefix in allowed):
        raise PreflightError(
            f"component {path!r} is not under {', '.join(allowed)}; add its module to "
            "config.extra_component_modules if it is yours"
        )
    module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise PreflightError(f"component {path!r} is not a dotted module path")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PreflightError(f"component {path!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise PreflightError(f"component {path!r}: {module_name} has no {attribute!r}") from exc


class ComponentSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One env component as a class path and its constructor keyword arguments."""

    cls: str
    kwargs: dict[str, JsonValue] = msgspec.field(default_factory=dict)

    def resolve(self, extra_modules: tuple[str, ...] = ()) -> Any:
        return resolve_component(self.cls, extra_modules)

    def recipe(self, extra_modules: tuple[str, ...] = ()) -> tuple[Any, dict[str, Any]]:
        """``(class, kwargs)`` in the shape ``royalegym.env.EnvFactory`` takes."""
        return self.resolve(extra_modules), dict(self.kwargs)

    def build(self, extra_modules: tuple[str, ...] = ()) -> Any:
        cls, kwargs = self.recipe(extra_modules)
        return cls(**kwargs)


class EnvFactorySpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The JSON-able env description: sent to workers, recorded in the checkpoint, and hashed
    into the ladder's context so that two policies rated against each other are known to have
    played the same game."""

    engine: ComponentSpec
    obs_builder: ComponentSpec
    action_parser: ComponentSpec
    reward_fn: ComponentSpec
    state_mutator: ComponentSpec
    termination: list[ComponentSpec]
    truncation: list[ComponentSpec] = msgspec.field(default_factory=list)
    decision_ms: int = 500

    def digest(self) -> str:
        """sha256 of the canonical JSON of this spec."""
        return digest_of(self)

    def factory(
        self,
        extra_modules: tuple[str, ...] = (),
        *,
        viser: str | None = None,
        recorder: Any = None,
    ) -> Any:
        """The ``royalegym.env.EnvFactory`` this spec describes.

        ``viser`` is not part of the recipe: the publisher is bound by the vec env, once per run
        (section 7.7), so it is passed to ``build_vec`` instead.

        ``recorder`` is not part of the recipe either, and for a sharper reason. ``EnvFactory``
        WOULD take it -- ``recorder`` is one of its component names -- but a component is built
        once per env, and the vec env builds one env per game. A recorder in the recipe is
        therefore 24 recorders on a 24-game shard rather than one, which is wrong three ways and
        only the first is about disk: ``ClashParallelEnv`` leaves the single multi-tick
        ``engine.step`` path whenever its recorder wants per-tick frames, so every battle in the
        shard would step one tick at a time for the whole run; ``keep`` bounds each instance
        separately, so retention is ``keep`` times the game count; and the saved name is
        ``{pid}-{completed}-tick{tick}``, whose pid exists to separate PROCESSES and cannot
        separate instances inside one. Two of them collide exactly when two battles end on the
        same tick, which is the step cap. So it is attached by ``build_vec``, to one env, the way
        the viewer is.
        """
        del viser, recorder
        from royalegym.env import EnvFactory

        recipe: dict[str, Any] = {
            "engine": self.engine.recipe(extra_modules),
            "obs_builder": self.obs_builder.recipe(extra_modules),
            "action_parser": self.action_parser.recipe(extra_modules),
            "reward_fn": self.reward_fn.recipe(extra_modules),
            "state_mutator": self.state_mutator.recipe(extra_modules),
            "decision_ms": self.decision_ms,
        }
        # A condition that is turned off is a key the factory is never given. ``EnvFactory``
        # keeps every key it is handed and builds it, so a None would reach the component
        # builder and be called -- which is how an evaluation env, whose truncation is empty by
        # definition, would fail at construction rather than run without one.
        for key, specs in (
            ("termination_cond", self.termination),
            ("truncation_cond", self.truncation),
        ):
            condition = self._condition(specs, extra_modules)
            if condition is not None:
                recipe[key] = condition
        return EnvFactory(**recipe)

    def build_vec(
        self,
        num_games: int,
        extra_modules: tuple[str, ...] = (),
        *,
        viser: Any = None,
        autoreset_seed_fn: Any = None,
        recorder: ComponentSpec | None = None,
    ) -> ClashSelfPlayVecEnv:
        """``ClashSelfPlayVecEnv(num_games)`` built from this spec.

        ``viser`` defaults to None -- no publisher -- because exactly one vec env in a run
        carries the viewer's state stream and the farm is what decides which (section 7.7); a
        second env binding the same UDP port raises at construction. ``"env"`` builds one from
        the environment, and a ``ViserPublisher`` is passed through as given, which is how a
        caller supplies one of its own: a paced publisher, for instance, since a battle is
        simulated faster than it is watched.
        """
        from royalegym.env import ClashSelfPlayVecEnv

        kwargs: dict[str, Any] = {"viser": viser}
        if autoreset_seed_fn is not None:
            # Without it an episode is reachable only by replaying the ones before it, because
            # the autoreset draws from the generator the shard's own seed set. With it, an
            # episode is addressed by its battle and its ordinal, which is what lets a resumed
            # run start the episode the original was about to start.
            kwargs["autoreset_seed_fn"] = autoreset_seed_fn
        vec = ClashSelfPlayVecEnv(num_games, self.factory(extra_modules), **kwargs)
        if recorder is not None:
            # ONE env, the same env the viewer gets, for the reasons in ``factory``. The vec env
            # does this for its own publisher a line after building its envs; a recorder needs
            # the same treatment and nothing in royalegym does it on our behalf.
            # ``ClashParallelEnv`` reads ``self.recorder`` at reset and at every step, so
            # attaching after construction and before the first reset is the whole job.
            vec.envs[0].recorder = recorder.build(extra_modules)
        return vec

    @staticmethod
    def _condition(specs: list[ComponentSpec], extra_modules: tuple[str, ...]) -> Any:
        """One DoneCondition recipe from a list of them.

        A single entry is passed through as itself; several are combined with ``AnyCondition``,
        which checks every child every step so that a child's counter keeps advancing. An empty
        list is None, and the caller leaves the key out of the recipe entirely, which is how a
        truncation is turned off.
        """
        if not specs:
            return None
        if len(specs) == 1:
            return specs[0].recipe(extra_modules)
        from royalegym.done_condition import AnyCondition

        return (AnyCondition, {"conditions": [s.build(extra_modules) for s in specs]})


#: What to run when the engine's compiled data and its data files disagree.
REBUILD_COMMAND = "maturin develop --release, in the RoyaleSim checkout"


def build_env(factory: EnvFactorySpec, extra_modules: tuple[str, ...] = ()) -> Any:
    """One environment of ``factory``, as a one-game vec env. A stale engine build dies here,
    with the rebuild command named, and not at the first cycle of a run.

    Public, so that anything reading an environment's facts -- its spec, its ``config()``, its
    catalogue -- builds it the way preflight does.
    """
    try:
        return factory.build_vec(1, extra_modules, viser=None)
    except RuntimeError as exc:
        text = str(exc)
        if "calibration" not in text and "build" not in text:
            raise
        raise StaleEngineBuild(text, rebuild_command=REBUILD_COMMAND) from exc


def _key_spec(space: Any) -> Any:
    """One ``ObsKeySpec`` from a gymnasium space, with per-channel bounds for a 3-D key."""
    import numpy as np

    from ..api.rollout import ObsKeySpec

    shape = tuple(int(n) for n in space.shape)
    low = np.asarray(getattr(space, "low", np.zeros(shape, dtype=np.float64)), dtype=np.float64)
    high = np.asarray(getattr(space, "high", np.ones(shape, dtype=np.float64)), dtype=np.float64)
    low = np.broadcast_to(low, shape)
    high = np.broadcast_to(high, shape)
    if len(shape) == 3:
        # Per channel: the minimum and the maximum the space declares anywhere on that plane.
        # A space with scalar bounds broadcasts, so this is a reduction of what was declared and
        # never a measurement of what was observed.
        lows = tuple(float(v) for v in low.reshape(shape[0], -1).min(axis=1))
        highs = tuple(float(v) for v in high.reshape(shape[0], -1).max(axis=1))
    else:
        lows = (float(low.min()),)
        highs = (float(high.max()),)
    return ObsKeySpec(shape=shape, dtype=str(space.dtype), low=lows, high=highs)


def read_env_spec(
    vec_env: ClashSelfPlayVecEnv,
    factory_spec: EnvFactorySpec,
    frame_stack: int = 1,
    *,
    seed: int | None = 0,
) -> EnvSpec:
    """Read the ``EnvSpec`` off a built vec env.

    The env is reset first, and the reset comes before anything is read: the engine's timing --
    the tick, the regulation and overtime lengths, and the decision granularity the env settles
    on against them -- is a property of a live battle, so reading the configuration earlier
    would record numbers that are about to change. Everything after that is the
    environment's own report of itself -- ``config()``, the observation space, the builder's
    ``vector_layout()`` and ``spatial_layout()``, the engine's state and catalogue -- and nothing
    is inferred from a sample.
    """
    from ..api.rollout import EnvSpec

    vec_env.reset(seed=seed)
    env = vec_env.envs[0]
    config = env.config()

    space = vec_env.single_observation_space
    obs_space = {key: _key_spec(sub) for key, sub in space.spaces.items()}
    for required in ("spatial", "vector", "action_mask", "mask_planes"):
        if required not in obs_space:
            raise PreflightError(
                f"the observation space has no {required!r} key; the harness's codec and trunk "
                f"are written against an observation of "
                "(spatial, vector, action_mask, mask_planes)"
            )

    offset = 0
    vector_layout: list[tuple[str, int, int]] = []
    for field in env.obs_builder.vector_layout():
        vector_layout.append((field.key, offset, int(field.size)))
        offset += int(field.size)

    hand_size, tiles_y, tiles_x = obs_space["mask_planes"].shape
    state = env.engine.state()
    spatial_shape = obs_space["spatial"].shape
    spec = EnvSpec(
        num_cards=len(env.engine.cards()),
        obs_space=obs_space,
        vector_layout=tuple(vector_layout),
        spatial_layout=tuple(env.obs_builder.spatial_layout()),
        frame_stack=frame_stack,
        vector_size=obs_space["vector"].shape[0],
        spatial_shape=(spatial_shape[0], spatial_shape[1], spatial_shape[2]),
        n_actions=int(vec_env.single_action_space.n),
        hand_size=hand_size,
        tiles=(tiles_y, tiles_x),
        decision_ms=int(config["decision_ms"]),
        tick_ms=int(state.tick_ms),
        decision_ticks=int(config["decision_ticks"]),
        regular_ticks=int(state.regular_ticks),
        overtime_ticks=int(state.overtime_ticks),
        engine_build_digest=str(config.get("build_digest") or ""),
        obs_digest="",
        env_factory=factory_spec,
    )
    return msgspec.structs.replace(spec, obs_digest=obs_digest(spec, config))


def obs_digest(spec: EnvSpec, env_config: dict[str, Any]) -> str:
    """The digest that says whether two policies saw the same observation.

    Over the observation space, both layouts, the frame stack, the action width, the catalogue
    size and the obs builder's own configuration -- so that a changed ``Reveal``, which adds
    vector slots and can add a spatial plane, moves it, and so does a builder parameter that
    changes nothing about the widths.
    """
    return digest_of(
        {
            "obs_space": {
                key: {"shape": k.shape, "dtype": k.dtype, "low": k.low, "high": k.high}
                for key, k in spec.obs_space.items()
            },
            "vector_layout": spec.vector_layout,
            "spatial_layout": spec.spatial_layout,
            "frame_stack": spec.frame_stack,
            "n_actions": spec.n_actions,
            "num_cards": spec.num_cards,
            "obs_builder": env_config.get("obs_builder"),
        }
    )
