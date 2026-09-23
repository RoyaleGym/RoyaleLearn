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
from typing import TYPE_CHECKING, Any

import msgspec

from ..errors import PreflightError

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


def digest_of(value: Any) -> str:
    """sha256 of ``canonical_json(value)``, as hex."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


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
        self, extra_modules: tuple[str, ...] = (), *, viser: str | None = None
    ) -> Any:
        """The ``royalegym.env.EnvFactory`` this spec describes.

        ``viser`` is not part of the recipe: the publisher is bound by the vec env, once per run
        (section 7.7), so it is passed to ``build_vec`` instead.
        """
        del viser
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
        return ClashSelfPlayVecEnv(num_games, self.factory(extra_modules), **kwargs)

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
