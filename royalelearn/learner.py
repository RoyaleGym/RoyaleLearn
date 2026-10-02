"""Train a bot in one call: ``Learner(build_env).learn(total_steps=...)``.

The quickest way to use RoyaleLearn. It builds a whole run from a function that makes your
environment and a few named settings, with defaults that train on a CPU::

    from royalegym import ClashParallelEnv, RustEngine
    from royalelearn import Learner

    def build_env():
        return ClashParallelEnv(RustEngine())

    if __name__ == "__main__":
        learner = Learner(build_env, save_dir="runs/my_bot")
        learner.learn(total_steps=200_000)
        learner.save("runs/my_bot/bot")

``build_env`` must be a function at the top level of a module (your script is fine), because
the run rebuilds environments from its name. It returns a two-seat ``ClashParallelEnv``: the
learner plays one seat and ``opponent`` decides who plays the other.

Everything a run writes goes in ``save_dir``: checkpoints, metric rows and the run's config.
Making a ``Learner`` on the same ``save_dir`` again carries the run on from its newest
checkpoint. For every setting there is, build a ``RunConfig`` and use ``LearningCoordinator``
or ``python -m royalelearn train`` instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from . import config as cfg
from .errors import PreflightError

__all__ = ["Bot", "Learner"]

#: Who sits in the other seat, by name.
OPPONENTS = ("random", "noop", "self")
#: The files a saved bot is.
POLICY_FILE = "policy.json"
WEIGHTS_FILE = "actor.safetensors"


def _dotted(env_fn: Any) -> str:
    """``module.function`` for a function the workers can import, or a refusal saying why not."""
    name = getattr(env_fn, "__qualname__", "")
    module = getattr(env_fn, "__module__", None)
    if not callable(env_fn) or not module or not name or "<" in name or "." in name:
        raise PreflightError(
            f"build_env must be a function defined at the top level of a module (your script is "
            f"fine), not {env_fn!r}: the run rebuilds environments from its name, and a lambda, "
            "a nested function or a method has no name it can be found by"
        )
    return f"{module}.{name}"


def _env_spec(env_fn: Any, path: str) -> cfg.EnvFactorySpec:
    """The env spec of a run whose environments ``env_fn`` builds, describing what it built."""
    env = env_fn()
    try:
        engine = type(env.engine)
        decision_ms = int(env.config().get("decision_ms", 500))
        truncation = _truncation(env.truncation)
    finally:
        env.close()
    spec = cfg.default_env_spec(f"{engine.__module__}.{engine.__qualname__}")
    return msgspec.structs.replace(
        spec, env_fn=path, decision_ms=decision_ms, truncation=truncation
    )


def _truncation(condition: Any) -> list[Any]:
    """The env's truncation as the spec's list, which replaces it in every env the run builds."""
    from .rollout.envspec import ComponentSpec

    if condition is None:
        return []
    kind = type(condition)
    path = f"{kind.__module__}.{kind.__qualname__}"
    if path == "royalegym.done_condition.StepLimitCondition":
        return [ComponentSpec(path, {"max_steps": int(condition.max_steps)})]
    params = condition.config() if callable(getattr(condition, "config", None)) else {}
    spec = ComponentSpec(path, dict(params))
    try:
        spec.build(("royalegym.", kind.__module__ + "."))
    except Exception as exc:
        raise PreflightError(
            f"build_env's truncation {kind.__name__} cannot be rebuilt from its config(), and "
            "the run rebuilds it in every env; use StepLimitCondition(max_steps), or give the "
            f"condition a config() that is its constructor's keywords ({exc})"
        ) from exc
    return [spec]


def _quiet(line: str) -> None:
    """The start-up report, cut to what a person training a first bot wants to see."""
    if line.startswith(("checkpoint", "device")):
        print(line)


def _ladder(opponent: str) -> cfg.LadderConfig:
    if opponent not in OPPONENTS:
        raise PreflightError(f"opponent {opponent!r} is not one of {', '.join(OPPONENTS)}")
    if opponent == "self":
        # Mirror battles, past selves from the pool and scripted bots: the full ladder.
        return cfg.LadderConfig(candidate_every_env_steps=1_000_000)
    bot = "scripted:random_legal" if opponent == "random" else "scripted:noop"
    never = 10**15
    return cfg.LadderConfig(
        mix=(0.0, 0.0, 1.0),
        scripted_opponents=(bot,),
        candidate_every_env_steps=never,
        floor_admit_every_env_steps=never,
    )


class Learner:
    """One training run, set up from ``build_env`` and a few settings.

    ``n_envs`` battles are played at once, in this process. ``steps_per_update`` decisions are
    collected between two updates of the network. ``opponent`` is "random" (a bot that plays a
    random legal move now and then), "noop" (one that never plays) or "self" (copies of the
    learner, past and present, and the scripted bots). ``device`` is "cpu" or "cuda". ``seed``
    makes a run repeatable. ``viser=True`` streams one battle to the viewer (run
    ``royaleviser`` in another terminal). ``resume=False`` refuses to carry on a run already in
    ``save_dir`` rather than continuing it. ``verbose=True`` prints the whole start-up report.
    """

    def __init__(
        self,
        build_env: Any,
        *,
        n_envs: int = 8,
        device: str = "cpu",
        save_dir: str | os.PathLike[str] = "runs/royalelearn",
        opponent: str = "random",
        steps_per_update: int = 1024,
        checkpoint_every: int = 50_000,
        seed: int | None = None,
        viser: bool = False,
        resume: bool = True,
        verbose: bool = False,
        _coordinator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        path = _dotted(build_env)
        self.build_env = build_env
        self.save_dir = Path(save_dir)
        self.device = device
        self.viser = bool(viser)
        self.resume = bool(resume)
        self.verbose = bool(verbose)
        self.steps = 0
        #: The checkpoint the last ``learn`` carried on from, or None for a fresh start.
        self.resumed_from: Path | None = None
        self._kwargs = dict(_coordinator_kwargs or {})
        self._model: Any = None
        self._spec: Any = None
        if steps_per_update < 2 * n_envs:
            raise PreflightError(
                f"steps_per_update {steps_per_update} is smaller than two decisions per battle "
                f"of n_envs {n_envs}"
            )
        width = 32
        settings: dict[str, Any] = {
            "run_name": self.save_dir.name,
            "runs_dir": str(self.save_dir.parent),
            "env": None,
            "extra_component_modules": [path.rsplit(".", 1)[0] + "."],
            "rollout": cfg.RolloutConfig(
                source="inline", workers=1, games_per_worker=int(n_envs), shards_per_worker=1
            ),
            "net": cfg.NetConfig(
                channels=width,
                blocks=2,
                norm_groups=4,
                card_embed=width,
                value_hidden=64,
                # One trunk for the actor and the critic: on a CPU the update is most of an
                # iteration, and two trunks double it.
                separate_trunks=False,
                autocast_dtype="float32",
                device=device,
            ),
            # Repeatable from a seed on one machine, without run_exact's bit-for-bit costs.
            "determinism": cfg.DeterminismConfig(tier="throughput"),
            "ppo": cfg.PPOConfig(
                timesteps_per_iteration=int(steps_per_update),
                batch_size=max(2, int(steps_per_update) // 2),
                minibatch_size=max(1, int(steps_per_update) // 8),
            ),
            "ladder": _ladder(opponent),
            "checkpoint": cfg.CheckpointConfig(every_env_steps=int(checkpoint_every), keep=3),
            "metrics": cfg.MetricsConfig(
                sinks=[cfg.SinkSpec("jsonl"), cfg.SinkSpec("console", options={"brief": True})]
                + ([cfg.SinkSpec("viser")] if viser else [])
            ),
        }
        if seed is not None:
            settings["master_seed"] = int(seed)
        self._env_fn_path = path
        self._settings = settings
        self._config: cfg.RunConfig | None = None

    @property
    def config(self) -> cfg.RunConfig:
        """The whole run's config, every setting spelled out (the env is built once to read)."""
        if self._config is None:
            settings = dict(self._settings)
            settings["env"] = _env_spec(self.build_env, self._env_fn_path)
            self._config = cfg.validate(cfg.RunConfig(**settings))
        return self._config

    def learn(self, total_steps: int) -> None:
        """Train until the run has taken ``total_steps`` decisions in all, then checkpoint.

        A run already in ``save_dir`` is carried on from its newest checkpoint, so calling this
        again, or in a new process, continues where the last one stopped.
        """
        from .checkpoint import DirCheckpointStore
        from .coordinator import LearningCoordinator
        from .determinism import apply_cublas_workspace_config

        # What the command line does before torch can start CUDA, for the same reason.
        apply_cublas_workspace_config()

        latest = DirCheckpointStore(self.save_dir).latest() if self.save_dir.is_dir() else None
        if latest is not None and not self.resume:
            raise PreflightError(
                f"{self.save_dir} already holds a run. Pass resume=True to carry it on, or "
                "give this run another save_dir."
            )
        if self.viser:
            os.environ.setdefault("ROYALEVISER", "1")
        self.resumed_from = latest
        config = self.config
        print(
            f"training in {self.save_dir} on {config.env.engine.cls.rsplit('.', 1)[-1]}: "
            f"{config.rollout.games_per_worker} battles at once, a line every "
            f"{config.ppo.timesteps_per_iteration:,} steps"
            + (f", carrying on from {latest}" if latest is not None else "")
        )
        kwargs = dict(self._kwargs)
        if not self.verbose:
            kwargs.setdefault("printer", _quiet)
        run = LearningCoordinator(
            self.config,
            device=self.device,
            resume=latest,
            run_dir=self.save_dir,
            install_signal_handler=False,
            **kwargs,
        )
        with run:
            run.learn(until_timesteps=int(total_steps))
            self.steps = int(run.cumulative_timesteps)
            self._model = run.model
            self._spec = run.spec

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the trained bot to the folder ``path``: weights and what they play. Load it with
        ``Learner.load_policy``. Call ``learn`` first."""
        from .ladder.snapshots import _encode_tensors

        if self._model is None:
            raise PreflightError("there is no trained bot to save yet: call learn() first")
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / WEIGHTS_FILE).write_bytes(_encode_tensors(self._model.actor.state_dict()))
        record = {"format": 1, "env_spec": self._spec, "net": self.config.net}
        (folder / POLICY_FILE).write_bytes(msgspec.json.encode(record))
        return folder

    @staticmethod
    def load_policy(path: str | os.PathLike[str], *, greedy: bool = False) -> Bot:
        """A bot saved by ``save``: call it on one seat's observation to get its action."""
        return Bot.load(path, greedy=greedy)


class Bot:
    """A trained policy that plays one seat: ``bot(obs)`` returns an action for that seat.

    ``obs`` is the observation the environment hands that seat. ``greedy=True`` plays the most
    likely legal move; otherwise it samples, as it was trained to.
    """

    def __init__(self, actor: Any, spec: Any, net: Any, *, greedy: bool, seed: int = 0) -> None:
        from .ladder.actors import EvalActors

        mode = "argmax" if greedy else "stochastic"
        self._play = EvalActors(spec, net, None, release_mode=mode).policy(actor)  # type: ignore[arg-type]
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs: Any) -> int:
        return int(self._play(obs, float(self._rng.random()), self._rng))

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, greedy: bool = False) -> Bot:
        from .api.rollout import EnvSpec
        from .ladder.snapshots import _decode_tensors
        from .learn.actor_critic import ClashActor
        from .learn.nets import ClashTrunk, PointerPolicyHead, resolve_dtype

        folder = Path(path)
        if not (folder / POLICY_FILE).is_file():
            raise PreflightError(f"{folder} is not a saved bot: it has no {POLICY_FILE}")
        record = msgspec.json.decode((folder / POLICY_FILE).read_bytes())
        spec = msgspec.convert(record["env_spec"], EnvSpec)
        net = msgspec.convert(record["net"], cfg.NetConfig)
        actor = ClashActor(
            ClashTrunk(spec, net), PointerPolicyHead(spec, net), resolve_dtype(net.autocast_dtype)
        )
        actor.load_state_dict(_decode_tensors((folder / WEIGHTS_FILE).read_bytes(), "cpu"))
        actor.eval()
        return cls(actor, spec, net, greedy=greedy)

