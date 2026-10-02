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

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from . import config as cfg
from .errors import IdentityMismatch, PreflightError

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


#: The most threads ``default_threads`` gives the network. Past this a battle-sized update gains
#: little, and the battles in the same process want cores too.
MAX_DEFAULT_THREADS = 8


#: Printed when ``device="auto"`` finds no GPU.
NO_GPU_LINE = (
    "No GPU that torch can use was found, so this trains on the CPU, which is much slower. "
    "With an NVIDIA card, install torch with CUDA (see Install)."
)


def resolve_device(device: str) -> str:
    """``device`` as torch names it: "auto" is "cuda" when torch can see a GPU, else "cpu",
    saying so in one printed line; anything else is passed through."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    print(NO_GPU_LINE)
    return "cpu"


def default_threads() -> int:
    """Threads for the network's arithmetic on a CPU: half the machine's logical cores (its
    physical ones, on most machines), at least one and at most ``MAX_DEFAULT_THREADS``."""
    return max(1, min(MAX_DEFAULT_THREADS, (os.cpu_count() or 2) // 2))


#: The run-identity fields a ``Learner`` carries a run on across: the update's settings (learning
#: rates, epochs, batch sizes, the entropy bonus, the discount). Anything else -- the network, the
#: environment, the opponents -- is another run.
TRAINING_SETTINGS = frozenset({"algo_digest"})


class Learner:
    """One training run, set up from ``build_env`` and named settings, each with a default.

    The run:

    - ``n_envs``: battles played at once, in this process.
    - ``device``: "auto" (a GPU when torch can see one, else the CPU, with a line saying so),
      "cuda" or "cpu". ``threads``: CPU threads for the network's arithmetic (default: half the
      machine's logical cores, at most eight).
    - ``save_dir``: where checkpoints, metric rows and the config go. Made again with the same
      folder, a run carries on; ``resume=False`` refuses to instead.
    - ``opponent``: "random" (a bot that plays a random legal move now and then), "noop" (one
      that never plays) or "self" (copies of the learner, past and present, and the scripted
      bots).
    - ``timestep_limit``: where ``learn()`` stops when it is given no ``total_steps``.
    - ``checkpoint_every``: decisions between checkpoints. ``n_checkpoints_to_keep``: how many.
    - ``seed``: makes a run repeatable on one machine.

    The network, one trunk shared by the policy and the critic:

    - ``trunk_channels``: width of the convolutional trunk over the board.
    - ``trunk_blocks``: residual blocks in it.
    - ``critic_hidden``: the critic's hidden layer.

    The update, named as rlgym-ppo names them:

    - ``steps_per_update``: decisions collected between two updates.
    - ``ppo_epochs``: passes over each update's decisions.
    - ``ppo_batch_size``, ``ppo_minibatch_size``: decisions per optimizer step, and per forward
      (a memory knob); by default half and an eighth of ``steps_per_update``.
    - ``policy_lr``, ``critic_lr``: learning rates.
    - ``ppo_ent_coef``: the entropy bonus; None keeps the default, which falls from 0.01 to 0.003
      over 30 million decisions.
    - ``ppo_clip_range``: the PPO clip.
    - ``gae_gamma``: the discount; None keeps the default, which rises from 0.997 to 0.999.
      ``gae_lambda``: GAE's lambda.
    - ``standardize_returns``: scale rewards by the spread of the returns.

    Watching:

    - ``log_to_wandb``: send each update's numbers to Weights and Biases, under
      ``wandb_project_name``, ``wandb_group_name`` and ``wandb_run_name``.
    - ``viser``: stream one battle to the viewer (run ``royaleviser`` in another terminal).
    - ``verbose``: print the whole start-up report.

    ``extensions`` sets an add-on's config sections by name, for example RoyaleImitate's
    ``warm_start``. Every other setting there is is a ``RunConfig`` field; ``learner.config``
    shows them all.
    """

    def __init__(
        self,
        build_env: Any,
        *,
        n_envs: int = 8,
        device: str = "auto",
        threads: int | None = None,
        save_dir: str | os.PathLike[str] = "runs/royalelearn",
        resume: bool = True,
        opponent: str = "random",
        timestep_limit: int | None = None,
        checkpoint_every: int = 50_000,
        n_checkpoints_to_keep: int = 3,
        seed: int | None = None,
        trunk_channels: int = 32,
        trunk_blocks: int = 2,
        critic_hidden: int = 64,
        steps_per_update: int = 1024,
        ppo_epochs: int = 3,
        ppo_batch_size: int | None = None,
        ppo_minibatch_size: int | None = None,
        policy_lr: float = 2e-4,
        critic_lr: float = 2e-4,
        ppo_ent_coef: float | None = None,
        ppo_clip_range: float = 0.2,
        gae_gamma: float | None = None,
        gae_lambda: float = 0.99,
        standardize_returns: bool = True,
        log_to_wandb: bool = False,
        wandb_project_name: str | None = None,
        wandb_group_name: str | None = None,
        wandb_run_name: str | None = None,
        viser: bool = False,
        verbose: bool = False,
        extensions: Mapping[str, Any] | None = None,
        _coordinator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        path = _dotted(build_env)
        self.build_env = build_env
        self.save_dir = Path(save_dir)
        device = resolve_device(device)
        self.device = device
        self.viser = bool(viser)
        self.resume = bool(resume)
        self.verbose = bool(verbose)
        self.timestep_limit = None if timestep_limit is None else int(timestep_limit)
        self.steps = 0
        #: The checkpoint the last ``learn`` carried on from, or None for a fresh start.
        self.resumed_from: Path | None = None
        self._kwargs = dict(_coordinator_kwargs or {})
        self._extensions = dict(extensions or {})
        #: The last ``learn``'s run, finished and closed: its model, spec and codec, for an add-on
        #: that writes something from the trained run. None before ``learn``.
        self.run: Any = None
        self._model: Any = None
        self._spec: Any = None
        threads = default_threads() if threads is None else int(threads)
        if threads < 1:
            raise PreflightError(f"threads is {threads}; the network needs at least one")
        if steps_per_update < 2 * n_envs:
            raise PreflightError(
                f"steps_per_update {steps_per_update} is smaller than two decisions per battle "
                f"of n_envs {n_envs}"
            )
        steps = int(steps_per_update)
        width = int(trunk_channels)
        ppo: dict[str, Any] = {
            "timesteps_per_iteration": steps,
            "batch_size": max(2, steps // 2) if ppo_batch_size is None else int(ppo_batch_size),
            "minibatch_size": (
                max(1, steps // 8) if ppo_minibatch_size is None else int(ppo_minibatch_size)
            ),
            "n_epochs": int(ppo_epochs),
            "lr_actor": float(policy_lr),
            "lr_critic": float(critic_lr),
            "clip_range": float(ppo_clip_range),
        }
        if ppo_ent_coef is not None:
            ppo["ent_coef"] = cfg.ConstantSpec(float(ppo_ent_coef))
        advantage: dict[str, Any] = {
            "gae_lambda": float(gae_lambda),
            "standardize_rewards": bool(standardize_returns),
        }
        if gae_gamma is not None:
            advantage["gamma"] = cfg.ConstantSpec(float(gae_gamma))
        sinks = [cfg.SinkSpec("jsonl"), cfg.SinkSpec("console", options={"brief": True})]
        if viser:
            sinks.append(cfg.SinkSpec("viser"))
        if log_to_wandb:
            names = {
                "project": wandb_project_name,
                "group": wandb_group_name,
                "name": wandb_run_name,
            }
            options = {key: value for key, value in names.items() if value is not None}
            sinks.append(cfg.SinkSpec("wandb", options=options))
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
                blocks=int(trunk_blocks),
                # GroupNorm's groups must divide the width; four for every width they can.
                norm_groups=math.gcd(width, 4),
                card_embed=width,
                value_hidden=int(critic_hidden),
                # One trunk for the actor and the critic: on a CPU the update is most of an
                # iteration, and two trunks double it.
                separate_trunks=False,
                autocast_dtype="float32",
                device=device,
            ),
            # Repeatable from a seed on one machine, without run_exact's bit-for-bit costs.
            "determinism": cfg.DeterminismConfig(tier="throughput", torch_threads=threads),
            "ppo": cfg.PPOConfig(**ppo),
            "advantage": cfg.AdvantageConfig(**advantage),
            "ladder": _ladder(opponent),
            "checkpoint": cfg.CheckpointConfig(
                every_env_steps=int(checkpoint_every), keep=int(n_checkpoints_to_keep)
            ),
            "metrics": cfg.MetricsConfig(sinks=sinks),
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
            config = cfg.RunConfig(**settings)
            if self._extensions:
                from .extensions import with_sections

                config = with_sections(config, **self._extensions)
            self._config = cfg.validate(config)
        return self._config

    def learn(self, total_steps: int | None = None) -> None:
        """Train until the run has taken ``total_steps`` decisions in all (``timestep_limit``
        when none is given), then checkpoint.

        A run already in ``save_dir`` is carried on from its newest checkpoint, so calling this
        again, or in a new process, continues where the last one stopped.
        """
        if total_steps is None:
            total_steps = self.timestep_limit
        if total_steps is None:
            raise PreflightError(
                "learn() needs to know where to stop: pass total_steps, or set timestep_limit "
                "on the Learner"
            )
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
        self._register_main()
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
        try:
            run = self._train(LearningCoordinator, latest, int(total_steps), kwargs)
        except IdentityMismatch as mismatch:
            if latest is None or set(mismatch.differences) - TRAINING_SETTINGS:
                raise PreflightError(
                    f"{self.save_dir} holds a run started with another network, environment or "
                    "opponent setup, so this one cannot carry it on. Use the settings it was "
                    "started with, or give this run another save_dir. What differs:\n"
                    + "\n".join(f"  {name}" for name in sorted(mismatch.differences))
                ) from mismatch
            print(
                f"training settings changed since the last run in {self.save_dir}; carrying on "
                "with the new ones"
            )
            kwargs["allow_identity_drift"] = True
            run = self._train(LearningCoordinator, latest, int(total_steps), kwargs)
        self.run = run

    def _train(
        self, coordinator: Any, latest: Path | None, total_steps: int, kwargs: dict[str, Any]
    ) -> Any:
        run = coordinator(
            self.config,
            device=self.device,
            resume=latest,
            run_dir=self.save_dir,
            install_signal_handler=False,
            **kwargs,
        )
        with run:
            run.learn(until_timesteps=total_steps)
            self.steps = int(run.cumulative_timesteps)
            self._model = run.model
            self._spec = run.spec
        return run

    def _register_main(self) -> None:
        """Make a ``build_env`` from ``__main__`` findable by its path.

        The run builds its environments from ``__main__.build_env``. A script run with python
        holds it there already; code run by ``exec`` in a namespace named ``__main__`` -- a docs
        checker, some IDE runners -- does not, and the battles, which are played in this
        process, would not find it. So the function itself is put there under its name.
        """
        import sys

        module, _, name = self._env_fn_path.rpartition(".")
        main = sys.modules.get("__main__")
        if (
            module == "__main__"
            and main is not None
            and getattr(main, name, None) is not self.build_env
        ):
            setattr(main, name, self.build_env)

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
        from .learn.nets import ClashTrunk, build_policy_head, resolve_dtype

        folder = Path(path)
        if not (folder / POLICY_FILE).is_file():
            raise PreflightError(f"{folder} is not a saved bot: it has no {POLICY_FILE}")
        record = msgspec.json.decode((folder / POLICY_FILE).read_bytes())
        spec = msgspec.convert(record["env_spec"], EnvSpec)
        net = msgspec.convert(record["net"], cfg.NetConfig)
        actor = ClashActor(
            ClashTrunk(spec, net), build_policy_head(spec, net), resolve_dtype(net.autocast_dtype)
        )
        actor.load_state_dict(_decode_tensors((folder / WEIGHTS_FILE).read_bytes(), "cpu"))
        actor.eval()
        return cls(actor, spec, net, greedy=greedy)

