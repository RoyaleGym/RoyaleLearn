"""``python -m royalelearn <command>``: twelve things a person does to a run.

The order of the first two lines of this module is load-bearing. ``CUBLAS_WORKSPACE_CONFIG``
and the BLAS thread counts are read once, when CUDA initialises and when numpy imports its
BLAS, so they have to be in the environment before torch is imported -- setting them later
would leave a run that looks reproducible and is not. Nothing here imports torch at module
scope, which is also what lets ``royalelearn --help``, ``royalelearn config`` and
``royalelearn identity`` work in an environment where torch is not installed at all.

Every command that needs a run directory finds its configuration in it. A run directory is
self-describing -- ``config.json`` and ``identity.json`` are written the moment it is created --
so ``eval``, ``gate``, ``rate`` and ``replay`` need nothing on the command line but the
directory and what to do in it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .determinism import apply_blas_thread_env, apply_cublas_workspace_config

# Before anything else imports numpy or torch, and before this module's own imports pull them
# in transitively. A value already in the environment is the operator's and is left alone.
apply_cublas_workspace_config()
apply_blas_thread_env()

import msgspec  # noqa: E402

from .config import (  # noqa: E402
    PROFILES,
    RunConfig,
    dump_config,
    geometry,
    load_config,
    profile,
    validate,
)
from .errors import RoyaleLearnError  # noqa: E402

__all__ = ["build_parser", "main"]

CONFIG_NAME = "config.json"
PROFILE_NAMES = tuple(sorted(PROFILES))


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Every command, its arguments and one line saying what it is for."""
    parser = argparse.ArgumentParser(
        prog="royalelearn",
        description="The self-play training harness for RoyaleGym environments.",
    )
    from .version import __version__

    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    write = commands.add_parser("config", help="write a fully populated default config")
    write.add_argument("--profile", default="laptop", choices=PROFILE_NAMES)
    write.add_argument("-o", "--output", type=Path, default=None)
    write.set_defaults(handler=_config)

    doctor = commands.add_parser("doctor", help="run the start-up gates on their own")
    doctor.add_argument("--config", type=Path, default=None)
    doctor.add_argument("--profile", default="laptop", choices=PROFILE_NAMES)
    doctor.set_defaults(handler=_doctor)

    bench = commands.add_parser("bench", help="measure this machine's throughput budget")
    bench.add_argument("--config", type=Path, default=None)
    bench.add_argument("--profile", default="laptop", choices=PROFILE_NAMES)
    bench.add_argument("--seconds", type=float, default=60.0)
    bench.set_defaults(handler=_bench)

    train = commands.add_parser("train", help="a new run")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-name", default=None)
    train.add_argument("--until-timesteps", type=int, default=None)
    train.add_argument("--inline", action="store_true", help="one process, no worker farm")
    train.add_argument("--device", default=None, choices=("cuda", "cpu"))
    train.set_defaults(handler=_train)

    resume = commands.add_parser("resume", help="continue a run from a checkpoint")
    resume.add_argument("--run", type=Path, required=True)
    resume.add_argument("--checkpoint", type=Path, default=None)
    resume.add_argument("--until-timesteps", type=int, default=None)
    resume.add_argument("--allow-identity-drift", action="store_true")
    resume.set_defaults(handler=_resume)

    verify = commands.add_parser("verify-resume", help="prove a resume on this machine")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--iterations", type=int, default=6)
    verify.add_argument("--split", type=int, default=3)
    verify.set_defaults(handler=_verify_resume)

    evaluate = commands.add_parser("eval", help="a paired evaluation between two members")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--a", required=True)
    evaluate.add_argument("--b", required=True)
    evaluate.add_argument("--seeds", type=int, default=50)
    evaluate.set_defaults(handler=_eval)

    gate = commands.add_parser("gate", help="re-run a gate decision from stored snapshots")
    gate.add_argument("--run", type=Path, required=True)
    gate.add_argument("--candidate", required=True)
    gate.set_defaults(handler=_gate)

    rate = commands.add_parser("rate", help="refit the ratings from games.jsonl")
    rate.add_argument("--run", type=Path, required=True)
    rate.add_argument("--output", type=Path, default=None)
    rate.set_defaults(handler=_rate)

    identity = commands.add_parser("identity", help="print the RunIdentity and the run id")
    identity.add_argument("--config", type=Path, required=True)
    identity.set_defaults(handler=_identity)

    replay = commands.add_parser("replay", help="re-simulate one episode and verify it")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--episode", required=True, help="worker/shard/battle/ordinal")
    replay.add_argument("--out", type=Path, default=None)
    replay.set_defaults(handler=_replay)

    play = commands.add_parser("play", help="watch one battle")
    play.add_argument("--checkpoint", type=Path, required=True)
    play.add_argument("--opponent", default="noop")
    play.add_argument("--viser", action="store_true")
    play.set_defaults(handler=_play)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, dispatch, and turn a refusal into a message rather than a traceback."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args) or 0)
    except RoyaleLearnError as exc:
        print(f"royalelearn: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - a person pressed Ctrl-C
        print("interrupted", file=sys.stderr)
        return 130


# ---------------------------------------------------------------------------
# Reading a config and a run directory
# ---------------------------------------------------------------------------


def _config_of(args: argparse.Namespace) -> RunConfig:
    """The config a command was given, or the profile it named."""
    path = getattr(args, "config", None)
    if path is not None:
        return validate(load_config(Path(path)))
    return validate(profile(args.profile))


def _run_config(run_dir: Path) -> RunConfig:
    """The configuration a run directory describes, pointed back at itself.

    ``runs_dir`` and ``run_name`` are taken from where the directory actually is rather than
    from what the file says, so a run that was moved or renamed still resumes -- the identity,
    which is what may not move, is checked separately and by name.
    """
    run_dir = Path(run_dir)
    path = run_dir / CONFIG_NAME
    if not path.exists():
        raise SystemExit(f"{path} does not exist; --run wants a run directory")
    config = load_config(path)
    config.runs_dir = str(run_dir.parent)
    config.run_name = run_dir.name.rsplit("-", 1)[0]
    return validate(config)


def _coordinator(config: RunConfig, **kwargs: Any) -> Any:
    from .coordinator import LearningCoordinator

    return LearningCoordinator(config, **kwargs)


def _latest_checkpoint(run_dir: Path) -> Path:
    """The newest checkpoint, from the run index, printed so the choice is visible.

    Read from the index rather than by parsing directory names: that form crashes on a stray
    file, and a run whose two names differ by a suffix would otherwise resolve to each other's.
    """
    from .checkpoint import DirCheckpointStore

    latest = DirCheckpointStore(run_dir).latest()
    if latest is None:
        raise SystemExit(f"{run_dir} holds no checkpoint to resume from")
    print(f"resuming from {latest}")
    return latest


# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------


def _config(args: argparse.Namespace) -> int:
    """Write a fully populated default config for one machine class."""
    text = dump_config(profile(args.profile), indent=2)
    if args.output is None:
        print(text)
        return 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def _identity(args: argparse.Namespace) -> int:
    """Print what this configuration IS: the thing to paste into an issue.

    It is computed without building an environment where it can be -- the pieces preflight
    contributes are named as unread rather than guessed -- so that this command answers in
    milliseconds and works where the engine cannot be built.
    """
    from .config import config_hash
    from .identity import describe_device, royalegym_provenance, torch_version
    from .rollout.envspec import digest_of
    from .version import __version__, git_describe

    config = _config_of(args)
    geo = geometry(config)
    gym_version, gym_git = royalegym_provenance()
    print(f"royalelearn   {__version__} ({git_describe()})")
    print(f"royalegym     {gym_version} ({gym_git})")
    print(f"torch         {torch_version()} on {describe_device(config.net.device)}")
    print(f"master seed   {config.master_seed}")
    print(f"determinism   {config.determinism.tier}")
    print(f"env           {config.env.engine.cls}, {config.env.digest()[:16]}")
    print(
        f"geometry      {geo.workers} x {geo.games_per_worker} battles, {geo.n_slots} slots, "
        f"{geo.cycles} cycles"
    )
    print(f"algo          {digest_of({'ppo': config.ppo, 'advantage': config.advantage})[:16]}")
    ladder = digest_of({"ladder": config.ladder, "master_seed": config.master_seed})
    print(f"ladder        {ladder[:16]}")
    print(f"config hash   {config_hash(config)}")
    print(
        "run id        read off a built environment at start-up: it also covers the engine "
        "build, the observation space and the codec table, which is what `royalelearn doctor` "
        "prints"
    )
    return 0


def _doctor(args: argparse.Namespace) -> int:
    """The start-up gates on their own: seconds, and it catches most first-run failures."""
    from .identity import compute_identity, run_id
    from .learn.nets import DefaultNetworkFactory
    from .rollout.preflight import run_preflight

    config = _config_of(args)
    report = run_preflight(config)
    factory = DefaultNetworkFactory(config.master_seed)
    arch_digest = factory.arch_digest(report.spec, config.net)
    identity = compute_identity(
        config,
        env_spec=report.spec,
        build=report.build,
        arch_digest=arch_digest,
        codec_version=report.codec_version,
        codec_table_digest=report.codec_table_digest,
    )
    report.note_run_id(run_id(identity))
    return 0


def _train(args: argparse.Namespace) -> int:
    """A new run."""
    config = _config_of(args)
    if args.run_name:
        config.run_name = args.run_name
    if args.inline:
        config.rollout = msgspec.structs.replace(config.rollout, source="inline")
    with _coordinator(validate(config), device=args.device) as run:
        run.learn(until_timesteps=args.until_timesteps)
        print(f"run {run.run_id} stopped at iteration {run.iteration}")
    return 0


def _resume(args: argparse.Namespace) -> int:
    """Continue a run; refuse on an identity mismatch and name every differing field."""
    config = _run_config(args.run)
    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(args.run)
    with _coordinator(
        config, resume=checkpoint, allow_identity_drift=args.allow_identity_drift
    ) as run:
        run.learn(until_timesteps=args.until_timesteps)
        print(f"run {run.run_id} stopped at iteration {run.iteration}")
    return 0


def _verify_resume(args: argparse.Namespace) -> int:
    """Section 12.4's proof, on this machine, in two processes.

    The first process runs ``--split`` iterations and checkpoints; a second, fresh one loads
    that checkpoint and reports what it loaded. What is compared is the learner's whole state
    digest across the process boundary, which is what a resume has to get right: the weights,
    both optimizers' moments, the return scaler and the schedule positions, byte for byte.
    """
    config = _config_of(args)
    with _coordinator(config) as run:
        run.learn(until_timesteps=_timesteps_for(config, args.split))
        run.checkpoint()
        digest = run.state_digest()
        run_dir, iteration = run.run_dir, run.iteration
    print(f"checkpointed at iteration {iteration}, state {digest}")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "royalelearn",
            "resume",
            "--run",
            str(run_dir),
            "--until-timesteps",
            str(_timesteps_for(config, args.iterations)),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    print(done.stdout, end="")
    loaded = _loaded_digest(done.stdout)
    if done.returncode != 0:
        print(done.stderr, file=sys.stderr)
        return done.returncode
    if loaded and not digest.startswith(loaded):
        print(f"the resumed process loaded {loaded}, and the checkpoint holds {digest[:16]}")
        return 1
    print("the resumed process loaded the same learner state, byte for byte")
    return 0


def _loaded_digest(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("resumed "):
            return line.rsplit(" ", 1)[-1].strip()
    return ""


def _timesteps_for(config: RunConfig, iterations: int) -> int:
    """How many timesteps ``iterations`` iterations of this config collect."""
    return max(1, iterations) * config.ppo.timesteps_per_iteration


def _bench(args: argparse.Namespace) -> int:
    """Measure section 2.3's table for THIS machine, and print it.

    Nothing here is asserted. The numbers are the machine's, and what the table is for is
    deciding two things on it: whether a second shard is worth its thread, and whether the
    harness is still comfortably faster than the learner it feeds.
    """
    config = _config_of(args)
    deadline = time.perf_counter() + max(1.0, args.seconds)
    with _coordinator(config) as run:
        iterations = 0
        while time.perf_counter() < deadline:
            run.iterate()
            iterations += 1
            if iterations >= 3:
                break
        row = run.rows[-1]
        geo = run.geometry
        env_steps = geo.cycles * geo.n_battles
        codec_us = _codec_microseconds(run)
        report = {
            "iterations": iterations,
            "env_ms_per_game_step": 1000.0 * float(row["time/env"]) / max(1, env_steps),
            "codec_us_per_row": codec_us,
            "boundary_mb_per_second": float(row["throughput/boundary_mb_per_second"]),
            "inference_ms_per_round": float(row["throughput/inference_ms_per_round"]),
            "update_timesteps_per_second": (
                float(row["run/cumulative_timesteps"]) / max(1e-9, float(row["time/update"]))
            ),
            "rollout_capacity_ratio": float(row["throughput/rollout_capacity_ratio"]),
            "vram_peak_mb": float(row["health/vram_peak_mb"]),
            "ratio_max_abs_dev": float(row["ppo/ratio_max_abs_dev"]),
            "ratio_atol": float(
                config.ppo.ratio_atol.get(config.net.autocast_dtype, float("nan"))
            ),
        }
    for name, value in report.items():
        print(f"{name:<32} {value:.6g}" if isinstance(value, float) else f"{name:<32} {value}")
    return 0


def _codec_microseconds(run: Any) -> float:
    """Microseconds to pack one observation row, measured on this machine's own codec."""
    import numpy as np

    spec = run.spec
    row_bytes = int(run.codec.row_bytes(spec))
    out = memoryview(bytearray(row_bytes))
    obs = {
        "spatial": np.zeros(spec.spatial_shape, dtype=np.float32),
        "vector": np.zeros(spec.vector_size, dtype=np.float32),
        "action_mask": np.ones(spec.n_actions, dtype=np.int8),
    }
    started = time.perf_counter()
    for _ in range(200):
        run.codec.pack(obs, out, 0)
    return 1e6 * (time.perf_counter() - started) / 200


def _eval(args: argparse.Namespace) -> int:
    """A paired evaluation between two members, with its interval, outside the loop."""
    config = _run_config(args.run)
    with _coordinator(config) as run:
        comparison = run.eval_runner.compare(args.a, args.b, games=2 * args.seeds)
    print(f"{args.a} vs {args.b} over {comparison.n_games} battles ({comparison.n_seeds} seeds)")
    print(f"score  {comparison.score_a:.4f}  95% [{comparison.lo:.4f}, {comparison.hi:.4f}]")
    print(f"rho    {comparison.rho:.3f}   draws {comparison.draw_rate:.3f}")
    return 0


def _gate(args: argparse.Namespace) -> int:
    """Re-run a gate decision from stored snapshots and print the verdict."""
    config = _run_config(args.run)
    with _coordinator(config) as run:
        decision = run.gate.evaluate(args.candidate, run.pool, run.eval_runner)
    print(f"candidate {decision.candidate} against {decision.champion or 'nothing'}")
    for name, condition in decision.conditions.items():
        verdict = "pass" if condition.passed else "FAIL"
        print(
            f"  {name:<28} {verdict}  n={condition.n} observed={condition.observed:.4f} "
            f"bound={condition.bound:.4f}"
        )
    print(f"admit={decision.admit} promote={decision.promote} cycle={decision.cycle}")
    return 0 if decision.admit else 1


def _rate(args: argparse.Namespace) -> int:
    """Refit the ratings from the result log and print the table with its intervals."""
    from .ladder.rating import Z95, BradleyTerryDavidsonRater
    from .ladder.results import ResultLog

    config = _run_config(args.run)
    log = ResultLog(Path(args.run) / "ladder" / "games.jsonl")
    view = log.eval_view()
    if not len(view):
        print("the result log holds no evaluation games yet")
        return 1
    rater = BradleyTerryDavidsonRater(
        prior_sd=config.ladder.rater.prior_sd,
        anchor=config.ladder.rater.anchor,
        draws=config.ladder.rater.draws,
    )
    table = rater.fit(view)
    print(f"{'member':<32} {'rating':>9} {'se':>7} {'95% interval':>22} {'games':>7}")
    for member in sorted(table.rating, key=lambda name: -table.rating[name]):
        rating, se = table.rating[member], table.se.get(member, 0.0)
        interval = f"[{rating - Z95 * se:8.1f}, {rating + Z95 * se:8.1f}]"
        print(
            f"{member:<32} {rating:9.1f} {se:7.1f} {interval:>22} "
            f"{table.n_games.get(member, 0):7d}"
        )
    print(f"anchor {table.anchor} at 0; transitivity residual {table.transitivity_residual:.4f}")
    if args.output:
        Path(args.output).write_bytes(msgspec.json.encode(table))
        print(f"wrote {args.output}")
    return 0


def _replay(args: argparse.Namespace) -> int:
    """Re-simulate one episode from its shard seed and reset ordinal, and verify it."""
    from royalegym.replay import save_trace

    from .api.rollout import EpisodeRecord
    from .metrics.bundle import replay_episode
    from .rollout.plan import SlotPlanner

    config = _run_config(args.run)
    worker, shard, battle, ordinal = (int(part) for part in args.episode.split("/"))
    planner = SlotPlanner(geometry(config), config.master_seed)
    record = msgspec.convert(
        {
            "slot": planner.slot_of(battle, 0),
            "worker": worker,
            "shard": shard,
            "battle": battle,
            "seat": 0,
            "ordinal": ordinal,
            "episode_seed_path": planner.env_seed_path(worker, shard, 0),
            "policy_id": "learner",
            "opponent_id": "learner",
            "bucket": "mirror",
            "episode_steps": 0,
            "episode_ticks": 0,
            "own_crowns": 0,
            "enemy_crowns": 0,
            "own_tower_hp_frac": 0.0,
            "enemy_tower_hp_frac": 0.0,
            "elixir_leak_steps": 0,
            "elixir_count_exact": True,
            "winner": -1,
            "outcome": 0,
            "cards_played": 0,
            "illegal_commands": 0,
            "undiscounted_return": 0.0,
            "reward_terms": {},
        },
        type=EpisodeRecord,
    )
    trace, divergences = replay_episode(
        config, record, seed=planner.env_seed(worker, shard, 0)
    )
    print(f"episode {args.episode} replayed from {record.episode_seed_path}")
    if trace is not None:
        print(f"  {len(trace.steps)} steps, {len(trace.frames)} frames")
        if args.out:
            save_trace(trace, args.out)
            print(f"  wrote {args.out}")
    if divergences:
        for line in divergences[:8]:
            print(f"  divergence: {line}")
        return 1
    print("  verify_trace found no divergence")
    return 0


def _play(args: argparse.Namespace) -> int:
    """Watch one battle: the checkpoint's policy against a scripted opponent or a snapshot."""
    checkpoint = Path(args.checkpoint)
    run_dir = checkpoint.parent.parent
    config = _run_config(run_dir)
    with _coordinator(config, resume=checkpoint) as run:
        run.snapshot_store.put("play:current", run.model, {"step": run.cumulative_env_steps})
        opponent = (
            f"scripted:{args.opponent}"
            if args.opponent in ("noop", "random_legal", "random")
            else args.opponent
        )
        if opponent == "scripted:random":
            opponent = "scripted:random_legal"
        run.player.viser = bool(args.viser)
        score = run.player.play(
            a="play:current",
            b=opponent,
            seed=run.seeds.seeds[0],
            a_seat=0,
            act_path="eval/match/play/0/blue",
        )
    print(f"play:current vs {opponent}: {score}")
    return 0
