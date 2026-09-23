"""What a run IS, as sixteen hex characters.

The identity is everything that can change a number the run produces. Two runs with the same
identity produce the same metric stream under ``determinism.tier = "run_exact"``; two runs with
different identities are different experiments and are never plotted as one. A resume compares
the identity field by field and refuses a difference by name.

What is in it, and why:

    master_seed, determinism_tier, device_kind, torch_version   each changes the byte stream
    the engine build and the card catalogue                     the game itself
    env_spec_digest, obs_digest, action_digest                  what the policy sees and can do
    frame_stack                                                 the trunk's input width
    arch_digest, codec_version, codec_table_digest              the network and how a row is stored
    algo_digest                                                 the PPO, GAE and schedule settings
    rollout_digest                                              workers, games, shards, T and R
    ladder_digest                                               who the learner plays

What is recorded and NOT in it, because none of it changes what the run is: ``run_name``,
``runs_dir``, ``timestep_limit``, the metrics sinks and their settings, ``checkpoint.keep``, the
alarm thresholds and severities, the doctor's budget, and the knobs that set how the collection
is scheduled rather than what it collects. An alarm can stop a run; it cannot alter a number. A
resume diffs those and prints every difference rather than refusing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import subprocess
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from .config import RunConfig, geometry
from .rollout.envspec import canonical_json, digest_of
from .version import UNKNOWN, __version__, git_describe

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .api.rollout import EnvSpec

__all__ = [
    "EXCLUDED_FROM_IDENTITY",
    "IDENTITY_FORMAT_VERSION",
    "EngineBuild",
    "RunIdentity",
    "catalogue_digest",
    "compute_identity",
    "describe_device",
    "dirty_sources",
    "engine_build",
    "identity_differences",
    "royalegym_provenance",
    "run_id",
    "torch_version",
]

IDENTITY_FORMAT_VERSION = 1

#: Config fields that are recorded in the checkpoint and left out of the identity hash. A resume
#: prints every difference in these and continues.
EXCLUDED_FROM_IDENTITY: tuple[str, ...] = (
    "run_name",
    "runs_dir",
    "timestep_limit",
    "extra_component_modules",
    "rollout.source",
    "rollout.spin_us",
    "rollout.round_timeout_s",
    "rollout.restart_failed_workers",
    "rollout.max_restarts_per_worker",
    "rollout.launch_delay_s",
    "rollout.stagger_first_reset",
    # overlap is the schedule of the collection rather than its configuration: with it on, a
    # cycle's actions come from the previous iteration's weights, so a run resumed with it
    # flipped is the same configuration collected differently. Diffed and printed, not refused.
    "rollout.overlap",
    "rollout.eval_workers",
    "rollout.eval_games_per_worker",
    "checkpoint",
    "metrics",
    "alarms",
    "doctor",
)


class EngineBuild(msgspec.Struct, frozen=True):
    """Which engine, built from which data, with which cards.

    Both digests come from ``ClashParallelEnv.config()``: ``calibration_digest`` over the
    calibration values as loaded, ``build_digest`` over the copies compiled into the extension.
    The harness hashes no data file of its own -- a digest computed here from the data directory
    would be a second opinion about what the engine is running on, and a second opinion is
    exactly what a stale build looks like. ``stale_build_differences`` MUST be empty; it is
    recorded so that a failure is auditable rather than reconstructed.
    """

    engine_class: str
    calibration_digest: str
    build_digest: str
    catalogue_sha256: str
    path_search: str | None
    stale_build_differences: list[str]


class RunIdentity(msgspec.Struct, frozen=True):
    """The identity itself. ``run_id`` is the first sixteen hex characters of its sha256."""

    format_version: int
    royalelearn_version: str
    royalelearn_git: str
    royalegym_version: str
    royalegym_git: str
    engine_build: EngineBuild
    env_spec_digest: str
    obs_digest: str
    action_digest: str
    frame_stack: int
    arch_digest: str
    codec_version: int
    codec_table_digest: str
    algo_digest: str
    rollout_digest: str
    ladder_digest: str
    master_seed: int
    determinism_tier: str
    torch_version: str
    device_kind: str


def run_id(identity: RunIdentity) -> str:
    """The short name of a run: sha256 of the identity's canonical JSON, first sixteen hex."""
    return hashlib.sha256(canonical_json(identity)).hexdigest()[:16]


def identity_differences(
    a: RunIdentity, b: RunIdentity
) -> dict[str, tuple[Any, Any]]:
    """Every field in which two identities differ, so that a refusal can name all of them."""
    out: dict[str, tuple[Any, Any]] = {}
    for name in a.__struct_fields__:
        left, right = getattr(a, name), getattr(b, name)
        if left != right:
            out[name] = (left, right)
    return out


def catalogue_digest(cards: Sequence[Any]) -> str:
    """sha256 over the card catalogue's identifying fields, in card order.

    Name, id, elixir and placement kind: enough that a catalogue with a card renamed, repriced,
    reordered or added is a different game, and little enough that a change to a statistic the
    engine already hashes into its calibration digest does not count twice.
    """
    return digest_of(
        [(card.name, card.card_id, card.elixir, int(card.placement)) for card in cards]
    )


def engine_build(
    env_config: Mapping[str, Any],
    cards: Sequence[Any],
    *,
    path_search: str | None = None,
    stale_build_differences: Sequence[str] = (),
) -> EngineBuild:
    """The ``EngineBuild`` for a built environment, from its own ``config()`` and catalogue."""
    engine = env_config.get("engine") or {}
    return EngineBuild(
        engine_class=str(engine.get("class", UNKNOWN)),
        calibration_digest=str(env_config.get("calibration_digest") or ""),
        build_digest=str(env_config.get("build_digest") or ""),
        catalogue_sha256=catalogue_digest(cards),
        path_search=path_search,
        stale_build_differences=list(stale_build_differences),
    )


@lru_cache(maxsize=8)
def _describe_repo(path: Path) -> str:
    """``git describe --always --dirty`` for the checkout a package was imported from.

    Cached: a checkout does not change inside a process, and this shells out."""
    try:
        done = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return done.stdout.strip() or UNKNOWN if done.returncode == 0 else UNKNOWN


def _repo_root(path: Path) -> Path | None:
    """The checkout ``path`` is in, or None. A package installed from a wheel is in none."""
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _uncommitted(path: Path) -> tuple[str, ...]:
    """The paths under ``path`` that git would not find in its last commit.

    ``git status --porcelain`` and not ``git describe --dirty``, because describe is blind to an
    untracked file, and an untracked file is the case that bites: the config a run is started from
    is often one, and it is the document that says what the run IS.
    """
    root = _repo_root(path)
    if root is None:
        return ()
    try:
        done = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if done.returncode != 0:
        return ()
    return tuple(line[3:].strip() for line in done.stdout.splitlines() if line.strip())


def _watched_paths(config_path: Path | None = None) -> list[tuple[str, Path]]:
    """What a run's behaviour is actually read from, each labelled for a message.

    The PACKAGES rather than their repositories: a docs edit in one of these checkouts cannot
    change what a run does, and a guard that fires on it is a guard people pass a flag to silence.
    The engine's DATA directory is here for the same reason it belongs in a run's identity -- it
    is read at runtime, not compiled in, so an edit to it moves the rules under a running process
    -- and it is resolved the way the code resolves it rather than guessed at. The run's own
    config file is watched because it is the document that says what the run is, and it is
    routinely untracked.
    """
    watched: list[tuple[str, Path]] = []
    for name in ("royalelearn", "royalegym", "royaleviser"):
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError):  # pragma: no cover - a broken install
            continue
        if found is not None and found.origin:
            watched.append((name, Path(found.origin).resolve().parent))
    try:
        from royalegym.protocol import data_dir

        watched.append(("the engine's data", data_dir().resolve()))
    except Exception:  # pragma: no cover - reported elsewhere if the data is missing
        pass
    if config_path is not None:
        watched.append((f"the run's config ({config_path.name})", config_path.resolve()))
    return watched


def dirty_sources(config_path: Path | None = None) -> tuple[str, ...]:
    """What this run would read that is not in any commit, each named with what changed.

    A run's identity records each repository by commit and everything else about the environment
    by NAME: the reward, the observation builder and the action parser are dotted paths, hashed as
    strings. So two different bodies of ``royalegym.reward.default_reward`` produce the same
    ``env_spec_digest``, the ladder files their games under one context, and a plot draws two
    objectives as one line. On 2026-09-22 an uncommitted change to a sibling repo sat under a live
    run for twenty minutes and nothing in the identity could have said so.

    What this covers, stated rather than implied: the python packages this process resolved, the
    engine's data directory, and the config file the run was started from. What it does not: any
    other file in those checkouts, a package installed from a wheel rather than a checkout, and
    anything a component reads at call time from somewhere else. Hashing each resolved component's
    source would be the thorough answer and it is a much larger one.

    The first version of this checked ``git describe --dirty`` on two hardcoded packages. It
    missed the engine's data, missed RoyaleViser, and was blind to untracked files -- including,
    at the time it was written, the config of the run then in flight. Found by the integrator,
    who measured all three rather than arguing them.
    """
    dirty: list[str] = []
    for label, path in _watched_paths(config_path):
        changed = _uncommitted(path)
        if not changed:
            continue
        shown = ", ".join(changed[:3])
        if len(changed) > 3:
            shown += f", and {len(changed) - 3} more"
        dirty.append(f"{label}: {shown}")
    return tuple(dirty)


@lru_cache(maxsize=1)
def royalegym_provenance() -> tuple[str, str]:
    """``(version, git description)`` of the royalegym this process imported.

    Read from the imported module rather than from a pinned number here, because an editable
    install from a sibling checkout is the normal case in this family and its version string
    alone does not say which commit.
    """
    try:
        import royalegym
    except ImportError:  # pragma: no cover - royalegym is a hard dependency of a real run
        return UNKNOWN, UNKNOWN
    version = getattr(royalegym, "__version__", UNKNOWN)
    if version is UNKNOWN:
        try:
            from importlib.metadata import version as metadata_version

            version = metadata_version("royalegym")
        except Exception:
            version = UNKNOWN
    source = Path(royalegym.__file__).resolve().parent.parent
    return str(version), _describe_repo(source)


def torch_version() -> str:
    """The torch this process would train with, or "absent"."""
    try:
        import torch
    except ImportError:
        return "absent"
    return str(torch.__version__)


def describe_device(device: str = "cuda") -> str:
    """``"cuda:<name>:sm_86"`` or ``"cpu:<machine>"``.

    The device is in the identity because a different GPU is a different set of kernels, and
    run-exactness is promised within a device class rather than across all of them.
    """
    if not device.startswith("cuda"):
        return f"cpu:{platform.machine()}"
    try:
        import torch
    except ImportError:
        return f"cpu:{platform.machine()}"
    if not torch.cuda.is_available():
        return f"cpu:{platform.machine()}"
    index = 0
    if ":" in device:
        index = int(device.split(":", 1)[1])
    props = torch.cuda.get_device_properties(index)
    return f"cuda:{props.name}:sm_{props.major}{props.minor}"


def compute_identity(
    config: RunConfig,
    *,
    env_spec: EnvSpec,
    build: EngineBuild,
    arch_digest: str,
    codec_version: int,
    codec_table_digest: str,
    torch_version_string: str | None = None,
    device_kind: str | None = None,
) -> RunIdentity:
    """The identity of the run ``config`` describes, against the environment that was built.

    The arguments beyond the config are the facts only preflight knows: what the environment
    turned out to be, what the codec decided to do with it, and what device and torch build the
    learner will use. They are passed in rather than probed here so that the function is pure
    and a test can vary one field at a time.
    """
    if env_spec.frame_stack != config.obs.frame_stack:
        raise ValueError(
            f"the env spec was read at frame_stack {env_spec.frame_stack}, the config says "
            f"{config.obs.frame_stack}"
        )
    geo = geometry(config)
    gym_version, gym_git = royalegym_provenance()
    return RunIdentity(
        format_version=IDENTITY_FORMAT_VERSION,
        royalelearn_version=__version__,
        royalelearn_git=git_describe(),
        royalegym_version=gym_version,
        royalegym_git=gym_git,
        engine_build=build,
        env_spec_digest=config.env.digest(),
        obs_digest=env_spec.obs_digest,
        action_digest=digest_of(
            {
                "action_parser": config.env.action_parser,
                "n_actions": env_spec.n_actions,
                "hand_size": env_spec.hand_size,
                "tiles": env_spec.tiles,
                "decision_ms": env_spec.decision_ms,
                "decision_ticks": env_spec.decision_ticks,
            }
        ),
        frame_stack=config.obs.frame_stack,
        arch_digest=arch_digest,
        codec_version=codec_version,
        codec_table_digest=codec_table_digest,
        # The PPO and advantage blocks whole. Nearly every leaf in them moves a number, and a
        # list of the two or three that do not would be one more thing to keep true.
        algo_digest=digest_of({"ppo": config.ppo, "advantage": config.advantage}),
        rollout_digest=digest_of(
            {
                "workers": geo.workers,
                "games_per_worker": geo.games_per_worker,
                "shards_per_worker": geo.shards_per_worker,
                "cycles": geo.cycles,
                "n_slots": geo.n_slots,
            }
        ),
        # The ladder block whole, plus the master seed: the evaluation seed set is drawn from
        # ``eval/seed_set`` and is therefore a pure function of the seed and the count, so
        # hashing those two says exactly what hashing the drawn set would and does not put a
        # second copy of the drawing rule in this file.
        ladder_digest=digest_of({"ladder": config.ladder, "master_seed": config.master_seed}),
        master_seed=config.master_seed,
        determinism_tier=config.determinism.tier,
        torch_version=torch_version() if torch_version_string is None else torch_version_string,
        device_kind=describe_device(config.net.device) if device_kind is None else device_kind,
    )
