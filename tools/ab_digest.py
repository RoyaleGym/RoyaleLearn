"""Run one small training config at two commits and say exactly what differs.

The cross-commit blind control of the imitation-extraction plan (RoyaleLive
docs/2026-09-24-il-extraction-plan.md, steps S0-S4). A refactor that claims to change nothing is
held to it column by column, and a step that claims a stated difference is held to exactly that
difference, written as an expect file (``tools/ab_expect/``).

    python tools/ab_digest.py run <commit> --out <dir> [--arm plain|il] [--engine mock|rust]
        [--gym <commit> | --gym-from <result.json>] [--artifacts <dir>]
        [--il-shape legacy|sections|<file>]
        [--imitate <commit>] [--forbid-module NAME]
    python tools/ab_digest.py artifacts <commit> --out <dir> [--gym ... | --gym-from ...]
    python tools/ab_digest.py compare <dirA> <dirB> [--expect expect.json]
    python tools/ab_digest.py check-expect expect.json
    python tools/ab_digest.py selftest --base <commit> --out <dir>
        [--arm plain|il --artifacts <dir>] [--engine mock|rust] [--only key,key,refusals]

HOW A SIDE RUNS. ``run`` resolves every commit to a full sha once, extracts RoyaleLearn and
RoyaleGym with ``git archive`` (and, with ``--imitate``, a ``git clone --shared`` of RoyaleImitate
at that commit) into a folder outside every repository, and runs a fixed tiny config in two child
processes of this venv: a fresh run of three iterations and a checkpoint, then a resume of that
checkpoint for two more. The resume builds its config the way ``royalelearn resume`` does, by
reloading ``<run>/config.json`` through ``cli._run_config`` (a replica only when cli imports
cleanly and lacks it; an unimportable cli fails the side), and refuses if the reloaded
config_hash differs from the fresh one. Each child drops the venv's editable finders for
royalelearn and royalegym before importing anything, so a module missing from the extracted tree
raises as it would on a clean install, and before it writes its record it refuses any
royalelearn*/royalegym* module loaded from outside its tree. Each child gets its own random
PYTHONHASHSEED (never the parent's, never its sibling's); the child records the seed it saw and
a hash probe, and the parent refuses a child that ran on another seed than planned. stdout
and stderr of every child go to ``<out>/<child>.log``.

WHAT A COMPARISON SEES (every column of ``diff``):
- commit, gym_commit, imitate_commit, il_shape: which code and which IL config shape ran.
- environment (never waivable): interpreter, torch/numpy/msgspec/royalesim versions, the resolved
  data dir, sha256 of every data file the engines read (calibration.json, derived/*.json,
  raw/*/csv_logic/*.csv, plus any file under the data dir that Python code opened), RoyaleSim
  HEAD and ``git status`` of its data/, and the royalesim binary when it was loaded. Two results
  are comparable only when this column is equal; a stored baseline is re-run when it is not.
- engine_binary (never waivable): identity.engine_build.binary_sha256.
- config_json_sha256, config_hash, resumed_config_hash, resume_config_via, config_keys (every
  leaf of config.json, {} and [] kept as leaves, lists by index, compared by JSON text so
  int/float/bool show; a key moved by a config_prefix rename is compared by value under its
  new name), config_order (never waivable: the leaf order of both files once the differing
  keys and the renamed keys are removed).
- run_id, resumed_run_id, identity_fields (every identity leaf, only_a/only_b/changed).
- weights, moments, state_digest and their resumed_* twins.
- metric_keys (presence per row over every key except throughput/gpu_util_frac and the
  conditional health/vram_* keys) and metric_values (every value by JSON text, NaN == NaN, except
  the values of time/*, throughput/* other than discarded_rows_frac, run/wall_seconds,
  health/rss_peak_mb and health/vram_*), row_count. A changed key's entry is [first row, A text,
  B text, rows changed, rows_sha]; rows_sha (16 hex) hashes every changed (row, A, B), so a value
  pin on the entry covers every row, not only the first.
- alarms and resumed_alarms (the AlarmSet name list of each process), alarm_order (never
  waivable: the order of the names on both sides, renamed names left out), alarm_duplicates
  (never waivable), alarm_params ((severity, patience, keys) per alarm per process, a one-sided
  alarm's as only_a/only_b), alarm_firings (every alarms.jsonl row, grouped by alarm name and
  in order within it: name, severity, iteration, consecutive, values, fresh and resumed; an
  alarm that fires on one side only is only_a/only_b and must be allowed by name). AlarmSet
  emits in table order, so with alarm_order equal the grouped rows are the file's rows.
- checkpoint: for the fresh and the resumed checkpoint, the manifest minus created_unix_ns,
  wall_seconds and gate_seconds (its files sha map minus rng/rng.json; its config and identity as
  equality flags against the run's config.json and identity.json, which the config and identity
  columns compare) and every JSON file in the checkpoint parsed (schedules, optimizers/misc,
  rollout, matchmaker, ladder, rating, advantage, metrics, arch, rng minus python_random).
- other_modules: every royale* module other than royalelearn/royalegym that was loaded, with its
  file.
- rename_problems (never waivable): an expect rename that matched nothing on side A.

WHAT IT CANNOT SEE: anything the tiny config does not run (rollout workers, snapshot opponents, the
ladder refit, eval and gate, the probe, CUDA and bfloat16, frame stacks, the recorder, long
schedules); the values of clock and machine keys; rng.json's python_random, which differs between
two identical runs; files the Rust engine reads natively outside the fixed data list; installed
distribution metadata, which still comes from the venv (royalegym_version; ``--imitate`` writes the
pinned clone's own entry points and ``direct_url.json`` beside it, which RoyaleLearn's discovery
takes, and a copy installed in the venv that declares the same entry points is merged with it); the
BLAS thread variables of the operator's shell (inherited, not recorded); a hash-order dependence the
two seeds happen to order alike (a different seed per process gives a chance, not a guarantee); the
content of the IL artifacts' writer (both sides read the same bytes, pinned by sha256 in the
config). A resumed AlarmSet restarts its patience counters by design, so a firing that needs more
than two rows of patience cannot happen in the resumed process.

``compare`` prints commit, gym_commit, arm and engine of both sides first, refuses sides whose arm
or engine differ, then prints every difference and, with ``--expect``, exits non-zero on any
difference the expect file does not allow and on any difference it requires that did not happen.
``selftest`` runs the base twice on different hash seeds (the null test) and then once per plant,
each planted in the extracted source at float32 resolution with the config unchanged; each plant
names the columns (and exact entries) it must move, and some name columns they must leave
equal. It then feeds plant diffs and mutated copies of the real ref results to ``unexpected``
with near-miss expect files (every column check and every expect rule, each refused for the
reason it was built to show, beside controls that must pass), feeds mutated ref records to
the parent's refusals, and on the plain arm runs sides that must fail inside a child: an
unimportable cli, a missing module (with and without the finder drop), an actor that never
trains, config.json edited between fresh and resume, alarms.jsonl rewritten, and a forbidden
module imported, plus a --forbid-module side that must pass and equal the ref. A comparator
that has not been seen failing is not a comparator.

Before each child process the tool waits (a minute at a time, ``--wait-minutes`` at most) until
``--min-free-mb`` of memory is free: the machine is shared with long jobs.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
LEARN_REPO = HERE.parent.parent
WORKSPACE = LEARN_REPO.parent

ITERATIONS, AFTER = 3, 2
MASTER_SEED = 20260925
#: Steps a Rust episode runs past the opening lockout, and steps an iteration collects past it.
RUST_MARGIN_STEPS, RUST_ITERATION_EXTRA = 16, 7
EXPECT_FORMAT = 2
RESULT_FORMAT = 2

# --------------------------------------------------------------------------
# Repositories, commits, memory
# --------------------------------------------------------------------------


def _is_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _find_repo(name: str) -> Path:
    """A sibling checkout: the workspace's, else the one beside a worktree
    (``.worktrees/RoyaleLearn-learn`` sits two levels below the workspace)."""
    for root in (WORKSPACE, WORKSPACE.parent):
        if _is_repo(root / name):
            return root / name
    raise SystemExit(f"no {name} checkout found near {LEARN_REPO}")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _learn_root() -> Path:
    return Path(_git(HERE.parent, "rev-parse", "--show-toplevel"))


def _resolve(repo: Path, commit: str) -> str:
    """A full sha, resolved once per invocation and passed everywhere after that."""
    try:
        return _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{commit!r} is not a commit of {repo}: {exc.stderr.strip()}") from exc


def _extract(repo: Path, sha: str, into: Path) -> None:
    """``git archive <sha>`` of ``repo`` into ``into``."""
    blob = subprocess.run(
        ["git", "archive", "--format=tar", sha], cwd=repo, capture_output=True, check=True
    ).stdout
    if into.exists():
        shutil.rmtree(into)
    into.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        archive.extractall(into, filter="data")


def _clone_imitate(sha: str, into: Path) -> None:
    """RoyaleImitate at ``sha`` as a real checkout (so its git provenance is a sha), made with
    ``git clone --shared`` so the source repository is not touched."""
    source = _find_repo("RoyaleImitate")
    if into.exists():
        shutil.rmtree(into)
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", "--quiet", str(source), str(into)],
        check=True,
    )
    subprocess.run(["git", "-C", str(into), "checkout", "--quiet", "--detach", sha], check=True)
    if not (into / "royaleimitate" / "__init__.py").exists():
        raise SystemExit(
            f"RoyaleImitate at {sha[:12]} has no royaleimitate package (step S4 has not landed "
            "there): --imitate has nothing to pin"
        )
    _write_install_metadata(into)


def _write_install_metadata(root: Path) -> None:
    """The install metadata ``pip install -e`` would leave for the pinned clone, written beside
    its package: the entry points its own pyproject.toml declares, and a ``direct_url.json``
    naming the clone, so RoyaleLearn's discovery finds the sections in the pinned code and not in
    whatever the venv has installed."""
    import json
    import tomllib

    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    info = root / f"{project['name']}-{project['version']}.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {project['name']}\nVersion: {project['version']}\n",
        encoding="utf-8",
    )
    groups = project.get("entry-points", {})
    lines = []
    for group, points in groups.items():
        lines.append(f"[{group}]")
        lines.extend(f"{name} = {value}" for name, value in points.items())
    (info / "entry_points.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    direct = {"url": root.resolve().as_uri(), "dir_info": {"editable": True}}
    (info / "direct_url.json").write_text(json.dumps(direct), encoding="utf-8")
    # Written into the clone, so it would read as an uncommitted edit of the package's checkout:
    # the clone's own exclude file keeps it out of git status, as a venv's copy lives elsewhere.
    exclude = root / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write(f"\n/{info.name}/\n")


def _refuse_inside_a_repo(out: Path) -> None:
    """The extracted trees must sit outside every git work tree.

    ``git describe`` walks up from the package's folder, so a tree inside a checkout would record
    that checkout's commit as its own ``royalelearn_git``. Outside every repo both sides read
    ``unknown``, which names no commit it did not run; the commit columns name the real ones.
    """
    probe = out
    while not probe.exists():
        probe = probe.parent
    done = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=probe, capture_output=True, text=True
    )
    if done.returncode == 0:
        raise SystemExit(
            f"{out} is inside the git work tree {done.stdout.strip()}: put the output outside "
            "every repository, or the trees' git provenance would be that repository's"
        )


def _free_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.virtual_memory().available / 2**20
    except Exception:
        pass
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong),
                ("available", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("extended", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.length = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return status.available / 2**20
    return None


def _ram_gate(minimum: float, wait_minutes: float) -> None:
    deadline = time.monotonic() + wait_minutes * 60.0
    while True:
        free = _free_mb()
        if free is None:
            raise SystemExit("cannot read free memory, and a comparison does not guess")
        if free >= minimum:
            return
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"only {free:.0f} MB free after waiting {wait_minutes:.0f} min and --min-free-mb "
                f"is {minimum:.0f}: not starting a child process on a shared machine"
            )
        print(f"  {free:.0f} MB free, below {minimum:.0f}: waiting a minute", flush=True)
        time.sleep(60)


_USED_SEEDS: set[int] = set()


def _hash_seed() -> int:
    """A PYTHONHASHSEED no other child of this invocation had and the parent does not have."""
    parent = os.environ.get("PYTHONHASHSEED", "")
    if parent.isdigit():
        _USED_SEEDS.add(int(parent))
    draw = random.SystemRandom()
    while True:
        seed = draw.randint(1, 2**32 - 1)
        if seed not in _USED_SEEDS:
            _USED_SEEDS.add(seed)
            return seed


# --------------------------------------------------------------------------
# The configs: fixed JSON, never a test helper, so both sides read the same bytes
# --------------------------------------------------------------------------

#: The IL arm's config fragments. Placeholders are filled from the artifacts folder. ``legacy``
#: is 490c74c's block; ``sections`` is S3's layout as of 7cd82d0 (royalelearn/imitation/config.py
#: WarmStartSection and ImitationSection): warm_start holds init, actor_lr_scale and the freeze
#: alarms' thresholds, imitation holds references, regularisers and the regularisers' alarms'
#: thresholds. Each ``alarms`` block is written out at the values ``legacy`` runs with (490c74c's
#: AlarmConfig.imitation_* defaults), so the two shapes name the same experiment.
IL_SHAPES: dict[str, dict[str, Any]] = {
    "legacy": {
        "imitation": {
            "init": {"path": "$INIT", "sha256": "$INIT_SHA256"},
            "actor_lr_scale": {"kind": "piecewise", "points": [[0, 0.0], [1, 1.0]]},
            "references": {
                "bc": {"kind": "snapshot", "path": "$REFERENCE", "sha256": "$REFERENCE_SHA256"}
            },
            "regularisers": [
                {
                    "kind": "reference_kl",
                    "name": "bc",
                    "reference": "bc",
                    "budget": {"kind": "constant", "value": 0.0},
                    "coef": {"start": 2.5, "max": 10.0, "up": 1.5},
                }
            ],
        }
    },
    "sections": {
        "warm_start": {
            "init": {"path": "$INIT", "sha256": "$INIT_SHA256"},
            "actor_lr_scale": {"kind": "piecewise", "points": [[0, 0.0], [1, 1.0]]},
            "alarms": {
                "handoff_window": 20,
                "handoff_kl": 0.05,
                "handoff_clip": 0.3,
                "ev_at_unfreeze": 0.3,
            },
        },
        "imitation": {
            "references": {
                "bc": {"kind": "snapshot", "path": "$REFERENCE", "sha256": "$REFERENCE_SHA256"}
            },
            "regularisers": [
                {
                    "kind": "reference_kl",
                    "name": "bc",
                    "reference": "bc",
                    "budget": {"kind": "constant", "value": 0.0},
                    "coef": {"start": 2.5, "max": 10.0, "up": 1.5},
                }
            ],
            "alarms": {"ref_kl_warn": 1.0, "lambda_saturated_patience": 10},
        },
    },
}


def _il_fragment(shape: str) -> tuple[dict[str, Any], dict[str, str]]:
    """The fragment for ``shape`` (a name above or a JSON file) and its record."""
    if shape in IL_SHAPES:
        fragment = copy.deepcopy(IL_SHAPES[shape])
        label = shape
    else:
        path = Path(shape)
        if not path.is_file():
            raise SystemExit(f"--il-shape {shape!r} is neither {sorted(IL_SHAPES)} nor a file")
        fragment = json.loads(path.read_text(encoding="utf-8"))
        label = f"file:{path.name}"
    text = json.dumps(fragment, sort_keys=True)
    return fragment, {"name": label, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def _fill(value: Any, made: dict[str, str]) -> Any:
    table = {
        "$INIT": made["init"],
        "$INIT_SHA256": made["init_sha256"],
        "$REFERENCE": made["reference"],
        "$REFERENCE_SHA256": made["reference_sha256"],
    }
    if isinstance(value, dict):
        return {k: _fill(v, made) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, made) for v in value]
    if isinstance(value, str) and value in table:
        return table[value]
    return value


def _regulariser_names(fragment: dict[str, Any]) -> list[str]:
    names = []
    for section in fragment.values():
        if isinstance(section, dict):
            for regulariser in section.get("regularisers", []) or []:
                names.append(regulariser["name"])
    return names


def _config(
    engine: str,
    arm: str,
    *,
    fragment: dict[str, Any] | None = None,
    rust: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The tiny config both sides run."""
    engine_class = {
        "mock": "royalegym.mock_engine.MockEngine",
        "rust": "royalegym.rust_engine.RustEngine",
    }[engine]
    max_steps, timesteps = 4, 16
    if engine == "rust":
        assert rust is not None
        max_steps, timesteps = rust["max_steps"], rust["timesteps_per_iteration"]
    config: dict[str, Any] = {
        "run_name": f"ab-{engine}-{arm}",
        "runs_dir": "runs",
        "master_seed": MASTER_SEED,
        "profile": "laptop",
        "env": {
            "engine": {"cls": engine_class, "kwargs": {}},
            "obs_builder": {"cls": "royalegym.obs.SpatialObsBuilder", "kwargs": {}},
            "action_parser": {"cls": "royalegym.action.TileActionParser", "kwargs": {}},
            "reward_fn": {"cls": "royalelearn.rewards.default_potential_reward", "kwargs": {}},
            "state_mutator": {"cls": "royalegym.state_mutator.DefaultStateMutator", "kwargs": {}},
            "termination": [{"cls": "royalegym.done_condition.GameOverCondition", "kwargs": {}}],
            "truncation": [
                {
                    "cls": "royalegym.done_condition.StepLimitCondition",
                    "kwargs": {"max_steps": max_steps},
                }
            ],
            "decision_ms": 500,
        },
        "rollout": {
            "source": "inline",
            "workers": 1,
            "games_per_worker": 2,
            "shards_per_worker": 1,
            "launch_delay_s": 0.0,
            "stagger_first_reset": False,
        },
        "net": {
            "channels": 8,
            "blocks": 1,
            "norm_groups": 4,
            "card_embed": 8,
            "value_hidden": 16,
            "autocast_dtype": "float32",
            "device": "cpu",
        },
        "ppo": {
            "n_epochs": 2,
            "timesteps_per_iteration": timesteps,
            "batch_size": 8,
            "minibatch_size": 4,
            "ratio_atol": {"fp32": 1e-4, "float32": 1e-4, "bfloat16": 2e-2},
        },
        "ladder": {
            "mix": [1.0, 0.0, 0.0],
            "candidate_every_env_steps": 1000000,
            "floor_admit_every_env_steps": 1000000,
            "eval_seed_count": 4,
            "refit_every_iterations": 1000000,
        },
        "checkpoint": {"every_env_steps": 1, "keep": 8},
        "metrics": {"sinks": [{"kind": "jsonl"}]},
        "determinism": {"tier": "run_exact"},
        "doctor": {"run_mask_disagreement_gate": False},
    }
    if arm == "il":
        if fragment is None:
            raise SystemExit("the il arm needs its artifacts and a shape")
        config.update(fragment)
    return config


def _rust_sizes(rules: dict[str, int]) -> dict[str, int]:
    """Episode and iteration lengths that get past the opening lockout.

    Every action is a forced no-op while ``tick < deploy_lockout_ticks``. An episode runs
    RUST_MARGIN_STEPS decisions past it, and an iteration collects more decisions per game than
    the lockout lasts, so every window of an iteration's length holds post-lockout decisions.
    """
    decision_ticks = max(1, -(-500 // max(1, rules["tick_ms"])))
    lock_steps = -(-rules["deploy_lockout_ticks"] // decision_ticks)
    per_game = lock_steps + RUST_ITERATION_EXTRA
    return {
        **rules,
        "decision_ticks": decision_ticks,
        "lock_steps": lock_steps,
        "max_steps": lock_steps + RUST_MARGIN_STEPS,
        "steps_per_game_per_iteration": per_game,
        # two games, two seats: four learner rows per decision
        "timesteps_per_iteration": 4 * per_game,
    }


# --------------------------------------------------------------------------
# The child processes
# --------------------------------------------------------------------------

#: Shared by every child. Runs before anything royale is imported.
CHILD_PRELUDE = r'''
import builtins, hashlib, io, json, os, subprocess, sys
from pathlib import Path

JOB = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
_OURS = ("royalelearn", "royalegym", "royaleimitate")
_early = sorted(n for n in sys.modules if n.split(".")[0] in _OURS)
if _early:
    raise SystemExit(f"imported before the finders were dropped: {_early}")


def _drop_editable_finders():
    """Remove only the editable finders whose MAPPING names royalelearn or royalegym: they
    resolve a submodule missing from the tree to the live checkout."""
    kept, dropped = [], []
    for finder in sys.meta_path:
        module = sys.modules.get(getattr(finder, "__module__", "") or "")
        mapping = getattr(module, "MAPPING", None) or {}
        if {"royalelearn", "royalegym"} & set(mapping):
            dropped.append(module.__name__)
        else:
            kept.append(finder)
    sys.meta_path[:] = kept
    return dropped


#: ``keep_finders`` exists for the self-test only: it shows that sweep_modules refuses the leak
#: the finder drop prevents.
DROPPED_FINDERS = [] if JOB.get("keep_finders") else _drop_editable_finders()
#: What this process actually runs on, not what the parent planned: the parent checks the two.
CHILD_SEED = {"env": os.environ.get("PYTHONHASHSEED"), "probe": hash("ab_digest")}

_OPENED = set()
_real_open = builtins.open


def _recording_open(file, *args, **kwargs):
    try:
        if isinstance(file, (str, bytes, os.PathLike)):
            _OPENED.add(os.path.abspath(os.fsdecode(file)))
    except Exception:
        pass
    return _real_open(file, *args, **kwargs)


builtins.open = _recording_open
io.open = _recording_open

ROOTS = {"royalelearn": JOB.get("learn_tree"), "royalegym": JOB["gym_tree"]}
if JOB.get("imitate_tree"):
    ROOTS["royaleimitate"] = JOB["imitate_tree"]


def _norm(path):
    return os.path.normcase(os.path.realpath(path))


def _inside(path, root):
    if not root:
        return False
    path, root = _norm(path), _norm(root)
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _where(path):
    places = [(f"<{name}>", root) for name, root in ROOTS.items() if root]
    places.append(("<venv>", sys.prefix))
    for label, root in places:
        if _inside(path, root):
            rel = os.path.relpath(_norm(path), _norm(root)).replace(os.sep, "/")
            return f"{label}/{rel}"
    return str(path).replace(os.sep, "/")


def sweep_modules():
    """Refuse a royalelearn*/royalegym* (and a pinned royaleimitate*) module from outside its
    tree; return every other royale* module with its file."""
    others = {}
    for name in sorted(sys.modules):
        top = name.split(".")[0]
        if not top.startswith("royale"):
            continue
        module = sys.modules[name]
        if module is None:
            continue
        place = getattr(module, "__file__", None)
        places = [place] if place else list(getattr(module, "__path__", []) or [])
        if top in ROOTS and ROOTS[top]:
            bad = [p for p in places if not _inside(p, ROOTS[top])]
            if bad or not places:
                where = bad or "nowhere"
                raise SystemExit(f"{name} was loaded from {where}, not from {ROOTS[top]}")
        if top not in ("royalelearn", "royalegym"):
            others[name] = [_where(p) for p in places]
    return others


def _sha_file(path):
    digest = hashlib.sha256()
    with _real_open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    """What the run read that is not either commit."""
    import importlib.metadata as metadata

    env = {"executable": sys.executable, "python": sys.version, "packages": {}}
    for name in ("torch", "numpy", "msgspec", "royalesim"):
        try:
            env["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            env["packages"][name] = None
    from royalegym.protocol import data_dir

    data = Path(data_dir()).resolve()
    env["data_dir"] = str(data)
    files = {data / "calibration.json"}
    files |= set((data / "derived").glob("*.json"))
    files |= set(data.glob("raw/*/csv_logic/*.csv"))
    files |= {Path(p) for p in _OPENED if _inside(p, data) and os.path.isfile(p)}
    env["data_files"] = {
        p.resolve().relative_to(data).as_posix(): _sha_file(p) for p in sorted(files) if p.is_file()
    }
    try:
        def git(*args, cwd):
            done = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
            )
            return done.stdout

        top = git("rev-parse", "--show-toplevel", cwd=data).strip()
        env["royalesim_head"] = git("rev-parse", "HEAD", cwd=top).strip()
        status = git("status", "--porcelain", "--", "data", cwd=top)
        env["royalesim_data_status"] = status.splitlines()
    except Exception as exc:
        env["royalesim_head"] = f"unreadable: {exc!r}"
        env["royalesim_data_status"] = None
    sim = sys.modules.get("royalesim")
    if sim is not None and getattr(sim, "__file__", None):
        folder = Path(sim.__file__).parent
        env["royalesim_binary"] = {
            p.name: _sha_file(p)
            for p in sorted(folder.iterdir())
            if p.suffix in (".pyd", ".so", ".dll")
        }
    return env


def flatten(value, prefix, out):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            flatten(item, f"{prefix}{key}.", out)
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            flatten(item, f"{prefix}{index}.", out)
    else:
        out[prefix[:-1]] = json.dumps(value, sort_keys=True)
    return out
'''

RUNNER = (
    CHILD_PRELUDE
    + r'''
MODE = JOB["mode"]
import royalelearn, royalegym
# What `python -m royalelearn` imports first (BLAS and CUBLAS env). Unguarded: cli.py exists at
# 8bf6a67 and every later commit, and a tree whose cli cannot be imported cannot start a run.
import royalelearn.cli
import torch
from royalelearn.config import config_hash, load_config
from royalelearn.coordinator import LearningCoordinator

PREFLIGHT = {"table_samples": 32, "mask_samples": 32, "min_table_states": 0}
QUIET = {"printer": None, "install_signal_handler": False, "preflight_kwargs": PREFLIGHT}


def tensors_sha(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(str(name).encode())
        if torch.is_tensor(tensor):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(tensor, sort_keys=True, default=repr).encode())
    return digest.hexdigest()


def moments_sha(run):
    digest = hashlib.sha256()
    for optimizer in run.update.optimizers:
        state = optimizer.state_dict()["state"]
        for key in sorted(state):
            digest.update(str(key).encode())
            digest.update(tensors_sha(dict(state[key])).encode())
    return digest.hexdigest()


def alarm_table(run):
    return {
        alarm.name: [alarm.severity, int(alarm.patience), list(alarm.keys)]
        for alarm in run.alarms.alarms
    }


def alarm_names(run):
    return [alarm.name for alarm in run.alarms.alarms]


def firings(run_dir):
    path = Path(run_dir) / "alarms.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            fields = ("name", "severity", "iteration", "consecutive", "values")
            rows.append({k: row.get(k) for k in fields})
    return rows


def checkpoint_record(path, run_dir):
    path, run_dir = Path(path), Path(run_dir)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for key, value in manifest.items():
        if key in ("created_unix_ns", "wall_seconds", "gate_seconds"):
            continue
        if key == "files":
            for name, sha in value.items():
                if name != "rng/rng.json":
                    out[f"manifest:files.{name}"] = json.dumps(sha)
            continue
        if key == "config":
            written = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            out["manifest:config==config.json"] = json.dumps(value == written)
            continue
        if key == "identity":
            written = json.loads((run_dir / "identity.json").read_text(encoding="utf-8"))
            out["manifest:identity==identity.json"] = json.dumps(value == written)
            continue
        flatten(value, f"manifest:{key}.", out)
    for file in sorted(path.rglob("*.json")):
        rel = file.relative_to(path).as_posix()
        if rel == "manifest.json":
            continue
        data = json.loads(file.read_text(encoding="utf-8"))
        if rel == "rng/rng.json" and isinstance(data, dict):
            data.pop("python_random", None)
        flatten(data, f"{rel}:", out)
    return out


def rows_of(run_dir):
    text = (Path(run_dir) / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def exercised(rows):
    """A side that never played a card or moved its actor compares nothing about either."""
    choice, grad = "policy/rollout_choice_frac", "ppo/grad_norm_actor"
    if not any(_positive(r.get(choice)) and _positive(r.get(grad)) for r in rows):
        raise SystemExit(
            "this side never exercised the actor: no row has policy/rollout_choice_frac > 0 and "
            "ppo/grad_norm_actor > 0, so a comparison would say nothing about action choice or "
            "the actor update"
        )
    for name in JOB.get("il_terms", []):
        kl, lam = f"imitation/{name}/kl", f"imitation/{name}/lambda"
        if not any(_positive(r.get(kl)) and _positive(r.get(lam)) for r in rows):
            raise SystemExit(
                f"the IL arm never exercised {name!r}: no row has {kl} > 0 and {lam} > 0"
            )


if MODE == "fresh":
    try:
        config = load_config(Path(JOB["config"]))
    except BaseException as exc:
        raise SystemExit(
            f"this commit's load_config refused the tool's config (arm {JOB['arm']}, IL shape "
            f"{JOB.get('il_shape')}): {exc!r}"
        )
    record = {}
    run = LearningCoordinator(config, **QUIET)
    with run:
        for _ in range(JOB["iterations"]):
            run.iterate()
        checkpoint = run.checkpoint()
        record.update(
            run_dir=str(run.run_dir),
            checkpoint=str(checkpoint),
            run_id=run.run_id,
            alarms=alarm_names(run),
            alarm_params=alarm_table(run),
            weights=tensors_sha(run.model.state_dict()),
            moments=moments_sha(run),
            state_digest=run.state_digest(),
        )
    run_dir = Path(record["run_dir"])
    written = (run_dir / "config.json").read_bytes()
    record["config_json_sha256"] = hashlib.sha256(written).hexdigest()
    record["config_json_text"] = written.decode("utf-8")
    record["config_hash"] = config_hash(config)
    record["identity"] = json.loads((run_dir / "identity.json").read_text(encoding="utf-8"))
    record["alarm_firings"] = {"fresh": firings(run_dir)}
    record["checkpoint_fresh"] = checkpoint_record(checkpoint, run_dir)
    record["environment"] = {"fresh": environment()}
    record["dropped_finders"] = {"fresh": DROPPED_FINDERS}
    record["child_hashseed"] = {"fresh": CHILD_SEED}
    record["modules"] = {"fresh": sweep_modules()}
else:
    record = json.loads(Path(JOB["result"]).read_text(encoding="utf-8"))
    run_dir = Path(record["run_dir"])
    # The replica is only for a cli that imports cleanly and has no _run_config; an ImportError
    # of cli itself is this tree's failure and propagates.
    _run_config = getattr(royalelearn.cli, "_run_config", None)
    if _run_config is not None:
        config = _run_config(run_dir)
        via = "royalelearn.cli._run_config"
    else:
        from royalelearn.config import validate

        config = load_config(run_dir / "config.json")
        config.runs_dir = str(run_dir.parent)
        config.run_name = run_dir.name.rsplit("-", 1)[0]
        config = validate(config)
        via = "replicated cli._run_config"
    reloaded = config_hash(config)
    if reloaded != record["config_hash"]:
        raise SystemExit(
            f"<run>/config.json does not load back to the config that wrote it: config_hash "
            f"{reloaded} after reloading, {record['config_hash']} fresh ({via})"
        )
    run = LearningCoordinator(
        config, resume=record["checkpoint"], run_dir=record["run_dir"], **QUIET
    )
    with run:
        for _ in range(JOB["iterations"]):
            run.iterate()
        checkpoint = run.checkpoint()
        record.update(
            resumed_checkpoint=str(checkpoint),
            resumed_weights=tensors_sha(run.model.state_dict()),
            resumed_moments=moments_sha(run),
            resumed_state_digest=run.state_digest(),
            resumed_run_id=run.run_id,
            resumed_alarms=alarm_names(run),
            resumed_alarm_params=alarm_table(run),
        )
    record["resumed_config_hash"] = reloaded
    record["resume_config_via"] = via
    every = firings(run_dir)
    fresh = record["alarm_firings"]["fresh"]
    if every[: len(fresh)] != fresh:
        raise SystemExit("alarms.jsonl lost or rewrote the fresh process's rows on resume")
    record["alarm_firings"]["resumed"] = every[len(fresh):]
    record["checkpoint_resumed"] = checkpoint_record(checkpoint, run_dir)
    record["rows"] = rows_of(run_dir)
    exercised(record["rows"])
    record["environment"]["resumed"] = environment()
    record["dropped_finders"]["resumed"] = DROPPED_FINDERS
    record["child_hashseed"]["resume"] = CHILD_SEED
    record["modules"]["resumed"] = sweep_modules()
Path(JOB["result"]).write_text(json.dumps(record, indent=1), encoding="utf-8")
'''
)

ARTIFACT_MAKER = (
    CHILD_PRELUDE
    + r"""
import numpy as np
import torch
from royalelearn.config import load_config
from royalelearn.coordinator import LearningCoordinator
try:
    from royalelearn.imitation.artifacts import ProbeSet, probe_log_probs, write_actor_artifact
except ImportError:
    from royalelearn.artifacts import ProbeSet, probe_log_probs, write_actor_artifact

PREFLIGHT = {"table_samples": 32, "mask_samples": 32, "min_table_states": 0}
config = load_config(Path(JOB["config"]))
out = Path(JOB["out_dir"])
quiet = {"printer": None, "install_signal_handler": False, "preflight_kwargs": PREFLIGHT}
with LearningCoordinator(config, **quiet) as run:
    seeded = {k: v.detach().clone() for k, v in run.model.actor.state_dict().items()}
    run.iterate()
    buffer = run.buffer
    cycles, slots = np.nonzero(buffer.trainable())
    rows, _ = buffer._stack_rows(cycles[:12], slots[:12])
    probe_rows = np.ascontiguousarray(buffer.obs_view[rows[:, 0]])
    template = run._artifact_spec() if hasattr(run, "_artifact_spec") else run.artifact_spec()
    codec = run.row_codec()
    made = {}
    # init is NOT the seeded actor: a run that silently skipped loading it would otherwise
    # start from the very weights the init holds and pass.
    for name, scale in (("init", 0.5), ("reference", 3.0)):
        state = {k: v * scale if v.is_floating_point() else v for k, v in seeded.items()}
        build = getattr(run, "_build_actor", None) or run.build_actor
        actor = build(run.device)
        actor.load_state_dict(state)
        actor.eval()
        log_probs, _mask = probe_log_probs(actor, codec, probe_rows)
        probe = ProbeSet(probe_rows, log_probs)
        digest = write_actor_artifact(out / name, state, template, probe=probe)
        made[name] = str(out / name)
        made[name + "_sha256"] = digest
    made["init_scale"], made["reference_scale"] = 0.5, 3.0
sweep_modules()
(out / "artifacts.json").write_text(json.dumps(made, indent=1), encoding="utf-8")
"""
)

PROBE = (
    CHILD_PRELUDE
    + r"""
from royalegym.protocol import default_calibration
from royalegym.rust_engine import RustEngine

engine = RustEngine()
rules = {
    "deploy_lockout_ticks": int(engine.rules().deploy_lockout_ticks),
    "tick_ms": int(default_calibration().int("time.TICK_MS")),
}
sweep_modules()
Path(JOB["result"]).write_text(json.dumps(rules), encoding="utf-8")
"""
)


def _import_lines(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if line.startswith("import time:")]


def _child(
    script: str,
    job: dict[str, Any],
    *,
    name: str,
    out: Path,
    trees: list[Path],
    run: RunOptions,
    hashseed: int,
) -> None:
    _ram_gate(run.min_free, run.wait_minutes)
    job_path = out / f"{name}.job.json"
    job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(tree) for tree in trees)
    env["ROYALESIM_DATA_DIR"] = str(_find_repo("RoyaleSim") / "data")
    env["PYTHONHASHSEED"] = str(hashseed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPROFILEIMPORTTIME", None)
    if run.forbid_module:
        env["PYTHONPROFILEIMPORTTIME"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", script, str(job_path)],
        cwd=out,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log = out / f"{name}.log"
    log.write_text(
        f"# {name}: exit {done.returncode}, PYTHONHASHSEED={hashseed}\n"
        f"== stdout ==\n{done.stdout}\n== stderr ==\n{done.stderr}",
        encoding="utf-8",
    )
    if done.returncode != 0:
        raise SystemExit(
            f"child {name} failed ({done.returncode}); its whole output is in {log}\n"
            f"{done.stdout[-2000:]}\n{done.stderr[-6000:]}"
        )
    if run.forbid_module:
        lines = _import_lines(done.stderr)
        if not lines:
            raise SystemExit(
                f"--forbid-module {run.forbid_module}: child {name} wrote no import-time lines, "
                "so PYTHONPROFILEIMPORTTIME did not reach it and an absence would prove nothing"
            )
        hits = []
        for line in lines:
            module = line.rsplit("|", 1)[-1].strip()
            if module == run.forbid_module or module.startswith(run.forbid_module + "."):
                hits.append(module)
        if hits:
            raise SystemExit(f"--forbid-module: child {name} imported {sorted(set(hits))} ({log})")


# --------------------------------------------------------------------------
# One side
# --------------------------------------------------------------------------


@dataclass
class RunOptions:
    min_free: float = 700.0
    wait_minutes: float = 20.0
    forbid_module: str | None = None


@dataclass
class Side:
    learn_sha: str
    gym_sha: str
    arm: str = "plain"
    engine: str = "mock"
    artifacts: Path | None = None
    il_shape: str = "legacy"
    imitate_sha: str | None = None
    learn_tree: Path | None = None  # a planted copy instead of the extracted commit
    planted: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: Self-test only: keep the editable finders, and a mutation between fresh and resume.
    keep_finders: bool = False
    between: str | None = None


def _between(kind: str, out: Path) -> None:
    """Self-test mutations applied to a side's run folder after its fresh process."""
    record = json.loads((out / "result.json").read_text(encoding="utf-8"))
    run_dir = out / record["run_dir"]  # relative to the child's cwd, which is ``out``
    if kind == "config_edit":
        path = run_dir / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["ppo"]["n_epochs"] += 1
        path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    elif kind == "alarms_rewrite":
        path = run_dir / "alarms.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        if not lines:
            raise SystemExit("between alarms_rewrite: the fresh process wrote no alarm row")
        row = json.loads(lines[0])
        row["consecutive"] = int(row.get("consecutive") or 0) + 100
        lines[0] = json.dumps(row)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise SystemExit(f"unknown between mutation {kind!r}")


def _check_child_seeds(record: dict[str, Any], planned: dict[str, int]) -> None:
    """Each child ran on the hash seed the parent planned for it (a shell-exported
    PYTHONHASHSEED, or a lost env entry, would otherwise go unrecorded)."""
    for process in ("fresh", "resume"):
        ran = (record.get("child_hashseed") or {}).get(process) or {}
        if ran.get("env") != str(planned[process]):
            raise SystemExit(
                f"the {process} child ran with PYTHONHASHSEED={ran.get('env')!r}, "
                f"not the planned {planned[process]}"
            )


def run_side(side: Side, out: Path, run: RunOptions) -> Path:
    """One side: extract, write the config, run fresh then resume. Returns result.json."""
    _refuse_inside_a_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "runs").exists() or (out / "result.json").exists():
        raise SystemExit(f"{out} already holds a run: a side runs in a fresh folder")
    trees = out / "tree"
    learn_path, gym_path = trees / "RoyaleLearn", trees / "RoyaleGym"
    if side.learn_tree is None:
        _extract(_learn_root(), side.learn_sha, learn_path)
    else:
        shutil.copytree(side.learn_tree, learn_path)
    _extract(_find_repo("RoyaleGym"), side.gym_sha, gym_path)
    path_trees = [learn_path, gym_path]
    imitate_path = None
    if side.imitate_sha:
        imitate_path = trees / "RoyaleImitate"
        _clone_imitate(side.imitate_sha, imitate_path)
        path_trees.insert(0, imitate_path)
    base_job = {
        "learn_tree": str(learn_path),
        "gym_tree": str(gym_path),
        "imitate_tree": str(imitate_path) if imitate_path else None,
    }

    rust = None
    if side.engine == "rust":
        probe_out = out / "rust_rules.json"
        _child(
            PROBE,
            {**base_job, "learn_tree": None, "result": str(probe_out)},
            name="rust_rules",
            out=out,
            trees=[gym_path],
            run=RunOptions(run.min_free, run.wait_minutes, None),
            hashseed=_hash_seed(),
        )
        rust = _rust_sizes(json.loads(probe_out.read_text(encoding="utf-8")))

    fragment, shape_record, il_terms = None, None, []
    if side.arm == "il":
        if side.artifacts is None:
            raise SystemExit("the il arm needs --artifacts")
        made = json.loads((side.artifacts / "artifacts.json").read_text(encoding="utf-8"))
        raw, shape_record = _il_fragment(side.il_shape)
        il_terms = _regulariser_names(raw)
        fragment = _fill(raw, made)
    config = _config(side.engine, side.arm, fragment=fragment, rust=rust)
    config_path = out / "config.json"
    config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    result = out / "result.json"
    seeds = {"fresh": _hash_seed(), "resume": _hash_seed()}
    if seeds["fresh"] == seeds["resume"]:
        raise SystemExit("the fresh and the resume process drew the same hash seed")
    job = {
        **base_job,
        "config": str(config_path),
        "result": str(result),
        "arm": side.arm,
        "il_shape": shape_record["name"] if shape_record else None,
        "il_terms": il_terms,
        "keep_finders": side.keep_finders,
    }
    _child(
        RUNNER,
        {**job, "mode": "fresh", "iterations": ITERATIONS},
        name="fresh",
        out=out,
        trees=path_trees,
        run=run,
        hashseed=seeds["fresh"],
    )
    if side.between:
        _between(side.between, out)
    _child(
        RUNNER,
        {**job, "mode": "resume", "iterations": AFTER},
        name="resume",
        out=out,
        trees=path_trees,
        run=run,
        hashseed=seeds["resume"],
    )
    record = json.loads(result.read_text(encoding="utf-8"))
    _check_child_seeds(record, seeds)
    _same_process_environment(record)
    record.update(
        format=RESULT_FORMAT,
        commit=side.learn_sha,
        gym_commit=side.gym_sha,
        imitate_commit=side.imitate_sha,
        arm=side.arm,
        engine=side.engine,
        il_shape=shape_record,
        rust_rules=rust,
        hashseeds=seeds,
        planted=side.planted,
        between=side.between,
        keep_finders=side.keep_finders,
        forbid_module=run.forbid_module,
        **side.extra,
    )
    result.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return result


def _same_process_environment(record: dict[str, Any]) -> None:
    """The fresh and the resumed process of one side read the same things."""
    fresh, resumed = record["environment"]["fresh"], record["environment"]["resumed"]
    for key in set(fresh) | set(resumed):
        if key == "data_files":
            a, b = fresh.get(key, {}), resumed.get(key, {})
            clash = sorted(k for k in set(a) & set(b) if a[k] != b[k])
            if clash:
                raise SystemExit(
                    f"a data file changed between the fresh and the resumed process: {clash}"
                )
        elif fresh.get(key) != resumed.get(key):
            raise SystemExit(
                f"environment {key!r} changed between the fresh and the resumed process: "
                f"{fresh.get(key)!r} -> {resumed.get(key)!r}"
            )
    ma, mb = record["modules"]["fresh"], record["modules"]["resumed"]
    clash = sorted(k for k in set(ma) & set(mb) if ma[k] != mb[k])
    if clash:
        raise SystemExit(f"a module came from another file on resume: {clash}")


def make_artifacts(learn_sha: str, gym_sha: str, out: Path, run: RunOptions) -> Path:
    """The il arm's init and reference, written once by ``learn_sha``'s own code."""
    _refuse_inside_a_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    trees = out / "tree"
    _extract(_learn_root(), learn_sha, trees / "RoyaleLearn")
    _extract(_find_repo("RoyaleGym"), gym_sha, trees / "RoyaleGym")
    config = _config("mock", "plain")
    config["run_name"] = "ab-artifacts"
    config_path = out / "config.json"
    config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    _child(
        ARTIFACT_MAKER,
        {
            "learn_tree": str(trees / "RoyaleLearn"),
            "gym_tree": str(trees / "RoyaleGym"),
            "config": str(config_path),
            "out_dir": str(out / "artifacts"),
        },
        name="artifacts",
        out=out,
        trees=[trees / "RoyaleLearn", trees / "RoyaleGym"],
        run=RunOptions(run.min_free, run.wait_minutes, None),
        hashseed=_hash_seed(),
    )
    return out / "artifacts"


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------

#: Columns holding one value. Only these may be listed in ``may_differ``.
SCALARS = (
    "commit",
    "gym_commit",
    "imitate_commit",
    "il_shape",
    "config_json_sha256",
    "config_hash",
    "resumed_config_hash",
    "resume_config_via",
    "run_id",
    "resumed_run_id",
    "weights",
    "moments",
    "state_digest",
    "resumed_weights",
    "resumed_moments",
    "resumed_state_digest",
    "row_count",
)
#: Columns of named entries, allowed name by name.
PARTS = (
    "config_keys",
    "identity_fields",
    "metric_keys",
    "metric_values",
    "alarms",
    "resumed_alarms",
    "alarm_params",
    "alarm_firings",
    "checkpoint",
    "other_modules",
)
#: Columns no expect file can allow.
NEVER = (
    "environment",
    "engine_binary",
    "config_order",
    "alarm_order",
    "resumed_alarm_order",
    "alarm_duplicates",
    "rename_problems",
)
SIDES = ("only_a", "only_b", "changed")


def _text(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _flat(value: Any, prefix: str = "", out: dict[str, str] | None = None) -> dict[str, str]:
    """Leaves by dotted path, as JSON text. {} and [] are leaves; lists go by index."""
    out = {} if out is None else out
    if isinstance(value, dict) and value:
        for key, item in value.items():
            _flat(item, f"{prefix}{key}.", out)
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            _flat(item, f"{prefix}{index}.", out)
    else:
        out[prefix[:-1]] = _text(value)
    return out


def _presence_exempt(key: str) -> bool:
    """Keys that exist or not depending on the machine."""
    return key == "throughput/gpu_util_frac" or (
        key.startswith("health/vram_") and key != "health/vram_peak_mb"
    )


def _value_exempt(key: str) -> bool:
    """Keys whose value reads the clock or the machine."""
    if key.startswith("time/") or key.startswith("health/vram_"):
        return True
    if key.startswith("throughput/"):
        return key != "throughput/discarded_rows_frac"
    return key in ("run/wall_seconds", "health/rss_peak_mb")


def _parts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    found = {
        "only_a": {k: a[k] for k in a if k not in b},
        "only_b": {k: b[k] for k in b if k not in a},
        "changed": {k: [a[k], b[k]] for k in a if k in b and a[k] != b[k]},
    }
    found = {part: names for part, names in found.items() if names}
    return found or None


class _Renamer:
    def __init__(self, rename: dict[str, Any] | None) -> None:
        rename = rename or {}
        self.metric = dict(rename.get("metric_keys", {}))
        self.alarm = dict(rename.get("alarms", {}))
        self.prefix = dict(rename.get("config_prefix", {}))
        self.used: set[tuple[str, str]] = set()

    def metric_key(self, key: str) -> str:
        if key in self.metric:
            self.used.add(("metric_keys", key))
            return self.metric[key]
        return key

    def alarm_name(self, name: str) -> str:
        if name in self.alarm:
            self.used.add(("alarms", name))
            return self.alarm[name]
        return name

    def config_key(self, key: str) -> str:
        for old, new in self.prefix.items():
            if key == old or key.startswith(old + "."):
                self.used.add(("config_prefix", old))
                return new + key[len(old) :]
        return key

    def unused(self) -> list[str]:
        every = [("metric_keys", k) for k in self.metric]
        every += [("alarms", k) for k in self.alarm]
        every += [("config_prefix", k) for k in self.prefix]
        return [
            f"rename {kind} {old!r} matched nothing on side A"
            for kind, old in every
            if (kind, old) not in self.used
        ]


def _renamed_dict(items: dict[str, Any], rename, problems: list[str], what: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items.items():
        new = rename(key)
        if new in out:
            problems.append(f"rename maps two {what} entries onto {new!r}")
        out[new] = value
    return out


def _alarm_columns(
    out: dict[str, Any],
    column: str,
    order_column: str,
    names_a: list[str],
    names_b: list[str],
    renamed: set[str],
) -> None:
    """``renamed`` (names side A reached through the rename map) are left out of the order: a
    stated rename may also move an alarm, and its firings are still compared by name."""
    duplicates = sorted(
        {f"a:{n}" for n in names_a if names_a.count(n) > 1}
        | {f"b:{n}" for n in names_b if names_b.count(n) > 1}
    )
    if duplicates:
        out.setdefault("alarm_duplicates", {})[column] = duplicates
    parts = _parts({n: True for n in names_a}, {n: True for n in names_b})
    if parts:
        out[column] = parts
    common_a = [n for n in dict.fromkeys(names_a) if n in set(names_b) and n not in renamed]
    common_b = [n for n in dict.fromkeys(names_b) if n in set(names_a) and n not in renamed]
    if common_a != common_b:
        out[order_column] = {"a": common_a, "b": common_b}


def _firings_by_name(record: dict[str, Any], renamer: _Renamer | None) -> dict[str, str]:
    """Every alarms.jsonl row, grouped by alarm name, in order within the name.

    AlarmSet.evaluate emits in table order, so with alarm_order equal the grouped rows are the
    file's rows. Grouping lets a one-sided alarm's firings be allowed (and required) by name
    rather than dropped, and a renamed alarm's firings be compared under its new name.
    """
    grouped: dict[str, list[str]] = {}
    for process in ("fresh", "resumed"):
        for row in record["alarm_firings"][process]:
            row = dict(row)
            if renamer is not None:
                row["name"] = renamer.alarm_name(row["name"])
                row["values"] = {
                    renamer.metric_key(k): v for k, v in (row.get("values") or {}).items()
                }
            grouped.setdefault(row["name"], []).append(f"{process}:{_text(row)}")
    return {name: _text(rows) for name, rows in grouped.items()}


def diff(
    a: dict[str, Any], b: dict[str, Any], rename: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Every difference between two results, column by column. ``rename`` is applied to side A."""
    for side, record in (("A", a), ("B", b)):
        if record.get("format") != RESULT_FORMAT:
            raise SystemExit(f"side {side} is not a format-{RESULT_FORMAT} result: re-run it")
    renamer = _Renamer(rename)
    problems: list[str] = []
    out: dict[str, Any] = {}
    for column in SCALARS:
        if column == "row_count":
            continue
        if _text(a.get(column)) != _text(b.get(column)):
            out[column] = [a.get(column), b.get(column)]

    # environment and engine: what makes two results comparable at all
    env_a = _flat({k: v for k, v in a["environment"].items()})
    env_b = _flat({k: v for k, v in b["environment"].items()})
    environment = _parts(env_a, env_b)
    if environment:
        out["environment"] = environment
    binary_a = a.get("identity", {}).get("engine_build", {}).get("binary_sha256")
    binary_b = b.get("identity", {}).get("engine_build", {}).get("binary_sha256")
    if binary_a != binary_b:
        out["engine_binary"] = [binary_a, binary_b]

    # config.json, leaf by leaf and in order. A key the rename map moved is compared by value
    # under its new name and left out of the order: a section move puts it elsewhere by design.
    flat_a = _flat(json.loads(a["config_json_text"]))
    moved = {renamer.config_key(k) for k in flat_a if renamer.config_key(k) != k}
    ca = _renamed_dict(flat_a, renamer.config_key, problems, "config")
    cb = _flat(json.loads(b["config_json_text"]))
    config = _parts(ca, cb)
    if config:
        out["config_keys"] = config
    order_a = [k for k in ca if k in cb and k not in moved]
    order_b = [k for k in cb if k in ca and k not in moved]
    if order_a != order_b:
        out["config_order"] = {
            "first": next(
                (i, x, y) for i, (x, y) in enumerate(zip(order_a, order_b, strict=True)) if x != y
            )
        }

    identity = _parts(_flat(a.get("identity", {})), _flat(b.get("identity", {})))
    if identity:
        out["identity_fields"] = identity

    # alarms: the table of each process, its parameters, and what fired
    renamed_alarms = {renamer.alarm.get(n) for n in a["alarms"] + a["resumed_alarms"]} - {None}
    _alarm_columns(
        out,
        "alarms",
        "alarm_order",
        [renamer.alarm_name(n) for n in a["alarms"]],
        b["alarms"],
        renamed_alarms,
    )
    _alarm_columns(
        out,
        "resumed_alarms",
        "resumed_alarm_order",
        [renamer.alarm_name(n) for n in a["resumed_alarms"]],
        b["resumed_alarms"],
        renamed_alarms,
    )
    params: dict[str, dict[str, str]] = {"a": {}, "b": {}}
    for label, record, ren in (("a", a, renamer), ("b", b, None)):
        for process, key in (("fresh", "alarm_params"), ("resumed", "resumed_alarm_params")):
            for name, (severity, patience, keys) in record[key].items():
                if ren is not None:
                    name = ren.alarm_name(name)
                    keys = [ren.metric_key(k) for k in keys]
                params[label][f"{process}:{name}"] = _text([severity, patience, keys])
    alarm_params = _parts(params["a"], params["b"])
    if alarm_params:
        out["alarm_params"] = alarm_params
    alarm_firings = _parts(_firings_by_name(a, renamer), _firings_by_name(b, None))
    if alarm_firings:
        out["alarm_firings"] = alarm_firings

    # metrics, row by row
    rows_a = [_renamed_dict(r, renamer.metric_key, problems, "metric") for r in a["rows"]]
    rows_b = b["rows"]
    if len(rows_a) != len(rows_b):
        out["row_count"] = [len(rows_a), len(rows_b)]
    presence: dict[str, dict[str, int]] = {"only_a": {}, "only_b": {}}
    values: dict[str, list[Any]] = {}
    changed_rows: dict[str, list[list[Any]]] = {}
    for index, (ra, rb) in enumerate(zip(rows_a, rows_b, strict=False), start=1):
        ka = {k for k in ra if not _presence_exempt(k)}
        kb = {k for k in rb if not _presence_exempt(k)}
        for key in ka - kb:
            presence["only_a"].setdefault(key, index)
        for key in kb - ka:
            presence["only_b"].setdefault(key, index)
        for key in sorted(set(ra) & set(rb)):
            if _value_exempt(key):
                continue
            ta, tb = _text(ra[key]), _text(rb[key])
            if ta != tb:
                changed_rows.setdefault(key, []).append([index, ta, tb])
                if key in values:
                    values[key][3] += 1
                else:
                    values[key] = [index, ta, tb, 1]
    # The first row alone would let a value pin pass while a later row moved.
    for key, changed in changed_rows.items():
        values[key].append(hashlib.sha256(_text(changed).encode()).hexdigest()[:16])
    presence = {part: names for part, names in presence.items() if names}
    if presence:
        out["metric_keys"] = presence
    if values:
        out["metric_values"] = {"changed": values}

    # checkpoints
    cka = {f"fresh:{k}": v for k, v in a["checkpoint_fresh"].items()}
    cka.update({f"resumed:{k}": v for k, v in a["checkpoint_resumed"].items()})
    ckb = {f"fresh:{k}": v for k, v in b["checkpoint_fresh"].items()}
    ckb.update({f"resumed:{k}": v for k, v in b["checkpoint_resumed"].items()})
    checkpoint = _parts(cka, ckb)
    if checkpoint:
        out["checkpoint"] = checkpoint

    ma = {**a["modules"]["fresh"], **a["modules"]["resumed"]}
    mb = {**b["modules"]["fresh"], **b["modules"]["resumed"]}
    modules = _parts({k: _text(v) for k, v in ma.items()}, {k: _text(v) for k, v in mb.items()})
    if modules:
        out["other_modules"] = modules

    problems += renamer.unused()
    if problems:
        out["rename_problems"] = problems
    return out


def _is_pattern(name: str) -> bool:
    return any(ch in name for ch in "*?[")


def _match(name: str, entry: str) -> bool:
    return fnmatch.fnmatchcase(name, entry) if _is_pattern(entry) else name == entry


def _entry(value: Any) -> tuple[str | None, bool]:
    """(why, optional) of an allow or may_differ entry."""
    if isinstance(value, str):
        return value, False
    if isinstance(value, dict):
        return value.get("why"), bool(value.get("optional", False))
    return None, False


#: The keys an entry written as a dict may hold. A pin under any other spelling would be ignored
#: silently, so an unknown key is refused. ``values`` pins a may_differ column to [A, B];
#: ``value`` pins an allow entry to the detail ``diff`` reports for it (a checkpoint leaf's JSON
#: text, [A, B] for a changed entry, a metric's [row, A, B, rows, rows_sha]).
ENTRY_KEYS = {
    "may_differ": {"why", "optional", "values"},
    "allow": {"why", "optional", "value"},
    "require": {"why"},
}
_UNPINNED = object()


def _pin(value: Any, key: str) -> Any:
    return value[key] if isinstance(value, dict) and key in value else _UNPINNED


def _entry_shape(block: str, where: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    problems = [
        f"expect: {where} has unknown key {k!r} (a dict entry holds {sorted(ENTRY_KEYS[block])})"
        for k in value
        if k not in ENTRY_KEYS[block]
    ]
    pinned = value.get("values", _UNPINNED)
    well_formed = isinstance(pinned, list) and len(pinned) == 2
    if block == "may_differ" and pinned is not _UNPINNED and not well_formed:
        problems.append(f"expect: {where} 'values' must be [side A, side B]")
    return problems


def lint_expect(expect: dict[str, Any]) -> list[str]:
    """The expect file's own mistakes, before anything is compared."""
    problems = []
    if expect.get("format") != EXPECT_FORMAT:
        problems.append(f"expect: 'format' must be {EXPECT_FORMAT}")
    known = {"format", "why", "may_differ", "rename", "allow", "require", "count", "note"}
    problems += [f"expect: unknown key {k!r}" for k in expect if k not in known]
    may = expect.get("may_differ", {})
    required_columns = expect.get("require", {}).get("columns", {})
    for column, value in may.items():
        why, optional = _entry(value)
        problems += _entry_shape("may_differ", f"may_differ {column!r}", value)
        if column not in SCALARS:
            problems.append(
                f"expect: may_differ {column!r} is not a single-value column "
                f"({'never waivable' if column in NEVER else 'allow its entries by name'})"
            )
        if not why:
            problems.append(f"expect: may_differ {column!r} has no why")
        if not optional and column not in required_columns:
            problems.append(f"expect: may_differ {column!r} is allowed without a require")
    for column, why in required_columns.items():
        if column not in may:
            problems.append(f"expect: require.columns {column!r} is required but not allowed")
        if not why:
            problems.append(f"expect: require.columns {column!r} has no why")
    allow = expect.get("allow", {})
    require = {k: v for k, v in expect.get("require", {}).items() if k != "columns"}
    for label, block in (("allow", allow), ("require", require)):
        for column, parts in block.items():
            if column not in PARTS:
                problems.append(f"expect: {label} {column!r} is not a named-entry column")
                continue
            for part, names in parts.items():
                if part not in SIDES:
                    problems.append(f"expect: {label} {column}.{part} is not only_a/only_b/changed")
                    continue
                for name, value in names.items():
                    why, optional = _entry(value)
                    problems += _entry_shape(label, f"{label} {column}.{part} {name!r}", value)
                    if _is_pattern(name) and _pin(value, "value") is not _UNPINNED:
                        problems.append(
                            f"expect: {label} {column}.{part} {name!r} pins a value on a pattern"
                        )
                    if not why:
                        problems.append(f"expect: {label} {column}.{part} {name!r} has no why")
                    unrequired = name not in require.get(column, {}).get(part, {})
                    if label == "allow" and not optional and unrequired:
                        problems.append(
                            f"expect: allow {column}.{part} {name!r} is allowed without a require"
                        )
                    if label == "require" and name not in allow.get(column, {}).get(part, {}):
                        problems.append(
                            f"expect: require {column}.{part} {name!r} is required but not allowed"
                        )
    for key, value in expect.get("count", {}).items():
        column, _, part = key.partition(".")
        if column not in PARTS or part not in SIDES:
            problems.append(f"expect: count {key!r} is not <column>.<only_a|only_b|changed>")
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("n"), int)
            or not value.get("why")
        ):
            problems.append(f"expect: count {key!r} must be {{'n': int, 'why': str}}")
    rename = expect.get("rename", {})
    for key in rename:
        if key not in ("metric_keys", "alarms", "config_prefix", "why"):
            problems.append(f"expect: rename {key!r} is not metric_keys/alarms/config_prefix")
    if any(k != "why" for k in rename) and not rename.get("why"):
        problems.append("expect: rename has no why")
    return problems


def unexpected(found: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    """What ``found`` holds that ``expect`` does not allow, and what ``expect`` requires that
    ``found`` does not hold. Empty means the comparison passes.

    Expect format 2 (every entry carries a why string; a name is exact unless it holds * ? or [,
    which is for families that are genuinely open-ended)::

        {"format": 2, "why": "...",
         "may_differ": {"commit": "why"},                       # single-value columns only
         "rename": {"metric_keys": {"old": "new"}, "alarms": {"old": "new"},
                    "config_prefix": {"old.prefix": "new.prefix"}, "why": "..."},
         "allow":   {"config_keys": {"only_b": {"imitation": "why"}}},
         "require": {"config_keys": {"only_b": {"imitation": "why"}},
                     "columns": {"commit": "why"}},
         "count":   {"alarms.only_a": {"n": 4, "why": "..."}}}

    ``allow`` permits, ``require`` obliges: an allowed entry without a matching require is a
    mistake unless the entry is written ``{"why": ..., "optional": true}``, and a required one
    that did not happen is ``expected difference missing``. An allowance can also pin what the
    difference is: ``"may_differ": {"run_id": {"why": ..., "values": [A, B]}}`` and
    ``"allow": {"checkpoint": {"only_b": {"<leaf>": {"why": ..., "value": "5.625"}}}}`` (the
    detail ``diff`` prints for that entry; exact names only). Without a pin an allowance admits
    any value, which is right only when no value is known. The rename map is applied to side A
    by ``diff`` before anything is compared, so a renamed entry is compared by value. The NEVER
    columns cannot be allowed at all.
    """
    problems = lint_expect(expect)
    may = expect.get("may_differ", {})
    allow = expect.get("allow", {})
    require = expect.get("require", {})
    for column, value in found.items():
        if column in NEVER:
            problems.append(f"{column} (never allowed): {json.dumps(value)[:400]}")
            continue
        if column in SCALARS:
            if column not in may:
                problems.append(f"{column}: {json.dumps(value)[:300]}")
            else:
                pinned = _pin(may[column], "values")
                if pinned is not _UNPINNED and _text(pinned) != _text(value):
                    problems.append(
                        f"{column}: pinned to {json.dumps(pinned)[:300]}, "
                        f"found {json.dumps(value)[:300]}"
                    )
            continue
        if column not in PARTS:
            problems.append(f"{column} (unknown column): {json.dumps(value)[:300]}")
            continue
        for part, names in value.items():
            entries = allow.get(column, {}).get(part, {})
            for name in names:
                detail = names[name] if isinstance(names, dict) else ""
                matched = [entry for entry in entries if _match(name, entry)]
                if not matched:
                    problems.append(f"{column}.{part}: {name} {json.dumps(detail)[:200]}")
                for entry in matched:
                    pinned = _pin(entries[entry], "value")
                    if pinned is not _UNPINNED and _text(pinned) != _text(detail):
                        problems.append(
                            f"{column}.{part}: {name} pinned to {json.dumps(pinned)[:200]}, "
                            f"found {json.dumps(detail)[:200]}"
                        )
    for column in require.get("columns", {}):
        if column not in found:
            problems.append(f"expected difference missing: column {column}")
    for column, parts in require.items():
        if column == "columns":
            continue
        for part, names in parts.items():
            present = list(found.get(column, {}).get(part, {}))
            for entry in names:
                if not any(_match(name, entry) for name in present):
                    problems.append(f"expected difference missing: {column}.{part} {entry}")
    for key, value in expect.get("count", {}).items():
        column, _, part = key.partition(".")
        n = len(found.get(column, {}).get(part, {}))
        if isinstance(value, dict) and n != value.get("n"):
            problems.append(f"count {key}: {n} found, {value.get('n')} expected")
    return problems


def _load_result(path: Path) -> dict[str, Any]:
    path = path / "result.json" if path.is_dir() else path
    return json.loads(path.read_text(encoding="utf-8"))


def _header(a: dict[str, Any], b: dict[str, Any]) -> str:
    lines = []
    for label, record in (("A", a), ("B", b)):
        shape = (record.get("il_shape") or {}).get("name")
        lines.append(
            f"{label}: commit {record.get('commit')}  gym_commit {record.get('gym_commit')}  "
            f"imitate_commit {record.get('imitate_commit')}  arm {record.get('arm')}  "
            f"engine {record.get('engine')}  il_shape {shape}  planted {record.get('planted')}  "
            f"hashseeds {record.get('hashseeds')}"
        )
    return "\n".join(lines)


def compare(a_path: Path, b_path: Path, expect_path: Path | None) -> int:
    a, b = _load_result(a_path), _load_result(b_path)
    print(_header(a, b))
    for key in ("arm", "engine"):
        if a.get(key) != b.get(key):
            print(f"REFUSED: the sides ran different {key}s ({a.get(key)} and {b.get(key)})")
            return 2
    expect = json.loads(expect_path.read_text(encoding="utf-8")) if expect_path else None
    found = diff(a, b, rename=(expect or {}).get("rename"))
    print(json.dumps(found, indent=1, sort_keys=True) if found else "identical on every column")
    if expect is None:
        return 1 if found else 0
    problems = unexpected(found, expect)
    print(
        "\n".join(["UNEXPECTED:", *problems])
        if problems
        else "every difference was expected, and every required one happened"
    )
    return 1 if problems else 0


# --------------------------------------------------------------------------
# The self-test: every column seen moving, every near-miss expect seen refused
# --------------------------------------------------------------------------


@dataclass
class Plant:
    key: str
    name: str
    arm: str
    file: str
    anchor: str
    replacement: str
    #: Columns this plant must move. ``metric_values:<key>`` needs that key changed,
    #: ``metric_values:<key>@<row>`` needs its first difference at that row, and
    #: ``<named-entry column>:<entry>`` needs that exact entry (a pattern if it holds * ? [).
    must_move: list[str]
    #: Columns this plant must leave equal (same syntax).
    must_stay: list[str] = field(default_factory=list)


PLANTS = [
    Plant(
        "weight",
        "one ulp on one actor weight before the second update (sched.iteration == 1)",
        "plain",
        "royalelearn/learn/ppo.py",
        '        """One iteration\'s whole update, and everything it reports about itself."""\n',
        '        """One iteration\'s whole update, and everything it reports about itself."""\n'
        "        if sched.iteration == 1:  # ab_digest plant\n"
        "            with torch.no_grad():\n"
        "                first = self.actor_params[0].view(-1)\n"
        "                first[0] = torch.nextafter(first[0], first[0] + 1)\n",
        [
            "weights",
            "state_digest",
            "metric_values:run/state_digest@2",
            "checkpoint:fresh:manifest:files.actor_critic/actor.safetensors",
        ],
    ),
    Plant(
        "resumed_weight",
        "one ulp on one actor weight before the first update of the resumed process "
        "(sched.iteration == 3)",
        "plain",
        "royalelearn/learn/ppo.py",
        '        """One iteration\'s whole update, and everything it reports about itself."""\n',
        '        """One iteration\'s whole update, and everything it reports about itself."""\n'
        "        if sched.iteration == 3:  # ab_digest plant\n"
        "            with torch.no_grad():\n"
        "                first = self.actor_params[0].view(-1)\n"
        "                first[0] = torch.nextafter(first[0], first[0] + 1)\n",
        [
            "resumed_weights",
            "resumed_moments",
            "resumed_state_digest",
            "metric_values:run/state_digest@4",
            "checkpoint:resumed:manifest:files.actor_critic/actor.safetensors",
        ],
        ["weights", "moments", "state_digest"],
    ),
    Plant(
        "rng_draw",
        "one extra torch draw just before rng.json is written",
        "plain",
        "royalelearn/checkpoint.py",
        '        (folder / "rng.json").write_bytes(msgspec.json.encode(self.capture()))\n',
        "        import torch as _ab_torch  # ab_digest plant\n"
        "        _ab_torch.rand(1)\n"
        '        (folder / "rng.json").write_bytes(msgspec.json.encode(self.capture()))\n',
        ["checkpoint:fresh:rng/rng.json:torch_cpu"],
    ),
    Plant(
        "moments",
        "one ulp on one critic exp_avg entry after each optimizer step of the third update",
        "plain",
        "royalelearn/learn/ppo.py",
        "            self.model_updates += 1\n",
        "            self.model_updates += 1\n"
        "            if sched.iteration == 2:  # ab_digest plant\n"
        "                with torch.no_grad():\n"
        "                    ab_state = self.critic_optimizer.state[self.critic_params[0]]\n"
        "                    moment = ab_state['exp_avg'].view(-1)\n"
        "                    moment[0] = torch.nextafter(moment[0], moment[0] + 1)\n",
        ["moments", "checkpoint:fresh:manifest:files.optimizers/critic_adam.pt"],
    ),
    Plant(
        "metric_key",
        "an extra metric key",
        "plain",
        "royalelearn/metrics/records.py",
        '        "ppo/samples_unused_frac": result.samples_unused_frac,\n',
        '        "ppo/samples_unused_frac": result.samples_unused_frac,\n'
        '        "ppo/ab_digest_plant": 0.0,\n',
        ["metric_keys"],
    ),
    Plant(
        "alarm_left_out",
        "an alarm left out of the table",
        "plain",
        "royalelearn/metrics/alarms.py",
        "        disabled = set(config.disabled)\n",
        '        disabled = set(config.disabled) | {"kl_dead"}\n',
        ["alarms", "resumed_alarms"],
    ),
    Plant(
        "alarm_threshold",
        "a core alarm threshold changed in code so reward_clipped fires on every row",
        "plain",
        "royalelearn/metrics/alarms.py",
        "            lambda frac: frac > 0.0,\n",
        "            lambda frac: frac >= 0.0,\n",
        ["alarm_firings:reward_clipped"],
    ),
    Plant(
        "alarm_duplicate",
        "one alarm registered twice",
        "plain",
        "royalelearn/metrics/alarms.py",
        "            self.alarms.append(alarm)\n",
        "            self.alarms.append(alarm)\n"
        '            if alarm.name == "kl_high":  # ab_digest plant\n'
        "                self.alarms.append(alarm)\n",
        ["alarm_duplicates"],
    ),
    Plant(
        "config_field",
        "a default-None RunConfig field",
        "plain",
        "royalelearn/config.py",
        "    doctor: DoctorConfig = DoctorConfig()\n",
        "    doctor: DoctorConfig = DoctorConfig()\n    ab_digest_plant: int | None = None\n",
        ["config_json_sha256", "config_hash", "resumed_config_hash", "config_keys"],
    ),
    Plant(
        "identity",
        "an identity that moves alone",
        "plain",
        "royalelearn/identity.py",
        "IDENTITY_FORMAT_VERSION = 1\n",
        "IDENTITY_FORMAT_VERSION = 2\n",
        ["run_id", "resumed_run_id", "identity_fields"],
    ),
    Plant(
        "schedule_field",
        "an extra defaulted ScheduleState field",
        "plain",
        "royalelearn/api/schedule.py",
        "    actor_lr_scale: float = 1.0\n",
        "    actor_lr_scale: float = 1.0\n    ab_digest_plant: float = 0.0\n",
        [
            "checkpoint:fresh:schedules/state.json:last.ab_digest_plant",
            "checkpoint:resumed:schedules/state.json:last.ab_digest_plant",
        ],
    ),
    Plant(
        "il_init_dropped",
        "the IL actor init not applied",
        "il",
        "royalelearn/coordinator.py",
        "        self._initialise_from_imitation()\n",
        "        pass  # ab_digest plant: the actor init is not applied\n",
        ["weights", "metric_values"],
    ),
    Plant(
        "il_lambda_ulp",
        "one float32 ulp on the regulariser's lambda each iteration",
        "il",
        "royalelearn/imitation/regularisers.py",
        "        self.lam = self.coef.current(env_steps)\n",
        "        self.lam = self.coef.current(env_steps)\n"
        "        import numpy as _np  # ab_digest plant\n"
        "        self.lam = float(_np.nextafter(_np.float32(self.lam), _np.float32(_np.inf)))\n",
        ["metric_values:imitation/bc/lambda"],
    ),
]


def _moved(found: dict[str, Any], spec: str) -> bool:
    column, _, detail = spec.partition(":")
    if column not in found:
        return False
    if not detail:
        return True
    if column == "metric_values":
        key, _, row = detail.partition("@")
        entry = found[column].get("changed", {}).get(key)
        return entry is not None and (not row or entry[0] == int(row))
    if column in PARTS:
        # a full entry name (a pattern when it holds * ? or [), never a substring
        return any(_match(name, detail) for part in found[column].values() for name in part)
    return False


def expect_from(found: dict[str, Any], why: str = "self-test") -> dict[str, Any]:
    """An expect that allows and requires exactly ``found`` (never the NEVER columns)."""
    expect: dict[str, Any] = {
        "format": EXPECT_FORMAT,
        "may_differ": {},
        "allow": {},
        "require": {"columns": {}},
    }
    for column, value in found.items():
        if column in SCALARS:
            expect["may_differ"][column] = why
            expect["require"]["columns"][column] = why
        elif column in PARTS:
            for part, names in value.items():
                for name in names:
                    expect["allow"].setdefault(column, {}).setdefault(part, {})[name] = why
                    expect["require"].setdefault(column, {}).setdefault(part, {})[name] = why
    return expect


def _near_misses(
    null: dict[str, Any],
    key_plant: dict[str, Any],
    ref: dict[str, Any],
    ref2: dict[str, Any],
    dup: dict[str, Any],
) -> list[str]:
    """Near-miss expect files must each be refused, for the reason they were built to show."""
    failures = []
    target = "ppo/ab_digest_plant"

    def case(label: str, found: dict[str, Any], expect: dict[str, Any], needle: str | None) -> None:
        problems = unexpected(found, expect)
        if needle is None:
            ok = not problems
            verdict = "passes" if ok else f"REFUSED WRONGLY: {problems}"
        else:
            hits = [p for p in problems if needle in p]
            ok = bool(hits)
            verdict = (
                f"refused ({hits[0]})" if ok else f"NOT REFUSED as expected; problems: {problems}"
            )
        print(f"  near-miss {label}: {verdict}")
        if not ok:
            failures.append(f"near-miss {label!r} was not handled")

    exact = expect_from(key_plant)
    case("control: the exact expect of the metric-key plant", key_plant, exact, None)

    wrong_name = copy.deepcopy(exact)
    for block in ("allow", "require"):
        names = wrong_name[block]["metric_keys"]["only_b"]
        names[target + "_x"] = names.pop(target)
    case("allowed under a wrong name", key_plant, wrong_name, f"metric_keys.only_b: {target}")

    wrong_side = copy.deepcopy(exact)
    for block in ("allow", "require"):
        why = wrong_side[block]["metric_keys"].pop("only_b")[target]
        wrong_side[block]["metric_keys"]["only_a"] = {target: why}
    case("allowed on the wrong side", key_plant, wrong_side, f"metric_keys.only_b: {target}")

    no_require = copy.deepcopy(exact)
    del no_require["require"]["metric_keys"]
    case("allowed without a require", key_plant, no_require, "allowed without a require")

    required = {
        "format": EXPECT_FORMAT,
        "allow": {"metric_keys": {"only_b": {target: "self-test"}}},
        "require": {"metric_keys": {"only_b": {target: "self-test"}}},
    }
    case("required but absent (null pair)", null, required, "expected difference missing")

    counted = copy.deepcopy(exact)
    counted["count"] = {"metric_keys.only_b": {"n": 2, "why": "self-test"}}
    case("an exact count that is wrong", key_plant, counted, "count metric_keys.only_b")

    def mutated(change: Any) -> dict[str, Any]:
        # a copy of the ref itself, so a control depends on nothing but the mutation
        side = copy.deepcopy(ref)
        change(side)
        return side

    def config_edit(edit: Any) -> Any:
        def change(side: dict[str, Any]) -> None:
            config = json.loads(side["config_json_text"])
            edit(config)
            side["config_json_text"] = json.dumps(config, indent=1)

        return change

    plain = {"format": EXPECT_FORMAT, "why": "self-test"}

    # require.columns: the only thing in an S1 expect that stops a same-commit compare passing
    same_commit = {
        **plain,
        "may_differ": {"commit": "self-test"},
        "require": {"columns": {"commit": "self-test"}},
    }
    case(
        "a required commit difference on the null pair",
        null,
        same_commit,
        "expected difference missing: column commit",
    )

    # renames
    bogus = {"metric_keys": {"ab_digest/no_such_key": "ab_digest/x"}, "why": "self-test"}
    case(
        "a rename that matches nothing on side A",
        diff(ref, ref2, rename=bogus),
        {**plain, "rename": bogus},
        "rename_problems",
    )
    key = "run/iteration"
    if any(key in keys for _, _, keys in ref["alarm_params"].values()):
        failures.append(f"near-miss metric rename: {key} feeds an alarm, pick another key")
    metric_rename = {"metric_keys": {key: key + "_v2"}, "why": "self-test"}

    def rename_rows(side: dict[str, Any]) -> None:
        side["rows"] = [
            {(key + "_v2" if k == key else k): v for k, v in r.items()} for r in side["rows"]
        ]

    renamed_side = mutated(rename_rows)
    case(
        "control: a renamed metric with equal values under its rename",
        diff(ref, renamed_side, rename=metric_rename),
        {**plain, "rename": metric_rename},
        None,
    )
    renamed_side["rows"][1][key + "_v2"] = 99
    case(
        "a renamed metric whose value changed",
        diff(ref, renamed_side, rename=metric_rename),
        {**plain, "rename": metric_rename},
        f"metric_values.changed: {key}_v2",
    )

    # value pins: an allowance can say what the difference is, not only where it is
    detail = key_plant["metric_keys"]["only_b"][target]
    pinned = copy.deepcopy(exact)
    pinned["allow"]["metric_keys"]["only_b"][target] = {"why": "self-test", "value": detail}
    case("control: an allowance pinned to the value it found", key_plant, pinned, None)
    other = copy.deepcopy(pinned)
    other["allow"]["metric_keys"]["only_b"][target]["value"] = detail + 1
    case(
        "an allowance pinned to another value",
        key_plant,
        other,
        f"metric_keys.only_b: {target} pinned to",
    )
    misspelt = copy.deepcopy(exact)
    misspelt["allow"]["metric_keys"]["only_b"][target] = {"why": "self-test", "valeu": detail + 1}
    case("a pin under a misspelt key", key_plant, misspelt, "unknown key 'valeu'")
    on_pattern = copy.deepcopy(exact)
    for block in ("allow", "require"):
        on_pattern[block]["metric_keys"]["only_b"] = {target[:-2] + "*": "self-test"}
    on_pattern["allow"]["metric_keys"]["only_b"][target[:-2] + "*"] = {
        "why": "self-test",
        "value": detail,
    }
    case("a value pinned on a pattern", key_plant, on_pattern, "pins a value on a pattern")

    def other_run_id(side: dict[str, Any]) -> None:
        side["run_id"] = "ab_digest_run_id"

    moved_id = diff(ref, mutated(other_run_id))
    id_pinned = expect_from(moved_id)
    id_pinned["may_differ"]["run_id"] = {
        "why": "self-test",
        "values": [ref["run_id"], "ab_digest_run_id"],
    }
    case("control: a may_differ column pinned to what it found", moved_id, id_pinned, None)
    id_other = copy.deepcopy(id_pinned)
    id_other["may_differ"]["run_id"]["values"] = [ref["run_id"], "ab_digest_other"]
    case("a may_differ column pinned to other values", moved_id, id_other, "run_id: pinned to")

    def two_rows(second: int) -> Any:
        def change(side: dict[str, Any]) -> None:
            side["rows"][1][key] = 1001
            side["rows"][2][key] = second

        return change

    rows_found = diff(ref, mutated(two_rows(1002)))
    rows_pinned = expect_from(rows_found)
    rows_pinned["allow"]["metric_values"]["changed"][key] = {
        "why": "self-test",
        "value": rows_found["metric_values"]["changed"][key],
    }
    case("control: a changed metric pinned to its entry", rows_found, rows_pinned, None)
    case(
        "the same pin when only a later row's value differs",
        diff(ref, mutated(two_rows(2002))),
        rows_pinned,
        f"metric_values.changed: {key} pinned to",
    )

    # types, rows
    def float_iteration(side: dict[str, Any]) -> None:
        side["rows"][2][key] = float(side["rows"][2][key])

    case(
        "a metric value that changed type only (int to float)",
        diff(ref, mutated(float_iteration)),
        plain,
        f"metric_values.changed: {key}",
    )

    def float_epochs(config: dict[str, Any]) -> None:
        config["ppo"]["n_epochs"] = float(config["ppo"]["n_epochs"])

    case(
        "a config leaf that changed type only (int to float)",
        diff(ref, mutated(config_edit(float_epochs))),
        plain,
        "config_keys.changed: ppo.n_epochs",
    )

    def drop_row(side: dict[str, Any]) -> None:
        side["rows"] = side["rows"][:-1]

    case("a metric row dropped from side B", diff(ref, mutated(drop_row)), plain, "row_count")

    # config order, and a section move under the rename map
    def swap_ppo(config: dict[str, Any]) -> None:
        config["ppo"] = dict(reversed(list(config["ppo"].items())))

    case(
        "two config keys swapped",
        diff(ref, mutated(config_edit(swap_ppo))),
        plain,
        "config_order",
    )
    section = {
        "config_prefix": {"ppo.batch_size": "ab_digest_section.batch_size"},
        "why": "self-test",
    }

    def move(delta: int) -> Any:
        def edit(config: dict[str, Any]) -> None:
            config["ab_digest_section"] = {"batch_size": config["ppo"].pop("batch_size") + delta}

        return config_edit(edit)

    case(
        "control: a key moved to a new section under a config_prefix rename",
        diff(ref, mutated(move(0)), rename=section),
        {**plain, "rename": section},
        None,
    )
    case(
        "the same section move with the moved value changed",
        diff(ref, mutated(move(1)), rename=section),
        {**plain, "rename": section},
        "config_keys.changed: ab_digest_section.batch_size",
    )

    # alarm table: order, a renamed alarm that also moved, parameters, a one-sided alarm's firings
    first, second = ref["alarms"][0], ref["alarms"][1]

    def swap_alarms(side: dict[str, Any]) -> None:
        side["alarms"][0], side["alarms"][1] = side["alarms"][1], side["alarms"][0]

    case("two alarms swapped in the table", diff(ref, mutated(swap_alarms)), plain, "alarm_order")
    alarm_rename = {"alarms": {first: first + "_v2"}, "why": "self-test"}

    def rename_and_move(side: dict[str, Any]) -> None:
        for column in ("alarms", "resumed_alarms"):
            side[column] = [n for n in side[column] if n != first] + [first + "_v2"]
        for column in ("alarm_params", "resumed_alarm_params"):
            side[column][first + "_v2"] = side[column].pop(first)

    case(
        "control: a renamed alarm that also moved to the end of the table",
        diff(ref, mutated(rename_and_move), rename=alarm_rename),
        {**plain, "rename": alarm_rename},
        None,
    )

    def patience(side: dict[str, Any]) -> None:
        side["alarm_params"][second][1] += 1

    case(
        "an alarm's patience changed",
        diff(ref, mutated(patience)),
        plain,
        f"alarm_params.changed: fresh:{second}",
    )
    new = "ab_digest_new_alarm"

    def one_sided(side: dict[str, Any]) -> None:
        for column in ("alarms", "resumed_alarms"):
            side[column].append(new)
        for column in ("alarm_params", "resumed_alarm_params"):
            side[column][new] = ["warn", 1, [key]]
        for process in ("fresh", "resumed"):
            side["alarm_firings"][process].append(
                {
                    "name": new,
                    "severity": "warn",
                    "iteration": 1,
                    "consecutive": 1,
                    "values": {key: 1},
                }
            )

    fired = diff(ref, mutated(one_sided))
    case(
        "control: a new alarm, its parameters and its firings, all allowed",
        fired,
        expect_from(fired),
        None,
    )
    unfired = expect_from(fired)
    for block in ("allow", "require"):
        unfired[block].pop("alarm_firings", None)
    case(
        "a new alarm allowed but its firings not",
        fired,
        unfired,
        f"alarm_firings.only_b: {new}",
    )

    # the never-waivable columns, from a real diff
    def data_file(side: dict[str, Any]) -> None:
        for process in ("fresh", "resumed"):
            files = side["environment"][process]["data_files"]
            files[sorted(files)[0]] = "0" * 64

    moved_env = diff(ref, mutated(data_file))
    case(
        "a data file's sha differs (everything else allowed)",
        moved_env,
        expect_from(moved_env),
        "environment (never allowed)",
    )
    case(
        "the environment waived through may_differ",
        moved_env,
        {**plain, "may_differ": {"environment": {"why": "self-test", "optional": True}}},
        "may_differ 'environment' is not a single-value column (never waivable)",
    )

    def binary(side: dict[str, Any]) -> None:
        side["identity"].setdefault("engine_build", {})["binary_sha256"] = "ab_digest"

    moved_binary = diff(ref, mutated(binary))
    case(
        "the engine binary differs (identity allowed)",
        moved_binary,
        expect_from(moved_binary),
        "engine_binary (never allowed)",
    )

    def module(side: dict[str, Any]) -> None:
        side["modules"]["fresh"]["royaleimitate.ab_digest"] = ["<venv>/ab_digest.py"]

    case(
        "a royale* module only side B loaded",
        diff(ref, mutated(module)),
        plain,
        "other_modules.only_b: royaleimitate.ab_digest",
    )

    def replica(side: dict[str, Any]) -> None:
        side["resume_config_via"] = "replicated cli._run_config"

    case(
        "side B resumed through the replica",
        diff(ref, mutated(replica)),
        plain,
        "resume_config_via",
    )

    rename = {"alarms": {"kl_high": "kl_high_v2"}, "why": "self-test"}
    renamed = diff(ref, dup, rename=rename)
    lenient = expect_from(renamed)
    lenient["rename"] = rename
    case("a duplicated alarm beside an allowed rename", renamed, lenient, "alarm_duplicates")
    return failures


def _refused(action: Any, needle: str) -> tuple[bool, str]:
    """(refused for the right reason, what it said)."""
    try:
        action()
    except SystemExit as exc:
        text = str(exc.code)
        return needle in text, text
    return False, "it did not refuse"


def _parent_refusals(ref: dict[str, Any]) -> list[str]:
    """The parent-side refusals, fed from the real ref result."""
    failures = []

    def case(label: str, change: Any, check: Any, needle: str | None) -> None:
        record = copy.deepcopy(ref)
        change(record)
        if needle is None:
            ok, said = _refused(lambda: check(record), "\0")
            ok = said == "it did not refuse"
        else:
            ok, said = _refused(lambda: check(record), needle)
        print(f"  parent refusal {label}: {'ok' if ok else 'NOT HANDLED'} ({said[:160]})")
        if not ok:
            failures.append(f"parent refusal {label!r} was not handled")

    def data(record: dict[str, Any]) -> None:
        files = record["environment"]["resumed"]["data_files"]
        files[sorted(files)[0]] = "0" * 64

    def head(record: dict[str, Any]) -> None:
        record["environment"]["resumed"]["royalesim_head"] = "ab_digest"

    def module(record: dict[str, Any]) -> None:
        record["modules"]["fresh"]["royaleimitate.ab_digest"] = ["a.py"]
        record["modules"]["resumed"]["royaleimitate.ab_digest"] = ["b.py"]

    same = _same_process_environment
    case("control: the ref's own environment", lambda r: None, same, None)
    case("a data file changed between the processes", data, same, "a data file changed")
    case(
        "RoyaleSim HEAD changed between the processes",
        head,
        same,
        "environment 'royalesim_head' changed",
    )
    case(
        "a module file changed between the processes",
        module,
        same,
        "a module came from another file",
    )
    planned = dict(ref["hashseeds"])
    case(
        "control: the children ran on the planned seeds",
        lambda r: None,
        lambda r: _check_child_seeds(r, planned),
        None,
    )
    case(
        "a child ran on another hash seed than planned",
        lambda r: None,
        lambda r: _check_child_seeds(r, {**planned, "resume": planned["resume"] + 1}),
        "ran with PYTHONHASHSEED",
    )
    return failures


@dataclass
class Refusal:
    """A side that must fail, for the reason named by ``needle``."""

    key: str
    needle: str
    file: str | None = None
    anchor: str | None = None
    replacement: str | None = None
    delete: str | None = None
    keep_finders: bool = False
    between: str | None = None
    forbid: str | None = None


_ZERO_ACTOR_GRADS = (
    "            for _ab_p in self.actor_params:  # ab_digest plant\n"
    "                if _ab_p.grad is not None:\n"
    "                    _ab_p.grad.zero_()\n"
    "            diagnostics.gradients(\n"
)

REFUSALS = [
    Refusal(
        "cli_unimportable",
        "_ab_digest_gone",
        file="royalelearn/cli.py",
        anchor="from __future__ import annotations\n",
        replacement="from __future__ import annotations\n"
        "from . import _ab_digest_gone  # noqa  ab_digest plant\n",
    ),
    Refusal(
        "module_missing",
        "No module named 'royalelearn.determinism'",
        delete="royalelearn/determinism.py",
    ),
    Refusal(
        "module_from_live_checkout",
        "royalelearn.determinism was loaded from",
        delete="royalelearn/determinism.py",
        keep_finders=True,
    ),
    Refusal(
        "actor_never_trained",
        "never exercised the actor",
        file="royalelearn/learn/ppo.py",
        anchor="            diagnostics.gradients(\n",
        replacement=_ZERO_ACTOR_GRADS,
    ),
    Refusal("config_edited_between", "does not load back", between="config_edit"),
    Refusal(
        "alarms_rewritten",
        "lost or rewrote",
        file="royalelearn/metrics/alarms.py",
        anchor="            lambda frac: frac > 0.0,\n",
        replacement="            lambda frac: frac >= 0.0,\n",
        between="alarms_rewrite",
    ),
    Refusal(
        "forbidden_module_imported",
        "--forbid-module: child fresh imported ['royalelearn.coordinator'",
        forbid="royalelearn.coordinator",
    ),
]


def _child_refusals(
    base_sha: str, common: dict[str, Any], out: Path, run: RunOptions, ref: dict[str, Any]
) -> list[str]:
    """Sides that must fail inside a child (or right after it), each for its named reason, and
    one --forbid-module side that must pass and equal the ref."""
    failures = []
    for case in REFUSALS:
        tree = None
        if case.file or case.delete:
            tree = out / f"refuse-{case.key}" / "src"
            _extract(_learn_root(), base_sha, tree)
            if case.file:
                path = tree / case.file
                text = path.read_text(encoding="utf-8")
                if case.anchor is None or text.count(case.anchor) != 1:
                    failures.append(f"refusal {case.key!r} NOT APPLIED")
                    print(failures[-1])
                    continue
                path.write_text(text.replace(case.anchor, case.replacement or ""), "utf-8")
            if case.delete:
                (tree / case.delete).unlink()
        side = Side(
            base_sha,
            learn_tree=tree,
            planted=f"refuse-{case.key}",
            keep_finders=case.keep_finders,
            between=case.between,
            **common,
        )
        options = RunOptions(run.min_free, run.wait_minutes, case.forbid)
        ok, said = _refused(
            lambda s=side, o=options, d=out / f"refuse-{case.key}" / "side": run_side(s, d, o),
            case.needle,
        )
        tail = said.strip().splitlines()[-1][:200] if said.strip() else said
        print(f"refusal {case.key}: {'REFUSED as required' if ok else 'NOT REFUSED'} ({tail})")
        if not ok:
            failures.append(
                f"refusal {case.key!r} was not refused for {case.needle!r}: {said[-600:]}"
            )
    absent = "royalelearn._ab_digest_absent"
    options = RunOptions(run.min_free, run.wait_minutes, absent)
    ok, said = _refused(
        lambda: run_side(
            Side(base_sha, planted="forbid-absent", **common),
            out / "forbid-absent" / "side",
            options,
        ),
        "\0",
    )
    if said != "it did not refuse":
        failures.append(
            f"--forbid-module {absent} refused a side that never imports it: {said[-600:]}"
        )
        print(failures[-1])
    else:
        other = diff(ref, _load_result(out / "forbid-absent" / "side"))
        print(f"forbid-absent: passed; against ref: {json.dumps(other) if other else 'identical'}")
        if other:
            failures.append("the --forbid-module side differs from the ref")
    return failures


def selftest(
    base_sha: str,
    gym_sha: str,
    out: Path,
    run: RunOptions,
    *,
    arm: str,
    engine: str,
    artifacts: Path | None,
    il_shape: str,
    only: list[str] | None,
) -> int:
    common = dict(gym_sha=gym_sha, arm=arm, engine=engine, artifacts=artifacts, il_shape=il_shape)
    ref_path = run_side(Side(base_sha, **common), out / "ref", run)
    ref2_path = run_side(Side(base_sha, **common), out / "ref2", run)
    ref, ref2 = _load_result(ref_path), _load_result(ref2_path)
    failures: list[str] = []
    seeds = [
        ref["hashseeds"]["fresh"],
        ref["hashseeds"]["resume"],
        ref2["hashseeds"]["fresh"],
        ref2["hashseeds"]["resume"],
    ]
    ran = [r["child_hashseed"][p] for r in (ref, ref2) for p in ("fresh", "resume")]
    print(f"hash seeds (ref fresh, ref resume, ref2 fresh, ref2 resume): {seeds}")
    print(
        f"  as the children saw them: {[s['env'] for s in ran]}, hash('ab_digest'): "
        f"{[s['probe'] for s in ran]}"
    )
    if len(set(seeds)) != 4:
        failures.append("the null pair did not run on four different hash seeds")
    if [s["env"] for s in ran] != [str(s) for s in seeds] or len({s["probe"] for s in ran}) != 4:
        failures.append("the children did not run on four different hash seeds")
    null = diff(ref, ref2)
    print(
        f"null test ({base_sha[:12]} against itself, {arm} arm, {engine}): "
        f"{json.dumps(null) if null else 'identical'}"
    )
    if null:
        failures.append("the null test differs")
    results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for plant in PLANTS:
        if plant.arm != arm or (only and plant.key not in only):
            continue
        tree = out / f"plant-{plant.key}" / "src"
        _extract(_learn_root(), base_sha, tree)
        path = tree / plant.file
        text = path.read_text(encoding="utf-8")
        if text.count(plant.anchor) != 1:
            failures.append(
                f"plant {plant.name!r} NOT APPLIED ({text.count(plant.anchor)} anchors)"
            )
            print(failures[-1])
            continue
        path.write_text(text.replace(plant.anchor, plant.replacement), encoding="utf-8")
        side = Side(base_sha, learn_tree=tree, planted=plant.key, **common)
        result = _load_result(run_side(side, out / f"plant-{plant.key}" / "side", run))
        found = diff(ref, result)
        results[plant.key] = (result, found)
        missing = [m for m in plant.must_move if not _moved(found, m)]
        # every plant leaves the environment equal: a plant that moved it measured the
        # machine (a data file edited during the self-test), not the plant
        moved = [m for m in [*plant.must_stay, "environment"] if _moved(found, m)]
        verdict = "CAUGHT" if not missing and not moved else f"MISSED {missing} MOVED {moved}"
        print(
            f"plant {plant.key} ({plant.name}): {verdict}; required {plant.must_move}"
            f"{'; must stay ' + str(plant.must_stay) if plant.must_stay else ''}; "
            f"columns moved: {sorted(found)}"
        )
        if "metric_values" in found and "run/state_digest" in found["metric_values"].get(
            "changed", {}
        ):
            print(
                "    run/state_digest first differs at row "
                f"{found['metric_values']['changed']['run/state_digest'][0]}"
            )
        if missing:
            failures.append(f"plant {plant.key!r} did not move {missing}")
        if moved:
            failures.append(f"plant {plant.key!r} moved {moved}, which it must leave equal")
    near = (
        "the near-miss expects were not run (they need the plain arm's metric_key and "
        "alarm_duplicate plants)"
    )
    if arm == "plain" and "metric_key" in results and "alarm_duplicate" in results:
        failures += _near_misses(
            null, results["metric_key"][1], ref, ref2, results["alarm_duplicate"][0]
        )
        near = "every near-miss expect was handled"
    failures += _parent_refusals(ref)
    refusals = "the child refusals were not run (plain arm on mock, all plants or --only refusals)"
    if arm == "plain" and engine == "mock" and (not only or "refusals" in only):
        failures += _child_refusals(base_sha, common, out, run, ref)
        refusals = "every child refusal refused for its reason"
    print(
        "\n".join(["SELFTEST FAILED:", *failures])
        if failures
        else "SELFTEST PASSED: every plant moved every column it names and left every column it "
        f"must leave; the null pair is identical on four hash seeds; {near}; {refusals}"
    )
    return 1 if failures else 0


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def _gym_sha(args: argparse.Namespace) -> str:
    if getattr(args, "gym_from", None):
        if args.gym is not None:
            raise SystemExit("give --gym or --gym-from, not both")
        recorded = _load_result(args.gym_from).get("gym_commit")
        if not recorded:
            raise SystemExit(f"{args.gym_from} records no gym_commit")
        return _resolve(_find_repo("RoyaleGym"), recorded)
    return _resolve(_find_repo("RoyaleGym"), args.gym or "HEAD")


def _run_options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(args.min_free_mb, args.wait_minutes, getattr(args, "forbid_module", None))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def machine(p: argparse.ArgumentParser) -> None:
        p.add_argument("--gym", default=None, help="RoyaleGym commit (default HEAD, resolved once)")
        p.add_argument(
            "--gym-from", type=Path, default=None, help="take gym_commit from this result"
        )
        p.add_argument("--min-free-mb", type=float, default=700.0)
        p.add_argument("--wait-minutes", type=float, default=20.0)

    run = sub.add_parser("run")
    run.add_argument("commit")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--arm", choices=("plain", "il"), default="plain")
    run.add_argument("--engine", choices=("mock", "rust"), default="mock")
    run.add_argument("--artifacts", type=Path, default=None, help="the il arm's artifacts folder")
    run.add_argument(
        "--il-shape", default="legacy", help="legacy, sections, or a JSON fragment file"
    )
    run.add_argument("--imitate", default=None, help="RoyaleImitate commit to pin on PYTHONPATH")
    run.add_argument("--forbid-module", default=None, help="fail if a child imports this module")
    machine(run)
    make = sub.add_parser("artifacts")
    make.add_argument("commit")
    make.add_argument("--out", type=Path, required=True)
    machine(make)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)
    cmp_.add_argument("--expect", type=Path, default=None)
    lint = sub.add_parser("check-expect")
    lint.add_argument("expect", type=Path)
    test = sub.add_parser("selftest")
    test.add_argument("--base", required=True)
    test.add_argument("--out", type=Path, required=True)
    test.add_argument("--arm", choices=("plain", "il"), default="plain")
    test.add_argument("--engine", choices=("mock", "rust"), default="mock")
    test.add_argument("--artifacts", type=Path, default=None)
    test.add_argument("--il-shape", default="legacy")
    test.add_argument("--only", default=None, help="comma-separated plant keys")
    machine(test)
    args = parser.parse_args(argv)

    if args.command == "compare":
        return compare(args.a, args.b, args.expect)
    if args.command == "check-expect":
        problems = lint_expect(json.loads(args.expect.read_text(encoding="utf-8")))
        print("\n".join(problems) if problems else "expect file is well formed")
        return 1 if problems else 0
    learn_root = _learn_root()
    gym_sha = _gym_sha(args)
    options = _run_options(args)
    if args.command == "run":
        learn_sha = _resolve(learn_root, args.commit)
        imitate_sha = _resolve(_find_repo("RoyaleImitate"), args.imitate) if args.imitate else None
        side = Side(
            learn_sha,
            gym_sha,
            arm=args.arm,
            engine=args.engine,
            artifacts=args.artifacts,
            il_shape=args.il_shape,
            imitate_sha=imitate_sha,
        )
        result = run_side(side, args.out, options)
        print(f"{result}  (commit {learn_sha[:12]}, gym {gym_sha[:12]}, {args.arm}/{args.engine})")
        return 0
    if args.command == "artifacts":
        learn_sha = _resolve(learn_root, args.commit)
        print(make_artifacts(learn_sha, gym_sha, args.out, options))
        return 0
    if args.arm == "il" and args.artifacts is None:
        raise SystemExit("selftest --arm il needs --artifacts")
    return selftest(
        _resolve(learn_root, args.base),
        gym_sha,
        args.out,
        options,
        arm=args.arm,
        engine=args.engine,
        artifacts=args.artifacts,
        il_shape=args.il_shape,
        only=args.only.split(",") if args.only else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
