"""What a resume restores, and what it cannot restore yet.

``docs/harness-spec.md`` section 12.4 is the claim these tests hold to account. Three of its
four clauses are testable today and are tested here: the learner's whole state comes back byte
for byte in a fresh process, a run is a pure function of its identity, and an identity that has
drifted is refused by name rather than continued.

The fourth clause -- that the iterations after a resume reproduce the original's metric rows --
does **not** hold, for a reason outside this repository. ``ClashSelfPlayVecEnv`` autoresets
without a seed, so a battle's RNG advances with every episode it has played, and a worker that
starts fresh cannot arrive at that state without replaying every episode before it. The
environment therefore continues from a different point even though every learner byte matches.
``test_the_environment_does_not_resume_and_says_so`` measures that gap rather than papering over
it, and it is section 16's ask of RoyaleGym: an ``autoreset_seed_fn`` makes an episode
addressable, and the clause becomes true the day it lands.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import royalelearn.config as cfg
from test_coordinator import coordinator, tiny_config

pytestmark = pytest.mark.slow

#: Iterations before the checkpoint, and after it.
SPLIT = 3
AFTER = 2


def resumable_config(tmp_path: Path) -> cfg.RunConfig:
    """``tiny_config``, with the settings a resume needs and nothing else changed."""
    return tiny_config(
        tmp_path,
        # The guarantee is about gradients as well as episodes, which is what this tier buys.
        determinism=cfg.DeterminismConfig(tier="run_exact"),
        # A checkpoint has to exist to resume from, and the split has to land on one.
        checkpoint=cfg.CheckpointConfig(every_env_steps=1, keep=8),
        # Two battles of a toy engine play about one card a match, which is a collapse by the
        # thresholds of a real run and is simply what this size of run looks like. The alarms
        # measure a policy; this file measures whether a resume reproduces one.
        alarms=cfg.AlarmConfig(enabled=False),
    )


def rows(run_dir: Path) -> list[dict[str, Any]]:
    """Every metric row a run wrote, in order."""
    text = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def timesteps_for(config: cfg.RunConfig, iterations: int) -> int:
    return config.ppo.timesteps_per_iteration * iterations


def run_to(config: cfg.RunConfig, iterations: int) -> Path:
    """Run from scratch for ``iterations`` iterations and return the run directory."""
    with coordinator(config) as run:
        run.learn(until_timesteps=timesteps_for(config, iterations))
        return Path(run.run_dir)


def resume(run_dir: Path, *, until: int | None = None) -> subprocess.CompletedProcess[str]:
    """Continue a run in a **fresh process**, which is the only resume worth having.

    One that works inside the interpreter that wrote the checkpoint skips the machinery being
    tested: restoring torch's generator, the schedules, and each shard's position.
    """
    command = [sys.executable, "-m", "royalelearn", "resume", "--run", str(run_dir)]
    if until is not None:
        command += ["--until-timesteps", str(until)]
    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def test_a_run_is_a_pure_function_of_its_identity(tmp_path: Path) -> None:
    """Two runs of one configuration agree row for row, which everything else rests on.

    If this failed, nothing about a resume could be tested at all: there would be no original
    for a continuation to be compared against.
    """
    first = rows(run_to(resumable_config(tmp_path / "a"), SPLIT))
    second = rows(run_to(resumable_config(tmp_path / "b"), SPLIT))
    assert len(first) == len(second) == SPLIT
    for index, (left, right) in enumerate(zip(first, second, strict=True), start=1):
        assert left["run/state_digest"] == right["run/state_digest"], (
            f"two runs of one configuration reached different states at iteration {index}"
        )
        differing = {
            key: (left.get(key), right.get(key))
            for key in sorted(set(left) | set(right))
            if not key.startswith(("time/", "throughput/", "run/wall_seconds"))
            and left.get(key) != right.get(key)
        }
        assert not differing, f"iteration {index} differs in {differing}"


def test_a_resume_restores_the_learner_byte_for_byte(tmp_path: Path) -> None:
    """The state digest a fresh process comes back with is the one the checkpoint recorded.

    That digest is the actor and critic weights, both optimizers' moments, the return scaler
    and the schedule positions hashed in a fixed order, so this is the whole learner and not a
    curve that looks similar.
    """
    config = resumable_config(tmp_path / "split")
    run_dir = run_to(config, SPLIT)
    checkpointed = rows(run_dir)[-1]["run/state_digest"]

    done = resume(run_dir, until=timesteps_for(config, SPLIT))
    assert done.returncode == 0, f"resume failed:\n{done.stdout}\n{done.stderr}"
    assert checkpointed[:16] in done.stdout, (
        f"a resumed run reported a different state than the checkpoint recorded: "
        f"expected {checkpointed[:16]}, got:\n{done.stdout}"
    )


def test_the_environment_does_not_resume_and_says_so(tmp_path: Path) -> None:
    """The gap, measured: the learner comes back and the battles do not.

    ``ClashSelfPlayVecEnv`` autoresets without a seed, so a battle's generator has advanced once
    per episode it has played and a fresh worker cannot arrive there without replaying them.
    The first iteration after a resume therefore collects different battles, and the run
    diverges from the one it is continuing -- in the environment's numbers, never in the
    learner's restoration.

    This test exists to fail the day that stops being true, which is the day RoyaleGym gives
    ``ClashSelfPlayVecEnv`` a seeded autoreset. Delete it then, and assert the row-for-row
    continuation section 12.4 describes.
    """
    config = resumable_config(tmp_path / "gap")
    whole = rows(run_to(resumable_config(tmp_path / "whole"), SPLIT + AFTER))

    split_dir = run_to(config, SPLIT)
    done = resume(split_dir, until=timesteps_for(config, SPLIT + AFTER))
    assert done.returncode == 0, f"resume failed:\n{done.stdout}\n{done.stderr}"
    continued = rows(split_dir)
    assert len(continued) == SPLIT + AFTER

    for index in range(SPLIT):
        assert continued[index]["run/state_digest"] == whole[index]["run/state_digest"], (
            f"the two runs disagreed at iteration {index + 1}, before the checkpoint, which is "
            f"a determinism failure rather than the resume gap this test measures"
        )

    after = continued[SPLIT]
    original = whole[SPLIT]
    environmental = {
        key
        for key in set(original) | set(after)
        if key.startswith(("env/", "policy/")) and original.get(key) != after.get(key)
    }
    assert environmental, (
        "the environment reproduced through a resume. If RoyaleGym has gained a seeded "
        "autoreset, this test has done its job: replace it with the row-for-row comparison "
        "of section 12.4 and update the spec."
    )


def test_a_resumed_run_refuses_an_identity_that_drifted(tmp_path: Path) -> None:
    """The other half of the promise: it says no rather than continuing something else.

    A checkpoint carries the identity of the run that wrote it, and a resume under a different
    one is a different experiment wearing the first one's curve. The refusal names the field.
    """
    config = resumable_config(tmp_path / "drift")
    run_dir = run_to(config, 1)

    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    stored["master_seed"] = int(stored["master_seed"]) + 1
    (run_dir / "config.json").write_text(json.dumps(stored, indent=2), encoding="utf-8")

    done = resume(run_dir)
    assert done.returncode != 0, "a drifted identity was resumed rather than refused"
    assert "master_seed" in (done.stdout + done.stderr), (
        f"the refusal did not name the field that drifted:\n{done.stdout}\n{done.stderr}"
    )
