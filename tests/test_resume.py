"""What a resume restores, and what it cannot restore yet.

``docs/harness-spec.md`` section 12.4 is the claim these tests hold to account. Three of its
four clauses are testable today and are tested here: the learner's whole state comes back byte
for byte in a fresh process, a run is a pure function of its identity, and an identity that has
drifted is refused by name rather than continued.

The fourth clause -- that the iterations after a resume reproduce the original's metric rows --
holds from an episode boundary and not from inside an episode, and the two tests below draw that
line. ``ClashSelfPlayVecEnv`` now takes an ``autoreset_seed_fn``, so an episode is addressed by
its battle and its ordinal rather than by how many came before it, and a resumed worker is put
back on the ordinal the original was about to play. What is not restored is an episode that was
half finished when the checkpoint was written: its transitions were already in the original's
buffer, and replaying it would count them twice.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec
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


def aligned_config(tmp_path: Path) -> cfg.RunConfig:
    """``resumable_config`` with every episode ending on an iteration boundary.

    Four-decision episodes against four cycles an iteration, and no warm-up, so a checkpoint is
    never written while a battle is mid-episode. That is the condition the row-for-row
    guarantee is stated under, and the next test measures what happens without it.
    """
    config = resumable_config(tmp_path)
    config.env = cfg.default_env_spec(cfg.MOCK_ENGINE, max_steps=4)
    config.rollout = msgspec.structs.replace(config.rollout, stagger_first_reset=False)
    return config


#: Keys that measure the MACHINE rather than the run, and so cannot be part of "two runs of one
#: configuration agree row for row". A second process has its own peak working set, and the free
#: memory of a shared card is whatever else is on it at that instant. They were compared until
#: 2026-09-22 and did not fail only because both read a constant zero: making them real
#: (health/rss_peak_mb in afe30e7) turned them into a flake, and the vram pair had already been
#: seen failing this comparison while another session held the GPU.
MACHINE_READINGS = frozenset(
    {
        "health/rss_peak_mb",
        "health/vram_available_mb",
        "health/vram_driver_free_mb",
        "health/vram_reserved_mb",
        "health/vram_peak_mb",
        "health/vram_inactive_split_mb",
        "health/vram_alloc_retries",
        "health/vram_needed_mb",
    }
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


def skip_if_a_sibling_moved(done: subprocess.CompletedProcess[str]) -> None:
    """A resume refused because a sibling checkout changed under it is not a defect here.

    The identity carries ``royalelearn_git`` and ``royalegym_git``, each ``git describe --dirty``,
    and a resume refuses when the identity moved. On this machine seven sessions share the five
    checkouts, so a sibling can go from clean to dirty between the run that writes the checkpoint
    and the subprocess that resumes it -- which is a true identity difference, correctly refused,
    and says nothing about the resume path this file tests. Seen on 2026-09-22: "royalegym_git:
    checkpoint '56da55d', now '56da55d-dirty'", while these same tests passed on their own minutes
    later.

    So it is a SKIP naming the field that moved, not a pass and not a red somebody has to chase.
    Only that one cause skips; any other refusal is still a failure.
    """
    if done.returncode == 0:
        return
    output = f"{done.stdout}\n{done.stderr}"
    moved = [field for field in ("royalelearn_git", "royalegym_git") if field in output]
    if moved:
        pytest.skip(f"a sibling checkout moved under the resume: {', '.join(moved)}")


def resume(run_dir: Path, *, until: int | None = None) -> subprocess.CompletedProcess[str]:
    """Continue a run in a **fresh process**, which is the only resume worth having.

    One that works inside the interpreter that wrote the checkpoint skips the machinery being
    tested: restoring torch's generator, the schedules, and each shard's position.
    """
    # --allow-dirty because this tests resume's MECHANICS, and a working tree is dirty while it is
    # worked on -- including by sibling sessions' edits this test cannot control. The refusal
    # itself is tested on its own, in test_user_code_identity.
    command = [
        sys.executable,
        "-m",
        "royalelearn",
        "resume",
        "--run",
        str(run_dir),
        "--allow-dirty",
    ]
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
            if key not in MACHINE_READINGS
            and not key.startswith(("time/", "throughput/", "run/wall_seconds"))
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
    skip_if_a_sibling_moved(done)
    assert done.returncode == 0, f"resume failed:\n{done.stdout}\n{done.stderr}"
    assert checkpointed[:16] in done.stdout, (
        f"a resumed run reported a different state than the checkpoint recorded: "
        f"expected {checkpointed[:16]}, got:\n{done.stdout}"
    )


def test_a_resume_continues_the_original_row_for_row(tmp_path: Path) -> None:
    """Six iterations straight through against three then three, compared field by field.

    Episodes are four decisions and an iteration is four cycles, so every episode ends on an
    iteration boundary and the checkpoint is never taken mid-episode -- which is the condition
    the guarantee is stated under. The warm-up is off for the same reason: it exists to spread
    the first episodes' phases apart, and a phase that differs is the one thing that cannot be
    restored.

    What is compared is every metric row of the second half and the state digest inside it, so
    a curve that merely looked similar would not pass.
    """
    whole = rows(run_to(aligned_config(tmp_path / "whole"), SPLIT + AFTER))

    config = aligned_config(tmp_path / "split")
    split_dir = run_to(config, SPLIT)
    done = resume(split_dir, until=timesteps_for(config, SPLIT + AFTER))
    skip_if_a_sibling_moved(done)
    assert done.returncode == 0, f"resume failed:\n{done.stdout}\n{done.stderr}"
    continued = rows(split_dir)
    assert len(continued) == len(whole) == SPLIT + AFTER

    for index in range(SPLIT, SPLIT + AFTER):
        original, resumed = whole[index], continued[index]
        assert resumed["run/state_digest"] == original["run/state_digest"], (
            f"iteration {index + 1} of a resumed run reached a different state than the same "
            f"iteration of a straight run"
        )
        differing = {
            key: (original.get(key), resumed.get(key))
            for key in sorted(set(original) | set(resumed))
            if key not in MACHINE_READINGS
            and not key.startswith(("time/", "throughput/", "run/wall_seconds"))
            and original.get(key) != resumed.get(key)
        }
        assert not differing, f"iteration {index + 1} differs in {differing}"


def test_an_episode_in_flight_is_not_replayed(tmp_path: Path) -> None:
    """The limit, stated by measuring it rather than by asserting it in prose.

    A checkpoint taken while battles are mid-episode restores the ordinal each battle is about
    to play, not the episode it was in the middle of: those transitions were already in the
    original's buffer and replaying them would count them twice. So the battles a resumed run
    plays are the right ones and their phase is not, and the rows differ in how many episodes
    completed rather than in which were played.

    If this ever stops being true -- if resuming inside an episode is offered -- this test says
    so by failing, and the guarantee above can be widened.
    """
    whole = rows(run_to(resumable_config(tmp_path / "whole"), SPLIT + AFTER))

    config = resumable_config(tmp_path / "gap")
    split_dir = run_to(config, SPLIT)
    done = resume(split_dir, until=timesteps_for(config, SPLIT + AFTER))
    skip_if_a_sibling_moved(done)
    assert done.returncode == 0, f"resume failed:\n{done.stdout}\n{done.stderr}"
    continued = rows(split_dir)

    for index in range(SPLIT):
        assert continued[index]["run/state_digest"] == whole[index]["run/state_digest"], (
            "the two runs disagreed before the checkpoint, which is a determinism failure "
            "rather than the phase difference this test measures"
        )

    phase = {
        key
        for key in ("env/episodes_completed", "env/episode_steps_mean")
        if whole[SPLIT].get(key) != continued[SPLIT].get(key)
    }
    assert phase, (
        "a resume taken mid-episode reproduced the original's episode phase. If resuming "
        "inside an episode is now offered, widen the guarantee above and delete this test."
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
