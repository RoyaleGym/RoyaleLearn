"""``ladder.rollback_after`` and ``ladder.stall_after``: back to the champion, and the stop.

Two straight gates the candidate loses below ``rollback_below`` against the champion, and the
learner goes back to the champion's weights and both optimizers' states, saved beside the
candidate before its gate. ``stall_after`` straight gates without a promotion, and the run stops
and saves itself. Only gates that played count; a promotion ends both streaks.

The run-level tests script the gate, because what is being tested is what the run does with a
verdict and not how the verdict is reached; the gate has its own tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn.api.ladder import ConditionResult, GateDecision
from royalelearn.config import (
    LadderConfig,
    RunConfig,
    check_consistency,
    rollback_after_of,
    rollback_below_of,
    stall_after_of,
)
from royalelearn.ladder.gate import CONDITION_CHAMPION, floor_decision
from royalelearn.ladder.recovery import ROLLBACK, STOP, Recovery, recovery_enabled
from royalelearn.testing import coordinator, tiny_config

torch = pytest.importorskip("torch")


def _played(candidate: str, champion: str, observed: float, *, promote: bool = False) -> Any:
    return GateDecision(
        candidate=candidate,
        champion=champion,
        admit=promote,
        promote=promote,
        cycle=False,
        conditions={
            CONDITION_CHAMPION: ConditionResult(
                passed=promote, n=800, observed=observed, bound=observed - 0.03, reference=0.5
            )
        },
        eval_seed_set_sha="",
        wall_seconds=0.0,
    )


def _first(candidate: str) -> Any:
    """Promoted into an empty ladder, without a game."""
    return GateDecision(
        candidate=candidate,
        champion="",
        admit=True,
        promote=True,
        cycle=False,
        conditions={},
        eval_seed_set_sha="",
        wall_seconds=0.0,
    )


def _ladder(**fields: Any) -> LadderConfig:
    return msgspec.structs.replace(LadderConfig(), **fields)


# -- the config ----------------------------------------------------------------


def test_unset_nothing_is_encoded_and_both_are_off() -> None:
    encoded = msgspec.json.encode(LadderConfig())
    for name in ("rollback_after", "rollback_below", "stall_after"):
        assert name.encode() not in encoded
    assert (rollback_after_of(LadderConfig()), stall_after_of(LadderConfig())) == (0, 0)
    assert rollback_below_of(LadderConfig()) == 0.45
    assert not recovery_enabled(LadderConfig())


@pytest.mark.parametrize(
    "fields,words",
    [
        ({"rollback_after": -1}, "rollback_after must not be negative"),
        ({"stall_after": -2}, "stall_after must not be negative"),
        ({"rollback_after": 2, "rollback_below": 1.5}, "rollback_below is 1.5"),
        ({"rollback_below": 0.4}, "the only rule that reads it"),
    ],
)
def test_a_setting_nothing_can_honour_is_named(fields: dict[str, Any], words: str) -> None:
    problems = check_consistency(RunConfig(ladder=_ladder(**fields)))
    assert any(words in problem for problem in problems)


# -- the rule ------------------------------------------------------------------


def _recovery(tmp_path: Path, **fields: Any) -> Recovery:
    return Recovery(_ladder(**fields), tmp_path / "learners", keep=2)


def _saved(recovery: Recovery, candidate: str) -> Path:
    folder = recovery.folder_for(candidate)
    folder.mkdir(parents=True)
    (folder / "weights").write_text(candidate)
    return folder


def test_two_straight_gates_lost_badly_roll_back_and_a_close_one_breaks_the_run(
    tmp_path: Path,
) -> None:
    recovery = _recovery(tmp_path, rollback_after=2)
    first = recovery.after_gate(_first("snap:v0"), previous=None, saved=_saved(recovery, "snap:v0"))
    assert first == ""
    assert recovery.learner_of("snap:v0") == recovery.folder_for("snap:v0")
    verdicts = [
        recovery.after_gate(_played(f"snap:v{i}", "snap:v0", score), previous="snap:v0", saved=None)
        for i, score in enumerate((0.40, 0.46, 0.30, 0.44, 0.30), start=1)
    ]
    # 0.40 below, 0.46 not (the streak ends), 0.30 below, 0.44 below: back; 0.30 starts again.
    assert verdicts == ["", "", "", ROLLBACK, ""]
    assert (recovery.state.below, recovery.state.since_promotion) == (1, 5)


def test_only_played_gates_count_and_a_promotion_ends_both_streaks(tmp_path: Path) -> None:
    recovery = _recovery(tmp_path, rollback_after=2, stall_after=3)
    recovery.after_gate(_first("snap:v0"), previous=None, saved=None)
    recovery.after_gate(_played("snap:v1", "snap:v0", 0.3), previous="snap:v0", saved=None)
    floor = floor_decision("snap:v2", "snap:v0")
    assert recovery.after_gate(floor, previous="snap:v0", saved=None) == ""
    assert (recovery.state.below, recovery.state.since_promotion) == (1, 1)
    won = _played("snap:v3", "snap:v0", 0.6, promote=True)
    assert recovery.after_gate(won, previous="snap:v0", saved=_saved(recovery, "snap:v3")) == ""
    assert (recovery.state.below, recovery.state.since_promotion) == (0, 0)


def test_the_stall_stops_the_run_and_comes_before_a_rollback(tmp_path: Path) -> None:
    recovery = _recovery(tmp_path, rollback_after=2, stall_after=2)
    recovery.after_gate(_played("snap:v1", "snap:v0", 0.3), previous="snap:v0", saved=None)
    assert recovery.after_gate(
        _played("snap:v2", "snap:v0", 0.3), previous="snap:v0", saved=None
    ) == STOP


def test_a_candidates_learner_is_kept_only_while_it_is_a_recent_champion(tmp_path: Path) -> None:
    recovery = _recovery(tmp_path, rollback_after=2)  # keep=2
    lost = _saved(recovery, "snap:v1")
    recovery.after_gate(_played("snap:v1", "snap:v0", 0.3), previous="snap:v0", saved=lost)
    assert not lost.exists()
    folders = []
    for index in (2, 3, 4):
        folder = _saved(recovery, f"snap:v{index}")
        won = _played(f"snap:v{index}", f"snap:v{index - 1}", 0.6, promote=True)
        recovery.after_gate(won, previous=f"snap:v{index - 1}", saved=folder)
        folders.append(folder)
    assert [folder.exists() for folder in folders] == [False, True, True]
    assert list(recovery.state.learners) == ["snap:v3", "snap:v4"]
    assert recovery.learner_of("snap:v2") is None


def test_the_streaks_go_through_a_checkpoint_and_start_at_zero_without_one(tmp_path: Path) -> None:
    recovery = _recovery(tmp_path, rollback_after=3)
    recovery.after_gate(_played("snap:v1", "snap:v0", 0.3), previous="snap:v0", saved=None)
    recovery.save_checkpoint(tmp_path / "ckpt" / "recovery")
    again = _recovery(tmp_path, rollback_after=3)
    again.load_checkpoint(tmp_path / "ckpt" / "recovery")
    assert again.state == recovery.state
    fresh = _recovery(tmp_path, rollback_after=3)
    fresh.load_checkpoint(tmp_path / "nowhere")
    assert (fresh.state.below, fresh.state.since_promotion) == (0, 0)


# -- a run ---------------------------------------------------------------------


class _ScriptedGate:
    """The gate's verdicts in order: "first", a score below or above, "won"."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)

    def evaluate(self, candidate: str, pool: Any, runner: Any) -> Any:
        step = self.script.pop(0)
        champion = pool.champion or ""
        if step == "first":
            return _first(candidate)
        if step == "won":
            return _played(candidate, champion, 0.7, promote=True)
        return _played(candidate, champion, float(step))


def _config(tmp_path: Path, **ladder: Any) -> Any:
    """A gate every iteration and no floor admissions. The alarms are off: a tiny run's policy
    stops playing cards within a few iterations, which is not what these tests are about."""
    base = tiny_config(tmp_path)
    return tiny_config(
        tmp_path,
        ladder=msgspec.structs.replace(
            base.ladder, candidate_every_env_steps=1, floor_admit_every_env_steps=0, **ladder
        ),
        alarms=msgspec.structs.replace(base.alarms, enabled=False),
    )


def _is_saved(run: Any, folder: Path) -> bool:
    """Whether the live learner is the one saved in ``folder``: weights and both moments."""
    from safetensors.torch import load_file

    stored: dict[str, Any] = {}
    for name in ("actor.safetensors", "critic.safetensors"):
        stored.update(load_file(str(folder / "actor_critic" / name)))
    live = run.model.state_dict()
    if set(stored) != set(live):
        return False
    if not all(torch.equal(live[k].cpu(), v) for k, v in stored.items()):
        return False
    for optimizer, name in zip(
        run.update.optimizers, (run.update.ACTOR_FILE, run.update.CRITIC_FILE), strict=True
    ):
        kept = torch.load(folder / "optimizers" / name, weights_only=True)["state"]
        now = optimizer.state_dict()["state"]
        for key, values in kept.items():
            for field, value in values.items():
                if torch.is_tensor(value) and not torch.equal(now[key][field].cpu(), value):
                    return False
    return True


@pytest.mark.parametrize("overlap", [False, True])
def test_a_run_rolls_back_to_the_champion_and_carries_on(tmp_path: Path, overlap: bool) -> None:
    """The learner after the second bad gate is the one the champion was snapshotted from,
    weights and moments; the clocks carry on; with overlap the batch collected beside the
    update is dropped and the next iteration collects afresh. Plant: restore the weights and not
    the moments, and the comparison fails on the first moment tensor."""
    base = _config(tmp_path, rollback_after=2)
    config = msgspec.structs.replace(
        base, rollout=msgspec.structs.replace(base.rollout, overlap=overlap)
    )
    lines: list[str] = []
    with coordinator(config, printer=lines.append) as run:
        run.gate = _ScriptedGate(["first", "won", 0.3, 0.3, 0.6])
        # What learn() sets: without a limit an iteration does not collect the next beside it.
        run._limit = 10**12
        run.iterate()  # the first champion, promoted without a game
        run.iterate()  # a champion that won its gate
        champion = run.pool.champion
        at_champion = run.recovery.learner_of(champion)
        assert at_champion is not None
        run.iterate()  # lost badly: one
        assert run.rows[-1]["ladder/gates_below"] == 1
        assert (run._prefetched is not None) is overlap
        before = (run.iteration, run.cumulative_env_steps, run.update.model_updates)
        run.iterate()  # lost badly: two, and back
        assert run.rows[-1]["ladder/rollbacks"] == 1
        assert run.rows[-1]["ladder/gates_below"] == 0
        assert run._prefetched is None
        assert run.iteration == before[0] + 1
        assert run.update.model_updates > before[2]
        # The learner is the champion's, weights and moments, as saved before its gate.
        assert _is_saved(run, at_champion)
        run.iterate()  # carries on from the champion's learner
        assert run.rows[-1]["ladder/gates_since_promotion"] == 3
    assert any(line.startswith("rollback      1:") for line in lines)


def test_a_run_that_stops_promoting_stops_and_saves_itself(tmp_path: Path) -> None:
    config = _config(tmp_path, stall_after=2)
    lines: list[str] = []
    with coordinator(config, printer=lines.append) as run:
        run.gate = _ScriptedGate(["first", 0.5, 0.5])
        assert run.iterate() == ""
        assert run.iterate() == ""
        assert run.iterate() == "q"
        saved = run._saved_at
    assert saved is not None and saved[1].is_dir()
    assert any("2 straight gates without a promotion" in line for line in lines)


def test_a_resume_keeps_the_streak_and_the_champions_learner(tmp_path: Path) -> None:
    config = _config(tmp_path, rollback_after=2)
    with coordinator(config) as run:
        run.gate = _ScriptedGate(["first", "won", 0.3])
        for _ in range(3):
            run.iterate()
        champion = run.pool.champion
        saved = run.checkpoint()
        at_champion = run.recovery.learner_of(champion)
    with coordinator(config, resume=saved) as resumed:
        assert resumed.recovery.state.below == 1
        assert resumed.recovery.learner_of(champion) == at_champion
        resumed.gate = _ScriptedGate([0.3])
        resumed.iterate()
        assert resumed.rows[-1]["ladder/rollbacks"] == 1


def test_turned_on_at_a_resume_it_starts_with_nothing_to_return_to(tmp_path: Path) -> None:
    """A checkpoint written with both rules off has no recovery folder, and a strict resume
    with them on reads it as zeros. Its champion has no saved learner, so a rollback before the
    next promotion is skipped and said so."""
    off = _config(tmp_path)
    with coordinator(off) as run:
        run.gate = _ScriptedGate(["first", "won"])
        run.iterate()
        run.iterate()
        saved = run.checkpoint()
    on = _config(tmp_path, rollback_after=1)
    lines: list[str] = []
    with coordinator(on, resume=saved, allow_identity_drift=True, printer=lines.append) as run:
        assert run.recovery.state.below == 0
        run.gate = _ScriptedGate([0.3])
        run.iterate()
        assert run.rows[-1]["ladder/rollbacks"] == 0
    assert any(line.startswith("rollback      skipped") for line in lines)
