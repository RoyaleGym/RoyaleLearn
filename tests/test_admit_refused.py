"""``ladder.admit_refused``: every due candidate is gated, and a refused one joins the pool.

Without it a floor admission due at the same step as a gate takes the gate's place, so a floor
cadence that is a multiple of the candidate cadence admits every second candidate ungated, and a
candidate the gate refuses is discarded. The run-level tests script the gate: what is tested is
what the run does with a verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn.api.ladder import ConditionResult, GateDecision
from royalelearn.config import LadderConfig, admit_refused_of
from royalelearn.ladder.gate import CONDITION_CHAMPION
from royalelearn.testing import coordinator, tiny_config


class _ScriptedGate:
    """Promotes the first candidate (as the gate does into an empty ladder), refuses the rest."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def evaluate(self, candidate: str, pool: Any, runner: Any) -> Any:
        self.asked.append(candidate)
        first = pool.champion is None
        return GateDecision(
            candidate=candidate,
            champion=pool.champion or "",
            admit=first,
            promote=first,
            cycle=False,
            conditions={}
            if first
            else {
                CONDITION_CHAMPION: ConditionResult(
                    passed=False, n=800, observed=0.4, bound=0.37, reference=0.5
                )
            },
            eval_seed_set_sha="",
            wall_seconds=0.0,
        )


def _config(tmp_path: Path, *, floor: int, refused: bool) -> Any:
    base = tiny_config(tmp_path)
    ladder = msgspec.structs.replace(
        base.ladder, candidate_every_env_steps=1, floor_admit_every_env_steps=floor
    )
    if refused:
        ladder = msgspec.structs.replace(ladder, admit_refused=True)
    return tiny_config(
        tmp_path, ladder=ladder, alarms=msgspec.structs.replace(base.alarms, enabled=False)
    )


def test_unset_nothing_is_encoded() -> None:
    assert b"admit_refused" not in msgspec.json.encode(LadderConfig())
    assert admit_refused_of(LadderConfig()) is False


@pytest.mark.parametrize("refused", [False, True])
def test_a_refused_candidate_joins_the_pool_only_under_the_setting(
    tmp_path: Path, refused: bool
) -> None:
    with coordinator(_config(tmp_path, floor=0, refused=refused)) as run:
        run.gate = _ScriptedGate()
        run.iterate()
        run.iterate()
        second = run.gate.asked[-1]
        assert run.pool.champion == run.gate.asked[0]
        assert (second in run.pool.sampler()) is refused
        assert run.pool.champion != second


@pytest.mark.parametrize("refused", [False, True])
def test_a_due_gate_is_not_replaced_by_the_floor_under_the_setting(
    tmp_path: Path, refused: bool
) -> None:
    """Both due at every step. Unset, the floor takes the gate's place and the gate is never
    asked; set, the gate is asked every time and the floor's clock moves with it."""
    with coordinator(_config(tmp_path, floor=1, refused=refused)) as run:
        run.gate = _ScriptedGate()
        for _ in range(3):
            run.iterate()
        assert len(run.gate.asked) == (3 if refused else 0)
        assert len(run.pool.sampler()) == 3
