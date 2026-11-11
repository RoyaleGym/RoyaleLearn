"""Rollback to the champion, and the stop when gates stop promoting (``ladder.rollback_after``,
``ladder.stall_after``).

A gate that the candidate loses badly says the learner has got worse than the champion it is
measured against, not merely that it has not got better. Two such gates in a row, and the learner
goes back to where the champion was snapshotted: its weights and both optimizers' moments, saved
beside the candidate before its gate and kept while it is champion. The clocks, the counters, the
coefficients and the pool do not go back; the learner starts again from the champion's state in
the run's present. A run of gates that promote nothing at all is a plateau, and past
``stall_after`` of them the run stops, says why, and saves itself.

Only gates that played count. A floor admission, the first champion promoted into an empty
ladder and a candidate that is the champion again decide nothing by playing, so they move
neither streak. A promotion ends both.

This module holds the rule and its state. The coordinator does the saving and loading of the
learner, because the learner is the coordinator's.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from ..config import rollback_after_of, rollback_below_of, stall_after_of
from .gate import CONDITION_CHAMPION

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.ladder import GateDecision
    from ..config import LadderConfig

__all__ = ["ROLLBACK", "STOP", "Recovery", "RecoveryState", "recovery_enabled"]

#: What ``Recovery.after_gate`` asks the coordinator to do.
ROLLBACK = "rollback"
STOP = "stop"


def recovery_enabled(ladder: LadderConfig) -> bool:
    """Whether either rule is on. Off, the run carries no recovery state at all."""
    return rollback_after_of(ladder) > 0 or stall_after_of(ladder) > 0


class RecoveryState(msgspec.Struct):
    """What the two rules count, and where each champion's learner is kept."""

    #: Straight played gates in which the candidate scored below ``rollback_below`` against the
    #: champion. Ended by a gate at or above it, by a promotion and by a rollback.
    below: int = 0
    #: Straight played gates without a promotion. Ended only by a promotion.
    since_promotion: int = 0
    rollbacks: int = 0
    #: champion id -> the folder holding its learner, newest last. The last ``keep`` champions'
    #: folders are kept, so a resume from an older checkpoint still finds the one it names.
    learners: dict[str, str] = msgspec.field(default_factory=dict)


class Recovery:
    """The rule, its state, and the folders of the learners it may return to."""

    FILE = "recovery.json"
    FORMAT_VERSION = 1
    #: A checkpoint written with both rules off has no folder for this, and a run that turns
    #: them on resumes from it: ``CheckpointStore.read`` hands the missing folder to
    #: ``load_checkpoint``, which starts the streaks at zero and says so.
    MAY_BE_ABSENT = True

    def __init__(self, config: LadderConfig, root: Path, *, keep: int) -> None:
        self.config = config
        self.root = Path(root)
        self.keep = max(1, int(keep))
        self.state = RecoveryState()

    # -- the learners --------------------------------------------------------

    def folder_for(self, candidate: str) -> Path:
        """Where a candidate's learner is saved before its gate."""
        return self.root / candidate.replace(":", "_")

    def learner_of(self, champion: str | None) -> Path | None:
        """The folder holding ``champion``'s learner, or None when it has none: promoted before
        either rule was on, or its folder is gone."""
        if champion is None:
            return None
        recorded = self.state.learners.get(champion)
        if recorded is None or not Path(recorded).is_dir():
            return None
        return Path(recorded)

    def _promoted(self, champion: str, folder: Path) -> None:
        learners = {k: v for k, v in self.state.learners.items() if k != champion}
        learners[champion] = str(folder)
        while len(learners) > self.keep:
            oldest = next(iter(learners))
            shutil.rmtree(learners.pop(oldest), ignore_errors=True)
        self.state.learners = learners

    # -- the rule ------------------------------------------------------------

    def after_gate(
        self, decision: GateDecision, *, previous: str | None, saved: Path | None
    ) -> str:
        """Count one gate's verdict, and say what the coordinator must do: ``ROLLBACK``,
        ``STOP`` or nothing (an empty string).

        ``previous`` is the champion the candidate was measured against, and ``saved`` the folder
        its learner was saved in before the gate, or None when it was not saved. A promoted
        candidate's folder becomes the champion's; any other is deleted.
        """
        promoted = decision.promote and decision.candidate != previous
        if saved is not None:
            if promoted:
                self._promoted(decision.candidate, saved)
            else:
                shutil.rmtree(saved, ignore_errors=True)
        if promoted:
            self.state.below = 0
            self.state.since_promotion = 0
            return ""
        condition = decision.conditions.get(CONDITION_CHAMPION)
        if condition is None or condition.skipped:
            return ""  # decided without playing: it moves neither streak
        self.state.since_promotion += 1
        if condition.observed < rollback_below_of(self.config):
            self.state.below += 1
        else:
            self.state.below = 0
        # The stop first: a run about to end gains nothing from going back.
        stall = stall_after_of(self.config)
        if stall > 0 and self.state.since_promotion >= stall:
            return STOP
        after = rollback_after_of(self.config)
        if after > 0 and self.state.below >= after:
            self.state.below = 0
            return ROLLBACK
        return ""

    # -- checkpoint ----------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / self.FILE).write_bytes(msgspec.json.encode(self.state))

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        """Put the streaks back. A checkpoint written with both rules off has none, and a run
        that turns them on resumes from it with both streaks at zero and no champion's learner
        to return to until its next promotion; it says so rather than refuse."""
        path = folder / self.FILE
        if not path.exists():
            print(
                f"no recovery state at {path}; the gate streaks start at zero and the champion "
                "has no saved learner to roll back to until the next promotion"
            )
            self.state = RecoveryState()
            return
        self.state = msgspec.json.decode(path.read_bytes(), type=RecoveryState)
