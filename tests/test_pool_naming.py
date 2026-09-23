"""Candidate names are issued once per run, and a resume does not start again at zero.

A candidate's id is how the result log, the rating fit and the snapshot archive all refer to one
set of weights. Two different candidates under one name is not a cosmetic problem: the gate would
compare a snapshot against its own predecessor's record, and the fit would average two players.
"""

from __future__ import annotations

import msgspec

from royalelearn.ladder.pool import LadderPool, PoolState
from royalelearn.ladder.results import GameResult, ResultLog


def _pool(tmp_path, folder: str = "ladder") -> LadderPool:
    return LadderPool(ResultLog(tmp_path / folder / "games.jsonl"), context="ctx")


def _snapshots(pool: LadderPool) -> list[str]:
    """The pool's snapshots, without the scripted anchors it is born holding."""
    return sorted(name for name in pool.members() if name.startswith("snap:"))


def test_a_resumed_pool_does_not_hand_out_a_name_it_has_already_used(tmp_path) -> None:
    pool = _pool(tmp_path)
    for step in range(3):
        pool.add(pool.issue_candidate_id(), step=step)
    assert _snapshots(pool) == ["snap:v0", "snap:v1", "snap:v2"]
    pool.save_checkpoint(tmp_path / "checkpoint")

    resumed = _pool(tmp_path, folder="resumed")
    resumed.load_checkpoint(tmp_path / "checkpoint")
    assert resumed.issue_candidate_id() == "snap:v3"


def test_an_older_checkpoint_reads_its_count_back_from_the_names_it_holds(tmp_path) -> None:
    """Format 1 did not store the counter, and the names it did store are the evidence.

    Read back from the members alone it would be wrong the moment a snapshot is evicted, so the
    ones it evicted and the chain of champions count too.
    """
    pool = _pool(tmp_path)
    for step in range(4):
        pool.add(pool.issue_candidate_id(), step=step)
    pool.promote("snap:v2")
    pool.save_checkpoint(tmp_path / "checkpoint")

    # An old checkpoint, with the newest snapshot evicted and no counter in the document.
    state_path = tmp_path / "checkpoint" / "champion.json"
    document = msgspec.json.decode(state_path.read_bytes())
    document.pop("snapshots_issued")
    document["sampled"] = [name for name in document["sampled"] if name != "snap:v3"]
    document["evicted"] = ["snap:v3"]
    state_path.write_bytes(msgspec.json.encode(document))
    assert "snapshots_issued" not in msgspec.json.decode(state_path.read_bytes())

    resumed = _pool(tmp_path, folder="resumed")
    resumed.load_checkpoint(tmp_path / "checkpoint")
    assert resumed.issue_candidate_id() == "snap:v4"


def test_an_older_checkpoint_counts_a_candidate_the_gate_rejected(tmp_path) -> None:
    """A rejected candidate is in no register the pool keeps, and its name was spent anyway.

    It is never added, so it is not a member, was never evicted and never became champion. The
    only trace it leaves is the games the gate played to reject it, which are in the result log
    under its id. A backfill that read the pool's own lists alone would hand that name out again,
    and the log would then hold two players under one id.
    """
    pool = _pool(tmp_path)
    kept = pool.issue_candidate_id()
    pool.add(kept, step=0)
    rejected = pool.issue_candidate_id()  # the gate auditions it and says no
    pool.record(
        [
            GameResult(
                a=rejected,
                b=kept,
                score_a=0.0,
                seed_index=index,
                side_a="blue",
                context="ctx",
                kind="eval",
                run_id="run",
                iteration=1,
                wall="",
            )
            for index in range(4)
        ]
    )
    pool.save_checkpoint(tmp_path / "checkpoint")

    state_path = tmp_path / "checkpoint" / "champion.json"
    document = msgspec.json.decode(state_path.read_bytes())
    document.pop("snapshots_issued")
    state_path.write_bytes(msgspec.json.encode(document))

    resumed = _pool(tmp_path)
    resumed.load_checkpoint(tmp_path / "checkpoint")
    assert resumed.issue_candidate_id() == "snap:v2", "it handed out the rejected name again"


def test_the_counter_counts_names_handed_out_and_not_snapshots_kept(tmp_path) -> None:
    """A candidate that fails its gate is never added, and its name is still spent."""
    pool = _pool(tmp_path)
    assert pool.issue_candidate_id() == "snap:v0"  # rejected by the gate, never added
    assert pool.issue_candidate_id() == "snap:v1"
    pool.add("snap:v1", step=10)
    assert _snapshots(pool) == ["snap:v1"]
    assert pool.state.snapshots_issued == 2


def test_a_fresh_pool_starts_at_v0(tmp_path) -> None:
    assert PoolState().snapshots_issued == 0
    assert _pool(tmp_path).issue_candidate_id() == "snap:v0"
