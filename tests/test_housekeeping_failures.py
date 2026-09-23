"""Housekeeping must not be able to end a run.

Two places delete files while a run is going: the episode log is gzipped and the original removed
at ``metrics.keep_episode_log_iterations``, and the checkpoint store prunes everything past
``checkpoint.keep``. Both are archiving. Neither produces a number, and neither is worth a run.

ON WINDOWS THEY CAN BOTH FAIL. ``unlink`` and ``rmtree`` raise ``PermissionError`` WinError 32
while any other process holds a handle on the file -- verified on this machine, and POSIX does
both regardless. The train session reads ``episodes.jsonl`` while a run is going, which is what
its job requires and what ``coordinator.py`` names in so many words.

WHAT THAT COST BEFORE 2026-09-23. The compaction fires at iteration 200 and raised from inside
``sinks.write``, which the coordinator calls between setting ``_learner_ahead_of_rows`` and
clearing it -- so the run fell over AND ``_emergency_refusal`` suppressed the emergency
checkpoint, losing back to the last periodic one. It was not one-shot either: the sink's
iteration counter is restored from the pre-200 checkpoint, so a resumed run reached 200 again and
died at the same line. Pruning is worse by frequency: it runs at every checkpoint, which is about
1,990 times in a long run.

Found by the train session's platform-portability hunt, reported with the measurement.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from royalelearn.metrics.sinks import EPISODES_NAME, JsonlSink

#: Holding a file open blocks its deletion on Windows and does not on POSIX, where the name goes
#: and the inode lives until the last handle closes. So a test that creates the failure by
#: holding a handle can only run on Windows, and the first version of this file did exactly that
#: and turned Linux CI red at 4bd05ed. The guard itself is platform-independent and is tested at
#: the bottom of this file by INJECTING the error, so the runner that certifies exercises the
#: code either way; these are the fidelity, asserting the mechanism is real rather than only
#: that the guard works.
windows_only = pytest.mark.skipif(
    os.name != "nt", reason="deleting an open file only fails on Windows"
)


def _sink(tmp_path: Path, keep: int = 2) -> JsonlSink:
    sink = JsonlSink(keep_episode_log_iterations=keep)
    sink.open(identity=None, config_json="{}", run_dir=tmp_path)  # type: ignore[arg-type]
    return sink


def _episode(index: int) -> dict[str, object]:
    return {"worker": 0, "shard": 0, "battle": index, "ordinal": index}


@windows_only
def test_a_held_episode_log_does_not_end_the_run(tmp_path: Path) -> None:
    """The case the train session creates by doing its job: a reader holding the file open."""
    sink = _sink(tmp_path)
    sink.write_episodes([_episode(0)])  # type: ignore[arg-type]
    sink.write({"run/iteration": 1})

    held = (tmp_path / EPISODES_NAME).open("rb")
    try:
        sink.write_episodes([_episode(1)])  # type: ignore[arg-type]
        sink.write({"run/iteration": 2})  # the compaction fires here
    finally:
        held.close()
        sink.close()

    assert (tmp_path / EPISODES_NAME).exists(), "the run lost its episode log to a failed archive"


@windows_only
def test_a_failed_compaction_leaves_no_archive_claiming_to_have_replaced_it(
    tmp_path: Path,
) -> None:
    """The gzip copy happens BEFORE the unlink, so a failure between them leaves both files.

    A complete episodes.jsonl beside a complete episodes.jsonl.gz that claims to have replaced
    it is the worst of the three outcomes: a later reader cannot tell which is authoritative,
    and concatenating them double-counts every episode in the archive.
    """
    sink = _sink(tmp_path)
    sink.write_episodes([_episode(0)])  # type: ignore[arg-type]
    sink.write({"run/iteration": 1})

    held = (tmp_path / EPISODES_NAME).open("rb")
    try:
        sink.write_episodes([_episode(1)])  # type: ignore[arg-type]
        sink.write({"run/iteration": 2})
    finally:
        held.close()
        sink.close()

    archive = (tmp_path / EPISODES_NAME).with_suffix(".jsonl.gz")
    assert not archive.exists(), (
        "a half-finished archive was left beside the file it did not replace"
    )


@windows_only
def test_a_compaction_that_could_not_run_is_tried_again(tmp_path: Path) -> None:
    """Otherwise the one attempt is spent and the log grows for the rest of the run."""
    sink = _sink(tmp_path)
    sink.write_episodes([_episode(0)])  # type: ignore[arg-type]
    sink.write({"run/iteration": 1})

    held = (tmp_path / EPISODES_NAME).open("rb")
    sink.write_episodes([_episode(1)])  # type: ignore[arg-type]
    sink.write({"run/iteration": 2})  # fails
    held.close()

    sink.write_episodes([_episode(2)])  # type: ignore[arg-type]
    sink.write({"run/iteration": 3})  # succeeds
    sink.close()

    archive = (tmp_path / EPISODES_NAME).with_suffix(".jsonl.gz")
    assert archive.exists(), "the archive was never written after the holder let go"


def _checkpoints(root: Path, steps: tuple[int, ...]) -> list[Path]:
    """Folders and an index that ``_entries`` will actually read.

    Through the INDEX rather than hand-written manifests. A ``Manifest`` carries a whole
    ``RunIdentity`` and the resolved config, and the first version of this test wrote a
    four-field imitation of one: ``_entries`` skipped every folder, ``prune`` removed nothing,
    and the test passed by asserting that the newest checkpoint -- which nothing had tried to
    delete -- was still there. It passed before the fix and after it and tested neither.
    """
    import msgspec

    from royalelearn.checkpoint import INDEX_NAME, CheckpointIndex, IndexEntry

    root.mkdir(parents=True, exist_ok=True)
    folders = []
    entries = []
    for index, value in enumerate(steps):
        folder = root / f"{value:012d}"
        folder.mkdir()
        (folder / "weights.bin").write_bytes(b"x" * 16)
        folders.append(folder)
        entries.append(
            IndexEntry(
                name=folder.name,
                iteration=index,
                cumulative_env_steps=value,
                created_unix_ns=value,
            )
        )
    (root / INDEX_NAME).write_bytes(msgspec.json.encode(CheckpointIndex(entries=entries)))
    return folders


def test_pruning_really_removes_the_old_ones(tmp_path: Path) -> None:
    """The control. Without it the test below can pass by pruning nothing at all."""
    from royalelearn.checkpoint import DirCheckpointStore

    store = DirCheckpointStore(tmp_path, keep=1)
    folders = _checkpoints(store.folder(), (100, 200))
    removed = store.prune(keep=1)

    assert [path.name for path in removed] == [folders[0].name]
    assert not folders[0].exists()
    assert folders[1].exists()


@windows_only
def test_a_held_checkpoint_does_not_end_the_run(tmp_path: Path) -> None:
    """Pruning runs at every checkpoint, so this is the one that fires ~1,990 times a run.

    ``shutil.rmtree`` raises ``PermissionError`` on Windows while any file inside the folder is
    open -- verified on this machine. Anything reading a checkpoint while a run continues, which
    is an ordinary thing to do, could end it.
    """
    from royalelearn.checkpoint import DirCheckpointStore

    store = DirCheckpointStore(tmp_path, keep=1)
    folders = _checkpoints(store.folder(), (100, 200))

    held = (folders[0] / "weights.bin").open("rb")
    try:
        removed = store.prune(keep=1)
    finally:
        held.close()

    assert folders[1].exists(), "the newest checkpoint was lost to a failed prune"
    assert folders[0].exists(), "a held checkpoint was somehow removed"
    assert removed == [], "a folder that could not be removed was reported as removed"
    assert store.prune_failures == 1


@windows_only
def test_a_checkpoint_that_could_not_be_pruned_is_tried_again(tmp_path: Path) -> None:
    """Otherwise a single held file leaves the directory growing for the rest of the run."""
    from royalelearn.checkpoint import DirCheckpointStore

    store = DirCheckpointStore(tmp_path, keep=1)
    folders = _checkpoints(store.folder(), (100, 200))

    held = (folders[0] / "weights.bin").open("rb")
    store.prune(keep=1)
    held.close()

    removed = store.prune(keep=1)
    assert [path.name for path in removed] == [folders[0].name]
    assert not folders[0].exists()


# -- the same guards, on every platform --------------------------------------
#
# The tests above create the failure by holding a handle, which only fails on Windows. These
# INJECT it, so the guard is exercised on the runner that certifies as well as on the machine
# where the defect is real. Both halves are worth having: an injected error proves the guard
# runs, and a real one proves the guard is needed.


def test_an_unarchivable_episode_log_is_survived_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import royalelearn.metrics.sinks as sinks

    sink = _sink(tmp_path)
    sink.write_episodes([_episode(0)])  # type: ignore[arg-type]
    sink.write({"run/iteration": 1})

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(32, "the process cannot access the file")

    monkeypatch.setattr(sinks.os, "replace", refuse)
    sink.write_episodes([_episode(1)])  # type: ignore[arg-type]
    sink.write({"run/iteration": 2})
    monkeypatch.undo()

    assert (tmp_path / EPISODES_NAME).exists()
    assert not (tmp_path / EPISODES_NAME).with_suffix(".jsonl.gz").exists()
    assert sink.compaction_failures == 1

    sink.write_episodes([_episode(2)])  # type: ignore[arg-type]
    sink.write({"run/iteration": 3})
    sink.close()
    assert (tmp_path / EPISODES_NAME).with_suffix(".jsonl.gz").exists(), (
        "the archive was never written after the obstacle went away"
    )


def test_an_unprunable_checkpoint_is_survived_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import royalelearn.checkpoint as checkpoint
    from royalelearn.checkpoint import DirCheckpointStore

    store = DirCheckpointStore(tmp_path, keep=1)
    folders = _checkpoints(store.folder(), (100, 200))

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(32, "the process cannot access the file")

    monkeypatch.setattr(checkpoint.shutil, "rmtree", refuse)
    removed = store.prune(keep=1)
    monkeypatch.undo()

    assert removed == []
    assert store.prune_failures == 1
    assert folders[0].exists() and folders[1].exists()

    # And the entry stayed in the index, so the next prune tries it again rather than losing it.
    assert [path.name for path in store.prune(keep=1)] == [folders[0].name]
