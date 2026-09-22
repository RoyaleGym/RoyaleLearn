"""A checkpoint is only worth having if a resume from it is the run continuing.

So what is asserted here is not that the files exist: it is that a manifest catches a changed
byte, that a missing component is a refusal under ``strict`` and a printed path without it,
that a crash between the write and the rename leaves the previous checkpoint intact, and that
pruning survives the stray files a real run directory accumulates.
"""

from __future__ import annotations

import time
from pathlib import Path

import msgspec
import pytest

from royalelearn.api.checkpoint import Manifest
from royalelearn.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    MANIFEST_NAME,
    DirCheckpointStore,
    RngComponent,
    check_described,
    check_resume,
)
from royalelearn.config import LadderConfig
from royalelearn.errors import CheckpointFormatError, IdentityMismatch
from royalelearn.identity import EngineBuild, RunIdentity
from royalelearn.ladder.matchmaker import MixMatchmaker
from royalelearn.ladder.pool import LadderPool
from royalelearn.ladder.rating import BradleyTerryDavidsonRater, EloReadout
from royalelearn.ladder.results import GameResult, ResultLog
from royalelearn.metrics.sinks import JsonlSink

IDENTITY = RunIdentity(
    format_version=1,
    royalelearn_version="0.1.0",
    royalelearn_git="unknown",
    royalegym_version="0.1.0",
    royalegym_git="unknown",
    engine_build=EngineBuild(
        engine_class="royalegym.mock_engine.MockEngine",
        calibration_digest="c" * 8,
        build_digest="b" * 8,
        catalogue_sha256="a" * 8,
        path_search=None,
        stale_build_differences=[],
    ),
    env_spec_digest="e" * 8,
    obs_digest="o" * 8,
    action_digest="d" * 8,
    frame_stack=1,
    arch_digest="r" * 8,
    codec_version=1,
    codec_table_digest="t" * 8,
    algo_digest="g" * 8,
    rollout_digest="l" * 8,
    ladder_digest="p" * 8,
    master_seed=7,
    determinism_tier="run_exact",
    torch_version="none",
    device_kind="cpu",
)


class Counter:
    """A component with one number, for the parts of a checkpoint the test owns."""

    FORMAT_VERSION = 3

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "counter.json").write_bytes(msgspec.json.encode({"value": self.value}))

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        path = folder / "counter.json"
        if not path.exists():
            if strict:
                raise CheckpointFormatError(f"no counter at {path}")
            print(f"no counter at {path}; continuing from zero")
            self.value = 0
            return
        self.value = int(msgspec.json.decode(path.read_bytes())["value"])


def _manifest(*, steps: int, iteration: int) -> Manifest:
    return Manifest(
        format_version=CHECKPOINT_FORMAT_VERSION,
        run_id="0123456789abcdef",
        run_name="test",
        identity=IDENTITY,
        config={"run_name": "test"},
        config_hash="f" * 16,
        iteration=iteration,
        cumulative_env_steps=steps,
        cumulative_timesteps=steps * 3 // 4,
        cumulative_model_updates=iteration * 8,
        wall_seconds=1.5,
        created_unix_ns=time.time_ns(),
        state_digest="s" * 16,
        component_versions={},
        files={},
    )


def _components(tmp_path, value: int = 1) -> dict:
    pool = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    pool.add("snap:v0", step=100, meta={"cycle": False})
    pool.promote("snap:v0")
    pool.record(
        [
            GameResult(
                a="learner",
                b="snap:v0",
                score_a=1.0,
                seed_index=index,
                side_a="blue",
                context="ctx",
                kind="eval",
                run_id="run",
                iteration=0,
                wall="",
            )
            for index in range(40)
        ]
    )
    rater = BradleyTerryDavidsonRater()
    rater.fit(pool.eval_view())
    elo = EloReadout()
    elo.update("learner", "snap:v0", 1.0)
    matchmaker = MixMatchmaker(7, LadderConfig())
    return {
        "counter": Counter(value),
        "ladder": pool,
        "ratings": rater,
        "elo": elo,
        "matchmaker": matchmaker,
        "rng": RngComponent(master_seed=7, eval_seed_set_sha="abc"),
    }


def test_every_component_round_trips(tmp_path) -> None:
    store = DirCheckpointStore(tmp_path)
    written = _components(tmp_path, value=11)
    path = store.write(written, _manifest(steps=2_000_000, iteration=4))
    assert path.name == "000002000000"

    loaded = _components(tmp_path / "other", value=0)
    manifest = store.read(path, loaded, strict=True)
    assert manifest.iteration == 4
    assert manifest.component_versions["counter"] == Counter.FORMAT_VERSION
    assert loaded["counter"].value == 11
    assert loaded["ladder"].champion == "snap:v0"
    assert loaded["ladder"].state.sampled == written["ladder"].state.sampled
    assert loaded["ratings"].table == written["ratings"].table
    assert loaded["elo"].ratings() == written["elo"].ratings()
    assert loaded["rng"].eval_seed_set_sha == "abc"
    assert manifest.files  # every file that was written is in the manifest
    assert all(len(digest) == 64 for digest in manifest.files.values())


def test_a_changed_byte_is_caught_by_the_manifest(tmp_path) -> None:
    store = DirCheckpointStore(tmp_path)
    path = store.write(_components(tmp_path), _manifest(steps=1000, iteration=1))
    target = path / "counter" / "counter.json"
    target.write_bytes(target.read_bytes().replace(b"1", b"2"))
    with pytest.raises(CheckpointFormatError) as excinfo:
        store.read(path, _components(tmp_path / "b"), strict=True)
    assert "counter.json" in str(excinfo.value)


def test_a_missing_component_raises_under_strict_and_prints_without_it(
    tmp_path, capsys
) -> None:
    store = DirCheckpointStore(tmp_path)
    path = store.write({"counter": Counter(5)}, _manifest(steps=1000, iteration=1))
    manifest = msgspec.json.decode((path / MANIFEST_NAME).read_bytes(), type=Manifest)
    manifest.files = {}
    (path / MANIFEST_NAME).write_bytes(msgspec.json.encode(manifest))
    (path / "counter" / "counter.json").unlink()

    with pytest.raises(CheckpointFormatError):
        store.read(path, {"counter": Counter(1)}, strict=True)
    tolerant = Counter(1)
    store.read(path, {"counter": tolerant}, strict=False)
    assert str(path / "counter" / "counter.json") in capsys.readouterr().out
    assert tolerant.value == 0


def test_a_missing_folder_is_named_under_strict(tmp_path) -> None:
    store = DirCheckpointStore(tmp_path)
    path = store.write({"counter": Counter(5)}, _manifest(steps=1000, iteration=1))
    with pytest.raises(CheckpointFormatError, match="'elo'"):
        store.read(path, {"counter": Counter(), "elo": EloReadout()}, strict=True)


def test_a_newer_format_version_is_refused_by_name(tmp_path) -> None:
    store = DirCheckpointStore(tmp_path)
    path = store.write({"counter": Counter(5)}, _manifest(steps=1000, iteration=1))
    raw = msgspec.json.decode((path / MANIFEST_NAME).read_bytes())
    raw["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    (path / MANIFEST_NAME).write_bytes(msgspec.json.encode(raw))
    with pytest.raises(CheckpointFormatError) as excinfo:
        store.read(path, {"counter": Counter()}, strict=True)
    assert str(CHECKPOINT_FORMAT_VERSION + 1) in str(excinfo.value)


def test_a_crash_before_the_rename_leaves_the_last_checkpoint_intact(
    tmp_path, monkeypatch
) -> None:
    store = DirCheckpointStore(tmp_path)
    good = store.write({"counter": Counter(1)}, _manifest(steps=1000, iteration=1))

    def explode(*args, **kwargs):
        raise OSError("the machine went away")

    monkeypatch.setattr("royalelearn.checkpoint.os.replace", explode)
    with pytest.raises(OSError, match="went away"):
        store.write({"counter": Counter(2)}, _manifest(steps=2000, iteration=2))
    monkeypatch.undo()

    assert store.latest() == good
    assert store.read(good, {"counter": (counter := Counter())}, strict=True)
    assert counter.value == 1
    # The half-written one is obviously named and is cleared by the next prune.
    partials = list((tmp_path / "checkpoints").glob("*.partial"))
    assert [path.name for path in partials] == ["000000002000.partial"]
    assert partials[0] in store.prune(keep=5)


def test_latest_and_prune_survive_a_stray_file(tmp_path) -> None:
    store = DirCheckpointStore(tmp_path, keep=2)
    paths = [
        store.write({"counter": Counter(step)}, _manifest(steps=step * 1000, iteration=step))
        for step in range(1, 5)
    ]
    (tmp_path / "checkpoints" / "notes.txt").write_text("left by a human", encoding="utf-8")
    assert store.latest() == paths[-1]
    removed = store.prune()
    assert set(removed) == {paths[0], paths[1]}
    assert not paths[0].exists()
    assert store.latest() == paths[-1]
    assert (tmp_path / "checkpoints" / "notes.txt").exists()
    assert len([path for path in (tmp_path / "checkpoints").iterdir() if path.is_dir()]) == 2


def test_latest_reads_the_manifests_when_the_index_is_gone(tmp_path) -> None:
    store = DirCheckpointStore(tmp_path)
    store.write({"counter": Counter(1)}, _manifest(steps=1000, iteration=1))
    newest = store.write({"counter": Counter(2)}, _manifest(steps=9000, iteration=2))
    (tmp_path / "checkpoints" / "index.json").unlink()
    (tmp_path / "checkpoints" / "stray").mkdir()
    assert store.latest() == newest


def test_an_empty_run_directory_has_no_latest(tmp_path) -> None:
    assert DirCheckpointStore(tmp_path).latest() is None
    assert DirCheckpointStore(tmp_path).prune() == []


def test_nothing_written_carries_a_pickle(tmp_path) -> None:
    """A byte scan, because the rule is about what a load can be made to do, not about intent."""
    store = DirCheckpointStore(tmp_path)
    sink = JsonlSink()
    sink.open(identity=IDENTITY, config_json='{"run_name": "test"}', run_dir=tmp_path / "run")
    sink.write({"run/iteration": 1})
    sink.close()
    path = store.write(
        {**_components(tmp_path), "metrics": sink}, _manifest(steps=1000, iteration=1)
    )
    for file in path.rglob("*"):
        if file.is_file():
            blob = file.read_bytes()
            assert b"\x80\x04" not in blob[:2], file
            assert b"__reduce__" not in blob, file


def test_a_resume_into_another_identity_is_refused_by_field(tmp_path) -> None:
    manifest = _manifest(steps=1000, iteration=1)
    moved = msgspec.structs.replace(IDENTITY, master_seed=8, frame_stack=2)
    with pytest.raises(IdentityMismatch) as excinfo:
        check_resume(manifest, moved, manifest.config)
    assert excinfo.value.differences.keys() == {"master_seed", "frame_stack"}
    assert "master_seed" in str(excinfo.value)
    # Drift is allowed only when it is asked for, and then it is recorded rather than refused.
    assert check_resume(manifest, moved, manifest.config, allow_drift=True) == {}


def _metric_file(path: Path, rows: list[dict]) -> Path:
    path.write_bytes(b"".join(msgspec.json.encode(row) + b"\n" for row in rows))
    return path


def test_a_checkpoint_no_metric_row_describes_is_refused(tmp_path) -> None:
    """The learner a checkpoint holds is one its run's record describes, or it is not resumed.

    Keyed on the iteration the manifest names, and satisfied by ANY row of that iteration:
    a run resumed from an earlier checkpoint writes that iteration's row a second time, and
    either copy describes a learner that really existed. What is refused is a digest none of
    them carries -- which is what a checkpoint of weights trained past their row looks like.
    """
    rows = _metric_file(
        tmp_path / "metrics.jsonl",
        [
            {"run/iteration": 1, "run/state_digest": "a" * 64},
            {"run/iteration": 2, "run/state_digest": "b" * 64},
        ],
    )
    trained_past = msgspec.structs.replace(
        _manifest(steps=2000, iteration=2), state_digest="c" * 64
    )
    with pytest.raises(CheckpointFormatError) as excinfo:
        check_described(trained_past, rows)
    assert "c" * 16 in str(excinfo.value)
    assert "b" * 16 in str(excinfo.value)
    check_described(msgspec.structs.replace(trained_past, state_digest="b" * 64), rows)

    # The same iteration twice, from a resume that replayed it: either copy is a description.
    with rows.open("ab") as handle:
        handle.write(msgspec.json.encode({"run/iteration": 2, "run/state_digest": "c" * 64}))
        handle.write(b"\n")
    check_described(trained_past, rows)


def test_a_checkpoint_with_nothing_to_compare_against_is_let_through(tmp_path, capsys) -> None:
    """Absence is not a mismatch: no row of that iteration, no metric file, a torn last line.

    Iteration zero has no row by construction, and a crash mid-write leaves the last line of
    the file cut short; neither says anything about the learner, so neither refuses it. It is
    said, though, past iteration zero: two smoke runs on disk hold an iteration-5 checkpoint
    beside a metric file that ends at row 4, and a resume from one should not look checked.
    """
    check_described(_manifest(steps=0, iteration=0), tmp_path / "missing.jsonl")
    assert capsys.readouterr().out == ""

    manifest = _manifest(steps=2000, iteration=2)
    check_described(manifest, tmp_path / "missing.jsonl")
    rows = _metric_file(tmp_path / "metrics.jsonl", [{"run/iteration": 1, "run/state_digest": "a"}])
    check_described(manifest, rows)
    with rows.open("ab") as handle:
        handle.write(b'{"run/iteration": 2, "run/state_dig')
    check_described(manifest, rows)
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 3
    assert all("iteration 2" in line and "not checked" in line for line in printed), printed


def test_every_config_difference_is_printed_and_the_run_continues(tmp_path, capsys) -> None:
    manifest = _manifest(steps=1000, iteration=1)
    manifest.config = {"run_name": "test", "checkpoint": {"keep": 10, "strict_load": True}}
    now = {"run_name": "test", "checkpoint": {"keep": 3, "strict_load": True}, "runs_dir": "r"}
    differences = check_resume(manifest, IDENTITY, now)
    assert differences == {"checkpoint.keep": (10, 3), "runs_dir": (None, "r")}
    printed = capsys.readouterr().out
    assert "checkpoint.keep" in printed
    assert "runs_dir" in printed
