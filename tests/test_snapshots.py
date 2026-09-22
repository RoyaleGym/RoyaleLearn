"""The archive: content addresses, a refusal by name, an LRU that never thrashes, and no pickle.

A snapshot is loaded months after it was written, by a run that may have changed underneath it.
So the interesting tests are the refusals: a snapshot whose architecture, observation or codec
table is not this run's is rejected with the field named rather than shape-errored halfway
through a load, and a pairing of two policies that saw different observations is refused before
a battle is played.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn.errors import IdentityMismatch, PreflightError
from royalelearn.ladder import snapshots as snapshots_module
from royalelearn.ladder.evaluate import EvalRunner, eval_seed_set
from royalelearn.ladder.snapshots import (
    COMPATIBILITY_FIELDS,
    SPEC_NAME,
    WEIGHTS_NAME,
    DiskSnapshotStore,
    SnapshotSpec,
    check_compatible,
)

TEMPLATE = SnapshotSpec(
    snapshot_id="",
    arch_digest="arch-1",
    obs_digest="obs-1",
    action_digest="action-1",
    codec_version=1,
    codec_table_digest="codec-1",
    frame_stack=2,
    num_cards=16,
    vector_size=229,
    context="ctx",
    run_id="run",
)


class FakeTensor:
    """Stands in for a tensor where the test is about the store and not about the weights."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def contiguous(self) -> FakeTensor:
        return self

    def to(self, device: Any) -> FakeTensor:
        return self


class FakeActorCritic:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def actor_state_dict_fp16(self) -> dict[str, FakeTensor]:
        return {"trunk.weight": FakeTensor(self.values)}


class FakeActor:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.evaluated = False

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state = state

    def eval(self) -> None:
        self.evaluated = True


@pytest.fixture
def json_tensors(monkeypatch):
    """Serialise the stand-in tensors as JSON.

    The real codec is safetensors and is exercised by the round trip below, which needs torch;
    everything else in this file is about the folder, the index and the refusals.
    """
    monkeypatch.setattr(
        snapshots_module,
        "_encode_tensors",
        lambda state: msgspec.json.encode({k: v.values for k, v in state.items()}),
    )
    monkeypatch.setattr(
        snapshots_module,
        "_decode_tensors",
        lambda payload, device: msgspec.json.decode(payload),
    )


def _store(tmp_path, **kwargs) -> DiskSnapshotStore:
    settings: dict[str, Any] = {"template": TEMPLATE, "max_resident": 2}
    settings.update(kwargs)
    return DiskSnapshotStore(tmp_path / "snapshots", **settings)


def test_a_snapshot_is_addressed_by_its_own_content(tmp_path, json_tensors) -> None:
    store = _store(tmp_path)
    digest = store.put("snap:v1", FakeActorCritic([1.0, 2.0]), {"step": 4_000_000})
    assert len(digest) == 16
    folder = tmp_path / "snapshots" / digest
    assert (folder / WEIGHTS_NAME).exists()
    assert store.list() == ["snap:v1"]
    assert store.digest("snap:v1") == digest
    spec = store.spec("snap:v1")
    assert spec.snapshot_id == "snap:v1"
    assert spec.step == 4_000_000
    assert spec.arch_digest == TEMPLATE.arch_digest
    assert spec.obs_digest == TEMPLATE.obs_digest

    # The same weights under another name are the same folder; different weights are not.
    assert store.put("snap:v1-again", FakeActorCritic([1.0, 2.0]), {"step": 1}) == digest
    assert store.put("snap:v2", FakeActorCritic([3.0]), {"step": 2}) != digest
    assert store.list() == ["snap:v1", "snap:v1-again", "snap:v2"]


def test_the_index_survives_a_new_process(tmp_path, json_tensors) -> None:
    digest = _store(tmp_path).put("snap:v1", FakeActorCritic([1.0]), {"step": 1})
    assert _store(tmp_path).digest("snap:v1") == digest
    with pytest.raises(KeyError, match="snap:missing"):
        _store(tmp_path).spec("snap:missing")


def test_a_snapshot_the_run_cannot_read_is_refused_by_field() -> None:
    for field in COMPATIBILITY_FIELDS:
        stored = msgspec.structs.replace(TEMPLATE, **{field: "something-else"})
        with pytest.raises(IdentityMismatch) as excinfo:
            check_compatible(stored, TEMPLATE)
        assert field in str(excinfo.value)
        assert "something-else" in str(excinfo.value)
    check_compatible(TEMPLATE, TEMPLATE)
    # Every field that moved is named at once, not one error message at a time.
    moved = msgspec.structs.replace(TEMPLATE, arch_digest="x", obs_digest="y")
    with pytest.raises(IdentityMismatch) as excinfo:
        check_compatible(moved, TEMPLATE)
    assert excinfo.value.differences.keys() == {"arch_digest", "obs_digest"}


def test_loading_an_incompatible_snapshot_refuses_before_it_builds(
    tmp_path, json_tensors
) -> None:
    written = _store(tmp_path)
    written.put("snap:v1", FakeActorCritic([1.0]), {"step": 1})
    folder = tmp_path / "snapshots" / written.digest("snap:v1")
    spec = msgspec.json.decode((folder / SPEC_NAME).read_bytes(), type=SnapshotSpec)
    (folder / SPEC_NAME).write_bytes(
        msgspec.json.encode(msgspec.structs.replace(spec, obs_digest="obs-2"))
    )
    built: list[FakeActor] = []

    def build(device: Any) -> FakeActor:
        actor = FakeActor()
        built.append(actor)
        return actor

    store = _store(tmp_path, build=build)
    with pytest.raises(IdentityMismatch, match="obs_digest"):
        store.get("snap:v1", "cpu")
    assert built == []


def test_the_cache_keeps_the_residents_and_evicts_the_rest(tmp_path, json_tensors) -> None:
    store = _store(tmp_path, build=lambda device: FakeActor(), max_resident=2)
    for index in range(3):
        store.put(f"snap:v{index}", FakeActorCritic([float(index)]), {"step": index})
    first = store.get("snap:v0", "cpu")
    assert first.evaluated
    assert first.state == {"trunk.weight": [0.0]}
    assert store.get("snap:v0", "cpu") is first  # the same module, not a second build
    store.get("snap:v1", "cpu")
    assert set(store.resident) == {"snap:v0", "snap:v1"}
    store.get("snap:v2", "cpu")
    assert set(store.resident) == {"snap:v1", "snap:v2"}
    assert store.get("snap:v0", "cpu") is not first


def test_a_store_without_a_builder_says_so(tmp_path, json_tensors) -> None:
    store = _store(tmp_path)
    store.put("snap:v1", FakeActorCritic([1.0]), {"step": 1})
    with pytest.raises(RuntimeError, match="not load them"):
        store.get("snap:v1", "cpu")


def test_an_evaluation_pairing_across_two_observations_is_refused(
    tmp_path, json_tensors
) -> None:
    store = _store(tmp_path)
    store.put("snap:v1", FakeActorCritic([1.0]), {"step": 1})
    other = msgspec.structs.replace(TEMPLATE, obs_digest="obs-2")
    _store(tmp_path, template=other).put("snap:v2", FakeActorCritic([2.0]), {"step": 2})

    runner = EvalRunner(
        player=None,
        seeds=eval_seed_set(1, 8),
        master_seed=1,
        obs_digest=_store(tmp_path).obs_digest,
    )
    with pytest.raises(PreflightError) as excinfo:
        runner.compare("snap:v1", "snap:v2", games=4)
    message = str(excinfo.value)
    assert "snap:v1" in message and "snap:v2" in message
    assert "obs-1" in message and "obs-2" in message


def test_nothing_written_is_a_pickle(tmp_path, json_tensors) -> None:
    store = _store(tmp_path)
    store.put("snap:v1", FakeActorCritic([1.0, 2.0]), {"step": 1, "cycle": True})
    for path in (tmp_path / "snapshots").rglob("*"):
        if path.is_file():
            blob = path.read_bytes()
            assert not blob.startswith(b"\x80")
            assert b"__reduce__" not in blob


def test_the_weights_round_trip_through_safetensors(tmp_path) -> None:
    """The real codec, on the real tensors, with no stand-in anywhere in the path."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    class Linear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(4, 3)

        def actor_state_dict_fp16(self) -> dict[str, Any]:
            return {name: value.half() for name, value in self.state_dict().items()}

    written = Linear()
    store = _store(tmp_path, build=lambda device: Linear().to(device))
    store.put("snap:v1", written, {"step": 7})
    loaded = store.get("snap:v1", torch.device("cpu"))
    for name, value in written.actor_state_dict_fp16().items():
        assert torch.equal(loaded.state_dict()[name].half(), value)
    for path in (tmp_path / "snapshots").rglob("*"):
        if path.is_file():
            assert not path.read_bytes().startswith(b"\x80")


def test_a_store_can_be_opened_read_only_on_an_archive(tmp_path, json_tensors) -> None:
    _store(tmp_path).put("snap:v1", FakeActorCritic([1.0]), {"step": 1})
    reopened = DiskSnapshotStore(tmp_path / "snapshots")
    assert reopened.list() == ["snap:v1"]
    assert reopened.obs_digest("snap:v1") == TEMPLATE.obs_digest
    assert Path(reopened.root, reopened.digest("snap:v1"), SPEC_NAME).exists()
