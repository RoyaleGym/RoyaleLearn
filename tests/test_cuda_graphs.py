"""``rollout.cuda_graphs``: each policy's rollout forward replayed as a CUDA graph.

A graph is a fixed shape, so a forward runs at a bucketed batch size: its rows are copied into
buffers the graph owns and its logits read back out. The padding is real rows, and the network
reads each row alone, so the answers are the eager path's. Without CUDA the same buffers are
called directly, which is how this file runs the path on the processor; the capture itself is
the ``gpu`` test at the end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import config as cfg
from royalelearn.api.rollout import GROUP_LEARNER, GROUP_SCRIPTED
from royalelearn.learn.graphs import ROW_BUCKETS, GraphedForward, bucket
from royalelearn.testing import build_model, coordinator, tiny_config

torch = pytest.importorskip("torch")

from test_buffer import SLOTS  # noqa: E402
from test_inference import (  # noqa: E402,F401
    _OneActorStore,
    _pool_plan,
    inference_for,
    observations,
    rect,
    round_with,
)


def _mixed() -> np.ndarray:
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    groups[1::4] = 0
    groups[2::4] = 1
    return groups


def test_unset_nothing_is_encoded() -> None:
    assert b"cuda_graphs" not in msgspec.json.encode(cfg.RolloutConfig())
    assert cfg.cuda_graphs_of(cfg.RolloutConfig()) is False


def test_a_batch_runs_at_the_smallest_bucket_that_holds_it() -> None:
    assert [bucket(n) for n in (1, 8, 9, 300, 512)] == [8, 8, 16, 512, 512]
    assert bucket(ROW_BUCKETS[-1] + 1) is None


def test_the_graphed_path_answers_as_the_eager_one(rect: Any) -> None:  # noqa: F811
    """Rounds of different sizes, so the padding holds rows earlier rounds left. Plant: read
    the whole bucket back instead of the round's rows, and the shapes stop agreeing; pad with
    rows the network mixes in, and the answers move."""
    model = build_model(rect.spec)
    store = _OneActorStore(build_model(rect.spec).actor)
    eager = inference_for(rect, model=model, snapshots=store)
    graphed = inference_for(rect, model=model, snapshots=store, cuda_graphs=True)
    rounds = [
        (0, _mixed()),
        (1, np.where(np.arange(SLOTS) % 2 == 0, GROUP_LEARNER, GROUP_SCRIPTED).astype(np.int8)),
        (2, _mixed()),
    ]
    for engine in (eager, graphed):
        engine.begin_iteration(_pool_plan(SLOTS, ("snap:a", "snap:b")))
    for cycle, groups in rounds:
        want = eager.act(round_with(groups, cycle=cycle, buffer=rect.buffer))
        got = graphed.act(round_with(groups, cycle=cycle, buffer=rect.buffer))
        assert np.array_equal(got.actions, want.actions)
        assert np.allclose(got.log_probs, want.log_probs, atol=1e-6)
    assert {key[0] for key in graphed._graphs} == {"learner", "frozen"}
    assert not eager._graphs


def test_a_graph_goes_with_its_actor(rect: Any) -> None:  # noqa: F811
    """A graph holds its actor, so one whose actor will not play this iteration is dropped:
    an opponent no longer resident, or the live learner once a behaviour snapshot samples."""
    from royalelearn.learn.actor_critic import BehaviourSnapshot

    model = build_model(rect.spec)
    engine = inference_for(
        rect, model=model, snapshots=_OneActorStore(build_model(rect.spec).actor), cuda_graphs=True
    )
    engine.begin_iteration(_pool_plan(SLOTS, ("snap:a", "snap:b")))
    engine.act(round_with(_mixed(), buffer=rect.buffer))
    assert {key[:2] for key in engine._graphs if key[0] == "frozen"} == {
        ("frozen", "snap:a"),
        ("frozen", "snap:b"),
    }
    engine.begin_iteration(_pool_plan(SLOTS, ("snap:a",)))
    assert ("frozen", "snap:b") not in {key[:2] for key in engine._graphs}
    assert any(key[0] == "learner" for key in engine._graphs)
    engine.begin_iteration(_pool_plan(SLOTS, ("snap:a",)), behaviour=BehaviourSnapshot.of(model))
    assert not any(key[0] == "learner" for key in engine._graphs)


def test_a_run_off_cuda_says_its_forwards_run_as_before(tmp_path: Path) -> None:
    base = tiny_config(tmp_path)
    config = tiny_config(tmp_path, rollout=msgspec.structs.replace(base.rollout, cuda_graphs=True))
    lines: list[str] = []
    with coordinator(config, printer=lines.append) as run:
        assert run.inference.cuda_graphs is False
    assert any(line.startswith("cuda graphs   off") for line in lines)


@pytest.mark.gpu
def test_a_captured_graph_replays_the_forward(env_spec: Any) -> None:
    """On a CUDA device: the replay is the eager forward on new rows, and it follows the
    weights when they change in place, as an optimizer step changes them."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from test_policy_transform import _obs

    actor = build_model(env_spec).actor.to("cuda").eval()
    first = _obs(env_spec, rows=5, seed=1)
    later = _obs(env_spec, rows=7, seed=2)
    first, later = (
        type(batch)(*(None if t is None else t.to("cuda") for t in batch))
        for batch in (first, later)
    )
    graph = GraphedForward(actor.logits, first, 8, keep=actor, capture=True)
    assert graph.graph is not None
    with torch.no_grad():
        for batch in (first, later):
            assert torch.allclose(graph(batch), actor.logits(batch), atol=1e-5)
        for parameter in actor.parameters():
            parameter.mul_(1.01)
        assert torch.allclose(graph(later), actor.logits(later), atol=1e-5)


def test_overlap_keeps_one_behaviour_actor_so_its_graphs_are_not_captured_again(
    rect: Any,  # noqa: F811
) -> None:
    """Each overlapped iteration brings a new behaviour snapshot. Its weights are copied into
    the one held since the first, so the learner's graph is the same graph, replaying the new
    weights. Plant: hold the snapshot handed in, and a new graph is captured every iteration,
    on the collection thread, while the update runs."""
    from royalelearn.learn.actor_critic import BehaviourSnapshot

    model = build_model(rect.spec)
    engine = inference_for(
        rect, model=model, snapshots=_OneActorStore(build_model(rect.spec).actor), cuda_graphs=True
    )
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    seen = []
    for cycle in range(3):
        snapshot = BehaviourSnapshot.of(model)
        engine.begin_iteration(_pool_plan(SLOTS, ("snap:a",)), behaviour=snapshot)
        assert engine.behaviour is not snapshot
        for held, given in zip(
            engine.behaviour.actor.state_dict().values(),
            snapshot.actor.state_dict().values(),
            strict=True,
        ):
            assert torch.equal(held, given)
        want = (
            inference_for(rect, model=model, snapshots=_OneActorStore(model.actor))
            .act(round_with(groups, cycle=cycle, buffer=rect.buffer))
        )
        got = engine.act(round_with(groups, cycle=cycle, buffer=rect.buffer))
        assert np.array_equal(got.actions, want.actions)
        seen.append({key: id(entry) for key, entry in engine._graphs.items()})
        with torch.no_grad():
            for parameter in model.actor.parameters():
                parameter.mul_(1.1)
    assert seen[0] == seen[1] == seen[2]
