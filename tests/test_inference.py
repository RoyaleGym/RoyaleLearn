"""The parent's half of a round: which policy answered, in what order, and from which uniforms.

Three properties carry the rollout's reproducibility and each is asserted directly rather than
through a played iteration:

- a seat's action depends on its slot and the cycle, and not on who else happened to be in the
  forward with it, which is what lets the farm's timing vary without the trajectory varying;
- the observation the parent runs the policy on is byte for byte the one the update will be
  shown for the same cell, which is the precondition the PPO ratio invariant rests on;
- one forward per distinct policy, and none at all for the seats the worker fills itself.

Nothing here knows a width. The rectangle is the environment's, the action space is the
environment's, and the network is built from the spec read off it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from royalelearn.api.rollout import (
    GROUP_DEAD,
    GROUP_LEARNER,
    GROUP_SCRIPTED,
    ROLE_MIRROR,
    Assignment,
    EnvSpec,
    SlotPlan,
)
from royalelearn.seeding import ACT_CYCLE, derive_generator, stream_path

torch = pytest.importorskip("torch")

from royalelearn.config import NetConfig  # noqa: E402
from royalelearn.learn.inference import BatchedInference, RectGather  # noqa: E402
from royalelearn.learn.nets import DefaultNetworkFactory  # noqa: E402
from test_buffer import CYCLES, SEED, SLOTS, Fixture, plan_for, round_for  # noqa: E402

#: The smallest network the architecture allows: the card embedding is one side of the pointer
#: head's inner product, so it is the channel count, and the norm groups have to divide it.
ARCH = NetConfig(
    channels=8,
    blocks=1,
    norm_groups=4,
    vector_embed=4,
    value_hidden=16,
    card_embed=8,
    device="cpu",
    autocast_dtype="float32",
)


@pytest.fixture(scope="module")
def observations(mock_env_spec: Any) -> list[dict[str, np.ndarray]]:
    """Real observations to fill the rectangle with, all different, from a played battle."""
    env = mock_env_spec.build_vec(2)
    try:
        batches = [env.reset(seed=11)[0]]
        actions = np.zeros(env.num_envs, dtype=np.int64)
        for _ in range(CYCLES * SLOTS):
            batches.append(env.step(actions)[0])
    finally:
        env.close()
    return [{key: value[0] for key, value in batch.items()} for batch in batches]


@pytest.fixture
def rect(env_spec: EnvSpec, observations: list[dict[str, np.ndarray]]) -> Any:
    """A filled rectangle with an iteration open on it."""
    built = Fixture(env_spec, observations)
    built.fill()
    built.buffer.begin_iteration(plan_for(SLOTS), CYCLES)
    try:
        yield built
    finally:
        built.close()


def build_model(spec: EnvSpec, *, seed: int = SEED) -> Any:
    """The shipped pair at the smallest size, on the CPU and in float32."""
    return DefaultNetworkFactory(seed).build(spec, ARCH, "cpu")


def inference_for(built: Fixture, **kwargs: Any) -> BatchedInference:
    model = kwargs.pop("model", None) or build_model(built.spec)
    engine = BatchedInference(built.buffer, model, master_seed=SEED, **kwargs)
    engine.begin_iteration(plan_for(SLOTS))
    return engine


def round_with(groups: np.ndarray, *, cycle: int = 0, buffer: Any) -> Any:
    """One shard-round covering every slot, with the controller of each seat written in."""
    slots = np.arange(groups.size, dtype=np.int64)
    built = round_for(
        cycle, slots, rows=np.array([buffer.layout.row_index(cycle, int(s)) for s in slots])
    )
    built.group = groups.astype(np.int8)
    return built


# --------------------------------------------------------------------------
# The gather
# --------------------------------------------------------------------------


def test_the_parent_sees_exactly_what_the_update_will_be_shown(rect: Fixture) -> None:
    """The precondition of the ratio invariant: one set of bytes, decoded twice.

    If these two disagreed, the importance ratio would be comparing a policy against itself on
    two different observations, and the whole update would be measuring nothing.
    """
    from test_buffer import collect

    buffer = rect.buffer
    slots = np.arange(SLOTS, dtype=np.int64)
    for cycle in range(CYCLES):
        buffer.record_round(
            round_for(
                cycle,
                slots,
                rows=np.array([buffer.layout.row_index(cycle, int(s)) for s in slots]),
            ),
            actions=np.zeros(SLOTS, dtype=np.int16),
            log_probs=np.zeros(SLOTS, dtype=np.float32),
        )
    gather = RectGather(buffer, rows=CYCLES * SLOTS)
    for batch in collect(buffer, 64, 64, 1):
        for minibatch in batch:
            cells = minibatch.cells.cpu().numpy()
            mine = gather.observations(cells // SLOTS, cells % SLOTS)
            assert torch.equal(mine.spatial, minibatch.obs.spatial)
            assert torch.equal(mine.vector, minibatch.obs.vector)
            assert torch.equal(mine.mask, minibatch.obs.mask)
            assert torch.equal(mine.mask_planes, minibatch.obs.mask_planes)


def test_a_gather_refuses_more_rows_than_it_staged_for(rect: Fixture) -> None:
    gather = RectGather(rect.buffer, rows=2)
    with pytest.raises(ValueError, match="stages 2 rows"):
        gather.observations(np.zeros(SLOTS, dtype=np.int64), np.arange(SLOTS, dtype=np.int64))


# --------------------------------------------------------------------------
# The uniforms
# --------------------------------------------------------------------------


def test_the_uniforms_are_the_named_stream_s(rect: Fixture) -> None:
    engine = inference_for(rect)
    for cycle in (0, 1, CYCLES - 1):
        path = stream_path(ACT_CYCLE, iteration=0, cycle=cycle)
        expected = derive_generator(SEED, path).random(SLOTS, dtype=np.float32)
        assert np.array_equal(engine.uniforms(cycle), expected)


def test_one_cycle_draws_one_vector_however_many_shards_read_it(rect: Fixture) -> None:
    """A cycle is several shard-rounds over disjoint slots, and all of them index one draw."""
    engine = inference_for(rect)
    first = engine.uniforms(3).copy()
    engine.uniforms(4)
    assert np.array_equal(engine.uniforms(3), first)


def test_a_new_iteration_draws_new_uniforms(rect: Fixture) -> None:
    engine = inference_for(rect)
    first = engine.uniforms(0).copy()
    engine.begin_iteration(plan_for(SLOTS, iteration=1))
    assert not np.array_equal(engine.uniforms(0), first)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def test_an_action_does_not_depend_on_who_shared_the_forward(rect: Fixture) -> None:
    """The property the slot-indexed uniforms and the ascending order exist for.

    The same seats, at the same cycle, with half the round taken away: a batch of four instead
    of a batch of eight, so a different kernel shape and a different arrival order, and the same
    actions.
    """
    engine = inference_for(rect)
    everyone = engine.act(round_with(np.full(SLOTS, GROUP_LEARNER), buffer=rect.buffer))
    half = np.full(SLOTS, GROUP_SCRIPTED, dtype=np.int8)
    kept = np.arange(0, SLOTS, 2)
    half[kept] = GROUP_LEARNER
    fewer = engine.act(round_with(half, buffer=rect.buffer))

    assert np.array_equal(fewer.actions[kept], everyone.actions[kept])
    assert np.array_equal(fewer.log_probs[kept], everyone.log_probs[kept])


def test_the_worker_s_own_seats_are_left_alone(rect: Fixture) -> None:
    """A scripted seat is filled by the worker and a dead one has no observation to read."""
    engine = inference_for(rect)
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    groups[1] = GROUP_SCRIPTED
    groups[2] = GROUP_DEAD
    answer = engine.act(round_with(groups, buffer=rect.buffer))

    assert answer.actions[1] == 0
    assert answer.actions[2] == 0
    assert answer.log_probs[1] == 0.0
    assert answer.log_probs[2] == 0.0
    assert engine.drain_stats().forwards == 1


def test_one_forward_per_distinct_policy(rect: Fixture) -> None:
    store = _OneActorStore(build_model(rect.spec).actor)
    engine = inference_for(rect, snapshots=store)
    engine.begin_iteration(_pool_plan(SLOTS, ("snap:a", "snap:b")))
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    groups[1::4] = 0
    groups[2::4] = 1
    engine.act(round_with(groups, buffer=rect.buffer))
    stats = engine.drain_stats()

    assert stats.forwards == 3
    assert stats.rounds == 1
    assert store.asked == ["snap:a", "snap:b"], "in ascending group order, not arrival order"


def test_only_the_learner_s_rows_carry_a_log_probability(rect: Fixture) -> None:
    """A frozen seat's transition is not trained on, and a log-probability from another
    policy's parameters would be a ratio against a denominator nothing optimised."""
    store = _OneActorStore(build_model(rect.spec).actor)
    engine = inference_for(rect, snapshots=store)
    engine.begin_iteration(_pool_plan(SLOTS, ("snap:a",)))
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    frozen = np.arange(1, SLOTS, 2)
    groups[frozen] = 0
    answer = engine.act(round_with(groups, buffer=rect.buffer))

    assert np.all(answer.log_probs[frozen] == 0.0)
    assert np.all(answer.log_probs[::2] != 0.0)


def test_a_frozen_seat_plays_the_snapshot_it_was_assigned(rect: Fixture) -> None:
    """With the learner's own actor in the archive, the frozen seats play exactly what the
    learner would have: the routing is what is under test, not the weights."""
    model = build_model(rect.spec)
    store = _OneActorStore(model.actor)
    engine = inference_for(rect, model=model, snapshots=store)
    engine.begin_iteration(_pool_plan(SLOTS, ("snap:a",)))
    everyone = engine.act(round_with(np.full(SLOTS, GROUP_LEARNER), buffer=rect.buffer))
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    groups[1::2] = 0
    mixed = engine.act(round_with(groups, buffer=rect.buffer))

    assert np.array_equal(mixed.actions, everyone.actions)


def test_a_group_the_plan_does_not_name_is_an_error(rect: Fixture) -> None:
    engine = inference_for(rect, snapshots=_OneActorStore(build_model(rect.spec).actor))
    engine.begin_iteration(_pool_plan(SLOTS, ("snap:a",)))
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    groups[0] = 4
    with pytest.raises(IndexError, match="group 4"):
        engine.act(round_with(groups, buffer=rect.buffer))


def test_a_pool_opponent_without_a_store_says_so(rect: Fixture) -> None:
    engine = inference_for(rect)
    engine.begin_iteration(_pool_plan(SLOTS, ("snap:a",)))
    groups = np.full(SLOTS, GROUP_LEARNER, dtype=np.int8)
    groups[0] = 0
    with pytest.raises(RuntimeError, match="snapshot store"):
        engine.act(round_with(groups, buffer=rect.buffer))


# --------------------------------------------------------------------------
# What is sampled
# --------------------------------------------------------------------------


def test_every_sampled_action_is_legal_under_the_row_s_own_mask(rect: Fixture) -> None:
    engine = inference_for(rect)
    gather = RectGather(rect.buffer, rows=SLOTS)
    slots = np.arange(SLOTS, dtype=np.int64)
    for cycle in range(CYCLES):
        learners = np.full(SLOTS, GROUP_LEARNER)
        answer = engine.act(round_with(learners, cycle=cycle, buffer=rect.buffer))
        obs = gather.observations(np.full(SLOTS, cycle, dtype=np.int64), slots)
        taken = torch.from_numpy(answer.actions).unsqueeze(-1)
        assert bool(obs.mask.gather(-1, taken).all())


def test_the_behaviour_snapshot_is_what_samples_when_one_is_set(rect: Fixture) -> None:
    """With overlapped collection the live actor is being updated while the rollout runs, so
    the rollout samples from a copy frozen at the iteration boundary."""
    from royalelearn.learn.actor_critic import BehaviourSnapshot

    model = build_model(rect.spec)
    engine = inference_for(rect, model=model)
    frozen = BehaviourSnapshot.of(model)
    before = engine.act(round_with(np.full(SLOTS, GROUP_LEARNER), buffer=rect.buffer))

    with torch.no_grad():
        for parameter in model.actor_parameters():
            parameter.add_(1.0)
    engine.begin_iteration(plan_for(SLOTS), behaviour=frozen)
    after = engine.act(round_with(np.full(SLOTS, GROUP_LEARNER), buffer=rect.buffer))

    assert np.array_equal(after.actions, before.actions)
    assert np.allclose(after.log_probs, before.log_probs)


# --------------------------------------------------------------------------
# Support
# --------------------------------------------------------------------------


def _pool_plan(slots: int, resident: tuple[str, ...]) -> SlotPlan:
    """A plan whose battles are the learner against the pool, with ``resident`` on the board."""
    assignments = tuple(
        Assignment(
            battle=slot // 2,
            ordinal=0,
            role=ROLE_MIRROR,
            opponent_id=None,
            group=(GROUP_LEARNER, GROUP_LEARNER),
            learner_seat=slot % 2,
        )
        for slot in range(slots)
    )
    return SlotPlan(
        iteration=0,
        n_battles=slots // 2,
        n_slots=slots,
        assignment=assignments,
        resident_snapshots=resident,
    )


class _OneActorStore:
    """A snapshot archive that hands back one actor and records what was asked for.

    The archive's own reading and writing is ``tests/test_snapshots.py``'s subject. What this
    file needs to see is which id the routing asked for, and in what order.
    """

    def __init__(self, actor: Any) -> None:
        self.actor = actor
        self.asked: list[str] = []

    def get(self, snapshot_id: str, device: Any) -> Any:
        self.asked.append(snapshot_id)
        return self.actor

    def put(self, snapshot_id: str, ac: Any, meta: dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError

    def digest(self, snapshot_id: str) -> str:  # pragma: no cover
        return snapshot_id

    def list(self) -> list[str]:  # pragma: no cover
        return sorted(set(self.asked))


def test_a_row_its_worker_never_wrote_is_not_handed_to_a_policy(rect: Fixture) -> None:
    """A worker that dies mid-round leaves its cells holding whatever was there before.

    The flag that says so is ``valid``. Routing on the plan's intent instead would read those
    cells, and a cell nobody wrote has no legal action in it -- not even the no-op the
    environment sets unconditionally -- so the first symptom would be an assertion inside the
    distribution rather than the worker that stopped.
    """
    engine = inference_for(rect)
    groups = np.full(SLOTS, GROUP_LEARNER)
    built = round_with(groups, cycle=0, buffer=rect.buffer)

    dead = np.zeros(SLOTS, dtype=bool)
    dead[[1, 4]] = True
    built.valid = ~dead
    for slot in np.flatnonzero(dead):
        row = rect.buffer.layout.row_index(0, int(slot))
        rect.buffer.obs_view[row] = 0  # what an unwritten cell looks like

    answer = engine.act(built)

    assert list(answer.actions[dead]) == [0, 0], "a seat that could not be asked plays the no-op"
    assert not answer.log_probs[dead].any(), "an unwritten row carries no log-probability"
    assert answer.actions[~dead].sum() >= 0
