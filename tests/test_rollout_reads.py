"""The rollout forward reads back from the device once, and every number it reports is unchanged.

Until 2026-09-26 a rollout forward made about twenty reads a round, each waiting for the stream:
the uniforms went up by a blocking copy, the distribution read its no-op check back, the policy
statistics read twelve sums and three boolean selections, and the actions and log-probabilities
came home in two more copies. Now the actions, the log-probabilities' bits and the no-op check
come home in one copy, and the statistics wait on the device until the iteration's collection is
drained. These tests hold the numbers to what the old reads gave, bit for bit.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from royalelearn.learn.distribution import MaskedCategorical  # noqa: E402
from royalelearn.learn.inference import BatchedInference, _StatAccumulator  # noqa: E402
from royalelearn.learn.nets import DefaultNetworkFactory  # noqa: E402
from royalelearn.seeding import derive_generator  # noqa: E402
from royalelearn.testing import ARCH, CYCLES, SEED, SLOTS, plan_for, round_for  # noqa: E402
from royalelearn.testing import RectFixture as Fixture  # noqa: E402
from royalelearn.testing import observations as fresh_observations  # noqa: E402


@pytest.fixture(scope="module")
def observations(mock_env_spec: Any) -> list[dict[str, np.ndarray]]:
    """Real observations, all different, to fill the rectangle with."""
    return fresh_observations(mock_env_spec, seed=3, steps=24)

FIELDS = (
    "rows",
    "choice_rows",
    "entropy",
    "p_noop",
    "n_legal",
    "hold",
    "hold_lift",
    "choice_n_legal",
    "gap",
    "gap_sq",
    "legal_log",
    "legal_log_sq",
    "gap_legal_log",
)


def _reference(distributions: list[MaskedCategorical]) -> dict[str, float]:
    """The per-round reads as they were before 2026-09-26: a boolean selection and an ``item``
    for every sum, added into Python floats round by round."""
    out = dict.fromkeys(FIELDS, 0.0)
    out["rows"] = out["choice_rows"] = 0
    for dist in distributions:
        p_noop, n_legal = dist.p_noop(), dist.n_legal()
        out["rows"] += int(n_legal.shape[0])
        out["entropy"] += float(dist.entropy().sum().item())
        out["p_noop"] += float(p_noop.sum().item())
        out["n_legal"] += float(n_legal.sum().item())
        choice = n_legal > 1
        held = p_noop[choice]
        widths = n_legal[choice].to(held.dtype)
        out["choice_rows"] += int(choice.sum().item())
        out["hold"] += float(held.sum().item())
        out["hold_lift"] += float((held * widths).sum().item())
        out["choice_n_legal"] += float(widths.sum().item())
        gap = dist.hold_gap()[choice].to(torch.float64)
        legal_log = widths.to(torch.float64).log()
        out["gap"] += float(gap.sum().item())
        out["gap_sq"] += float((gap * gap).sum().item())
        out["legal_log"] += float(legal_log.sum().item())
        out["legal_log_sq"] += float((legal_log * legal_log).sum().item())
        out["gap_legal_log"] += float((gap * legal_log).sum().item())
    return out


def _rounds(seed: str) -> list[MaskedCategorical]:
    """Rounds of random policies over random masks, a third of the rows forced."""
    rng = derive_generator(SEED, seed)
    rounds = []
    for _ in range(7):
        rows, width = 12, 40
        mask = rng.random((rows, width)) < 0.4
        mask[:, 0] = True
        mask[rng.random(rows) < 0.33, 1:] = False
        logits = torch.from_numpy(rng.normal(size=(rows, width)).astype(np.float32) * 3.0)
        rounds.append(MaskedCategorical(logits, torch.from_numpy(mask)))
    return rounds


@pytest.mark.parametrize("route", ["host_index", "device_mask"])
def test_the_deferred_statistics_are_the_per_round_ones_bit_for_bit(route: str) -> None:
    rounds = _rounds("test/rollout-reads/stats")
    stats = _StatAccumulator()
    for dist in rounds:
        rows = int(dist.n_legal().shape[0])
        if route == "host_index":
            host = np.flatnonzero(dist.mask.numpy().sum(axis=1) > 1)
            stats.policy(dist, rows, torch.from_numpy(host), int(host.size))
        else:
            stats.policy(dist, rows)
    drained = stats.drain()
    want = _reference(rounds)
    for field in FIELDS:
        assert getattr(drained, field) == want[field], field
    assert want["choice_rows"] not in (0, want["rows"]), "the draw must have both kinds of row"


def test_the_host_and_the_device_must_agree_on_the_choice_rows() -> None:
    """The two counts come from the same stored bits by different routes; a disagreement is a
    codec or bit-order fault, and it stops the drain rather than skewing a statistic."""
    dist = _rounds("test/rollout-reads/disagree")[0]
    host = np.flatnonzero(dist.mask.numpy().sum(axis=1) > 1)
    stats = _StatAccumulator()
    stats.policy(dist, 12, torch.from_numpy(host[1:]), int(host.size) - 1)
    with pytest.raises(AssertionError, match="read the mask differently"):
        stats.drain()


def _engine(
    env_spec: Any, observations: Any, device: str, *, forced: tuple[tuple[int, int], ...] = ()
) -> tuple[Fixture, Any, Any]:
    from royalelearn.testing import plant_forced

    built = Fixture(env_spec, observations, device=device)
    built.fill()
    if forced:
        plant_forced(built, forced)
    model = DefaultNetworkFactory(SEED).build(built.spec, ARCH, device)
    engine = BatchedInference(built.buffer, model, master_seed=SEED)
    plan = plan_for(SLOTS)
    built.buffer.begin_iteration(plan, CYCLES)
    engine.begin_iteration(plan)
    return built, model, engine


def _round(built: Fixture, cycle: int) -> Any:
    slots = np.arange(SLOTS, dtype=np.int64)
    rows = np.array([built.buffer.layout.row_index(cycle, int(s)) for s in slots])
    return round_for(cycle, slots, rows=rows)


def test_a_rollout_round_answers_what_the_distribution_would_have_bit_for_bit(
    env_spec: Any, observations: Any
) -> None:
    """The actions and log-probabilities that come home packed are the ones the distribution
    gives when asked directly, with the log-probabilities compared as bit patterns.

    Two seats are forced (only the no-op legal), because a round of nothing but choice rows
    cannot tell a host that counted the right rows from one that counted the wrong ones."""
    built, model, engine = _engine(env_spec, observations, "cpu", forced=((0, 2), (0, 5)))
    try:
        played = _round(built, 0)
        answer = engine.act(played)
        slots = np.asarray(played.slots, dtype=np.int64)
        obs = engine.gather.observations(np.zeros(slots.size, dtype=np.int64), slots)
        with torch.no_grad():
            dist = MaskedCategorical(model.logits(obs).float(), obs.mask)
            actions = dist.sample(torch.from_numpy(engine.uniforms(0)[slots]))
            log_probs = dist.log_prob(actions)
        assert np.array_equal(answer.actions, actions.numpy())
        assert np.array_equal(answer.log_probs.view(np.int32), log_probs.numpy().view(np.int32))
        # The host counted the choice rows from the stored bits; the drain checks that count
        # against the decoded masks', so a host that counted other rows stops here.
        stats = engine.drain_stats()
        want = _reference([dist])
        for field in FIELDS:
            assert getattr(stats, field) == want[field], field
        assert 0 < stats.choice_rows < stats.rows, "the round must hold both kinds of row"
    finally:
        built.close()


def test_a_rollout_row_without_its_noop_is_still_refused(
    env_spec: Any, observations: Any
) -> None:
    """The distribution no longer reads its own check back; the answer rides home with the
    actions, and a live row with the no-op cleared must still stop the round."""
    built, _model, engine = _engine(env_spec, observations, "cpu")
    try:
        row = built.buffer.layout.row_index(0, 3)
        built.buffer.obs_view[row, built.codec.layout.mask_start] &= 0xFE
        with pytest.raises(AssertionError, match=r"mask\[NOOP\]"):
            engine.act(_round(built, 0))
    finally:
        built.close()


@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature")
def test_on_cuda_a_rollout_forward_reads_back_once(env_spec: Any, observations: Any) -> None:
    """Counted with torch's sync debug mode in warn: one synchronising call per forward, the
    copy the actions come home in. The mode does not see every synchronisation, so this is a
    ceiling on what it can see, not a proof that nothing else waits."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine, so there is no stream to wait for")
    built, _model, engine = _engine(env_spec, observations, "cuda")
    try:
        engine.act(_round(built, 0))  # the first builds the ring, the caches and the kernels
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                engine.act(_round(built, 1))
            finally:
                torch.cuda.set_sync_debug_mode("default")
        syncs = [w for w in caught if "called a synchronizing CUDA operation" in str(w.message)]
        assert len(syncs) == 1, [str(w.message) for w in syncs]
    finally:
        built.close()
