"""Reward scaling: the running statistic, and the one thing it must never do.

The statistic is checked against numpy over a hundred thousand samples fed in uneven batches, so
that the batched merge is held to the same answer as a single pass. Everything else here is about
the mean: it is recorded, it is logged, and it is never subtracted.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalelearn.seeding import derive_generator

torch = pytest.importorskip("torch")

import msgspec  # noqa: E402

from royalelearn.learn.returns import (  # noqa: E402
    WelfordReturnScaler,
    WelfordState,
    discounted_returns,
)

SEED = 20260921
SAMPLES = 100_000


def test_a_reward_exactly_on_the_bound_is_not_reported_clipped() -> None:
    """The clip is ``> bound``, and the number reported is the number applied.

    ``advantage/clipped_reward_frac`` is read as a bug report rather than a tolerance -- a
    clipped reward means the scale is wrong -- so a count that fires on a value the clamp left
    alone is the wrong direction to be wrong in.
    """
    scaler = WelfordReturnScaler(standardize=False, clip=10.0)
    on_the_bound = torch.full((2, 2), 10.0)
    scaled, fraction = scaler.scale(on_the_bound)
    assert fraction == 0.0
    assert torch.equal(scaled, on_the_bound)


def test_the_reported_clip_share_is_over_the_cells_it_is_asked_for() -> None:
    """One comparison against the bound, restricted to the cells the statistics describe."""
    scaler = WelfordReturnScaler(standardize=False, clip=10.0)
    rewards = torch.tensor([[11.0, 0.0], [99.0, 99.0]])
    trainable = torch.tensor([[True, True], [False, False]])
    _, fraction = scaler.scale(rewards.clone(), over=trainable)
    assert fraction == pytest.approx(0.5)
    _, whole = scaler.scale(rewards.clone())
    assert whole == pytest.approx(0.75)


def test_the_running_statistic_matches_numpy_over_many_samples() -> None:
    rng = derive_generator(SEED, "test/returns/welford")
    values = rng.normal(loc=7.0, scale=3.0, size=SAMPLES)
    scaler = WelfordReturnScaler()
    start = 0
    for size in rng.integers(1, 5000, size=200):
        chunk = values[start : start + int(size)]
        if chunk.size == 0:
            break
        scaler.update(torch.as_tensor(chunk))
        start += int(size)
    seen = values[:start]
    assert scaler.count == seen.size
    assert scaler.mean == pytest.approx(float(seen.mean()), rel=1e-10)
    assert scaler.variance == pytest.approx(float(seen.var(ddof=1)), rel=1e-8)
    assert scaler.std == pytest.approx(float(seen.std(ddof=1)), rel=1e-8)
    assert scaler.divisor == pytest.approx(scaler.std)


def test_the_statistic_is_the_same_however_the_batches_fall() -> None:
    rng = derive_generator(SEED, "test/returns/batching")
    values = rng.normal(size=10_000)
    whole = WelfordReturnScaler()
    whole.update(values)
    pieces = WelfordReturnScaler()
    for chunk in np.array_split(values, 37):
        pieces.update(chunk)
    assert pieces.count == whole.count
    assert pieces.mean == pytest.approx(whole.mean, rel=1e-10)
    assert pieces.variance == pytest.approx(whole.variance, rel=1e-10)


def test_an_empty_batch_changes_nothing() -> None:
    scaler = WelfordReturnScaler()
    scaler.update(np.array([]))
    assert scaler.count == 0
    assert scaler.divisor == 1.0


def test_the_mean_is_never_subtracted() -> None:
    """A zero-sum reward's sign structure is the whole of what it says. Subtracting a mean from
    it turns a loss into a smaller win."""
    scaler = WelfordReturnScaler(standardize=True, clip=1e9)
    rewards = torch.tensor([[1.0, -1.0], [3.0, -3.0]])
    scaler.update(rewards + 100.0)
    scaled, _ = scaler.scale(rewards)
    assert torch.allclose(scaled, rewards / scaler.divisor)
    assert torch.allclose(scaled + scaled.flip(1), torch.zeros_like(scaled))
    assert scaler.mean != pytest.approx(0.0)


def test_the_clip_is_applied_after_the_scaling_and_is_counted() -> None:
    scaler = WelfordReturnScaler(standardize=False, clip=2.0)
    rewards = torch.tensor([0.0, 1.0, 5.0, -5.0])
    scaled, fraction = scaler.scale(rewards)
    assert torch.equal(scaled, torch.tensor([0.0, 1.0, 2.0, -2.0]))
    assert fraction == pytest.approx(0.5)


def test_standardisation_can_be_turned_off_without_changing_the_reward() -> None:
    scaler = WelfordReturnScaler(standardize=False, clip=1e9)
    scaler.update(np.array([1.0, 5.0, 9.0]))
    rewards = torch.tensor([2.0, -4.0])
    scaled, _ = scaler.scale(rewards)
    assert scaler.divisor == 1.0
    assert torch.equal(scaled, rewards)


def test_the_state_round_trips_through_json() -> None:
    scaler = WelfordReturnScaler()
    scaler.update(np.array([1.0, 2.0, 4.0, 8.0]))
    encoded = msgspec.json.encode(scaler.state())
    restored = WelfordReturnScaler()
    restored.load_state(msgspec.json.decode(encoded, type=WelfordState))
    assert restored.count == scaler.count
    assert restored.mean == pytest.approx(scaler.mean)
    assert restored.divisor == pytest.approx(scaler.divisor)


def test_the_state_round_trips_through_a_checkpoint_folder(tmp_path: object) -> None:
    from pathlib import Path

    folder = Path(str(tmp_path)) / "return_scaler"
    scaler = WelfordReturnScaler()
    scaler.update(np.array([1.0, 2.0, 4.0, 8.0]))
    scaler.save_checkpoint(folder)
    restored = WelfordReturnScaler()
    restored.load_checkpoint(folder, strict=True)
    assert restored.state() == scaler.state()


def test_a_missing_checkpoint_is_a_refusal_when_strict_and_a_default_otherwise(
    tmp_path: object,
) -> None:
    from pathlib import Path

    from royalelearn.errors import CheckpointFormatError

    folder = Path(str(tmp_path)) / "empty"
    folder.mkdir()
    scaler = WelfordReturnScaler()
    with pytest.raises(CheckpointFormatError):
        scaler.load_checkpoint(folder, strict=True)
    scaler.load_checkpoint(folder, strict=False)
    assert scaler.count == 0


# --------------------------------------------------------------------------
# The returns the statistic is taken over
# --------------------------------------------------------------------------


def test_the_discounted_return_stops_at_the_end_of_its_own_episode() -> None:
    rewards = torch.tensor([[0.0], [1.0], [0.0], [10.0]])
    ended = torch.tensor([[False], [True], [False], [False]])
    gamma = 0.5
    out = discounted_returns(rewards, ended, gamma)
    assert out[1, 0].item() == pytest.approx(1.0)
    assert out[0, 0].item() == pytest.approx(gamma * 1.0)
    assert out[3, 0].item() == pytest.approx(10.0)
    assert out[2, 0].item() == pytest.approx(gamma * 10.0)


def test_the_discounted_return_of_an_iteration_with_no_ends_is_the_whole_tail() -> None:
    rewards = torch.ones((5, 2))
    ended = torch.zeros((5, 2), dtype=torch.bool)
    gamma = 0.9
    out = discounted_returns(rewards, ended, gamma)
    expected = sum(gamma**k for k in range(5))
    assert out[0, 0].item() == pytest.approx(expected)
