"""The masked categorical: what it must never do, checked on the ways it could.

Three of these are regressions against mistakes that are easy to make and silent to have made --
probability leaking onto an illegal action, NaN in the entropy sum, and an argmax taken over an
unmasked softmax. The fourth is the property the whole determinism story rests on: an action is a
function of the uniform it was given, and of nothing else.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

LN2 = float(np.log(2.0))


def random_mask(torch: Any, batch: int, width: int, generator: Any, p: float = 0.3) -> Any:
    """A mask with the no-op always set, as the environment guarantees."""
    mask = torch.rand(batch, width, generator=generator) < p
    mask[:, 0] = True
    return mask


def test_illegal_actions_have_exactly_zero_probability(torch: Any) -> None:
    from royalelearn.learn.distribution import MaskedCategorical

    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(16, 128, generator=generator) * 5.0
    mask = random_mask(torch, 16, 128, generator)
    distribution = MaskedCategorical(logits, mask)
    probabilities = distribution.log_probs.exp()
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))
    assert torch.allclose(probabilities.sum(-1), torch.ones(16), atol=1e-6)
    # An illegal action's log-probability is not merely small: exponentiating it gives zero, so
    # no sampler and no importance ratio can reach it.
    assert bool((distribution.log_probs[~mask] < torch.finfo(torch.float32).min / 2).all())


def test_a_masked_logit_cannot_be_rescued_by_being_enormous(torch: Any) -> None:
    from royalelearn.learn.distribution import MaskedCategorical

    logits = torch.zeros(1, 8)
    logits[0, 5] = 1e30
    mask = torch.ones(1, 8, dtype=torch.bool)
    mask[0, 5] = False
    distribution = MaskedCategorical(logits, mask)
    assert float(distribution.log_probs[0, 5].exp()) == 0.0
    assert int(distribution.mode()) != 5
    assert torch.isfinite(distribution.entropy()).all()


def test_entropy_is_finite_when_only_the_noop_is_legal(torch: Any) -> None:
    """The game-over shape: ``action_mask`` returns the no-op alone once the battle is decided,
    and every row of that iteration goes through here."""
    from royalelearn.learn.distribution import MaskedCategorical

    generator = torch.Generator().manual_seed(2)
    logits = torch.randn(32, 64, generator=generator)
    mask = torch.zeros(32, 64, dtype=torch.bool)
    mask[:, 0] = True
    distribution = MaskedCategorical(logits, mask)
    assert torch.allclose(distribution.entropy(), torch.zeros(32), atol=0.0)
    assert torch.allclose(distribution.p_noop(), torch.ones(32))
    assert torch.equal(distribution.mode(), torch.zeros(32, dtype=torch.int64))
    assert torch.equal(
        distribution.sample(torch.rand(32, generator=generator)),
        torch.zeros(32, dtype=torch.int64),
    )
    assert torch.equal(distribution.n_legal(), torch.ones(32, dtype=torch.int64))


def test_mode_respects_the_mask(torch: Any) -> None:
    from royalelearn.learn.distribution import MaskedCategorical

    generator = torch.Generator().manual_seed(3)
    for _ in range(50):
        logits = torch.randn(8, 256, generator=generator) * 3.0
        mask = random_mask(torch, 8, 256, generator, p=0.1)
        chosen = MaskedCategorical(logits, mask).mode()
        assert bool(mask.gather(-1, chosen.unsqueeze(-1)).all())


def test_sampling_matches_a_brute_force_inverse_cdf(torch: Any, env_spec: Any) -> None:
    """Against an independent implementation in float64, on a narrow legal set.

    Narrow so that the cumulative intervals are wide compared with the difference between a
    float32 running sum and a float64 one: what is being checked is the search and the clamping,
    not the last bit of a cumulative sum.
    """
    from royalelearn.learn.distribution import MaskedCategorical

    generator = torch.Generator().manual_seed(4)
    batch, legal = 1024, 16
    width = env_spec.n_actions
    logits = torch.randn(batch, width, generator=generator) * 2.0
    mask = torch.zeros(batch, width, dtype=torch.bool)
    columns = torch.stack(
        [torch.randperm(width, generator=generator)[:legal] for _ in range(batch)]
    )
    mask.scatter_(1, columns, True)
    mask[:, 0] = True
    distribution = MaskedCategorical(logits, mask)
    uniforms = torch.rand(batch, generator=generator)

    probabilities = distribution.log_probs.exp().double().numpy()
    cdf = probabilities.cumsum(-1)
    cdf /= cdf[:, -1:]
    clamped = np.clip(uniforms.numpy().astype(np.float64), 0.0, 1.0 - 1e-7)
    expected = np.array(
        [np.searchsorted(cdf[row], clamped[row], side="right") for row in range(batch)]
    )
    drawn = distribution.sample(uniforms)
    assert np.array_equal(drawn.numpy(), expected)
    assert bool(mask.gather(-1, drawn.unsqueeze(-1)).all())


def test_sampling_never_leaves_the_legal_set_over_ten_thousand_masks(
    torch: Any, env_spec: Any
) -> None:
    """Ten thousand rows of every legal-set size from one to the whole board.

    Nothing here may be NaN and nothing may be illegal. A fully masked row is the one way both
    could happen, and the environment makes it impossible by setting the no-op unconditionally.
    """
    from royalelearn.learn.distribution import MaskedCategorical

    generator = torch.Generator().manual_seed(5)
    chunk = 500
    width = env_spec.n_actions
    drawn_noop = 0
    for step in range(20):
        density = (step + 1) / 20.0
        logits = torch.randn(chunk, width, generator=generator) * 4.0
        mask = random_mask(torch, chunk, width, generator, p=density)
        distribution = MaskedCategorical(logits, mask)
        entropy = distribution.entropy()
        noop_entropy = distribution.noop_entropy()
        assert torch.isfinite(entropy).all()
        assert bool((entropy >= 0.0).all())
        assert torch.isfinite(noop_entropy).all()
        assert bool(((noop_entropy >= 0.0) & (noop_entropy <= LN2 + 1e-6)).all())
        actions = distribution.sample(torch.rand(chunk, generator=generator))
        assert bool(mask.gather(-1, actions.unsqueeze(-1)).all())
        assert torch.isfinite(distribution.log_prob(actions)).all()
        drawn_noop += int((actions == 0).sum())
    assert drawn_noop < 20 * chunk, "every draw was the no-op; the sampler is not sampling"


def test_entropy_matches_the_definition_over_the_legal_set(torch: Any) -> None:
    from royalelearn.learn.distribution import MaskedCategorical

    generator = torch.Generator().manual_seed(6)
    logits = torch.randn(64, 512, generator=generator)
    mask = random_mask(torch, 64, 512, generator, p=0.2)
    distribution = MaskedCategorical(logits, mask)
    reference = torch.full_like(logits, float("-inf"))
    reference[mask] = logits[mask]
    log_probabilities = torch.log_softmax(reference.double(), dim=-1)
    probabilities = log_probabilities.exp()
    expected = -(probabilities * torch.where(mask, log_probabilities, 0.0)).sum(-1)
    assert torch.allclose(distribution.entropy().double(), expected, atol=1e-6)


def test_noop_entropy_is_the_binary_entropy_of_playing_at_all(torch: Any) -> None:
    from royalelearn.learn.distribution import MaskedCategorical

    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(64, 256, generator=generator) * 2.0
    mask = random_mask(torch, 64, 256, generator)
    distribution = MaskedCategorical(logits, mask)
    p = distribution.p_noop().double()
    expected = -(p * p.log() + (1 - p) * (1 - p).log())
    assert torch.allclose(distribution.noop_entropy().double(), expected, atol=1e-5)


def test_a_fully_masked_row_is_refused(torch: Any) -> None:
    from royalelearn.learn.distribution import MaskedCategorical

    logits = torch.zeros(4, 16)
    mask = torch.ones(4, 16, dtype=torch.bool)
    mask[2, 0] = False
    with pytest.raises(AssertionError, match=r"mask\[NOOP\]"):
        MaskedCategorical(logits, mask)


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16", "float64"])
def test_logits_that_are_not_float32_are_refused(torch: Any, dtype_name: str) -> None:
    """Masking in bf16 is the bug this catches: ``finfo(bfloat16).min`` and a ``log_softmax``
    over it are not exact, and the leak is small enough to look like noise."""
    from royalelearn.learn.distribution import MaskedCategorical

    logits = torch.zeros(2, 8, dtype=getattr(torch, dtype_name))
    mask = torch.ones(2, 8, dtype=torch.bool)
    with pytest.raises(TypeError, match="float32"):
        MaskedCategorical(logits, mask)


def test_a_mask_of_the_wrong_kind_is_refused(torch: Any) -> None:
    from royalelearn.learn.distribution import MaskedCategorical

    logits = torch.zeros(2, 8)
    with pytest.raises(TypeError, match="bool"):
        MaskedCategorical(logits, torch.ones(2, 8, dtype=torch.int8))
    with pytest.raises(TypeError, match="bool"):
        MaskedCategorical(logits, torch.ones(2, 9, dtype=torch.bool))


def test_the_environments_own_masks_always_admit_the_noop(mock_env_spec: Any) -> None:
    """The guarantee the assertion above relies on, taken from a running environment.

    RoyaleGym sets the no-op bit unconditionally, so no observation the harness can ever receive
    produces a fully masked row. This is a few hundred real masks; the exhaustive statement about
    the action space lives with the rest of the environment contract.

    Its own environment, because this one is stepped: a session-wide env that another module
    reads its state off would not be where that module left it.
    """
    env = mock_env_spec.build_vec(2)
    try:
        observations, _ = env.reset(seed=17)
        noop = np.zeros(env.num_envs, dtype=np.int64)
        for _ in range(40):
            assert bool(observations["action_mask"][:, 0].all())
            observations = env.step(noop)[0]
    finally:
        env.close()
