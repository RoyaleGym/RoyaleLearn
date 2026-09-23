"""The one place in the package where a mask meets a logit.

Masking a categorical policy is three lines and every one of them has a failure mode that is
silent rather than loud, so they are written once, here, and everything else in the harness goes
through this class.

The mask is applied BEFORE the softmax. Masking afterwards is gradient-identical -- a masked
logit receives exactly zero gradient either way -- and numerically it is not: a post-softmax mask
combined with the usual ``clamp(probs, min=1e-11)`` leaves every illegal action sitting at
p = 1e-11 with a finite log-probability, and a sampler draws one eventually. Masking before the
softmax removes them from the normalisation instead, so an illegal action's probability is
exactly zero and no draw can reach it.

The fill value is ``torch.finfo(dtype).min`` and never ``-inf``. An entropy sum contains
``p * log p`` at every index; at a masked index ``p`` is exactly zero, and ``0 * -inf`` is NaN
while ``0 * finfo.min`` is zero. A distribution that survives only because some library clamps
internally is not one whose correctness anybody can point at.

Sampling takes its uniforms from the caller. The rollout path draws an ``(R,)`` vector per cycle
from a named stream and indexes it by slot, so the action a policy takes is a function of the run
seed, the cycle and the slot -- not of how many rows happened to share the forward, and not of
torch's global generator.
"""

from __future__ import annotations

import torch
from torch import Tensor

from royalegym.action import NOOP

from ..api.policy import ActionDistribution

__all__ = ["MaskedCategorical"]


def _noop_violation(mask: Tensor) -> str:
    """What to say when a row arrives with its no-op masked out.

    The environment sets the no-op unconditionally, so a row without it did not come from the
    environment: it was read from somewhere that was never written, or written by something
    other than the codec. Which of those it is shows in whether the row has any legal action
    at all -- a row of zeros is a cell nobody filled, and a row with legal actions but no no-op
    is a corruption of one that was.
    """
    bad = (~mask[:, NOOP]).nonzero(as_tuple=False).flatten().tolist()
    legal = mask.sum(dim=-1)
    empty = [row for row in bad if int(legal[row]) == 0]
    lines = [
        "mask[NOOP] must be True on every row "
        "(royalegym.action.GridActionParser.action_mask sets it unconditionally); "
        f"{len(bad)} of {mask.shape[0]} rows in this batch do not have it",
        f"  rows without the no-op:    {bad[:16]}{' ...' if len(bad) > 16 else ''}",
        f"  of those, entirely empty:  {len(empty)} "
        f"{empty[:16]}{' ...' if len(empty) > 16 else ''}",
    ]
    if empty:
        lines.append(
            "  an entirely empty row is a cell that was never written rather than a bad mask: "
            "look at which slots those rows belong to and whether their worker reported them"
        )
    else:
        lines.append(
            "  every offending row has legal actions but not the no-op, which the environment "
            "cannot produce: suspect the codec or the bit order rather than an unwritten cell"
        )
    return "\n".join(lines)


class MaskedCategorical(ActionDistribution):
    """A categorical over ``(B, A)`` logits with the illegal actions removed.

    ``logits`` must be float32 even when the forward that produced them ran in bf16: the cast is
    what makes the masking and the ``log_softmax`` exact, and exactness there is what keeps a
    filled logit from leaking probability and keeps the entropy sum finite.
    """

    __slots__ = ("_logp", "_mask")

    def __init__(self, logits: Tensor, mask: Tensor) -> None:
        if logits.dtype is not torch.float32:
            raise TypeError(f"logits must be float32, they are {logits.dtype}")
        if mask.dtype is not torch.bool or mask.shape != logits.shape:
            raise TypeError(
                f"mask must be a bool tensor shaped like the logits {tuple(logits.shape)}; "
                f"it is {mask.dtype} {tuple(mask.shape)}"
            )
        # No row can be fully masked, because the action mask sets the no-op unconditionally --
        # including after game over (royalegym.action.GridActionParser.action_mask). A row that
        # violates it produces NaN everywhere downstream, so it is caught at the boundary.
        if not bool(mask[:, NOOP].all()):
            raise AssertionError(_noop_violation(mask))
        self._mask = mask
        self._logp = torch.log_softmax(
            logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1
        )

    @property
    def log_probs(self) -> Tensor:
        """``(B, A)`` float32: the normalised masked log-probabilities themselves."""
        return self._logp

    @property
    def mask(self) -> Tensor:
        """``(B, A)`` bool: the mask this distribution was built with, not a recomputed one."""
        return self._mask

    def sample(self, uniforms: Tensor) -> Tensor:
        """``(B,)`` int64 by inverse CDF of caller-supplied uniforms in [0, 1).

        ``right=True`` is what makes an illegal action unreachable: its CDF interval has zero
        width, and a zero-width interval cannot contain a point under a strict comparison.
        """
        cdf = self._logp.exp().cumsum(-1)
        cdf = cdf / cdf[..., -1:].clamp_min(1e-30)
        u = uniforms.to(device=cdf.device, dtype=cdf.dtype).unsqueeze(-1).clamp(0.0, 1.0 - 1e-7)
        index = torch.searchsorted(cdf.contiguous(), u, right=True)
        return index.squeeze(-1).clamp_(max=cdf.shape[-1] - 1)

    def mode(self) -> Tensor:
        """``(B,)`` int64: the argmax of the MASKED log-probabilities, so it is legal by
        construction. An argmax over an unmasked softmax would emit illegal actions in every
        evaluation game."""
        return self._logp.argmax(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        return self._logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> Tensor:
        """``(B,)`` float32 over the whole legal set.

        The ``where`` drops the masked terms rather than relying on ``0 * finfo.min`` being zero,
        so the sum stays finite whatever a future fill value is.
        """
        p = self._logp.exp()
        return -(p * torch.where(self._mask, self._logp, torch.zeros_like(self._logp))).sum(-1)

    def noop_entropy(self) -> Tensor:
        """``(B,)`` float32: the binary entropy of p(no-op) against p(play anything).

        Bounded by 0.693 nats, which is why the coefficient acting on it is not the joint
        entropy's, and it is the leading indicator of no-op collapse: it falls while the joint
        entropy is still comfortable.
        """
        p = self._logp[:, NOOP].exp().clamp(1e-7, 1 - 1e-7)
        return -(p * p.log() + (1 - p) * (1 - p).log())

    def p_noop(self) -> Tensor:
        """``(B,)`` float32: p(no-op), unclamped.

        Summed over the learner's rollout rows into ``policy/rollout_hold_rate`` and, against
        each row's own uniform baseline, ``policy/rollout_hold_lift``. It said it was behind
        ``policy/cards_per_match`` until 2026-09-22 and was behind nothing: the accumulator
        summed it and no reader read it.
        """
        return self._logp[:, NOOP].exp()

    def logit_std(self) -> Tensor:
        """``(B,)`` float32: the spread of the logits over each row's LEGAL set.

        How far apart the policy is willing to put its options, which is the quantity entropy
        is a saturating function of. Over 250 legal actions the entropy deficit goes as the
        square of this and starts invisibly small: measured on hog26-3, ``entropy_normalised``
        spanned 0.999991 to 0.994957 across a run whose policy became 560 times less uniform,
        so the climb lives in the fifth decimal place of that key and is legible in this one.

        Computed from the normalised log-probabilities rather than from the raw logits, which
        is the same number: ``log_softmax`` subtracts a constant per row and a constant does
        not move a standard deviation. Doing it here means it is over the same masked set the
        entropy is over, rather than over a tensor that still carries the fill value.
        """
        mask = self._mask
        n = mask.sum(-1, keepdim=True).clamp_min(1).to(self._logp.dtype)
        centred = self._logp.masked_fill(~mask, 0.0)
        mean = centred.sum(-1, keepdim=True) / n
        var = (((centred - mean) ** 2) * mask).sum(-1) / n.squeeze(-1)
        return var.clamp_min(0.0).sqrt()

    def n_legal(self) -> Tensor:
        """``(B,)`` int64. Entropy falling is ambiguous without it: a policy that has learnt to
        wait sees fewer legal actions, and its entropy falls for that reason alone."""
        return self._mask.sum(-1)
