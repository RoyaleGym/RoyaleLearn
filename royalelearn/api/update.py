"""One optimisation pass over one iteration's experience, and what it reports about itself.

``UpdateResult`` is wide on purpose. Most of its fields are diagnostics rather than losses,
because a PPO update fails quietly: a mask disagreement pins the clip fraction, a stale weight
version moves the ratio, a dead critic shows up only as an explained variance that never rises.
Each of those is a number here, logged every iteration, with an alarm behind it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import msgspec

from .checkpoint import Checkpointable
from .schedule import ScheduleState

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .buffer import ExperienceBuffer

__all__ = ["Update", "UpdateResult"]


class UpdateResult(msgspec.Struct):
    """What one update did, in the units the metric row uses.

    ``kl_by_epoch`` and ``clip_fraction_by_epoch`` are per epoch rather than averaged: the rule
    for ``n_epochs`` is read off them ("if epoch three's clip fraction is more than twice epoch
    one's, lower it"), and an average cannot answer that. ``samples_unused_frac`` reads zero by
    construction and is logged so that it can be seen to.
    """

    policy_loss: float
    value_loss: float
    entropy: float
    noop_entropy: float
    entropy_normalised: float
    #: Spread of the logits over each row's legal set, averaged over the rows that had a choice.
    #: The un-saturated companion to ``entropy_normalised``, which is within 1e-5 of its maximum
    #: for the first hundred iterations of every run so far.
    logit_std: float
    kl: float
    clip_fraction: float
    dual_clip_fraction: float
    explained_variance: float
    ratio_max_abs_dev: float
    grad_norm_actor: float
    grad_norm_critic: float
    update_magnitude_actor: float
    update_magnitude_critic: float
    kl_by_epoch: list[float]
    clip_fraction_by_epoch: list[float]
    n_minibatches: int
    n_optimizer_steps: int
    n_samples: int
    samples_unused_frac: float
    seconds: float
    #: Seconds inside ``step`` before the epochs begin: the critic's pass over every collected
    #: cell, and the advantage recursion over the rectangle. They were published as a hardcoded
    #: 0.0, which reads as "it cost nothing" rather than "nobody measured it", and they are part
    #: of the update's own time rather than of the residual it is compared against.
    critic_pass_seconds: float = 0.0
    gae_seconds: float = 0.0
    #: Share of each optimizer's parameters whose second moment sits under its own ``eps``, where
    #: Adam stops normalising and the step becomes proportional to the gradient again. ``None``
    #: before an optimizer has stepped: no second moments is not "nothing is on the floor".
    adam_eps_floor_frac_actor: float | None = None
    adam_eps_floor_frac_critic: float | None = None
    #: What the ACTOR was shown, which is the one thing ``ppo.forced_rows`` changes that no
    #: gradient, loss or KL can show: the point of its exact arm is that they do not move.
    #: ``actor_rows`` counts rows over the whole update, epochs included, and ``actor_forwards``
    #: counts the forwards those rows arrived in.
    actor_rows: int = 0
    actor_forwards: int = 0
    #: The share of trainable cells whose mask left one action, from the stored column rather
    #: than from the rollout's own sample of the same quantity.
    forced_frac: float = 0.0
    #: The two quantities above conditioned on the cells that had a choice, so that they mean
    #: the same thing whichever population the arm optimises over.
    policy_loss_choice: float = 0.0
    explained_variance_choice: float = 0.0


class Update(ABC, Checkpointable):
    """The learner's optimisation step. Owns the optimizers, so it is checkpointable."""

    @abstractmethod
    def step(self, buffer: ExperienceBuffer, sched: ScheduleState) -> UpdateResult:
        """Consume one iteration's rectangle and return what happened.

        Implementers may assume the buffer's advantages and returns are already computed and
        that every cell it hands out is trainable and valid. They MUST apply exactly the mask
        read from the buffer, never a recomputed one, and MUST leave the buffer unchanged.
        """
