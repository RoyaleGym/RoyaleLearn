"""One optimisation pass over one iteration's experience, and what it reports about itself.

``UpdateResult`` is wide on purpose. Most of its fields are diagnostics rather than losses,
because a PPO update fails quietly: a mask disagreement pins the clip fraction, a stale weight
version moves the ratio, a dead critic shows up only as an explained variance that never rises.
Each of those is a number here, logged every iteration, with an alarm behind it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import msgspec

from .checkpoint import Checkpointable
from .schedule import ScheduleState

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor

    from .buffer import ExperienceBuffer
    from .policy import ObsBatch

__all__ = ["ActorLossTerm", "ActorTermInputs", "Update", "UpdateResult", "check_actor_terms"]


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
    #: False on an iteration the actor was frozen (section 19.5): every quantity computed from
    #: the actor's forward in the update was not measured, and its key is left out of the row.
    actor_trained: bool = True
    #: Keys the update adds beyond the core ``ppo/`` group: each extra actor-loss term's own,
    #: named under its extension, and the freeze's, when the run schedules the actor's rate.
    extra: dict[str, float] = {}


class ActorTermInputs(NamedTuple):
    """What an extra actor-loss term is handed, identically in both of the update's actor paths.

    ``log_probs`` and ``mask`` are the actor's masked log-probabilities and mask on the
    minibatch's choice rows, in ``rows`` order, with the graph attached. ``actor`` is the live
    actor, for a term that needs a forward of its own on other rows. ``actor_scale`` is the
    per-batch scale the policy term is multiplied by, and ``weight`` the minibatch's share of
    its batch.
    """

    obs: ObsBatch
    rows: Tensor
    log_probs: Tensor
    mask: Tensor
    actor: Any
    actor_scale: float
    weight: float


class ActorLossTerm(Protocol):
    """A term an extension adds to the actor's loss.

    ``loss`` returns ``(coefficient, raw)``. The update adds ``coefficient * raw * scale``, where
    ``scale`` is ``actor_scale`` for a term whose raw value is a sum over the choice rows
    (``scaling == "rows"``, which keeps minibatch size a pure memory knob and matches the policy
    term under every ``ppo.forced_rows`` value) and the minibatch weight for one whose raw value
    is a mean of its own (``scaling == "minibatch"``). The gradient ratio, when asked for, is
    measured on ``raw * scale``, before the coefficient.

    ``finish`` returns the term's keys for the row; every one must start with ``<extension>/``.
    ``actor_trained`` is False on a frozen iteration, when ``loss`` was not called at all.
    ``state`` is what a checkpoint keeps (an empty dict keeps nothing); ``format_version`` is
    saved beside it and a different one refuses the resume.
    """

    extension: str
    name: str
    format_version: int
    scaling: str
    measure_grad_ratio: bool

    def begin(self, env_steps: int, device: Any) -> None: ...

    def loss(self, inputs: ActorTermInputs, *, epoch: int, measure: bool) -> tuple[float, Any]: ...

    def finish(
        self,
        *,
        iteration: int,
        actor_trained: bool,
        explained_variance: float,
        grad_ratio: float | None,
    ) -> dict[str, float]: ...

    def state(self) -> dict[str, Any]: ...

    def load_state(self, state: Mapping[str, Any]) -> None: ...


class Update(ABC, Checkpointable):
    """The learner's optimisation step. Owns the optimizers, so it is checkpointable."""

    @abstractmethod
    def step(self, buffer: ExperienceBuffer, sched: ScheduleState) -> UpdateResult:
        """Consume one iteration's rectangle and return what happened.

        Implementers may assume the buffer's advantages and returns are already computed and
        that every cell it hands out is trainable and valid. They MUST apply exactly the mask
        read from the buffer, never a recomputed one, and MUST leave the buffer unchanged.
        """


def check_actor_terms(terms: Sequence[ActorLossTerm]) -> None:
    """Refuse a set of terms the update could not hold apart.

    Two terms with one ``(extension, name)`` would share a checkpoint entry and a gradient
    ratio, and a scaling the update does not know would be applied as neither. Called where the
    terms are first assembled, before anything that needs cleaning up exists, and again by the
    update itself.
    """
    from ..metrics.schema import core_groups

    keys = [(term.extension, term.name) for term in terms]
    doubled = sorted({key for key in keys if keys.count(key) > 1})
    if doubled:
        raise ValueError(f"two actor-loss terms share an (extension, name): {doubled}")
    reserved = core_groups()
    for term in terms:
        # A term's keys are held to '<extension>/', so an extension named after a core group
        # could write that group's keys -- ppo/kl, which an alarm halts on.
        if not term.extension or "/" in term.extension or term.extension in reserved:
            raise ValueError(
                f"actor-loss term {term.extension}/{term.name}: {term.extension!r} cannot be an "
                f"extension's name; it is empty, has a '/', or is a core metric group "
                f"({', '.join(sorted(reserved))})"
            )
        if term.scaling not in ("rows", "minibatch"):
            raise ValueError(
                f"actor-loss term {term.extension}/{term.name} declares scaling "
                f"{term.scaling!r}; it is 'rows' or 'minibatch'"
            )
