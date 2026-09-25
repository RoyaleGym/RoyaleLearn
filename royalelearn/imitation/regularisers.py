"""The reference-KL regulariser and its adaptive coefficient (sections 19.7-19.8).

For each regulariser, on each choice row it covers, KL(pi_ref || pi_theta): FORWARD, so the policy
is charged for taking probability away from where the reference puts it. It is exact over the
row's masked legal set under ``factor: joint``, and exact over the play/wait pair under
``factor: noop_marginal``.

Each regulariser is an actor-loss term (``api.update.ActorLossTerm``): it hands the update
``(lambda, sum(KL))``, and the update adds ``lambda * sum(KL) * actor_scale`` to the actor's loss
in both of its actor paths -- ``actor_scale`` being the same per-batch scale the policy term is
multiplied by -- so the term's weight against the policy term is one number under every
``ppo.forced_rows`` value, and minibatch size stays a pure memory knob. The gradient ratio the
update measures is on ``sum(KL) * actor_scale``, before lambda.

Everything that is only reported (the chain-rule parts, agreement, the reference's own hold
probability) is computed in the first epoch under ``no_grad``, from tensors the loss already has.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from ..errors import PreflightError
from ..obs_layout import field_slice

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor

    from ..api.policy import ObsBatch
    from ..api.rollout import EnvSpec
    from ..api.update import ActorTermInputs
    from ..config import CoefSpec, ImitationConfig, ReferenceKLSpec
    from .references import Reference

__all__ = [
    "AdaptiveCoefficient",
    "KLParts",
    "ReferenceKL",
    "RowFilter",
    "build_terms",
    "joint_kl",
    "joint_kl_parts",
    "log1mexp",
    "noop_kl",
]


def log1mexp(log_p: Tensor) -> Tensor:
    """``log(1 - exp(log_p))`` for ``log_p <= 0``, without the cancellation of the naive form.

    The two branches are the standard split at log(1/2). ``log_p`` is held a hair below zero,
    as ``MaskedCategorical.hold_gap`` holds it, because at exactly zero the answer is minus
    infinity and a row that certainly holds has no play probability to take a log of.
    """
    import torch

    x = log_p.clamp(max=-1e-7)
    return torch.where(x > -math.log(2.0), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def noop_kl(ref_noop: Tensor, ref_play: Tensor, noop: Tensor, play: Tensor) -> Tensor:
    """``(B,)``: the KL between two play/wait Bernoullis, every argument a log-probability."""
    return ref_noop.exp() * (ref_noop - noop) + ref_play.exp() * (ref_play - play)


def joint_kl(ref: Tensor, policy: Tensor, mask: Tensor) -> Tensor:
    """``(B,)``: sum over legal a of p_ref(a) (log p_ref(a) - log pi(a)).

    Over the legal set only, by ``where`` rather than by trusting that ``0 * fill`` is zero: an
    illegal action's reference probability is exactly zero, and its log-probability is the fill
    value's residue, which is finite today and need not be tomorrow.
    """
    import torch

    terms = ref.exp() * (ref - policy)
    return torch.where(mask, terms, torch.zeros_like(terms)).sum(-1)


class KLParts(NamedTuple):
    """The joint KL split by the chain rule over the action layout: no-op, hand slot, tile."""

    noop: Tensor
    card: Tensor
    tile: Tensor


def joint_kl_parts(
    ref: Tensor, policy: Tensor, mask: Tensor, *, hand_size: int, tiles: int
) -> KLParts:
    """``kl_noop + kl_card + kl_tile == joint_kl``, exactly up to float rounding.

    ``kl_card`` is p_ref(play) times the KL of the slot given a play, and ``kl_tile`` is the
    reference-weighted KL of the tile given the slot. Reporting only; nothing is differentiated.
    """
    import torch

    batch = ref.shape[0]
    grid = mask[:, 1:].reshape(batch, hand_size, tiles)
    ref_tile = ref[:, 1:].reshape(batch, hand_size, tiles)
    pol_tile = policy[:, 1:].reshape(batch, hand_size, tiles)
    minus_inf = torch.full_like(ref_tile, -math.inf)
    ref_slot = torch.logsumexp(torch.where(grid, ref_tile, minus_inf), dim=-1)
    pol_slot = torch.logsumexp(torch.where(grid, pol_tile, minus_inf), dim=-1)
    ref_noop, pol_noop = ref[:, 0], policy[:, 0]
    ref_play, pol_play = log1mexp(ref_noop), log1mexp(pol_noop)
    noop = noop_kl(ref_noop, ref_play, pol_noop, pol_play)
    slot_legal = grid.any(-1)
    card_terms = ref_slot.exp() * ((ref_slot - ref_play[:, None]) - (pol_slot - pol_play[:, None]))
    card = torch.where(slot_legal, card_terms, torch.zeros_like(card_terms)).sum(-1)
    tile_terms = ref_tile.exp() * (
        (ref_tile - ref_slot[..., None]) - (pol_tile - pol_slot[..., None])
    )
    tile = torch.where(grid, tile_terms, torch.zeros_like(tile_terms)).sum((-1, -2))
    return KLParts(noop=noop, card=card, tile=tile)


_OPS = {
    "<": lambda column, value: column < value,
    "<=": lambda column, value: column <= value,
    ">": lambda column, value: column > value,
    ">=": lambda column, value: column >= value,
}


class RowFilter:
    """``exclude_when``: a row is left out when every one of its conditions holds."""

    def __init__(self, conditions: Sequence[Any], spec: EnvSpec, *, what: str) -> None:
        self.columns: list[tuple[int, str, float]] = []
        for condition in conditions:
            part = field_slice(spec, condition.field)
            width = part.stop - part.start
            if not 0 <= condition.index < width:
                raise PreflightError(
                    f"{what}: exclude_when index {condition.index} is outside field "
                    f"{condition.field!r}, which is {width} wide"
                )
            self.columns.append((part.start + condition.index, condition.op, condition.value))

    def keep(self, vector: Tensor) -> Tensor:
        """``(B,)`` float32: 1 for a row the regulariser covers, 0 for one it leaves out."""
        import torch

        if not self.columns:
            return torch.ones(vector.shape[0], dtype=torch.float32, device=vector.device)
        excluded = torch.ones(vector.shape[0], dtype=torch.bool, device=vector.device)
        for column, op, value in self.columns:
            excluded &= _OPS[op](vector[:, column].float(), value)
        return (~excluded).to(torch.float32)


class AdaptiveCoefficient:
    """Section 19.8: lambda, moved once an iteration by what epoch 1 measured."""

    def __init__(self, spec: CoefSpec) -> None:
        from ..learn.schedules import build_schedule

        self.spec = spec
        self.floor = build_schedule(spec.min)
        #: The state a checkpoint carries.
        self.value = float(spec.start)

    def clamp(self, value: float, env_steps: int) -> float:
        return min(max(value, self.floor.value(env_steps)), self.spec.max)

    def current(self, env_steps: int) -> float:
        """The coefficient an iteration at this clock uses."""
        return self.clamp(self.value, env_steps)

    def observe(self, measured: float | None, budget: float, env_steps: int) -> float:
        """Move and store lambda from one iteration's measured KL. No measurement, no move."""
        value = self.current(env_steps)
        if measured is not None:
            if measured > self.spec.band * budget:
                value *= self.spec.up
            elif measured < budget / self.spec.band:
                value /= self.spec.down
        self.value = self.clamp(value, env_steps)
        return self.value


class ReferenceKL:
    """One ``reference_kl`` regulariser: the term, its first-epoch measurement and its lambda.

    An actor-loss term of the ``imitation`` extension. Its state is lambda, kept in the update's
    checkpoint folder and in the state digest like the backoff's.
    """

    extension = "imitation"
    format_version = 1
    #: ``sum(KL)`` over the choice rows, so the update scales it by ``actor_scale``.
    scaling = "rows"
    measure_grad_ratio = True

    def __init__(self, spec: ReferenceKLSpec, reference: Reference, env: EnvSpec) -> None:
        from ..learn.schedules import build_schedule

        what = f"imitation.regularisers[{spec.name!r}]"
        if spec.factor == "joint" and reference.kind != "snapshot":
            raise PreflightError(f"{what}: factor 'joint' needs a snapshot reference")
        self.name = spec.name
        self.factor = spec.factor
        self.reference = reference
        self.filter = RowFilter(spec.exclude_when, env, what=what)
        self.budget = build_schedule(spec.budget)
        self.coef = AdaptiveCoefficient(spec.coef)
        self.hand_size = int(env.hand_size)
        self.tiles = (int(env.n_actions) - 1) // max(1, self.hand_size)
        self.lam = self.coef.current(0)
        self.kappa = self.budget.value(0)
        self._sums: dict[str, Tensor] = {}
        self._choice: Tensor | None = None

    # -- one iteration -------------------------------------------------------

    def begin(self, env_steps: int, device: Any) -> None:
        import torch

        self.env_steps = int(env_steps)
        self.lam = self.coef.current(env_steps)
        self.kappa = self.budget.value(env_steps)
        names = ("kl", "rows", "noop", "top1", "ref_p_noop")
        if self.factor == "joint":
            names += ("card", "tile")
        self._sums = {name: torch.zeros((), dtype=torch.float32, device=device) for name in names}
        self._choice = torch.zeros((), dtype=torch.float32, device=device)

    def loss(self, inputs: ActorTermInputs, *, epoch: int, measure: bool) -> tuple[float, Tensor]:
        """``(lambda, sum(KL))`` on the minibatch's choice rows."""
        return self.lam, self.kl_sum(
            inputs.obs, inputs.rows, inputs.log_probs, inputs.mask, measure=measure
        )

    def kl_sum(
        self,
        obs: ObsBatch,
        rows: Tensor,
        policy: Tensor,
        mask: Tensor,
        *,
        measure: bool,
    ) -> Tensor:
        """The sum of KL over the covered rows among ``rows``, differentiable in ``policy``.

        ``rows`` indexes ``obs`` and are the choice rows the actor's distribution covers, in the
        same order as ``policy``'s rows. ``measure`` adds them to this iteration's epoch-1 means.
        Rows are left out by a zero weight rather than by indexing, because an index needs its
        size on the host, and reading it would put a synchronisation inside the minibatch loop.
        """
        import torch

        from royalegym.action import NOOP

        from ..api.policy import ObsBatch as _Obs

        sub = _Obs(*(None if t is None else t.index_select(0, rows) for t in obs))
        keep = self.filter.keep(sub.vector)
        ref: Tensor | None = None
        if self.factor == "joint":
            ref = self.reference.log_probs(sub)  # type: ignore[union-attr]
            kl = joint_kl(ref, policy, mask)
            ref_noop = ref[:, NOOP]
            ref_play = log1mexp(ref_noop)
        else:
            ref_noop, ref_play = self.reference.noop_log_probs(sub)
            pol_noop = policy[:, NOOP]
            kl = noop_kl(ref_noop, ref_play, pol_noop, log1mexp(pol_noop))
        if measure:
            with torch.no_grad():
                self._measure(kl.detach(), keep, policy.detach(), mask, ref, ref_noop, ref_play)
        return (kl * keep).sum()

    def _measure(
        self,
        kl: Tensor,
        keep: Tensor,
        policy: Tensor,
        mask: Tensor,
        ref: Tensor | None,
        ref_noop: Tensor,
        ref_play: Tensor,
    ) -> None:
        from royalegym.action import NOOP

        sums = self._sums
        assert self._choice is not None
        self._choice += float(kl.shape[0])
        sums["kl"] += (kl * keep).sum()
        sums["rows"] += keep.sum()
        pol_noop = policy[:, NOOP]
        if ref is not None:
            parts = joint_kl_parts(ref, policy, mask, hand_size=self.hand_size, tiles=self.tiles)
            sums["noop"] += (parts.noop * keep).sum()
            sums["card"] += (parts.card * keep).sum()
            sums["tile"] += (parts.tile * keep).sum()
            agree = (ref.argmax(-1) == policy.argmax(-1)).float()
        else:
            pol_play = log1mexp(pol_noop)
            sums["noop"] += (noop_kl(ref_noop, ref_play, pol_noop, pol_play) * keep).sum()
            agree = ((ref_play > ref_noop) == (pol_play > pol_noop)).float()
        sums["top1"] += (agree * keep).sum()
        sums["ref_p_noop"] += (ref_noop.exp() * keep).sum()

    def finish(
        self,
        *,
        iteration: int,
        actor_trained: bool,
        explained_variance: float,
        grad_ratio: float | None,
    ) -> dict[str, float]:
        """This iteration's metrics, and lambda moved for the next one when the actor trained.

        On a frozen iteration there was no policy term to anchor, no KL was measured, and lambda
        does not move.
        """
        prefix = f"imitation/{self.name}/"
        fields: dict[str, float] = {
            prefix + "lambda": float(self.lam),
            prefix + "lambda_at_max": 1.0 if self.lam >= self.coef.spec.max else 0.0,
            prefix + "budget": float(self.kappa),
        }
        rows = float(self._sums["rows"].item()) if self._sums else 0.0
        choice = float(self._choice.item()) if self._choice is not None else 0.0
        measured: float | None = None
        if rows > 0:
            measured = float(self._sums["kl"].item()) / rows
            fields[prefix + "kl"] = measured
            for name in ("noop", "card", "tile"):
                if name in self._sums:
                    fields[prefix + f"kl_{name}"] = float(self._sums[name].item()) / rows
            fields[prefix + "top1_agree"] = float(self._sums["top1"].item()) / rows
            fields[prefix + "ref_p_noop"] = float(self._sums["ref_p_noop"].item()) / rows
        if choice > 0:
            fields[prefix + "rows_frac"] = rows / choice
        if grad_ratio is not None:
            fields[prefix + "grad_ratio"] = grad_ratio
        if actor_trained:
            self.coef.observe(measured, self.kappa, self.env_steps)
        return fields

    def state(self) -> dict[str, Any]:
        return {"lambda": self.coef.value}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.coef.value = float(state["lambda"])


def build_terms(
    imitation: ImitationConfig | None,
    references: Mapping[str, Reference],
    env: EnvSpec,
) -> tuple[ReferenceKL, ...]:
    """One term per configured regulariser, in config order; none without the block."""
    if imitation is None:
        return ()
    return tuple(
        ReferenceKL(spec, references[spec.reference], env) for spec in imitation.regularisers
    )
