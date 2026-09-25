"""The reference-KL regulariser and its adaptive coefficient (sections 19.7-19.8).

For each regulariser, on each choice row it covers, KL(pi_ref || pi_theta): FORWARD, so the policy
is charged for taking probability away from where the reference puts it. It is exact over the
row's masked legal set under ``factor: joint``, and exact over the play/wait pair under
``factor: noop_marginal``.

The PPO update adds ``lambda * sum(KL) * actor_scale`` to the actor's loss in both of its actor
paths -- ``actor_scale`` being the same per-batch scale the policy term is multiplied by -- so the
term's weight against the policy term is one number under every ``ppo.forced_rows`` value, and
minibatch size stays a pure memory knob.

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
    from ..config import CoefSpec, ImitationConfig, ReferenceKLSpec
    from .references import Reference

__all__ = [
    "AdaptiveCoefficient",
    "ImitationTerms",
    "KLParts",
    "ReferenceKL",
    "RowFilter",
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
    """One ``reference_kl`` regulariser: the term, its first-epoch measurement and its lambda."""

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
        self.grad_ratio: float | None = None

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
        self.grad_ratio = None

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

    def finish(self, *, adapt: bool) -> dict[str, float]:
        """This iteration's metrics, and lambda moved for the next one when ``adapt``.

        ``adapt`` is False on a frozen iteration: there was no policy term to anchor, no KL was
        measured, and lambda does not move.
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
        if self.grad_ratio is not None:
            fields[prefix + "grad_ratio"] = self.grad_ratio
        if adapt:
            self.coef.observe(measured, self.kappa, self.env_steps)
        return fields


class ImitationTerms:
    """What the PPO update holds for section 19: the regularisers and the freeze's bookkeeping.

    Its state -- each lambda, and where the last frozen stretch ended -- is update state like the
    backoff's, saved in the update's checkpoint folder and in the state digest.
    """

    def __init__(self, regularisers: Sequence[ReferenceKL], *, scheduled_scale: bool) -> None:
        self.regularisers = list(regularisers)
        #: Whether the block sets ``actor_lr_scale`` at all; the freeze keys exist only then.
        self.scheduled_scale = bool(scheduled_scale)
        #: The first unfrozen iteration after the last frozen stretch, or None.
        self.unfrozen_at: int | None = None
        #: Whether the previous iteration was frozen. None before any iteration ran.
        self.last_frozen: bool | None = None

    @classmethod
    def from_config(
        cls,
        imitation: ImitationConfig | None,
        references: Mapping[str, Reference],
        env: EnvSpec,
    ) -> ImitationTerms | None:
        # Nothing for the update to hold when the block only initialises the actor: the update
        # is then the one a run without the block runs, state digest included.
        if imitation is None or (not imitation.regularisers and imitation.actor_lr_scale is None):
            return None
        regularisers = [
            ReferenceKL(spec, references[spec.reference], env) for spec in imitation.regularisers
        ]
        return cls(regularisers, scheduled_scale=imitation.actor_lr_scale is not None)

    def begin(self, env_steps: int, device: Any) -> None:
        for regulariser in self.regularisers:
            regulariser.begin(env_steps, device)

    def finish(
        self, *, iteration: int, scale: float, explained_variance: float
    ) -> dict[str, float]:
        """Every ``imitation/`` key of this iteration's row."""
        frozen = scale == 0.0
        fields: dict[str, float] = {}
        for regulariser in self.regularisers:
            fields.update(regulariser.finish(adapt=not frozen))
        if self.scheduled_scale:
            fields["imitation/actor_lr_scale"] = float(scale)
            fields["imitation/actor_frozen"] = 1.0 if frozen else 0.0
        if not frozen and self.last_frozen:
            self.unfrozen_at = int(iteration)
            fields["imitation/ev_at_unfreeze"] = float(explained_variance)
        if not frozen and self.unfrozen_at is not None:
            fields["imitation/iterations_since_unfreeze"] = float(iteration - self.unfrozen_at + 1)
        self.last_frozen = frozen
        return fields

    def state(self) -> dict[str, Any]:
        return {
            "lambda": {reg.name: reg.coef.value for reg in self.regularisers},
            "unfrozen_at": self.unfrozen_at,
            "last_frozen": self.last_frozen,
        }

    def load_state(self, state: Mapping[str, Any] | None) -> None:
        """Restore; a checkpoint written without the block restores each ``coef.start``."""
        if not state:
            return
        values = state.get("lambda") or {}
        for reg in self.regularisers:
            if reg.name in values:
                reg.coef.value = float(values[reg.name])
        self.unfrozen_at = state.get("unfrozen_at")
        self.last_frozen = state.get("last_frozen")
