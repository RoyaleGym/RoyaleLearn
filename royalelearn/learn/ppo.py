"""One optimisation pass over one iteration's experience.

The order of an update is not free, and most of what is written here is that order. The critic
runs over the whole rectangle once, in chunks, before anything else, so that every advantage is
computed against one set of values rather than against values that moved while they were being
used. GAE runs on those values. The advantages are standardised **once per iteration**, over
every trainable cell, and never per minibatch: a per-minibatch standardisation makes the
gradient depend on how the samples were partitioned, which is exactly the property
``minibatch_size`` is supposed not to have.

MINIBATCH SIZE IS A PURE MEMORY KNOB, and this file is where that is true or not. Gradients
accumulate over the minibatches of a batch weighted by ``n / batch_size``, and one
``optimizer.step()`` is taken per batch, so the optimisation is identical whatever the
minibatch size is. ``tests/test_ppo.py`` holds it to that by comparing the accumulated gradient
against the one-batch gradient. On a 4 GB device that is not a nicety, it is the mechanism that
makes the run fit.

THE LOSS, for one minibatch of ``n`` samples in a batch of ``N``::

    r_i       = exp( log pi_new(a_i | s_i) - log pi_old(a_i | s_i) )
    surr_i    = min( r_i * A_i, clip(r_i, 1 - eps, 1 + eps) * A_i )
    L_i       = surr_i                      if A_i >= 0
                max( surr_i, c * A_i )      if A_i <  0
    L_policy  = -(1/n) sum_i L_i                      * (n/N)
    L_value   = vf_coef * (1/n) sum_i (V_i - G_i)^2   * (n/N)
    L_entropy = -( ent_coef * mean(H) + ent_coef_noop * mean(H2) ) * (n/N)

The dual clip is the second half of the ratio bound. For a positive advantage the standard
minimum already bounds the loss; for a negative one it does not, and in a masked space this
wide a rarely-sampled action's ratio can be enormous. ``max(surr, c*A)`` puts a floor under it.

The actor and the critic have disjoint parameters, so summing the three terms and calling
``backward()`` once is safe: the entropy term's gradient into the critic's parameters is
exactly zero because there is no path between them.

Every gradient norm is recorded BEFORE clipping -- a post-clip norm is the clip bound and says
nothing -- and the two parameter sets are clipped separately, so a large critic gradient cannot
shrink the policy's step.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, parameters_to_vector

from ..api.update import Update, UpdateResult
from ..config import PPOConfig
from ..errors import CheckpointFormatError
from ..seeding import PPO_MINIBATCH, derive_generator, stream_path
from .inference import RectGather

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch.nn import Parameter
    from torch.optim import Optimizer

    from ..api.advantage import AdvantageEstimator, AdvantageStats
    from ..api.policy import ActorCritic, BackpropResult
    from ..api.schedule import ScheduleState
    from .buffer import Minibatch, RectBuffer
    from .schedules import LrBackoff

__all__ = [
    "PPOUpdate",
    "approx_kl",
    "chunked_critic_pass",
    "clipped_fraction",
    "dual_clipped_fraction",
    "explained_variance",
    "standardise",
    "surrogate",
    "value_error",
]

#: What ``ppo.ratio_atol`` calls each precision -- torch's own spelling, so that one vocabulary
#: names the dtype in ``net.autocast_dtype`` and the tolerance beside it.
PRECISION_NAMES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}

#: The three causes a ratio deviation has, in the order they are worth checking. Every one of
#: them moves a log-probability by a quantity of order one, which is why a tolerance of a
#: percent still detects all three.
RATIO_CAUSES = (
    "a mask mismatch between the rollout and the update: a masked-out action's log-probability "
    "is finfo.min, so its ratio is zero or astronomical",
    "a codec or codec-table mismatch: the update is being shown a different observation from "
    "the one the policy acted on",
    "a weight-version mismatch: the parameters that acted are not the parameters being "
    "updated, which moves log-probabilities by the size of a whole update",
)


# --------------------------------------------------------------------------
# The loss, and the numbers read off it
# --------------------------------------------------------------------------


def surrogate(
    ratio: Tensor, advantages: Tensor, *, clip_range: float, dual_clip_c: float
) -> tuple[Tensor, Tensor]:
    """The clipped surrogate and the dual-clipped one, elementwise.

    Returned as a pair because the difference between them is a diagnostic: where they differ is
    exactly where the lower bound bound something, and that is ``dual_clip_fraction``.
    """
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    surr = torch.min(ratio * advantages, clipped * advantages)
    dual = torch.where(advantages < 0, torch.max(surr, dual_clip_c * advantages), surr)
    return surr, dual


def approx_kl(ratio: Tensor) -> Tensor:
    """Schulman's k3 estimator of ``KL(pi_old || pi_new)``, elementwise.

    ``(r - 1) - log r`` rather than ``-log r``: it is non-negative for every ratio and has far
    lower variance, so a threshold set on it means the same thing from iteration to iteration.
    The naive estimator is signed, and a mean of it crossing a threshold says as much about the
    batch's noise as about the policy.
    """
    log_ratio = ratio.log()
    return (ratio - 1.0) - log_ratio


def value_error(
    values: Tensor, old_values: Tensor, returns: Tensor, *, clip_range: float | None
) -> Tensor:
    """The critic's squared error, optionally held near the value the batch was collected under.

    Without a clip range this is the plain mean squared error and the critic is free to move as
    far as one update asks. With one, the prediction is also measured after being pulled back to
    within ``clip_range`` of the old value, and the LARGER of the two errors is the one optimised.
    That is a pessimistic bound rather than a cap: the gradient of a step that overshoots comes
    from the clipped branch, so the critic is not rewarded for a large move it cannot justify on
    the far side of the clip.

    ``ppo.value_clipping`` has been a configuration field with nothing reading it since the
    learner landed, so a config could ask for this and silently not get it. It is off by default
    and it stays off: the published evidence for it is thin, and this repo's own critic clips on
    its gradient norm every step already (``ppo/grad_norm_critic`` sits at 21-35 against a
    ``max_grad_norm`` of 0.5), so a second brake on the same wheel is a thing to measure rather
    than to assume. The field now does what it says, which is the point.
    """
    if clip_range is None:
        return F.mse_loss(values, returns)
    pulled_back = old_values + (values - old_values).clamp(-clip_range, clip_range)
    return torch.max((values - returns).square(), (pulled_back - returns).square()).mean()


def clipped_fraction(ratio: Tensor, clip_range: float) -> Tensor:
    """Elementwise: was this sample's ratio outside the clip band."""
    return ((ratio - 1.0).abs() > clip_range).to(ratio.dtype)


def dual_clipped_fraction(
    advantages: Tensor, surr: Tensor, *, dual_clip_c: float
) -> Tensor:
    """Elementwise: did the lower bound bind. It can only bind where the advantage is negative."""
    return ((advantages < 0) & (dual_clip_c * advantages > surr)).to(surr.dtype)


def explained_variance(returns: Tensor, values: Tensor) -> float:
    """``1 - Var(G - V) / Var(G)``: how much of the return the critic accounts for.

    Zero is a critic no better than predicting the mean, and negative is one that is worse.
    Guarded below, because a batch whose returns have no variance has nothing to explain.
    """
    if returns.numel() < 2:
        return 0.0
    residual = (returns - values).var(unbiased=False)
    total = returns.var(unbiased=False)
    return float(1.0 - residual / total.clamp_min(1e-8))


def standardise(advantages: Tensor, mask: Tensor) -> Tensor:
    """Centre and scale the advantages by the statistics of the cells that reach the update.

    Over the trainable cells and not over the whole rectangle: an opponent's rows are in the
    rectangle and are not trained on, and letting them set the scale would make the gradient
    depend on the iteration's opponent mix.
    """
    selected = advantages[mask]
    if selected.numel() < 2:
        return advantages
    return (advantages - selected.mean()) / (selected.std() + 1e-8)


# --------------------------------------------------------------------------
# The critic pass
# --------------------------------------------------------------------------


def chunked_critic_pass(
    buffer: RectBuffer, model: ActorCritic, gather: RectGather, *, chunk: int
) -> Tensor:
    """``V`` over every cell of the rectangle, including the bootstrap cycle.

    In chunks because the rectangle is a hundred thousand rows wide at the shipped geometry and
    one forward over all of them is a multi-gigabyte activation spike on a device that has four.
    The value of a cell no worker ever wrote is zero rather than whatever the network makes of a
    row of zero bytes: a dead worker's rows are flagged ended and carry no reward, and a zero
    there is what lets the GAE recursion run over them inertly instead of needing a case.
    """
    cycles, slots = buffer.cycles, buffer.n_slots
    values = torch.zeros((cycles + 1, slots), dtype=torch.float32, device=buffer.device)
    flat = values.view(-1)
    total = (cycles + 1) * slots
    step = max(1, int(chunk))
    for start in range(0, total, step):
        cells = np.arange(start, min(start + step, total), dtype=np.int64)
        obs = gather.observations(cells // slots, cells % slots)
        with torch.no_grad():
            flat[start : start + cells.size] = model.value(obs).float()
    return values * _row_validity(buffer)


def _row_validity(buffer: RectBuffer) -> Tensor:
    """``(T+1, R)`` float32: which cells a worker actually wrote.

    The bootstrap cycle takes the last collected cycle's validity. It is read only by the cells
    below it, and a slot whose worker died before the last cycle has those flagged ended.
    """
    collected = buffer.valid[: buffer.cycles]
    rows = np.concatenate([collected, collected[-1:]], axis=0) if buffer.cycles else collected
    return torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32)).to(buffer.device)


# --------------------------------------------------------------------------
# The update
# --------------------------------------------------------------------------


class PPOUpdate(Update):
    """The shipped update: a critic pass, GAE, and the epochs over the rectangle.

    Owns the two optimizers and therefore its own checkpoint folder. The actor and the critic
    get one each: their gradients are clipped separately, their learning rates move together
    under the backoff but are configured apart, and a single optimizer over both sets would make
    ``vf_coef`` and the critic's learning rate the same knob.
    """

    FORMAT_VERSION = 1

    #: What the checkpoint's folder holds.
    ACTOR_FILE = "actor_adam.pt"
    CRITIC_FILE = "critic_adam.pt"
    STATE_FILE = "misc.json"

    def __init__(
        self,
        model: ActorCritic,
        gae: AdvantageEstimator,
        config: PPOConfig,
        *,
        master_seed: int,
        backoff: LrBackoff | None = None,
        device: torch.device | str | None = None,
        optimizer_factory: Callable[..., Optimizer] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        #: Said once per epoch while the update runs, because nothing else is said during it.
        self.progress = progress
        self.model = model
        self.gae = gae
        self.config = config
        self.master_seed = int(master_seed)
        self.backoff = backoff
        self.device = torch.device(device) if device is not None else _model_device(model)
        self.actor_params: list[Parameter] = list(model.actor_parameters())
        self.critic_params: list[Parameter] = list(model.critic_parameters())
        factory = optimizer_factory if optimizer_factory is not None else _adam
        # Kept beside the optimizers because a resume restores them OVER a loaded state dict:
        # without that, changing a learning rate for the resumed half of a run does nothing.
        self.actor_kwargs: dict[str, Any] = {"lr": config.lr_actor, "eps": config.adam_eps}
        self.critic_kwargs: dict[str, Any] = {"lr": config.lr_critic, "eps": config.adam_eps}
        self.actor_optimizer = factory(self.actor_params, **self.actor_kwargs)
        self.critic_optimizer = factory(self.critic_params, **self.critic_kwargs)
        self.optimizers: tuple[Optimizer, Optimizer] = (
            self.actor_optimizer,
            self.critic_optimizer,
        )
        #: Optimizer steps over the whole run; the manifest reports it.
        self.model_updates = 0
        #: The advantage spread before standardisation, which is the thing standardisation
        #: hides and the thing a collapsing policy shows first.
        self.advantage_std_pre_norm = 0.0
        #: What the estimator saw last: the raw return statistics, the divisor it applied and
        #: how much of the batch the reward clip touched. Kept here rather than folded into
        #: ``UpdateResult`` because it describes the rewards, not the optimisation.
        self.advantage_stats: AdvantageStats | None = None
        self._gather: RectGather | None = None
        self._n_slots = 1
        self._final_cells: list[np.ndarray] = []
        self._final_rows: list[np.ndarray] = []

    # -- what the collection hands over -------------------------------------

    def record_final_observations(
        self, cycle: int, slots: np.ndarray, rows: np.ndarray
    ) -> None:
        """The packed observation a truncated episode actually ended on.

        Not in the rectangle, and not reconstructible from it: what the rectangle holds at a
        truncated cell is the first observation of the NEXT episode, because the environment
        resets on the step that ends one. The caller collects these per round from the rollout
        source and hands them over; a truncated cell for which nothing arrives -- a dead
        worker's, whose bootstrap is the value of the cycle it did publish -- gets its final
        value set directly instead.
        """
        slots = np.asarray(slots, dtype=np.int64)
        if slots.size == 0:
            return
        cells = np.stack([np.full(slots.size, int(cycle), dtype=np.int64), slots], axis=1)
        self._final_cells.append(cells)
        self._final_rows.append(np.ascontiguousarray(rows, dtype=np.uint8))

    def _take_final_observations(self) -> tuple[np.ndarray, np.ndarray]:
        cells = (
            np.concatenate(self._final_cells)
            if self._final_cells
            else np.zeros((0, 2), dtype=np.int64)
        )
        rows = np.concatenate(self._final_rows) if self._final_rows else np.zeros((0, 0), np.uint8)
        self._final_cells = []
        self._final_rows = []
        return cells, rows

    # -- the step ------------------------------------------------------------

    def step(self, buffer: RectBuffer, sched: ScheduleState) -> UpdateResult:
        """One iteration's whole update, and everything it reports about itself."""
        started = time.perf_counter()
        config = self.config
        self._n_slots = buffer.n_slots
        self._apply_learning_rates(sched)
        gather = self._gather_for(buffer)

        values = chunked_critic_pass(buffer, self.model, gather, chunk=config.critic_chunk)
        buffer.set_values(values)
        cells, rows = self._take_final_observations()
        if cells.size:
            buffer.set_final_values(
                cells, self.critic_on_final_obs(buffer, gather, cells, rows)
            )

        inputs = buffer.advantage_inputs()
        advantages, returns, self.advantage_stats = self.gae.compute(
            rewards=inputs.rewards,
            values=inputs.values,
            final_values=inputs.final_values,
            terminated=inputs.terminated,
            truncated=inputs.truncated,
            trainable=inputs.trainable,
            gamma=sched.gamma,
            lam=sched.gae_lambda,
        )
        mask = buffer.trainable_mask() & buffer.valid_mask()
        selected = advantages[mask]
        self.advantage_std_pre_norm = (
            float(selected.std(unbiased=True).item()) if selected.numel() > 1 else 0.0
        )
        if config.advantage_standardization:
            advantages = standardise(advantages, mask)
        buffer.set_advantages(advantages, returns)

        before_actor = parameters_to_vector(self.actor_params).detach().clone()
        before_critic = parameters_to_vector(self.critic_params).detach().clone()

        n_samples = int(mask.sum().item())
        # The buffer's own rule, imported rather than restated. This used to be a math.ceil of
        # the same division, which was right while an epoch ended in a remainder batch and wrong
        # the moment it stopped: with the rows over spread across whole batches there are FEWER
        # batches than the ceiling, so the labels ran ahead and the last epoch was never named.
        # It reaches the progress line and the per-epoch diagnostics, not a gradient, but a
        # diagnostic that says epoch 2 of 3 for the last third of an update is one a reader will
        # act on.
        from .buffer import batch_count

        per_epoch = batch_count(n_samples, config.batch_size) if n_samples else 0
        diagnostics = _Diagnostics(config.n_epochs, self.device)
        checking = self._asserts_due(sched.iteration)
        batches = buffer.batches(
            config.batch_size,
            config.minibatch_size,
            config.n_epochs,
            self._rng_for_epoch(sched.iteration),
        )
        started_at = time.perf_counter()
        said_epoch = -1
        for index, batch in enumerate(batches):
            epoch = index // per_epoch if per_epoch else 0
            if self.progress is not None and epoch != said_epoch:
                # The update is the long quiet phase -- on a contended machine it has run for
                # three quarters of an hour -- and its own timing only reaches the metric
                # stream once the iteration ends, so a run interrupted inside it can say
                # nothing about how long it was. One line per epoch is enough to tell a
                # waiting run from a stopped one and to size the phase from outside.
                said_epoch = epoch
                self.progress(
                    f"updating      epoch {epoch + 1}/{config.n_epochs}, "
                    f"{n_samples} samples, {time.perf_counter() - started_at:.0f}s in"
                )
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            for position, minibatch in enumerate(batch):
                first = index == 0 and position == 0
                self._minibatch(
                    minibatch, sched, diagnostics, epoch, check=checking, first=first
                )
            diagnostics.gradients(
                clip_grad_norm_(self.actor_params, config.max_grad_norm),
                clip_grad_norm_(self.critic_params, config.max_grad_norm),
            )
            for optimizer in self.optimizers:
                optimizer.step()
            self.model_updates += 1
        if self.device.type == "cuda":  # pragma: no cover - the suite runs on the CPU
            # One synchronisation for the whole update, and every .item() below is behind it.
            torch.cuda.synchronize(self.device)

        after_actor = parameters_to_vector(self.actor_params).detach()
        after_critic = parameters_to_vector(self.critic_params).detach()
        result = diagnostics.result(
            n_samples=n_samples,
            epochs=config.n_epochs,
            explained=explained_variance(returns[mask], values[: buffer.cycles][mask]),
            update_actor=float((after_actor - before_actor).norm().item()),
            update_critic=float((after_critic - before_critic).norm().item()),
            seconds=time.perf_counter() - started,
        )
        if self.backoff is not None and self.backoff.observe(result.kl):
            print(
                f"lr_backoff: KL stayed above {self.backoff.config.kl_threshold} for "
                f"{self.backoff.config.patience} iterations; the rates are now "
                f"{self.backoff.lr_actor:g} and {self.backoff.lr_critic:g}"
            )
        return result

    def _minibatch(
        self,
        minibatch: Minibatch,
        sched: ScheduleState,
        diagnostics: _Diagnostics,
        epoch: int,
        *,
        check: bool,
        first: bool,
    ) -> None:
        """One forward, one backward, and the diagnostics that come off them."""
        config = self.config
        weight = minibatch.weight
        if check:
            # The action that was taken was legal under the mask that was stored with it. An
            # unmasked update pins the clip fraction at one; a mask applied to the wrong row
            # gives a log-probability of finfo.min and NaN. Both are silent for hours.
            assert bool(
                minibatch.obs.mask.gather(-1, minibatch.actions.unsqueeze(-1)).all()
            ), _mask_message(minibatch, self._n_slots)
        result: BackpropResult = self.model.backprop(minibatch.obs, minibatch.actions)
        log_probs = result.log_probs
        if check:
            assert bool(torch.isfinite(log_probs).all()), (
                "a log-probability in this minibatch is not finite, which means an action was "
                "scored under a mask that forbade it"
            )
        ratio = torch.exp(log_probs - minibatch.log_probs)
        if first:
            self._ratio_invariant(ratio, minibatch, diagnostics, sched.iteration)
        surr, dual = surrogate(
            ratio,
            minibatch.advantages,
            clip_range=config.clip_range,
            dual_clip_c=config.dual_clip_c,
        )
        mean_squared_error = value_error(
            result.values,
            minibatch.values,
            minibatch.returns,
            clip_range=config.clip_range if config.value_clipping else None,
        )
        policy_loss = -dual.mean() * weight
        value_loss = config.vf_coef * mean_squared_error * weight
        entropy_loss = (
            -(
                sched.ent_coef * result.entropy.mean()
                + sched.ent_coef_noop * result.noop_entropy.mean()
            )
            * weight
        )
        (policy_loss + value_loss + entropy_loss).backward()
        with torch.no_grad():
            diagnostics.minibatch(
                epoch=epoch,
                n=minibatch.n,
                ratio=ratio,
                advantages=minibatch.advantages,
                surr=surr,
                dual=dual,
                result=result,
                value_loss=mean_squared_error.detach(),
                clip_range=config.clip_range,
                dual_clip_c=config.dual_clip_c,
            )

    def _ratio_invariant(
        self,
        ratio: Tensor,
        minibatch: Minibatch,
        diagnostics: _Diagnostics,
        iteration: int,
    ) -> None:
        """At the first minibatch of the first epoch every ratio should be one.

        Nothing has changed since the rollout: the parameters are the ones that acted and the
        input bytes are the ones they acted on, because the policy acted on the decoded form of
        exactly these rows. The tolerance is a property of the precision and not a fudge factor
        -- under bf16 the two forwards are differently shaped, cuDNN picks a kernel per shape,
        and three significant digits put the fp32 logits about a percent apart -- and every
        failure this is aimed at moves the ratio by a quantity of order one.
        """
        with torch.no_grad():
            diagnostics.ratio_deviation = (ratio - 1.0).abs().max()
            if not self._ratio_check_due(iteration):
                return
            atol = self._ratio_atol()
            value = float(diagnostics.ratio_deviation.item())
            assert value <= atol, _ratio_message(value, ratio, minibatch, atol, self._n_slots)

    # -- the knobs -----------------------------------------------------------

    def _asserts_due(self, iteration: int) -> bool:
        return iteration < self.config.debug_assert_iterations

    def _ratio_check_due(self, iteration: int) -> bool:
        if iteration < self.config.debug_assert_iterations:
            return True
        every = self.config.check_ratio_invariant_every
        return every > 0 and iteration % every == 0

    def _ratio_atol(self) -> float:
        dtype = getattr(self.model, "autocast_dtype", torch.float32)
        name = PRECISION_NAMES.get(dtype, str(dtype))
        try:
            return float(self.config.ratio_atol[name])
        except KeyError:
            raise KeyError(
                f"ppo.ratio_atol has no tolerance for {name}; it lists "
                f"{', '.join(sorted(self.config.ratio_atol))}"
            ) from None

    def _apply_learning_rates(self, sched: ScheduleState) -> None:
        """Take this iteration's rates. They move only under the backoff, which is why they
        arrive on the schedule state rather than being read from the config here."""
        self.actor_kwargs["lr"] = sched.lr_actor
        self.critic_kwargs["lr"] = sched.lr_critic
        for optimizer, rate in (
            (self.actor_optimizer, sched.lr_actor),
            (self.critic_optimizer, sched.lr_critic),
        ):
            for group in optimizer.param_groups:
                group["lr"] = rate

    def _rng_for_epoch(self, iteration: int) -> Callable[[int], np.random.Generator]:
        """The permutation of one epoch, from its own named stream.

        Addressed by iteration and epoch rather than drawn from a generator the update carries,
        so that adding a consumer anywhere else in the harness does not move it.
        """

        def generator(epoch: int) -> np.random.Generator:
            path = stream_path(PPO_MINIBATCH, iteration=iteration, epoch=epoch)
            return derive_generator(self.master_seed, path)

        return generator

    def _gather_for(self, buffer: RectBuffer) -> RectGather:
        if self._gather is None or self._gather.buffer is not buffer:
            self._gather = RectGather(buffer, rows=max(1, self.config.critic_chunk))
        return self._gather

    def critic_on_final_obs(
        self, buffer: RectBuffer, gather: RectGather, cells: np.ndarray, rows: np.ndarray
    ) -> Tensor:
        """``V(final_obs)`` for the cells that truncated, in the order they were recorded."""
        truncated = buffer.truncated[: buffer.cycles]
        if not bool(truncated[cells[:, 0], cells[:, 1]].all()):
            raise ValueError(
                "a final observation was recorded for a cell the rectangle does not have "
                "marked truncated; a final observation belongs to the transition that was cut"
            )
        out = torch.empty(cells.shape[0], dtype=torch.float32, device=self.device)
        step = max(1, self.config.critic_chunk)
        for start in range(0, cells.shape[0], step):
            block = cells[start : start + step]
            obs = gather.observations(
                block[:, 0], block[:, 1], current=rows[start : start + step]
            )
            with torch.no_grad():
                out[start : start + block.shape[0]] = self.model.value(obs).float()
        return out

    # -- checkpoint ----------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        """The two optimizer states and the update counter.

        ``torch.save`` for the optimizer state and nothing else: the moments are tensors keyed
        by parameter index, which is what ``weights_only=True`` can read back, and the weights
        themselves are safetensors in the folder next door.
        """
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor_optimizer.state_dict(), folder / self.ACTOR_FILE)
        torch.save(self.critic_optimizer.state_dict(), folder / self.CRITIC_FILE)
        state = {
            "format_version": self.FORMAT_VERSION,
            "cumulative_model_updates": self.model_updates,
        }
        (folder / self.STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load_checkpoint(self, folder: Path, *, strict: bool) -> None:
        """Restore both optimizers, then the configured keyword arguments over them.

        The order is the point. An optimizer state dict carries the learning rate the run was
        saved at, so loading it last would silently undo a rate the user changed for the resumed
        half of the run.
        """
        for optimizer, name, kwargs in (
            (self.actor_optimizer, self.ACTOR_FILE, self.actor_kwargs),
            (self.critic_optimizer, self.CRITIC_FILE, self.critic_kwargs),
        ):
            path = folder / name
            if not path.exists():
                if strict:
                    raise CheckpointFormatError(f"the optimizer state is not at {path}")
                print(f"no optimizer state at {path}; its moments start from nothing")
                continue
            optimizer.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            for group in optimizer.param_groups:
                group.update(kwargs)
        path = folder / self.STATE_FILE
        if not path.exists():
            if strict:
                raise CheckpointFormatError(f"the update's own state is not at {path}")
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        version = int(state.get("format_version", 0))
        if version > self.FORMAT_VERSION:
            raise CheckpointFormatError(
                f"{path} was written at update format {version} and this build reads "
                f"{self.FORMAT_VERSION}"
            )
        self.model_updates = int(state.get("cumulative_model_updates", 0))


# --------------------------------------------------------------------------
# The diagnostics
# --------------------------------------------------------------------------


class _Diagnostics:
    """Sums over the minibatches of one iteration, kept on the device until the end.

    Every quantity is accumulated as a tensor and read out once, after the single
    synchronisation: a ``.item()`` inside the minibatch loop would put a device synchronisation
    between every backward pass and the next forward, which costs more than the numbers do.
    """

    def __init__(self, epochs: int, device: torch.device) -> None:
        self.device = device
        self.epochs = max(1, int(epochs))
        self.samples = 0
        self.minibatches = 0
        self.steps = 0
        #: Rows whose mask offered more than the no-op. The play/wait entropy is a mean over
        #: these and not over every row; see ``_Diagnostics.minibatch``.
        self._chose = torch.zeros((), dtype=torch.float32, device=device)
        self._sums = {
            name: torch.zeros((), dtype=torch.float32, device=device)
            for name in (
                "policy_loss",
                "value_loss",
                "entropy",
                "noop_entropy",
                "forced_rows",
                "entropy_normalised",
                "kl",
                "clip_fraction",
                "dual_clip_fraction",
            )
        }
        self._epoch_kl = [
            torch.zeros((), dtype=torch.float32, device=device) for _ in range(self.epochs)
        ]
        self._epoch_clip = [
            torch.zeros((), dtype=torch.float32, device=device) for _ in range(self.epochs)
        ]
        self._epoch_n = [0 for _ in range(self.epochs)]
        self._grad_actor: list[Tensor] = []
        self._grad_critic: list[Tensor] = []
        #: The FIRST minibatch's worst ratio deviation, and deliberately only that one.
        #:
        #: It reads exactly 0.0 on a healthy update -- 124 of 124 metric rows on disk as of
        #: 2026-09-22 -- which looks like a dead metric and is not. At the first minibatch of
        #: the first epoch nothing has been stepped, so the stored log-probabilities and a fresh
        #: forward over the stored bytes must agree; a non-zero value there means a mask, codec
        #: or weight-version mismatch, and `ratio_invariant` halts the run on it. Zero is the
        #: whole point.
        #:
        #: What it is NOT is a blow-up detector, and it was briefly changed into one here by
        #: taking the worst deviation over every minibatch instead. That reads 0.237 on a
        #: perfectly healthy update -- a policy that moved enough to clip, at a clip_range of
        #: 0.2 -- and it halted a run on its first outing. An invariant that must hold and a
        #: diagnostic that is expected to move cannot share a key: the alarm reads this one, so
        #: this one stays the invariant. A worst-over-update figure would be a new key with its
        #: own threshold, and is not worth one until something wants it.
        self.ratio_deviation: Tensor | None = None
        self.ratio_value = 0.0

    def minibatch(
        self,
        *,
        epoch: int,
        n: int,
        ratio: Tensor,
        advantages: Tensor,
        surr: Tensor,
        dual: Tensor,
        result: BackpropResult,
        value_loss: Tensor,
        clip_range: float,
        dual_clip_c: float,
    ) -> None:
        # Every quantity derived from the ratio is conditioned on the rows that had a choice,
        # for one structural reason: where the mask leaves a single action the distribution is
        # a point mass at it before and after the update, so the ratio is exactly one and the
        # row contributes exactly zero KL and zero clipping. It is not a policy that did not
        # move, it is a policy that could not. On this environment nine rows in ten are that
        # one, so an unconditioned mean is the choice-bearing mean times a fraction set by the
        # elixir economy -- a fraction that moves as the policy learns to hold elixir, as the
        # deck changes, and in overtime at double rate. A threshold against it is therefore
        # wrong in a way that re-tuning cannot fix, because the baseline is not stationary.
        # This matters past the dashboard: `lr_backoff` reads this KL, so a diluted one makes
        # the brake that stops a blow-up unreachable by the same factor.
        chose = (result.n_legal > 1).to(ratio.dtype)
        chose_n = chose.sum()
        denominator = chose_n.clamp_min(1.0)
        kl = (approx_kl(ratio) * chose).sum() / denominator
        clip = (clipped_fraction(ratio, clip_range) * chose).sum() / denominator
        # The normaliser is the entropy of a uniform policy over this row's legal set, so a
        # policy that has learnt to wait -- and therefore sees fewer legal actions -- is not
        # read as one that has collapsed.
        legal = result.n_legal.clamp_min(2).to(ratio.dtype).log()
        # The play/wait entropy is conditioned on the rows that HAD a choice. On the shipped
        # environment about nine decisions in ten leave exactly one legal action -- the elixir
        # bar can afford nothing -- and on those the binary entropy is zero by construction,
        # not by anything the policy did. Averaging over all rows measures the elixir curve:
        # it reads near zero on a healthy run, so a floor set on it fires permanently, and a
        # gate that really had collapsed would move it by a fraction of what it moves here.
        self.samples += n
        self.minibatches += 1
        self._chose += chose_n
        self._sums["policy_loss"] += -dual.mean() * n
        self._sums["value_loss"] += value_loss * n
        # Both entropies are conditioned on the rows that HAD a choice, for the same reason the
        # KL is. A forced row's entropy is exactly zero -- one legal action is a point mass --
        # so averaging over every row multiplies the answer by the choosing fraction and the
        # published number becomes a reading of the elixir bar. Measured 2026-09-22 over the 124
        # metric rows on disk: `entropy_normalised` equalled `1 - forced_noop_frac` to within
        # 0.0075 everywhere, because a forced row contributes 0/log2 and a choosing row
        # contributes entropy/log(n_legal), which is 1 for a policy still uniform on its legal
        # set. The key was an algebraic restatement of a different key, and its documented
        # healthy band of 0.3-0.8 could not be reached by any policy at this forced fraction.
        self._sums["entropy"] += (result.entropy * chose).sum()
        self._sums["noop_entropy"] += (result.noop_entropy * chose).sum()
        self._sums["forced_rows"] += (n - chose_n)
        self._sums["entropy_normalised"] += ((result.entropy / legal) * chose).sum()
        self._sums["kl"] += kl * chose_n
        self._sums["clip_fraction"] += clip * chose_n
        self._sums["dual_clip_fraction"] += (
            dual_clipped_fraction(advantages, surr, dual_clip_c=dual_clip_c) * chose
        ).sum() / denominator * chose_n
        index = min(epoch, self.epochs - 1)
        self._epoch_kl[index] += kl * chose_n
        self._epoch_clip[index] += clip * chose_n
        self._epoch_n[index] += float(chose_n.item())

    def gradients(self, actor: Tensor, critic: Tensor) -> None:
        self._grad_actor.append(actor.detach())
        self._grad_critic.append(critic.detach())
        self.steps += 1

    def result(
        self,
        *,
        n_samples: int,
        epochs: int,
        explained: float,
        update_actor: float,
        update_critic: float,
        seconds: float,
    ) -> UpdateResult:
        total = max(1, self.samples)
        means = {name: float((value / total).item()) for name, value in self._sums.items()}
        # Most of them are not means over every row. Everything the policy could have moved is
        # summed over the rows that had a choice and divided by those; the forced-row count is
        # a count. Only `policy_loss` and `value_loss` stay means over the whole batch, because
        # they are the quantities actually optimised and their denominator is part of the
        # optimisation rather than part of the reporting.
        chose = float(self._chose.item())
        divisor = max(chose, 1.0)
        for name in (
            "entropy",
            "entropy_normalised",
            "noop_entropy",
            "kl",
            "clip_fraction",
            "dual_clip_fraction",
        ):
            means[name] = float((self._sums[name] / divisor).item()) if chose else 0.0
        means["forced_rows"] = float(self._sums["forced_rows"].item())
        if self.ratio_deviation is not None:
            self.ratio_value = float(self.ratio_deviation.item())
        expected = n_samples * max(1, epochs)
        unused = 1.0 - (self.samples / expected) if expected else 0.0
        return UpdateResult(
            policy_loss=means["policy_loss"],
            value_loss=means["value_loss"],
            entropy=means["entropy"],
            noop_entropy=means["noop_entropy"],
            entropy_normalised=means["entropy_normalised"],
            kl=means["kl"],
            clip_fraction=means["clip_fraction"],
            dual_clip_fraction=means["dual_clip_fraction"],
            explained_variance=explained,
            ratio_max_abs_dev=self.ratio_value,
            grad_norm_actor=_mean_of(self._grad_actor),
            grad_norm_critic=_mean_of(self._grad_critic),
            update_magnitude_actor=update_actor,
            update_magnitude_critic=update_critic,
            kl_by_epoch=_per_epoch(self._epoch_kl, self._epoch_n),
            clip_fraction_by_epoch=_per_epoch(self._epoch_clip, self._epoch_n),
            n_minibatches=self.minibatches,
            n_optimizer_steps=self.steps,
            n_samples=n_samples,
            samples_unused_frac=max(0.0, unused),
            seconds=seconds,
        )


def _per_epoch(sums: Sequence[Tensor], counts: Sequence[int]) -> list[float]:
    return [
        float((value / count).item()) if count else 0.0
        for value, count in zip(sums, counts, strict=True)
    ]


def _mean_of(values: Sequence[Tensor]) -> float:
    if not values:
        return 0.0
    return float(torch.stack([value.reshape(()) for value in values]).mean().item())


def _adam(params: Iterable[Parameter], **kwargs: Any) -> Optimizer:
    """The default factory: one Adam over one parameter set, at the configured rate and epsilon.

    A factory rather than a constructed optimizer because the keyword arguments are what a
    resume restores over a loaded state dict, and because it is the seam anything else -- a
    different optimizer, a fused one, a parameter group split -- is swapped in through.
    """
    return torch.optim.Adam(list(params), **kwargs)


def _model_device(model: ActorCritic) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    for parameter in model.actor_parameters():
        return parameter.device
    return torch.device("cpu")  # pragma: no cover - a model with no parameters


def _cell(minibatch: Minibatch, row: int, n_slots: int) -> str:
    """One sample, named the way the rest of the harness names a cell."""
    cell = int(minibatch.cells[row])
    return f"cycle {cell // n_slots} slot {cell % n_slots}"


def _mask_message(minibatch: Minibatch, n_slots: int) -> str:
    """Which samples took an action their own stored mask forbids."""
    legal = minibatch.obs.mask.gather(-1, minibatch.actions.unsqueeze(-1)).squeeze(-1)
    offending = torch.nonzero(~legal).reshape(-1)[:10].tolist()
    lines = [
        f"  {_cell(minibatch, row, n_slots)} took action {int(minibatch.actions[row])}, "
        "which its stored mask forbids"
        for row in offending
    ]
    return (
        f"{int((~legal).sum())} of {legal.numel()} samples took an action their stored mask "
        "forbids. The mask the update applies is the one the transition carried, so this is a "
        "rollout that sampled outside its own mask:\n" + "\n".join(lines)
    )


def _ratio_message(
    deviation: float, ratio: Tensor, minibatch: Minibatch, atol: float, n_slots: int
) -> str:
    """The worst ten samples, and the three causes in the order they are worth checking."""
    ratio = ratio.detach()
    worst = (ratio - 1.0).abs().argsort(descending=True)[:10].tolist()
    lines = [
        f"  {_cell(minibatch, row, n_slots)}, action {int(minibatch.actions[row])}: "
        f"ratio {float(ratio[row]):.6g}"
        for row in worst
    ]
    causes = "\n".join(f"  {index}. {cause}" for index, cause in enumerate(RATIO_CAUSES, 1))
    return (
        f"the importance ratio deviates from one by {deviation:.3g} at the first minibatch of "
        f"the first epoch, against a tolerance of {atol:.3g}. Nothing has changed since the "
        "rollout, so it should be one.\n"
        + "\n".join(lines)
        + "\nIn order of likelihood:\n"
        + causes
    )
