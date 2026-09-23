"""The parent's half of a round: one batched forward per distinct policy.

A round arrives as a rectangle of slots with a ``group`` column saying who controls each seat.
The parent groups by that column, runs one forward per group, and writes the actions back in
slot order. Three properties are what this file is for, and each of them is a decision rather
than an implementation detail:

**One forward per distinct policy, not one per battle.** A round of a few hundred seats holds
at most a handful of policies -- the learner, and the snapshots the matchmaker put on the board
this iteration -- so grouping by policy turns a few hundred small forwards into a few large
ones. Group sizes change from round to round because assignments change at episode boundaries;
that costs a little kernel-shape churn and nothing else, and it is the price of binding one
policy to a battle for a whole episode, which is the thing that must not be given up.

**Ascending slot order inside a group.** Never arrival order, and never the order a dictionary
happened to iterate. The batch a policy sees is then a function of the slot indices alone, so a
worker that answered a millisecond later does not reorder anybody's rows.

**The uniforms come from a named stream, indexed by slot.** ``act/iteration/{i}/cycle/{t}`` is
drawn once per cycle as an ``(R,)`` vector and every shard of that cycle indexes its own slots
out of it. A sampled action is therefore a pure function of the master seed, the iteration, the
cycle, the slot and the logits -- not of how many rows shared the forward, and not of torch's
global generator. That is what makes a trajectory reproducible whatever ``rollout.overlap`` is
set to and whatever the farm's timing does.

Frozen snapshots run under ``torch.inference_mode`` in half precision. They are played against
and never trained from, so there is no autograd graph to build and no gradient to keep exponent
range for; fp16 is the right half precision here for the same reason bf16 is the right one for
the learner.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
from torch import Tensor

from ..api.policy import ObsBatch
from ..api.rollout import GROUP_DEAD, GROUP_LEARNER, GROUP_SCRIPTED
from ..seeding import ACT_CYCLE, derive_generator, stream_path
from .buffer import _StagingRing
from .distribution import MaskedCategorical

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.ladder import SnapshotStore
    from ..api.policy import Actor, ActorCritic
    from ..api.rollout import RolloutRound, SlotPlan
    from .actor_critic import BehaviourSnapshot
    from .buffer import RectBuffer

__all__ = [
    "BatchedInference",
    "InferenceResult",
    "RectGather",
    "RoundStats",
    "frozen_dtype",
]


def frozen_dtype(device: torch.device) -> torch.dtype:
    """The precision a pool snapshot is played at on this device.

    Half on a device that has half-precision arithmetic in hardware, and float32 anywhere else:
    on a CPU, fp16 is an emulation and is slower than the thing it would be saving, so the whole
    test suite would pay for a speed-up it cannot have.
    """
    return torch.float16 if device.type == "cuda" else torch.float32


class RectGather:
    """Observations for arbitrary cells of the rectangle, unpacked on the device.

    The minibatch path has its own gather, which also carries the scalar columns; this is the
    one the parent uses everywhere else -- the rollout forward, the whole-iteration critic pass,
    and the critic pass over the observation a truncated episode ended on. All three want the
    same thing and none of them wants an action or an advantage.

    The rows a cell's frame stack reaches, which of them are live, and the static planes that are
    scattered into every frame are the rectangle's own rules, asked of the rectangle rather than
    restated here: a second copy of the liveness rule is a second answer to "does this frame
    belong to this episode", and the one that is wrong would be silent.
    """

    def __init__(self, buffer: RectBuffer, *, rows: int, staging_slots: int = 4) -> None:
        self.buffer = buffer
        self.device = buffer.device
        self.capacity = max(1, int(rows))
        self.frames = buffer.frame_stack
        self.row_bytes = buffer.layout.row_bytes
        self._ring = _StagingRing(
            staging_slots, self.capacity, self.frames, self.row_bytes, self.device
        )

    def observations(
        self,
        cycles: np.ndarray,
        slots: np.ndarray,
        *,
        current: np.ndarray | None = None,
    ) -> ObsBatch:
        """The stacked observation of each ``(cycle, slot)``, on the device.

        ``current`` replaces frame zero with packed rows of the caller's own -- the observation
        a truncated episode ended on, which is not in the rectangle, because what the rectangle
        holds at that cell is the first observation of the next episode. The history frames
        behind it are still the rectangle's, which is what makes the stack the one the episode
        actually ended with.
        """
        cycles = np.asarray(cycles, dtype=np.int64)
        slots = np.asarray(slots, dtype=np.int64)
        count = int(cycles.size)
        if count > self.capacity:
            raise ValueError(
                f"this gather stages {self.capacity} rows at a time and was asked for {count}"
            )
        rows, live = self.buffer._stack_rows(cycles, slots)
        slab, view = self._ring.next()
        staged = view[:count]
        np.take(
            self.buffer.obs_view,
            rows.reshape(-1),
            axis=0,
            out=staged.reshape(count * self.frames, self.row_bytes),
        )
        if current is not None:
            staged[:, 0] = current
        if not live.all():
            staged[~live] = 0
        raw = slab[:count].to(self.device, non_blocking=True)
        self._ring.copied()
        out = self.buffer._empty_obs(count)
        self.buffer.codec.unpack_to_device(raw, self.buffer._statics(), out)
        return out


class RoundStats(NamedTuple):
    """What the rollout forwards of one iteration cost and what the policy looked like in them.

    Sums rather than means, so that draining them at the iteration boundary is an addition and
    the mean is taken once, over the samples it is a mean of. ``rows`` counts the learner's own
    seats: a frozen opponent's entropy belongs to the run that trained it.
    """

    rows: int
    forwards: int
    rounds: int
    seconds: float
    entropy: float
    p_noop: float
    n_legal: float
    #: The three below are over the rows whose mask offered more than the no-op, and they are
    #: separate sums rather than a ratio because the mean has to be taken over the rows it is a
    #: mean of. ``hold_lift`` is the sum of p(no-op) * n_legal: each row's hold mass against the
    #: uniform baseline of its OWN width, so averaging them is an average of comparable numbers.
    choice_rows: int
    hold: float
    hold_lift: float
    choice_n_legal: float


class InferenceResult(NamedTuple):
    """One round's answer, in the round's own slot order.

    ``log_probs`` is written for the learner's seats and left at zero elsewhere: a frozen or
    scripted seat's transition is not trained on, and a log-probability from another policy's
    parameters would be an importance ratio against a denominator nothing ever optimised.
    """

    actions: np.ndarray
    log_probs: np.ndarray


class BatchedInference:
    """The policy side of the round protocol.

    Holds the gather, the learner's network, the snapshot store the frozen opponents are loaded
    from, and the plan that says which snapshot a group index means. It draws its own uniforms
    unless it is handed some, so that the stream a sampled action comes from is a property of
    this object rather than of whoever called it.
    """

    def __init__(
        self,
        buffer: RectBuffer,
        model: ActorCritic,
        *,
        master_seed: int,
        snapshots: SnapshotStore | None = None,
        device: torch.device | str | None = None,
        staging_slots: int = 4,
    ) -> None:
        self.buffer = buffer
        self.model = model
        self.snapshots = snapshots
        self.master_seed = int(master_seed)
        self.device = torch.device(device) if device is not None else buffer.device
        self.n_slots = buffer.n_slots
        self.gather = RectGather(buffer, rows=self.n_slots, staging_slots=staging_slots)
        self.plan: SlotPlan | None = None
        self.iteration = 0
        #: The actor the learner's seats sample from. The live one until a behaviour snapshot is
        #: set, which is what overlapped collection does at every iteration boundary.
        self.behaviour: BehaviourSnapshot | None = None
        self._uniforms: np.ndarray | None = None
        self._uniform_cycle = -1
        self._stats = _StatAccumulator()

    # -- the iteration ------------------------------------------------------

    def begin_iteration(
        self, plan: SlotPlan, *, behaviour: BehaviourSnapshot | None = None
    ) -> None:
        """Take the iteration's opening table and, with overlap, the actor to sample from.

        The plan is what turns a non-negative group index into a snapshot id: the workers carry
        one byte per slot and the parent owns the table it indexes.
        """
        self.plan = plan
        self.iteration = plan.iteration
        self.behaviour = behaviour
        self._uniforms = None
        self._uniform_cycle = -1

    def uniforms(self, cycle: int) -> np.ndarray:
        """The ``(R,)`` uniform vector of one cycle, drawn once and shared by its shards.

        A cycle is several shard-rounds, each covering its own slots, and all of them index this
        one vector. Drawing per shard would make an action depend on how the battles were
        distributed over the workers.
        """
        if self._uniforms is None or self._uniform_cycle != cycle:
            path = stream_path(ACT_CYCLE, iteration=self.iteration, cycle=cycle)
            self._uniforms = derive_generator(self.master_seed, path).random(
                self.n_slots, dtype=np.float32
            )
            self._uniform_cycle = cycle
        return self._uniforms

    # -- one round ----------------------------------------------------------

    def act(self, round_: RolloutRound, uniforms: np.ndarray | None = None) -> InferenceResult:
        """Answer one shard-round: one forward per distinct policy, in slot order."""
        started = time.perf_counter()
        draw = self.uniforms(round_.cycle) if uniforms is None else np.asarray(uniforms)
        slots = np.asarray(round_.slots, dtype=np.int64)
        group = np.asarray(round_.group)
        actions = np.zeros(slots.size, dtype=np.int64)
        log_probs = np.zeros(slots.size, dtype=np.float32)
        # A row is routed on what its worker REPORTED, never on what the plan intended for it.
        # `valid` is false where the worker wrote nothing this round -- it died, it timed out,
        # or it is being restarted -- and the cell it did not write holds whatever the
        # rectangle held before, which for a fresh iteration is zeros. Reading one of those
        # would hand the policy an observation with no legal action in it, including the no-op
        # the environment always sets, and the first thing to notice would be an assertion
        # inside the distribution rather than the worker that stopped. The action left behind
        # is the no-op, which is what a seat that could not be asked should do.
        live = np.asarray(round_.valid, dtype=bool)
        forwards = 0
        for gid in sorted(int(value) for value in np.unique(group[live])):
            if gid in (GROUP_SCRIPTED, GROUP_DEAD):
                # The worker fills its own scripted seats, and a dead worker's slots have no
                # observation to read. Neither is a policy the parent holds.
                continue
            rows = np.flatnonzero((group == gid) & live)
            if rows.size == 0:  # pragma: no cover - np.unique only reports values that occur
                continue
            forwards += 1
            row_slots = slots[rows]
            obs = self.gather.observations(
                np.full(rows.size, round_.cycle, dtype=np.int64), row_slots
            )
            u = torch.from_numpy(np.ascontiguousarray(draw[row_slots])).to(self.device)
            if gid == GROUP_LEARNER:
                chosen, chosen_log_probs = self._act_learner(obs, u)
                log_probs[rows] = chosen_log_probs
            else:
                chosen = self._act_frozen(self._snapshot(gid), obs, u)
            actions[rows] = chosen
        self._stats.round(forwards=forwards, seconds=time.perf_counter() - started)
        return InferenceResult(actions=actions, log_probs=log_probs)

    def _act_learner(self, obs: ObsBatch, uniforms: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Sample the learner's seats, and record what the policy looked like doing it.

        Under ``no_grad`` and not ``inference_mode``: the actor sampled from here is the one the
        update differentiates, and a tensor produced in inference mode carries a flag that
        refuses to take part in a later autograd graph.
        """
        actor: Any = self.behaviour if self.behaviour is not None else self.model
        with torch.no_grad():
            distribution = MaskedCategorical(actor.logits(obs).float(), obs.mask)
            actions = distribution.sample(uniforms)
            log_probs = distribution.log_prob(actions)
            self._stats.policy(distribution, actions.numel())
            return _numpy(actions, np.int64), _numpy(log_probs, np.float32)

    def _act_frozen(self, actor: Actor, obs: ObsBatch, uniforms: Tensor) -> np.ndarray:
        """Sample a pool opponent's seats. Nothing here is ever differentiated."""
        with torch.inference_mode():
            distribution = MaskedCategorical(actor.logits(obs).float(), obs.mask)
            return _numpy(distribution.sample(uniforms), np.int64)

    def _snapshot(self, group: int) -> Actor:
        """The frozen actor a group index names, through the plan's resident table."""
        if self.plan is None:
            raise RuntimeError(
                "a round carried a pool opponent before begin_iteration() was given the plan "
                "that says which snapshot its group index means"
            )
        if self.snapshots is None:
            raise RuntimeError(
                "a round carried a pool opponent and this inference was built without a "
                "snapshot store to load one from"
            )
        resident = self.plan.resident_snapshots
        if not 0 <= group < len(resident):
            raise IndexError(
                f"a round carried group {group}, and this iteration's plan made "
                f"{len(resident)} snapshots resident"
            )
        actor = self.snapshots.get(resident[group], self.device)
        dtype = frozen_dtype(self.device)
        if dtype is not torch.float32:
            actor.autocast_dtype = dtype  # type: ignore[attr-defined]
        return actor

    # -- what it cost -------------------------------------------------------

    def drain_stats(self) -> RoundStats:
        """The rollout forwards since the last drain, and the accumulator reset."""
        return self._stats.drain()


class _StatAccumulator:
    """Running sums over the rounds of one iteration."""

    __slots__ = (
        "choice_n_legal",
        "choice_rows",
        "entropy",
        "forwards",
        "hold",
        "hold_lift",
        "n_legal",
        "p_noop",
        "rounds",
        "rows",
        "seconds",
    )

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.rows = 0
        self.forwards = 0
        self.rounds = 0
        self.seconds = 0.0
        self.entropy = 0.0
        self.p_noop = 0.0
        self.n_legal = 0.0
        self.choice_rows = 0
        self.hold = 0.0
        self.hold_lift = 0.0
        self.choice_n_legal = 0.0

    def round(self, *, forwards: int, seconds: float) -> None:
        self.forwards += forwards
        self.rounds += 1
        self.seconds += seconds

    def policy(self, distribution: MaskedCategorical, rows: int) -> None:
        """The learner's own rows only: a frozen opponent's entropy is not this run's."""
        self.rows += rows
        p_noop = distribution.p_noop()
        n_legal = distribution.n_legal()
        self.entropy += float(distribution.entropy().sum().item())
        self.p_noop += float(p_noop.sum().item())
        self.n_legal += float(n_legal.sum().item())
        # A row with one legal action holds with probability 1 and lifts by exactly 1, whatever
        # the policy is. Nine decisions in ten are that row on this environment, so a mean over
        # every row would report the elixir curve dragging a concentrated policy back towards
        # uniform -- the same defect that took `noop_entropy` and `entropy_normalised` onto
        # choice rows only.
        choice = n_legal > 1
        held = p_noop[choice]
        widths = n_legal[choice].to(held.dtype)
        self.choice_rows += int(choice.sum().item())
        self.hold += float(held.sum().item())
        self.hold_lift += float((held * widths).sum().item())
        self.choice_n_legal += float(widths.sum().item())

    def drain(self) -> RoundStats:
        stats = RoundStats(
            rows=self.rows,
            forwards=self.forwards,
            rounds=self.rounds,
            seconds=self.seconds,
            entropy=self.entropy,
            p_noop=self.p_noop,
            n_legal=self.n_legal,
            choice_rows=self.choice_rows,
            hold=self.hold,
            hold_lift=self.hold_lift,
            choice_n_legal=self.choice_n_legal,
        )
        self._reset()
        return stats


def _numpy(tensor: Tensor, dtype: Any) -> np.ndarray:
    """A tensor as a plain numpy array of its own.

    The copy is not incidental: the frozen path produces its tensors inside
    ``torch.inference_mode``, and an array that aliased one of those would outlive the guard
    holding a view of memory the allocator is free to reuse.
    """
    return tensor.detach().to("cpu").numpy().astype(dtype, copy=True)
