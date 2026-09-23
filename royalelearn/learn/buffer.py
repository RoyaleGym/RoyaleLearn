"""The iteration's experience, as one rectangle.

``(T + k)`` cycles by ``R`` slots of packed observation rows over a single shared-memory block,
plus the scalar columns beside it in the parent's own numpy arrays. Cycle ``T`` holds observations
only -- it is the bootstrap row -- and the ``k - 1`` rows below cycle zero are the history carried
down from the previous iteration, so an iteration boundary is not a discontinuity in what the
policy sees.

ONE ROW IS ONE TIMESTEP. Row ``t`` of a slot holds the observation ``s_t``, the action taken in
it, the log-probability that action was drawn with, the reward that action earned, the flag that
says whether it ended the episode and the engine's answer to the command it carried. Nothing in
the rectangle is stored a row away from the row it describes, and every column may be read
against every other at the same index. A rollout round carries two timesteps' worth of news --
its own state, and the step that arrived at it -- and ``record_round`` is the one place that is
taken apart.

Three further properties are worth stating because each is a decision:

**Every row of the rectangle is stored, including the seats a frozen or scripted opponent
played.** That costs a quarter more memory than storing only learner rows and it buys a buffer
index that IS the slot index: no second map from slot to row to get wrong, and no scratch area for
rows that turn out to be discarded. What decides whether a cell reaches the update is
``trainable``, never where the row was written.

**Frame stacking is a gather, not a second copy.** A slot's rows sit at a fixed stride of ``R`` in
the index, so the stack for a cell is rows ``t, t-1, ... t-k+1`` of the same slot, assembled at
unpack time in the same kernel that dequantises. ``episode_end[p]`` is the boundary AFTER row
``p`` -- row ``p`` is the last row of the episode it ended -- so the frame at row ``p`` belongs
to the cell above it exactly while ``episode_end[p]`` is none. The stack is zero-filled from the
first boundary back, so the first frame of an episode has a zero history and the policy is never
shown the tail of the battle before it.

**Minibatches gather per minibatch and drop nothing.** Fancy-indexing a whole batch of these rows
materialises gigabytes; taking the remainder of an epoch as a smaller final batch, weighted by its
true sample count, is the difference between training on every collected timestep and quietly
discarding up to a batch of them per epoch.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
from torch import Tensor

from ..api.buffer import ExperienceBuffer
from ..api.policy import ObsBatch
from ..api.rollout import EPISODE_END_NONE, GROUP_DEAD, GROUP_LEARNER
from ..errors import CheckpointFormatError
from ..rollout.layout import BufferHandle, BufferLayout, buffer_segment_name

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.buffer import ObsCodec
    from ..api.rollout import EnvSpec, RolloutRound, SlotPlan

__all__ = ["AdvantageInputs", "Batch", "Minibatch", "RectBuffer", "batch_count"]


class Minibatch(NamedTuple):
    """One device-resident forward's worth of experience.

    ``weight`` is the minibatch's share of its batch, so that a loss averaged over the minibatch
    and scaled by it accumulates into exactly the full batch's gradient. That is what makes
    ``minibatch_size`` a pure memory knob.
    """

    obs: ObsBatch
    actions: Tensor
    log_probs: Tensor
    advantages: Tensor
    returns: Tensor
    values: Tensor
    #: How many actions each row's mask left, from the column the critic's pass filled. Carried
    #: beside the row rather than recomputed from ``obs.mask`` because the update reads it
    #: before it has decided which rows to unpack at all.
    n_legal: Tensor
    #: Which rows of this minibatch had more than one legal action, as an index into it. Built
    #: on the host from the same column, because reading it off the device -- a ``nonzero`` or
    #: an ``any`` over a mask -- is a synchronisation inside the minibatch loop, and the loop is
    #: where an update's whole wall clock is.
    choice_index: Tensor
    #: How many rows that index holds, as a plain integer for the same reason.
    n_choice: int
    cells: Tensor
    n: int
    weight: float


class Batch:
    """One optimizer step's worth of cells, iterated as device-resident minibatches.

    The cells are gathered a minibatch at a time, when the minibatch is asked for. A batch knows
    how many samples it really holds, which is what lets the last batch of an epoch be a smaller
    one instead of a discarded one.
    """

    def __init__(
        self,
        cells: np.ndarray,
        minibatch_size: int,
        gather: Callable[[np.ndarray, float], Minibatch],
        *,
        n_choice: int,
    ) -> None:
        self.cells = cells
        self.minibatch_size = max(1, int(minibatch_size))
        self._gather = gather
        #: How many of this batch's cells had more than one legal action. A count of the batch
        #: and not of a minibatch, because it is a denominator and a denominator taken per
        #: minibatch would make the gradient a function of the partition.
        self.n_choice = int(n_choice)

    @property
    def n_samples(self) -> int:
        return int(self.cells.size)

    def __len__(self) -> int:
        return (self.n_samples + self.minibatch_size - 1) // self.minibatch_size

    def __iter__(self) -> Iterator[Minibatch]:
        total = self.n_samples
        for start in range(0, total, self.minibatch_size):
            chunk = self.cells[start : start + self.minibatch_size]
            yield self._gather(chunk, chunk.size / total)


class AdvantageInputs(NamedTuple):
    """Exactly the arguments ``AdvantageEstimator.compute`` takes, on the device."""

    rewards: Tensor
    values: Tensor
    final_values: Tensor
    terminated: Tensor
    truncated: Tensor
    trainable: Tensor


class _StagingRing:
    """A ring of pinned host slabs the gather stages through.

    Pinned so that the copy to the device is asynchronous, and a ring so that the transfer of one
    minibatch overlaps the compute of the one before it. Four slabs at the shipped minibatch size
    is a few tens of megabytes, allocated once per run.

    A slab is not handed out again until the copy that last read it has finished. That is the
    whole point of the events: the host loop runs ahead of the device, and a pinned slab rewritten
    while its transfer is still pending gives a minibatch another minibatch's observations beside
    its own actions and advantages -- a wrong importance ratio, with nothing raised anywhere.
    On a device that is not CUDA the copy is synchronous, the slabs are not pinned and there is
    nothing to wait for, so the events are not created at all.
    """

    def __init__(
        self,
        slots: int,
        rows: int,
        frames: int,
        row_bytes: int,
        device: torch.device | None = None,
    ) -> None:
        self.device = device if device is not None else torch.device("cpu")
        pin = self.device.type == "cuda" and torch.cuda.is_available()
        self.slabs = [
            torch.empty((rows, frames, row_bytes), dtype=torch.uint8, pin_memory=pin)
            for _ in range(slots)
        ]
        self.views = [slab.numpy() for slab in self.slabs]
        self.events: list[Any] = [torch.cuda.Event() if pin else None for _ in range(slots)]
        self._next = 0
        self._held = 0

    def next(self) -> tuple[Tensor, np.ndarray]:
        """The next slab, once whatever last read it is done with it."""
        index = self._next
        self._next = (self._next + 1) % len(self.slabs)
        self._held = index
        event = self.events[index]
        if event is not None:
            event.synchronize()
        return self.slabs[index], self.views[index]

    def copied(self) -> None:
        """Called once the copy out of the slab just handed out has been enqueued."""
        event = self.events[self._held]
        if event is not None:
            event.record()


def batch_count(n_samples: int, batch_size: int) -> int:
    """How many optimizer steps one epoch over ``n_samples`` cells should take.

    An epoch used to be cut at every ``batch_size`` rows, which left a remainder batch of
    whatever was over. That batch is not a small step: the loss is a mean over the rows it holds
    and the optimizer takes a step per batch, so a few dozen leftover rows would move the weights
    as far as a full batch did, on a gradient estimated from a fraction of the sample. On the
    shipped laptop profile that was 3 of every 27 steps an iteration, and the count itself moved
    between iterations -- the train session measured 6 to 9 on ITS geometry -- as the number of
    trainable rows crossed a multiple of the batch size, so the number of Adam steps an iteration
    took was not a property of the config at all. That contaminates any comparison between two
    runs, which is what the profiles exist for. The 3-in-27 and 6-to-9 figures come from
    different geometries and neither is the other's arithmetic.

    So the epoch is cut into as many WHOLE batches as it can fill, and the rows over are spread
    one each across those batches rather than made into a short one. Every batch is then at least
    ``batch_size``, the step count is ``n // batch_size`` for any n at or above it, and no row is
    dropped. An epoch smaller than one batch is one batch, which is what it already was.
    """
    return max(1, int(n_samples) // max(1, int(batch_size)))


class RectBuffer(ExperienceBuffer):
    """The shipped experience buffer.

    Owns the shared-memory segment the workers write observation rows into, the scalar columns
    beside it, and the staging ring the learner gathers through. Constructed once per run: the
    rectangle is cleared and reused every iteration, so each timestep is trained on exactly
    ``n_epochs`` times and then discarded, which is what makes the KL and clip-fraction
    diagnostics mean what they say.
    """

    FORMAT_VERSION = 1

    #: What the checkpoint's own files are called.
    STATE_FILE = "buffer.json"
    HISTORY_FILE = "history.npy"

    def __init__(
        self,
        spec: EnvSpec,
        codec: ObsCodec,
        *,
        run_id: str,
        cycles: int,
        n_slots: int,
        device: torch.device | str = "cpu",
        discard_opponent_rows: bool = True,
        staging_slots: int = 4,
        segment_name: str | None = None,
    ) -> None:
        self.spec = spec
        self.codec = codec
        self.device = torch.device(device)
        self.discard_opponent_rows = discard_opponent_rows
        self.capacity = int(cycles)
        self.cycles = int(cycles)
        self.n_slots = int(n_slots)
        self.frame_stack = spec.frame_stack
        self.history_rows = spec.frame_stack - 1
        self.layout = BufferLayout.from_spec(
            spec,
            run_id=run_id,
            cycles=self.capacity,
            n_slots=self.n_slots,
            row_bytes=codec.row_bytes(spec),
            codec_version=codec.codec_version,
        )
        self._name = segment_name or buffer_segment_name(run_id)
        self.shm = shared_memory.SharedMemory(
            name=self._name, create=True, size=self.layout.total_bytes
        )
        self.layout.write_header(self.shm.buf)
        # One row per cell, in the rectangle's own index order, so that gathering a minibatch is
        # a single take along axis zero.
        self.obs_view = np.frombuffer(
            self.shm.buf,
            dtype=np.uint8,
            count=self.layout.obs_bytes,
            offset=self.layout.obs_offset,
        ).reshape(self.layout.rows * self.n_slots, self.layout.row_bytes)

        shape = (self.capacity, self.n_slots)
        self.action = np.zeros(shape, dtype=np.int16)
        self.log_prob = np.zeros(shape, dtype=np.float32)
        self.reward = np.zeros(shape, dtype=np.float32)
        self.advantage = np.zeros(shape, dtype=np.float32)
        self.ret = np.zeros(shape, dtype=np.float32)
        self.final_value = np.zeros(shape, dtype=np.float32)
        self.value = np.zeros((self.capacity + 1, self.n_slots), dtype=np.float32)
        # How many actions the cell's mask left, written once per iteration by the critic's
        # pass over the rectangle. int16 because the count is bounded by the action space and
        # the column is one per cell of a rectangle that is a hundred thousand cells wide.
        self.n_legal = np.zeros(shape, dtype=np.int16)
        self.terminated = np.zeros(shape, dtype=bool)
        self.truncated = np.zeros(shape, dtype=bool)
        self.valid = np.zeros(shape, dtype=bool)
        self.group = np.full(shape, GROUP_DEAD, dtype=np.int8)
        self.deploy_status = np.zeros(shape, dtype=np.int8)
        self.tick = np.zeros(shape, dtype=np.int32)
        # The history strip rides on the front of this one column: the gather needs to know
        # whether a carried-down row belongs to the episode the cell above it is in, and that
        # question is asked of cycles below zero exactly as it is of cycles above them.
        self._episode_end = np.zeros(
            (self.history_rows + self.capacity, self.n_slots), dtype=np.int8
        )
        self._history_valid = np.zeros((self.history_rows, self.n_slots), dtype=bool)

        self.plan: SlotPlan | None = None
        self.iteration = -1
        # Whether this process has collected an iteration whose last cycles are worth carrying
        # down. It is not ``iteration >= 0``: a restored buffer has an iteration number and an
        # empty rectangle, and carrying that rectangle's last cycles would write zeros over the
        # history the checkpoint just restored.
        self._carry_ready = False
        self._static_planes: Tensor | None = None
        self._rounds_recorded = 0
        self._staging_slots = max(1, int(staging_slots))
        self._ring: _StagingRing | None = None
        self._ring_rows = 0

    # -- the segment -------------------------------------------------------

    def shared_handle(self) -> BufferHandle:
        """What a worker is told: the name and the numbers the offsets follow from."""
        return self.layout.handle(self._name)

    def close(self) -> None:
        """Release the segment. Safe to call twice.

        The view over the block goes first: a shared-memory segment refuses to close while a
        buffer is still exported from it, and the view is the only export this object holds.
        """
        self.obs_view = None  # type: ignore[assignment]
        shm, self.shm = getattr(self, "shm", None), None  # type: ignore[assignment]
        if shm is None:
            return
        try:
            shm.close()
        except BufferError:
            # Something outside this object still holds a view over the block. The mapping goes
            # when that view does; closing it under the view's feet is what must not happen.
            return
        with contextlib.suppress(FileNotFoundError, OSError):
            # Unlinking is what a platform with a name for the segment needs and what one
            # without a name does not have.
            shm.unlink()

    # -- the iteration -----------------------------------------------------

    def begin_iteration(self, plan: SlotPlan, cycles: int) -> None:
        """Clear the scalars, carry the history down, and open the iteration.

        The carry happens here rather than at the end of the previous iteration because the rows
        it copies are the ones that were just collected: they are still in place, and copying them
        at the moment the next iteration opens means there is no state to keep between the two.
        """
        if not 1 <= cycles <= self.capacity:
            raise ValueError(
                f"this rectangle holds {self.capacity} cycles and was asked to collect {cycles}"
            )
        if plan.n_slots != self.n_slots:
            raise ValueError(
                f"the plan covers {plan.n_slots} slots and this rectangle has {self.n_slots}"
            )
        self._carry_history()
        self.cycles = int(cycles)
        self.plan = plan
        self.iteration = plan.iteration
        self._rounds_recorded = 0
        for column in (
            self.action,
            self.log_prob,
            self.reward,
            self.advantage,
            self.ret,
            self.final_value,
            self.value,
            self.n_legal,
            self.terminated,
            self.truncated,
            self.valid,
            self.deploy_status,
            self.tick,
        ):
            column.fill(0)
        self.group.fill(GROUP_DEAD)
        self._episode_end[self.history_rows :].fill(EPISODE_END_NONE)
        self._carry_ready = True

    def _carry_history(self) -> None:
        """Copy the last ``k - 1`` collected cycles into the rows below cycle zero.

        Their episode ends come with them, and they are the ends of the rows being copied, not
        of the rows above them: cycle ``T - 1``'s end is the boundary between it and the
        observation the next iteration opens at, which is why row ``-1`` can answer for the
        first cell of the new rectangle at all.
        """
        if self.history_rows == 0 or not self._carry_ready:
            return
        for offset in range(1, self.history_rows + 1):
            source = self.cycles - offset
            if source < 0:
                continue
            target = -offset
            rows = slice(
                self.layout.row_index(source, 0), self.layout.row_index(source, 0) + self.n_slots
            )
            into = slice(
                self.layout.row_index(target, 0), self.layout.row_index(target, 0) + self.n_slots
            )
            self.obs_view[into] = self.obs_view[rows]
            self._episode_end[self.history_rows + target] = self._episode_end[
                self.history_rows + source
            ]
            self._history_valid[self.history_rows + target] = self.valid[source]

    def record_round(
        self,
        r: RolloutRound,
        actions: np.ndarray | None,
        log_probs: np.ndarray | None,
    ) -> None:
        """Write one shard-round's scalars into the rows they describe.

        A round is two timesteps' worth of news. Its observation is the state at its own cycle,
        and the action chosen from it is that row's; its reward, its two done flags, its deploy
        status and its episode end all describe the step that ARRIVED at it, which began one
        cycle earlier. So the arriving half goes to row ``cycle - 1``, beside the action that
        earned it, and the round's own half stays at ``cycle``.

        Both ends of the iteration follow from that and neither is a special case worth a flag.
        The round at cycle zero has no transition behind it, so its arriving half is dropped:
        there is no row below zero for it, and what it carries is the cleared record a worker
        publishes when it re-opens. The round at cycle ``T`` has no row of its own -- ``T`` is
        the bootstrap row, observations only -- but it is the ONLY place the last transition's
        reward and flags exist, and they belong to row ``T - 1``.

        ``actions`` and ``log_probs`` are None for that trailing round, which was never acted
        on.
        """
        cycle = r.cycle
        if not 0 <= cycle <= self.cycles:
            raise IndexError(
                f"cycle {cycle} is outside the {self.cycles} this iteration collects and the "
                "bootstrap cycle above them"
            )
        slots = r.slots
        if cycle > 0:
            acted = cycle - 1
            self.reward[acted, slots] = r.reward
            self.terminated[acted, slots] = r.terminated
            self.truncated[acted, slots] = r.truncated
            self.valid[acted, slots] = r.valid
            self.deploy_status[acted, slots] = r.deploy_status
            self._episode_end[self.history_rows + acted, slots] = r.episode_end
        if cycle < self.cycles:
            self.group[cycle, slots] = r.group
            self.tick[cycle, slots] = r.tick
            if actions is not None:
                self.action[cycle, slots] = actions
            if log_probs is not None:
                self.log_prob[cycle, slots] = log_probs
        self._rounds_recorded += 1

    @property
    def episode_end(self) -> np.ndarray:
        """``(T, R)``: how the episode this cell's action ENDED ended, from the seat's own view.

        A cell that did not end one reads ``EPISODE_END_NONE``, so a non-zero entry at row ``p``
        says two things at once: row ``p`` is the last row of an episode, and the boundary sits
        between ``p`` and ``p + 1``.
        """
        return self._episode_end[self.history_rows :][: self.cycles]

    def set_values(self, values: Tensor) -> None:
        self.value[: self.cycles + 1] = _to_numpy(values, (self.cycles + 1, self.n_slots))

    def set_n_legal(self, counts: Tensor) -> None:
        """How many actions each cell's mask left, from the pass that produced the values.

        Takes the whole ``(T+1, R)`` the critic's pass covers and keeps the collected cycles.
        The bootstrap row is dropped rather than stored: no action was taken in it, so a count
        of what was legal there names nothing the update can ask about.
        """
        shape = (self.cycles + 1, self.n_slots)
        wide = counts.detach().to(device="cpu", dtype=torch.int32).numpy()
        if wide.shape != shape:
            raise ValueError(f"expected {shape} and was given {wide.shape}")
        # The column is int16 to keep the rectangle small, and an action space above 32,767 would
        # wrap into a negative count, which reads as "forced" and quietly drops a choice row from
        # the actor. The shipped space is 2,305, so this is a guard against a future catalogue
        # rather than a live hazard, and a loud one because the failure it prevents is silent.
        largest = int(wide.max()) if wide.size else 0
        if largest > 32_767:
            raise ValueError(
                f"a cell's mask left {largest} actions legal and this column holds up to 32,767; "
                "widen RectBuffer.n_legal before growing the action space"
            )
        array = wide.astype(np.int16)
        self.n_legal[: self.cycles] = array[: self.cycles]

    def set_final_values(self, cells: np.ndarray, values: Tensor) -> None:
        """``V(final_obs)`` for the cells that truncated; ``cells`` is ``(k, 2)`` of
        ``(cycle, slot)``."""
        if cells.size == 0:
            return
        flat = values.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
        self.final_value[cells[:, 0], cells[:, 1]] = flat

    def truncated_cells(self) -> np.ndarray:
        """``(k, 2)`` of ``(cycle, slot)`` for every truncated cell, ascending."""
        rows, columns = np.nonzero(self.truncated[: self.cycles])
        return np.stack([rows, columns], axis=1).astype(np.int64)

    def set_advantages(self, adv: Tensor, ret: Tensor) -> None:
        shape = (self.cycles, self.n_slots)
        self.advantage[: self.cycles] = _to_numpy(adv, shape)
        self.ret[: self.cycles] = _to_numpy(ret, shape)

    # -- what reaches the update -------------------------------------------

    def trainable(self) -> np.ndarray:
        """``(T, R)`` bool, as numpy: the seat's group and the cell's validity, nothing else."""
        group = self.group[: self.cycles]
        if self.discard_opponent_rows:
            learner = group == GROUP_LEARNER
        else:
            learner = group != GROUP_DEAD
        return learner & self.valid[: self.cycles]

    def trainable_mask(self) -> Tensor:
        return torch.from_numpy(self.trainable()).to(self.device)

    def valid_mask(self) -> Tensor:
        return torch.from_numpy(self.valid[: self.cycles].copy()).to(self.device)

    def choice_mask(self) -> Tensor:
        """``(T, R)`` bool: cells whose mask offered more than the no-op.

        Off the column the critic's pass filled, so it is the same classification the minibatches
        carry and not a second opinion about it. It says nothing about whether a cell is
        trainable; a caller that wants both asks for both.
        """
        return torch.from_numpy(self.n_legal[: self.cycles] > 1).to(self.device)

    def advantage_inputs(self) -> AdvantageInputs:
        """The estimator's arguments, on the device, with a dead row's rewards and values
        already zero because nothing ever wrote them."""
        cycles = self.cycles
        return AdvantageInputs(
            rewards=self._tensor(self.reward[:cycles]),
            values=self._tensor(self.value[: cycles + 1]),
            final_values=self._tensor(self.final_value[:cycles]),
            terminated=self._tensor(self.terminated[:cycles]),
            truncated=self._tensor(self.truncated[:cycles]),
            trainable=self.trainable_mask(),
        )

    def _tensor(self, array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    # -- minibatching ------------------------------------------------------

    def batches(
        self,
        batch_size: int,
        minibatch_size: int,
        epochs: int,
        rng_for_epoch: Callable[[int], np.random.Generator],
        *,
        choice_first: bool = False,
    ) -> Iterator[Batch]:
        """Every trainable cell, once per epoch, in batches that never straddle an epoch.

        The permutation of each epoch comes from that epoch's own named stream, so it is a
        function of the run's identity and the epoch number rather than of how many random draws
        happened to have been made before it.

        ``choice_first`` reorders the cells WITHIN each batch so that the ones whose mask offered
        more than the no-op come first, keeping the permutation's order inside each class. It
        moves no cell between batches: every batch holds the cells it held, so every batch-level
        denominator is the one it was, and a caller that skips the forced rows then skips whole
        minibatches of them rather than nine rows in every ten of every minibatch.
        """
        cells = np.flatnonzero(self.trainable().reshape(-1))
        if cells.size == 0:
            # Nothing to train on is no batches, not one empty one. The coordinator's invariants
            # refuse an iteration this short before the update sees it, so this is about the
            # contract rather than about a run: a caller that counts batches should count none.
            return
        self._ensure_ring(minibatch_size)
        has_choice = self.n_legal.reshape(-1) > 1
        for epoch in range(epochs):
            order = cells[rng_for_epoch(epoch).permutation(cells.size)]
            for part in np.array_split(order, batch_count(order.size, batch_size)):
                chose = has_choice[part]
                if choice_first:
                    part = np.concatenate([part[chose], part[~chose]])
                yield Batch(
                    part, minibatch_size, self._gather, n_choice=int(np.count_nonzero(chose))
                )

    def _ensure_ring(self, minibatch_size: int) -> None:
        rows = max(1, int(minibatch_size))
        if self._ring is None or self._ring_rows != rows:
            self._ring = _StagingRing(
                self._staging_slots,
                rows,
                self.frame_stack,
                self.layout.row_bytes,
                self.device,
            )
            self._ring_rows = rows

    def _gather(self, cells: np.ndarray, weight: float) -> Minibatch:
        """One minibatch: stage the rows through a pinned slab, unpack them on the device."""
        cells = np.sort(cells)
        count = int(cells.size)
        cycle = cells // self.n_slots
        slot = cells % self.n_slots
        rows, live = self._stack_rows(cycle, slot)

        assert self._ring is not None  # batches() allocates it before yielding anything
        slab, view = self._ring.next()
        np.take(
            self.obs_view,
            rows.reshape(-1),
            axis=0,
            out=view[:count].reshape(count * self.frame_stack, self.layout.row_bytes),
        )
        if not live.all():
            staged = view[:count]
            staged[~live] = 0
        raw = slab[:count].to(self.device, non_blocking=True)
        self._ring.copied()

        obs = self._empty_obs(count)
        statics = self._statics()
        self.codec.unpack_to_device(raw, statics, obs)
        # After the sort, so the index names rows of this minibatch as it is handed over.
        legal = np.take(self.n_legal.reshape(-1), cells)
        choice = np.flatnonzero(legal > 1).astype(np.int64)
        return Minibatch(
            obs=obs,
            actions=self._gather_column(self.action, cells, torch.int64),
            log_probs=self._gather_column(self.log_prob, cells, torch.float32),
            advantages=self._gather_column(self.advantage, cells, torch.float32),
            returns=self._gather_column(self.ret, cells, torch.float32),
            values=self._gather_column(self.value[: self.cycles], cells, torch.float32),
            n_legal=torch.from_numpy(legal.astype(np.int64)).to(self.device),
            choice_index=torch.from_numpy(choice).to(self.device),
            n_choice=int(choice.size),
            cells=torch.from_numpy(cells.astype(np.int64)).to(self.device),
            n=count,
            weight=weight,
        )

    def _stack_rows(self, cycle: np.ndarray, slot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The rectangle rows of each cell's frame stack, and which of them are live.

        Frame ``j`` is the row ``j`` cycles earlier in the same slot. It is live while no episode
        ended between it and the cell, while it was collected at all, and -- below cycle zero --
        while there was a previous iteration to carry it down from.

        "No episode ended between them" is asked of the frame's OWN row, because
        ``episode_end[p]`` is the boundary after row ``p``: a frame that ended an episode is the
        last row of that episode, and the cell above it opened the next one. Asking it of the
        row above instead is the same question about the wrong boundary -- it zeroes the history
        of the second row of every episode and stacks the tail of the previous battle behind the
        first.
        """
        count = cycle.size
        frames = self.frame_stack
        rows = np.empty((count, frames), dtype=np.int64)
        live = np.zeros((count, frames), dtype=bool)
        base = self.history_rows * self.n_slots + slot
        live[:, 0] = True
        for j in range(frames):
            rows[:, j] = base + (cycle - j) * self.n_slots
            if j == 0:
                continue
            previous = cycle - j
            ended = self._episode_end[self.history_rows + previous, slot] != EPISODE_END_NONE
            collected = self.valid[np.clip(previous, 0, self.cycles - 1), slot]
            carried = self._history_valid[
                np.clip(self.history_rows + previous, 0, self.history_rows - 1), slot
            ]
            available = np.where(previous >= 0, collected, carried)
            live[:, j] = live[:, j - 1] & ~ended & available
        return rows, live

    def _gather_column(self, column: np.ndarray, cells: np.ndarray, dtype: torch.dtype) -> Tensor:
        taken = np.take(column.reshape(-1), cells)
        return torch.from_numpy(np.ascontiguousarray(taken)).to(device=self.device, dtype=dtype)

    def _empty_obs(self, count: int) -> ObsBatch:
        """The tensors one minibatch is unpacked into.

        Allocated per minibatch rather than reused from a ring: the device copy allocates anyway,
        the caching allocator hands back the same blocks, and a reused tensor would be overwritten
        while a backward pass still held it.
        """
        spec = self.spec
        frames = self.frame_stack
        planes, tiles_y, tiles_x = spec.spatial_shape
        return ObsBatch(
            spatial=torch.empty(
                (count, frames * planes, tiles_y, tiles_x),
                dtype=torch.float32,
                device=self.device,
            ),
            mask_planes=torch.empty(
                (count, frames * spec.hand_size, tiles_y, tiles_x),
                dtype=torch.float32,
                device=self.device,
            ),
            vector=torch.empty((count, spec.vector_size), dtype=torch.float32, device=self.device),
            mask=torch.empty((count, spec.n_actions), dtype=torch.bool, device=self.device),
        )

    def _statics(self) -> Tensor:
        if self._static_planes is None:
            raise RuntimeError(
                "the static planes were never set; preflight reads them once off a real "
                "observation and hands them to the buffer with set_static_planes()"
            )
        return self._static_planes

    def set_static_planes(self, planes: np.ndarray | Tensor) -> None:
        """The declared-static planes, read once at preflight and held for the run.

        One copy, not one per seat: the observation is in the acting player's own frame and the
        arena is symmetric under the seat rotation, which preflight asserts rather than assumes.
        """
        if isinstance(planes, np.ndarray):
            planes = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32))
        self._static_planes = planes.to(device=self.device, dtype=torch.float32)

    # -- checkpoint --------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        """The rectangle itself is not checkpointed; the history it carries is.

        The iteration boundary is the resume point, so there is no mid-iteration experience worth
        preserving. The ``k - 1`` history rows are the one thing that crosses a boundary, and
        without them the first cycle after a resume would see a zero stack where the run it
        continues saw a full one.
        """
        folder.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "cycles": self.cycles,
            "n_slots": self.n_slots,
            "row_bytes": self.layout.row_bytes,
            "frame_stack": self.frame_stack,
            "codec_version": self.codec.codec_version,
            "iteration": self.iteration,
        }
        if self.history_rows:
            # The bytes alone are not the history. A carried row is stacked only where the
            # gather can see that it was collected and that no episode ended on it, so saving
            # the rows without those two columns saves something that restores as a zero.
            state["history_valid"] = [int(flag) for flag in self._history_valid.reshape(-1)]
            state["history_episode_end"] = [
                int(end) for end in self._episode_end[: self.history_rows].reshape(-1)
            ]
        (folder / self.STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        if self.history_rows:
            rows = self.history_rows * self.n_slots
            np.save(folder / self.HISTORY_FILE, np.asarray(self.obs_view[:rows]))

    def load_checkpoint(self, folder: Path, *, strict: bool) -> None:
        path = folder / self.STATE_FILE
        if not path.exists():
            if strict:
                raise CheckpointFormatError(f"the buffer's state is not at {path}")
            print(f"no buffer state at {path}; the next iteration starts with a zero history")
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        version = int(state.get("format_version", 0))
        if version > self.FORMAT_VERSION:
            raise CheckpointFormatError(
                f"{path} was written at buffer format {version} and this build reads "
                f"{self.FORMAT_VERSION}"
            )
        for field, mine in (
            ("n_slots", self.n_slots),
            ("row_bytes", self.layout.row_bytes),
            ("frame_stack", self.frame_stack),
            ("codec_version", self.codec.codec_version),
        ):
            if state.get(field) != mine:
                raise CheckpointFormatError(
                    f"{path} records {field}={state.get(field)} and this buffer is {mine}"
                )
        self.iteration = int(state["iteration"])
        history = folder / self.HISTORY_FILE
        if self.history_rows and history.exists():
            rows = self.history_rows * self.n_slots
            self.obs_view[:rows] = np.load(history, allow_pickle=False)
            shape = (self.history_rows, self.n_slots)
            valid = state.get("history_valid")
            if valid is not None:
                self._history_valid[:] = np.asarray(valid, dtype=bool).reshape(shape)
            ends = state.get("history_episode_end")
            if ends is not None:
                self._episode_end[: self.history_rows] = np.asarray(
                    ends, dtype=np.int8
                ).reshape(shape)
        # Nothing has been collected in this process, so the first iteration to open must not
        # copy this rectangle's own last cycles -- which are empty -- over what was restored.
        self._carry_ready = False


def _to_numpy(values: Tensor, shape: tuple[int, int]) -> np.ndarray:
    array = values.detach().to(device="cpu", dtype=torch.float32).numpy()
    if array.shape != shape:
        raise ValueError(f"expected {shape} and was given {array.shape}")
    return array
