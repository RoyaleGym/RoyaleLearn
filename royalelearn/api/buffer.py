"""How an observation is stored, and the rectangle it is stored in.

The two are one subject. The worker packs a row straight into its final resting place in the
experience buffer, so there is no intermediate copy to agree about; the learner unpacks it on
the device inside the same kernel that gathers the frame stack. What the codec may decide is
how each key is stored, and that is decided from the observation space at preflight rather than
from a list of plane indices written down here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

import msgspec
import numpy as np

from .checkpoint import Checkpointable

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor

    from ..learn.buffer import Batch
    from ..rollout.layout import BufferHandle
    from .policy import ObsBatch
    from .rollout import EnvSpec, RolloutRound, SlotPlan

__all__ = ["MIN_TABLE_STATES", "CodecTable", "ExperienceBuffer", "ObsCodec"]

#: How many real observations a codec table may be decided from. Storage is decided per plane
#: from what the sample contains, so the sample has to be large enough, and played rather than
#: idle, to have reached the states in which a plane takes the values that decide it.
MIN_TABLE_STATES = 1000


class CodecTable(msgspec.Struct, frozen=True, omit_defaults=True):
    """How each observation key is stored, decided from ``EnvSpec.obs_space`` at preflight
    rather than from a list of plane indices. Logged, hashed and written into every snapshot.

    ``plane`` is one entry per spatial plane: its name, its storage in
    {"uint8", "float16", "static", "derived"}, and the divisor that takes the stored integer
    back to the value the environment produced. Two runs whose tables differ are not comparable,
    and ``digest()`` is what says so.
    """

    plane: tuple[tuple[str, str, float], ...]
    vector: str
    mask: str
    #: How ``card_ids`` is stored: "uint8", exact, or None when the observation has none. It is
    #: omitted from the encoding when None (``omit_defaults``), so every table decided before
    #: card identity existed hashes exactly as it did and no saved run's digest moves.
    ids: str | None = None

    def digest(self) -> str:
        """sha256 of the canonical JSON of this table."""
        from ..rollout.envspec import digest_of

        return digest_of(self)


class ObsCodec(ABC):
    """Quantisation of one observation row. The worker packs; the learner unpacks on the GPU.

    Implementers MUST be exact round-trips for the integer-valued channels and MUST declare
    ``row_bytes`` as a constant given an ``EnvSpec`` and its ``CodecTable``.
    """

    @abstractmethod
    def table(
        self,
        spec: EnvSpec,
        sample: Sequence[dict[str, np.ndarray]],
        *,
        min_states: int = MIN_TABLE_STATES,
    ) -> CodecTable:
        """Decide storage per key from the declared bounds and a sample of real observations.

        Storage is decided from a sample; EXISTENCE is decided from the declaration. A plane the
        layout does not declare static is stored even when it is constant across the sample,
        because the tower planes are constant in any sample in which no tower falls.

        Implementers MUST refuse a sample of fewer than ``min_states`` observations. The row
        size of the whole run follows from this one decision, and a sample too small or too
        idle to have reached the states a plane varies in decides it wrongly and in silence.
        """

    @abstractmethod
    def row_bytes(self, spec: EnvSpec) -> int: ...

    @abstractmethod
    def pack(self, obs: dict[str, np.ndarray], out: memoryview, row: int) -> None: ...

    @abstractmethod
    def static_planes(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """The planes ``EnvSpec.spatial_layout`` declares static; stored once per seat, never
        per row."""

    @abstractmethod
    def unpack_to_device(self, raw: Tensor, statics: Tensor, out: ObsBatch) -> None:
        """Dequantise, scatter the static planes in, reshape the stored mask into the mask
        planes, and gather the frame-stack history."""

    @property
    @abstractmethod
    def codec_version(self) -> int:
        """The RULE's version. The table it produces is data and travels separately."""


class ExperienceBuffer(ABC, Checkpointable):
    """A rectangle of ``(T + frame_stack)`` cycles x ``R`` slots: ``T`` collected cycles, one
    bootstrap row, and ``frame_stack - 1`` history rows carried over from the previous
    iteration. Owns the shared-memory block the workers write into.

    Implementers may assume each ``(cycle, slot)`` cell is written exactly once, by the worker
    that owns that slot; the learner only reads.
    """

    @abstractmethod
    def shared_handle(self) -> BufferHandle:
        """Name, size and the numbers the offsets follow from; picklable, and sent to workers."""

    @abstractmethod
    def begin_iteration(self, plan: SlotPlan, cycles: int) -> None: ...

    @abstractmethod
    def record_round(
        self, r: RolloutRound, actions: np.ndarray, log_probs: np.ndarray
    ) -> None:
        """Scalars only: observations are already in place. O(n), no observation copy."""

    @abstractmethod
    def set_values(self, values: Tensor) -> None:
        """``(T+1, R)`` float32, from the whole-iteration critic pass."""

    @abstractmethod
    def set_n_legal(self, counts: Tensor) -> None:
        """``(T+1, R)`` integer: how many actions each cell's mask left, from the critic's pass.

        Implementers keep the collected cycles and may drop the bootstrap row, in which no
        action was taken. The count is what tells the update which rows had a choice at all
        without unpacking an observation to find out.
        """

    @abstractmethod
    def set_final_values(self, cells: np.ndarray, values: Tensor) -> None:
        """V(final_obs) for truncated cells; ``cells`` is int64[(k, 2)] of (cycle, slot)."""

    @abstractmethod
    def set_advantages(self, adv: Tensor, ret: Tensor) -> None:
        """``(T, R)`` float32 each."""

    @abstractmethod
    def trainable_mask(self) -> Tensor:
        """``(T, R)`` bool: which cells reach the update. What decides it is the seat's group
        and the cell's validity, never where the row was written."""

    @abstractmethod
    def batches(
        self,
        batch_size: int,
        minibatch_size: int,
        epochs: int,
        rng_for_epoch: Callable[[int], np.random.Generator],
        *,
        choice_first: bool = False,
    ) -> Iterator[Batch]:
        """Yield batches; each Batch knows its true sample count and iterates device-resident
        minibatches. A batch never straddles an epoch boundary, and an epoch holds as many whole
        batches as it can fill with the rows over spread one each across them, so every batch is
        at least ``batch_size`` and an epoch is exactly ``n // batch_size`` optimizer steps. An
        epoch with nothing trainable in it yields no batches at all. Each minibatch is weighted by
        its share of its own batch. Gathers per MINIBATCH, never per batch.

        ``choice_first`` reorders each batch's cells so that the ones with more than one legal
        action come first, keeping the permutation's order inside each class. Implementers MUST
        move no cell between batches: it is a reordering, so that a caller skipping the forced
        rows skips whole minibatches of them, and every batch-level denominator is unchanged."""
