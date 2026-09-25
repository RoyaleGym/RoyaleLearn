"""Codec rows, packed the way the rollout packs them and decoded the way the update decodes them.

A cloned actor, a probe set and a demonstration shard all hand the network observations that did
not come through the rollout buffer. If they reached the network by any other path -- the float32
observation straight from the environment, say -- the network would be trained or tested on
inputs a few quanta away from the ones it acts on, and nothing would say so. So they go through
the run's own codec, pack then unpack, which is exactly the path a rollout row takes into the
update.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.policy import ObsBatch
    from ..api.rollout import EnvSpec, ObsCodec

__all__ = ["RowCodec"]


class RowCodec:
    """One run's codec, its static planes and its device, as a pack and a decode.

    Single frames only: a row here is one observation, and a stacked history is something the
    rollout buffer assembles from consecutive cells, which none of these callers have.
    """

    def __init__(self, spec: EnvSpec, codec: ObsCodec, statics: Any, device: Any) -> None:
        import torch

        self.spec = spec
        self.codec = codec
        self.device = torch.device(device)
        self.row_bytes = int(codec.row_bytes(spec))
        self.statics = torch.as_tensor(np.asarray(statics), dtype=torch.float32).to(self.device)

    def pack(self, observations: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
        """``(N, row_bytes)`` uint8: each observation as the rollout would have written it."""
        out = np.zeros((len(observations), self.row_bytes), dtype=np.uint8)
        view = memoryview(out.reshape(-1))
        for row, obs in enumerate(observations):
            self.codec.pack(dict(obs), view, row)
        return out

    def decode(self, rows: np.ndarray | Any) -> ObsBatch:
        """The batch the networks read, from ``(N, row_bytes)`` packed rows."""
        import torch

        from ..learn.buffer import empty_obs

        raw = torch.as_tensor(np.ascontiguousarray(rows, dtype=np.uint8)).to(self.device)
        if raw.ndim != 2 or raw.shape[1] != self.row_bytes:
            raise ValueError(
                f"expected (N, {self.row_bytes}) packed rows for this run's codec, got "
                f"{tuple(raw.shape)}"
            )
        out = empty_obs(self.spec, int(raw.shape[0]), frames=1, device=self.device)
        self.codec.unpack_to_device(raw, self.statics, out)
        return out
