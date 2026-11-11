"""``rollout.cuda_graphs``: the rollout's policy forwards replayed as CUDA graphs.

A rollout forward is a few rows through a deep network, and run op by op its cost is the host
launching a few hundred kernels, not the device running them. A CUDA graph records one forward
once and replays the whole of it in one launch. A graph is a fixed shape, so a forward of ``n``
rows runs at the smallest of ``ROW_BUCKETS`` that holds them: the rows are copied into the
first ``n`` places of buffers the graph owns, and the logits of those rows are read back out.
The places after them hold whatever earlier forwards left there, every one a real row with its
no-op legal; the network computes each row from its own inputs, so they do not touch the rows
read out. More than the largest bucket runs as it always did.

One graph per policy and bucket: the learner's (its live actor, or the overlap's behaviour
snapshot, captured again when that changes) and each frozen opponent's. A graph holds its actor:
replaying one whose weights had been freed would read freed memory. The learner's graph keeps
working across updates, because the optimizer changes the weights in place. Only the forward is
captured; the distribution, the sample and the statistics after it run as before, so the
learner's statistics are exactly what they were.

On a device other than CUDA there is no graph, and ``GraphedForward(capture=False)`` is the same
buffers and the same padding called directly: what the suite runs the path through.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from royalegym.action import NOOP

if TYPE_CHECKING:
    from torch import Tensor

    from ..api.policy import ObsBatch

__all__ = ["ROW_BUCKETS", "GraphedForward", "bucket"]

#: The batch sizes a forward is captured at.
ROW_BUCKETS = (8, 16, 32, 64, 128, 256, 512)


def bucket(rows: int) -> int | None:
    """The smallest bucket that holds ``rows``, or None when none does."""
    return next((size for size in ROW_BUCKETS if size >= rows), None)


class GraphedForward:
    """``fn`` of an ``ObsBatch``, captured at ``rows`` rows, with buffers of its own.

    ``keep`` is what ``fn`` reads (its actor), held for as long as the graph is: a graph does
    not hold the tensors it was captured on.
    """

    __slots__ = ("fn", "graph", "inputs", "keep", "output", "rows")

    def __init__(
        self,
        fn: Callable[[ObsBatch], Tensor],
        template: ObsBatch,
        rows: int,
        *,
        keep: Any,
        capture: bool,
        pool: Any = None,
    ) -> None:
        import torch

        from ..api.policy import ObsBatch as _Obs
        from .actor_critic import GRAPH_CAPTURE

        self.fn = fn
        self.keep = keep
        self.rows = int(rows)
        self.inputs = _Obs(
            *(
                None
                if t is None
                else torch.zeros((self.rows, *t.shape[1:]), dtype=t.dtype, device=t.device)
                for t in template
            )
        )
        # A place no forward has filled yet is a row whose only legal action is the no-op.
        self.inputs.mask[:, NOOP] = True
        self._fill(template)
        self.graph: Any = None
        self.output: Tensor | None = None
        if not capture:
            return
        current = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(current)
        # Warmed up off the capture: cuDNN picks its algorithms, the allocator grows, and a
        # check a function runs on its first batch runs here rather than inside the graph.
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):
                fn(self.inputs)
        graph = torch.cuda.CUDAGraph()
        token = GRAPH_CAPTURE.set(True)
        # Captured with the graph's own begin and end rather than ``torch.cuda.graph``, whose
        # entry synchronises the whole device and empties the allocator's cache, and in
        # thread-local mode. Under rollout.overlap the capture runs on the collection thread
        # while the update runs on the main one: a device-wide synchronise there waits on the
        # update, an emptied cache sends the update's next allocation to cudaMalloc, and in the
        # default global mode a cudaMalloc from any other thread during a capture is an error.
        # Thread-local mode forbids only this thread's unsafe calls, and the capture's own
        # stream (``side``, which has waited on the current one) orders it after the warm-up.
        try:
            with torch.no_grad(), torch.cuda.stream(side):
                graph.capture_begin(pool=pool, capture_error_mode="thread_local")
                try:
                    self.output = fn(self.inputs)
                finally:
                    graph.capture_end()
        finally:
            GRAPH_CAPTURE.reset(token)
        current.wait_stream(side)
        self.graph = graph

    def _fill(self, obs: ObsBatch) -> int:
        count = int(obs.mask.shape[0])
        for static, value in zip(self.inputs, obs, strict=True):
            if static is not None and value is not None:
                static[:count].copy_(value)
        return count

    def __call__(self, obs: ObsBatch) -> Tensor:
        """``fn(obs)`` for ``obs``'s rows, through the graph."""
        import torch

        count = self._fill(obs)
        if self.graph is None:
            with torch.no_grad():
                self.output = self.fn(self.inputs)
        else:
            self.graph.replay()
        assert self.output is not None
        return self.output[:count]
