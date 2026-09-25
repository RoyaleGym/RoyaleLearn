"""Artifacts the imitation tests need, made from a real run's own actor and codec.

Everything here goes through the run: the rows are ones its codec packed, the log-probabilities
are the ones its actor produces on them, and the spec is its snapshot template. An artifact
assembled by hand would test the hand.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from royalelearn import config as cfg
from royalelearn.imitation.artifacts import ProbeSet, probe_log_probs, write_actor_artifact


def learner_rows(run: Any, count: int) -> np.ndarray:
    """Up to ``count`` packed rows the run collected for its own seats, after an iteration."""
    buffer = run.buffer
    cycles, slots = np.nonzero(buffer.trainable())
    take = min(count, int(cycles.size))
    rows, _live = buffer._stack_rows(cycles[:take], slots[:take])
    return np.ascontiguousarray(buffer.obs_view[rows[:, 0]])


def actor_state(run: Any) -> dict[str, torch.Tensor]:
    """A detached copy of the run's actor tensors, as they are right now."""
    return {name: tensor.detach().clone() for name, tensor in run.model.actor.state_dict().items()}


def write_from_run(
    run: Any,
    folder: Path,
    state: Mapping[str, torch.Tensor],
    rows: np.ndarray,
    *,
    edit: Callable[[dict[str, torch.Tensor]], None] | None = None,
    probe: bool = True,
) -> str:
    """An actor artifact holding ``state``, with probe log-probabilities computed on ``rows``
    by a bare actor of the run's architecture loaded with that state.

    ``edit`` changes the tensors AFTER the probe was recorded, which is what a file that no
    longer matches its own probe looks like.
    """
    actor = run._build_actor(run.device)
    actor.load_state_dict(dict(state))
    actor.eval()
    log_probs, _mask = probe_log_probs(actor, run.row_codec(), rows)
    tensors = {name: tensor.clone() for name, tensor in state.items()}
    if edit is not None:
        edit(tensors)
    return write_actor_artifact(
        folder,
        tensors,
        run.snapshot_template,
        probe=ProbeSet(rows=rows, log_probs=log_probs) if probe else None,
    )


def seeded_artifact(
    config: cfg.RunConfig,
    folder: Path,
    coordinator: Callable[..., Any],
    *,
    edit: Callable[[dict[str, torch.Tensor]], None] | None = None,
    probe: bool = True,
) -> str:
    """An artifact holding the actor exactly as ``config``'s seeded build makes it."""
    with coordinator(config) as run:
        state = actor_state(run)
        run.iterate()
        rows = learner_rows(run, 12)
        return write_from_run(run, folder, state, rows, edit=edit, probe=probe)


def with_imitation(config: cfg.RunConfig, **block: Any) -> cfg.RunConfig:
    """``config`` with an ``imitation`` block built from keyword arguments."""
    import msgspec

    return msgspec.structs.replace(
        config, imitation=msgspec.convert(block, type=cfg.ImitationConfig)
    )
