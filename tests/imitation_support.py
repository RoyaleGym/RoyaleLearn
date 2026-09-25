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
from royalelearn.artifacts import ProbeSet, probe_log_probs, write_actor_artifact
from royalelearn.testing import actor_state, learner_rows


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
    actor = run.build_actor(run.device)
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


#: The keys of the pre-extension ``imitation`` block that moved to the ``warm_start`` section.
WARM_START_KEYS = ("init", "actor_lr_scale")


def with_imitation(config: cfg.RunConfig, **block: Any) -> cfg.RunConfig:
    """``config`` with the IL sections built from keyword arguments.

    ``init`` and ``actor_lr_scale`` go to ``warm_start`` and everything else to ``imitation``;
    a section with no key given is left out, as a run without it has none.
    """
    from royalelearn.extensions import with_sections

    warm = {key: block.pop(key) for key in WARM_START_KEYS if key in block}
    return with_sections(config, warm_start=warm or None, imitation=block or None)
