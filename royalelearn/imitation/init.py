"""What a run does with its ``imitation`` block before the first iteration (section 19.4).

Two things, in this order. At every start, fresh or resumed, every folder the block names is
hashed and compared with the digest the config states for it: the identity carries the stated
digest, so this is what makes the identity true of the files. Then, on a fresh start only, the
actor is loaded from ``imitation.init`` and made to reproduce the log-probabilities recorded on
the artifact's own probe rows. On a resume the checkpoint's weights win.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import PreflightError
from .artifacts import (
    check_actor_artifact,
    load_actor_state,
    read_actor_artifact,
    self_test,
    verify_artifact,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..config import ImitationConfig, InitSpec
    from ..ladder.snapshots import SnapshotSpec
    from .rows import RowCodec

__all__ = ["initialise_actor", "verify_imitation_files"]


def verify_imitation_files(imitation: ImitationConfig | None) -> dict[str, Path]:
    """Every folder the block names, checked against its stated digest. Keyed by role.

    All of them before any refusal is raised, so a config with two stale digests is told about
    both at once.
    """
    if imitation is None:
        return {}
    named: list[tuple[str, str, str]] = []
    if imitation.init is not None:
        named.append(("imitation.init", imitation.init.path, imitation.init.sha256))
    for name, reference in sorted(imitation.references.items()):
        named.append((f"imitation.references.{name}", reference.path, reference.sha256))
    found: dict[str, Path] = {}
    problems: list[str] = []
    for role, path, digest in named:
        try:
            found[role] = verify_artifact(path, digest, what=role)
        except PreflightError as exc:
            problems.append(str(exc))
    if problems:
        raise PreflightError("\n".join(problems))
    return found


def initialise_actor(
    model: Any,
    init: InitSpec,
    *,
    current: SnapshotSpec,
    codec: RowCodec,
    printer: Callable[[str], None] | None = print,
) -> float:
    """Load the init artifact into ``model.actor`` and run the probe-logit self-test.

    Returns the self-test's largest difference. The critic is left exactly as the seeded build
    made it, which is what makes a run initialised this way comparable with one started from
    scratch at the same seed: the two differ in the actor's starting weights and in nothing else.
    """
    what = "imitation.init"
    artifact = read_actor_artifact(init.path)
    check_actor_artifact(artifact, current, what=what)
    if artifact.probe is None:
        raise PreflightError(
            f"{what}: {init.path} has no probe rows. An init is refused without them: the "
            "self-test on them is the only check that the network the weights were loaded into "
            "computes the function that was saved"
        )
    load_actor_state(model.actor, artifact, what=what)
    worst = self_test(model.actor, codec, artifact.probe, atol=init.self_test_atol, what=what)
    if printer is not None:
        printer(
            f"imitation     actor initialised from {init.path} ({init.sha256[:12]}); probe "
            f"self-test on {artifact.probe.rows.shape[0]} rows, largest difference {worst:.3g}"
        )
    return worst
