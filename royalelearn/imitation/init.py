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
    PROBE_CHUNK,
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

__all__ = ["initialise_actor", "loaded_ratio_guard", "verify_imitation_files"]


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


def loaded_ratio_guard(
    model: Any,
    codec: RowCodec,
    rows: Any,
    *,
    atol: float,
    precision: str,
    say: Callable[[str], None],
) -> float:
    """Section 19.9: the ratio guard of preflight, with p_max and |logit| MEASURED.

    Preflight predicts the importance ratio's arithmetic floor from ``net.noop_bias``, which is
    what a seeded actor's largest probability is. A loaded actor's is its own -- a cloned policy
    that holds on most rows puts far more than the bias's share on the no-op -- so the prediction
    is made from the actor itself, on the artifact's probe rows. Returns the prediction.
    """
    import torch

    from ..rollout.preflight import judge_ratio_precision, precision_bound

    p_max = 0.0
    magnitude = 0.0
    with torch.no_grad():
        for start in range(0, int(rows.shape[0]), PROBE_CHUNK):
            obs = codec.decode(rows[start : start + PROBE_CHUNK])
            logits = model.actor.logits(obs).float()
            legal = obs.mask
            probs = model.actor.distribution(obs).log_probs.exp()
            p_max = max(p_max, float(torch.where(legal, probs, torch.zeros_like(probs)).max()))
            magnitude = max(
                magnitude,
                float(torch.where(legal, logits.abs(), torch.zeros_like(logits)).max()),
            )
    predicted = precision_bound(p_max=p_max, magnitude=magnitude, dtype_name=precision)
    say(
        f"ratio guard   {precision} measured on {int(rows.shape[0])} probe rows: p_max "
        f"{p_max:.4f}, largest legal |logit| {magnitude:.3g}, predicts a deviation of "
        f"{predicted:.2e} against ppo.ratio_atol {atol:g}"
    )
    trouble = (
        f"ppo.ratio_atol[{precision!r}] is {atol:g}, and the loaded actor's own arithmetic "
        f"predicts {predicted:.2e}: it puts up to {p_max:.4f} of a row's mass on one action, and "
        f"an error in that action's logit is multiplied into every other action's "
        f"log-probability by that share. Set net.autocast_dtype to float32"
    )
    judge_ratio_precision(predicted, atol, trouble=trouble, say=say)
    return predicted
