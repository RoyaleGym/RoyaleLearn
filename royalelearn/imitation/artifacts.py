"""The folders a cloned actor and a field model live in, and how a config names them.

Section 19.2 of ``docs/harness-spec.md``. An artifact is a folder, and a config names it by a
path and a digest. The digest is over every file in the folder, not over the weights alone,
because the other files change numbers too: a field model's normalisation is in its
``spec.json``, and an init's self-test is in its probe rows.

An **actor artifact** is the snapshot layout of section 11.6 -- ``actor.safetensors`` beside a
``SnapshotSpec`` -- with the weights in float32 rather than half precision, and a probe set when
it is meant to be loaded as an init. A pool snapshot is played against and never trained from;
an init is trained from, and a file that rounded the seeded weights could not reproduce a run
started from them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec
import numpy as np

from ..errors import PreflightError
from ..ladder.snapshots import SPEC_NAME, WEIGHTS_NAME, SnapshotSpec
from ..rollout.envspec import digest_of

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor

    from ..api.policy import ObsBatch
    from .rows import RowCodec

__all__ = [
    "FIELD_MODEL_NAME",
    "PROBE_CHUNK",
    "PROBE_NAME",
    "ActorArtifact",
    "FieldModelSpec",
    "ProbeSet",
    "artifact_digest",
    "build_field_mlp",
    "check_actor_artifact",
    "load_actor_state",
    "probe_log_probs",
    "read_actor_artifact",
    "read_field_model",
    "self_test",
    "verify_artifact",
    "write_actor_artifact",
    "write_field_model",
]

PROBE_NAME = "probe.safetensors"
FIELD_MODEL_NAME = "model.safetensors"
#: The forward the probe log-probabilities are taken in, on both sides. A batch's shape can
#: choose the kernel a convolution runs with, and a self-test that compared a 1,024-row forward
#: against a 256-row one would be measuring that choice.
PROBE_CHUNK = 256


def artifact_digest(folder: str | Path) -> str:
    """sha256 of the canonical JSON of ``{relative path: sha256 of the file}`` over the folder.

    Every file, at any depth, named by its path relative to the folder with forward slashes, so
    the digest is the same on every platform and does not depend on where the folder sits.
    """
    root = Path(folder)
    if not root.is_dir():
        raise PreflightError(f"no artifact folder at {root}")
    listing = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not listing:
        raise PreflightError(f"the artifact folder {root} is empty")
    return digest_of(listing)


def verify_artifact(path: str | Path, stated: str, *, what: str) -> Path:
    """The folder, if its digest is the one the config states; a refusal naming both if not.

    Checked at every start, fresh or resumed. The identity carries the stated digest, so without
    this the identity is only as true as the config's claim about the file.
    """
    folder = Path(path)
    found = artifact_digest(folder)
    if found != stated:
        raise PreflightError(
            f"{what}: the artifact at {folder} has digest {found}, and the config states "
            f"{stated}. Either the folder changed since the config was written or the config "
            "names the wrong one; `royalelearn artifact-digest <folder>` prints a folder's digest"
        )
    return folder


# --------------------------------------------------------------------------
# Actor artifacts
# --------------------------------------------------------------------------


class ProbeSet(NamedTuple):
    """Rows packed by the run's codec, and the masked log-probabilities recorded on them."""

    rows: np.ndarray
    log_probs: np.ndarray


class ActorArtifact(NamedTuple):
    folder: Path
    spec: SnapshotSpec
    state: dict[str, Tensor]
    probe: ProbeSet | None


def probe_log_probs(
    actor: Any, codec: RowCodec, rows: np.ndarray, *, chunk: int = PROBE_CHUNK
) -> tuple[np.ndarray, np.ndarray]:
    """``(N, n_actions)`` float32 masked log-probabilities and the ``(N, n_actions)`` mask.

    The one function both sides call: the tool that writes an artifact records what it returns,
    and the self-test compares against what it returns now. Two implementations of "the
    log-probabilities on these rows" would be two answers.
    """
    import torch

    out: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(rows.shape[0]), chunk):
            obs: ObsBatch = codec.decode(rows[start : start + chunk])
            distribution = actor.distribution(obs)
            out.append(distribution.log_probs.float().cpu().numpy())
            masks.append(obs.mask.cpu().numpy())
    if not out:
        width = codec.spec.n_actions
        return np.zeros((0, width), np.float32), np.zeros((0, width), bool)
    return np.concatenate(out), np.concatenate(masks)


def self_test(actor: Any, codec: RowCodec, probe: ProbeSet, *, atol: float, what: str) -> float:
    """The probe-logit self-test of section 19.4. Returns the largest difference it found.

    A compatible spec says the tensors have the right shapes; only a forward on known rows says
    the network computes the function that was saved. Compared on the legal actions only: an
    illegal action's log-probability is the fill value's residue and carries nothing.
    """
    if probe.rows.shape[0] == 0:
        raise PreflightError(f"{what}: the artifact's probe set is empty")
    measured, mask = probe_log_probs(actor, codec, probe.rows)
    if measured.shape != probe.log_probs.shape:
        raise PreflightError(
            f"{what}: the probe records log-probabilities of shape {probe.log_probs.shape} and "
            f"this run's actor produces {measured.shape}"
        )
    difference = np.where(mask, np.abs(measured - probe.log_probs), 0.0)
    worst = float(difference.max())
    if not worst <= atol:
        row = int(np.unravel_index(int(difference.argmax()), difference.shape)[0])
        raise PreflightError(
            f"{what}: the probe-logit self-test failed. On probe row {row} a legal action's "
            f"log-probability differs by {worst:.3g} from the one recorded when the artifact "
            f"was written, against a tolerance of {atol:g}. The weights loaded, so the network "
            "they were loaded into does not compute the function that was saved: a different "
            "architecture behind the same digest, a codec that decodes the rows differently, or "
            "a different precision or device"
        )
    return worst


def write_actor_artifact(
    folder: str | Path,
    state: Mapping[str, Tensor],
    spec: SnapshotSpec,
    *,
    probe: ProbeSet | None = None,
) -> str:
    """Write an actor artifact into a folder that does not exist yet. Returns its digest.

    The folder must be new: an artifact half overwritten by a second writer is a folder whose
    digest matches nothing anybody wrote down.
    """
    import torch
    from safetensors.torch import save

    root = Path(folder)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} already holds files; an artifact is written once")
    root.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        for name, tensor in state.items()
    }
    (root / WEIGHTS_NAME).write_bytes(save(tensors))
    meta = dict(spec.meta)
    meta["dtype"] = "float32"
    (root / SPEC_NAME).write_bytes(msgspec.json.encode(msgspec.structs.replace(spec, meta=meta)))
    if probe is not None:
        (root / PROBE_NAME).write_bytes(
            save(
                {
                    "rows": torch.from_numpy(np.ascontiguousarray(probe.rows, dtype=np.uint8)),
                    "log_probs": torch.from_numpy(
                        np.ascontiguousarray(probe.log_probs, dtype=np.float32)
                    ),
                }
            )
        )
    return artifact_digest(root)


def read_actor_artifact(folder: str | Path) -> ActorArtifact:
    """An actor artifact's spec, its tensors on the CPU, and its probe set if it has one."""
    from safetensors.torch import load

    root = Path(folder)
    for name in (WEIGHTS_NAME, SPEC_NAME):
        if not (root / name).is_file():
            raise PreflightError(f"{root} is not an actor artifact: it has no {name}")
    spec = msgspec.json.decode((root / SPEC_NAME).read_bytes(), type=SnapshotSpec)
    state = load((root / WEIGHTS_NAME).read_bytes())
    probe = None
    if (root / PROBE_NAME).is_file():
        blob = load((root / PROBE_NAME).read_bytes())
        probe = ProbeSet(rows=blob["rows"].numpy(), log_probs=blob["log_probs"].numpy())
    return ActorArtifact(folder=root, spec=spec, state=dict(state), probe=probe)


def check_actor_artifact(artifact: ActorArtifact, current: SnapshotSpec, *, what: str) -> None:
    """``check_compatible`` plus the action layout, which an init or a reference also needs.

    A pool snapshot shares the run's action space by construction; an artifact brought in from
    outside the run is exactly the thing that might not.
    """
    from ..ladder.snapshots import check_compatible

    check_compatible(artifact.spec, current)
    if artifact.spec.action_digest and artifact.spec.action_digest != current.action_digest:
        raise PreflightError(
            f"{what}: the artifact was written for action space {artifact.spec.action_digest} and "
            f"this run's is {current.action_digest}"
        )


def load_actor_state(actor: Any, artifact: ActorArtifact, *, what: str) -> None:
    """``load_state_dict(strict=True)``, with a refusal that names the keys that did not match."""
    reference = actor.state_dict()
    missing = sorted(set(reference) - set(artifact.state))
    unexpected = sorted(set(artifact.state) - set(reference))
    if missing or unexpected:
        raise PreflightError(
            f"{what}: the artifact's tensors do not match this run's actor. Missing: "
            f"{missing[:8]}; unexpected: {unexpected[:8]}"
        )
    shapes = [
        name
        for name, tensor in artifact.state.items()
        if tuple(tensor.shape) != tuple(reference[name].shape)
    ]
    if shapes:
        raise PreflightError(f"{what}: tensors of the wrong shape: {shapes[:8]}")
    actor.load_state_dict(
        {name: tensor.to(reference[name].device) for name, tensor in artifact.state.items()},
        strict=True,
    )


# --------------------------------------------------------------------------
# Field models
# --------------------------------------------------------------------------


class FieldModelSpec(msgspec.Struct, frozen=True, kw_only=True):
    """A small MLP over named observation fields: ``spec.json`` of a field-model artifact.

    ``fields`` are ``(name, width)`` in input order. The run's vector layout must hold every one
    at that width, or the model would read somebody else's numbers.
    """

    fields: list[tuple[str, int]]
    mean: list[float]
    std: list[float]
    hidden: list[int]
    activation: str = "tanh"
    #: What the output is the logit of. Only "play" -- p(play) on the row -- is defined.
    target: str = "play"
    format_version: int = 1
    meta: dict[str, Any] = {}


def write_field_model(folder: str | Path, state: Mapping[str, Tensor], spec: FieldModelSpec) -> str:
    """Write a field-model artifact into a new folder. Returns its digest."""
    import torch
    from safetensors.torch import save

    root = Path(folder)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} already holds files; an artifact is written once")
    root.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        for name, tensor in state.items()
    }
    (root / FIELD_MODEL_NAME).write_bytes(save(tensors))
    (root / SPEC_NAME).write_bytes(msgspec.json.encode(spec))
    return artifact_digest(root)


def read_field_model(folder: str | Path) -> tuple[FieldModelSpec, dict[str, Tensor]]:
    """A field-model artifact's spec and tensors."""
    from safetensors.torch import load

    root = Path(folder)
    for name in (FIELD_MODEL_NAME, SPEC_NAME):
        if not (root / name).is_file():
            raise PreflightError(f"{root} is not a field-model artifact: it has no {name}")
    spec = msgspec.json.decode((root / SPEC_NAME).read_bytes(), type=FieldModelSpec)
    return spec, dict(load((root / FIELD_MODEL_NAME).read_bytes()))


def build_field_mlp(spec: FieldModelSpec) -> Any:
    """The module a field-model spec describes, untrained. ``torch.nn.Sequential``."""
    from torch import nn

    activations: dict[str, Callable[[], nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}
    if spec.activation not in activations:
        raise PreflightError(
            f"field model activation {spec.activation!r} is not one of {sorted(activations)}"
        )
    width = sum(size for _, size in spec.fields)
    layers: list[nn.Module] = []
    for hidden in spec.hidden:
        layers += [nn.Linear(width, hidden), activations[spec.activation]()]
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)
