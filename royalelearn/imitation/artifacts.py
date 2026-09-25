"""Field models: a small MLP over named observation fields, stored as an artifact (19.6).

The actor-artifact half of this file -- the folder digest, the probe self-test, actor folders --
is generic and lives in ``royalelearn.artifacts``; what is here is the imitation half.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from ..artifacts import artifact_digest
from ..errors import PreflightError
from ..ladder.snapshots import SPEC_NAME

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor

__all__ = [
    "FIELD_MODEL_NAME",
    "FieldModelSpec",
    "build_field_mlp",
    "read_field_model",
    "write_field_model",
]

FIELD_MODEL_NAME = "model.safetensors"




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
