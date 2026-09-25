"""``royalelearn fit-field-reference``: the play/wait model a ``field_mlp`` reference loads.

Section 19.12. Rows are named observation fields and a play label: at this decision, did the
demonstrator play? The model is a small MLP giving the logit of p(play), fitted by weighted
binary cross-entropy on the CPU, and written as a field-model artifact (section 19.2).

Two things make it the run's model rather than a model of the data. The inputs are the run's own
vector fields by name, checked against the run's layout when the reference is loaded; and each
input is rounded to half precision before fitting, because that is the codec's rule for the
vector and so the precision the reference will be evaluated at inside a run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from ..errors import PreflightError
from .artifacts import FieldModelSpec, build_field_mlp, write_field_model
from .split import is_validation

__all__ = ["FitConfig", "fit_field_reference", "load_rows"]


class FitConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """How the model is fitted. Recorded in the artifact's ``meta``."""

    hidden: list[int] = msgspec.field(default_factory=lambda: [32, 32])
    activation: str = "tanh"
    epochs: int = 20
    batch_rows: int = 4096
    lr: float = 3e-3
    seed: int = 20260924


def load_rows(path: str | Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    """The rows file: one ``(N, width)`` column per field, ``label``, ``weight`` and ``group``.

    A one-dimensional field column is a field of width one. Every column must have the same row
    count, the labels must be 0 or 1, and the weights finite and non-negative.
    """
    blob = np.load(path)
    missing = [name for name in (*fields, "label", "weight", "group") if name not in blob.files]
    if missing:
        raise PreflightError(f"{path} has no column {missing[0]!r}; it has {sorted(blob.files)}")
    out: dict[str, np.ndarray] = {}
    for name in fields:
        column = np.asarray(blob[name], dtype=np.float32)
        out[name] = column.reshape(column.shape[0], -1)
    out["label"] = np.asarray(blob["label"], dtype=np.float32).reshape(-1)
    out["weight"] = np.asarray(blob["weight"], dtype=np.float32).reshape(-1)
    out["group"] = np.asarray(blob["group"], dtype=np.uint64).reshape(-1)
    rows = {column.shape[0] for column in out.values()}
    if len(rows) != 1:
        raise PreflightError(f"{path}: the columns have different row counts {sorted(rows)}")
    if not np.isin(out["label"], (0.0, 1.0)).all():
        raise PreflightError(f"{path}: a label is not 0 or 1")
    weight = out["weight"]
    if not (np.isfinite(weight).all() and (weight >= 0).all()):
        raise PreflightError(f"{path}: a weight is negative or not finite")
    return out


def _metrics(p: np.ndarray, label: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    """Weighted NLL and Brier score, and the unweighted rank AUC, of predictions ``p``."""
    total = float(weight.sum())
    if total <= 0:
        return {}
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    nll = -(weight * (label * np.log(clipped) + (1 - label) * np.log(1 - clipped))).sum() / total
    brier = (weight * (p - label) ** 2).sum() / total
    out = {"nll": float(nll), "brier": float(brier), "mean_p": float((weight * p).sum() / total)}
    out["mean_label"] = float((weight * label).sum() / total)
    positives, negatives = label == 1, label == 0
    if positives.any() and negatives.any():
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(p.size, dtype=np.float64)
        ranks[order] = np.arange(1, p.size + 1)
        n_pos, n_neg = int(positives.sum()), int(negatives.sum())
        out["auc"] = float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return out


def fit_field_reference(
    rows_path: str | Path,
    fields: Sequence[str],
    out: str | Path,
    *,
    config: FitConfig | None = None,
    printer: Callable[[str], None] | None = print,
) -> tuple[str, dict[str, Any]]:
    """Fit, write the artifact, and return its digest and the validation metrics."""
    import torch

    config = config if config is not None else FitConfig()
    say = printer or (lambda _line: None)
    columns = load_rows(rows_path, fields)
    widths = [(name, int(columns[name].shape[1])) for name in fields]
    inputs = np.concatenate([columns[name] for name in fields], axis=1)
    # The codec's rule for the vector: the reference will read these values in half precision.
    inputs = inputs.astype(np.float16).astype(np.float32)
    label, weight = columns["label"], columns["weight"]
    held = is_validation(columns["group"])
    train = ~held
    if not train.any() or not held.any():
        raise PreflightError(
            f"{rows_path}: the split by group left {int(train.sum())} training and "
            f"{int(held.sum())} validation rows; both need some"
        )
    mean = inputs[train].mean(axis=0)
    std = np.maximum(inputs[train].std(axis=0), 1e-6)
    spec = FieldModelSpec(
        fields=widths,
        mean=[float(v) for v in mean],
        std=[float(v) for v in std],
        hidden=list(config.hidden),
        activation=config.activation,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        module = build_field_mlp(spec)
    x = torch.from_numpy((inputs - mean) / std)
    y = torch.from_numpy(label)
    w = torch.from_numpy(weight)
    optimizer = torch.optim.Adam(module.parameters(), lr=config.lr)
    order_rng = np.random.default_rng(config.seed)
    train_index = np.flatnonzero(train)
    for epoch in range(config.epochs):
        order = train_index[order_rng.permutation(train_index.size)]
        for start in range(0, order.size, config.batch_rows):
            batch = torch.from_numpy(order[start : start + config.batch_rows])
            logits = module(x[batch]).squeeze(-1)
            weights = w[batch]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[batch], weight=weights, reduction="sum"
            ) / weights.sum().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch == config.epochs - 1 or epoch % 5 == 0:
            said = f"epoch {epoch + 1}/{config.epochs}, last batch loss {float(loss.detach()):.4f}"
            say(f"fit           {said}")
    with torch.no_grad():
        p = torch.sigmoid(module(x).squeeze(-1)).numpy().astype(np.float64)
    metrics = {
        "train": _metrics(p[train], label[train], weight[train]),
        "validation": _metrics(p[held], label[held], weight[held]),
        "rows": {"train": int(train.sum()), "validation": int(held.sum())},
    }
    rows_sha = hashlib.sha256(Path(rows_path).read_bytes()).hexdigest()
    spec = msgspec.structs.replace(
        spec,
        meta={"fit": msgspec.to_builtins(config), "metrics": metrics, "rows_sha256": rows_sha},
    )
    digest = write_field_model(out, module.state_dict(), spec)
    validation = {key: validation_value for key, validation_value in metrics["validation"].items()}
    nan = float("nan")
    say(
        f"fit           validation over {metrics['rows']['validation']} rows: nll "
        f"{validation.get('nll', nan):.4f}, brier {validation.get('brier', nan):.4f}, "
        f"auc {validation.get('auc', nan):.4f}"
    )
    say(f"artifact      {out}  sha256 {digest}")
    return digest, metrics
