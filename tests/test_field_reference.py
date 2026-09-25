"""Section 19.12: ``fit-field-reference``, and the ``field_mlp`` reference that loads it.

The rows here are synthetic on purpose. The property under test is the tool's, not a dataset's:
that a play rate which depends on elixir is learnt as depending on elixir, that the split is by
group, and that the artifact loads into a run as a reference which reads the same field the model
was fitted on -- and refuses one the run's observation does not carry at that width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import config as cfg
from royalelearn.artifacts import artifact_digest
from royalelearn.errors import PreflightError
from royalelearn.imitation.artifacts import read_field_model
from royalelearn.imitation.split import is_validation

torch = pytest.importorskip("torch")

from royalelearn.imitation.fit import FitConfig, fit_field_reference  # noqa: E402
from royalelearn.imitation.references import _field_mlp  # noqa: E402


def _rows(tmp_path: Path, *, n: int = 6000, seed: int = 3) -> Path:
    """Elixir on [0, 1] (the vector's own scale), and a play probability that rises with it."""
    rng = np.random.default_rng(seed)
    elixir = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    p = 0.05 + 0.6 * elixir[:, 0] ** 2
    label = (rng.uniform(size=n) < p).astype(np.float32)
    path = tmp_path / "rows.npz"
    np.savez(
        path,
        own_elixir=elixir,
        label=label,
        weight=np.ones(n, dtype=np.float32),
        group=(np.arange(n) // 20).astype(np.uint64),
    )
    return path


def test_the_split_is_by_group_and_stable() -> None:
    groups = np.repeat(np.arange(2000, dtype=np.uint64), 3)
    held = is_validation(groups)
    assert np.array_equal(held, is_validation(groups))
    per_group = held.reshape(-1, 3)
    assert (per_group.all(axis=1) | ~per_group.any(axis=1)).all(), "a group was split"
    assert 0.02 < held.mean() < 0.08


def test_a_fitted_model_learns_the_rate_and_loads_as_a_reference(
    tmp_path: Path, env_spec: Any
) -> None:
    out = tmp_path / "timing"
    digest, metrics = fit_field_reference(
        _rows(tmp_path),
        ["own_elixir"],
        out,
        config=FitConfig(hidden=[8], epochs=40, seed=1),
        printer=None,
    )
    assert digest == artifact_digest(out)
    assert metrics["validation"]["auc"] > 0.7
    spec, _ = read_field_model(out)
    assert spec.fields == [("own_elixir", 1)]
    assert spec.meta["metrics"]["rows"]["validation"] > 0

    reference = _field_mlp("timing", out, env_spec, "cpu")
    from royalelearn.api.policy import ObsBatch
    from royalelearn.obs_layout import field_slice

    column = field_slice(env_spec, "own_elixir").start
    vector = torch.zeros(3, env_spec.vector_size)
    vector[:, column] = torch.tensor([0.05, 0.5, 0.95])
    obs = ObsBatch(spatial=None, mask_planes=None, vector=vector, mask=None)  # type: ignore[arg-type]
    log_noop, log_play = reference.noop_log_probs(obs)
    play = log_play.exp()
    assert float(play[0]) < float(play[1]) < float(play[2])
    assert torch.allclose(log_noop.exp() + play, torch.ones(3), atol=1e-6)


def test_a_field_model_the_run_cannot_read_is_refused(tmp_path: Path, env_spec: Any) -> None:
    rows = _rows(tmp_path)
    out = tmp_path / "wide"
    blob = dict(np.load(rows))
    blob["own_elixir"] = np.repeat(blob["own_elixir"], 2, axis=1)
    np.savez(tmp_path / "wide.npz", **blob)
    fit_field_reference(
        tmp_path / "wide.npz", ["own_elixir"], out, config=FitConfig(epochs=1), printer=None
    )
    with pytest.raises(PreflightError, match="is 1 wide in this run's observation"):
        _field_mlp("timing", out, env_spec, "cpu")


def test_a_rows_file_with_a_bad_label_or_weight_is_refused(tmp_path: Path) -> None:
    rows = dict(np.load(_rows(tmp_path)))
    for column, value in (("label", 0.5), ("weight", -1.0)):
        broken = dict(rows)
        broken[column] = broken[column].copy()
        broken[column][0] = value
        np.savez(tmp_path / f"{column}.npz", **broken)
        with pytest.raises(PreflightError, match=column if column == "label" else "weight"):
            fit_field_reference(
                tmp_path / f"{column}.npz", ["own_elixir"], tmp_path / f"out-{column}", printer=None
            )


def test_a_run_anchors_its_hold_rate_to_a_fitted_model(tmp_path: Path) -> None:
    """End to end: fit, name the artifact in a config, and train one iteration against it."""
    from imitation_support import with_imitation
    from test_coordinator import coordinator, tiny_config

    out = tmp_path / "timing"
    digest, _ = fit_field_reference(
        _rows(tmp_path), ["own_elixir"], out, config=FitConfig(epochs=2), printer=None
    )
    block = {
        "references": {"timing": {"kind": "field_mlp", "path": str(out), "sha256": digest}},
        "regularisers": [
            {
                "kind": "reference_kl",
                "name": "timing",
                "reference": "timing",
                "factor": "noop_marginal",
                "budget": {"kind": "constant", "value": 0.05},
                "coef": {"start": 1.0},
            }
        ],
    }
    config = with_imitation(tiny_config(tmp_path / "run"), **block)
    with coordinator(config) as run:
        run.iterate()
        row = run.rows[-1]
    assert row["imitation/timing/kl"] > 0.0
    assert row["imitation/timing/kl_noop"] == pytest.approx(row["imitation/timing/kl"])
    assert "imitation/timing/kl_card" not in row
    assert msgspec.convert(block, cfg.ImitationConfig).regularisers[0].factor == "noop_marginal"


def test_inputs_are_fitted_at_the_precision_a_run_reads_them(tmp_path: Path) -> None:
    """The vector reaches a run in half precision, so a difference below it is no difference.

    Two rows files whose fields differ only below float16's resolution give the same artifact,
    digest and all. Plant: fitting on the float32 values makes the two digests differ.
    """
    rows = dict(np.load(_rows(tmp_path)))
    # On float16's grid first, so a nudge of a millionth stays inside every value's rounding
    # interval: a value near a rounding midpoint would otherwise be pushed across it.
    rows["own_elixir"] = rows["own_elixir"].astype(np.float16).astype(np.float32)
    np.savez(tmp_path / "rows.npz", **rows)
    nudged = dict(rows)
    nudged["own_elixir"] = rows["own_elixir"] + np.float32(1e-6)
    assert not np.array_equal(nudged["own_elixir"], rows["own_elixir"])
    np.savez(tmp_path / "nudged.npz", **nudged)
    config = FitConfig(hidden=[4], epochs=2, seed=5)
    fit_field_reference(
        tmp_path / "rows.npz", ["own_elixir"], tmp_path / "a", config=config, printer=None
    )
    fit_field_reference(
        tmp_path / "nudged.npz", ["own_elixir"], tmp_path / "b", config=config, printer=None
    )
    spec_a, state_a = read_field_model(tmp_path / "a")
    spec_b, state_b = read_field_model(tmp_path / "b")
    assert spec_a.mean == spec_b.mean and spec_a.std == spec_b.std
    for name, tensor in state_a.items():
        assert torch.equal(tensor, state_b[name]), name
