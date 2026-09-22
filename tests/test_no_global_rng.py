"""No module in this package draws from a global generator.

Every random draw in the harness comes from a named stream: ``derive_generator(master_seed,
path)``. A single ``np.random.random()`` anywhere would make a run's trajectory depend on how
many times something else had drawn first, which is exactly the property name-addressed seeding
exists to buy and exactly the kind of regression nobody notices for a month.

A source scan rather than a runtime check, because the failure is a line of code and not a
state: a call that runs once every thousand iterations would never show up in a fixture.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "royalelearn"

#: ``np.random.<name>`` is a global draw except for these, which are the constructors and the
#: types name-addressed seeding is built out of.
ALLOWED_NUMPY = frozenset({"Generator", "PCG64", "SeedSequence", "default_rng"})

#: ``torch.<name>`` that draws from the global generator unless it is handed one.
TORCH_DRAWS = frozenset(
    {"rand", "randn", "randint", "randperm", "rand_like", "randn_like", "randint_like"}
)


def _sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _modules() -> list[tuple[Path, ast.Module]]:
    return [(path, ast.parse(path.read_text(encoding="utf-8"), str(path))) for path in _sources()]


def _attribute_path(node: ast.AST) -> str:
    """``np.random.default_rng`` as a dotted string, or "" for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def test_no_module_calls_a_global_numpy_draw() -> None:
    offences: list[str] = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _attribute_path(node.func)
            if not dotted.startswith(("np.random.", "numpy.random.")):
                continue
            if dotted.rsplit(".", 1)[-1] in ALLOWED_NUMPY:
                continue
            offences.append(f"{path.name}:{node.lineno}: {dotted}")
    assert offences == [], "global numpy draws: " + ", ".join(offences)


def test_no_module_calls_the_bare_random_module() -> None:
    """``random.random()`` and friends. ``random.getstate`` is the checkpoint's, not a draw."""
    allowed = {"getstate", "setstate"}
    offences: list[str] = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _attribute_path(node.func)
            if dotted.startswith("random.") and dotted.split(".", 1)[1] not in allowed:
                offences.append(f"{path.name}:{node.lineno}: {dotted}")
    assert offences == [], "bare random draws: " + ", ".join(offences)


def test_no_module_draws_from_torchs_global_generator() -> None:
    """``torch.rand*`` without a ``generator=``.

    Action sampling takes caller-supplied uniforms for the same reason: a sampled action is a
    function of the master seed, the iteration, the cycle and the slot, and of nothing else.
    """
    offences: list[str] = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _attribute_path(node.func)
            if not dotted.startswith("torch."):
                continue
            if dotted.rsplit(".", 1)[-1] not in TORCH_DRAWS:
                continue
            if any(keyword.arg == "generator" for keyword in node.keywords):
                continue
            offences.append(f"{path.name}:{node.lineno}: {dotted}")
    assert offences == [], "unseeded torch draws: " + ", ".join(offences)


def test_nothing_seeds_a_generator_from_the_clock() -> None:
    """``time.time()`` as a seed is the other way a run stops being reproducible."""
    offences: list[str] = []
    for path, tree in _modules():
        seeding_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _attribute_path(node.func).endswith(("seed", "manual_seed"))
        ]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _attribute_path(node.func)
            if dotted not in ("time.time", "time.time_ns", "time.monotonic"):
                continue
            if any(node in call.args for call in seeding_calls):
                offences.append(f"{path.name}:{node.lineno}: {dotted}")
    assert offences == []


@pytest.mark.parametrize("name", ["seeding.py", "determinism.py"])
def test_the_modules_that_own_seeding_exist_where_the_scan_looks(name: str) -> None:
    """The scan is worthless if it is pointed at the wrong tree."""
    assert (PACKAGE / name).is_file()
    assert len(_sources()) > 20
