"""The package imports without torch or rlgym_ppo; the seed classes fail clearly without them."""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

import royalelearn

SEED_DEPS = ("torch", "rlgym_ppo")


def _missing_deps() -> set[str]:
    return {dep for dep in SEED_DEPS if importlib.util.find_spec(dep) is None}


def test_import_pulls_no_seed_module() -> None:
    assert royalelearn.__version__ == "0.1.0"
    loaded = [m for m in sys.modules if m.startswith("royalelearn.")]
    assert loaded == [], f"the package import loaded seed modules: {loaded}"


def test_layers_below_are_importable() -> None:
    # RoyaleLearn -> RoyaleGym -> RoyaleSim: both siblings are in the workspace venv.
    importlib.import_module("royalegym")
    importlib.import_module("royalesim")


@pytest.mark.parametrize("name", sorted(royalelearn.SEED_CLASSES))
def test_seed_class_access(name: str) -> None:
    missing = _missing_deps()
    if not missing:
        cls = getattr(royalelearn, name)
        assert cls.__name__ == name
        return
    with pytest.raises(ImportError) as info:
        getattr(royalelearn, name)
    message = str(info.value)
    assert name in message
    assert royalelearn.SEED_CLASSES[name] in message
    assert info.value.name in missing


def test_unknown_attribute_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = royalelearn.NotAClass
