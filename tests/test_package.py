"""The package's own contract: it imports without torch and its public names are honest.

``import royalelearn`` is what the CLI's config, identity and doctor commands do before they
know whether torch is installed, so the import itself must not pull a learner module in. A
public name that cannot be provided has to say which package is missing rather than pretend
it was never a name at all.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import royalelearn

REPO = Path(__file__).resolve().parent.parent

# The public surface, taken from the package rather than retyped, so that adding an export
# without a test is not possible.
PUBLIC_NAMES = [n for n in royalelearn.__all__ if not n.startswith("_")]


def _probe(expression: str) -> str:
    """Evaluate ``expression`` in a fresh interpreter that has just imported the package.

    Fresh, because what is being asked is what ``import royalelearn`` alone loads, and this
    interpreter has already imported half the tree to run the rest of the suite.
    """
    completed = subprocess.run(
        [sys.executable, "-c", f"import royalelearn, sys; print({expression})"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_import_loads_no_learner_module() -> None:
    loaded = _probe("sorted(m for m in sys.modules if m.startswith('royalelearn.'))")
    assert loaded == "['royalelearn.version']", f"the package import loaded {loaded}"


def test_import_does_not_import_torch() -> None:
    assert _probe("'torch' in sys.modules") == "False"


def test_version_is_a_string() -> None:
    assert isinstance(royalelearn.__version__, str)
    assert royalelearn.__version__


def test_git_describe_is_a_string() -> None:
    assert isinstance(royalelearn.git_describe(), str)


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_resolves_or_names_what_it_needs(name: str) -> None:
    try:
        value = getattr(royalelearn, name)
    except ImportError as exc:
        assert exc.name, f"royalelearn.{name} raised an ImportError naming nothing: {exc}"
        return
    assert value is not None


def test_unknown_attribute_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = royalelearn.NotAName


@pytest.mark.parametrize("marker", ["slow", "engine"])
def test_marker_is_registered(pytestconfig: pytest.Config, marker: str) -> None:
    registered = {line.split(":", 1)[0] for line in pytestconfig.getini("markers")}
    assert marker in registered
