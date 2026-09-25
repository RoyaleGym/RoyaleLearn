"""``package_provenance`` and ``section_digest``: what an add-on's record in a run's identity says.

The provenance of a package is the commit of the repository it is tracked in, and nothing else.
``git describe`` from the package's folder walks up into whatever repository encloses it, so an
installed copy inside a checkout's virtual environment reads as that checkout's commit; these
tests hold the stricter rule to each way that can go wrong.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from royalelearn.identity import package_provenance, section_digest


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    return root


def _package(folder: Path, name: str = "addon") -> ModuleType:
    package = folder / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        f"_prov_{abs(hash(package))}", package / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _commit(root: Path) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "c")
    return _git(root, "rev-parse", "HEAD")


def test_a_tracked_package_at_its_repositorys_top_is_named_by_its_commit(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    module = _package(root)
    sha = _commit(root)
    assert package_provenance(module) == sha


def test_an_uncommitted_file_in_the_package_marks_it_dirty(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    module = _package(root)
    sha = _commit(root)
    (root / "addon" / "new.py").write_text("x = 2\n", encoding="utf-8")
    assert package_provenance(module) == f"{sha}-dirty"


def test_a_package_inside_someone_elses_repository_is_unknown(tmp_path: Path) -> None:
    """The review's case: an installed copy in a checkout's virtual environment.

    ``git describe`` from there answers with the enclosing checkout's commit. Plant: drop the
    top-level check and this reads that commit."""
    outer = _repo(tmp_path / "outer")
    (outer / "README").write_text("x\n", encoding="utf-8")
    _commit(outer)
    module = _package(outer / ".venv" / "site-packages")
    assert package_provenance(module) == "unknown"


def test_an_untracked_package_at_a_repositorys_top_is_unknown(tmp_path: Path) -> None:
    """Plant: drop the ls-files check and this reads the repository's commit."""
    root = _repo(tmp_path / "repo")
    (root / "README").write_text("x\n", encoding="utf-8")
    _commit(root)
    module = _package(root)
    assert package_provenance(module) == "unknown"


def test_a_module_with_no_file_is_unknown() -> None:
    assert package_provenance(sys) == "unknown"


def test_a_section_digest_follows_content_and_drops_the_named_keys() -> None:
    base = {"a": {"path": "x", "sha256": "1"}, "b": [{"path": "y", "n": 2}]}
    moved = {"a": {"path": "elsewhere", "sha256": "1"}, "b": [{"path": "z", "n": 2}]}
    other = {"a": {"path": "x", "sha256": "2"}, "b": [{"path": "y", "n": 2}]}
    assert section_digest(base) == section_digest(moved)
    assert section_digest(base) != section_digest(other)
    assert section_digest(base, drop=()) != section_digest(moved, drop=())
    named = {"file": "a.bin", "digest": "1"}
    assert section_digest(named, drop=("file",)) == section_digest({"digest": "1"}, drop=())
