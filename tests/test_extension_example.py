"""The worked example in ``docs/extensions.md`` runs as printed.

The page is the only copy of the example. This reads its fences, writes the package and its
``pyproject.toml`` into a git repository in a temporary folder, installs it the way ``pip install
-e`` leaves it, and then does what the page tells a reader to: loads the page's config section,
starts a run, and runs the page's own test. An example nothing runs is found by the first reader
who pastes it, which is how two of this project's doc examples stayed broken for a day and a half.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn import config as cfg
from royalelearn.errors import PreflightError
from royalelearn.extensions import ENTRY_POINT_GROUP
from royalelearn.testing import coordinator, tiny_config
from test_extensions import _git, _install

pytest.importorskip("torch")

PAGE = Path(__file__).resolve().parents[1] / "docs" / "extensions.md"
FENCE = re.compile(r"^```(\w+)\n(.*?)^```$", re.M | re.S)


def _fences() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for language, body in FENCE.findall(PAGE.read_text(encoding="utf-8")):
        found.setdefault(language, []).append(body)
    return found


class Example:
    """What the page gives a reader: the package, its declaration, a section and a test."""

    def __init__(self) -> None:
        fences = _fences()
        counts = {language: len(bodies) for language, bodies in fences.items()}
        # Which fence is which is decided by position, so a fence added or dropped must fail here
        # rather than hand the wrong block to the wrong step.
        assert counts.get("python") == 2 and counts.get("toml") == 1, counts
        assert counts.get("json") == 1, counts
        self.package_code, self.test_code = fences["python"]
        self.pyproject_text = fences["toml"][0]
        self.pyproject = tomllib.loads(self.pyproject_text)
        self.section: dict[str, Any] = json.loads(fences["json"][0])
        self.points: dict[str, str] = self.pyproject["project"]["entry-points"][ENTRY_POINT_GROUP]
        (self.key,) = self.points
        self.module = self.points[self.key].partition(":")[0]


@pytest.fixture
def example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The page's package, committed in a repository of its own and installed editable."""
    page = Example()
    repo = tmp_path / "confidence-cap"
    (repo / page.module).mkdir(parents=True)
    (repo / page.module / "__init__.py").write_text(page.package_code, encoding="utf-8")
    (repo / "pyproject.toml").write_text(page.pyproject_text, encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "the example")
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(repo))
    _install(site, page.pyproject["project"]["name"], page.points, code=repo)
    importlib.invalidate_caches()
    page.repo = repo
    page.sha = _git(repo, "rev-parse", "HEAD")
    try:
        yield page
    finally:
        sys.modules.pop(page.module, None)


def _document(folder: Path, section: dict[str, Any]) -> dict[str, Any]:
    """A config file as a reader has one: a small run's keys, with the page's section added."""
    return {**msgspec.to_builtins(tiny_config(folder)), **section}


def test_the_page_section_loads_and_runs_through_the_installed_package(
    example: Any, tmp_path: Path
) -> None:
    config = cfg.validate(cfg.load_config(_document(tmp_path, example.section), say=None))
    with coordinator(config) as run:
        run.iterate()
    record = run.identity.extensions[example.key]
    assert (record.distribution, record.version, record.git) == (
        example.pyproject["project"]["name"],
        example.pyproject["project"]["version"],
        example.sha,
    )
    row = run.rows[-1]
    assert run.schema.unknown_keys(row) == ()
    assert 0.0 <= row["confidence_cap/over_cap_frac"] <= 1.0
    assert "confidence_cap/grad_ratio" in row
    assert "confidence_cap_crowded" in [alarm.name for alarm in run.alarms.alarms]


def test_the_page_test_passes(
    example: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace: dict[str, Any] = {}
    exec(compile(example.test_code, str(PAGE), "exec"), namespace)
    tests = [value for name, value in namespace.items() if name.startswith("test_")]
    assert len(tests) == 1
    tests[0](tmp_path, monkeypatch)


def test_the_page_problems_refuse_a_bad_cap(example: Any, tmp_path: Path) -> None:
    bad = {example.key: {**example.section[example.key], "cap": 1.5}}
    refusal = re.escape("confidence_cap.cap must be above 0 and at most 1, not 1.5")
    with pytest.raises(PreflightError, match=refusal):
        cfg.validate(cfg.load_config(_document(tmp_path, bad), say=None))
    typo = {example.key: {**example.section[example.key], "capp": 0.5}}
    with pytest.raises(msgspec.ValidationError, match="capp"):
        cfg.load_config(_document(tmp_path, typo), say=None)
