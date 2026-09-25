"""Extensions (``royalelearn.extensions``): how a section finds its provider, and what a provider
that is installed but unused, or used, does to a run.

Discovery here goes through real install metadata: each test that needs an installed package
writes a ``.dist-info`` folder with an ``entry_points.txt`` onto ``sys.path``, which is what
``pip install`` leaves behind, so the entry-point walk is the one a user's install goes through.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn import config as cfg
from royalelearn import extensions as registry
from royalelearn import identity as I
from royalelearn.errors import CheckpointFormatError, PreflightError
from royalelearn.extensions import ENTRY_POINT_GROUP, active_extensions, with_sections
from royalelearn.testing import StubExtension, coordinator, tiny_config, use_extensions
from test_user_code_identity import facts  # noqa: F401 - a fixture

DATA = Path(__file__).parent / "data"


def _install(
    site: Path, distribution: str, points: dict[str, str], *, code: Path | None = None
) -> None:
    """What ``pip install -e`` leaves for an entry point: a dist-info folder on the path, and a
    ``direct_url.json`` naming the checkout the package is imported from."""
    info = site / f"{distribution}-0.1.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.1\n", encoding="utf-8"
    )
    lines = [f"[{ENTRY_POINT_GROUP}]"] + [f"{key} = {value}" for key, value in points.items()]
    (info / "entry_points.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if code is not None:
        direct = {"url": code.resolve().as_uri(), "dir_info": {"editable": True}}
        (info / "direct_url.json").write_text(json.dumps(direct), encoding="utf-8")


def _package(root: Path, name: str, *, marker: Path | None = None) -> None:
    """A package whose ``EXTENSION`` is a ``StubExtension`` named ``name``, provided by itself.
    Importing it writes ``marker``, so a test can see whether anything did."""
    folder = root / name
    folder.mkdir(parents=True)
    body = [
        "import sys",
        "from royalelearn.testing import StubExtension",
        '__version__ = "0.1"',
        f"EXTENSION = StubExtension({name!r}, package=sys.modules[__name__])",
    ]
    if marker is not None:
        body.insert(0, f"open({str(marker)!r}, 'w').close()")
    (folder / "__init__.py").write_text("\n".join(body) + "\n", encoding="utf-8")


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "site"
    folder.mkdir()
    monkeypatch.syspath_prepend(str(folder))
    importlib.invalidate_caches()
    return folder


def _forget(*modules: str) -> None:
    for name in modules:
        sys.modules.pop(name, None)


# -- finding a provider ------------------------------------------------------------------------


def test_a_section_no_provider_claims_is_refused_by_name_at_load() -> None:
    """Before anything is built: load_config is the first thing a run does with its file."""
    with pytest.raises(PreflightError, match="'nosuch' is not a RoyaleLearn config key"):
        cfg.load_config({"nosuch": {"a": 1}})


def test_a_misspelt_key_inside_a_section_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    use_extensions(monkeypatch, {"stubsec": StubExtension("stubsec")})
    with pytest.raises(msgspec.ValidationError, match="coefficent"):
        cfg.load_config({"stubsec": {"coefficent": 1.0}})


def test_a_null_section_loads_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A claimed key that is null is an absent section; so is the ``"imitation": null`` every
    config of 0ab7a29..S2 carries, dropped by the shim with nothing installed to claim it."""
    use_extensions(monkeypatch, {"stubsec": StubExtension("stubsec")})
    loaded = cfg.load_config({"imitation": None, "stubsec": None}, say=None)
    assert type(loaded) is cfg.RunConfig
    assert loaded == cfg.laptop()
    assert active_extensions(loaded) == ()


def test_two_packages_providing_one_key_are_refused(site: Path) -> None:
    _install(site, "first-ext", {"twice": "first_ext:EXTENSION"})
    _install(site, "second-ext", {"twice": "second_ext:EXTENSION"})
    with pytest.raises(PreflightError, match="'twice' is provided by more than one"):
        cfg.load_config({"twice": {}})


def test_the_same_declaration_seen_twice_is_one_provider(
    site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale second copy of one distribution's metadata names the same object: not a clash."""
    code = tmp_path / "code"
    _package(code, "dup_ext")
    monkeypatch.syspath_prepend(str(code))
    _install(site, "dup-ext", {"dup_ext": "dup_ext:EXTENSION"}, code=code)
    _install(site / "stale", "Dup_Ext", {"dup_ext": "dup_ext:EXTENSION"}, code=code)
    monkeypatch.syspath_prepend(str(site / "stale"))
    try:
        loaded = cfg.load_config({"dup_ext": {}})
        assert [(a.name, a.distribution) for a in active_extensions(loaded)] == [
            ("dup_ext", "dup-ext")
        ], "one distribution under two spellings is one provider, named canonically"
    finally:
        _forget("dup_ext")


def test_another_api_version_and_a_wrong_name_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    old = StubExtension("old")
    old.api_version = registry.EXTENSION_API_VERSION + 1
    use_extensions(monkeypatch, {"old": old, "renamed": StubExtension("other")})
    with pytest.raises(PreflightError) as refused:
        cfg.load_config({"old": {}, "renamed": {}})
    message = str(refused.value)
    assert "'old' is provided at extension API version" in message
    assert "'renamed' is provided by an extension that calls itself 'other'" in message


# -- installed but unused ----------------------------------------------------------------------


def _plain_run(config: cfg.RunConfig, folder: Path) -> dict[str, Any]:
    with coordinator(config, run_dir=folder) as run:
        run.iterate()
        assert run.identity is not None
        return {
            "run_id": run.run_id,
            "config.json": (folder / "config.json").read_bytes(),
            "identity.json": (folder / "identity.json").read_bytes(),
            "alarms": [alarm.name for alarm in run.alarms.alarms],
            "schema": (sorted(run.schema.metrics), [p.template for p in run.schema.patterns]),
            "state_digest": run.rows[-1]["run/state_digest"],
        }


def test_an_installed_but_unused_extension_changes_nothing(
    site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same plain run without the package installed and with it: run_id, config.json,
    identity.json, the alarm table, the schema and the state digest are equal, and the package's
    module is never imported. Seen failing: an identity that reads the installed entry points."""
    marker = tmp_path / "imported"
    code = tmp_path / "code"
    _package(code, "unused_ext", marker=marker)
    monkeypatch.syspath_prepend(str(code))
    config = cfg.load_config(cfg.dump_config(tiny_config(tmp_path / "plain")))
    assert type(config) is cfg.RunConfig
    without = _plain_run(config, tmp_path / "without")
    _install(site, "unused-ext", {"unused_ext": "unused_ext:EXTENSION"}, code=code)
    importlib.invalidate_caches()
    with_it = _plain_run(cfg.load_config(cfg.dump_config(config)), tmp_path / "with")
    assert with_it == without
    assert len(with_it["alarms"]) == 24
    assert "unused_ext" not in sys.modules
    assert not marker.exists()
    # and it IS findable: naming it loads it
    try:
        named = cfg.load_config({"unused_ext": {}})
        assert [a.name for a in active_extensions(named)] == ["unused_ext"]
        assert marker.exists()
    finally:
        _forget("unused_ext")


# -- used: the identity names the code, and edits to it are seen -------------------------------


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_an_active_extension_is_recorded_by_commit_and_watched(
    site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, facts: Any  # noqa: F811
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _package(repo, "watched_ext")
    # As every repository here has: bytecode is not source, and importing writes it.
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c")
    sha = _git(repo, "rev-parse", "HEAD")
    monkeypatch.syspath_prepend(str(repo))
    _install(site, "watched-ext", {"watched_ext": "watched_ext:EXTENSION"}, code=repo)
    try:
        plain = cfg.RunConfig(env=cfg.default_env_spec(cfg.MOCK_ENGINE))
        used = with_sections(plain, watched_ext={"coefficient": 0.5, "alarms": {"x": 1.0}})
        identity = I.compute_identity(used, **facts)
        assert identity.extensions is not None
        record = identity.extensions["watched_ext"]
        assert (record.distribution, record.version, record.git) == ("watched-ext", "0.1", sha)
        # the section by value, with its alarms left out of the identity
        louder = with_sections(plain, watched_ext={"coefficient": 0.5, "alarms": {"x": 9.0}})
        assert I.compute_identity(louder, **facts).extensions == identity.extensions
        other = with_sections(plain, watched_ext={"coefficient": 0.25})
        assert I.compute_identity(other, **facts).extensions != identity.extensions

        def named(config: cfg.RunConfig) -> list[str]:
            return [line for line in I.dirty_sources(None, config=config) if "watched_ext" in line]

        assert named(used) == []
        (repo / "watched_ext" / "extra.py").write_text("x = 1\n", encoding="utf-8")
        assert named(used), "an uncommitted edit to the extension's package was not seen"
        assert named(plain) == [], "a run without the section is stopped by the package's edit"
        assert I.compute_identity(used, **facts).extensions["watched_ext"].git == f"{sha}-dirty"
    finally:
        _forget("watched_ext")


def test_an_extension_whose_commit_cannot_be_named_refuses_the_run(
    site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, facts: Any  # noqa: F811
) -> None:
    """Installed from a wheel, or inside somebody else's checkout: 'unknown' would make two runs
    on two different builds of it one run."""
    code = tmp_path / "loose"
    _package(code, "loose_ext")
    monkeypatch.syspath_prepend(str(code))
    _install(site, "loose-ext", {"loose_ext": "loose_ext:EXTENSION"}, code=code)
    try:
        used = with_sections(cfg.RunConfig(env=cfg.default_env_spec(cfg.MOCK_ENGINE)), loose_ext={})
        with pytest.raises(PreflightError, match="cannot be named"):
            I.compute_identity(used, **facts)
    finally:
        _forget("loose_ext")


# -- the identity refuses what it does not know ---------------------------------------------------


def test_a_resume_refuses_an_identity_field_this_build_does_not_know(tmp_path: Path) -> None:
    """A manifest written by a build with a field since removed (``imitation_digest``, say)
    would otherwise decode with it silently gone, and compare equal to this run's identity."""
    config = tiny_config(tmp_path / "run")
    with coordinator(config) as run:
        run.iterate()
        saved = run.checkpoint()
    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    manifest["identity"]["imitation_digest"] = "0" * 64
    (saved / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (
        pytest.raises(CheckpointFormatError, match="imitation_digest"),
        coordinator(config, resume=saved),
    ):
        pass


# -- the one-release shim -------------------------------------------------------------------------


def test_a_config_an_older_build_wrote_loads_to_the_same_run() -> None:
    """``examples/configs/laptop.json`` as 490c74c shipped it: ``"imitation": null`` and six
    ``alarms.imitation_*`` keys at their defaults. They are dropped with a notice, and the hash is
    the one the profile has, as it was before any of them existed."""
    said: list[str] = []
    loaded = cfg.load_config(DATA / "config-490c74c-laptop.json", say=said.append)
    assert cfg.config_hash(loaded) == cfg.config_hash(cfg.laptop())
    assert cfg.config_hash(loaded).startswith("9b55b7465c98")
    assert len(said) == 1
    assert '"imitation": null' in said[0] and "alarms.imitation_handoff_kl" in said[0]


def test_a_moved_threshold_that_was_changed_is_refused_with_its_new_home() -> None:
    with pytest.raises(PreflightError, match=r"now warm_start\.alarms\.handoff_kl"):
        cfg.load_config({"alarms": {"imitation_handoff_kl": 0.1}})


def test_the_keys_that_moved_to_warm_start_are_named() -> None:
    with pytest.raises(PreflightError, match=r"imitation\.init is now warm_start\.init"):
        cfg.load_config({"imitation": {"init": {"path": "x", "sha256": "y"}}})


# -- the surface --------------------------------------------------------------------------------


def test_the_supported_surface_resolves_and_importing_it_imports_nothing_else() -> None:
    """Every name in ``__all__`` resolves; and the registry module alone does not import torch or
    the core modules that import it."""
    for name in registry.__all__:
        assert getattr(registry, name) is not None, name
    probe = (
        "import sys, royalelearn.extensions as e; "
        "bad = [m for m in ('torch', 'royalelearn.coordinator', 'royalelearn.config') "
        "if m in sys.modules]; print(bad)"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]", done.stdout + done.stderr



# -- what the S3 review found --------------------------------------------------------------------


def test_a_null_key_no_provider_claims_is_still_refused() -> None:
    """A null value is how an absent section is written, not a way past a misspelt key."""
    with pytest.raises(PreflightError, match="'nosuch' is not a RoyaleLearn config key"):
        cfg.load_config({"nosuch": None})


def test_a_section_type_that_would_accept_a_misspelt_field_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Loose(msgspec.Struct):
        coefficient: float = 0.0

    loose = StubExtension("loose")
    loose.section_type = Loose  # type: ignore[assignment]
    use_extensions(monkeypatch, {"loose": loose})
    with pytest.raises(PreflightError, match="refuses unknown fields"):
        cfg.load_config({"loose": {"coefficent": 0.5}})


def test_a_declaration_whose_module_is_gone_is_refused_by_its_key(site: Path) -> None:
    """An editable install whose checkout was moved leaves its metadata behind."""
    _install(site, "ghost-ext", {"ghost": "ghost_mod:EXTENSION"})
    with pytest.raises(PreflightError, match="'ghost' is declared by ghost-ext as ghost_mod"):
        cfg.load_config({"ghost": {}})


def test_one_key_declared_by_two_distributions_is_refused(
    site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which of the two the identity named would depend on the path, and so would the run id."""
    code = tmp_path / "code"
    _package(code, "twin_ext")
    monkeypatch.syspath_prepend(str(code))
    _install(site, "a-ext", {"twin_ext": "twin_ext:EXTENSION"}, code=code)
    _install(site, "b-ext", {"twin_ext": "twin_ext:EXTENSION"}, code=code)
    try:
        with pytest.raises(
            PreflightError, match="declared by more than one distribution: a-ext, b-ext"
        ):
            cfg.load_config({"twin_ext": {}})
    finally:
        _forget("twin_ext")


def test_a_package_imported_from_outside_its_distribution_is_refused(
    site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale copy earlier on the path shadows the installed one: the code that would run is
    not the code the identity would name."""
    installed = tmp_path / "installed"
    stale = tmp_path / "stale"
    _package(installed, "shadow_ext")
    _package(stale, "shadow_ext")
    monkeypatch.syspath_prepend(str(installed))
    monkeypatch.syspath_prepend(str(stale))
    _install(site, "shadow-ext", {"shadow_ext": "shadow_ext:EXTENSION"}, code=installed)
    try:
        with pytest.raises(PreflightError, match="which that distribution does not own"):
            cfg.load_config({"shadow_ext": {}})
    finally:
        _forget("shadow_ext")


def test_an_extension_need_not_be_hashable(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unhashable(StubExtension):
        __hash__ = None  # type: ignore[assignment]

    use_extensions(monkeypatch, {"plain_eq": Unhashable("plain_eq")})
    loaded = cfg.load_config({"plain_eq": {}})
    assert [a.name for a in active_extensions(loaded)] == ["plain_eq"]


def test_a_section_set_to_none_encodes_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    use_extensions(monkeypatch, {"stubsec": StubExtension("stubsec")})
    sectioned = with_sections(cfg.laptop(), stubsec={"coefficient": 0.5})
    cleared = msgspec.structs.replace(sectioned, stubsec=None)
    assert cfg.dump_config(cleared) == cfg.dump_config(cfg.laptop())
    assert cfg.config_hash(cleared) == cfg.config_hash(cfg.laptop())
    assert type(with_sections(cleared)) is cfg.RunConfig
    with pytest.raises(PreflightError, match="RoyaleLearn's own config keys"):
        with_sections(cfg.laptop(), ppo={})


@pytest.mark.parametrize(
    "path",
    [
        ("identity", "imitation_digest"),
        ("identity", "engine_build", "built_by"),
        ("identity", "extensions", "stubsec", "signed_by"),
    ],
    ids=["identity", "engine_build", "extension_record"],
)
def test_a_resume_refuses_an_unknown_field_at_every_level_of_the_identity(
    tmp_path: Path, path: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seen failing: without forbid_unknown_fields on EngineBuild or ExtensionRecord, the nested
    cases resume."""
    use_extensions(monkeypatch, {"stubsec": StubExtension("stubsec")})
    config = with_sections(
        tiny_config(tmp_path / "run"),
        stubsec={"actor_lr_scale": {"kind": "constant", "value": 1.0}},
    )
    with coordinator(config) as run:
        run.iterate()
        saved = run.checkpoint()
    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    node = manifest
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = "0" * 16
    (saved / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (
        pytest.raises(CheckpointFormatError, match=path[-1]),
        coordinator(config, resume=saved),
    ):
        pass


def test_a_checkpoint_this_build_cannot_read_is_still_found_without_an_index(
    tmp_path: Path,
) -> None:
    """With no index, the store rebuilds its list from the manifests; one this build cannot read
    in full is still a checkpoint, so the resume refuses it by name instead of finding none."""
    from royalelearn.checkpoint import INDEX_NAME, DirCheckpointStore

    config = tiny_config(tmp_path / "run")
    with coordinator(config) as run:
        run.iterate()
        saved = run.checkpoint()
    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    manifest["identity"]["imitation_digest"] = "0" * 16
    (saved / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (saved.parent / INDEX_NAME).unlink()
    store = DirCheckpointStore(saved.parent.parent, keep=2)
    assert [entry.name for entry in store._entries(saved.parent)] == [saved.name]


def test_a_resume_does_not_report_keys_the_shim_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run made between 0ab7a29 and S2 stored the six retired alarm keys and a null imitation
    in its checkpoint's config; resumed now, they are not 'config differs' lines."""
    config = tiny_config(tmp_path / "run")
    with coordinator(config) as run:
        run.iterate()
        saved = run.checkpoint()
    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    for key, (default, _home) in cfg.RETIRED_ALARM_KEYS.items():
        manifest["config"]["alarms"][key] = default
    manifest["config"]["imitation"] = None
    (saved / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    capsys.readouterr()
    with coordinator(config, resume=saved):
        pass
    printed = capsys.readouterr().out
    assert "config differs" not in printed, printed
