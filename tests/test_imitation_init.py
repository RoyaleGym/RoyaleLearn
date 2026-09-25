"""Section 19.1-19.4: the ``imitation`` and ``warm_start`` sections, their identity, their
artifacts and the actor init.

The properties, each held by a test that has been seen failing on a plant:

- a run without the sections encodes its identity exactly as before, so no ``run_id`` moves;
- a section's digest follows the files' content and not where they sit;
- a folder whose content is not the digest the config states is refused at every start, and
  every stale folder is named at once;
- an init loads the artifact into the actor, leaves the critic as the seed built it, and refuses
  an artifact whose weights no longer produce its own recorded probe log-probabilities;
- an init from the seeded weights is the run without the block, bit for bit, for three
  iterations (the init-identity control of section 19.15).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

import msgspec
import numpy as np
import pytest

from imitation_support import actor_state, seeded_artifact, with_imitation
from royalelearn import config as cfg
from royalelearn import identity as I
from royalelearn.artifacts import artifact_digest, read_actor_artifact
from royalelearn.errors import IdentityMismatch, PreflightError
from royalelearn.extensions import active_extensions, with_sections
from royalelearn.rollout.envspec import canonical_json
from test_coordinator import coordinator, tiny_config
from test_user_code_identity import facts  # noqa: F401 - a fixture

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def _block(**overrides: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "references": {"bc": {"kind": "snapshot", "path": "a/bc", "sha256": "1" * 64}},
        "regularisers": [
            {
                "kind": "reference_kl",
                "name": "bc",
                "reference": "bc",
                "budget": {"kind": "constant", "value": 0.1},
                "coef": {"start": 0.3},
            }
        ],
    }
    block.update(overrides)
    return block


def _config(**block: Any) -> cfg.RunConfig:
    """The ``imitation`` section from ``_block``; ``init`` and ``actor_lr_scale`` go to
    ``warm_start``, the section that owns them."""
    warm = {key: block.pop(key) for key in ("init", "actor_lr_scale") if key in block}
    return cfg.load_config({"imitation": _block(**block), **({"warm_start": warm} if warm else {})})


# -- the block -----------------------------------------------------------------------------


def test_a_well_formed_block_has_no_problems() -> None:
    assert cfg.check_consistency(_config()) == []


REG = _block()["regularisers"][0]


@pytest.mark.parametrize(
    ("block", "named"),
    [
        ({"regularisers": [{**REG, "reference": "nope"}]}, "reference 'nope' is not declared"),
        ({"regularisers": [{**REG, "name": "a/b"}]}, "one metric segment"),
        ({"regularisers": [REG, REG]}, "two regularisers are named 'bc'"),
        ({"regularisers": [{**REG, "factor": "reverse"}]}, "factor 'reverse'"),
        (
            {"references": {"bc": {"kind": "field_mlp", "path": "a", "sha256": "b"}}},
            "can back factor 'noop_marginal' and not 'joint'",
        ),
        (
            {"regularisers": [{**REG, "budget": {"kind": "constant", "value": -0.1}}]},
            "the budget takes a negative value",
        ),
        (
            {
                "regularisers": [
                    {
                        **REG,
                        "coef": {
                            "start": 0.3,
                            "max": 0.1,
                            "min": {"kind": "constant", "value": 0.5},
                        },
                    }
                ]
            },
            "coef.max 0.1 is below the largest floor",
        ),
        (
            {"regularisers": [{**REG, "coef": {"start": 0.3, "band": 1.0}}]},
            "coef.band must be above 1",
        ),
        ({"regularisers": [{**REG, "coef": {"start": -1.0}}]}, "is negative"),
        (
            {
                "regularisers": [
                    {**REG, "exclude_when": [{"field": "clock", "op": "==", "value": 1}]}
                ]
            },
            "exclude_when op '=='",
        ),
        (
            {"actor_lr_scale": {"kind": "constant", "value": -1.0}},
            "actor_lr_scale takes a negative",
        ),
    ],
)
def test_each_malformed_block_is_refused_by_name(block: dict[str, Any], named: str) -> None:
    problems = cfg.check_consistency(_config(**block))
    assert any(named in problem for problem in problems), problems


def test_an_unknown_field_in_the_block_is_refused_at_load() -> None:
    with pytest.raises(msgspec.ValidationError):
        cfg.load_config({"imitation": {"inits": {}}})


# -- identity ------------------------------------------------------------------------------


def _identity(config: cfg.RunConfig, facts: dict[str, Any]) -> I.RunIdentity:  # noqa: F811
    return I.compute_identity(config, **facts)


def test_without_the_block_the_identity_encodes_exactly_as_before(facts: Any) -> None:  # noqa: F811
    """The encoding of every field that existed before, in their order, and nothing else.

    Plant: without ``omit_defaults`` on ``RunIdentity`` the encoding gains
    ``"extensions":null`` and every run id in every run directory would move.
    """
    identity = _identity(cfg.RunConfig(env=cfg.default_env_spec(cfg.MOCK_ENGINE)), facts)
    assert identity.extensions is None
    before = [name for name in I.RunIdentity.__struct_fields__ if name != "extensions"]
    Before = msgspec.defstruct("Before", [(name, Any) for name in before])
    old = Before(**{name: getattr(identity, name) for name in before})
    assert canonical_json(identity) == canonical_json(old)
    assert b"extensions" not in canonical_json(identity)


def test_the_block_digest_follows_content_not_paths(facts: Any) -> None:  # noqa: F811
    env = cfg.default_env_spec(cfg.MOCK_ENGINE)
    base = msgspec.structs.replace(_config(), env=env)
    moved_block = _block(
        references={"bc": {"kind": "snapshot", "path": "elsewhere/bc", "sha256": "1" * 64}}
    )
    moved = msgspec.structs.replace(cfg.load_config({"imitation": moved_block}), env=env)
    other_file = msgspec.structs.replace(
        cfg.load_config(
            {
                "imitation": _block(
                    references={"bc": {"kind": "snapshot", "path": "a/bc", "sha256": "2" * 64}}
                )
            }
        ),
        env=env,
    )
    other_budget = msgspec.structs.replace(
        cfg.load_config(
            {
                "imitation": _block(
                    regularisers=[{**REG, "budget": {"kind": "constant", "value": 0.2}}]
                )
            }
        ),
        env=env,
    )
    def digest(config: cfg.RunConfig) -> str:
        records = _identity(config, facts).extensions
        assert records is not None
        return records["imitation"].digest

    assert digest(moved) == digest(base)
    assert digest(other_file) != digest(base)
    assert digest(other_budget) != digest(base)


def test_a_resume_refuses_a_block_added_or_changed(facts: Any) -> None:  # noqa: F811
    from royalelearn.checkpoint import check_resume

    env = cfg.default_env_spec(cfg.MOCK_ENGINE)
    without = _identity(cfg.RunConfig(env=env), facts)
    added = _identity(msgspec.structs.replace(_config(), env=env), facts)
    assert set(I.identity_differences(without, added)) == {"extensions"}

    class _Manifest:
        identity = without
        config: ClassVar[dict[str, Any]] = {}

    with pytest.raises(IdentityMismatch) as refused:
        check_resume(_Manifest(), added, {})  # type: ignore[arg-type]
    assert "extensions" in refused.value.differences


# -- artifact digests ----------------------------------------------------------------------


def _folder(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True)
    for name, blob in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return root


def test_the_artifact_digest_covers_every_file_and_not_the_location(tmp_path: Path) -> None:
    files = {"actor.safetensors": b"weights", "spec.json": b"{}", "sub/probe.safetensors": b"p"}
    here = _folder(tmp_path / "here", files)
    digest = artifact_digest(here)
    there = tmp_path / "there" / "deeper"
    shutil.copytree(here, there)
    assert artifact_digest(there) == digest
    for name in files:
        edited = _folder(
            tmp_path / f"edit-{name.replace('/', '-')}", {**files, name: files[name] + b"!"}
        )
        assert artifact_digest(edited) != digest, name
    extra = _folder(tmp_path / "extra", {**files, "notes.txt": b""})
    assert artifact_digest(extra) != digest


def test_every_stale_digest_is_named_at_once(tmp_path: Path) -> None:
    one = _folder(tmp_path / "one", {"a": b"1"})
    two = _folder(tmp_path / "two", {"a": b"2"})
    config = with_sections(
        cfg.RunConfig(),
        warm_start={"init": {"path": str(one), "sha256": "0" * 64}},
        imitation={
            "references": {
                "bc": {"kind": "snapshot", "path": str(two), "sha256": "0" * 64},
                "ok": {"kind": "snapshot", "path": str(one), "sha256": artifact_digest(one)},
            }
        },
    )
    message = "\n".join(
        problem for a in active_extensions(config) for problem in a.extension.verify(a.section)
    )
    assert "warm_start.init" in message and "imitation.references.bc" in message
    assert "imitation.references.ok" not in message
    assert artifact_digest(two) in message


# -- the init ------------------------------------------------------------------------------


def _init_config(tmp_path: Path, folder: Path, digest: str, **overrides: Any) -> cfg.RunConfig:
    return with_imitation(
        tiny_config(tmp_path / "run", **overrides),
        init={"path": str(folder), "sha256": digest},
    )


def _run_rows(config: cfg.RunConfig, iterations: int) -> tuple[list[str], np.ndarray]:
    digests: list[str] = []
    with coordinator(config) as run:
        for _ in range(iterations):
            run.iterate()
            digests.append(str(run.rows[-1]["run/state_digest"]))
        actions = run.buffer.action[: run.buffer.cycles].copy()
    return digests, actions


def test_an_init_from_the_seeded_weights_is_the_run_without_it(tmp_path: Path) -> None:
    """The init-identity control: the init path adds nothing but the weights it loads.

    Plant: a changed weight in the artifact makes the self-test refuse (the next test), and an
    init that also re-seeded or touched the critic would part the state digests here.
    """
    folder = tmp_path / "seeded"
    digest = seeded_artifact(tiny_config(tmp_path / "donor"), folder, coordinator)
    plain = _run_rows(tiny_config(tmp_path / "plain"), 3)
    initialised = _run_rows(_init_config(tmp_path, folder, digest), 3)
    assert plain[0] == initialised[0]
    assert np.array_equal(plain[1], initialised[1])


def test_an_init_loads_the_actor_and_leaves_the_critic_as_seeded(tmp_path: Path) -> None:
    donor = tiny_config(tmp_path / "donor", master_seed=7)
    folder = tmp_path / "donor-artifact"
    digest = seeded_artifact(donor, folder, coordinator)
    expected = read_actor_artifact(folder).state
    with coordinator(tiny_config(tmp_path / "plain")) as plain:
        critic_seeded = {k: v.clone() for k, v in plain.model.critic.state_dict().items()}
        actor_seeded = actor_state(plain)
    with coordinator(_init_config(tmp_path, folder, digest)) as run:
        loaded = actor_state(run)
        for name, tensor in expected.items():
            assert torch.equal(loaded[name], tensor), name
        assert any(not torch.equal(loaded[n], actor_seeded[n]) for n in loaded)
        for name, tensor in run.model.critic.state_dict().items():
            assert torch.equal(tensor, critic_seeded[name]), name
        assert run.extension_facts["warm_start"]["self_test"] == 0.0


def test_an_artifact_whose_weights_no_longer_match_its_probe_is_refused(tmp_path: Path) -> None:
    def nudge(tensors: dict[str, Any]) -> None:
        name = next(n for n, t in tensors.items() if t.is_floating_point() and t.numel() > 1)
        tensors[name].view(-1)[0] += 0.5

    folder = tmp_path / "nudged"
    digest = seeded_artifact(tiny_config(tmp_path / "donor"), folder, coordinator, edit=nudge)
    with (
        pytest.raises(PreflightError, match="probe-logit self-test failed"),
        coordinator(_init_config(tmp_path, folder, digest)),
    ):
        pass


def test_an_init_without_probe_rows_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "bare"
    digest = seeded_artifact(tiny_config(tmp_path / "donor"), folder, coordinator, probe=False)
    with (
        pytest.raises(PreflightError, match="has no probe rows"),
        coordinator(_init_config(tmp_path, folder, digest)),
    ):
        pass


def test_an_init_of_another_architecture_is_refused_by_name(tmp_path: Path) -> None:
    folder = tmp_path / "wide"
    wide = tiny_config(
        tmp_path / "donor",
        net=cfg.NetConfig(
            channels=16,
            blocks=1,
            norm_groups=4,
            card_embed=16,
            value_hidden=16,
            autocast_dtype="float32",
            device="cpu",
        ),
    )
    digest = seeded_artifact(wide, folder, coordinator)
    with (
        pytest.raises(IdentityMismatch) as refused,
        coordinator(_init_config(tmp_path, folder, digest)),
    ):
        pass
    assert "arch_digest" in refused.value.differences


def test_a_stale_digest_is_refused_before_anything_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "seeded"
    seeded_artifact(tiny_config(tmp_path / "donor"), folder, coordinator)

    def no_preflight(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preflight ran before the digests were checked")

    monkeypatch.setattr("royalelearn.coordinator.run_preflight", no_preflight)
    with (
        pytest.raises(PreflightError, match="has digest"),
        coordinator(_init_config(tmp_path, folder, "0" * 64)),
    ):
        pass


def test_a_resume_keeps_the_checkpoint_weights_and_still_checks_the_digest(tmp_path: Path) -> None:
    folder = tmp_path / "seeded"
    digest = seeded_artifact(tiny_config(tmp_path / "donor", master_seed=7), folder, coordinator)
    config = _init_config(tmp_path, folder, digest)
    with coordinator(config) as run:
        run.iterate()
        saved = run.checkpoint()
        trained = actor_state(run)
    said: list[str] = []
    with coordinator(config, resume=saved, printer=said.append) as resumed:
        for name, tensor in actor_state(resumed).items():
            assert torch.equal(tensor, trained[name]), name
        # Preflight deferred its predicted ratio guard to the section, and on a resume the
        # section measures it on the checkpoint's actor rather than skipping it.
        assert "ratio_precision" in resumed.extension_facts["warm_start"]
        assert "self_test" not in resumed.extension_facts["warm_start"]
    assert any("measured on" in line and "probe rows" in line for line in said)
    (folder / "spec.json").write_bytes((folder / "spec.json").read_bytes() + b" ")
    with pytest.raises(PreflightError, match="has digest"), coordinator(config, resume=saved):
        pass


# -- the ratio guard, measured on the loaded actor ---------------------------------------------


def test_the_ratio_guard_is_measured_on_a_loaded_actor(tmp_path: Path) -> None:
    """Section 19.9: an initialised actor's p_max is its own, not the one noop_bias implies.

    A cloned policy that holds on most rows puts far more than the bias's share on one action.
    The same rows pass in float32 and are refused in bfloat16 once the head is pushed there.
    Plant: taking p_max from noop_bias instead of the rows must let the bfloat16 case through.
    """
    from royalelearn.imitation.init import loaded_ratio_guard

    folder = tmp_path / "seeded"
    digest = seeded_artifact(tiny_config(tmp_path / "donor"), folder, coordinator)
    rows = read_actor_artifact(folder).probe.rows
    said: list[str] = []
    with coordinator(_init_config(tmp_path, folder, digest), printer=said.append) as run:
        assert any("measured on" in line and "probe rows" in line for line in said)
        with torch.no_grad():
            run.model.actor.head.noop.bias.fill_(40.0)
        codec = run.row_codec()
        quiet = loaded_ratio_guard(
            run.model, codec, rows, atol=1e-4, precision="float32", say=said.append
        )
        assert quiet < 1e-4 / 4
        with pytest.raises(PreflightError, match="the loaded actor's own arithmetic"):
            loaded_ratio_guard(
                run.model, codec, rows, atol=2e-2, precision="bfloat16", say=said.append
            )


def test_preflight_defers_the_ratio_guard_to_the_init(env_spec: Any) -> None:
    """The noop_bias estimate is about a seeded actor; with an init it is not this run's."""
    from royalelearn.rollout.preflight import _ratio_precision_gate

    heavy = cfg.RunConfig(
        env=cfg.default_env_spec(cfg.MOCK_ENGINE),
        net=cfg.NetConfig(noop_bias=40.0, autocast_dtype="bfloat16"),
    )
    with pytest.raises(PreflightError):
        _ratio_precision_gate(heavy, env_spec, lambda _line: None)
    initialised = with_sections(heavy, warm_start={"init": {"path": "x", "sha256": "y"}})
    said: list[str] = []
    _ratio_precision_gate(initialised, env_spec, said.append)
    assert said and "measured by warm_start on the loaded actor" in said[0]
