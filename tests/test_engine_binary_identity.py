"""Two runs on two different engines used to write the same identity.

``build_digest`` hashes the DATA compiled into the extension, not the Rust it was compiled from.
Measured 2026-09-23 across a rebuild from an edited ``state.rs``: ``build_digest`` read
``8952c1c4aa7d7923`` on both sides, and the binary's own hash went to ``04d42611db5d6e5a``. The
engine had carried that hash in ``RustEngine.config()`` since RoyaleGym added it, and
``engine_build`` dropped it on the floor, so two runs either side of the rebuild wrote identical
``engine_build`` blocks and played different games.

This file pins three things. The binary reaches the identity. A checkpoint written before it did
still resumes, and says it could not check. And a worker cannot quietly run a different binary
from the one the identity names, which is the case a rebuild during a run produces: the parent
measured the file at start, and a restarted worker loads whatever is on disk by then.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

from royalelearn import config as C
from royalelearn import identity as I
from royalelearn.api.rollout import EnvSpec
from royalelearn.checkpoint import check_resume
from royalelearn.errors import IdentityMismatch, PreflightError
from royalelearn.rollout.envspec import EnvFactorySpec

BEFORE = "8952c1c4aa7d7923"
BINARY_A = "04d42611db5d6e5a"
BINARY_B = "5f0e2b7c1a9d3e46"


def _env_config(binary: str | None) -> dict[str, Any]:
    """The shape ``ClashParallelEnv.config()`` has: the engine's own config under params."""
    params: dict[str, Any] = {"cards": ["Knight"], "build_digest": BEFORE}
    if binary is not None:
        params["engine_binary_sha256"] = binary
    return {
        "engine": {"class": "royalegym.rust_engine.RustEngine", "params": params},
        "calibration_digest": "c" * 16,
        "build_digest": BEFORE,
    }


def _build(binary: str | None) -> I.EngineBuild:
    return I.engine_build(_env_config(binary), [])


# -- the binary reaches the identity ----------------------------------------------------------


def test_the_binary_the_engine_reports_is_recorded() -> None:
    assert _build(BINARY_A).binary_sha256 == BINARY_A


def test_the_real_env_reports_its_binary_and_it_is_recorded() -> None:
    """Not a hand-built dict: the shape ``engine_build`` reads is the one the env produces."""
    env = C.default_env_spec(C.RUST_ENGINE).factory()()
    try:
        config = env.config()
        build = I.engine_build(config, env.engine.cards())
    finally:
        env.close()
    stated = config["engine"]["params"]["engine_binary_sha256"]
    assert build.binary_sha256 == stated
    assert build.binary_sha256 not in (I.NOT_STATED, I.NOT_RECORDED, "")


def test_an_engine_that_states_no_binary_says_so() -> None:
    """MockEngine has no compiled file. It is recorded as unstated, not as an empty string."""
    assert _build(None).binary_sha256 == I.NOT_STATED


@pytest.fixture
def base(env_spec: EnvSpec, mock_env_spec: EnvFactorySpec) -> tuple[C.RunConfig, dict[str, Any]]:
    facts = {
        "env_spec": env_spec,
        "arch_digest": "a" * 64,
        "codec_version": 1,
        "codec_table_digest": "b" * 64,
        "torch_version_string": "2.11.0+cu128",
        "device_kind": "cuda:A Card:sm_86",
    }
    return C.RunConfig(env=mock_env_spec), facts


def _identity(base: tuple[C.RunConfig, dict[str, Any]], build: I.EngineBuild) -> I.RunIdentity:
    config, facts = base
    return I.compute_identity(config, build=build, **facts)


def test_same_data_different_binary_is_a_different_run(base: Any) -> None:
    """The measured case: identical data digests, two binaries. It used to be one run id."""
    a, b = _build(BINARY_A), _build(BINARY_B)
    assert (a.build_digest, a.calibration_digest) == (b.build_digest, b.calibration_digest)

    first, second = _identity(base, a), _identity(base, b)
    assert I.run_id(first) != I.run_id(second)
    assert set(I.identity_differences(first, second)) == {"engine_build"}


# -- resuming across the change ----------------------------------------------------------------


def _manifest(identity: I.RunIdentity) -> Any:
    return SimpleNamespace(identity=identity, config={})


def _recorded_before_the_field(identity: I.RunIdentity) -> I.RunIdentity:
    """Round-trip through JSON with the field removed, the way an old ``identity.json`` reads."""
    raw = msgspec.json.decode(msgspec.json.encode(identity))
    del raw["engine_build"]["binary_sha256"]
    return msgspec.convert(raw, I.RunIdentity)


def test_a_checkpoint_written_before_the_binary_was_recorded_still_decodes(base: Any) -> None:
    old = _recorded_before_the_field(_identity(base, _build(BINARY_A)))
    assert old.engine_build.binary_sha256 == I.NOT_RECORDED


def test_it_resumes_and_says_it_could_not_check(
    base: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unrecorded is neither a match nor a mismatch.

    Refusing would strand every checkpoint written before today for a difference nobody can
    show. Passing in silence would claim a check that never happened. So it passes, and says so.
    """
    now = _identity(base, _build(BINARY_A))
    old = _recorded_before_the_field(now)

    check_resume(_manifest(old), now, {})

    said = capsys.readouterr().out
    assert "engine binary" in said and "cannot" in said, said


def test_an_unrecorded_binary_does_not_hide_a_difference_in_the_rest_of_the_build(
    base: Any,
) -> None:
    """Masking the binary must mask the binary and nothing else."""
    now = _identity(base, _build(BINARY_A))
    old = _recorded_before_the_field(now)
    old = msgspec.structs.replace(
        old,
        engine_build=msgspec.structs.replace(old.engine_build, build_digest="f" * 16),
    )
    with pytest.raises(IdentityMismatch):
        check_resume(_manifest(old), now, {})


def test_a_recorded_binary_that_differs_is_refused(base: Any) -> None:
    before = _identity(base, _build(BINARY_A))
    after = _identity(base, _build(BINARY_B))
    with pytest.raises(IdentityMismatch) as refused:
        check_resume(_manifest(before), after, {})
    assert "engine_build" in refused.value.differences


# -- the workers ---------------------------------------------------------------------------


def _report(worker: int, binary: str) -> Any:
    return SimpleNamespace(worker=worker, engine_binary_sha256=binary)


def test_workers_that_agree_with_the_identity_pass() -> None:
    from royalelearn.rollout.farm import check_worker_binaries

    check_worker_binaries(BINARY_A, [_report(0, BINARY_A), _report(1, BINARY_A)])


def test_a_worker_on_another_binary_is_refused_and_named() -> None:
    """Every worker's hash is in the message, so the odd one out is readable at a glance."""
    from royalelearn.rollout.farm import check_worker_binaries

    with pytest.raises(PreflightError) as refused:
        check_worker_binaries(BINARY_A, [_report(0, BINARY_A), _report(1, BINARY_B)])
    message = str(refused.value)
    assert "worker 1" in message and BINARY_B in message and BINARY_A in message


def test_an_engine_without_a_binary_is_not_held_to_one() -> None:
    """MockEngine states none in the parent and none in a worker; there is nothing to compare."""
    from royalelearn.rollout.farm import check_worker_binaries

    check_worker_binaries(I.NOT_STATED, [_report(0, I.NOT_STATED)])


def test_the_restart_path_checks_the_replacement_against_the_run() -> None:
    """A restarted worker is the one that can load a rebuilt file; it is checked like the rest."""
    from royalelearn.rollout.farm import check_worker_binaries

    with pytest.raises(PreflightError, match=r"replacement|restart|changed"):
        check_worker_binaries(BINARY_A, [_report(3, BINARY_B)], restarted=True)
