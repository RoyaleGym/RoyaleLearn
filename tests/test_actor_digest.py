"""``actor_digest``: what an actor's weights compute, for loading them into another run.

``arch_digest`` is the run's architecture whole: what its checkpoints and its resume are held
to. It includes settings that do not change what a set of actor weights computes -- the autocast
precision, the device, the initialisation, whether the head runs in float32 -- and the critic's
shape, which an actor folder does not carry. A warm start held to it refuses a bfloat16 student
the float32 clone it was meant to start from.

``actor_digest`` leaves those out and keeps everything else, ``net.noop_bias`` included: it is a
constant added in the forward, so it changes what the weights compute. ``check_compatible``
compares it when both sides state one, and ``arch_digest`` otherwise, as before.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

from royalelearn.errors import IdentityMismatch  # noqa: E402
from royalelearn.ladder.snapshots import SnapshotSpec, check_compatible  # noqa: E402
from royalelearn.learn.nets import actor_digest_of, arch_digest_of  # noqa: E402
from royalelearn.testing import ARCH  # noqa: E402

BASE = msgspec.structs.replace(ARCH, separate_trunks=True)

#: What a set of actor weights computes the same under, and an actor folder does not carry.
NEUTRAL = {
    "autocast_dtype": "bfloat16",
    "device": "cuda",
    "init": "orthogonal",
    "policy_head_float32": True,
    "factored_act_init": 0.3,
    "value_hidden": 7,
    "critic_channels": 4,
    "critic_blocks": 0,
}
#: What changes the actor's shape or what its weights compute.
DECISIVE = {"channels": 12, "card_embed": 12, "blocks": 2, "noop_bias": 8.0, "trunk_stride": 2}


@pytest.mark.parametrize(("field", "value"), sorted(NEUTRAL.items()))
def test_a_setting_the_weights_do_not_feel_leaves_the_actor_digest(
    env_spec: Any, field: str, value: Any
) -> None:
    """Plant: digest the arch whole, and a bfloat16 student refuses its float32 clone."""
    changed = msgspec.structs.replace(BASE, **{field: value})
    assert actor_digest_of(env_spec, changed) == actor_digest_of(env_spec, BASE)


@pytest.mark.parametrize(("field", "value"), sorted(DECISIVE.items()))
def test_a_setting_the_weights_do_feel_moves_it(env_spec: Any, field: str, value: Any) -> None:
    """``noop_bias`` among them. Plant: neutralise it like ``init``, and a clone trained under
    one bias loads into a run that adds another to every no-op logit."""
    fields = {field: value}
    if field == "channels":
        fields["card_embed"] = value
    changed = msgspec.structs.replace(BASE, **fields)
    assert actor_digest_of(env_spec, changed) != actor_digest_of(env_spec, BASE)


def test_the_arch_digest_is_what_it_was(env_spec: Any) -> None:
    """The run's identity is untouched: ``arch_digest_of`` is the factory's ``arch_digest``."""
    from royalelearn.learn.nets import DefaultNetworkFactory

    assert arch_digest_of(env_spec, BASE) == DefaultNetworkFactory(1).arch_digest(env_spec, BASE)
    bf16 = msgspec.structs.replace(BASE, autocast_dtype="bfloat16")
    assert arch_digest_of(env_spec, bf16) != arch_digest_of(env_spec, BASE)


def _spec(arch_digest: str, actor_digest: Any = msgspec.UNSET) -> SnapshotSpec:
    return SnapshotSpec(
        snapshot_id="x",
        arch_digest=arch_digest,
        actor_digest=actor_digest,
        obs_digest="obs",
        codec_table_digest="table",
    )


def test_compatibility_reads_the_actor_digest_when_both_sides_state_one() -> None:
    check_compatible(_spec("float32-run", "actor"), _spec("bfloat16-run", "actor"))
    with pytest.raises(IdentityMismatch) as refused:
        check_compatible(_spec("same", "actor-a"), _spec("same", "actor-b"))
    assert set(refused.value.differences) == {"actor_digest"}


def test_without_it_on_either_side_compatibility_is_what_it_was() -> None:
    """A folder written before the field existed is held to ``arch_digest``, as it always was."""
    for stored, current in (
        (_spec("float32-run"), _spec("bfloat16-run", "actor")),
        (_spec("float32-run", "actor"), _spec("bfloat16-run")),
    ):
        with pytest.raises(IdentityMismatch) as refused:
            check_compatible(stored, current)
        assert set(refused.value.differences) == {"arch_digest"}


def test_an_unstated_actor_digest_is_not_written() -> None:
    """Every folder written before keeps its bytes, and so its sha256."""
    assert b"actor_digest" not in msgspec.json.encode(_spec("a"))
    assert b'"actor_digest":"d"' in msgspec.json.encode(_spec("a", "d"))


def test_a_runs_actor_folders_state_their_actor_digest(tmp_path: Any) -> None:
    from royalelearn.testing import coordinator, tiny_config

    config = tiny_config(tmp_path)
    with coordinator(config) as run:
        spec = run.artifact_spec()
        assert spec.actor_digest == actor_digest_of(run.spec, config.net)
        assert run.snapshot_template.actor_digest == spec.actor_digest
