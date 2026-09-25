"""The environment is identified by what it IS, not by how its config happened to be spelled.

The run identity and the ladder's result context both hashed the env spec as written. So
``"kwargs": {}`` and the same reward with its default weights written out were two identities,
``{"crown": 1}`` and ``{"crown": 1.0}`` were two, and -- the dangerous direction -- changing a
default in the code changed the objective of every config that relied on it without moving a
single digest. Found by the train session's review of the 2026-09-24 placeholder commits.

Now each component's kwargs are bound to its real signature, defaults applied and numbers
normalised, before hashing. These tests pin all three properties and the wiring into both places
the digest is read.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

import royalelearn.config as cfg
from royalelearn.rollout.envspec import ComponentSpec, env_value_digest

REWARD = "royalelearn.rewards.default_potential_reward"


def _with_reward(kwargs: dict[str, Any]) -> Any:
    spec = cfg.default_env_spec(cfg.MOCK_ENGINE)
    return msgspec.structs.replace(spec, reward_fn=ComponentSpec(REWARD, kwargs))


def test_the_defaults_written_out_are_the_defaults_left_out() -> None:
    assert env_value_digest(_with_reward({})) == env_value_digest(
        _with_reward({"crown": 0.2, "tower_hp": 0.1, "elixir": 0.05})
    )
    # and the literal spelling still differs, which is exactly what this replaces
    assert _with_reward({}).digest() != _with_reward({"crown": 0.2}).digest()


def test_an_integer_and_the_same_float_are_one_value() -> None:
    assert env_value_digest(_with_reward({"crown": 1})) == env_value_digest(
        _with_reward({"crown": 1.0})
    )


def test_a_different_weight_is_a_different_environment() -> None:
    """The control: normalising must not collapse values that really differ."""
    assert env_value_digest(_with_reward({"crown": 0.3})) != env_value_digest(_with_reward({}))


def test_changing_a_default_in_the_code_moves_the_digest_of_a_config_that_relied_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direction that was silent: every ``{}`` config's objective moved and nothing said so."""
    from royalelearn import rewards

    before = env_value_digest(_with_reward({}))
    defaults = dict(rewards.default_potential_reward.__kwdefaults__)
    monkeypatch.setattr(
        rewards.default_potential_reward, "__kwdefaults__", {**defaults, "elixir": 0.3}
    )
    assert env_value_digest(_with_reward({})) != before


def test_every_component_is_bound_not_only_the_reward() -> None:
    """The truncation's step limit is written out in the shipped spec; left to its default it is
    the same environment only if the default is that number, and a different one otherwise."""
    spec = cfg.default_env_spec(cfg.MOCK_ENGINE)
    limit = spec.truncation[0]
    doubled = msgspec.structs.replace(
        spec,
        truncation=[
            ComponentSpec(limit.cls, {**limit.kwargs, "max_steps": 2 * limit.kwargs["max_steps"]})
        ],
    )
    assert env_value_digest(doubled) != env_value_digest(spec)


def test_the_run_identity_reads_the_value_digest(env_spec: Any) -> None:
    """Through ``compute_identity``, the function that fills the field, not the helper it calls.

    The first version of this test called the helper directly, so ``compute_identity`` could go
    back to hashing the spelling with this test still green; that plant was not caught.
    """
    from royalelearn import identity

    facts = {
        "env_spec": env_spec,
        "build": identity.EngineBuild(
            engine_class="royalegym.mock_engine.MockEngine",
            calibration_digest="0" * 16,
            build_digest="0" * 16,
            catalogue_sha256="c" * 64,
            path_search=None,
            stale_build_differences=[],
            binary_sha256=identity.NOT_STATED,
        ),
        "arch_digest": "a" * 64,
        "codec_version": 1,
        "codec_table_digest": "b" * 64,
        "torch_version_string": "2.11.0+cu128",
        "device_kind": "cpu:x86_64",
    }
    left_out = identity.compute_identity(cfg.RunConfig(env=_with_reward({})), **facts)
    written_out = identity.compute_identity(
        cfg.RunConfig(env=_with_reward({"crown": 0.2, "tower_hp": 0.1, "elixir": 0.05})),
        **facts,
    )
    assert left_out.env_spec_digest == written_out.env_spec_digest
    assert identity.run_id(left_out) == identity.run_id(written_out)


def test_the_ladder_context_reads_the_value_digest(tmp_path: Any) -> None:
    """The second place the env is identified. Two configs that differ only in spelling must
    file their games under ONE context, or a resumed run would stop seeing its own ladder."""
    from test_coordinator import coordinator, tiny_config

    contexts = []
    for kwargs in ({}, {"crown": 0.2, "tower_hp": 0.1, "elixir": 0.05}):
        config = tiny_config(tmp_path / str(len(contexts)))
        config = msgspec.structs.replace(
            config,
            env=msgspec.structs.replace(config.env, reward_fn=ComponentSpec(REWARD, kwargs)),
        )
        with coordinator(config) as run:
            contexts.append(run.context)
    assert contexts[0] == contexts[1]
