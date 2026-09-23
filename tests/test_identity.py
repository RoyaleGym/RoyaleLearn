"""The run identity: order-independent, and moved by exactly the fields that can move a number.

The table below is the whole point of the file. Every included field is varied on its own and
must change the run id; every excluded field is varied on its own and must not. A field that
moves from one list to the other is a decision, and it fails here until it is made deliberately.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import msgspec
import pytest

from royalelearn import config as C
from royalelearn import identity as I
from royalelearn.api.rollout import EnvSpec
from royalelearn.rollout.envspec import EnvFactorySpec

FACTS: dict[str, Any] = {
    "arch_digest": "a" * 64,
    "codec_version": 1,
    "codec_table_digest": "b" * 64,
    "torch_version_string": "2.11.0+cu128",
    "device_kind": "cuda:A Card:sm_86",
}


def _build() -> I.EngineBuild:
    return I.EngineBuild(
        engine_class="royalegym.mock_engine.MockEngine",
        calibration_digest="0123456789abcdef",
        build_digest="0123456789abcdef",
        catalogue_sha256="c" * 64,
        path_search=None,
        stale_build_differences=[],
    )


@pytest.fixture
def base(env_spec: EnvSpec, mock_env_spec: EnvFactorySpec) -> tuple[C.RunConfig, dict[str, Any]]:
    config = C.RunConfig(env=mock_env_spec)
    facts = {"env_spec": env_spec, "build": _build(), **FACTS}
    return config, facts


def _identity(config: C.RunConfig, facts: dict[str, Any]) -> I.RunIdentity:
    return I.compute_identity(config, **facts)


def _id(config: C.RunConfig, facts: dict[str, Any]) -> str:
    return I.run_id(_identity(config, facts))


# ``name -> change``. Each change returns a new (config, facts) pair differing in one thing.
Change = Callable[[C.RunConfig, dict[str, Any]], tuple[C.RunConfig, dict[str, Any]]]


def _config_change(**fields: Any) -> Change:
    return lambda config, facts: (msgspec.structs.replace(config, **fields), facts)


def _fact_change(**fields: Any) -> Change:
    return lambda config, facts: (config, {**facts, **fields})


def _frame_stack(config: C.RunConfig, facts: dict[str, Any]) -> tuple[C.RunConfig, dict[str, Any]]:
    """The frame stack is in the config and in the spec that was read at it; both move."""
    spec = msgspec.structs.replace(facts["env_spec"], frame_stack=2)
    spec = msgspec.structs.replace(spec, obs_digest=spec.obs_digest[::-1])
    return (
        msgspec.structs.replace(config, obs=C.ObsConfig(frame_stack=2)),
        {**facts, "env_spec": spec},
    )


def _rollout_change(**fields: Any) -> Change:
    """One rollout field, with the rest of the block left at its default."""
    return _config_change(rollout=msgspec.structs.replace(C.RolloutConfig(), **fields))


def _other_env(config: C.RunConfig, facts: dict[str, Any]) -> tuple[C.RunConfig, dict[str, Any]]:
    env = msgspec.structs.replace(config.env, decision_ms=250)
    return msgspec.structs.replace(config, env=env), facts


INCLUDED: dict[str, Change] = {
    "master_seed": _config_change(master_seed=7),
    "env": _other_env,
    "obs.frame_stack": _frame_stack,
    "ppo.n_epochs": _config_change(ppo=msgspec.structs.replace(C.PPOConfig(), n_epochs=4)),
    "ppo.lr_actor": _config_change(ppo=msgspec.structs.replace(C.PPOConfig(), lr_actor=1e-4)),
    # Two runs on different forced-row arms optimise different objectives on the same rows, so
    # they are not the same run and a resume across the change is refused by name.
    "ppo.forced_rows": _config_change(
        ppo=msgspec.structs.replace(C.PPOConfig(), forced_rows="critic_only_choice_mean")
    ),
    "advantage.gae_lambda": _config_change(
        advantage=msgspec.structs.replace(C.AdvantageConfig(), gae_lambda=0.95)
    ),
    "rollout.workers": _rollout_change(workers=4),
    "rollout.games_per_worker": _rollout_change(games_per_worker=16),
    "rollout.shards_per_worker": _rollout_change(shards_per_worker=1),
    "ladder.mix": _config_change(
        ladder=msgspec.structs.replace(C.LadderConfig(), mix=(0.4, 0.45, 0.15))
    ),
    "ladder.gate.champion_lower_bound": _config_change(
        ladder=msgspec.structs.replace(
            C.LadderConfig(),
            gate=msgspec.structs.replace(C.GateConfig(), champion_lower_bound=0.55),
        )
    ),
    "ladder.eval_seed_count": _config_change(
        ladder=msgspec.structs.replace(C.LadderConfig(), eval_seed_count=250)
    ),
    "determinism.tier": _config_change(determinism=C.DeterminismConfig(tier="throughput")),
    "arch_digest": _fact_change(arch_digest="d" * 64),
    "codec_version": _fact_change(codec_version=2),
    "codec_table_digest": _fact_change(codec_table_digest="e" * 64),
    "torch_version": _fact_change(torch_version_string="2.10.0+cu124"),
    "device_kind": _fact_change(device_kind="cpu:x86_64"),
}

EXCLUDED: dict[str, Change] = {
    "run_name": _config_change(run_name="another-name"),
    "runs_dir": _config_change(runs_dir="/somewhere/else"),
    "timestep_limit": _config_change(timestep_limit=1),
    "extra_component_modules": _config_change(extra_component_modules=["mypackage.components"]),
    "rollout.source": _rollout_change(source="inline"),
    "rollout.spin_us": _rollout_change(spin_us=0),
    "rollout.round_timeout_s": _rollout_change(round_timeout_s=120.0),
    "rollout.restart_failed_workers": _rollout_change(restart_failed_workers=False),
    "rollout.max_restarts_per_worker": _rollout_change(max_restarts_per_worker=1),
    "rollout.launch_delay_s": _rollout_change(launch_delay_s=0.0),
    "rollout.stagger_first_reset": _rollout_change(stagger_first_reset=False),
    "rollout.overlap": _rollout_change(overlap=True),
    "rollout.eval_workers": _rollout_change(eval_workers=1),
    "rollout.eval_games_per_worker": _rollout_change(eval_games_per_worker=8),
    "checkpoint.keep": _config_change(
        checkpoint=msgspec.structs.replace(C.CheckpointConfig(), keep=2)
    ),
    "metrics.image_every": _config_change(
        metrics=msgspec.structs.replace(C.MetricsConfig(), image_every=5)
    ),
    "alarms.kl_high": _config_change(alarms=msgspec.structs.replace(C.AlarmConfig(), kl_high=0.5)),
    "doctor.ram_budget_mb": _config_change(
        doctor=msgspec.structs.replace(C.DoctorConfig(), ram_budget_mb=1000)
    ),
}


@pytest.mark.parametrize("name", sorted(INCLUDED))
def test_an_included_field_changes_the_run_id(
    base: tuple[C.RunConfig, dict[str, Any]], name: str
) -> None:
    config, facts = base
    changed_config, changed_facts = INCLUDED[name](config, facts)
    assert _id(changed_config, changed_facts) != _id(config, facts)


@pytest.mark.parametrize("name", sorted(EXCLUDED))
def test_an_excluded_field_does_not_change_the_run_id(
    base: tuple[C.RunConfig, dict[str, Any]], name: str
) -> None:
    config, facts = base
    changed_config, changed_facts = EXCLUDED[name](config, facts)
    assert _id(changed_config, changed_facts) == _id(config, facts)


def test_the_two_lists_cover_the_documented_exclusions() -> None:
    """Everything ``EXCLUDED_FROM_IDENTITY`` names is either tested here or a whole block of it."""
    for field in I.EXCLUDED_FROM_IDENTITY:
        assert any(name == field or name.startswith(field + ".") for name in EXCLUDED) or any(
            name.startswith(field) for name in EXCLUDED
        ), field


def test_the_engine_build_is_in_the_identity(base: tuple[C.RunConfig, dict[str, Any]]) -> None:
    config, facts = base
    other = msgspec.structs.replace(facts["build"], build_digest="fedcba9876543210")
    assert _id(config, {**facts, "build": other}) != _id(config, facts)


def test_the_observation_digest_is_in_the_identity(
    base: tuple[C.RunConfig, dict[str, Any]]
) -> None:
    config, facts = base
    spec = msgspec.structs.replace(facts["env_spec"], obs_digest="f" * 64)
    assert _id(config, {**facts, "env_spec": spec}) != _id(config, facts)


def test_a_reveal_changes_the_identity(base: tuple[C.RunConfig, dict[str, Any]]) -> None:
    """A revealed field can leave every width alone, so what catches it is the obs digest."""
    from royalegym.env import ClashSelfPlayVecEnv, EnvFactory
    from royalegym.mock_engine import MockEngine
    from royalegym.obs import Reveal, SpatialObsBuilder
    from royalelearn.rollout.envspec import read_env_spec

    config, facts = base
    factory = EnvFactory(
        engine=MockEngine,
        obs_builder=(SpatialObsBuilder, {"reveal": Reveal(enemy_elixir=True)}),
        decision_ms=config.env.decision_ms,
    )
    revealed_env = ClashSelfPlayVecEnv(1, factory, viser=None)
    try:
        revealed = read_env_spec(revealed_env, config.env, frame_stack=config.obs.frame_stack)
    finally:
        revealed_env.close()
    assert revealed.vector_size == facts["env_spec"].vector_size
    assert _id(config, {**facts, "env_spec": revealed}) != _id(config, facts)


def test_the_identity_json_is_order_independent(
    base: tuple[C.RunConfig, dict[str, Any]]
) -> None:
    config, facts = base
    identity = _identity(config, facts)
    encoded = msgspec.json.encode(identity, order="deterministic")
    decoded = msgspec.json.decode(encoded, type=I.RunIdentity)
    assert decoded == identity
    assert msgspec.json.encode(decoded, order="deterministic") == encoded
    assert I.run_id(decoded) == I.run_id(identity)


def test_the_run_id_is_sixteen_hex_characters(base: tuple[C.RunConfig, dict[str, Any]]) -> None:
    value = _id(*base)
    assert len(value) == 16
    assert set(value) <= set("0123456789abcdef")


def test_differences_name_every_field_that_moved(
    base: tuple[C.RunConfig, dict[str, Any]]
) -> None:
    config, facts = base
    mine = _identity(config, facts)
    theirs = _identity(*INCLUDED["master_seed"](config, facts))
    differences = I.identity_differences(mine, theirs)
    # The seed moves the ladder digest with it: the evaluation seed set is drawn from
    # ``eval/seed_set``, so it is a different set of battles under a different master seed.
    assert set(differences) == {"master_seed", "ladder_digest"}
    assert differences["master_seed"] == (config.master_seed, 7)
    assert I.identity_differences(mine, mine) == {}


def test_a_spec_read_at_another_frame_stack_is_refused(
    base: tuple[C.RunConfig, dict[str, Any]]
) -> None:
    config, facts = base
    spec = msgspec.structs.replace(facts["env_spec"], frame_stack=2)
    with pytest.raises(ValueError, match="frame_stack"):
        _identity(config, {**facts, "env_spec": spec})


def test_the_catalogue_digest_follows_the_cards(vec_env: object) -> None:
    cards = list(vec_env.envs[0].engine.cards())
    assert I.catalogue_digest(cards) == I.catalogue_digest(list(cards))
    assert I.catalogue_digest(cards[:-1]) != I.catalogue_digest(cards)
    assert I.catalogue_digest(cards[::-1]) != I.catalogue_digest(cards)


def test_the_engine_build_reads_the_envs_own_report(vec_env: object) -> None:
    env = vec_env.envs[0]
    build = I.engine_build(env.config(), env.engine.cards())
    assert build.engine_class == "royalegym.mock_engine.MockEngine"
    assert build.calibration_digest == env.config()["calibration_digest"]
    assert build.stale_build_differences == []
    assert build.catalogue_sha256 == I.catalogue_digest(list(env.engine.cards()))


def test_the_device_description_falls_back_to_the_cpu() -> None:
    assert I.describe_device("cpu").startswith("cpu:")
    assert I.torch_version()


def test_royalegym_provenance_is_reported(base: tuple[C.RunConfig, dict[str, Any]]) -> None:
    identity = _identity(*base)
    assert identity.royalegym_version
    assert identity.royalegym_git
    assert identity.royalelearn_version
    assert identity.format_version == I.IDENTITY_FORMAT_VERSION
