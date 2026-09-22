"""The config tree: it round-trips, it refuses a typo wherever one is made, its hash does not
depend on how the JSON was written, and every shipped profile is internally consistent."""

from __future__ import annotations

import json
import subprocess
import sys

import msgspec
import pytest

from royalelearn import config as C
from royalelearn.errors import PreflightError


@pytest.mark.parametrize("name", sorted(C.PROFILES))
def test_every_profile_is_consistent(name: str) -> None:
    assert C.check_consistency(C.profile(name)) == []


@pytest.mark.parametrize("name", sorted(C.PROFILES))
def test_every_profile_collects_what_it_asks_for(name: str) -> None:
    """The rectangle has to hold at least ``timesteps_per_iteration`` learner transitions, and
    a batch has to be a whole number of minibatches for the minibatch to be a memory knob."""
    config = C.profile(name)
    geometry = C.geometry(config)
    assert geometry.cycles * geometry.learner_rows >= config.ppo.timesteps_per_iteration
    assert config.ppo.batch_size % config.ppo.minibatch_size == 0
    assert geometry.n_slots == 2 * geometry.n_battles
    assert geometry.games_per_shard * geometry.shards_per_worker == geometry.games_per_worker


@pytest.mark.parametrize("name", sorted(C.PROFILES))
def test_every_profile_round_trips(name: str) -> None:
    config = C.profile(name)
    text = C.dump_config(config)
    assert C.load_config(text) == config
    assert C.dump_config(C.load_config(text)) == text


def test_the_dump_is_canonical_and_idempotent() -> None:
    config = C.laptop()
    once = C.dump_config(config)
    twice = C.dump_config(C.load_config(once))
    assert once == twice
    assert json.loads(once) == json.loads(twice)


def test_the_hash_ignores_key_order() -> None:
    """A reformatted file is the same run."""
    config = C.laptop()
    document = json.loads(C.dump_config(config))
    shuffled = dict(reversed(list(document.items())))
    shuffled["rollout"] = dict(reversed(list(document["rollout"].items())))
    assert C.config_hash(C.load_config(json.dumps(shuffled))) == C.config_hash(config)


def test_the_hash_moves_when_a_value_does() -> None:
    config = C.laptop()
    other = msgspec.structs.replace(config, master_seed=config.master_seed + 1)
    assert C.config_hash(other) != C.config_hash(config)


@pytest.mark.parametrize(
    "document",
    [
        '{"bogus": 1}',
        '{"rollout": {"workerz": 4}}',
        '{"ppo": {"lr_backoff": {"patients": 3}}}',
        '{"ladder": {"gate": {"champion_gamez": 10}}}',
        '{"metrics": {"sinks": [{"kind": "jsonl", "enabledd": true}]}}',
        '{"env": {"engine": {"cls": "royalegym.mock_engine.MockEngine", "kwarg": {}}}}',
        '{"ppo": {"ent_coef": {"kind": "linear", "start": 1, "end": 0, "over_env_stepz": 1}}}',
    ],
    ids=[
        "top",
        "rollout",
        "ppo.lr_backoff",
        "ladder.gate",
        "metrics.sinks",
        "env.engine",
        "ppo.ent_coef.schedule",
    ],
)
def test_a_typo_is_refused_at_every_level(document: str) -> None:
    with pytest.raises(msgspec.ValidationError):
        C.load_config(document)


def test_a_document_overlays_the_profile_it_names() -> None:
    """Setting one number keeps the rest of that machine class, not the laptop's."""
    config = C.load_config('{"profile": "workstation", "rollout": {"workers": 4}}')
    workstation = C.workstation()
    assert config.rollout.workers == 4
    assert config.rollout.games_per_worker == workstation.rollout.games_per_worker
    assert config.rollout.overlap == workstation.rollout.overlap
    assert config.net.channels == workstation.net.channels


def test_an_empty_document_is_the_laptop_profile() -> None:
    assert C.load_config("{}") == C.laptop()


def test_a_list_is_replaced_rather_than_merged() -> None:
    config = C.load_config('{"metrics": {"sinks": [{"kind": "jsonl"}]}}')
    assert [sink.kind for sink in config.metrics.sinks] == ["jsonl"]


def test_an_unknown_profile_is_refused_by_name() -> None:
    with pytest.raises(PreflightError, match="nosuch"):
        C.profile("nosuch")


def test_validate_names_everything_that_is_wrong() -> None:
    broken = msgspec.structs.replace(
        C.laptop(),
        ppo=msgspec.structs.replace(C.PPOConfig(), minibatch_size=300),
        determinism=C.DeterminismConfig(tier="fast"),
        ladder=msgspec.structs.replace(C.LadderConfig(), mix=(0.5, 0.2, 0.2)),
    )
    problems = C.check_consistency(broken)
    assert len(problems) == 3
    with pytest.raises(PreflightError) as excinfo:
        C.validate(broken)
    for problem in problems:
        assert problem in str(excinfo.value)


def test_validate_returns_a_good_config() -> None:
    config = C.laptop()
    assert C.validate(config) is config


def test_geometry_refuses_shards_that_do_not_divide_the_games() -> None:
    config = msgspec.structs.replace(
        C.laptop(), rollout=msgspec.structs.replace(C.RolloutConfig(), games_per_worker=33)
    )
    with pytest.raises(ValueError, match="divisible"):
        C.geometry(config)
    assert any("divisible" in problem for problem in C.check_consistency(config))


def test_the_learner_row_fraction_follows_the_mixture() -> None:
    """Both seats of a mirror battle are the learner's; one seat of every other battle is."""
    config = C.laptop()
    geometry = C.geometry(config)
    mirror = config.ladder.mix[0]
    assert geometry.learner_row_fraction == pytest.approx(mirror + (1 - mirror) / 2)
    assert geometry.learner_rows == int(geometry.n_slots * geometry.learner_row_fraction)

    all_mirror = msgspec.structs.replace(
        config, ladder=msgspec.structs.replace(C.LadderConfig(), mix=(1.0, 0.0, 0.0))
    )
    assert C.geometry(all_mirror).learner_rows == C.geometry(all_mirror).n_slots


def test_a_jsonl_sink_is_required() -> None:
    config = C.load_config('{"metrics": {"sinks": [{"kind": "console"}]}}')
    assert any("jsonl" in problem for problem in C.check_consistency(config))


def test_the_default_environment_is_the_rust_engine_with_a_step_limit() -> None:
    env = C.laptop().env
    assert env.engine.cls == C.RUST_ENGINE
    assert [spec.cls for spec in env.termination] == [
        "royalegym.done_condition.GameOverCondition"
    ]
    assert env.truncation[0].kwargs["max_steps"] > 0


def test_a_config_loads_from_a_mapping_and_from_bytes() -> None:
    document = json.loads(C.dump_config(C.laptop()))
    assert C.load_config(document) == C.laptop()
    assert C.load_config(C.dump_config(C.laptop()).encode("utf-8")) == C.laptop()


def test_a_config_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(PreflightError, match="JSON object"):
        C.load_config("[1, 2, 3]")


def test_schedules_round_trip_through_their_tag() -> None:
    config = C.load_config('{"ppo": {"ent_coef": {"kind": "constant", "value": 0.005}}}')
    assert config.ppo.ent_coef == C.ConstantSpec(0.005)
    assert C.load_config(C.dump_config(config)).ppo.ent_coef == C.ConstantSpec(0.005)


def test_a_schedule_of_another_kind_replaces_rather_than_merges() -> None:
    # The laptop profile's ent_coef is linear; naming a constant one must not leave its
    # endpoints behind, which is what a field-by-field merge of the two would do.
    assert isinstance(C.laptop().ppo.ent_coef, C.LinearSpec)
    config = C.load_config('{"ppo": {"ent_coef": {"kind": "constant", "value": 0.005}}}')
    assert json.loads(C.dump_config(config))["ppo"]["ent_coef"] == {
        "kind": "constant",
        "value": 0.005,
    }


def test_a_schedule_of_the_same_kind_can_be_overridden_field_by_field() -> None:
    base = C.laptop().ppo.ent_coef
    assert isinstance(base, C.LinearSpec)
    config = C.load_config('{"ppo": {"ent_coef": {"kind": "linear", "end": 0.0}}}')
    assert config.ppo.ent_coef == msgspec.structs.replace(base, end=0.0)


TORCH_FREE_MODULES = (
    "royalelearn.api",
    "royalelearn.config",
    "royalelearn.determinism",
    "royalelearn.errors",
    "royalelearn.identity",
    "royalelearn.metrics.schema",
    "royalelearn.obs_layout",
    "royalelearn.rollout.envspec",
    "royalelearn.rollout.layout",
    "royalelearn.seeding",
)


def test_the_spine_imports_without_torch() -> None:
    """The ABCs, the config tree and the seeding run on numpy and msgspec alone.

    In a fresh interpreter, so that another test having imported torch cannot hide a module-scope
    import here, and with this process's environment, so that torch being installed and available
    is exactly the thing the check has to survive.
    """
    lines = [
        "import importlib, sys",
        f"for name in {TORCH_FREE_MODULES!r}:",
        "    importlib.import_module(name)",
        "assert 'torch' not in sys.modules, sorted(sys.modules)",
        "print('clean')",
    ]
    program = "\n".join(lines)
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "clean"


def test_torch_is_importable_in_this_environment_or_the_check_is_vacuous() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("torch is not installed, so the import check above proves less than it says")
    assert probe.stdout.strip()
