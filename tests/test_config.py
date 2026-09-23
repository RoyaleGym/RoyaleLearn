"""The config tree: it round-trips, it refuses a typo wherever one is made, its hash does not
depend on how the JSON was written, and every shipped profile is internally consistent."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

from royalelearn import config as C
from royalelearn.errors import PreflightError

EXAMPLE_CONFIGS = Path(__file__).resolve().parents[1] / "examples" / "configs"


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


def _leaves(tree: Any, prefix: str = "") -> dict[str, Any]:
    """A decoded config as ``{"ppo.minibatch_size": 256, ...}``, so a disagreement names the
    field rather than printing two trees of two hundred leaves side by side. Lists are leaves:
    the loader replaces a list whole, so a list is one setting."""
    if isinstance(tree, dict):
        leaves: dict[str, Any] = {}
        for key, value in tree.items():
            leaves.update(_leaves(value, f"{prefix}.{key}" if prefix else key))
        return leaves
    return {prefix: tree}


def _disagreements(ours: dict[str, Any], theirs: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    absent = "<absent>"
    return {
        key: (ours.get(key, absent), theirs.get(key, absent))
        for key in sorted(set(ours) | set(theirs))
        if ours.get(key, absent) != theirs.get(key, absent)
    }


@pytest.mark.parametrize("name", ["laptop", "workstation"])
def test_the_shipped_example_is_its_profile_written_out_in_full(name: str) -> None:
    """``examples/configs/<profile>.json`` and ``--profile <profile>`` are the same run.

    They are two routes to one machine class and nothing forces them to agree: the file is what
    ``train --config`` runs, the profile is what ``config``, ``doctor`` and ``bench`` use when no
    file is given. They did drift -- the file carried the measured minibatch while the code kept
    the one that spills past a 4 GB card -- so a doctor run without ``--config`` checked a
    configuration nobody trains.

    Two comparisons, because each is blind where the other sees. Loading the file overlays it on
    the profile, so a field the file leaves out is filled from the code and the loaded tree cannot
    notice the omission; the raw document can, and a field the file omits is a setting its reader
    cannot see. No field is excluded: the run name is the default in both.
    """
    path = EXAMPLE_CONFIGS / f"{name}.json"
    shipped = C.profile(name)
    profile_leaves = _leaves(json.loads(C.dump_config(shipped)))

    loaded = C.load_config(path)
    assert _disagreements(profile_leaves, _leaves(json.loads(C.dump_config(loaded)))) == {}
    assert loaded == shipped

    written = _leaves(json.loads(path.read_text(encoding="utf-8")))
    assert _disagreements(profile_leaves, written) == {}


def test_each_profiles_minibatch_is_the_value_it_was_chosen_at() -> None:
    """The measured values, written down, so that the pair above cannot agree on a wrong one.

    ``test_the_shipped_example_is_its_profile_written_out_in_full`` compares the code with the
    file. That catches drift between them and nothing else: move both to 512 and it still passes,
    which is exactly the state this repo was in until 582ce96. A number that came from a
    measurement belongs in a test beside the measurement.

    laptop 256: measured on the 4 GB card (``docs/harness-spec.md`` section 6). 512 reserved 4243
    MB of a 4294 MB card and took 180-233 s an update against 47-49 s at 256. workstation 2048 and
    many_core 8192 are UNMEASURED: no such machine has run one. They are here to be changed by a
    measurement rather than by an edit, and on a card too small for them the preflight refuses the
    run before it starts.
    """
    assert C.profile("laptop").ppo.minibatch_size == 256
    assert C.profile("workstation").ppo.minibatch_size == 2048
    assert C.profile("many_core").ppo.minibatch_size == 8192


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


def test_every_collected_row_reaches_the_actor_unless_a_run_asks_otherwise() -> None:
    """``ppo.forced_rows`` lands at ``all``, which is the update every run so far has taken.

    The two other values change what the actor is trained on, so neither is arrived at by
    upgrading: a run that wants one names it, and the run identity then says which it had.
    """
    assert C.PPOConfig().forced_rows == "all"
    for name in sorted(C.PROFILES):
        assert C.profile(name).ppo.forced_rows == "all"


@pytest.mark.parametrize("value", ["critic", "none", "", "ALL"])
def test_a_forced_row_arm_that_does_not_exist_is_refused_by_name(value: str) -> None:
    """The three arms are three different updates, and a fourth spelling is not a fourth arm.

    It is a string rather than an enum because every other choice in this file is -- see
    ``rollout.source`` and ``determinism.tier`` -- and the check is what makes the string safe.
    """
    config = msgspec.structs.replace(
        C.laptop(), ppo=msgspec.structs.replace(C.PPOConfig(), forced_rows=value)
    )

    problems = C.check_consistency(config)

    assert len(problems) == 1
    assert repr(value) in problems[0]
    assert all(arm in problems[0] for arm in C.FORCED_ROW_ARMS)


@pytest.mark.parametrize("value", ["critic_only", "critic_only_choice_mean"])
def test_letting_the_actor_skip_rows_needs_a_trunk_it_does_not_share(value: str) -> None:
    """A skipped row is only skipped if the forward it skips was the actor's alone.

    With one trunk under both heads the critic's forward IS the actor's, so there is nothing to
    save and the value loss of a skipped row still reaches the parameters the policy gradient
    uses. Only the policy head could be left out, and under the choice-mean arm the two terms
    would then be weighted against each other inside the trunk by the forced fraction, which is
    a quantity the elixir economy moves from iteration to iteration.
    """
    config = msgspec.structs.replace(
        C.laptop(),
        ppo=msgspec.structs.replace(C.PPOConfig(), forced_rows=value),
        net=msgspec.structs.replace(C.NetConfig(), separate_trunks=False),
    )

    problems = C.check_consistency(config)

    assert len(problems) == 1
    assert "net.separate_trunks" in problems[0] and value in problems[0]
    allowed = msgspec.structs.replace(config, net=C.NetConfig())
    assert C.check_consistency(allowed) == [], "the same arm is fine with a trunk each"


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


def test_the_learner_rows_are_a_count_of_battles_and_not_a_share() -> None:
    """Both seats of a mirror battle are the learner's; one seat of every other battle is.

    The mirror battles are a count, so the rows are a count: the rectangle is the size it is
    rather than the size it is expected to be.
    """
    config = C.laptop()
    geometry = C.geometry(config)
    assert geometry.mirror_battles == 48
    assert geometry.learner_rows == geometry.n_battles + geometry.mirror_battles
    assert geometry.learner_row_fraction == geometry.learner_rows / geometry.n_slots

    all_mirror = msgspec.structs.replace(
        config, ladder=msgspec.structs.replace(C.LadderConfig(), mix=(1.0, 0.0, 0.0))
    )
    assert C.geometry(all_mirror).learner_rows == C.geometry(all_mirror).n_slots


def test_the_rectangle_is_the_size_the_matchmaker_actually_fills() -> None:
    """``geometry`` and ``MixMatchmaker`` must not each do this arithmetic their own way.

    They share ``role_counts``; what this checks is that the matchmaker's own assignments add
    up to the number the config sized the iteration from, which is the claim that recomputing
    the arithmetic in two places could not make.
    """
    from royalelearn.api.rollout import ROLE_MIRROR
    from royalelearn.ladder.matchmaker import MixMatchmaker

    for name in sorted(C.PROFILES):
        config = C.profile(name)
        geometry = C.geometry(config)
        matchmaker = MixMatchmaker(11, config.ladder, n_battles=geometry.n_battles)
        mirror = sum(
            matchmaker.role_of(battle) == ROLE_MIRROR for battle in range(geometry.n_battles)
        )
        assert mirror == geometry.mirror_battles
        assert geometry.n_battles + mirror == geometry.learner_rows
        assert matchmaker.learner_rows == geometry.learner_rows
        assert geometry.cycles * geometry.learner_rows >= config.ppo.timesteps_per_iteration


def test_the_role_counts_sum_to_the_battles_on_every_mixture() -> None:
    """Whatever the mixture and however few the battles, the three counts are the rectangle."""
    mixes = [
        (0.50, 0.35, 0.15),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.33, 0.33, 0.34),
        (0.9, 0.05, 0.05),
        (0.05, 0.05, 0.90),
    ]
    for mix in mixes:
        for n_battles in range(1, 130):
            counts = C.role_counts(mix, n_battles)
            assert sum(counts) == n_battles
            assert all(count >= 0 for count in counts)
            for count, share in zip(counts, mix, strict=True):
                assert abs(count - share * n_battles) <= 1.0


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
