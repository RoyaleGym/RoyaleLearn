"""The loop: that it runs, that it records, and that each of its invariants fires.

The helpers at the top are what the other coordinator tests build on. They are here rather than
in a module of their own because what they describe is one thing -- the smallest run that is
still a run -- and a second copy of it would be a second answer to what this harness does per
iteration.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import config as cfg
from royalelearn.determinism import apply_cublas_workspace_config
from royalelearn.metrics import schema
from royalelearn.metrics.records import unknown_keys

pytest.importorskip("torch")
pytest.importorskip("safetensors")

#: What preflight is told for a test: every gate still runs, on a sample a test can afford.
PREFLIGHT = {"table_samples": 32, "mask_samples": 32, "min_table_states": 0}


def tiny_config(tmp_path: Path, **overrides: Any) -> cfg.RunConfig:
    """The smallest run that is still a run: MockEngine, one worker, two battles, four cycles.

    The mixture is mirror-only. That is not a simplification for its own sake: the learner-row
    count of a mirror battle is exactly two, so the rectangle collects exactly what the config
    asks for, and the per-iteration invariant that says so can be asserted rather than given a
    tolerance that only means anything over thousands of battles.
    """
    settings: dict[str, Any] = {
        "run_name": "test",
        "runs_dir": str(tmp_path / "runs"),
        "master_seed": 4242,
        "env": cfg.default_env_spec(cfg.MOCK_ENGINE, max_steps=6),
        "rollout": cfg.RolloutConfig(
            source="inline",
            workers=1,
            games_per_worker=2,
            shards_per_worker=1,
            launch_delay_s=0.0,
        ),
        "net": cfg.NetConfig(
            channels=8,
            blocks=1,
            norm_groups=4,
            card_embed=8,
            value_hidden=16,
            autocast_dtype="float32",
            device="cpu",
        ),
        # ``net.autocast_dtype`` is torch's spelling and ``ppo.ratio_atol``'s keys are the
        # update's names for the same precisions, which agree on bfloat16 and not on float32.
        # Naming both is what lets a CPU run assert the ratio invariant at fp32's tolerance.
        "ppo": cfg.PPOConfig(
            n_epochs=2,
            timesteps_per_iteration=16,
            batch_size=8,
            minibatch_size=4,
            ratio_atol={"fp32": 1e-4, "float32": 1e-4, "bfloat16": 2e-2},
        ),
        "ladder": cfg.LadderConfig(
            mix=(1.0, 0.0, 0.0),
            candidate_every_env_steps=1_000_000,
            floor_admit_every_env_steps=1_000_000,
            eval_seed_count=4,
            refit_every_iterations=1_000_000,
        ),
        "checkpoint": cfg.CheckpointConfig(every_env_steps=1_000_000, keep=2),
        "metrics": cfg.MetricsConfig(sinks=[cfg.SinkSpec("jsonl")]),
        "determinism": cfg.DeterminismConfig(tier="throughput"),
        # The exhaustive mask gate is four thousand engine queries and is what
        # ``tests/test_env_contract.py`` and ``royalelearn doctor`` are for; a file that builds
        # twenty coordinators pays for it twenty times and learns nothing new after the first.
        "doctor": cfg.DoctorConfig(run_mask_disagreement_gate=False),
    }
    settings.update(overrides)
    return cfg.RunConfig(**settings)


@contextlib.contextmanager
def coordinator(config: cfg.RunConfig, **kwargs: Any) -> Iterator[Any]:
    """A coordinator, entered, with the entry point's own environment already applied."""
    from royalelearn.coordinator import LearningCoordinator

    apply_cublas_workspace_config()
    kwargs.setdefault("preflight_kwargs", PREFLIGHT)
    kwargs.setdefault("printer", None)
    kwargs.setdefault("install_signal_handler", False)
    run = LearningCoordinator(config, **kwargs)
    with run:
        yield run


@pytest.fixture
def run(tmp_path: Path) -> Iterator[Any]:
    with coordinator(tiny_config(tmp_path)) as coordinated:
        yield coordinated


def test_a_named_run_directory_is_the_one_opened(tmp_path: Path) -> None:
    """A tool pointed at a run has to reach THAT run, whatever the code has done since.

    A run directory is ``<run_name>-<run_id>`` and the run id hashes the identity, which carries
    the commit of every repository. So a command pointed at a run that started before the code
    moved computed a different id, created an empty directory beside the real one and worked in
    it: ``royalelearn eval --run <a live run>`` reached nothing, which is why the train session
    reads episodes.jsonl with a tool of its own instead. When a caller names a directory, that
    directory is the answer, and the identity is still checked separately and by name.
    """
    named = tmp_path / "somebody-elses-name"
    with coordinator(tiny_config(tmp_path), run_dir=named) as run:
        assert run.run_dir == named
        assert named.is_dir()
        assert run.run_dir.name != f"{run.config.run_name}-{run.run_id}"

    with coordinator(tiny_config(tmp_path)) as run:
        assert run.run_dir.name == f"{run.config.run_name}-{run.run_id}"


# -- one iteration -----------------------------------------------------------


def test_an_iteration_produces_a_row_the_schema_knows_in_full(run: Any) -> None:
    run.iterate()
    row = run.rows[-1]
    assert unknown_keys(row) == ()
    # Every key the schema declares, except the ones it declares conditional and says why.
    # A key may be absent only by being on that list: absence is how this harness reports
    # "nothing to say", and an undeclared absence would be a silently dropped measurement.
    assert set(schema.METRICS) - set(schema.CONDITIONAL) <= set(row)
    assert set(schema.CONDITIONAL) <= set(schema.METRICS), (
        "a key is declared conditional that the schema does not declare at all"
    )
    for key, value in row.items():
        if isinstance(value, bool | str):
            continue
        assert np.isfinite(float(value)), f"{key} is {value}"


def test_the_vram_gate_names_the_one_legal_minibatch_below(run: Any) -> None:
    """The refusal has to say what to change it to, and there is usually one answer.

    A minibatch must divide ``batch_size`` or it stops being a pure memory knob, and at the
    shipped 4096 there is no divisor between 256 and 512 -- so "lower it" means 256 and nothing
    else. A reader should not have to discover that by trying 384 and being refused, which is
    exactly what happened to the measurement this gate came out of.
    """
    cases = {(4096, 512): 256, (4096, 256): 128, (4096, 1): None, (3072, 384): 256}
    for (batch, minibatch), expected in cases.items():
        run.config = msgspec.structs.replace(
            run.config,
            ppo=msgspec.structs.replace(
                run.config.ppo, batch_size=batch, minibatch_size=minibatch
            ),
        )
        assert run._next_legal_minibatch() == expected, (batch, minibatch)


def test_the_counters_advance_by_what_the_rectangle_holds(run: Any) -> None:
    geometry = run.geometry
    run.iterate()
    assert run.iteration == 1
    assert run.cumulative_timesteps == geometry.cycles * geometry.n_slots
    assert run.cumulative_env_steps == geometry.cycles * geometry.n_battles
    run.iterate()
    assert run.cumulative_timesteps == 2 * geometry.cycles * geometry.n_slots


def test_the_metric_file_carries_one_line_per_iteration(run: Any) -> None:
    run.iterate()
    run.iterate()
    lines = (run.run_dir / "metrics.jsonl").read_bytes().splitlines()
    assert len(lines) == 2
    assert (run.run_dir / "identity.json").exists()
    assert (run.run_dir / "config.json").exists()


def test_the_run_directory_is_named_for_the_run_and_its_identity(run: Any) -> None:
    from royalelearn.coordinator import run_directory

    assert run.run_dir == run_directory(run.config, run.run_id)
    assert run.run_dir.name.endswith(run.run_id)


def test_two_runs_of_one_identity_take_the_same_actions(tmp_path: Path) -> None:
    """The trajectory is a function of the master seed and nothing this process did first."""
    digests = []
    for index in range(2):
        with coordinator(tiny_config(tmp_path / f"run{index}")) as run:
            run.iterate()
            digests.append((run.run_id, run.rows[-1]["run/state_digest"]))
            actions = run.buffer.action[: run.buffer.cycles].copy()
            if index == 0:
                first = actions
            else:
                assert np.array_equal(first, actions)
    assert digests[0] == digests[1]


# -- the invariants ----------------------------------------------------------


def _collection(run: Any, **overrides: Any) -> dict[str, Any]:
    base = {
        "episodes": [],
        "rounds": run.geometry.cycles * run.geometry.shards_per_worker,
        "seconds": 0.1,
        "env_seconds": 0.05,
        "command": "",
    }
    base.update(overrides)
    return base


def test_a_dropped_cycle_is_named(run: Any) -> None:
    from royalelearn.coordinator import RoundsMissing

    run.iterate()
    trainable = run.buffer.trainable()
    with pytest.raises(RoundsMissing):
        run._check_iteration(_collection(run, rounds=1), trainable, [])


def test_a_short_rectangle_is_named(run: Any) -> None:
    from royalelearn.coordinator import TrainableRowsShort

    run.iterate()
    trainable = run.buffer.trainable()
    trainable[:, :] = False
    with pytest.raises(TrainableRowsShort):
        run._check_iteration(_collection(run), trainable, [])


def test_a_non_finite_log_probability_names_its_cell(run: Any) -> None:
    from royalelearn.coordinator import NonFiniteLogProb

    run.iterate()
    run.buffer.log_prob[1, 2] = -np.inf
    with pytest.raises(NonFiniteLogProb) as excinfo:
        run._check_iteration(_collection(run), run.buffer.trainable(), [])
    assert (excinfo.value.cycle, excinfo.value.slot) == (1, 2)


def test_a_refused_deploy_names_its_cell(run: Any) -> None:
    from royalelearn.coordinator import DeployRefused

    run.iterate()
    run.buffer.deploy_status[0, 1] = 7
    with pytest.raises(DeployRefused) as excinfo:
        run._check_iteration(_collection(run), run.buffer.trainable(), [])
    assert (excinfo.value.cycle, excinfo.value.slot) == (0, 1)


def test_an_unpaired_episode_is_named(run: Any) -> None:
    from royalelearn.coordinator import EpisodesUnpaired

    run.iterate()
    episodes = list(run.recent_episodes)
    assert episodes, "the tiny config's step limit ends episodes inside an iteration"
    with pytest.raises(EpisodesUnpaired):
        run._check_iteration(_collection(run), run.buffer.trainable(), episodes[:1])


def test_a_drifting_mixture_is_named(run: Any) -> None:
    """The rectangle kept every row and the configured mixture says a third are the frozen
    seat's. One of the two is wrong, and which is not knowable from here -- which is why the
    check compares them rather than trusting either."""
    from royalelearn.coordinator import MixtureDrifted
    from royalelearn.ladder.matchmaker import MixMatchmaker

    run.iterate()
    mixed = msgspec.structs.replace(run.config.ladder, mix=(0.34, 0.33, 0.33))
    run.matchmaker = MixMatchmaker(run.config.master_seed, mixed)
    with pytest.raises(MixtureDrifted):
        run._check_iteration(_collection(run), run.buffer.trainable(), [])


def test_an_assignment_inside_an_episode_is_named(run: Any) -> None:
    from royalelearn.api.rollout import EPISODE_END_NONE, GROUP_SCRIPTED
    from royalelearn.coordinator import AssignmentInsideEpisode

    run.iterate()
    buffer = run.buffer
    buffer.episode_end[:, :] = EPISODE_END_NONE
    buffer.group[2:, 0] = GROUP_SCRIPTED
    trainable = buffer.trainable()
    trainable[:, :] = True
    with pytest.raises(AssignmentInsideEpisode):
        run._check_iteration(_collection(run), trainable, [])


def test_a_stored_action_illegal_under_its_own_mask_is_named(run: Any) -> None:
    """The probe reads the mask that travelled with the transition, not a recomputed one."""
    from royalelearn.coordinator import StoredActionIllegal

    run.iterate()
    buffer = run.buffer
    actions = buffer.action[: buffer.cycles].copy()
    # The no-op is legal in every state, so an action that is not the no-op and not one the
    # mask allows is the thing being checked for. The last index never is: the grid's last
    # tile of the last hand slot is legal only when that card is affordable and placeable.
    actions[:, :] = buffer.spec.n_actions - 1
    trainable = buffer.trainable()
    with pytest.raises(StoredActionIllegal):
        run.probe.measure(buffer, run.inference.gather, actions, trainable)


# -- interactive control and crash discipline --------------------------------


def test_the_control_file_asks_for_a_checkpoint(run: Any) -> None:
    (run.run_dir / "control").write_text("c", encoding="utf-8")
    run.iterate()
    checkpoints = sorted((run.run_dir / "checkpoints").glob("0*"))
    assert checkpoints, "a c in the control file writes a checkpoint at the end of the round"
    assert not (run.run_dir / "control").exists()


def test_a_q_in_the_control_file_stops_the_run(run: Any) -> None:
    (run.run_dir / "control").write_text("q\n", encoding="utf-8")
    run.learn(until_timesteps=10_000)
    assert run.iteration == 1
    assert sorted((run.run_dir / "checkpoints").glob("0*"))


def test_a_keyboard_interrupt_saves_before_it_propagates(run: Any) -> None:
    """``KeyboardInterrupt`` is not an ``Exception``; a bare except would skip this save.

    It arrives while the next iteration is collecting, which is where a run spends the time it
    is not updating: the learner is still the one the last row describes, and that state is
    not on disk yet, so the save has something to keep.
    """
    run.iterate()
    _after_collection(run, 1, _interrupt)
    with pytest.raises(KeyboardInterrupt):
        run.learn(until_timesteps=10_000)
    (last,) = _metric_rows(run)
    (manifest,) = _manifests(run)
    assert manifest.state_digest == last["run/state_digest"]


def test_closing_twice_is_allowed(run: Any) -> None:
    run.close()
    run.close()


# -- what a crash leaves behind ----------------------------------------------


def _metric_rows(run: Any) -> list[dict[str, Any]]:
    """Every row the run's metric file holds, which is the record a checkpoint answers to."""
    path = run.run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [msgspec.json.decode(line) for line in path.read_bytes().splitlines() if line]


def _manifests(run: Any) -> list[Any]:
    """Every finished checkpoint's manifest, oldest first."""
    from royalelearn.api.checkpoint import Manifest

    root = run.run_dir / "checkpoints"
    if not root.is_dir():
        return []
    return [
        msgspec.json.decode((folder / "manifest.json").read_bytes(), type=Manifest)
        for folder in sorted(root.iterdir())
        if folder.is_dir() and (folder / "manifest.json").is_file()
    ]


def _after_collection(run: Any, number: int, effect: Any) -> None:
    """Run ``effect`` once the ``number``-th collection from now has returned.

    That is the moment the iteration's invariants are about to read the rectangle, so an
    effect that corrupts the buffer here is indistinguishable, to everything after it, from a
    collection that went wrong by itself.
    """
    original = run._collect
    seen = count(1)

    def collect(*args: Any, **kwargs: Any) -> Any:
        collection = original(*args, **kwargs)
        if next(seen) == number:
            effect(run)
        return collection

    run._collect = collect


def _interrupt(_run: Any) -> None:
    raise KeyboardInterrupt


def _short(run: Any) -> None:
    # A worker that died for a cycle leaves that cycle's rows invalid: the real way a
    # rectangle comes up short, and the way the run this test comes from did.
    run.buffer.valid[0, :] = False


def _non_finite(run: Any) -> None:
    run.buffer.log_prob[1, 2] = -np.inf


def _illegal(run: Any) -> None:
    # The probe's check, which ran inside the row and so after the update.
    run.buffer.action[: run.buffer.cycles] = run.buffer.spec.n_actions - 1


@pytest.mark.parametrize(
    ("corrupt", "refusal"),
    [
        (_short, "TrainableRowsShort"),
        (_non_finite, "NonFiniteLogProb"),
        (_illegal, "StoredActionIllegal"),
    ],
    ids=["short", "non_finite", "illegal"],
)
def test_a_refused_batch_is_never_trained_on(tmp_path: Path, corrupt: Any, refusal: str) -> None:
    """The invariants judge the batch before the update sees it, and the save says so.

    None of them reads anything the update produces, so there is no reason to run it first --
    and running it first is how a real run trained a refused batch of 22,627 rows for 18
    optimizer steps and then saved those weights under the previous iteration's counters. What
    is asserted is the record: the learner in memory, and the emergency checkpoint on disk, are
    both exactly the one the last metric row describes.
    """
    import royalelearn.coordinator as coordinator_module

    error = getattr(coordinator_module, refusal)
    with coordinator(tiny_config(tmp_path)) as run:
        _after_collection(run, 2, corrupt)
        with pytest.raises(error):
            run.learn(until_timesteps=10_000)
        (last,) = _metric_rows(run)
        assert run.update.model_updates == last["run/cumulative_updates"], (
            "the update ran on a batch the invariants then refused"
        )
        assert run.state_digest() == last["run/state_digest"]
        (manifest,) = _manifests(run)
        assert manifest.state_digest == last["run/state_digest"]
        assert manifest.cumulative_model_updates == last["run/cumulative_updates"]
        assert manifest.iteration == last["run/iteration"]
        assert manifest.cumulative_env_steps == last["run/cumulative_env_steps"]
        # The schedule it reports as the last one run is the last row's, not the refused one's.
        folder = run.run_dir / "checkpoints" / f"{manifest.cumulative_env_steps:012d}"
        schedule = msgspec.json.decode((folder / "schedules" / "state.json").read_bytes())
        assert schedule["last"]["iteration"] == last["run/iteration"] - 1


def test_an_update_that_raises_leaves_no_checkpoint_of_the_weights_it_moved(
    tmp_path: Path,
) -> None:
    """A crash inside the update, after the weights moved and before a row said so.

    The actor takes one optimizer step and then the device runs out of memory, which is the
    crash this phase actually has on a small card. The weights in memory are then half an
    update past the last row -- the critic has not moved and the step counter has not either --
    and no checkpoint may hold them: every manifest on disk must carry a digest some metric row
    carries. The one written at the end of the first iteration is the one to resume from, and
    the run says so rather than replacing it.
    """
    lines: list[str] = []
    config = tiny_config(tmp_path, checkpoint=cfg.CheckpointConfig(every_env_steps=1, keep=2))
    with coordinator(config, printer=lines.append) as run:
        run.iterate()
        (saved,) = _manifests(run)
        optimizer = run.update.actor_optimizer
        original = optimizer.step

        def step(*args: Any, **kwargs: Any) -> Any:
            original(*args, **kwargs)
            raise RuntimeError("out of device memory, on purpose")

        optimizer.step = step
        with pytest.raises(RuntimeError, match="on purpose"):
            run.learn(until_timesteps=10_000)

        rows = _metric_rows(run)
        assert run.state_digest() != rows[-1]["run/state_digest"], (
            "the premise: the weights moved inside the update before it raised"
        )
        described = {row["run/state_digest"] for row in rows}
        for manifest in _manifests(run):
            assert manifest.state_digest in described, (
                f"checkpoint at step {manifest.cumulative_env_steps} holds a learner no metric "
                f"row describes"
            )
        assert _manifests(run) == [saved]
        assert any("no metrics row describes" in line for line in lines), lines[-5:]


def test_a_crash_while_collecting_keeps_the_checkpoint_already_on_disk(tmp_path: Path) -> None:
    """The same learner is already saved, and saved at the iteration boundary: keep that one.

    Collection moves every battle's ordinal past the episodes that finish in it. An emergency
    save of the same learner at the same step would replace the boundary's checkpoint with one
    whose battles have skipped ahead -- the same weights, and no longer able to replay the
    iteration that failed.
    """
    config = tiny_config(tmp_path, checkpoint=cfg.CheckpointConfig(every_env_steps=1, keep=2))
    with coordinator(config) as run:
        run.iterate()
        (before,) = _manifests(run)
        folder = run.run_dir / "checkpoints" / f"{before.cumulative_env_steps:012d}"
        ordinals = msgspec.json.decode((folder / "rollout" / "state.json").read_bytes())
        _after_collection(run, 1, _interrupt)
        with pytest.raises(KeyboardInterrupt):
            run.learn(until_timesteps=10_000)
        battles = range(run.geometry.n_battles)
        moved = {str(battle): run.matchmaker.ordinal(battle) for battle in battles}
        assert moved != ordinals["ordinals"], (
            "the premise: an episode finished during the collection that was interrupted"
        )
        (after,) = _manifests(run)
        assert after == before
        assert msgspec.json.decode((folder / "rollout" / "state.json").read_bytes()) == ordinals


def test_a_halt_leaves_its_row_on_disk_beside_its_checkpoint(run: Any) -> None:
    """A halt checkpoints the iteration it halted on, so that iteration's row must be written.

    The alarms read the row before the sinks did, and a halt raises out of that read: the
    checkpoint the halt wrote held a learner whose only row was in memory and in the bundle,
    and the run's own metric file ended one iteration earlier.
    """
    from royalelearn.api.metrics import AlarmResult
    from royalelearn.errors import AlarmHalt

    def halt(row: Any, *, on_halt: Any = None, on_dump: Any = None, on_fired: Any = None) -> Any:
        result = AlarmResult(
            name="on_purpose",
            severity="halt",
            iteration=int(row["run/iteration"]),
            fired=True,
            consecutive=1,
            message="halted on purpose",
            values={},
        )
        # In the real AlarmSet the fired list reaches the writer before the bundle is built,
        # so the stub does it too: a stub that skipped it would be testing a contract the code
        # does not have.
        if on_fired is not None:
            on_fired([result])
        raise AlarmHalt(result.name, result.message, result.iteration, on_halt(result))

    run.alarms.evaluate = halt
    with pytest.raises(AlarmHalt):
        run.learn(until_timesteps=10_000)
    rows = _metric_rows(run)
    assert len(rows) == 1, "the halting iteration's row never reached the metric file"
    (manifest,) = _manifests(run)
    assert manifest.state_digest == rows[0]["run/state_digest"]
    assert manifest.iteration == rows[0]["run/iteration"]


def test_a_halt_leaves_its_alarms_on_disk_too(run: Any) -> None:
    """The halting iteration's alarms, in the run's own alarms file, through the real AlarmSet.

    ``evaluate`` raises the halt from inside itself, and this loop used to write the alarms file
    from what it RETURNED. So every iteration's alarms were recorded except the one that stopped
    the run -- the halting alarm and every warning firing beside it, which is the pair a reader
    opens the file for. The alarm below always holds, so what is under test is the wiring rather
    than a threshold.
    """
    import json

    from royalelearn.api.metrics import Alarm
    from royalelearn.errors import AlarmHalt
    from royalelearn.metrics.alarms import HALT, AlarmSet
    from royalelearn.metrics.sinks import ALARMS_NAME

    class _Always(Alarm):
        name = "on_purpose"
        severity = HALT
        patience = 1
        keys = ("run/iteration",)

        def holds(self, row: Any) -> bool:
            return True

    run.alarms = AlarmSet(cfg.AlarmConfig(), alarms=[_Always()], printer=None)
    with pytest.raises(AlarmHalt):
        run.learn(until_timesteps=10_000)

    path = Path(run.run_dir) / ALARMS_NAME
    assert path.exists(), "the halting iteration's alarms never reached the run's alarms file"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert [row["name"] for row in rows] == ["on_purpose"]
    assert rows[0]["severity"] == HALT
    assert rows[0]["iteration"] == 1


def test_a_resume_refuses_a_checkpoint_no_metric_row_describes(tmp_path: Path) -> None:
    """What a checkpoint written by the old ordering looks like, refused by the resume.

    Such a checkpoint's manifest carries the digest of a learner trained past its row, beside
    the row's own iteration. Resuming it would continue a run from a state its record does not
    contain, so the resume names both digests and stops.
    """
    from royalelearn.coordinator import LearningCoordinator
    from royalelearn.errors import CheckpointFormatError

    config = tiny_config(tmp_path)
    with coordinator(config) as run:
        run.iterate()
        path = run.checkpoint()
    manifest_path = path / "manifest.json"
    raw = msgspec.json.decode(manifest_path.read_bytes())
    raw["state_digest"] = "9d" * 32
    manifest_path.write_bytes(msgspec.json.encode(raw))

    apply_cublas_workspace_config()
    resumed = LearningCoordinator(
        config,
        resume=path,
        printer=None,
        preflight_kwargs=PREFLIGHT,
        install_signal_handler=False,
    )
    try:
        with pytest.raises(CheckpointFormatError, match="no metrics row"):
            resumed.__enter__()
    finally:
        resumed.close()


# -- the checkpoint ----------------------------------------------------------


def test_a_checkpoint_holds_every_component_and_verifies(run: Any) -> None:
    run.iterate()
    path = run.checkpoint()
    for folder in (
        "actor_critic",
        "optimizers",
        "schedules",
        "advantage",
        "rollout",
        "ladder",
        "matchmaker",
        "rating",
        "metrics",
        "rng",
    ):
        assert (path / folder).is_dir(), folder
    manifest = run.store.read(path, {}, strict=False)
    assert manifest.iteration == run.iteration
    assert manifest.state_digest == run.state_digest()
    assert manifest.run_id == run.run_id


def test_no_file_a_checkpoint_writes_is_a_pickle(run: Any) -> None:
    """safetensors for weights, msgspec JSON for everything else, and torch's weights_only
    reader for the optimizer moments."""
    run.iterate()
    path = run.checkpoint()
    for candidate in path.rglob("*"):
        if not candidate.is_file() or candidate.suffix == ".pt":
            continue
        head = candidate.read_bytes()[:2]
        assert head not in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"), candidate


def test_the_state_digest_moves_only_when_the_learner_does(run: Any) -> None:
    run.iterate()
    first = run.state_digest()
    assert run.state_digest() == first
    run.iterate()
    assert run.state_digest() != first


# -- the diagnostic bundle ---------------------------------------------------


def test_a_bundle_carries_the_rows_the_episodes_and_the_replay(run: Any) -> None:
    run.iterate()
    folder = Path(run._dump_bundle(note="on purpose"))
    assert (folder / "metrics.jsonl").exists()
    assert (folder / "episodes.jsonl").exists()
    assert (folder / "config.json").exists()
    assert (folder / "identity.json").exists()
    import msgspec

    summary = msgspec.json.decode((folder / "bundle.json").read_bytes())
    assert summary["note"] == "on purpose"
    assert summary["trace_divergences"] == []
    assert (folder / "episode.msgpack").exists()


def test_the_run_writes_its_evaluation_seed_set_once(run: Any) -> None:
    import msgspec

    from royalelearn.ladder.evaluate import SeedSet

    stored = msgspec.json.decode(
        (run.run_dir / "ladder" / "eval_seeds.json").read_bytes(), type=SeedSet
    )
    assert stored == run.seeds
    assert len(stored) == run.config.ladder.eval_seed_count


def test_a_run_never_leaves_its_shared_segment_behind(tmp_path: Path) -> None:
    """Two runs in one process would collide on a segment name that outlived the first."""
    names = []
    for _ in range(2):
        with coordinator(tiny_config(tmp_path)) as run:
            run.iterate()
            names.append(run.buffer.shared_handle().name)
    assert names[0] != names[1], "two runs shared one segment name"
