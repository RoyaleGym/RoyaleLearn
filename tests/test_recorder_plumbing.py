"""Does a replay recorder reach exactly one env, and change nothing a policy is compared on?

A recorder saves finished battles so they can be watched afterwards. Watching a run live cannot
work: `time/collection` is about 5% of an iteration and those seconds hold tens of thousands of
engine ticks, so it is a firehose between silences rather than slow gameplay.

Two properties carry the whole design and neither is visible in ordinary use.

THE FIRST IS COMPARABILITY. A recorder does not change the game, so it must not enter
``EnvFactorySpec`` and must not move ``env_spec_digest``. If it did, a policy trained with one
would be incomparable with every policy trained without -- the ladder's ratings are only meaningful
between policies known to have played the same game.

THE SECOND IS THAT ONE IS NOT ALL. ``EnvFactory`` takes ``recorder`` as a component, and a
component is built once per env, and the vec env builds one env per game. A recorder in the
recipe is 24 recorders on a 24-game shard, which costs throughput before it costs disk:
``ClashParallelEnv`` leaves the single multi-tick ``engine.step`` path whenever its recorder
wants per-tick frames.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

import royalelearn.config as cfg
from royalelearn.rollout.envspec import ComponentSpec

RECORDER = "royalegym.replay.SavingReplayRecorder"


def test_a_recorder_is_not_a_field_of_the_hashed_spec() -> None:
    """The structural half: it cannot move the digest if it cannot be put there."""
    fields = set(cfg.EnvFactorySpec.__struct_fields__)

    assert "recorder" not in fields, (
        "a recorder in EnvFactorySpec makes every policy trained with one incomparable with "
        "every policy trained without, because the spec is hashed into the ladder's context"
    )
    assert "viser" not in fields, "the same argument, and the precedent this follows"


def test_configuring_a_recorder_leaves_the_env_digest_alone(tmp_path: Any) -> None:
    """The behavioural half, on a config that actually carries one."""
    spec = cfg.default_env_spec(cfg.MOCK_ENGINE)
    plain = cfg.RunConfig(env=spec)
    recording = msgspec.structs.replace(
        plain,
        rollout=msgspec.structs.replace(
            plain.rollout,
            recorder=ComponentSpec(RECORDER, {"out_dir": str(tmp_path), "every": 1, "keep": 2}),
        ),
    )

    assert recording.rollout.recorder is not None, "the test did not configure what it claims to"
    assert recording.env.digest() == plain.env.digest()
    assert recording.env == plain.env


def test_the_recorder_survives_the_spawn_boundary_as_json() -> None:
    """A worker is handed a description, not an object, and decodes it in another process.

    The decoder is built where the struct is USED, which is the subprocess -- so an unresolvable
    annotation here does not fail at import, it fails at worker spawn, hours into a run. That is
    how this exact field shipped broken for ten minutes: the module imported fine and
    ``msgspec.msgpack.Decoder(WorkerConfig)`` raised NameError.
    """
    from royalelearn.rollout.inline import WorkerConfig

    decoder = msgspec.msgpack.Decoder(WorkerConfig)
    assert decoder is not None

    given = ComponentSpec(RECORDER, {"out_dir": "runs/x/replays", "every": 25, "keep": 4})
    back = msgspec.msgpack.decode(msgspec.msgpack.encode(given), type=ComponentSpec)
    assert back == given
    assert back.kwargs["every"] == 25


@pytest.mark.parametrize("struct", ["WorkerConfig"])
def test_every_struct_crossing_the_spawn_boundary_has_a_working_decoder(struct: str) -> None:
    """The general form of the trap above, so the next field added does not have to find it.

    ``from __future__ import annotations`` makes every annotation a string, so a name that does
    not exist costs nothing until something resolves it. Importing the module is not that moment.
    """
    from royalelearn.rollout import inline

    msgspec.msgpack.Decoder(getattr(inline, struct))


def test_a_recorder_reaches_one_env_and_no_others(tmp_path: Any) -> None:
    """One recorder per SHARD, not one per game.

    ``keep`` bounds each instance on its own, and the saved name is
    ``{pid}-{completed:06d}-tick{tick}`` whose pid separates processes and cannot separate
    instances inside one -- two collide exactly when two battles end on the same tick, which is
    the step cap. Above all, a recorder that wants per-tick frames takes its env off the
    multi-tick step path, so 24 of them is 24 battles stepping one tick at a time.
    """
    spec = cfg.default_env_spec(cfg.MOCK_ENGINE)
    recorder = ComponentSpec(RECORDER, {"out_dir": str(tmp_path), "every": 1, "keep": 1})

    vec = spec.build_vec(4, viser=None, recorder=msgspec.structs.replace(recorder))
    try:
        attached = [env.recorder is not None for env in vec.envs]
    finally:
        vec.close()

    assert attached == [True, False, False, False], (
        "a recorder on every env steps every battle in the shard tick by tick"
    )


def test_no_recorder_attaches_nothing() -> None:
    """The control. An assertion that passes either way measures the test, not the code."""
    spec = cfg.default_env_spec(cfg.MOCK_ENGINE)

    vec = spec.build_vec(3, viser=None)
    try:
        attached = [env.recorder is not None for env in vec.envs]
    finally:
        vec.close()

    assert attached == [False, False, False]


def test_the_factory_refuses_to_carry_it(tmp_path: Any) -> None:
    """Passing it to ``factory`` must not quietly put it in the recipe.

    ``EnvFactory`` WOULD accept it -- ``recorder`` is one of its component names -- which is what
    makes the quiet version possible.
    """
    spec = cfg.default_env_spec(cfg.MOCK_ENGINE)
    recorder = ComponentSpec(RECORDER, {"out_dir": str(tmp_path), "every": 1, "keep": 1})

    factory = spec.factory(recorder=recorder)

    assert "recorder" not in factory.components, (
        "the recorder went into the recipe, so every env in the shard would build one"
    )


def test_only_worker_zero_carries_the_recorder(tmp_path: Any) -> None:
    """Two workers, one recorder. The rule that decides whether the run writes one set or two.

    This is the same rule the viewer uses, and it is one line in each of four places, so it is
    worth a test that reads the WorkerConfig a farm would actually send rather than trusting that
    four copies of a conditional agree.
    """
    from royalelearn.rollout.farm import ProcessRolloutSource

    given = ComponentSpec(RECORDER, {"out_dir": str(tmp_path), "every": 1, "keep": 1})
    config = cfg.RunConfig(env=cfg.default_env_spec(cfg.MOCK_ENGINE))
    config = msgspec.structs.replace(
        config, rollout=msgspec.structs.replace(config.rollout, recorder=given)
    )

    class _Layout:
        def handle(self, name: str) -> Any:
            return None

    class _Segment:
        name = "seg"

    class _Worker:
        def __init__(self, index: int) -> None:
            self.index = index
            self.layout = _Layout()
            self.segment = _Segment()

    class _Farm:
        def __init__(self) -> None:
            self.config = config
            self.generation = {0: 0, 1: 0}
            self.run_id = "r"
            self.geometry = cfg.Geometry(
                workers=2,
                games_per_worker=4,
                shards_per_worker=2,
                games_per_shard=2,
                n_battles=8,
                n_slots=16,
                mirror_battles=0,
                learner_row_fraction=1.0,
                learner_rows=16,
                cycles=2,
                timesteps_per_iteration=32,
            )
            self._spec = None
            self.codec_path = "royalelearn.rollout.codec.SpatialObsCodec"
            self.codec_table = None
            self.extra_modules = ()
            self.viser = False
            self.ordinals = None

    farm = _Farm()
    first = ProcessRolloutSource._worker_config(farm, _Worker(0), None)
    second = ProcessRolloutSource._worker_config(farm, _Worker(1), None)

    assert first.recorder == given
    assert second.recorder is None, (
        "every worker got a recorder, so the run writes one set of replays per worker and each "
        "worker's shard steps tick by tick"
    )
