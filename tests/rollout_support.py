"""What the rollout tests build their environments out of.

Three things live here rather than in a test module, because a worker process imports them by
name: a codec, a rectangle for the workers to write into, and two environment components that
fail on purpose.

The codec follows the storage rule of ``docs/harness-spec.md`` section 7.2 -- static planes are
not stored, an integer-valued plane whose declared bound fits a byte is stored as one, anything
else is float16, the vector is float16 and the mask is bit-packed -- and it decides all of that
from the observation space and a sample, so it runs on either catalogue without knowing which
it has.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from royalelearn.api.buffer import MIN_TABLE_STATES, CodecTable, ObsCodec
from royalelearn.api.rollout import EnvSpec
from royalelearn.errors import PreflightError
from royalelearn.rollout.inline import rectangle_view
from royalelearn.rollout.layout import BufferHandle, BufferLayout, buffer_segment_name

UINT8 = "uint8"
FLOAT16 = "float16"
STATIC = "static"
DERIVED = "derived"


class ReferenceCodec(ObsCodec):
    """One observation row, packed the way section 7.2 says and unpacked with numpy.

    Storage is decided from the declared bounds and a sample; EXISTENCE is decided from the
    declaration, so a plane the layout does not call static is stored even when nothing in the
    sample ever changed it.
    """

    def __init__(self, table: CodecTable | None = None) -> None:
        self.spec: EnvSpec | None = None
        self._table = table
        self.clipped = 0

    def bind(self, spec: EnvSpec, table: CodecTable | None = None) -> CodecTable:
        """Bind to one environment and one table, and work out where a row's parts go."""
        bound = table if table is not None else self._table
        if bound is None:
            raise RuntimeError("this codec has no table; compute one with table(spec, sample)")
        self.spec = spec
        self._table = bound
        self._bind(bound)
        return bound

    # -- the table ----------------------------------------------------------

    def table(
        self,
        spec: EnvSpec,
        sample: Sequence[dict[str, np.ndarray]],
        *,
        min_states: int = MIN_TABLE_STATES,
    ) -> CodecTable:
        per_state = int(np.prod(spec.spatial_shape))
        states = sum(
            int(np.asarray(entry["spatial"]).size // per_state) for entry in sample
        )
        if states < min_states:
            raise PreflightError(
                f"this sample holds {states} observations against a minimum of {min_states}"
            )
        highs = spec.obs_space["spatial"].high
        planes = []
        stacked = (
            np.stack([np.asarray(row["spatial"]) for row in sample])
            if sample
            else np.zeros((0, *spec.spatial_shape), dtype=np.float32)
        )
        for index, (name, static) in enumerate(spec.spatial_layout):
            if static:
                planes.append((name, STATIC, 1.0))
                continue
            values = stacked[:, index] if stacked.size else np.zeros(1)
            integral = bool(np.all(np.mod(values, 1.0) == 0.0))
            if highs[index] <= np.iinfo(np.uint8).max and integral:
                planes.append((name, UINT8, 1.0))
            else:
                planes.append((name, FLOAT16, 1.0))
        decided = CodecTable(plane=tuple(planes), vector=FLOAT16, mask=UINT8)
        self.bind(spec, decided)
        return decided

    def _bind(self, table: CodecTable) -> None:
        spec = self.spec
        assert spec is not None
        self._u8 = np.array(
            [i for i, (_, kind, _) in enumerate(table.plane) if kind == UINT8], dtype=np.int64
        )
        self._f16 = np.array(
            [i for i, (_, kind, _) in enumerate(table.plane) if kind == FLOAT16], dtype=np.int64
        )
        self._static = np.array(
            [i for i, (_, kind, _) in enumerate(table.plane) if kind == STATIC], dtype=np.int64
        )
        self._cells = spec.tiles[0] * spec.tiles[1]
        self._mask_bytes = -(-spec.n_actions // 8)
        self._u8_bytes = int(self._u8.size) * self._cells
        self._f16_bytes = 2 * int(self._f16.size) * self._cells
        self._vector_bytes = 2 * spec.vector_size
        self._row = self._u8_bytes + self._f16_bytes + self._vector_bytes + self._mask_bytes

    # -- the ABC ------------------------------------------------------------

    @property
    def codec_version(self) -> int:
        return 1

    def row_bytes(self, spec: EnvSpec) -> int:
        if self._table is None:
            raise RuntimeError("this codec has no table yet; call table() first")
        return self._row

    def static_planes(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(obs["spatial"])[self._static]

    def pack(self, obs: dict[str, np.ndarray], out: memoryview, row: int) -> None:
        spatial = np.asarray(obs["spatial"])
        dest = np.frombuffer(out, dtype=np.uint8, count=self._row, offset=row * self._row)
        cursor = 0
        if self._u8.size:
            values = spatial[self._u8].reshape(-1)
            clipped = np.clip(values, 0, np.iinfo(np.uint8).max)
            self.clipped += int(np.count_nonzero(clipped != values))
            dest[cursor : cursor + self._u8_bytes] = clipped.astype(np.uint8)
            cursor += self._u8_bytes
        if self._f16.size:
            block = spatial[self._f16].reshape(-1).astype(np.float16)
            dest[cursor : cursor + self._f16_bytes] = block.view(np.uint8)
            cursor += self._f16_bytes
        vector = np.asarray(obs["vector"]).astype(np.float16)
        dest[cursor : cursor + self._vector_bytes] = vector.view(np.uint8)
        cursor += self._vector_bytes
        bits = np.packbits(np.asarray(obs["action_mask"]).astype(bool), bitorder="little")
        dest[cursor : cursor + self._mask_bytes] = bits

    def unpack_to_device(self, raw: Any, statics: Any, out: Any) -> None:
        raise NotImplementedError(
            "the reference codec unpacks on the CPU; unpack_row is what the tests read"
        )

    # -- what the tests read ------------------------------------------------

    def unpack_row(
        self, rows: np.ndarray, row: int, statics: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """One packed row back, as the environment's own keys."""
        spec = self.spec
        raw = np.asarray(rows).reshape(-1, self._row)[row]
        cursor = 0
        spatial = np.zeros(spec.spatial_shape, dtype=np.float32)
        if self._u8.size:
            block = raw[cursor : cursor + self._u8_bytes].reshape(self._u8.size, *spec.tiles)
            spatial[self._u8] = block.astype(np.float32)
            cursor += self._u8_bytes
        if self._f16.size:
            block = (
                raw[cursor : cursor + self._f16_bytes]
                .view(np.float16)
                .reshape(self._f16.size, *spec.tiles)
            )
            spatial[self._f16] = block.astype(np.float32)
            cursor += self._f16_bytes
        if statics is not None and self._static.size:
            spatial[self._static] = statics
        vector = raw[cursor : cursor + self._vector_bytes].view(np.float16).astype(np.float32)
        cursor += self._vector_bytes
        mask = np.unpackbits(
            raw[cursor : cursor + self._mask_bytes], count=spec.n_actions, bitorder="little"
        )
        return {
            "spatial": spatial,
            "vector": vector,
            "action_mask": mask.astype(np.int8),
            "mask_planes": mask[1:].reshape(spec.hand_size, *spec.tiles).astype(np.int8),
        }


class SharedRectangle:
    """The experience buffer, as much of it as a rollout needs.

    The workers write observation rows into it and the source computes where; what a real
    ``RectBuffer`` adds -- the scalar columns, the advantage arrays, the minibatching -- is the
    learner's side of it and nothing here touches it.
    """

    def __init__(self, spec: EnvSpec, *, run_id: str, cycles: int, n_slots: int, row_bytes: int):
        from multiprocessing import shared_memory

        self.layout = BufferLayout.from_spec(
            spec,
            run_id=run_id,
            cycles=cycles,
            n_slots=n_slots,
            row_bytes=row_bytes,
            codec_version=1,
        )
        name = buffer_segment_name(run_id)
        self.segment = shared_memory.SharedMemory(
            create=True, size=self.layout.total_bytes, name=name
        )
        self.layout.write_header(self.segment.buf)
        self.rows = rectangle_view(self.layout, self.segment.buf)

    def shared_handle(self) -> BufferHandle:
        return self.layout.handle(self.segment.name)

    def close(self) -> None:
        self.rows = None
        self.segment.close()
        with contextlib.suppress(FileNotFoundError, OSError):
            self.segment.unlink()


class ExplodingReward:
    """A reward function that raises on a chosen step, to see what a worker does about it."""

    def __init__(self, at_step: int = 3, message: str = "deliberate") -> None:
        self.at_step = at_step
        self.message = message
        self.steps = 0

    def bind(self, engine: Any) -> None:
        return None

    def reset(self, state: Any) -> None:
        return None

    def config(self) -> dict[str, Any]:
        return {"at_step": self.at_step, "message": self.message}

    def get_reward(self, team: int, prev: Any, state: Any, results: Any) -> float:
        self.steps += 1
        if self.steps > 2 * self.at_step:
            raise RuntimeError(self.message)
        return 0.0


class HangingReward:
    """A reward function that stops answering, to see that a wait has a deadline."""

    def __init__(self, at_step: int = 3, seconds: float = 30.0) -> None:
        self.at_step = at_step
        self.seconds = seconds
        self.steps = 0

    def bind(self, engine: Any) -> None:
        return None

    def reset(self, state: Any) -> None:
        return None

    def config(self) -> dict[str, Any]:
        return {"at_step": self.at_step, "seconds": self.seconds}

    def get_reward(self, team: int, prev: Any, state: Any, results: Any) -> float:
        self.steps += 1
        if self.steps > 2 * self.at_step:
            time.sleep(self.seconds)
        return 0.0


# ---------------------------------------------------------------------------
# Driving a source
# ---------------------------------------------------------------------------

#: Where a worker process finds the pieces above. It is this module, named the way a worker's
#: component allow-list takes it: the tests sit on the path a spawned child inherits.
SUPPORT_MODULE = "rollout_support"
CODEC = f"{SUPPORT_MODULE}.ReferenceCodec"


def rollout_config(
    *,
    workers: int = 1,
    games_per_worker: int = 2,
    shards_per_worker: int = 1,
    max_steps: int = 6,
    source: str = "inline",
    reward: Any = None,
    master_seed: int = 4242,
    **rollout: Any,
) -> Any:
    """A run small enough to drive in a test and shaped like a real one.

    MockEngine, so the widths are not the catalogue a real run uses and a literal from either
    would fail on the other; a short step limit, so episodes end inside a handful of cycles.
    """
    from royalelearn import config as cfg

    env = cfg.default_env_spec(cfg.MOCK_ENGINE, max_steps=max_steps)
    if reward is not None:
        env = _replace(env, reward_fn=reward)
    return cfg.RunConfig(
        run_name="test",
        master_seed=master_seed,
        env=env,
        extra_component_modules=[SUPPORT_MODULE],
        rollout=cfg.RolloutConfig(
            source=source,
            workers=workers,
            games_per_worker=games_per_worker,
            shards_per_worker=shards_per_worker,
            launch_delay_s=0.0,
            **rollout,
        ),
        ppo=cfg.PPOConfig(
            n_epochs=1, timesteps_per_iteration=24, batch_size=8, minibatch_size=4
        ),
        determinism=cfg.DeterminismConfig(tier="throughput"),
    )


def _replace(struct: Any, **fields: Any) -> Any:
    import msgspec

    return msgspec.structs.replace(struct, **fields)


def preflight(config: Any, codec: str = CODEC, **kwargs: Any) -> Any:
    """Preflight with sample counts a test can afford; every gate still runs."""
    from royalelearn.rollout.preflight import run_preflight

    kwargs.setdefault("table_samples", 32)
    kwargs.setdefault("mask_samples", 32)
    # The rule's own sample size is what tests/test_codec.py asserts. What these tests need from
    # a table is a row size, so they say plainly that they are deciding one from fewer states
    # rather than paying for a thousand of them in every rollout test.
    kwargs.setdefault("min_table_states", 0)
    kwargs.setdefault("printer", None)
    return run_preflight(config, codec=codec, **kwargs)


class Drive:
    """What one run of a source produced, as rectangles a test can assert on.

    Every column is ``(cycles + 1, n_slots)``: the iteration's ``T`` cycles and the trailing
    one, which carries the bootstrap observation and the last transition's reward.
    """

    def __init__(self, cycles: int, n_slots: int, row_bytes: int) -> None:
        shape = (cycles + 1, n_slots)
        self.cycles = cycles
        self.group = np.zeros(shape, dtype=np.int8)
        #: What the worker published, kept apart from ``group``, which the parent overwrites
        #: with its own routing table before the round is recorded.
        self.published_group = np.zeros(shape, dtype=np.int8)
        self.reward = np.zeros(shape, dtype=np.float32)
        self.tick = np.zeros(shape, dtype=np.int32)
        self.terminated = np.zeros(shape, dtype=bool)
        self.truncated = np.zeros(shape, dtype=bool)
        self.valid = np.zeros(shape, dtype=bool)
        self.deploy_status = np.zeros(shape, dtype=np.int8)
        self.episode_end = np.zeros(shape, dtype=np.int8)
        self.obs_rows = np.zeros(shape, dtype=np.int32)
        self.action = np.zeros(shape, dtype=np.int16)
        self.shard = np.zeros(shape, dtype=np.int8)
        self.episodes: list[Any] = []
        self.finals: dict[tuple[int, int], np.ndarray] = {}
        self.rows: np.ndarray | None = None

    def record(self, round_: Any) -> None:
        slots = round_.slots
        cycle = round_.cycle
        self.group[cycle, slots] = round_.group
        self.reward[cycle, slots] = round_.reward
        self.tick[cycle, slots] = round_.tick
        self.terminated[cycle, slots] = round_.terminated
        self.truncated[cycle, slots] = round_.truncated
        self.valid[cycle, slots] = round_.valid
        self.deploy_status[cycle, slots] = round_.deploy_status
        self.episode_end[cycle, slots] = round_.episode_end
        self.obs_rows[cycle, slots] = round_.obs_rows
        self.shard[cycle, slots] = round_.shard
        self.episodes.extend(round_.episodes)


def drive(
    source: Any,
    buffer: SharedRectangle,
    report: Any,
    *,
    cycles: int,
    policy: Any = None,
    matchmaker: Any = None,
    gamma: float = 0.99,
) -> Drive:
    """Run one iteration through a source, playing the parent's part.

    The parent's part is the whole of section 7.5: draw an assignment for every battle whose
    episode just ended, fold it into the group column before routing, choose an action for
    every slot, and hand the worker the result.
    """
    from royalelearn.api.rollout import GROUP_SCRIPTED, Step
    from royalelearn.rollout.plan import SlotPlanner

    geo = report.geometry
    planner = SlotPlanner(geo, source.config.master_seed)
    out = Drive(cycles, geo.n_slots, report.row_bytes)
    group = np.full(geo.n_slots, -1, dtype=np.int8)
    opponent = np.full(geo.n_slots, -1, dtype=np.int8)
    seat = np.full(geo.n_battles, -1, dtype=np.int8)
    ordinal = np.zeros(geo.n_battles, dtype=np.int64)
    for _ in range(cycles):
        for _shard in range(geo.shards_per_worker):
            round_ = source.next_round(30.0)
            slots = round_.slots
            assigned = False
            ended = slots[round_.episode_end != 0]
            for battle in sorted({int(planner.slot_battle[s]) for s in ended}):
                ordinal[battle] += 1
                if matchmaker is not None:
                    pair, ix, learner = matchmaker(battle, int(ordinal[battle]))
                    group[2 * battle : 2 * battle + 2] = pair
                    opponent[2 * battle : 2 * battle + 2] = ix
                    seat[battle] = learner
                    assigned = True
            # The worker's own reported group, before the parent's table is folded in: it is
            # a published field of the scalar record like every other column, so it is
            # differenced like every other column.
            out.published_group[round_.cycle, slots] = round_.group
            round_.group[:] = group[slots]
            out.record(round_)
            if policy is None:
                actions = np.zeros(slots.size, dtype=np.int16)
            else:
                actions = policy(round_, buffer)
            actions[round_.group == GROUP_SCRIPTED] = 0
            out.action[round_.cycle, slots] = actions
            final_slots, final_rows = source.final_rows(round_)
            if final_slots.size:
                out.finals[(round_.cycle, round_.shard)] = np.array(final_rows, copy=True)
            source.submit(
                Step(
                    actions=actions,
                    gamma=gamma,
                    group=group[slots] if assigned else np.zeros(0, dtype=np.int8),
                    opponent_ix=opponent[slots] if assigned else np.zeros(0, dtype=np.int8),
                    learner_seat=(
                        seat[planner.slot_battle[slots][::2]]
                        if assigned
                        else np.zeros(0, dtype=np.int8)
                    ),
                )
            )
    for round_ in source.finish_iteration():
        out.published_group[round_.cycle, round_.slots] = round_.group
        round_.group[:] = group[round_.slots]
        out.record(round_)
    out.rows = np.array(buffer.rows, copy=True)
    return out
