"""One shard of battles, and the rollout source that drives shards in this process.

``ShardRunner`` is the whole worker side of the boundary: it owns a vec env, plays the
scripted seats itself, packs observations straight into their final resting place in the
experience buffer, and writes one fixed-size scalar record per slot per round. It does not
know whether it is running in a child process or in the learner's own, which is what makes
``InlineRolloutSource`` and ``rollout.farm.ProcessRolloutSource`` two drivers of one
implementation rather than two implementations.

``InlineRolloutSource`` is the reference. It is what the fast tests run, it is what a new
process worker -- or a Rust one -- is held against, and the acceptance criterion for either is
that the bytes it produces are identical to these from the same seed
(``tests/test_rollout_farm.py``).

THE ROUND, IN ONE PLACE. A round is one shard across the whole farm at one cycle. The worker
publishes the observation the policy is to act on at cycle ``t``; the parent answers with that
cycle's actions; the worker steps and publishes cycle ``t + 1``.

A round's observation is therefore the state at its own cycle, and its scalars -- the reward,
the two done flags, the deploy status and the episode end -- are the step that ARRIVED at it,
which is the transition that began one cycle earlier. The round at cycle 0 has no transition
behind it and carries zeros; the trailing round at cycle ``T`` carries the last transition's
reward and the bootstrap observation, and ``finish_iteration`` is what waits for it. An
iteration is ``T`` cycles of the loop plus that one wait.
"""

from __future__ import annotations

import struct
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np

from ..api.buffer import CodecTable
from ..api.rollout import (
    BUCKETS,
    EPISODE_END_DRAW,
    EPISODE_END_LOSS,
    EPISODE_END_NONE,
    EPISODE_END_WIN,
    GROUP_DEAD,
    GROUP_LEARNER,
    GROUP_SCRIPTED,
    ROLE_MIRROR,
    ROLE_POOL,
    ROLE_SCRIPTED,
    Close,
    Defer,
    EnvSpec,
    EpisodeRecord,
    Plan,
    RolloutRound,
    RolloutSource,
    SetState,
    SlotPlan,
    Spaces,
    Step,
    WorkerCommand,
    WorkerFailure,
)
from ..config import Geometry
from ..errors import PreflightError
from ..seeding import ENV_STAGGER, derive_generator, stream_path
from .envspec import EnvFactorySpec, resolve_component
from .layout import (
    ASSIGN,
    CONTROL,
    ERR_EXCEPTION,
    ERR_NONE,
    FLAG_TERMINATED,
    FLAG_TRUNCATED,
    FLAG_VALID,
    STATE_ACTIONS_READY,
    STATE_CLOSED,
    STATE_OBS_READY,
    BufferHandle,
    BufferLayout,
    ControlHandle,
    ControlLayout,
    read_error,
    write_error,
)
from .plan import NO_OPPONENT, SlotPlanner, opponent_index, read_plan, write_plan
from .scripted import SCRIPTED_NAMES, ScriptedSeats, scripted_id

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.buffer import ExperienceBuffer, ObsCodec

__all__ = [
    "ASSIGN_DTYPE",
    "COMMAND_CLOSE",
    "COMMAND_DEFER",
    "COMMAND_PLAN",
    "COMMAND_SET_STATE",
    "COMMAND_SPACES",
    "COMMAND_STEP",
    "CYCLE_UNBOUND",
    "InlineRolloutSource",
    "PlanMessage",
    "RolloutSourceBase",
    "ShardRunner",
    "WorkerConfig",
    "assignments_constant_within_episodes",
    "build_codec",
    "check_actions_legal",
    "check_bootstrap",
    "check_round",
    "command_of",
    "plan_view",
    "rectangle_view",
]

#: What the parent writes into the control word's ``state`` to say what this round is. Two of
#: them are the layout's own states -- a step is actions, a close is closed -- and the rest sit
#: above the range the worker publishes in, so a command and a state can never be read as one
#: another.
COMMAND_STEP = STATE_ACTIONS_READY
COMMAND_CLOSE = STATE_CLOSED
COMMAND_PLAN = 16
COMMAND_SET_STATE = 17
COMMAND_SPACES = 18
COMMAND_DEFER = 19

#: The cycle a publication carries before an iteration has been begun: an observation that is
#: real but has nowhere in the rectangle to live yet, because the rectangle belongs to an
#: iteration and there is not one.
CYCLE_UNBOUND = -1
_CYCLE_SENTINEL = (1 << 64) - 1

#: The assignment region as a structured array: the parent's per-slot answer to "who plays
#: this seat", written for every slot of a shard whenever any of its battles has just reset.
ASSIGN_DTYPE = np.dtype(
    {
        "names": [f.name for f in ASSIGN.fields],
        "formats": ["i1", "i1", "i1"],
        "offsets": [f.offset for f in ASSIGN.fields],
        "itemsize": ASSIGN.size,
    }
)
if ASSIGN_DTYPE.itemsize != ASSIGN.size:  # pragma: no cover - a typo in the table above
    raise RuntimeError("ASSIGN_DTYPE does not match the ASSIGN record")


def rectangle_view(layout: BufferLayout, buffer: Any) -> np.ndarray:
    """Segment A as ``uint8[(cycles + frame_stack) * n_slots, row_bytes]``, over its memory.

    One row per cell of the rectangle, in the order ``BufferLayout.row_index`` computes, so a
    cell's packed observation is a row and nothing has to know how wide a row is to find one.
    """
    return layout.obs_view(buffer)


def plan_view(layout: ControlLayout, control: Any) -> memoryview:
    """One worker's plan region: the whole opening table for its slots, once per iteration.

    It sits outside the per-shard, per-parity regions because it is written once and read by
    every shard, so ``ControlLayout.view`` -- which addresses a shard and a parity -- does not
    reach it.
    """
    return memoryview(control)[layout.plan_offset : layout.plan_offset + layout.plan_bytes]


def command_of(command: WorkerCommand) -> int:
    """The control word a command is sent as."""
    if isinstance(command, Step):
        return COMMAND_STEP
    if isinstance(command, Plan):
        return COMMAND_PLAN
    if isinstance(command, SetState):
        return COMMAND_SET_STATE
    if isinstance(command, Spaces):
        return COMMAND_SPACES
    if isinstance(command, Defer):
        return COMMAND_DEFER
    if isinstance(command, Close):
        return COMMAND_CLOSE
    raise TypeError(f"{type(command).__name__} is not a WorkerCommand this boundary carries")


class PlanMessage(msgspec.Struct, frozen=True):
    """What a ``Plan`` carries that does not fit a fixed-width per-slot record.

    The table itself goes through the plan region, a byte at a time per slot. The snapshot ids
    are strings of unbounded length, they change once per iteration rather than once per round,
    and the worker needs them only to name an opponent in an episode record -- so they travel
    beside the table rather than in it.
    """

    iteration: int
    resident_snapshots: tuple[str, ...] = ()


class WorkerConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Everything a worker is told at start-up, as one msgspec value.

    A description rather than a closure: it encodes to bytes, so a child is handed it over the
    spawn boundary without pickling anything the parent built, and a checkpoint can record what
    the workers were.
    """

    worker: int
    generation: int
    run_id: str
    master_seed: int
    geometry: Geometry
    env: EnvFactorySpec
    spec: EnvSpec
    codec: str
    codec_table: CodecTable
    buffer: BufferHandle
    control: ControlHandle
    extra_component_modules: tuple[str, ...] = ()
    spin_us: int = 100
    stagger_first_reset: bool = True
    #: Decisions of warm-up, drawn per battle, so that episode ends spread across cycles
    #: instead of arriving in one spike at whatever the episode length turns out to be. One
    #: decision of warm-up costs one decision, so the bound is a start-up cost of at most this
    #: many steps per battle and is paid once per run.
    stagger_max_steps: int = 64
    #: Whether this worker's first shard carries the viewer's state stream. Exactly one vec
    #: env in a run may: the publisher binds one UDP port, and a second one raises at
    #: construction.
    viser: bool = False


def build_codec(
    path: str,
    spec: EnvSpec,
    table: CodecTable | None = None,
    extra_modules: Sequence[str] = (),
) -> ObsCodec:
    """The observation codec named by ``path``, built for one environment and one table.

    The class is named rather than sent because a codec is not a value: it holds the table's
    divisors and offsets as arrays and is rebuilt identically on both sides of the boundary
    from the two things that decide it, which are the environment's own spec and the table
    preflight computed from a sample of its observations. A codec built without a table is one
    that has not been told what the table is yet -- it can compute one and it cannot pack --
    which is the state preflight builds it in.
    """
    cls = resolve_component(path, tuple(extra_modules))
    codec = cls(table)
    if table is not None:
        codec.bind(spec, table)
    return codec


# ---------------------------------------------------------------------------
# The worker side of one shard
# ---------------------------------------------------------------------------


class _TermRecorder:
    """A reward function that records each seat's weighted breakdown as it is produced.

    The vec env resets a finished game inside the same ``step`` that ended it, and a reset
    clears the reward function's stored breakdown, so the terminal step -- the one that carries
    the win -- is gone by the time the step returns. Wrapping the function catches every call
    at the moment it is made, and changes nothing about what the environment computes or
    reports: the scalar rewards, the config and every other attribute are the wrapped
    function's.
    """

    def __init__(self, inner: Any, sink: dict[int, dict[str, float]], slot_of_team: dict[int, int]):
        self._inner = inner
        self._sink = sink
        self._slot_of_team = slot_of_team

    def get_reward(self, team: int, prev: Any, state: Any, results: Any) -> float:
        value = self._inner.get_reward(team, prev, state, results)
        terms = getattr(self._inner, "terms_for", None)
        if terms is not None:
            totals = self._sink[self._slot_of_team[team]]
            for name, amount in terms(team).items():
                totals[name] = totals.get(name, 0.0) + float(amount)
        return value

    def reset(self, state: Any) -> None:
        self._inner.reset(state)

    def bind(self, engine: Any) -> None:  # pragma: no cover - the env binds before the wrap
        self._inner.bind(engine)

    def config(self) -> dict[str, Any]:
        return self._inner.config()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ShardRunner:
    """One shard: a vec env of ``games_per_shard`` battles, and the bytes it publishes.

    The runner holds the parent's assignment table for its own slots and nothing else about
    the run. It never decides what a battle plays; it is told, on the round after the battle's
    episode ended, and it applies what it is told before the first action of the new episode is
    taken.
    """

    def __init__(
        self,
        config: WorkerConfig,
        shard: int,
        *,
        planner: SlotPlanner,
        codec: ObsCodec,
        buffer: Any,
        control: Any,
        viser: str | None = None,
    ) -> None:
        self.config = config
        self.worker = config.worker
        self.shard = shard
        self.planner = planner
        self.codec = codec
        self.buffer_layout: BufferLayout = config.buffer.layout()
        self.control_layout: ControlLayout = config.control.layout()
        self.buffer_layout.check_header(buffer)
        self.obs_rows = rectangle_view(self.buffer_layout, buffer)
        self.obs_view = memoryview(self.obs_rows.reshape(-1))
        self.control = control
        self.slots = planner.shard_slots(config.worker, shard)
        self.n_slots = int(self.slots.size)
        self.games = config.geometry.games_per_shard
        self.battles = planner.shard_battles(config.worker, shard)
        self.vec = config.env.build_vec(
            self.games, tuple(config.extra_component_modules), viser=viser
        )
        self.scripted = ScriptedSeats(planner, config.worker, self.slots, config.generation)
        self.noop = int(self.vec.envs[0].action_parser.noop())

        # The parent's table for this shard's slots, and the counters no one else can keep.
        self.group = np.full(self.n_slots, GROUP_LEARNER, dtype=np.int8)
        self.opponent_ix = np.full(self.n_slots, NO_OPPONENT, dtype=np.int8)
        self.learner_seat = np.full(self.games, -1, dtype=np.int8)
        self.resident_snapshots: tuple[str, ...] = ()
        self.ordinal = np.zeros(self.games, dtype=np.int64)
        self.episode_steps = np.zeros(self.n_slots, dtype=np.int32)
        self.cards_played = np.zeros(self.n_slots, dtype=np.int32)
        self.illegal_commands = np.zeros(self.n_slots, dtype=np.int32)
        self.undiscounted_return = np.zeros(self.n_slots, dtype=np.float64)
        self.reward_terms: dict[int, dict[str, float]] = {i: {} for i in range(self.n_slots)}
        self.episodes: list[EpisodeRecord] = []
        self.pending_snapshots: tuple[bytes | None, ...] = ()

        # This round's scalars, kept as columns so that publishing is a handful of writes.
        self.reward = np.zeros(self.n_slots, dtype=np.float32)
        self.tick = np.zeros(self.n_slots, dtype=np.int32)
        self.terminated = np.zeros(self.n_slots, dtype=bool)
        self.truncated = np.zeros(self.n_slots, dtype=bool)
        self.deploy_status = np.full(self.n_slots, -1, dtype=np.int8)
        self.episode_end = np.zeros(self.n_slots, dtype=np.int8)
        self.round_steps = np.zeros(self.n_slots, dtype=np.int32)
        self.round_cards = np.zeros(self.n_slots, dtype=np.int32)
        self.round_leak = np.zeros(self.n_slots, dtype=np.int32)
        self.final_slots = np.zeros(0, dtype=np.int32)

        self.obs: dict[str, np.ndarray] = {}
        self.cycle = CYCLE_UNBOUND
        self.publications = 0
        self.gamma: float | None = None
        self.t_env_ns = 0
        self.closed = False

    # -- start-up -----------------------------------------------------------

    def start(self) -> None:
        """Reset the shard's battles and leave it holding the first observation.

        The seed is the shard's one named draw. Every episode after the first starts from the
        vec env's own generator, which that seed set, so a shard replays exactly from its seed
        and its respawn generation and nothing else.
        """
        from royalegym.protocol import BLUE, RED

        for index, env in enumerate(self.vec.envs):
            seats = {int(BLUE): 2 * index, int(RED): 2 * index + 1}
            env.reward_fn = _TermRecorder(env.reward_fn, self.reward_terms, seats)
        seed = self.planner.env_seed(self.worker, self.shard, self.config.generation)
        obs, infos = self.vec.reset(seed=seed)
        self.obs = obs
        self._clear_round()
        self.tick[:] = np.asarray(infos["tick"], dtype=np.int32)
        if self.config.stagger_first_reset:
            self._stagger(seed)

    def _stagger(self, seed: int) -> None:
        """Advance each battle by a drawn number of decisions, so that episodes end apart.

        Without it every battle of a shard starts at the same phase, so their episodes end on
        the same cycle and the iteration's episode statistics come in spikes. The draw is from
        a stream of its own rather than the shard's env seed, and the warm-up is played as
        no-ops so that nothing a policy did reaches the buffer: the run starts from the
        observation this leaves.
        """
        generator = derive_generator(
            self.planner.master_seed,
            stream_path(
                ENV_STAGGER,
                worker=self.worker,
                shard=self.shard,
                generation=self.config.generation,
            ),
        )
        bound = max(0, int(self.config.stagger_max_steps))
        if bound == 0:
            return
        noop = {"blue": self.noop, "red": self.noop}
        for index, env in enumerate(self.vec.envs):
            steps = int(generator.integers(0, bound + 1))
            if steps == 0:
                continue
            obs: dict[str, Any] = {}
            for _ in range(steps):
                obs, _, terminated, truncated, infos = env.step(noop)
                if terminated["blue"] or truncated["blue"]:
                    obs, infos = env.reset()
                    self.ordinal[index] += 1
            for seat, agent in enumerate(("blue", "red")):
                row = 2 * index + seat
                for key, value in obs[agent].items():
                    self.obs[key][row] = value
                self.tick[row] = int(infos[agent]["tick"])
            self.episode_steps[2 * index : 2 * index + 2] = 0
            self.undiscounted_return[2 * index : 2 * index + 2] = 0.0
            for row in (2 * index, 2 * index + 1):
                self.reward_terms[row] = {}

    def close(self) -> None:
        """Close the battles and give back every view into the two segments.

        A numpy array over shared memory keeps that memory exported, and a segment with an
        exported pointer refuses to close, so the views go before the vec env does and the
        owner of the segment can free it.
        """
        if self.closed:
            return
        self.closed = True
        self.obs_view.release()
        self.obs_rows = None
        self.obs = {}
        self.vec.close()

    # -- the round ----------------------------------------------------------

    def _clear_round(self) -> None:
        self.reward[:] = 0.0
        self.terminated[:] = False
        self.truncated[:] = False
        self.deploy_status[:] = -1
        self.episode_end[:] = EPISODE_END_NONE
        self.round_steps[:] = self.episode_steps
        self.round_cards[:] = self.cards_played
        self.round_leak[:] = 0
        self.final_slots = np.zeros(0, dtype=np.int32)
        self.episodes = []

    def set_gamma(self, gamma: float) -> None:
        """Thread the discount the schedule is at into every reward function that takes one."""
        if self.gamma is not None and gamma == self.gamma:
            return
        self.gamma = gamma
        for env in self.vec.envs:
            setter = getattr(env.reward_fn, "set_gamma", None)
            if setter is not None:
                setter(gamma)

    def apply_plan(self, message: PlanMessage) -> None:
        """Take the iteration's opening table out of the plan region."""
        self.resident_snapshots = tuple(message.resident_snapshots)
        table = read_plan(plan_view(self.control_layout, self.control), self.control_layout.n_slots)
        offset = self.shard * self.n_slots
        rows = table[offset : offset + self.n_slots]
        self.group[:] = rows["group"]
        self.opponent_ix[:] = rows["opponent_ix"]

    def apply_assignments(self, parity: int) -> None:
        """Take the parent's answer for the battles whose episode started on the last round."""
        view = self.control_layout.view(self.control, self.shard, parity, "assign")
        rows = np.frombuffer(view, dtype=ASSIGN_DTYPE, count=self.n_slots)
        self.group[:] = rows["group"]
        self.opponent_ix[:] = rows["opponent_ix"]
        self.learner_seat[:] = rows["learner_seat"][::2]

    def apply_state(self, snapshots: Sequence[bytes | None]) -> None:
        """Start each battle from a recorded position. Off the hot path, by construction."""
        for index, blob in enumerate(snapshots[: self.games]):
            if blob is None:
                continue
            env = self.vec.envs[index]
            obs, infos = env.reset(options={"snapshot": blob})
            for seat, agent in enumerate(("blue", "red")):
                row = 2 * index + seat
                for key, value in obs[agent].items():
                    self.obs[key][row] = value
                self.tick[row] = int(infos[agent]["tick"])
            self.ordinal[index] += 1
            self.episode_steps[2 * index : 2 * index + 2] = 0

    def check_spaces(self) -> None:
        """Re-read the spaces and refuse a shard whose environment has changed under it."""
        spec = self.config.spec
        space = self.vec.single_observation_space
        for key, declared in spec.obs_space.items():
            shape = tuple(int(n) for n in space[key].shape)
            if shape != declared.shape:
                raise PreflightError(
                    f"worker {self.worker} shard {self.shard}: observation key {key!r} is "
                    f"{shape}, and this run was built against {declared.shape}"
                )
        if int(self.vec.single_action_space.n) != spec.n_actions:
            raise PreflightError(
                f"worker {self.worker} shard {self.shard}: the action space is "
                f"{int(self.vec.single_action_space.n)}, and this run was built against "
                f"{spec.n_actions}"
            )

    def step(self, actions: np.ndarray, gamma: float) -> None:
        """Play one decision in every battle of the shard."""
        self.set_gamma(gamma)
        self._clear_round()
        taken = np.asarray(actions, dtype=np.int16).copy()
        self.scripted.fill(taken, self.obs, self.group, self.opponent_ix)
        started = time.perf_counter_ns()
        obs, reward, terminated, truncated, infos = self.vec.step(taken)
        self.t_env_ns = time.perf_counter_ns() - started

        self.obs = obs
        self.reward[:] = reward
        self.terminated[:] = terminated
        self.truncated[:] = truncated
        self.tick[:] = np.asarray(infos["tick"], dtype=np.int32)
        status = np.asarray(infos["deploy_status"], dtype=np.int8)
        self.deploy_status[:] = status

        self.episode_steps += 1
        self.cards_played += (taken != self.noop) & (status == 0)
        self.illegal_commands += status > 0
        self.undiscounted_return += reward

        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        ended = np.flatnonzero(done)
        if ended.size:
            self._record_episodes(ended, infos)
        self.round_steps[:] = self.episode_steps
        self.round_cards[:] = self.cards_played
        self.episode_steps[ended] = 0
        self.cards_played[ended] = 0
        self.illegal_commands[ended] = 0
        self.undiscounted_return[ended] = 0.0
        for row in ended:
            self.reward_terms[int(row)] = {}

    def _record_episodes(self, ended: np.ndarray, infos: dict[str, Any]) -> None:
        """One record per seat whose episode finished on this round, in ascending slot order.

        The terminal scalars are copied out of ``final_info`` rather than reconstructed: the
        environment already computes them, its ``EPISODE_STAT_KEYS`` is the authority on the
        set, and a second implementation of the same summary would be a second answer.
        """
        final = infos.get("final_info")
        if final is None:
            raise KeyError(
                "the vec env reported an episode end without final_info; the harness reads the "
                "terminal statistics from there (royalegym.env.ClashSelfPlayVecEnv)"
            )
        valid = np.asarray(final.get("_episode_steps"), dtype=bool)
        truncated_rows = []
        for row in ended:
            row = int(row)
            if not valid[row]:
                raise KeyError(
                    f"slot {int(self.slots[row])} ended without its episode statistics; "
                    "final_info's validity mask does not cover it"
                )
            game, seat = divmod(row, 2)
            outcome, winner = self._outcome(final, row)
            self.episode_end[row] = (
                EPISODE_END_WIN
                if outcome > 0
                else EPISODE_END_LOSS
                if outcome < 0
                else EPISODE_END_DRAW
            )
            self.round_leak[row] = int(final["elixir_leak_steps"][row])
            self.episodes.append(
                EpisodeRecord(
                    slot=int(self.slots[row]),
                    worker=self.worker,
                    shard=self.shard,
                    battle=int(self.battles[game]),
                    seat=seat,
                    ordinal=int(self.ordinal[game]),
                    episode_seed_path=self.planner.env_seed_path(
                        self.worker, self.shard, self.config.generation
                    ),
                    policy_id=self._policy_id(row),
                    opponent_id=self._policy_id(2 * game + (1 - seat)),
                    bucket=BUCKETS[self._role(game)],
                    episode_steps=int(final["episode_steps"][row]),
                    episode_ticks=int(final["episode_ticks"][row]),
                    own_crowns=int(final["own_crowns"][row]),
                    enemy_crowns=int(final["enemy_crowns"][row]),
                    own_tower_hp_frac=float(final["own_tower_hp_frac"][row]),
                    enemy_tower_hp_frac=float(final["enemy_tower_hp_frac"][row]),
                    elixir_leak_steps=int(final["elixir_leak_steps"][row]),
                    elixir_count_exact=bool(final["elixir_count_exact"][row]),
                    winner=winner,
                    outcome=outcome,
                    cards_played=int(self.cards_played[row]),
                    illegal_commands=int(self.illegal_commands[row]),
                    undiscounted_return=float(self.undiscounted_return[row]),
                    reward_terms=dict(self.reward_terms[row]),
                )
            )
            if self.truncated[row]:
                truncated_rows.append(row)
        for game in np.unique(ended // 2):
            self.ordinal[int(game)] += 1
        self.final_slots = np.asarray(truncated_rows, dtype=np.int32)
        if truncated_rows:
            self._pack_finals(infos, truncated_rows)

    @staticmethod
    def _outcome(final: dict[str, Any], row: int) -> tuple[int, int]:
        """``(outcome, winner)`` for one finished seat.

        The environment writes them only when the engine ended the battle, so a truncation has
        neither: it is recorded as a draw with no winner, which is what it is from the seat's
        point of view and what keeps every finished episode countable.
        """
        mask = final.get("_outcome")
        if mask is None or not bool(np.asarray(mask, dtype=bool)[row]):
            return 0, -1
        return int(final["outcome"][row]), int(final["winner"][row])

    def _role(self, game: int) -> int:
        """What a battle is, read off the two seats' groups rather than carried separately."""
        blue, red = int(self.group[2 * game]), int(self.group[2 * game + 1])
        if blue == GROUP_SCRIPTED or red == GROUP_SCRIPTED:
            return ROLE_SCRIPTED
        if blue >= 0 or red >= 0:
            return ROLE_POOL
        return ROLE_MIRROR

    def _policy_id(self, row: int) -> str:
        group = int(self.group[row])
        if group == GROUP_LEARNER:
            return "learner"
        if group == GROUP_SCRIPTED:
            index = int(self.opponent_ix[row])
            return scripted_id(SCRIPTED_NAMES[index] if 0 <= index < len(SCRIPTED_NAMES) else "?")
        if group == GROUP_DEAD:
            return "dead"
        if group < len(self.resident_snapshots):
            return self.resident_snapshots[group]
        return f"snap:#{group}"

    def _pack_finals(self, infos: dict[str, Any], rows: Sequence[int]) -> None:
        """The final observation of every truncated seat, for the learner's bootstrap value.

        In ascending slot order, which is the order the parent recomputes from the truncation
        flags it was handed, so the correspondence needs no index of its own.
        """
        finals = infos["final_obs"]
        view = self.control_layout.view(
            self.control, self.shard, self.publish_parity(), "finals"
        )
        capacity = self.control_layout.finals_per_round
        if len(rows) > capacity:  # pragma: no cover - the region covers every slot of a shard
            raise IndexError(
                f"{len(rows)} truncated rows on one round, and the finals region holds "
                f"{capacity}"
            )
        for index, row in enumerate(rows):
            self.codec.pack(finals[row], view, index)

    # -- publishing ---------------------------------------------------------

    def publish_parity(self) -> int:
        """The parity the next publication goes to."""
        return self.publications % 2

    def open_parity(self) -> int:
        """The parity of the round the parent is answering: the last one published.

        Every region of a round -- the actions, the assignment, the final observations, the
        control word that carries the command -- is addressed by this, so a shard's two
        parities alternate without either side counting rounds the other cannot see.
        """
        return (self.publications - 1) % 2

    def publish(self, cycle: int) -> None:
        """Write this round's observations and scalars, and say so in the control word.

        The control word is written last and read first. It is the only thing either side
        polls, so the payload is complete by the time the state that announces it is visible.
        """
        parity = self.publish_parity()
        self.cycle = cycle
        if 0 <= cycle <= self.buffer_layout.cycles:
            for row in range(self.n_slots):
                one = {key: value[row] for key, value in self.obs.items()}
                self.codec.pack(
                    one, self.obs_view, self.buffer_layout.row_index(cycle, int(self.slots[row]))
                )
        scalars = self.control_layout.scalars(self.control, self.shard, parity)
        scalars["reward"] = self.reward
        scalars["tick"] = self.tick
        scalars["episode_steps"] = self.round_steps
        scalars["cards_played"] = np.minimum(self.round_cards, np.iinfo(np.int16).max)
        scalars["elixir_leak"] = np.minimum(self.round_leak, np.iinfo(np.int16).max)
        scalars["group"] = self.group
        scalars["flags"] = (
            FLAG_VALID
            + FLAG_TERMINATED * self.terminated
            + FLAG_TRUNCATED * self.truncated
        ).astype(np.uint8)
        scalars["deploy_status"] = self.deploy_status
        scalars["episode_end"] = self.episode_end
        self._write_control(parity, STATE_OBS_READY, cycle, ERR_NONE, 0)
        self.publications += 1

    def publish_error(self, text: str, code: int = ERR_EXCEPTION) -> None:
        """Hand the parent a traceback instead of a round, and stop."""
        parity = self.publish_parity()
        view = self.control_layout.view(self.control, self.shard, parity, "error")
        length = write_error(view, text)
        self._write_control(parity, STATE_OBS_READY, self.cycle, code, length)
        self.publications += 1

    def _write_control(
        self, parity: int, state: int, cycle: int, err_code: int, err_len: int
    ) -> None:
        view = self.control_layout.view(self.control, self.shard, parity, "control")
        CONTROL.pack_into(
            view,
            0,
            {
                "cycle": _CYCLE_SENTINEL if cycle < 0 else cycle,
                "t_env_ns": self.t_env_ns,
                "state": state,
                "n_slots": self.n_slots,
                "err_code": err_code,
                "err_len": err_len,
            },
        )

    # -- what a driver calls ------------------------------------------------

    def read_command(self, parity: int) -> tuple[int, float]:
        """The command word and the round parameter the parent wrote."""
        view = self.control_layout.view(self.control, self.shard, parity, "control")
        word = CONTROL.unpack_from(view, 0)
        return int(word["state"]), _gamma_from_word(int(word["t_env_ns"]))

    def read_actions(self, parity: int) -> np.ndarray:
        return self.control_layout.actions(self.control, self.shard, parity)

    def handle(self, command: int, gamma: float, parity: int, plan: PlanMessage | None) -> bool:
        """Apply one command. Returns False when the shard has been told to close.

        Every command but ``CLOSE`` ends in a publication, so the parent's wait is answered
        exactly once per command whatever the command was.
        """
        if command == COMMAND_CLOSE:
            self._write_control(parity, STATE_CLOSED, self.cycle, ERR_NONE, 0)
            self.close()
            return False
        if command == COMMAND_PLAN:
            if plan is None:
                raise ValueError("a PLAN round arrived without its message")
            self.apply_plan(plan)
            self._clear_round()
            self.publish(0)
            return True
        if command == COMMAND_DEFER:
            self.publish(self.cycle)
            return True
        if command == COMMAND_SPACES:
            self.check_spaces()
            self.publish(self.cycle)
            return True
        if command == COMMAND_SET_STATE:
            self.apply_state(self.pending_snapshots)
            self._clear_round()
            self.publish(self.cycle)
            return True
        if command == COMMAND_STEP:
            self.apply_assignments(parity)
            self.step(self.read_actions(parity), gamma)
            self.publish(self.cycle + 1)
            return True
        raise ValueError(f"command word {command} is not one this worker knows")


def _gamma_to_word(gamma: float) -> int:
    """The round's discount as the eight bytes the command carries it in.

    The control record's wide field is the child's environment time on the way up and the
    round's parameter on the way down; the two directions never overlap, because a command is
    written only after the publication it answers has been read.
    """
    return int(struct.unpack("<Q", struct.pack("<d", float(gamma)))[0])


def _gamma_from_word(word: int) -> float:
    return float(struct.unpack("<d", struct.pack("<Q", word & ((1 << 64) - 1)))[0])


# ---------------------------------------------------------------------------
# The parent side
# ---------------------------------------------------------------------------


class _RoundBuffers:
    """The parent's arrays for one shard's rounds, allocated once and refilled.

    A round's scalars come from as many segments as there are workers, so they are assembled
    rather than viewed; the observations are not, because every worker wrote them straight into
    the one rectangle.
    """

    def __init__(self, slots: np.ndarray) -> None:
        n = int(slots.size)
        self.slots = slots
        self.obs_rows = np.zeros(n, dtype=np.int32)
        self.group = np.zeros(n, dtype=np.int8)
        self.reward = np.zeros(n, dtype=np.float32)
        self.terminated = np.zeros(n, dtype=bool)
        self.truncated = np.zeros(n, dtype=bool)
        self.valid = np.zeros(n, dtype=bool)
        self.deploy_status = np.zeros(n, dtype=np.int8)
        self.tick = np.zeros(n, dtype=np.int32)
        self.episode_end = np.zeros(n, dtype=np.int8)
        self.episode_steps = np.zeros(n, dtype=np.int32)
        self.cards_played = np.zeros(n, dtype=np.int32)
        self.elixir_leak = np.zeros(n, dtype=np.int32)


class RolloutSourceBase(RolloutSource):
    """What the two shipped sources share: the rectangle, the round protocol and the table.

    Everything here is byte-level and driver-independent. The subclasses differ only in where
    a shard runs and how a publication is waited for.
    """

    def __init__(
        self,
        config: Any,
        spec: EnvSpec,
        codec_table: CodecTable,
        *,
        run_id: str,
        codec: str,
        geometry: Geometry | None = None,
    ) -> None:
        from ..config import geometry as compute_geometry

        self.config = config
        self.geometry = geometry if geometry is not None else compute_geometry(config)
        self._spec = spec
        self.codec_table = codec_table
        self.codec_path = codec
        self.run_id = run_id
        self.planner = SlotPlanner(self.geometry, config.master_seed)
        self.extra_modules = tuple(config.extra_component_modules)
        self.codec = build_codec(codec, spec, codec_table, self.extra_modules)
        self.row_bytes = int(self.codec.row_bytes(spec))
        self.buffer_layout: BufferLayout | None = None
        self.iteration = -1
        self.plan: SlotPlan | None = None
        self.generation = [0] * self.geometry.workers
        self.restarts = [0] * self.geometry.workers
        #: Whether a worker's slots take part in rounds, and whether it is waiting to. A
        #: restarted worker has reset its battles from a new seat of the seed, so the match its
        #: slots were collecting is gone: it rejoins at the next iteration boundary, and until
        #: then its cells arrive invalid rather than arriving wrong.
        self.live = [True] * self.geometry.workers
        self.rejoining = [False] * self.geometry.workers
        self.failures: list[WorkerFailure] = []
        self._rounds = {
            shard: _RoundBuffers(self.planner.round_slots(shard))
            for shard in range(self.geometry.shards_per_worker)
        }
        self._round = {
            shard: RolloutRound(
                cycle=CYCLE_UNBOUND,
                shard=shard,
                slots=self._rounds[shard].slots,
                obs_rows=self._rounds[shard].obs_rows,
                group=self._rounds[shard].group,
                reward=self._rounds[shard].reward,
                terminated=self._rounds[shard].terminated,
                truncated=self._rounds[shard].truncated,
                valid=self._rounds[shard].valid,
                deploy_status=self._rounds[shard].deploy_status,
                tick=self._rounds[shard].tick,
                episode_end=self._rounds[shard].episode_end,
            )
            for shard in range(self.geometry.shards_per_worker)
        }
        # The parent owns the assignment table and is the only thing that knows it. It is
        # written in full on every step: a worker that had to remember which entries it was
        # last told about would be a second place where what a battle is playing lives.
        self._group = np.full(self.geometry.n_slots, GROUP_LEARNER, dtype=np.int8)
        self._opponent = np.full(self.geometry.n_slots, NO_OPPONENT, dtype=np.int8)
        self._learner_seat = np.full(self.geometry.n_battles, -1, dtype=np.int8)
        self._final_slots: dict[int, np.ndarray] = {}
        self._final_rows: dict[int, np.ndarray] = {}
        self._cycle = 0
        self._next_shard = 0
        self._open_shard: int | None = None
        self._buffer_name = ""
        self._wait_ns = 0
        self._env_ns = 0
        self._rounds_read = 0
        self.closed = False

    # -- the ABC ------------------------------------------------------------

    def spec(self) -> EnvSpec:
        return self._spec

    def stats(self) -> dict[str, float]:
        if self._rounds_read == 0:
            return {}
        env_ms = self._env_ns / 1e6 / self._rounds_read
        wait_ms = self._wait_ns / 1e6 / self._rounds_read
        return {
            "env_ms": env_ms,
            "wait_ms": wait_ms,
            "idle_frac": wait_ms / max(env_ms + wait_ms, 1e-9),
            "bytes_out": float(self.row_bytes * self.geometry.n_slots),
        }

    def drain_failures(self) -> list[WorkerFailure]:
        out, self.failures = self.failures, []
        return out

    def begin_iteration(
        self, plan: SlotPlan, buffer: ExperienceBuffer, iteration: int
    ) -> None:
        handle = buffer.shared_handle()
        self._attach_buffer(handle)
        for worker in range(self.geometry.workers):
            if self.rejoining[worker]:
                self.live[worker] = True
                self.rejoining[worker] = False
        self.plan = plan
        self.iteration = iteration
        for assignment in plan.assignment:
            battle = assignment.battle
            self._group[2 * battle : 2 * battle + 2] = assignment.group
            self._opponent[2 * battle : 2 * battle + 2] = opponent_index(plan, assignment)
            self._learner_seat[battle] = assignment.learner_seat
        self._cycle = 0
        self._next_shard = 0
        self._open_shard = None
        self._start(handle)
        for worker in range(self.geometry.workers):
            self._write_plan(worker, plan)
        # Every shard is holding an observation -- its first, or the bootstrap row of the
        # iteration that just ended -- and republishes it into cycle 0 of the new rectangle,
        # so an iteration boundary is a change of table and not a discontinuity in the battle.
        for shard in range(self.geometry.shards_per_worker):
            self._await_publication(shard, self.config.rollout.round_timeout_s)
            self._send(shard, COMMAND_PLAN, 0.0, PlanMessage(iteration, plan.resident_snapshots))

    def finish_iteration(self, timeout_s: float | None = None) -> list[RolloutRound]:
        """Wait for the trailing publication of every shard, and return them.

        It is not bookkeeping. The last cycle's actions were submitted and the observation
        they led to is the bootstrap row -- the one a truncated or unfinished trajectory is
        bootstrapped from -- and the reward of that last transition arrives with it. Reading
        the rectangle before this has returned reads a row the worker is still writing.
        """
        if self._open_shard is not None:
            raise RuntimeError(
                f"finish_iteration() with shard {self._open_shard} still awaiting its submit"
            )
        timeout = self.config.rollout.round_timeout_s if timeout_s is None else timeout_s
        rounds = []
        for shard in range(self.geometry.shards_per_worker):
            self._await_publication(shard, timeout)
            rounds.append(self._assemble(shard))
        return rounds

    def next_round(self, timeout_s: float = 30.0) -> RolloutRound:
        if self._open_shard is not None:
            raise RuntimeError(
                f"next_round() was called with shard {self._open_shard} still awaiting its "
                "submit; the protocol is exactly one submit per round"
            )
        shard = self._next_shard
        self._await_publication(shard, timeout_s)
        round_ = self._assemble(shard)
        self._open_shard = shard
        return round_

    def submit(self, command: WorkerCommand) -> None:
        shard = self._open_shard
        if shard is None:
            raise RuntimeError("submit() without a round to answer")
        self._open_shard = None
        word = command_of(command)
        gamma = 0.0
        message: PlanMessage | None = None
        if isinstance(command, Step):
            gamma = float(command.gamma)
            self._write_step(shard, command)
        elif isinstance(command, Plan):
            message = PlanMessage(command.plan.iteration, command.plan.resident_snapshots)
            for worker in range(self.geometry.workers):
                self._write_plan(worker, command.plan)
        elif isinstance(command, SetState):
            self._write_snapshots(shard, command.snapshots)
        self._send(shard, word, gamma, message)
        if word == COMMAND_STEP:
            self._next_shard += 1
            if self._next_shard == self.geometry.shards_per_worker:
                self._next_shard = 0
                self._cycle += 1

    def final_rows(self, round_: RolloutRound) -> tuple[np.ndarray, np.ndarray]:
        """The packed final observation of every truncated seat of a round.

        Slots ascending, and one packed row each, in the same order. A truncated transition is
        bootstrapped from the value of the observation the episode actually ended on, and that
        observation is not in the rectangle: what the rectangle holds at that cell is the first
        observation of the next episode, because the environment resets on the step that ends
        one.
        """
        slots = self._final_slots.get(round_.shard)
        rows = self._final_rows.get(round_.shard)
        if slots is None or rows is None:  # pragma: no cover - a round is always assembled first
            return np.zeros(0, dtype=np.int32), np.zeros((0, self.row_bytes), dtype=np.uint8)
        return slots, rows

    # -- assembling a round -------------------------------------------------

    def _assemble(self, shard: int) -> RolloutRound:
        buffers = self._rounds[shard]
        layout = self.buffer_layout
        assert layout is not None
        cycle = -1
        per_worker = self.geometry.slots_per_shard
        finals_slots: list[np.ndarray] = []
        finals_rows: list[np.ndarray] = []
        episodes: list[EpisodeRecord] = []
        env_ns = 0
        for worker in range(self.geometry.workers):
            lo = worker * per_worker
            hi = lo + per_worker
            if not self._alive(worker):
                buffers.valid[lo:hi] = False
                buffers.group[lo:hi] = GROUP_DEAD
                buffers.reward[lo:hi] = 0.0
                buffers.terminated[lo:hi] = False
                buffers.truncated[lo:hi] = False
                buffers.deploy_status[lo:hi] = -1
                buffers.episode_end[lo:hi] = EPISODE_END_NONE
                continue
            word, scalars = self._read_publication(worker, shard)
            env_ns += int(word["t_env_ns"])
            if int(word["err_code"]) != ERR_NONE:
                self._fail_worker(worker, shard, word)
                buffers.valid[lo:hi] = False
                buffers.group[lo:hi] = GROUP_DEAD
                continue
            published = _cycle_of(word)
            if cycle >= 0 and published != cycle:
                raise ValueError(
                    f"shard {shard} was published at cycle {published} by worker {worker} and "
                    f"at cycle {cycle} by an earlier one; a round is one cycle of one shard"
                )
            cycle = published
            buffers.valid[lo:hi] = (scalars["flags"] & FLAG_VALID) != 0
            buffers.terminated[lo:hi] = (scalars["flags"] & FLAG_TERMINATED) != 0
            buffers.truncated[lo:hi] = (scalars["flags"] & FLAG_TRUNCATED) != 0
            buffers.group[lo:hi] = scalars["group"]
            buffers.reward[lo:hi] = scalars["reward"]
            buffers.deploy_status[lo:hi] = scalars["deploy_status"]
            buffers.tick[lo:hi] = scalars["tick"]
            buffers.episode_end[lo:hi] = scalars["episode_end"]
            buffers.episode_steps[lo:hi] = scalars["episode_steps"]
            buffers.cards_played[lo:hi] = scalars["cards_played"]
            buffers.elixir_leak[lo:hi] = scalars["elixir_leak"]
            ends = int(np.count_nonzero(scalars["episode_end"]))
            if ends:
                episodes.extend(self._read_episodes(worker, shard, ends))
            truncated = np.flatnonzero((scalars["flags"] & FLAG_TRUNCATED) != 0)
            if truncated.size:
                finals_slots.append(buffers.slots[lo:hi][truncated])
                finals_rows.append(self._read_finals(worker, shard, truncated.size))
        if cycle >= 0:
            buffers.obs_rows[:] = (cycle + layout.history_rows) * layout.n_slots + buffers.slots
        else:
            buffers.obs_rows[:] = -1
        self._final_slots[shard] = (
            np.concatenate(finals_slots) if finals_slots else np.zeros(0, dtype=np.int32)
        )
        self._final_rows[shard] = (
            np.concatenate(finals_rows)
            if finals_rows
            else np.zeros((0, self.row_bytes), dtype=np.uint8)
        )
        round_ = self._round[shard]
        round_.cycle = cycle
        round_.episodes = episodes
        round_.timings = {"env_ms": env_ns / 1e6}
        self._env_ns += env_ns
        self._rounds_read += 1
        return round_

    def _write_step(self, shard: int, command: Step) -> None:
        """The round's actions and the whole assignment table, into every live worker.

        The command's arrays are the parent's answer for the battles that just reset, or empty
        when none did; either way what crosses is the table as it now stands, because a table
        assembled from updates on the far side would be a copy that could fall behind.
        """
        buffers = self._rounds[shard]
        slots = buffers.slots
        per_worker = self.geometry.slots_per_shard
        actions = np.asarray(command.actions, dtype=np.int16)
        if actions.size != slots.size:
            raise ValueError(
                f"a Step for shard {shard} carried {actions.size} actions, and the round has "
                f"{slots.size} slots"
            )
        group = np.asarray(command.group, dtype=np.int8)
        opponent = np.asarray(command.opponent_ix, dtype=np.int8)
        seat = np.asarray(command.learner_seat, dtype=np.int8)
        if group.size:
            if group.size != slots.size or opponent.size != slots.size:
                raise ValueError(
                    "a Step's group and opponent_ix are either empty or one entry per slot of "
                    f"the round; got {group.size} and {opponent.size} against {slots.size}"
                )
            self._group[slots] = group
            self._opponent[slots] = opponent
        if seat.size:
            battles = slots[::2] // 2
            if seat.size != battles.size:
                raise ValueError(
                    "a Step's learner_seat is either empty or one entry per battle of the "
                    f"round; got {seat.size} against {battles.size} battles"
                )
            self._learner_seat[battles] = seat
        for worker in range(self.geometry.workers):
            if not self._alive(worker):
                continue
            lo = worker * per_worker
            hi = lo + per_worker
            block = slots[lo:hi]
            parity = self._parity(worker, shard)
            self._actions_view(worker, shard, parity)[:] = actions[lo:hi]
            rows = self._assign_view(worker, shard, parity)
            rows["group"] = self._group[block]
            rows["opponent_ix"] = self._opponent[block]
            rows["learner_seat"] = np.repeat(self._learner_seat[block[::2] // 2], 2)

    def _write_plan(self, worker: int, plan: SlotPlan) -> None:
        layout = self._control_layout(worker)
        view = plan_view(layout, self._control_memory(worker))
        write_plan(view, plan, self.planner.worker_slots(worker), self.planner)

    def _write_snapshots(self, shard: int, snapshots: Sequence[bytes | None]) -> None:
        raise NotImplementedError

    # -- what a driver implements -------------------------------------------

    def _attach_buffer(self, handle: BufferHandle) -> None:
        if self.buffer_layout is None:
            if handle.row_bytes != self.row_bytes:
                raise ValueError(
                    f"the buffer's rows are {handle.row_bytes} B and this codec packs "
                    f"{self.row_bytes} B"
                )
            self.buffer_layout = handle.layout()
        elif handle.name != self._buffer_name:
            raise ValueError(
                f"the rectangle changed under the source: it was {self._buffer_name!r} and is "
                f"now {handle.name!r}"
            )
        self._buffer_name = handle.name

    def _alive(self, worker: int) -> bool:
        """Whether this worker's slots are part of a round right now."""
        return self.live[worker]

    def _start(self, handle: BufferHandle) -> None:
        raise NotImplementedError

    def _parity(self, worker: int, shard: int) -> int:
        raise NotImplementedError

    def _control_layout(self, worker: int) -> ControlLayout:
        raise NotImplementedError

    def _control_memory(self, worker: int) -> Any:
        raise NotImplementedError

    def _await_publication(self, shard: int, timeout_s: float) -> None:
        raise NotImplementedError

    def _read_publication(self, worker: int, shard: int) -> tuple[dict[str, Any], np.ndarray]:
        raise NotImplementedError

    def _read_episodes(self, worker: int, shard: int, count: int) -> list[EpisodeRecord]:
        raise NotImplementedError

    def _read_finals(self, worker: int, shard: int, count: int) -> np.ndarray:
        raise NotImplementedError

    def _send(
        self, shard: int, command: int, gamma: float, message: PlanMessage | None
    ) -> None:
        raise NotImplementedError

    def _fail_worker(self, worker: int, shard: int, word: dict[str, Any]) -> None:
        raise NotImplementedError

    def _actions_view(self, worker: int, shard: int, parity: int) -> np.ndarray:
        return self._control_layout(worker).actions(
            self._control_memory(worker), shard, parity
        )

    def _assign_view(self, worker: int, shard: int, parity: int) -> np.ndarray:
        view = self._control_layout(worker).view(
            self._control_memory(worker), shard, parity, "assign"
        )
        return np.frombuffer(view, dtype=ASSIGN_DTYPE, count=self.geometry.slots_per_shard)


def _cycle_of(word: dict[str, Any]) -> int:
    cycle = int(word["cycle"])
    return CYCLE_UNBOUND if cycle == _CYCLE_SENTINEL else cycle


# ---------------------------------------------------------------------------
# What a round has to be true of
# ---------------------------------------------------------------------------


def check_round(round_: RolloutRound, *, cycle: int, slots: np.ndarray) -> None:
    """Refuse a round that is not the one that was asked for.

    A dropped cycle and a slot count that does not match the geometry are the two ways the
    rectangle silently stops being a rectangle: the first writes a transition into the wrong
    row and the second leaves a column of it never written. Both are one comparison.
    """
    if round_.cycle != cycle:
        raise ValueError(
            f"shard {round_.shard} published cycle {round_.cycle} where cycle {cycle} was due; "
            "a cycle the buffer never receives is a row of it that is never written"
        )
    if round_.slots.size != np.asarray(slots).size:
        raise ValueError(
            f"shard {round_.shard} covered {round_.slots.size} slots at cycle {cycle} and the "
            f"geometry says {np.asarray(slots).size}"
        )
    if not np.array_equal(round_.slots, np.asarray(slots)):
        raise ValueError(
            f"shard {round_.shard} covered slots that are not this shard's, or not in "
            f"ascending order, at cycle {cycle}"
        )


def check_actions_legal(actions: np.ndarray, mask: np.ndarray, slots: np.ndarray) -> None:
    """Refuse an action that its own stored mask says is illegal.

    The mask that travels with the transition is the one the update applies, so an action that
    is illegal under it makes ``log pi`` negative infinity at the first minibatch. Checking it
    where the action is chosen is what turns that into a message naming the slot.
    """
    actions = np.asarray(actions)
    mask = np.asarray(mask)
    taken = mask[np.arange(actions.size), actions.astype(np.int64)]
    wrong = np.flatnonzero(~taken.astype(bool))
    if wrong.size:
        first = int(wrong[0])
        raise ValueError(
            f"{wrong.size} action(s) are illegal under the mask that was handed out; slot "
            f"{int(np.asarray(slots)[first])} took action {int(actions[first])}"
        )


def assignments_constant_within_episodes(
    group: np.ndarray, episode_end: np.ndarray
) -> None:
    """Refuse a rectangle in which a seat changed hands in the middle of an episode.

    One vectorised comparison over ``(T, R)``: a slot's group may differ from the cycle before
    it only at a cycle that reported an episode end. ``episode_end[t]`` is the end of the step
    that ARRIVED at cycle ``t``, so the observation at ``t`` is already the next episode's
    first and the assignment drawn for it belongs at ``t`` -- which is why the end and the
    change are on the same row rather than one apart.

    It is what makes "one policy for one whole episode" checkable rather than merely intended,
    and it catches every way an assignment could reach the middle of a trajectory.
    """
    group = np.asarray(group)
    episode_end = np.asarray(episode_end)
    if group.ndim != 2 or group.shape != episode_end.shape:
        raise ValueError(
            f"group is {group.shape} and episode_end is {episode_end.shape}; both are "
            "(cycles, slots)"
        )
    changed = group[1:] != group[:-1]
    allowed = episode_end[1:] != EPISODE_END_NONE
    offending = np.argwhere(changed & ~allowed)
    if offending.size:
        cycle, slot = (int(v) for v in offending[0])
        raise ValueError(
            f"{len(offending)} assignment change(s) landed inside an episode; slot {slot} "
            f"changed from {int(group[cycle, slot])} to {int(group[cycle + 1, slot])} between "
            f"cycles {cycle} and {cycle + 1} with no episode end on {cycle}"
        )


def check_bootstrap(truncated: np.ndarray, final_slots: np.ndarray, slots: np.ndarray) -> None:
    """Refuse a truncated transition with no final observation to bootstrap from.

    A termination bootstraps from zero and a truncation bootstraps from the value of the state
    the episode was cut off in. That state is not in the rectangle -- the environment resets on
    the step that ends an episode, so what the rectangle holds is the next episode's first
    observation -- so a truncation whose final row is missing would be bootstrapped from
    another battle.
    """
    truncated = np.asarray(truncated, dtype=bool)
    expected = np.asarray(slots)[truncated]
    got = np.asarray(final_slots)
    if not np.array_equal(np.sort(expected), np.sort(got)):
        missing = sorted(set(expected.tolist()) - set(got.tolist()))
        extra = sorted(set(got.tolist()) - set(expected.tolist()))
        raise ValueError(
            f"the truncated slots and the final observations do not match: {len(missing)} "
            f"truncation(s) with no final row {missing[:8]}, {len(extra)} final row(s) with no "
            f"truncation {extra[:8]}"
        )


class InlineRolloutSource(RolloutSourceBase):
    """Every shard in the learner's own process, over the same bytes.

    No processes, no shared memory for the control segments -- a plain ``bytearray`` has the
    same offsets -- and no waiting: a publication is complete by the time the call that
    produced it returns. What it does have is the same rectangle, the same scalar records, the
    same table and the same seeds, which is what makes it the reference a process farm is
    checked against rather than a convenience.
    """

    def __init__(
        self,
        config: Any,
        spec: EnvSpec,
        codec_table: CodecTable,
        *,
        run_id: str = "inline",
        codec: str = "royalelearn.rollout.codec.SpatialObsCodec",
        geometry: Geometry | None = None,
        viser: bool = False,
    ) -> None:
        super().__init__(
            config, spec, codec_table, run_id=run_id, codec=codec, geometry=geometry
        )
        self.viser = viser
        self._control: dict[int, bytearray] = {}
        self._layouts: dict[int, ControlLayout] = {}
        self._runners: dict[tuple[int, int], ShardRunner] = {}
        self._buffer: Any = None
        self._started = False

    def _start(self, handle: BufferHandle) -> None:
        if self._started:
            return
        from multiprocessing import shared_memory

        self._buffer = shared_memory.SharedMemory(name=handle.name)
        for worker in range(self.geometry.workers):
            config = self._worker_config(worker, handle)
            layout = config.control.layout()
            self._layouts[worker] = layout
            self._control[worker] = bytearray(layout.total_bytes)
            planner = self.planner
            for shard in range(self.geometry.shards_per_worker):
                runner = ShardRunner(
                    config,
                    shard,
                    planner=planner,
                    codec=self.codec,
                    buffer=self._buffer.buf,
                    control=self._control[worker],
                    viser="env" if (self.viser and worker == 0 and shard == 0) else None,
                )
                runner.start()
                runner.publish(CYCLE_UNBOUND)
                self._runners[(worker, shard)] = runner
        self._started = True

    def _worker_config(self, worker: int, handle: BufferHandle) -> WorkerConfig:
        control = ControlLayout(
            run_id=self.run_id,
            worker=worker,
            shards=self.geometry.shards_per_worker,
            slots_per_shard=self.geometry.slots_per_shard,
            row_bytes=self.row_bytes,
        ).handle()
        rollout = self.config.rollout
        return WorkerConfig(
            worker=worker,
            generation=self.generation[worker],
            run_id=self.run_id,
            master_seed=self.config.master_seed,
            geometry=self.geometry,
            env=self.config.env,
            spec=self._spec,
            codec=self.codec_path,
            codec_table=self.codec_table,
            buffer=handle,
            control=control,
            extra_component_modules=self.extra_modules,
            spin_us=rollout.spin_us,
            stagger_first_reset=rollout.stagger_first_reset,
            viser=self.viser and worker == 0,
        )

    def _parity(self, worker: int, shard: int) -> int:
        return self._runners[(worker, shard)].open_parity()

    def _control_layout(self, worker: int) -> ControlLayout:
        return self._layouts[worker]

    def _control_memory(self, worker: int) -> Any:
        return self._control[worker]

    def _await_publication(self, shard: int, timeout_s: float) -> None:
        return None

    def _read_publication(self, worker: int, shard: int) -> tuple[dict[str, Any], np.ndarray]:
        layout = self._layouts[worker]
        parity = self._runners[(worker, shard)].open_parity()
        word = CONTROL.unpack_from(layout.view(self._control[worker], shard, parity, "control"), 0)
        return word, layout.scalars(self._control[worker], shard, parity)

    def _read_episodes(self, worker: int, shard: int, count: int) -> list[EpisodeRecord]:
        episodes = self._runners[(worker, shard)].episodes
        if len(episodes) != count:  # pragma: no cover - the two are written together
            raise ValueError(
                f"worker {worker} shard {shard} reported {count} episode ends and produced "
                f"{len(episodes)} records"
            )
        return list(episodes)

    def _read_finals(self, worker: int, shard: int, count: int) -> np.ndarray:
        layout = self._layouts[worker]
        parity = self._runners[(worker, shard)].open_parity()
        view = layout.view(self._control[worker], shard, parity, "finals")
        return np.frombuffer(view, dtype=np.uint8, count=count * self.row_bytes).reshape(
            count, self.row_bytes
        )

    def _send(
        self, shard: int, command: int, gamma: float, message: PlanMessage | None
    ) -> None:
        for worker in range(self.geometry.workers):
            runner = self._runners[(worker, shard)]
            parity = runner.open_parity()
            view = self._layouts[worker].view(self._control[worker], shard, parity, "control")
            CONTROL.pack_into(
                view,
                0,
                {
                    "cycle": _CYCLE_SENTINEL,
                    "t_env_ns": _gamma_to_word(gamma),
                    "state": command,
                    "n_slots": runner.n_slots,
                    "err_code": ERR_NONE,
                    "err_len": 0,
                },
            )
            word, round_gamma = runner.read_command(parity)
            runner.handle(word, round_gamma, parity, message)

    def _write_snapshots(self, shard: int, snapshots: Sequence[bytes | None]) -> None:
        for worker in range(self.geometry.workers):
            runner = self._runners[(worker, shard)]
            battles = self.planner.shard_battles(worker, shard)
            runner.pending_snapshots = tuple(
                snapshots[int(battle)] if int(battle) < len(snapshots) else None
                for battle in battles
            )

    def _fail_worker(self, worker: int, shard: int, word: dict[str, Any]) -> None:
        layout = self._layouts[worker]
        parity = self._runners[(worker, shard)].open_parity()
        view = layout.view(self._control[worker], shard, parity, "error")
        self.failures.append(
            WorkerFailure(
                worker=worker,
                shard=shard,
                cycle=_cycle_of(word),
                kind="exception",
                message=read_error(view, int(word["err_len"])),
            )
        )

    def restart(self, worker: int) -> None:
        """Rebuild one worker's shards from the next generation.

        In this process a restart cannot recover a crash -- an exception here reached the
        caller -- but it is the same operation the farm performs, and doing it the same way is
        what keeps the two sources' seeds in step when a test drives a restart.
        """
        self.generation[worker] += 1
        self.restarts[worker] += 1
        self.live[worker] = False
        self.rejoining[worker] = True
        handle = BufferHandle(
            name=self._buffer_name,
            size=self._buffer.size,
            run_id=self.run_id,
            cycles=self.buffer_layout.cycles,
            n_slots=self.buffer_layout.n_slots,
            row_bytes=self.row_bytes,
            codec_version=self.buffer_layout.codec_version,
            frame_stack=self.buffer_layout.frame_stack,
        )
        config = self._worker_config(worker, handle)
        for shard in range(self.geometry.shards_per_worker):
            self._runners[(worker, shard)].close()
            runner = ShardRunner(
                config,
                shard,
                planner=self.planner,
                codec=self.codec,
                buffer=self._buffer.buf,
                control=self._control[worker],
                viser=None,
            )
            runner.start()
            runner.publish(CYCLE_UNBOUND)
            self._runners[(worker, shard)] = runner

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for runner in self._runners.values():
            runner.close()
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None
