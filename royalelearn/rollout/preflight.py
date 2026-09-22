"""The start-up gates: what is checked before a run is allowed to collect anything.

Every gate here answers a question whose wrong answer is silent. A stale engine build produces
a battle that is not the one the calibration describes; a mask that disagrees with the engine
produces a policy trained to take actions the game refuses; a vector field resolved by
arithmetic rather than by name produces a pointer head reading somebody else's numbers. None
of those shows up as an exception later -- they show up as a training curve that means
nothing -- so they are checked once, at the front, and a failure is a refusal to start.

The order matters and is the order of ``docs/harness-spec.md`` section 7.7. In particular the
environment is reset before anything is read off it: a freshly constructed env reports a
decision granularity it is about to revise, because the granularity is a function of the tick
length the engine reports for a live battle.

Nothing here holds a width, a plane count or a field offset. Every number in the report was
read from the environment that was built, which is why the whole suite runs against MockEngine,
whose catalogue is not the one a real run uses.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..api.buffer import MIN_TABLE_STATES, CodecTable
from ..config import Geometry, RunConfig, geometry
from ..errors import PreflightError, StaleEngineBuild
from ..identity import EngineBuild, engine_build
from ..obs_layout import hand_fields
from ..seeding import PREFLIGHT_ENV, PREFLIGHT_SAMPLE, derive_generator, derive_int, stream_path
from .envspec import ComponentSpec, EnvFactorySpec, read_env_spec
from .inline import build_codec
from .scripted import RANDOM_LEGAL_NOOP_PROB

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.buffer import ObsCodec
    from ..api.rollout import EnvSpec

__all__ = [
    "DEFAULT_CODEC",
    "PreflightReport",
    "ram_ledger",
    "run_preflight",
]

#: The codec a run uses unless it is told otherwise.
DEFAULT_CODEC = "royalelearn.rollout.codec.SpatialObsCodec"

#: What to do about a build that no longer matches the data it was compiled from. The engine
#: reports which keys differ; the harness hashes no data file of its own, because a second
#: opinion about what the engine is running on is exactly what a stale build looks like.
REBUILD_COMMAND = "maturin develop --release, in the RoyaleSim checkout"

#: The measured resident cost of the pieces of a run, from ``docs/harness-spec.md`` section
#: 2.4. They are what ``doctor`` projects a run's footprint from; the buffer, which is the one
#: that scales, is computed from the codec's own row size rather than taken from here.
PARENT_MB = 1400.0
WORKER_ENGINE_MB_PER_BATTLE = 7.8
WORKER_OVERHEAD_MB = 40.0
INTERPRETER_MB = 400.0


@dataclass(slots=True)
class PreflightReport:
    """What preflight learned, in the order the run needs it.

    A value rather than a printout: the coordinator puts ``spec``, ``table`` and ``build`` into
    the run identity and the checkpoint, the farm builds workers from ``spec`` and ``table``,
    and ``lines`` is what a person reads.
    """

    spec: EnvSpec
    env_config: dict[str, Any]
    table: CodecTable
    codec: str
    codec_version: int
    row_bytes: int
    build: EngineBuild
    statics: np.ndarray
    geometry: Geometry
    ram: dict[str, float]
    lines: list[str] = field(default_factory=list)
    #: The run this preflight belongs to, once it is known. Empty until it is.
    run_id: str = ""

    @property
    def codec_table_digest(self) -> str:
        return self.table.digest()

    def note_run_id(self, value: str, *, printer: Callable[[str], None] | None = print) -> None:
        """Record the run's identity and write the last line of the start-up summary.

        The identity cannot be computed while preflight runs -- it is a function of what
        preflight found and of the architecture the network factory builds -- so the line that
        names it is written once both are in hand. Recording it here is what makes the string a
        person read at start-up the same string that goes into the checkpoint.
        """
        self.run_id = value
        line = f"run id        {value}"
        self.lines.append(line)
        if printer is not None:
            printer(line)


def run_preflight(
    config: RunConfig,
    *,
    codec: str = DEFAULT_CODEC,
    table_samples: int = 1000,
    mask_samples: int = 1000,
    printer: Callable[[str], None] | None = print,
    mask_disagreement_gate: bool | None = None,
    min_table_states: int = MIN_TABLE_STATES,
    run_id: str = "",
) -> PreflightReport:
    """Build one environment, read everything off it, and refuse the run or return the facts.

    The environment built here is preflight's own: it is seeded and played from preflight's own
    two streams and closed before a worker starts. It does not borrow worker 0's, because two
    consumers on one path advance each other's stream and the first battle of the run would
    then depend on how many observations a gate happened to sample.

    ``run_id`` is printed when the caller already knows it, which a resume does. A fresh run
    does not: the identity is a function of what this returns and of the architecture the
    network factory builds, so the coordinator names it through ``PreflightReport.note_run_id``
    once both are in hand.
    """
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)
        if printer is not None:
            printer(text)

    geo = geometry(config)
    vec = _construct(config.env, tuple(config.extra_component_modules))
    try:
        spec = read_env_spec(
            vec,
            config.env,
            frame_stack=config.obs.frame_stack,
            seed=derive_int(config.master_seed, stream_path(PREFLIGHT_ENV)),
        )
        env = vec.envs[0]
        env_config = env.config()
        say(f"engine        {env_config['engine']['class']} on {spec.num_cards} cards")
        say(
            f"timing        {spec.decision_ms} ms per decision = {spec.decision_ticks} ticks "
            f"of {spec.tick_ms} ms; {spec.regular_ticks} + {spec.overtime_ticks} ticks a match"
        )

        obs_codec = build_codec(codec, spec, None, config.extra_component_modules)
        sample = _sample(vec, config.master_seed, table_samples)
        # The codec decides the table from the sample and binds itself to it, so what computes
        # the row size and reads the static planes below is the table that was just decided.
        table = obs_codec.table(spec, sample, min_states=min_table_states)
        row_bytes = int(obs_codec.row_bytes(spec))
        for line in _table_lines(spec, table, row_bytes):
            say(line)

        hand = hand_fields(spec)
        say(
            f"hand fields   one-hot {hand.card_onehot.start}:{hand.card_onehot.stop} of "
            f"{hand.onehot_width} per slot, cost {hand.cost.start}:{hand.cost.stop}, "
            f"affordable {hand.affordable.start}:{hand.affordable.stop}"
        )

        run_gate = (
            config.doctor.run_mask_disagreement_gate
            if mask_disagreement_gate is None
            else mask_disagreement_gate
        )
        if run_gate:
            _mask_disagreement_gate(env, say)
        _noop_gate(env, config, sample, spec, mask_samples, say)
        _action_layout_gate(env, spec, sample, say)

        statics = _statics(obs_codec, sample, say)
        build = engine_build(env_config, env.engine.cards())
        say(
            f"engine build  calibration {build.calibration_digest[:12]} build "
            f"{(build.build_digest or 'none')[:12]} catalogue {build.catalogue_sha256[:12]}"
        )
    finally:
        vec.close()

    ram = ram_ledger(config, geo, row_bytes)
    for line in _ram_lines(ram):
        say(line)
    if ram["total_mb"] > config.doctor.ram_budget_mb:
        raise PreflightError(
            f"this run projects {ram['total_mb']:.0f} MB resident and doctor.ram_budget_mb is "
            f"{config.doctor.ram_budget_mb} MB. Lower rollout.workers, "
            f"rollout.games_per_worker or ppo.timesteps_per_iteration, or raise the budget if "
            f"the machine really has the memory."
        )

    say(
        f"geometry      {geo.workers} workers x {geo.games_per_worker} battles = "
        f"{geo.n_battles} battles, {geo.n_slots} slots, {geo.learner_rows} of them learner rows"
    )
    say(
        f"iteration     {geo.cycles} cycles for {geo.timesteps_per_iteration} timesteps; "
        f"credit horizon {_credit_horizon_s(config, spec):.1f} s"
    )
    say(_viser_line(config))
    report = PreflightReport(
        spec=spec,
        env_config=env_config,
        table=table,
        codec=codec,
        codec_version=int(obs_codec.codec_version),
        row_bytes=row_bytes,
        build=build,
        statics=statics,
        geometry=geo,
        ram=ram,
        lines=lines,
    )
    if run_id:
        report.note_run_id(run_id, printer=printer)
    return report


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def _construct(factory: EnvFactorySpec, extra_modules: tuple[str, ...]) -> Any:
    """Gate 1: one engine, constructed. A stale build dies here and not at cycle 0."""
    try:
        return factory.build_vec(1, extra_modules, viser=None)
    except RuntimeError as exc:
        text = str(exc)
        if "calibration" not in text and "build" not in text:
            raise
        raise StaleEngineBuild(text, rebuild_command=REBUILD_COMMAND) from exc


def _sample(vec: Any, master_seed: int, count: int) -> list[dict[str, np.ndarray]]:
    """Real observations to decide storage from, and real states to check the mask on.

    Played by the random-legal opponent on both seats, which is the only sampler that reaches
    states a no-op walk never does: a board with units on it, a hand that has cycled and a
    tower that has taken damage.
    """
    generator = derive_generator(master_seed, stream_path(PREFLIGHT_SAMPLE))
    rows: list[dict[str, np.ndarray]] = []
    # No seed: the environment was seeded when its spec was read, and this continues that
    # stream rather than starting a second one beside it.
    obs_batch, _ = vec.reset(seed=None)
    noop = int(vec.envs[0].action_parser.noop())
    while len(rows) < count:
        for index in range(vec.num_envs):
            rows.append({key: value[index].copy() for key, value in obs_batch.items()})
        actions = np.full(vec.num_envs, noop, dtype=np.int64)
        for index in range(vec.num_envs):
            legal = np.flatnonzero(obs_batch["action_mask"][index])
            legal = legal[legal != noop]
            if legal.size and generator.random() >= RANDOM_LEGAL_NOOP_PROB:
                actions[index] = int(legal[generator.integers(legal.size)])
        obs_batch, _, _, _, infos = vec.step(actions)
        finals = infos.get("final_obs")
        if finals is not None:
            mask = np.asarray(infos["_final_obs"], dtype=bool)
            for index in np.flatnonzero(mask):
                rows.append({key: value.copy() for key, value in finals[index].items()})
    return rows[:count]


def _mask_disagreement_gate(env: Any, say: Callable[[str], None]) -> None:
    """Gate 5: the mask and the engine agree about every action, for both seats.

    Exhaustive, because a policy trained against a wrong mask is worthless and because the
    check is already written in RoyaleGym and nobody runs it.
    """
    from royalegym.action import mask_disagreements
    from royalegym.protocol import TEAMS

    parser = env.action_parser
    if not hasattr(parser, "encode"):
        raise PreflightError(
            f"{type(parser).__name__} is not a grid action parser, and the harness's pointer "
            "head and mask planes are written against one"
        )
    state = env.engine.state()
    total = 0
    for team in TEAMS:
        disagreements = mask_disagreements(env.engine, parser, state, int(team))
        total += len(disagreements)
        if disagreements:
            shown = ", ".join(
                f"action {a} mask {m} engine {s}" for a, m, s in disagreements[:8]
            )
            raise PreflightError(
                f"the action mask and the engine disagree on {len(disagreements)} of "
                f"{int(parser.space.n) - 1} actions for team {int(team)}: {shown}"
            )
    say(f"mask gate     {2 * (int(parser.space.n) - 1)} actions checked, {total} disagreements")


def _noop_gate(
    env: Any,
    config: RunConfig,
    sample: Sequence[dict[str, np.ndarray]],
    spec: EnvSpec,
    count: int,
    say: Callable[[str], None],
) -> None:
    """Gate 6: the no-op is legal in every state, including one the engine has ended.

    It is the precondition the whole masking scheme rests on: a state with no legal action is a
    distribution over nothing, and what it produces is a NaN in the first backward pass.
    """
    noop = int(env.action_parser.noop())
    checked = 0
    for row in sample[:count]:
        if not row["action_mask"][noop]:
            raise PreflightError(
                "a sampled state has no legal no-op; every masked distribution in the harness "
                "assumes there is always one action to take"
            )
        checked += 1
    finished, parser = _drive_to_game_over(config, spec)
    for team in (0, 1):
        mask = parser.action_mask(finished, team)
        if not mask[noop]:
            raise PreflightError(
                "the no-op is not legal in a finished battle, and a finished battle is the one "
                "state every episode passes through"
            )
    say(f"no-op gate    {checked} sampled states and a finished battle, all legal")


def _drive_to_game_over(config: RunConfig, spec: EnvSpec) -> tuple[Any, Any]:
    """A state with ``game_over`` set, reached the way a battle reaches one.

    It runs the run's own environment with its step limit raised past the length of a match,
    because a truncation ends an episode before the engine does and this gate is about the
    state the engine ends one in. The bound is the match's own length -- regulation, overtime
    and a decision of slack, read off the environment rather than written down -- so a rules
    change moves it.
    """
    import msgspec

    limit = (spec.regular_ticks + spec.overtime_ticks) // spec.decision_ticks + 2
    factory = msgspec.structs.replace(
        config.env,
        truncation=[
            ComponentSpec(
                "royalegym.done_condition.StepLimitCondition", {"max_steps": limit + 1}
            )
        ],
    )
    vec = factory.build_vec(1, tuple(config.extra_component_modules), viser=None)
    try:
        env = vec.envs[0]
        env.reset()
        noop = {"blue": env.action_parser.noop(), "red": env.action_parser.noop()}
        for _ in range(limit):
            state = env.engine.state()
            if state.game_over:
                return state, env.action_parser
            env.step(noop)
        state = env.engine.state()
        if not state.game_over:
            raise PreflightError(
                f"a battle of no-ops did not end within {limit} decisions, and the mask gate "
                "needs a finished battle to check the no-op on"
            )
        return state, env.action_parser
    finally:
        vec.close()


def _action_layout_gate(
    env: Any, spec: EnvSpec, sample: Sequence[dict[str, np.ndarray]], say: Callable[[str], None]
) -> None:
    """Gate 7: the head's arithmetic and the parser's encoding are the same index.

    Exhaustive over every non-no-op action. It is the most load-bearing identity in the
    harness: the pointer head writes a logit per (hand slot, tile) and the environment reads an
    integer, and a transposition between them is a policy that plays a different card in a
    different place from the one it learned to.
    """
    parser = env.action_parser
    hand, tiles_y, tiles_x = spec.hand_size, spec.tiles[0], spec.tiles[1]
    slots, ys, xs = np.meshgrid(
        np.arange(hand), np.arange(tiles_y), np.arange(tiles_x), indexing="ij"
    )
    flat = np.ravel_multi_index((slots, ys, xs), (hand, tiles_y, tiles_x)).reshape(-1) + 1
    encoded = np.array(
        [
            parser.encode(int(s), int(x), int(y))
            for s, y, x in zip(slots.reshape(-1), ys.reshape(-1), xs.reshape(-1), strict=True)
        ]
    )
    wrong = np.flatnonzero(flat != encoded)
    if wrong.size:
        first = int(wrong[0])
        raise PreflightError(
            f"the action layout disagrees with the parser on {wrong.size} of {flat.size} "
            f"actions: a one-hot at index {flat[first]} encodes to {encoded[first]}"
        )
    if int(parser.space.n) != flat.size + 1:
        raise PreflightError(
            f"the action space is {int(parser.space.n)} and the grid is {flat.size} + the "
            "no-op"
        )
    planes = 0
    for row in sample:
        if "mask_planes" not in row:
            break
        expected = row["action_mask"][1:].reshape(hand, tiles_y, tiles_x)
        if not np.array_equal(row["mask_planes"], expected):
            raise PreflightError(
                "mask_planes is not action_mask[1:] reshaped; the codec stores the mask once "
                "and the learner reshapes it at unpack, so the two must be the same bits"
            )
        planes += 1
    say(
        f"action layout {flat.size} actions exhaustive, mask planes checked on {planes} states"
    )


def _statics(
    codec: ObsCodec, sample: Sequence[dict[str, np.ndarray]], say: Callable[[str], None]
) -> np.ndarray:
    """The planes the layout declares static, held once instead of once per transition.

    They are asserted equal between the two seats, because the observation is in the acting
    player's own frame and the arena is symmetric under the seat rotation. If they are ever
    not, the codec has to hold one pair per seat, and this is where that is found out.
    """
    blue = np.asarray(codec.static_planes(sample[0]))
    red = np.asarray(codec.static_planes(sample[1]))
    if blue.shape != red.shape or not np.array_equal(blue, red):
        say(
            "warning       the static planes differ between the two seats; the codec holds one "
            "pair per seat"
        )
    say(f"static planes {blue.shape[0] if blue.ndim else 0} held once, not per row")
    return blue


# ---------------------------------------------------------------------------
# What preflight prints
# ---------------------------------------------------------------------------


def _table_lines(spec: EnvSpec, table: CodecTable, row_bytes: int) -> list[str]:
    lines = ["codec table   plane                          storage      divisor"]
    for name, storage, divisor in table.plane:
        lines.append(f"              {name:<30} {storage:<12} {divisor:g}")
    lines.append(
        f"              vector {spec.vector_size} wide as {table.vector}, mask "
        f"{spec.n_actions} wide as {table.mask}"
    )
    lines.append(f"              {row_bytes} B a row, {table.digest()[:12]}")
    return lines


def ram_ledger(config: RunConfig, geo: Geometry, row_bytes: int) -> dict[str, float]:
    """What this run projects to hold resident, in megabytes.

    The buffer is arithmetic on the codec's own row; the rest are the measured figures of
    ``docs/harness-spec.md`` section 2.4. Overlap costs a second rectangle, which is why it is
    off by default on a machine with eight gigabytes.
    """
    rectangles = 2 if config.rollout.overlap else 1
    rows = geo.cycles + config.obs.frame_stack
    buffer_mb = rectangles * rows * geo.n_slots * row_bytes / 1e6
    workers_mb = config.rollout.workers * (
        WORKER_ENGINE_MB_PER_BATTLE * config.rollout.games_per_worker + WORKER_OVERHEAD_MB
    )
    total = buffer_mb + PARENT_MB + workers_mb + INTERPRETER_MB
    return {
        "buffer_mb": buffer_mb,
        "parent_mb": PARENT_MB,
        "workers_mb": workers_mb,
        "interpreters_mb": INTERPRETER_MB,
        "total_mb": total,
    }


def _ram_lines(ram: dict[str, float]) -> list[str]:
    return [
        f"memory        experience buffer            {ram['buffer_mb']:8.0f} MB",
        f"              parent, torch and CUDA       {ram['parent_mb']:8.0f} MB",
        f"              workers                      {ram['workers_mb']:8.0f} MB",
        f"              interpreters                 {ram['interpreters_mb']:8.0f} MB",
        f"              projected total              {ram['total_mb']:8.0f} MB",
    ]


def _credit_horizon_s(config: RunConfig, spec: EnvSpec) -> float:
    """``1 / (1 - gamma * lambda)`` decisions, in seconds of game time.

    It is the number that says what the agent is being asked to plan over, and it is printed
    because the discount and the decision granularity are set in different places and their
    product is what matters.
    """
    schedule = config.advantage.gamma
    gamma = float(getattr(schedule, "start", getattr(schedule, "value", 0.0)) or 0.0)
    span = 1.0 / max(1e-9, 1.0 - gamma * config.advantage.gae_lambda)
    return span * spec.decision_ms / 1000.0


def _viser_line(config: RunConfig) -> str:
    """Which vec env carries the viewer's state stream, if any.

    Exactly one may: the publisher binds one fixed UDP port, so a second env constructing one
    is an ``OSError`` rather than a second view.
    """
    import os

    if not os.environ.get("ROYALEVISER"):
        return "viewer        off; set ROYALEVISER=host:port to watch a battle"
    source = "worker 0 shard 0 battle 0" if config.rollout.source == "process" else "battle 0"
    return f"viewer        on, published by {source}"
