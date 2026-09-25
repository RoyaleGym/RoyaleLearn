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

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..api.buffer import MIN_TABLE_STATES, CodecTable
from ..config import Geometry, RunConfig, geometry
from ..errors import PreflightError
from ..identity import EngineBuild, engine_build
from ..obs_layout import hand_fields
from ..seeding import PREFLIGHT_ENV, PREFLIGHT_SAMPLE, derive_generator, derive_int, stream_path
from .envspec import ComponentSpec, EnvFactorySpec, build_env, read_env_spec
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
    free_mb = available_memory_mb()
    if free_mb is not None:
        say(f"              free right now               {free_mb:8.0f} MB")
        if ram["total_mb"] > free_mb:
            # A warning and not a refusal. `ram_budget_mb` is a statement about the machine
            # and is worth refusing over; this is a statement about the machine at this
            # moment, and whatever else is using it may well stop before the run needs the
            # memory. Refusing here would mean a run that could have finished does not start
            # because something unrelated was busy for a minute, which is the more expensive
            # mistake. What starting over the line costs is a worker killed mid-round, which
            # the run survives: its slots report nothing, the round continues without them,
            # and health/worker_restarts counts it.
            say(
                f"              WARNING: the projection is {ram['total_mb'] - free_mb:.0f} MB "
                f"over what is free."
            )
            say(
                "              Something else on this machine is using it. A worker killed "
                "for memory costs its seats for a round and shows up in "
                "health/worker_restarts; if that number climbs, this is why."
            )

    say(
        f"geometry      {geo.workers} workers x {geo.games_per_worker} battles = "
        f"{geo.n_battles} battles, {geo.n_slots} slots, {geo.learner_rows} of them learner rows"
    )
    say(
        f"iteration     {geo.cycles} cycles for {geo.timesteps_per_iteration} timesteps; "
        f"credit horizon {_credit_horizon_s(config, spec):.1f} s"
    )
    _ratio_precision_gate(config, spec, say)
    say(_ladder_cost_line(config, geo))
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


#: EXPLICIT mantissa bits, which is what sets the spacing between two representable values:
#: ulp(x) = 2**floor(log2 x) * 2**-bits. Named here rather than read from torch, so this can be
#: asked of a config in a process with no torch installed.
MANTISSA_BITS: dict[str, int] = {"bfloat16": 7, "float16": 10, "float32": 23}


def ratio_precision(*, noop_bias: float, n_actions: int, dtype_name: str) -> float:
    """How far from one the importance ratio can sit from ARITHMETIC ALONE, before a run starts.

    PPO's update recomputes each stored action's log-probability and asserts the ratio is one,
    because nothing has changed since the rollout. What makes that assertion fragile is not the
    guard but the shape of a softmax:

        log p_i = z_i - logsumexp(z)      so      d(log p_i) / d(z_noop) = -p_noop

    An error in the LARGEST logit is multiplied into every other action's log-probability by the
    probability that logit holds. So the deviation is ``p_max`` times the spacing between two
    representable logits at that magnitude -- and both factors are knowable from the config and
    the action space, with nothing running.

    IT IS AN UPPER BOUND AND ITS ACCURACY IS UNEVEN, which is worth knowing before trusting a
    borderline number. Against the two measurements of 2026-09-23, on 451 actions at
    ``noop_bias`` 8.0:

        float32     predicts 8.3e-7    measured 9.5e-7 over eight iterations     within 15%
        bfloat16    predicts 0.054     measured 0.0261 when a run died at it     2x conservative

    The bfloat16 side is high because the observed figure is the worst of the actions a minibatch
    happened to sample, while this is the worst the arithmetic allows. Erring high is the right
    direction for a gate -- it refuses slightly early rather than slightly late -- and the two
    cases it has to tell apart are five orders of magnitude apart, so a factor of two changes
    nothing about the decision.

    At ``noop_bias`` 0 the no-op holds 1/451 of the mass and the same arithmetic is invisible,
    which is why 383 iterations across three runs never came near the guard.

    ``n_actions`` is the WHOLE action space and the mask makes the set smaller at runtime, so the
    real ``p_max`` is higher than this assumes -- 0.87 over the ~450 legal actions of that run
    against 0.56 over its 2,305-wide space. That pushes the estimate down where the ulp term
    pushes it up, and the two partly cancel. The space is what a config knows before an
    environment exists, which is the point of asking here.
    """
    import math

    others = max(1, n_actions - 1)
    # p_max at initialisation: the biased action against a field of roughly equal ones. The rest
    # of the head is small next to a bias of several nats, so this is the bias alone.
    p_max = math.exp(noop_bias) / (math.exp(noop_bias) + others)
    return precision_bound(p_max=p_max, magnitude=abs(noop_bias), dtype_name=dtype_name)


def precision_bound(*, p_max: float, magnitude: float, dtype_name: str) -> float:
    """``p_max`` times the spacing between representable logits at ``magnitude``.

    The arithmetic of ``ratio_precision`` with its two inputs given rather than derived from
    ``noop_bias``: an actor loaded from an artifact (section 19.4) has a p_max and logits of its
    own, and those are measured on its probe rows instead of assumed.
    """
    import math

    bits = MANTISSA_BITS.get(dtype_name, MANTISSA_BITS["float32"])
    # A magnitude below one still has a spacing, so the floor keeps this a statement about the
    # head rather than about zero.
    magnitude = max(float(magnitude), 1.0)
    spacing = 2.0 ** (math.floor(math.log2(magnitude))) * 2.0 ** -bits
    return float(p_max) * spacing


def precision_name(config: RunConfig) -> str:
    """The update's name for the run's autocast precision, the key of ``ppo.ratio_atol``."""
    from ..learn.ppo import PRECISION_NAMES

    name = str(config.net.autocast_dtype)
    for dtype, spelling in PRECISION_NAMES.items():  # normalise torch's spelling to the config's
        if spelling == name or str(dtype).endswith(name):
            return spelling
    return name


def judge_ratio_precision(
    predicted: float, atol: float, *, trouble: str, say: Callable[[str], None]
) -> None:
    """Refuse above the tolerance, warn within a factor of four of it, else say nothing more."""
    if predicted <= atol / 4.0:
        return
    if predicted > atol:
        raise PreflightError(
            "this run would fail its own importance-ratio assertion within a few iterations: "
            + trouble
        )
    say(f"              WARNING: within a factor of four of the guard. {trouble}")


def _ratio_precision_gate(config: RunConfig, spec: EnvSpec, say: Callable[[str], None]) -> None:
    """Refuse a run whose own arithmetic cannot carry its own importance-ratio guard.

    This costs nothing and it replaces five hours of discovery with a printed number. The run it
    was written for died at iteration 6 with a deviation of 0.0261 against a tolerance of 0.02,
    and everything needed to predict that was in the config before it started.
    """
    name = precision_name(config)
    from ..extensions import active_extensions

    starters = [
        a.name for a in active_extensions(config) if a.extension.sets_starting_weights(a.section)
    ]
    if starters:
        # The estimate below is from noop_bias alone, which is what a SEEDED actor's p_max is.
        # A loaded actor's is its own, and the section that loads it measures it on the loaded
        # weights, fresh or resumed.
        say(f"ratio guard   {name}: measured by {', '.join(starters)} on the loaded actor")
        return
    predicted = ratio_precision(
        noop_bias=config.net.noop_bias, n_actions=spec.n_actions, dtype_name=name
    )
    atol = float(config.ppo.ratio_atol.get(name, 1e-4))
    say(
        f"ratio guard   {name} at noop_bias {config.net.noop_bias:g} over {spec.n_actions} "
        f"actions predicts a deviation of {predicted:.2e} against ppo.ratio_atol {atol:g}"
    )
    if predicted <= atol / 4.0:
        return
    unbiased = ratio_precision(noop_bias=0.0, n_actions=spec.n_actions, dtype_name=name)
    amplification = predicted / max(1e-30, unbiased)
    trouble = (
        f"ppo.ratio_atol[{name!r}] is {atol:g} and this run's arithmetic predicts a deviation of "
        f"{predicted:.2e}. log p = z - logsumexp(z), so an error in the largest logit is "
        f"multiplied into every other action's log-probability by the probability that logit "
        f"holds: at noop_bias {config.net.noop_bias:g} over {spec.n_actions} actions that is "
        f"{amplification:.0f}x what an unbiased head would see. Set net.autocast_dtype to "
        f"float32, measured at 9.5e-7 on this action space, or lower net.noop_bias"
    )
    judge_ratio_precision(predicted, atol, trouble=trouble, say=say)


def _ladder_cost_line(config: RunConfig, geo: Geometry) -> str:
    """What the ladder will cost this run, before it costs it.

    The candidate cadence is a BOUNDARY between two economics rather than a budget, and nothing
    said so: the first candidate into an empty pool is admitted unconditionally and costs
    milliseconds, and every one after it plays a full gate. At the shipped settings that is
    2,200 battles, and at a measured 8.02 s a battle it is about five hours EACH. A 500-iteration
    run at the old cadence fired six of them -- 29 hours of evaluation against 4 hours of
    training -- and the first anybody knew was a measurement on 2026-09-23, because no run of
    this project had ever reached a second candidate.

    So it is printed. A reader choosing a step budget can see which side of the boundary it
    falls on instead of discovering it at hour six.
    """
    from ..ladder.gate import EVAL_BATTLE_SECONDS, gate_battles
    from ..ladder.pool import SCRIPTED_IDS

    env_per_iteration = max(1, geo.cycles * geo.n_battles)
    every = config.ladder.candidate_every_env_steps
    if every <= 0:
        return "ladder        no candidates: ladder.candidate_every_env_steps is off"
    battles = gate_battles(config.ladder.gate, anchors=len(SCRIPTED_IDS))
    hours = battles * EVAL_BATTLE_SECONDS / 3600.0
    probe_every = config.ladder.probe_every_iterations
    probe = (
        f"; a probe every {probe_every} iterations plays "
        f"{config.ladder.probe_games * len(config.ladder.probe_opponents)} battles"
        if probe_every > 0
        else "; no probes"
    )
    return (
        f"ladder        a candidate every {every // env_per_iteration} iterations "
        f"({every} env steps). The FIRST into an empty pool is free; every one after plays "
        f"{battles} battles, about {hours:.1f} h at {EVAL_BATTLE_SECONDS:.1f} s a battle{probe}"
    )


def _construct(factory: EnvFactorySpec, extra_modules: tuple[str, ...]) -> Any:
    """Gate 1: one engine, constructed. A stale build dies here and not at cycle 0."""
    return build_env(factory, extra_modules)


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
    """Gate 5: the mask and the engine agree about every action, for both seats, AT ONE STATE.

    Exhaustive over ACTIONS and a single sample over STATES, and the difference matters enough
    to be in the first line rather than discovered. A policy trained against a wrong mask is
    worthless, and the check is already written in RoyaleGym where nobody runs it.

    The one state is the opening one -- ``env.engine.state()`` on a freshly built environment --
    which is the right sample for the disagreement most likely to exist, because rules that
    gate the START of a battle apply there and nowhere else. RoyaleGym found one on 2026-09-23:
    the mask offered all four cards while the engine refused them all with ``TOO_EARLY``, for
    the length of a deploy lockout, so a seat was penalised for obeying its own mask. This gate
    stands exactly where that bites and will refuse a run that reintroduces it.

    WHAT IT CANNOT SEE is a disagreement that only appears LATER: a rule that opens when a tower
    falls, a card whose legality depends on what is already on the board, a zone that changes at
    overtime. Sampling several states through a played-out battle would cover those, and would
    need the engine, which is why it is named here rather than quietly absent.
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


def available_memory_mb() -> float | None:
    """Memory a new process could actually obtain now, or None where that cannot be asked.

    Not the machine's size: what is free on it at this moment. A training run sized against
    the total is sized against a machine nobody else is using, and the way it fails when that
    is untrue is a worker being killed in the middle of a round rather than a refusal at the
    start.
    """
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPhys / (1024 * 1024)
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024
    except OSError:
        return None
    return None


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
