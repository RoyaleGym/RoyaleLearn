"""One directory, written on any halt, holding everything somebody would ask for.

A learner that is eighty per cent right produces a bot that loses for reasons nobody can
attribute. This is the concrete answer to that: when a run stops, it leaves behind the last
fifty metric rows, the alarms that were firing, the resolved configuration, the run identity,
the last episodes, the worst importance ratios with the cell each came from, a replay of one
offending episode, and the state digest -- in one folder, attachable to an issue.

The replay is what makes the folder more than a log. An episode is re-simulated from its own
shard seed and reset ordinal into a ``royalegym.replay.Trace``, and the trace is then verified
against a fresh engine, so what the bundle carries is a battle somebody else can run rather
than a description of one.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from ..api.metrics import AlarmResult, MetricRow
from ..api.rollout import EpisodeRecord

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..config import RunConfig
    from ..identity import RunIdentity

__all__ = [
    "BUNDLES_DIR",
    "KEEP_EPISODES",
    "KEEP_ROWS",
    "RatioOutlier",
    "replay_episode",
    "write_bundle",
]

BUNDLES_DIR = "bundles"
#: How much of each stream the bundle carries. Fifty iterations is enough to see a metric turn
#: and small enough that the folder stays attachable to an issue.
KEEP_ROWS = 50
KEEP_EPISODES = 20
KEEP_OUTLIERS = 10


class RatioOutlier(msgspec.Struct, frozen=True):
    """One sample whose importance ratio was furthest from one, and where it came from.

    The cell is what makes it actionable: a cluster of outliers in one slot is a worker, a
    cluster in one cycle is an iteration boundary, and a spread over both is the codec.
    """

    ratio: float
    deviation: float
    cycle: int
    slot: int
    ordinal: int = -1


def write_bundle(
    run_dir: str | Path,
    iteration: int,
    *,
    identity: RunIdentity | None = None,
    config_json: str = "",
    rows: Sequence[MetricRow] = (),
    alarms: Sequence[AlarmResult] = (),
    episodes: Sequence[EpisodeRecord] = (),
    outliers: Sequence[RatioOutlier] = (),
    state_digest: str = "",
    trace: Any = None,
    divergences: Sequence[str] | None = None,
    note: str = "",
) -> Path:
    """Write ``<run>/bundles/<iteration>/`` and return it.

    Nothing here raises on a piece that is missing. The bundle is written from a crash handler,
    and a bundle that failed to write because one of its ten parts was unavailable would cost
    the other nine at exactly the moment they are wanted.
    """
    folder = Path(run_dir) / BUNDLES_DIR / f"{iteration:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    encoder = msgspec.json.Encoder()

    _lines(folder / "metrics.jsonl", encoder, list(rows)[-KEEP_ROWS:])
    _lines(folder / "alarms.jsonl", encoder, alarms)
    _lines(folder / "episodes.jsonl", encoder, list(episodes)[-KEEP_EPISODES:])
    if outliers:
        (folder / "ratio_outliers.json").write_bytes(
            encoder.encode(list(outliers)[:KEEP_OUTLIERS])
        )
    if config_json:
        (folder / "config.json").write_text(config_json, encoding="utf-8")
    if identity is not None:
        (folder / "identity.json").write_bytes(encoder.encode(identity))
    if trace is not None:
        from royalegym.replay import save_trace

        save_trace(trace, folder / "episode.msgpack")
    (folder / "bundle.json").write_bytes(
        encoder.encode(
            {
                "iteration": iteration,
                "state_digest": state_digest,
                "written_unix_ns": time.time_ns(),
                "metric_rows": min(len(rows), KEEP_ROWS),
                "episodes": min(len(episodes), KEEP_EPISODES),
                "alarms": [alarm.name for alarm in alarms],
                "trace_divergences": list(divergences) if divergences is not None else None,
                "note": note,
            }
        )
    )
    return folder


def _lines(path: Path, encoder: msgspec.json.Encoder, records: Iterable[Any]) -> None:
    records = list(records)
    if not records:
        return
    with path.open("wb") as handle:
        for record in records:
            handle.write(encoder.encode(dict(record) if isinstance(record, dict) else record))
            handle.write(b"\n")


def replay_episode(
    config: RunConfig,
    record: EpisodeRecord,
    *,
    seed: int,
    policy: Any = None,
    frame_every_tick: bool = True,
    max_steps: int | None = None,
) -> tuple[Any, list[str]]:
    """Re-simulate one episode from its shard seed and reset ordinal, and verify the trace.

    ``seed`` is the seed of the shard the episode belongs to, which the run's own plan derives
    from ``EpisodeRecord.episode_seed_path``; the vec env is advanced past the ``ordinal``
    episodes that came before it, so the battle starts in the position it started in. The
    actions come from ``policy`` -- the run's actor, a scripted opponent, or the no-op when
    nothing is given.

    The trace is then re-simulated on a fresh engine and the divergences returned. An empty
    list is the whole promise the engine makes: the battle in the folder is the battle that
    happened.
    """
    from royalegym.replay import ReplayRecorder, verify_trace

    traces: list[Any] = []
    recorder = ReplayRecorder(frame_every_tick=frame_every_tick, on_finish=traces.append)
    factory = config.env.factory(tuple(config.extra_component_modules))
    env = factory()
    env.recorder = recorder
    try:
        obs, _info = env.reset(seed=seed)
        # The reset ordinal counts the episodes this battle played before the one wanted, and
        # an env resets itself on the step that ends one, so replaying to it is replaying
        # through them -- which is also what makes the start state the one the run saw.
        played = 0
        steps = 0
        limit = max_steps if max_steps is not None else _step_limit(config)
        while played <= record.ordinal and steps < limit * (record.ordinal + 1):
            actions = _actions(env, obs, policy)
            obs, _rewards, terminated, truncated, _infos = env.step(actions)
            steps += 1
            if terminated["blue"] or truncated["blue"]:
                played += 1
                if played <= record.ordinal:
                    obs, _info = env.reset()
        trace = traces[-1] if traces else recorder.trace
    finally:
        env.close()
    if trace is None:
        return None, ["the episode produced no trace"]
    engine = config.env.engine.build(tuple(config.extra_component_modules))
    return trace, list(verify_trace(trace, engine))


def _actions(env: Any, obs: dict[str, Any], policy: Any) -> dict[str, int]:
    """One decision for both seats: the policy's, or the no-op."""
    noop = int(env.action_parser.noop())
    if policy is None:
        return {"blue": noop, "red": noop}
    return {agent: int(policy(agent, obs[agent])) for agent in ("blue", "red")}


def _step_limit(config: RunConfig) -> int:
    """How many decisions one episode may take before the replay gives up.

    Read off the truncation the config declares where it declares one, so a replay of a run
    whose episodes are capped is capped the same way, and generous otherwise: a replay that
    stopped early would produce a trace of part of a battle and call it the battle.
    """
    for spec in config.env.truncation:
        steps = spec.kwargs.get("max_steps")
        if steps is not None:
            return int(steps) + 1
    return 10_000
