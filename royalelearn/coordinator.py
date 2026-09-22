"""The loop: the only place in the package where the phases of an iteration are ordered.

Everything else in this repository is a piece that can be replaced -- a codec, an estimator, a
matchmaker, a sink. This file is the sentence they are read in. It holds no policy of its own
beyond that ordering and the invariants that go with it, and it is the only module that knows
that a critic pass comes before GAE, that a gate belongs at an env-step cadence rather than an
iteration one, and that a checkpoint is written after the alarms have had their look at the row.

The invariants are the point of having one such place. Each of them is checked once per
iteration over the whole rectangle and raises an exception that names the cell it failed on,
because every failure they catch is otherwise silent for hours: an assignment that reached the
middle of a trajectory, an action that is illegal under the mask stored beside it, a cycle that
never arrived, a mixture that is not the one the config describes.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import signal
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np

from .api.checkpoint import Checkpointable, Manifest
from .api.ladder import GateDecision, RatingTable
from .api.metrics import AlarmResult, MetricValue
from .api.rollout import (
    GROUP_LEARNER,
    EnvSpec,
    EpisodeRecord,
    RolloutRound,
    SlotPlan,
    Step,
)
from .api.schedule import ScheduleState
from .checkpoint import CHECKPOINT_FORMAT_VERSION, DirCheckpointStore, check_resume
from .config import RunConfig, config_hash, dump_config, geometry, validate
from .errors import RoyaleLearnError
from .identity import RunIdentity, compute_identity, describe_device
from .identity import run_id as run_id_of
from .ladder.evaluate import EvalRunner, SeedSet, eval_seed_set
from .ladder.eviction import HallOfFameEviction
from .ladder.gate import WilsonGate, floor_decision
from .ladder.matchmaker import MixMatchmaker
from .ladder.pool import LEARNER_ID, LadderPool
from .ladder.rating import BradleyTerryDavidsonRater, EloReadout
from .ladder.results import ResultLog, context_digest
from .ladder.snapshots import DiskSnapshotStore, SnapshotSpec
from .metrics.alarms import AlarmSet
from .metrics.bundle import RatioOutlier, replay_episode, write_bundle
from .metrics.records import (
    IterationMetrics,
    episode_fields,
    ladder_fields,
    schedule_fields,
    update_fields,
)
from .metrics.sinks import build_sinks
from .obs_layout import HAND_CARD_ONEHOT, field_slice, hand_fields
from .rollout.inline import (
    InlineRolloutSource,
    assignments_constant_within_episodes,
    build_codec,
    check_round,
)
from .rollout.layout import buffer_segment_name
from .rollout.preflight import DEFAULT_CODEC, PreflightReport, run_preflight
from .seeding import TORCH_CUDA, TORCH_GLOBAL, derive_int, stream_path

if TYPE_CHECKING:  # pragma: no cover - annotations only
    import torch

    from .api.metrics import MetricsSink
    from .api.rollout import RolloutSource

__all__ = [
    "AssignmentInsideEpisode",
    "CheckpointComponents",
    "DeployRefused",
    "EnvBattlePlayer",
    "EpisodesUnpaired",
    "IterationInvariant",
    "LearningCoordinator",
    "MixtureDrifted",
    "NonFiniteLogProb",
    "PolicyProbe",
    "RoundsMissing",
    "StoredActionIllegal",
    "TrainableRowsShort",
    "run_directory",
]

#: The share of ``ppo.timesteps_per_iteration`` an iteration must actually collect.
MIN_TRAINABLE_FRACTION = 0.98
#: How far ``throughput/discarded_rows_frac`` may sit from what the mixture implies.
MIXTURE_TOLERANCE = 0.05
#: Cells the policy probe unpacks per iteration to measure the things only an observation
#: carries. Bounded, because it is a measurement and not a phase of the update.
PROBE_ROWS = 4096
#: Where the interactive control letter is read from, inside the run directory.
CONTROL_FILE = "control"


# ---------------------------------------------------------------------------
# What an iteration has to be true of
# ---------------------------------------------------------------------------


class IterationInvariant(RoyaleLearnError):
    """An iteration was not the rectangle it is supposed to be.

    Every subclass carries the cell it failed on, because the difference between "the mixture
    drifted" and "slot 118 changed hands at cycle 40" is the difference between a thing to
    think about and a thing to fix.
    """

    def __init__(self, message: str, *, cycle: int = -1, slot: int = -1) -> None:
        self.cycle = int(cycle)
        self.slot = int(slot)
        where = "" if cycle < 0 and slot < 0 else f" (cycle {cycle}, slot {slot})"
        super().__init__(f"{message}{where}")


class RoundsMissing(IterationInvariant):
    """The iteration did not read one round per shard per cycle."""


class TrainableRowsShort(IterationInvariant):
    """The rectangle collected fewer trainable transitions than the config asked for."""


class NonFiniteLogProb(IterationInvariant):
    """A stored log-probability is not finite, so its action was scored under a mask that
    forbade it."""


class StoredActionIllegal(IterationInvariant):
    """A stored action is illegal under the mask stored beside it."""


class DeployRefused(IterationInvariant):
    """The engine refused a command the mask allowed; under a correct mask that cannot
    happen."""


class EpisodesUnpaired(IterationInvariant):
    """A battle reported an episode from one seat and not the other."""


class MixtureDrifted(IterationInvariant):
    """The share of collected rows that were discarded is not the share the mixture implies."""


class AssignmentInsideEpisode(IterationInvariant):
    """A seat changed hands in the middle of an episode."""


# ---------------------------------------------------------------------------
# Checkpoint components the coordinator owns
# ---------------------------------------------------------------------------


class _ModelComponent:
    """The actor and the critic, as safetensors, beside the architecture that shaped them.

    Split by whether a tensor belongs to the actor rather than by two state dicts, so that the
    shared-trunk architecture writes the same two files as the separate-trunk one and a reader
    does not have to know which it has.
    """

    FORMAT_VERSION = 1
    ACTOR_FILE = "actor.safetensors"
    CRITIC_FILE = "critic.safetensors"
    ARCH_FILE = "arch.json"

    def __init__(self, model: Any, *, arch: Any, arch_digest: str, device: Any) -> None:
        self.model = model
        self.arch = arch
        self.arch_digest = arch_digest
        self.device = device

    def _split(self) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self.model.state_dict()
        actor = {name: t for name, t in state.items() if name.startswith("actor.")}
        critic = {name: t for name, t in state.items() if not name.startswith("actor.")}
        return actor, critic

    def save_checkpoint(self, folder: Path) -> None:
        from safetensors.torch import save_file

        folder.mkdir(parents=True, exist_ok=True)
        actor, critic = self._split()
        for name, tensors in ((self.ACTOR_FILE, actor), (self.CRITIC_FILE, critic)):
            save_file(
                {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
                str(folder / name),
            )
        (folder / self.ARCH_FILE).write_bytes(
            msgspec.json.encode(
                {
                    "format_version": self.FORMAT_VERSION,
                    "arch_digest": self.arch_digest,
                    "arch": self.arch,
                }
            )
        )

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        from safetensors.torch import load_file

        from .errors import CheckpointFormatError

        arch_path = folder / self.ARCH_FILE
        if arch_path.exists():
            stored = msgspec.json.decode(arch_path.read_bytes())
            if stored.get("arch_digest") and stored["arch_digest"] != self.arch_digest:
                raise CheckpointFormatError(
                    f"{arch_path} holds weights built from arch_digest "
                    f"{stored['arch_digest']} and this run's architecture hashes to "
                    f"{self.arch_digest}"
                )
        state: dict[str, Any] = {}
        for name in (self.ACTOR_FILE, self.CRITIC_FILE):
            path = folder / name
            if not path.exists():
                if strict:
                    raise CheckpointFormatError(f"the weights are not at {path}")
                print(f"no weights at {path}; keeping the ones this run started with")
                continue
            state.update(load_file(str(path), device=str(self.device)))
        if state:
            self.model.load_state_dict(state)


class _ScheduleComponent:
    """The scheduled quantities: the backoff's own state, and the values of the last iteration.

    The schedules themselves are pure functions of the cumulative env step, which the manifest
    carries, so what has to be written here is the one thing that is not -- the learning-rate
    backoff -- plus the values the last iteration actually ran at, which is what makes "what was
    gamma at iteration k" a lookup rather than a recomputation from a config that may since have
    been edited.
    """

    FORMAT_VERSION = 1
    STATE_FILE = "state.json"

    def __init__(self, schedules: Any) -> None:
        self.schedules = schedules
        self.last: ScheduleState | None = None

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / self.STATE_FILE).write_bytes(
            msgspec.json.encode(
                {
                    "format_version": self.FORMAT_VERSION,
                    "backoff": self.schedules.backoff.state(),
                    "last": self.last,
                }
            )
        )

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        from .errors import CheckpointFormatError
        from .learn.schedules import BackoffState

        path = folder / self.STATE_FILE
        if not path.exists():
            if strict:
                raise CheckpointFormatError(f"the schedule state is not at {path}")
            print(f"no schedule state at {path}; the rates start from the configured ones")
            return
        payload = msgspec.json.decode(path.read_bytes())
        self.schedules.backoff.load_state(msgspec.convert(payload["backoff"], BackoffState))
        stored = payload.get("last")
        self.last = msgspec.convert(stored, ScheduleState) if stored else None


class _RolloutComponent:
    """Where each shard's stream had got to: its seed path, its reset ordinal and its
    respawn generation.

    What this does NOT carry is a battle's position inside an episode. The worker protocol has
    no way to read one out, so a resumed run's battles start from the beginning of their own
    seeded sequence. The learner resumes exactly; the environment resumes deterministically but
    not from mid-episode, and ``docs/checkpoints.md`` is where that is stated rather than here.
    """

    FORMAT_VERSION = 1
    STATE_FILE = "state.json"

    def __init__(self, source: Any, planner: Any, matchmaker: Any) -> None:
        self.source = source
        self.planner = planner
        self.matchmaker = matchmaker

    def shard_streams(self) -> list[dict[str, Any]]:
        streams: list[dict[str, Any]] = []
        for worker in range(self.source.geometry.workers):
            generation = self.source.generation[worker]
            for shard in range(self.source.geometry.shards_per_worker):
                streams.append(
                    {
                        "worker": worker,
                        "shard": shard,
                        "generation": int(generation),
                        "seed_path": self.planner.env_seed_path(worker, shard, generation),
                        "restarts": int(self.source.restarts[worker]),
                    }
                )
        return streams

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / self.STATE_FILE).write_bytes(
            msgspec.json.encode(
                {
                    "format_version": self.FORMAT_VERSION,
                    "shards": self.shard_streams(),
                    "ordinals": {
                        str(battle): self.matchmaker.ordinal(battle)
                        for battle in range(self.source.geometry.n_battles)
                    },
                }
            )
        )

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        from .errors import CheckpointFormatError

        path = folder / self.STATE_FILE
        if not path.exists():
            if strict:
                raise CheckpointFormatError(f"the rollout state is not at {path}")
            print(f"no rollout state at {path}; every shard restarts at generation zero")
            return
        payload = msgspec.json.decode(path.read_bytes())
        for entry in payload["shards"]:
            worker = int(entry["worker"])
            self.source.generation[worker] = int(entry["generation"])
            self.source.restarts[worker] = int(entry.get("restarts", 0))
        # The ordinals are the half of this file that decides which battles get played: each
        # one addresses the stream its episode's seed comes from. Writing them and not reading
        # them back leaves every battle on episode zero, so a resumed run replays the opening
        # battles under weights that have moved on -- a divergence that shows in the
        # environment's numbers while every learner digest still matches.
        self.matchmaker.restore_ordinals(payload.get("ordinals", {}))


class _GroupComponent:
    """Several checkpointables in one folder, for the pieces that belong together.

    The ladder is three objects and one subject: the pool, the archive it indexes and the
    counters over both. They write distinct file names, so one folder holds them and a reader
    of ``ladder/`` sees the ladder rather than three subdirectories.
    """

    FORMAT_VERSION = 1

    def __init__(self, members: Sequence[Checkpointable]) -> None:
        self.members = list(members)

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for member in self.members:
            member.save_checkpoint(folder)

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        for member in self.members:
            member.load_checkpoint(folder, strict=strict)


CheckpointComponents = dict[str, Checkpointable]


# ---------------------------------------------------------------------------
# The measurements only an observation carries
# ---------------------------------------------------------------------------


class PolicyProbe:
    """What the policy did, measured on a bounded sample of the iteration's own cells.

    Three of the metric groups cannot be read off a scalar column: how many actions were legal,
    how much elixir was in the bar, and which card each play was. All three are in the stored
    observation, so they are read from it -- through the same gather the update uses, on an
    evenly spaced sample of the trainable cells rather than on all of them, because this is a
    measurement and not a phase of the update.

    The sample is a stride and not a draw. A random sample would need a stream of its own and
    would make a diagnostic move the run, and a stride over an ascending cell list is spread
    over every slot and every cycle by construction.
    """

    def __init__(self, spec: EnvSpec, *, rows: int = PROBE_ROWS) -> None:
        self.spec = spec
        self.rows = max(1, int(rows))
        self.hand = hand_fields(spec)
        self.elixir = field_slice(spec, "own_elixir")
        self.onehot = field_slice(spec, HAND_CARD_ONEHOT)
        self.tiles = spec.tiles[0] * spec.tiles[1]
        self.max_mana = _max_mana()
        self.card_plays: dict[int, int] = {}
        self.tile_plays: dict[int, int] = {}
        self.card_tile_plays: dict[tuple[int, int], int] = {}

    def measure(
        self, buffer: Any, gather: Any, actions: np.ndarray, trainable: np.ndarray
    ) -> dict[str, MetricValue]:
        """The ``policy/`` group and the two elixir fields, plus the legality check.

        Returns the fields; raises ``StoredActionIllegal`` where a sampled action is illegal
        under the mask stored with it, which is the same assertion the update makes over every
        cell during its debug iterations and this makes over a sample on every one.
        """
        import torch

        cells = np.argwhere(trainable)
        fields: dict[str, MetricValue] = {}
        self.card_plays, self.tile_plays, self.card_tile_plays = {}, {}, {}
        if cells.size == 0:
            return _empty_policy_fields()
        step = max(1, cells.shape[0] // self.rows)
        sampled = cells[::step][: self.rows]
        cycles, slots = sampled[:, 0].copy(), sampled[:, 1].copy()
        taken = actions[cycles, slots].astype(np.int64)

        legal_counts: list[np.ndarray] = []
        elixir: list[np.ndarray] = []
        onehots: list[np.ndarray] = []
        # The gather's staging ring is one round wide, so the probe reads in blocks of at
        # most that: it borrows the update's gather rather than allocating a second one.
        chunk = max(1, min(self.rows, buffer.n_slots))
        for start in range(0, cycles.size, chunk):
            block = slice(start, start + chunk)
            obs = gather.observations(cycles[block], slots[block])
            with torch.no_grad():
                mask = obs.mask
                legal_counts.append(mask.sum(dim=-1).to("cpu").numpy())
                chosen = mask.gather(
                    -1,
                    torch.from_numpy(taken[block]).to(mask.device).unsqueeze(-1),
                ).reshape(-1)
                wrong = np.flatnonzero(~chosen.to("cpu").numpy().astype(bool))
                if wrong.size:
                    first = start + int(wrong[0])
                    raise StoredActionIllegal(
                        f"{wrong.size} stored action(s) in this iteration's sample are illegal "
                        "under the mask stored with them",
                        cycle=int(cycles[first]),
                        slot=int(slots[first]),
                    )
                vector = obs.vector.to("cpu").numpy()
            elixir.append(vector[:, self.elixir].reshape(-1))
            onehots.append(vector[:, self.onehot])

        legal = np.concatenate(legal_counts).astype(np.float64)
        fields["policy/legal_actions_mean"] = float(legal.mean())
        for percentile, name in ((5, "p05"), (50, "p50"), (95, "p95")):
            fields[f"policy/legal_actions_{name}"] = float(np.percentile(legal, percentile))
        fields["policy/forced_noop_frac"] = float(np.mean(legal <= 1))

        bar = np.concatenate(elixir).astype(np.float64)
        fields["env/mean_elixir_at_decision"] = float(bar.mean() * self.max_mana)
        fields["env/frac_elixir_above_99"] = float(np.mean(bar >= 0.99))

        fields.update(self._plays(taken, np.concatenate(onehots, axis=0)))
        return fields

    def _plays(self, actions: np.ndarray, onehot: np.ndarray) -> dict[str, MetricValue]:
        """Where the cards went, and which cards they were.

        The action index says the hand slot and the tile; the card in that slot is read out of
        the observation's own one-hot, so the counters are per card rather than per hand
        position -- which is the only one of the two that means anything across a cycle.
        """
        fields: dict[str, MetricValue] = {}
        played = actions > 0
        fields["policy/noop_rate"] = float(np.mean(~played))
        if not played.any():
            fields["policy/tile_entropy"] = 0.0
            fields["policy/tile_top1_share"] = 0.0
            fields["policy/card_tile_top10_share"] = 0.0
            return fields
        index = actions[played] - 1
        slot = index // self.tiles
        tile = index % self.tiles
        block = self.hand.onehot_width
        rows = np.arange(onehot.shape[0])[played]
        cards = np.array(
            [
                int(np.argmax(onehot[row, s * block : (s + 1) * block]))
                for row, s in zip(rows, slot, strict=True)
            ],
            dtype=np.int64,
        )
        counts = np.bincount(tile, minlength=self.tiles).astype(np.float64)
        share = counts / counts.sum()
        positive = share[share > 0]
        fields["policy/tile_entropy"] = float(-(positive * np.log(positive)).sum())
        fields["policy/tile_top1_share"] = float(share.max())

        pairs: dict[tuple[int, int], int] = {}
        for card, position in zip(cards, tile, strict=True):
            pairs[(int(card), int(position))] = pairs.get((int(card), int(position)), 0) + 1
            self.card_plays[int(card)] = self.card_plays.get(int(card), 0) + 1
        self.card_tile_plays = pairs
        self.tile_plays = {int(t): int(c) for t, c in enumerate(counts) if c}
        top = sorted(pairs.values(), reverse=True)[:10]
        fields["policy/card_tile_top10_share"] = float(sum(top) / sum(pairs.values()))
        total = float(sum(self.card_plays.values()))
        for card, played_count in sorted(self.card_plays.items()):
            fields[f"policy/card_play_frac/{card}"] = played_count / total
        return fields

    def heatmap(self) -> dict[str, Any]:
        """The per-card play counts over the tile grid, as the artifact a sink is handed."""
        tiles_y, tiles_x = self.spec.tiles
        return {
            "tiles": [tiles_y, tiles_x],
            "counts": {
                f"{card}/{tile}": n for (card, tile), n in sorted(self.card_tile_plays.items())
            },
        }


def _empty_policy_fields() -> dict[str, MetricValue]:
    """What the probe reports for an iteration with no trainable cell in it.

    Zeros rather than absences: an iteration that collected nothing is a fact about the run and
    the alarms that read these keys are about a policy that has stopped playing, which is
    exactly what "nothing was collected" looks like from outside.
    """
    return {
        "policy/legal_actions_mean": 0.0,
        "policy/legal_actions_p05": 0.0,
        "policy/legal_actions_p50": 0.0,
        "policy/legal_actions_p95": 0.0,
        "policy/forced_noop_frac": 0.0,
        "policy/noop_rate": 0.0,
        "policy/tile_entropy": 0.0,
        "policy/tile_top1_share": 0.0,
        "policy/card_tile_top10_share": 0.0,
        "env/mean_elixir_at_decision": 0.0,
        "env/frac_elixir_above_99": 0.0,
    }


def _max_mana() -> float:
    """The full elixir bar, in elixir.

    The observation carries the bar as a fraction of it, and the metric is in elixir, so the
    number has to come from somewhere: it comes from the calibration the engine itself reads,
    never from a constant in this repository.
    """
    from royalegym.protocol import default_calibration

    return float(default_calibration().int("match.MAX_MANA"))


# ---------------------------------------------------------------------------
# Evaluation battles
# ---------------------------------------------------------------------------


class _RecordingEvalRunner(EvalRunner):
    """An evaluation runner that keeps its last comparison.

    ``paired_rho`` and the evaluation draw rate are properties of a comparison and the gate
    reports only its verdict, so the number is kept where it is produced instead of being
    reconstructed from the result log afterwards.
    """

    last: Any = None

    def compare(self, a: str, b: str, *, games: int, iteration: int = 0) -> Any:
        comparison = super().compare(a, b, games=games, iteration=iteration)
        self.last = comparison
        return comparison


class EnvBattlePlayer:
    """One evaluation battle, played end to end in this process.

    The separations of section 11.5 are what this is for: a full match under the real rules
    with no truncation, a seed from the frozen set, no transition recorded anywhere, and both
    actions drawn from a stream addressed by the comparison, the seed index and the side. It
    holds one environment and reuses it, because building one costs a catalogue parse and a
    gate plays hundreds of battles.
    """

    def __init__(
        self,
        spec: EnvSpec,
        factory_spec: Any,
        actors: Callable[[str], Any],
        *,
        master_seed: int,
        extra_modules: Sequence[str] = (),
        device: Any = "cpu",
        release_mode: str = "stochastic",
        max_decisions: int = 2000,
        viser: bool = False,
    ) -> None:
        self.spec = spec
        self.factory_spec = factory_spec
        self.actors = actors
        self.master_seed = int(master_seed)
        self.extra_modules = tuple(extra_modules)
        self.device = device
        self.release_mode = release_mode
        self.max_decisions = int(max_decisions)
        #: Whether this battle carries the viewer's state stream. One vec env in a process may:
        #: the publisher binds one fixed UDP port, so a second would be an OSError.
        self.viser = bool(viser)
        self.battles = 0
        self._env: Any = None
        self._vec: Any = None

    @property
    def env(self) -> Any:
        """The environment, built on first use and kept.

        With the viewer on it is the single battle of a one-game vec env, because the publisher
        belongs to the vec env and is handed to game zero; without it there is no reason to
        build the wrapper at all.
        """
        if self._env is None:
            if self.viser:
                self._vec = self.factory_spec.build_vec(1, self.extra_modules, viser="env")
                self._env = self._vec.envs[0]
            else:
                self._env = self.factory_spec.factory(self.extra_modules)()
        return self._env

    def close(self) -> None:
        if self._vec is not None:
            self._vec.close()
        elif self._env is not None:
            self._env.close()
        self._env = None
        self._vec = None

    def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
        """A's score in one battle: 1 for a win, 0.5 for a draw, 0 for a loss."""
        from .seeding import derive_generator

        env = self.env
        seats = {a_seat: a, 1 - a_seat: b}
        policies = {seat: self.actors(name) for seat, name in seats.items()}
        rng = derive_generator(self.master_seed, act_path)
        obs, _info = env.reset(seed=seed)
        agents = ("blue", "red")
        outcome = 0
        for _ in range(self.max_decisions):
            uniforms = rng.random(2, dtype=np.float32)
            actions = {
                agents[seat]: int(
                    policies[seat](obs[agents[seat]], float(uniforms[seat]), rng)
                )
                for seat in (0, 1)
            }
            obs, _reward, terminated, truncated, infos = env.step(actions)
            if terminated[agents[a_seat]] or truncated[agents[a_seat]]:
                outcome = int(infos[agents[a_seat]].get("outcome", 0) or 0)
                break
        self.battles += 1
        return 0.5 + 0.5 * float(np.sign(outcome))


def _scripted_policy(name: str) -> Callable[[dict[str, Any], float, Any], int]:
    """A scripted opponent behind the player's uniform signature."""
    from .rollout.scripted import build_opponent

    opponent = build_opponent(name)

    def act(obs: dict[str, Any], _uniform: float, rng: Any) -> int:
        return int(opponent.act(obs, obs["action_mask"], rng))

    return act


# ---------------------------------------------------------------------------
# Interactive control
# ---------------------------------------------------------------------------


class _Control:
    """``c`` checkpoint, ``q`` checkpoint and quit, ``p`` pause, read once per round.

    A sentinel file and a signal handler, so that a run needs neither a terminal attached nor a
    thread spinning on one. The file is read and removed, so a letter acts once; the handler
    turns Ctrl-C into ``q``, which is a checkpoint rather than a lost afternoon.
    """

    def __init__(self, run_dir: Path, *, install_signal: bool = True) -> None:
        self.path = run_dir / CONTROL_FILE
        self.pending: str = ""
        self.resume = threading.Event()
        self._previous: Any = None
        self._installed = False
        if install_signal:
            self._install()

    def _install(self) -> None:
        try:
            self._previous = signal.signal(signal.SIGINT, self._on_sigint)
            self._installed = True
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            self._installed = False

    def _on_sigint(self, _signum: int, _frame: Any) -> None:
        self.pending = "q"
        self.resume.set()

    def poll(self) -> str:
        """The next command, or the empty string. One stat call per round."""
        if self.pending:
            command, self.pending = self.pending, ""
            return command
        try:
            if not self.path.exists():
                return ""
            letter = self.path.read_text(encoding="utf-8").strip()[:1].lower()
            self.path.unlink()
        except OSError:
            return ""
        return letter if letter in ("c", "q", "p") else ""

    def wait_while_paused(self, printer: Callable[[str], None]) -> str:
        """Block until something says to go on, and return what it said.

        A blocking wait on an Event rather than a sleep loop: a paused run costs nothing, and
        the event is what a signal sets.
        """
        printer("paused; write c, q or p to the control file, or press Ctrl-C")
        while True:
            self.resume.wait(0.25)
            self.resume.clear()
            command = self.poll()
            if command and command != "p":
                return command
            if self.pending:  # pragma: no cover - set between the poll and the check
                return self.poll()

    def close(self) -> None:
        if self._installed and self._previous is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGINT, self._previous)
            self._installed = False


# ---------------------------------------------------------------------------
# The coordinator
# ---------------------------------------------------------------------------


_SEGMENTS = count()


def run_directory(config: RunConfig, run_id: str) -> Path:
    """``<runs_dir>/<run_name>-<run_id>``: where everything one run writes ends up."""
    return Path(config.runs_dir) / f"{config.run_name}-{run_id}"


class LearningCoordinator:
    """One run, from preflight to the last checkpoint.

    Constructed from a config and entered as a context manager: ``__enter__`` runs the start-up
    gates, builds every component and opens the run directory, and ``__exit__`` closes the
    workers whatever happened. ``learn`` is the loop.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        printer: Callable[[str], None] | None = print,
        device: str | None = None,
        codec: str = DEFAULT_CODEC,
        resume: str | Path | None = None,
        allow_identity_drift: bool = False,
        source: RolloutSource | None = None,
        preflight_kwargs: Mapping[str, Any] | None = None,
        install_signal_handler: bool = True,
    ) -> None:
        self.config = validate(config)
        self.printer = printer or (lambda _line: None)
        self.geometry = geometry(self.config)
        self.codec_path = codec
        self.resume_from = Path(resume) if resume is not None else None
        self.allow_identity_drift = bool(allow_identity_drift)
        self.requested_device = device
        self.preflight_kwargs = dict(preflight_kwargs or {})
        self.install_signal_handler = install_signal_handler
        self._given_source = source

        self.report: PreflightReport | None = None
        self.identity: RunIdentity | None = None
        self.run_id = ""
        self.run_dir = Path()
        self.iteration = 0
        self.cumulative_timesteps = 0
        self.cumulative_env_steps = 0
        self.wall_seconds = 0.0
        self.resumed_with_drift = False
        self.rows: list[dict[str, MetricValue]] = []
        self.alarm_rows: list[AlarmResult] = []
        self.recent_episodes: list[EpisodeRecord] = []
        self.last_decision: GateDecision | None = None
        self.ratings: RatingTable | None = None
        self.snapshots_taken = 0
        self.last_gate_seconds = 0.0
        self.gate_seconds_total = 0.0
        self._entered = False
        self._closed = False
        self._last_checkpoint_step = 0
        self._last_checkpoint_seconds = 0.0
        self._last_candidate_step = 0
        self._last_floor_step = 0

    # -- construction --------------------------------------------------------

    def __enter__(self) -> LearningCoordinator:
        if self._entered:
            return self
        import torch

        from .determinism import apply as apply_determinism
        from .learn.buffer import RectBuffer
        from .learn.gae import GAE
        from .learn.inference import BatchedInference
        from .learn.nets import DefaultNetworkFactory
        from .learn.ppo import PPOUpdate
        from .learn.schedules import ScheduleSet

        config = self.config
        self.determinism = apply_determinism(
            config.determinism.tier,
            torch_threads=config.determinism.torch_threads,
        )
        torch.manual_seed(derive_int(config.master_seed, stream_path(TORCH_GLOBAL)))
        if torch.cuda.is_available():  # pragma: no cover - the suite runs on the CPU
            torch.cuda.manual_seed_all(derive_int(config.master_seed, stream_path(TORCH_CUDA)))

        self.device = self._resolve_device(torch)
        self.report = run_preflight(
            config,
            codec=self.codec_path,
            # The caller's own preflight settings win over the coordinator's, so a test that
            # asks for a smaller sample is not fighting the printer it also passed.
            **{"printer": self.printer, **self.preflight_kwargs},
        )
        report = self.report
        self.spec = report.spec

        self.factory = DefaultNetworkFactory(config.master_seed)
        self.arch_digest = self.factory.arch_digest(self.spec, config.net)
        self.model = self.factory.build(self.spec, config.net, device=self.device)

        self.identity = compute_identity(
            config,
            env_spec=self.spec,
            build=report.build,
            arch_digest=self.arch_digest,
            codec_version=report.codec_version,
            codec_table_digest=report.codec_table_digest,
            device_kind=describe_device(str(self.device)),
        )
        self.run_id = run_id_of(self.identity)
        report.note_run_id(self.run_id, printer=self.printer)
        self.run_dir = run_directory(config, self.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.codec = build_codec(
            self.codec_path, self.spec, report.table, tuple(config.extra_component_modules)
        )
        self.buffer = RectBuffer(
            self.spec,
            self.codec,
            run_id=self.run_id,
            cycles=self.geometry.cycles,
            n_slots=self.geometry.n_slots,
            device=self.device,
            discard_opponent_rows=config.ppo.discard_opponent_rows,
            segment_name=f"{buffer_segment_name(self.run_id)}-{os.getpid():x}"
            f"-{next(_SEGMENTS)}",
        )
        self.buffer.set_static_planes(report.statics)
        self.source = self._build_source()
        self.planner = self.source.planner

        self.schedules = ScheduleSet.from_config(config)
        self.gae = GAE(
            standardize_rewards=config.advantage.standardize_rewards,
            reward_clip=config.advantage.reward_clip,
        )
        self.update = PPOUpdate(
            self.model,
            self.gae,
            config.ppo,
            master_seed=config.master_seed,
            backoff=self.schedules.backoff,
            device=self.device,
            progress=self.printer,
        )
        self._build_ladder()
        self.inference = BatchedInference(
            self.buffer,
            self.model,
            master_seed=config.master_seed,
            snapshots=self.snapshot_store,
            device=self.device,
        )
        self.probe = PolicyProbe(self.spec)
        self.sinks: MetricsSink = build_sinks(
            config.metrics.sinks,
            jsonl={"keep_episode_log_iterations": config.metrics.keep_episode_log_iterations},
        )
        self.config_json = dump_config(config, indent=2)
        self.sinks.open(
            identity=self.identity, config_json=self.config_json, run_dir=self.run_dir
        )
        self.alarms = AlarmSet(
            config.alarms, ratio_atol=self._ratio_atol(), printer=self.printer
        )
        self.store = DirCheckpointStore(self.run_dir, keep=config.checkpoint.keep)
        self.components = self._components()
        self.control = _Control(self.run_dir, install_signal=self.install_signal_handler)
        if self.resume_from is not None:
            self._load(self.resume_from)
        self._entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Shut everything down, once, whatever state it is in.

        Called from ``__exit__`` and safe to call again: a close that raised because a worker
        was already gone would take the emergency checkpoint's process down with it.
        """
        if self._closed:
            return
        self._closed = True
        for shutdown in (
            getattr(self, "control", None),
            getattr(self, "source", None),
            getattr(self, "player", None),
            getattr(self, "sinks", None),
            getattr(self, "results", None),
            getattr(self, "buffer", None),
        ):
            if shutdown is None:
                continue
            with contextlib.suppress(Exception):
                shutdown.close()

    def _ratio_atol(self) -> float:
        """The tolerance the ratio invariant is asserted at, under this run's precision.

        Read through the update's own name for each precision rather than through the config's
        spelling of the autocast dtype: the alarm and the assertion have to be about the same
        number, and two ways of naming float32 is how they would stop being.
        """
        from .learn.nets import resolve_dtype
        from .learn.ppo import PRECISION_NAMES
        from .metrics.alarms import DEFAULT_RATIO_ATOL

        dtype = resolve_dtype(self.config.net.autocast_dtype)
        name = PRECISION_NAMES.get(dtype, str(self.config.net.autocast_dtype))
        return float(self.config.ppo.ratio_atol.get(name, DEFAULT_RATIO_ATOL))

    def _resolve_device(self, torch: Any) -> torch.device:
        """The device this run trains on.

        A config that asks for CUDA on a machine without it is answered with the CPU and a
        line saying so, rather than with an exception: the shipped profile names CUDA and the
        whole test suite runs where there is none.
        """
        wanted = self.requested_device or self.config.net.device
        if wanted.startswith("cuda") and not torch.cuda.is_available():
            self.printer(f"device        {wanted} is not available; this run uses the CPU")
            wanted = "cpu"
        return torch.device(wanted)

    def _build_source(self) -> RolloutSource:
        if self._given_source is not None:
            return self._given_source
        assert self.report is not None
        viser = bool(os.environ.get("ROYALEVISER"))
        kwargs: dict[str, Any] = {
            "run_id": self.run_id,
            "codec": self.codec_path,
            "geometry": self.geometry,
            "viser": viser,
        }
        if self.config.rollout.source == "inline":
            return InlineRolloutSource(self.config, self.spec, self.report.table, **kwargs)
        from .rollout.farm import ProcessRolloutSource

        return ProcessRolloutSource(self.config, self.spec, self.report.table, **kwargs)

    def _build_ladder(self) -> None:
        """The pool, the archive, the fit, the seed set and the gate.

        The context is what says two results may be pooled, and it is computed from the same
        ``config()`` dict preflight already read: the env recipe, what the policy was shown and
        which engine build produced it.
        """
        config = self.config
        assert self.report is not None
        ladder_dir = self.run_dir / "ladder"
        ladder_dir.mkdir(parents=True, exist_ok=True)
        self.context = context_digest(
            config.env.digest(),
            self.spec.obs_digest,
            config.ladder.release_mode,
            self.report.build.build_digest or self.report.build.calibration_digest,
        )
        self.snapshot_template = SnapshotSpec(
            snapshot_id="",
            arch_digest=self.arch_digest,
            obs_digest=self.spec.obs_digest,
            action_digest=self.identity.action_digest if self.identity else "",
            codec_version=self.report.codec_version,
            codec_table_digest=self.report.codec_table_digest,
            frame_stack=self.spec.frame_stack,
            num_cards=self.spec.num_cards,
            vector_size=self.spec.vector_size,
            context=self.context,
            run_id=self.run_id,
        )
        self.snapshot_store = DiskSnapshotStore(
            self.run_dir / "snapshots",
            template=self.snapshot_template,
            build=self._build_actor,
            max_resident=config.ladder.max_resident_opponents + 2,
        )
        self.results = ResultLog(ladder_dir / "games.jsonl")
        self.pool = LadderPool(
            self.results,
            snapshots=self.snapshot_store,
            context=self.context,
            run_id=self.run_id,
        )
        self.matchmaker = MixMatchmaker(config.master_seed, config.ladder)
        self.rater = BradleyTerryDavidsonRater(
            prior_sd=config.ladder.rater.prior_sd,
            anchor=config.ladder.rater.anchor,
            draws=config.ladder.rater.draws,
        )
        self.elo = EloReadout()
        self.eviction = HallOfFameEviction()
        self.seeds: SeedSet = eval_seed_set(config.master_seed, config.ladder.eval_seed_count)
        (ladder_dir / "eval_seeds.json").write_bytes(msgspec.json.encode(self.seeds))
        eval_env = config.eval_env if config.eval_env is not None else msgspec.structs.replace(
            config.env, truncation=[]
        )
        self.eval_env_spec = eval_env
        self.player = EnvBattlePlayer(
            self.spec,
            eval_env,
            self._eval_actor,
            master_seed=config.master_seed,
            extra_modules=tuple(config.extra_component_modules),
            device=self.device,
            release_mode=config.ladder.release_mode,
        )
        self.eval_runner = _RecordingEvalRunner(
            self.player,
            self.seeds,
            master_seed=config.master_seed,
            context=self.context,
            run_id=self.run_id,
            log=self.results,
            bootstrap_resamples=config.ladder.gate.bootstrap_resamples,
            release_mode=config.ladder.release_mode,
            obs_digest=self._snapshot_obs_digest,
        )
        self.gate = WilsonGate(
            config.ladder.gate,
            self.rater,
            gates_dir=ladder_dir / "gates",
        )

    def _build_actor(self, device: Any) -> Any:
        """A bare actor of this run's architecture, for a snapshot to load into."""
        from .learn.actor_critic import ClashActor
        from .learn.nets import ClashTrunk, PointerPolicyHead, resolve_dtype

        actor = ClashActor(
            ClashTrunk(self.spec, self.config.net),
            PointerPolicyHead(self.spec, self.config.net),
            resolve_dtype(self.config.net.autocast_dtype),
        )
        return actor.to(device)

    def _snapshot_obs_digest(self, member: str) -> str:
        """What a member saw, for the evaluation runner's pairing check.

        A scripted opponent sees whatever it is shown and is comparable with anything, so it
        answers with this run's own digest rather than with an empty string that would read as
        a disagreement.
        """
        if member in self.pool.anchors or member == LEARNER_ID:
            return self.spec.obs_digest
        try:
            return self.snapshot_store.obs_digest(member)
        except KeyError:
            return self.spec.obs_digest

    def _eval_actor(self, member: str) -> Callable[[dict[str, Any], float, Any], int]:
        """One evaluation seat's policy: a script, or a frozen actor from the archive."""
        if member.startswith("scripted:"):
            return _scripted_policy(member.split(":", 1)[1])
        actor = self.snapshot_store.get(member, self.device)
        return self._actor_policy(actor)

    def _actor_policy(self, actor: Any) -> Callable[[dict[str, Any], float, Any], int]:
        """A frozen actor behind the battle player's signature.

        One observation at a time and on the device the run uses. Evaluation is a few hundred
        battles at a gate, against an update of tens of thousands of rows, so a batch of one is
        the right trade for a path with no rectangle behind it.
        """
        import torch

        from .learn.distribution import MaskedCategorical

        def act(obs: dict[str, Any], uniform: float, _rng: Any) -> int:
            batch = self._obs_batch(obs)
            with torch.inference_mode():
                distribution = MaskedCategorical(actor.logits(batch).float(), batch.mask)
                if self.config.ladder.release_mode == "argmax":
                    return int(distribution.mode()[0].item())
                draw = torch.tensor([uniform], dtype=torch.float32, device=batch.mask.device)
                return int(distribution.sample(draw)[0].item())

        return act

    def _obs_batch(self, obs: Mapping[str, Any]) -> Any:
        """One environment observation as the batch of one the networks take.

        Frames are stacked the way the rectangle stacks them -- the current frame first -- and
        an evaluation battle carries no history, so the older frames are zero on the first
        decision and the previous observations after it. It is built here rather than through
        the codec because nothing is being stored: quantising an observation in order to
        dequantise it again would be a round trip for its own sake.
        """
        import torch

        spatial = np.asarray(obs["spatial"], dtype=np.float32)
        mask = np.asarray(obs["action_mask"]).astype(bool)
        planes = mask[1:].reshape(self.spec.hand_size, *self.spec.tiles).astype(np.float32)
        frames = self.spec.frame_stack
        history = getattr(self, "_eval_history", None)
        if frames > 1:
            if history is None or history[0].shape != spatial.shape:
                history = [np.zeros_like(spatial) for _ in range(frames - 1)]
                self._eval_history = history
            spatial_stack = np.concatenate([spatial, *history], axis=0)
            plane_stack = np.concatenate(
                [planes, *[np.zeros_like(planes) for _ in history]], axis=0
            )
            history.insert(0, spatial)
            del history[frames - 1 :]
        else:
            spatial_stack, plane_stack = spatial, planes
        from .api.policy import ObsBatch

        def tensor(array: np.ndarray, dtype: Any) -> Any:
            return torch.from_numpy(np.ascontiguousarray(array)).to(
                device=self.device, dtype=dtype
            )[None]

        return ObsBatch(
            spatial=tensor(spatial_stack, torch.float32),
            mask_planes=tensor(plane_stack, torch.float32),
            vector=tensor(np.asarray(obs["vector"], dtype=np.float32), torch.float32),
            mask=tensor(mask, torch.bool),
        )

    def _components(self) -> CheckpointComponents:
        """Every piece of the run that goes into a checkpoint, under its own folder name."""
        from .checkpoint import RngComponent

        self.schedule_component = _ScheduleComponent(self.schedules)
        self.rollout_component = _RolloutComponent(
            self.source, self.source.planner, self.matchmaker
        )
        self.rng = RngComponent(
            master_seed=self.config.master_seed, eval_seed_set_sha=self.seeds.sha()
        )
        return {
            "actor_critic": _ModelComponent(
                self.model,
                arch=self.config.net,
                arch_digest=self.arch_digest,
                device=self.device,
            ),
            "optimizers": self.update,
            "schedules": self.schedule_component,
            "advantage": self.gae,
            "rollout": self.rollout_component,
            "ladder": self.pool,
            "matchmaker": self.matchmaker,
            "rating": _GroupComponent([self.rater, self.elo]),
            "metrics": self.sinks,
            "rng": self.rng,
        }

    # -- the loop ------------------------------------------------------------

    def learn(self, until_timesteps: int | None = None) -> None:
        """Collect, update, rate and record until the timestep limit.

        The whole loop is wrapped, ``KeyboardInterrupt`` explicitly: it is not an ``Exception``,
        so a bare ``except Exception`` would skip its own emergency save on Ctrl-C, which is
        the one crash a long run actually experiences.
        """
        if not self._entered:
            raise RuntimeError("a LearningCoordinator runs inside its own with-block")
        limit = int(until_timesteps or self.config.timestep_limit)
        started = time.perf_counter()
        try:
            while self.cumulative_timesteps < limit:
                if self._iterate(started) == "q":
                    break
        except (Exception, KeyboardInterrupt) as exc:
            self._emergency(exc)
            raise
        finally:
            self.close()

    def iterate(self) -> str:
        """One iteration, for a caller that drives the loop itself.

        ``bench`` is the caller that does: it wants a measured iteration and not a run, and a
        second copy of the ordering inside it would be a second answer to what an iteration is.
        """
        return self._iterate(time.perf_counter())

    def checkpoint(self) -> Path:
        """Write a checkpoint now. What the interactive ``c`` does, and what a proof does."""
        return self._checkpoint()

    def _iterate(self, run_started: float) -> str:
        """One whole iteration, and whatever the interactive control asked for during it."""
        began = time.perf_counter()
        sched = self.schedules.state(
            iteration=self.iteration,
            cumulative_env_steps=self.cumulative_env_steps,
            cumulative_timesteps=self.cumulative_timesteps,
        )
        self.schedule_component.last = sched
        self.rng.iteration = self.iteration
        self.rng.shard_streams = self.rollout_component.shard_streams()

        plan = self.matchmaker.plan(
            self.iteration, self.pool, self.ratings, self.geometry
        )
        self.buffer.begin_iteration(plan, self.geometry.cycles)
        self.source.begin_iteration(plan, self.buffer, self.iteration)
        self.inference.begin_iteration(plan)

        collection = self._collect(plan, sched)
        episodes = collection["episodes"]
        self.recent_episodes = (self.recent_episodes + episodes)[-200:]

        result = self.update.step(self.buffer, sched)
        trainable = self.buffer.trainable()
        self._check_iteration(collection, trainable, episodes)

        self.cumulative_timesteps += int(trainable.sum())
        self.cumulative_env_steps += self.geometry.cycles * self.geometry.n_battles
        self.iteration += 1
        self.wall_seconds += time.perf_counter() - began

        self._record_training_results(episodes)
        gate_seconds = self._maybe_gate()
        self._maybe_refit()

        row = self._row(
            sched=sched,
            result=result,
            episodes=episodes,
            collection=collection,
            trainable=trainable,
            iteration_seconds=time.perf_counter() - began,
            gate_seconds=gate_seconds,
            wall=time.perf_counter() - run_started,
        )
        self.rows.append(row)
        fired = self.alarms.evaluate(row, on_halt=self._halt_bundle, on_dump=self._dump_bundle)
        self.alarm_rows.extend(fired)
        self.sinks.write(row)
        self.sinks.write_episodes(episodes)
        if fired:
            self.sinks.write_alarms(fired)
        self._maybe_heatmap()
        command = collection["command"]
        if command in ("c", "q") or self._checkpoint_due():
            self._checkpoint()
        return command

    def _say_collected(self, started: float) -> None:
        """Say that collection is done and the update has it, before the long quiet part.

        An iteration prints nothing until its metrics row, and at the shipped laptop profile
        that row is minutes away -- almost all of it inside one call to the update, which
        collects no rounds and answers no control letter. A run silent for that long is
        indistinguishable from one that has stopped, and the person watching kills it, which
        has happened twice on this project for two unrelated reasons. One line, naming which
        phase the silence belongs to, is the difference between waiting and killing it.
        """
        elapsed = time.perf_counter() - started
        geo = self.geometry
        self.printer(
            f"collected     {geo.cycles} cycles, {geo.cycles * geo.learner_rows} timesteps in "
            f"{elapsed:.1f}s ({geo.cycles * geo.n_slots / max(elapsed, 1e-9):.0f} env steps/s); "
            f"updating"
        )

    def _collect(self, plan: SlotPlan, sched: ScheduleState) -> dict[str, Any]:
        """The rectangle, one shard-round at a time.

        The assignment step comes before routing in the same round, so the first observation of
        a new episode is already routed by that episode's own assignment and no transition is
        ever produced under a stale one.
        """
        geo = self.geometry
        source = self.source
        buffer = self.buffer
        timeout = self.config.rollout.round_timeout_s
        group = np.full(geo.n_slots, GROUP_LEARNER, dtype=np.int8)
        opponent = np.full(geo.n_slots, -1, dtype=np.int8)
        seat = np.full(geo.n_battles, 0, dtype=np.int8)
        for assignment in plan.assignment:
            battle = assignment.battle
            group[2 * battle : 2 * battle + 2] = assignment.group
            opponent[2 * battle : 2 * battle + 2] = _opponent_index(plan, assignment)
            seat[battle] = assignment.learner_seat

        episodes: list[EpisodeRecord] = []
        rounds = 0
        command = ""
        started = time.perf_counter()
        env_seconds = 0.0
        for cycle in range(geo.cycles):
            for _shard in range(geo.shards_per_worker):
                round_ = source.next_round(timeout)
                check_round(
                    round_, cycle=cycle, slots=self.planner.round_slots(round_.shard)
                )
                env_seconds += float(round_.timings.get("env_ms", 0.0)) / 1000.0
                assigned = self._assign(plan, round_, group, opponent, seat, episodes)
                round_.group[:] = group[round_.slots]
                answer = self.inference.act(round_)
                slots = round_.slots
                source.submit(
                    Step(
                        actions=answer.actions.astype(np.int16),
                        gamma=sched.gamma,
                        group=group[slots] if assigned else np.zeros(0, dtype=np.int8),
                        opponent_ix=(
                            opponent[slots] if assigned else np.zeros(0, dtype=np.int8)
                        ),
                        learner_seat=(
                            seat[self.planner.slot_battle[slots][::2]]
                            if assigned
                            else np.zeros(0, dtype=np.int8)
                        ),
                    )
                )
                buffer.record_round(round_, answer.actions, answer.log_probs)
                final_slots, final_rows = source.final_rows(round_)
                self.update.record_final_observations(round_.cycle, final_slots, final_rows)
                rounds += 1
                command = self.control.poll() or command
                if command == "p":
                    command = self.control.wait_while_paused(self.printer)
        for round_ in source.finish_iteration():
            for record in round_.episodes:
                self.matchmaker.on_episode(record)
            episodes.extend(round_.episodes)
        self._handle_failures()
        self._say_collected(started)
        return {
            "episodes": episodes,
            "rounds": rounds,
            "seconds": time.perf_counter() - started,
            "env_seconds": env_seconds,
            "command": command,
        }

    def _assign(
        self,
        plan: SlotPlan,
        round_: RolloutRound,
        group: np.ndarray,
        opponent: np.ndarray,
        seat: np.ndarray,
        episodes: list[EpisodeRecord],
    ) -> bool:
        """Draw a fresh assignment for every battle whose episode just ended."""
        for record in round_.episodes:
            self.matchmaker.on_episode(record)
        episodes.extend(round_.episodes)
        ended = round_.slots[np.asarray(round_.episode_end) != 0]
        if ended.size == 0:
            return False
        battles = sorted({int(self.planner.slot_battle[slot]) for slot in ended})
        for battle in battles:
            assignment = self.matchmaker.assign(
                battle, self.matchmaker.ordinal(battle), self.pool, self.ratings
            )
            group[2 * battle : 2 * battle + 2] = assignment.group
            opponent[2 * battle : 2 * battle + 2] = _opponent_index(plan, assignment)
            seat[battle] = assignment.learner_seat
        return True

    def _handle_failures(self) -> None:
        """Restart what died, count it, and stop if one worker keeps dying."""
        failures = self.source.drain_failures()
        if not failures:
            return
        self.failures_by_kind = getattr(self, "failures_by_kind", {})
        for failure in failures:
            self.failures_by_kind[failure.kind] = self.failures_by_kind.get(failure.kind, 0) + 1
            # What is lost is the worker, not the shard that raised. A child that fails
            # anywhere closes every shard it holds and exits, because a protocol error means
            # the bytes it was reading cannot be trusted and the other shard reads the same
            # segment. Naming only the shard understates the blast radius by half, and the
            # first place that shows is a short iteration nobody can account for.
            geo = self.geometry
            self.printer(
                f"worker {failure.worker} failed at cycle {failure.cycle} ({failure.kind}) "
                f"in shard {failure.shard}: {failure.message.splitlines()[-1:]}"
            )
            self.printer(
                f"              it held {geo.shards_per_worker} shards and "
                f"{geo.games_per_worker} battles; all of them are out until it restarts, "
                f"which is about "
                f"{100 * geo.games_per_worker / max(1, geo.n_battles):.0f}% of this iteration's "
                f"rows for the cycles it misses"
            )
            if self.config.rollout.restart_failed_workers:
                self.source.restart(failure.worker)

    # -- the invariants ------------------------------------------------------

    def _check_iteration(
        self,
        collection: Mapping[str, Any],
        trainable: np.ndarray,
        episodes: Sequence[EpisodeRecord],
    ) -> None:
        geo = self.geometry
        buffer = self.buffer
        expected_rounds = geo.cycles * geo.shards_per_worker
        if collection["rounds"] != expected_rounds:
            raise RoundsMissing(
                f"this iteration read {collection['rounds']} rounds and the rectangle is "
                f"{geo.cycles} cycles of {geo.shards_per_worker} shards"
            )
        wanted = self.config.ppo.timesteps_per_iteration * MIN_TRAINABLE_FRACTION
        if trainable.sum() < wanted:
            raise TrainableRowsShort(
                f"this iteration collected {int(trainable.sum())} trainable transitions "
                f"against the {self.config.ppo.timesteps_per_iteration} asked for"
            )
        log_probs = buffer.log_prob[: buffer.cycles]
        bad = np.argwhere(trainable & ~np.isfinite(log_probs))
        if bad.size:
            cycle, slot = (int(v) for v in bad[0])
            raise NonFiniteLogProb(
                f"{len(bad)} stored log-probability(ies) are not finite", cycle=cycle, slot=slot
            )
        refused = np.argwhere(trainable & (buffer.deploy_status[: buffer.cycles] > 0))
        if refused.size:
            cycle, slot = (int(v) for v in refused[0])
            raise DeployRefused(
                f"the engine refused {len(refused)} command(s) the mask allowed",
                cycle=cycle,
                slot=slot,
            )
        _check_pairs(episodes)
        expected = self.matchmaker.expected_discarded_rows_frac
        observed = _discarded_frac(buffer, trainable)
        if abs(observed - expected) > MIXTURE_TOLERANCE:
            raise MixtureDrifted(
                f"{observed:.3f} of the collected rows were discarded and the mixture "
                f"{self.config.ladder.mix} implies {expected:.3f}"
            )
        try:
            assignments_constant_within_episodes(
                buffer.group[: buffer.cycles], buffer.episode_end
            )
        except ValueError as exc:
            raise AssignmentInsideEpisode(str(exc)) from exc

    # -- the ladder ----------------------------------------------------------

    def _record_training_results(self, episodes: Sequence[EpisodeRecord]) -> None:
        """File the training battles, tagged so the fit leaves them out."""
        outcomes: list[tuple[str, str, float]] = []
        for record in episodes:
            if record.policy_id != LEARNER_ID or record.opponent_id == LEARNER_ID:
                continue
            outcomes.append(
                (LEARNER_ID, record.opponent_id, 0.5 + 0.5 * float(np.sign(record.outcome)))
            )
            self.elo.update(LEARNER_ID, record.opponent_id, outcomes[-1][2])
        if outcomes:
            self.pool.record_training_results(outcomes, iteration=self.iteration)

    def _maybe_gate(self) -> float:
        """Snapshot a candidate at the env-step cadence and put it through the gate."""
        config = self.config.ladder
        started = time.perf_counter()
        floor_due = (
            config.floor_admit_every_env_steps > 0
            and self.cumulative_env_steps - self._last_floor_step
            >= config.floor_admit_every_env_steps
        )
        candidate_due = (
            self.cumulative_env_steps - self._last_candidate_step
            >= config.candidate_every_env_steps
        )
        if not (candidate_due or floor_due):
            return 0.0
        self._last_candidate_step = self.cumulative_env_steps
        candidate = f"snap:v{self.snapshots_taken}"
        self.snapshots_taken += 1
        self.snapshot_store.put(
            candidate,
            self.model,
            {"step": self.cumulative_env_steps, "iteration": self.iteration},
        )
        if floor_due:
            self._last_floor_step = self.cumulative_env_steps
            decision = floor_decision(candidate, self.pool.champion)
        else:
            decision = self.gate.evaluate(candidate, self.pool, self.eval_runner)
        self.pool.apply(decision)
        self.last_decision = decision
        self._evict()
        seconds = time.perf_counter() - started
        self.gate_seconds_total += seconds
        self.printer(
            f"gate          {candidate}: admit={decision.admit} promote={decision.promote} "
            f"cycle={decision.cycle} in {seconds:.1f}s"
        )
        return seconds

    def _evict(self) -> None:
        if self.ratings is None:
            return
        evicted = self.eviction.select_for_eviction(
            pool=self.pool,
            ratings=self.ratings,
            max_sampled=self.config.ladder.pool_working_size,
        )
        if evicted:
            self.pool.evict(evicted)

    def _maybe_refit(self) -> None:
        every = max(1, self.config.ladder.refit_every_iterations)
        if self.iteration % every:
            return
        view = self.pool.eval_view()
        if not len(view):
            return
        self.ratings = self.rater.fit(view)
        self.pool.note_refit(self.ratings)
        folder = self.run_dir / "ladder" / "ratings"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.iteration:06d}.json").write_bytes(
            msgspec.json.encode(self.ratings)
        )

    # -- the row -------------------------------------------------------------

    def _row(
        self,
        *,
        sched: ScheduleState,
        result: Any,
        episodes: Sequence[EpisodeRecord],
        collection: Mapping[str, Any],
        trainable: np.ndarray,
        iteration_seconds: float,
        gate_seconds: float,
        wall: float,
    ) -> dict[str, MetricValue]:
        """One iteration as one flat row, merged from all three sources."""
        buffer = self.buffer
        geo = self.geometry
        stats = self.inference.drain_stats()
        timesteps = int(trainable.sum())
        env_steps = geo.cycles * geo.n_battles
        collection_seconds = max(1e-9, float(collection["seconds"]))
        update_seconds = max(1e-9, float(result.seconds))
        source_stats = self.source.stats()

        metrics = IterationMetrics(iteration=self.iteration)
        metrics.run = dict(schedule_fields(sched, decision_ms=self.spec.decision_ms))
        metrics.run["run/iteration"] = self.iteration
        metrics.run["run/cumulative_timesteps"] = self.cumulative_timesteps
        metrics.run["run/cumulative_env_steps"] = self.cumulative_env_steps
        metrics.run["run/cumulative_updates"] = self.update.model_updates
        metrics.run["run/wall_seconds"] = self.wall_seconds
        metrics.run["run/determinism_tier"] = self.config.determinism.tier
        metrics.run["run/resumed_with_drift"] = self.resumed_with_drift
        metrics.run["run/state_digest"] = self.state_digest()

        inference_seconds = stats.seconds
        ipc_seconds = max(
            0.0, collection_seconds - inference_seconds - float(collection["env_seconds"])
        )
        metrics.time = {
            "time/iteration": iteration_seconds,
            "time/collection": collection_seconds,
            "time/inference": inference_seconds,
            "time/env": float(collection["env_seconds"]),
            "time/codec": float(source_stats.get("codec_ms", 0.0))
            * collection["rounds"]
            / 1000.0,
            "time/ipc": ipc_seconds,
            "time/critic_pass": 0.0,
            "time/gae": 0.0,
            "time/update": update_seconds,
            "time/checkpoint": self._take_checkpoint_seconds(),
            "time/gate": gate_seconds,
            "time/overlap_saved": 0.0,
        }
        attributed = sum(
            float(metrics.time[key])
            for key in ("time/collection", "time/update", "time/checkpoint", "time/gate")
        )
        metrics.time["time/residual"] = max(0.0, iteration_seconds - attributed)

        metrics.throughput = {
            "throughput/overall_steps_per_second": timesteps / max(1e-9, iteration_seconds),
            "throughput/collected_steps_per_second": timesteps / collection_seconds,
            "throughput/engine_ticks_per_second": (
                env_steps * self.spec.decision_ticks / collection_seconds
            ),
            "throughput/rollout_capacity_ratio": update_seconds / collection_seconds,
            "throughput/boundary_mb_per_second": (
                self.buffer.layout.row_bytes
                * geo.n_slots
                * collection["rounds"]
                / 1e6
                / collection_seconds
            ),
            "throughput/parent_wait_frac": float(source_stats.get("parent_wait_frac", 0.0)),
            "throughput/inference_ms_per_round": (
                1000.0 * inference_seconds / max(1, collection["rounds"])
            ),
            "throughput/discarded_rows_frac": _discarded_frac(buffer, trainable),
            "throughput/gpu_util_frac": _gpu_util(),
        }

        metrics.ppo = dict(update_fields(result))
        metrics.ppo["ppo/advantage_std_pre_norm"] = float(self.update.advantage_std_pre_norm)
        stats_of = self.update.advantage_stats
        metrics.ppo["ppo/return_running_mean"] = (
            float(stats_of.raw_return_mean) if stats_of else 0.0
        )
        metrics.ppo["ppo/return_running_std"] = (
            float(stats_of.raw_return_std) if stats_of else 0.0
        )
        metrics.ppo["ppo/reward_clip_frac"] = (
            float(stats_of.clipped_reward_frac) if stats_of else 0.0
        )
        metrics.ppo["ppo/lr_backoff_events"] = self.schedules.backoff.events

        aggregate = episode_fields(
            episodes, truncation_steps=_truncation_steps(self.config)
        )
        metrics.env = {
            key: value for key, value in aggregate.fields.items() if key.startswith("env/")
        }
        metrics.policy = {
            key: value for key, value in aggregate.fields.items() if key.startswith("policy/")
        }
        probe = self.probe.measure(
            buffer, self.inference.gather, buffer.action[: buffer.cycles], trainable
        )
        for key, value in probe.items():
            (metrics.env if key.startswith("env/") else metrics.policy)[key] = value
        metrics.env.setdefault("env/illegal_action_rate", 0.0)
        for key, default in _ENV_DEFAULTS.items():
            metrics.env.setdefault(key, default)
        metrics.policy.setdefault("policy/cards_per_match", 0.0)

        metrics.ladder = dict(
            ladder_fields(
                self.pool,
                ratings=self.ratings,
                decision=self.last_decision,
                elo=self.elo.rating(LEARNER_ID),
                paired_rho=self._paired_rho(),
                gate_seconds_frac=self.gate_seconds_total / max(1e-9, wall),
            )
        )

        failures = getattr(self, "failures_by_kind", {})
        metrics.health = {
            "health/illegal_action_rate": float(metrics.env["env/illegal_action_rate"]),
            "health/mask_disagreements": 0,
            "health/worker_restarts": int(sum(getattr(self.source, "restarts", []) or [0])),
            "health/rows_dropped_dead_worker": int(
                (~buffer.valid[: buffer.cycles]).sum()
            ),
            "health/obs_codec_clipped": int(getattr(self.codec, "clipped", 0)),
            "health/samples_unused_frac": float(result.samples_unused_frac),
            "health/nan_guard_trips": _nan_guard_trips(result),
            "health/vram_peak_mb": _vram_peak_mb(),
            "health/rss_peak_mb": _rss_peak_mb(),
            "health/buffer_fill_frac": _fill_frac(buffer, collection["rounds"], geo),
        }
        for kind, n in sorted(failures.items()):
            metrics.health[f"health/worker_failures/{kind}"] = int(n)
        return metrics.row()

    def _paired_rho(self) -> float:
        """How much of a battle's outcome the start state decided, from the last comparison.

        It is the number that sets the gate's real effective sample size, and it is worth
        knowing on its own, so it is kept off the last comparison the evaluation runner made
        rather than recomputed from the result log.
        """
        comparison = self.eval_runner.last
        return float(comparison.rho) if comparison is not None else 0.0

    def _take_checkpoint_seconds(self) -> float:
        """What the last checkpoint cost, once. A checkpoint is written after the row it
        belongs to, so it is reported on the row after it and not on every one since."""
        seconds, self._last_checkpoint_seconds = self._last_checkpoint_seconds, 0.0
        return seconds

    # -- checkpoints ---------------------------------------------------------

    def _checkpoint_due(self) -> bool:
        every = self.config.checkpoint.every_env_steps
        return bool(
            every > 0 and self.cumulative_env_steps - self._last_checkpoint_step >= every
        )

    def manifest(self) -> Manifest:
        """What one checkpoint is, beside the folders that hold it."""
        assert self.identity is not None
        return Manifest(
            format_version=CHECKPOINT_FORMAT_VERSION,
            run_id=self.run_id,
            run_name=self.config.run_name,
            identity=self.identity,
            config=msgspec.json.decode(self.config_json.encode("utf-8")),
            config_hash=config_hash(self.config),
            iteration=self.iteration,
            cumulative_env_steps=self.cumulative_env_steps,
            cumulative_timesteps=self.cumulative_timesteps,
            cumulative_model_updates=self.update.model_updates,
            wall_seconds=self.wall_seconds,
            created_unix_ns=time.time_ns(),
            state_digest=self.state_digest(),
            component_versions={},
            files={},
        )

    def _checkpoint(self) -> Path:
        started = time.perf_counter()
        self.rng.iteration = self.iteration
        self.rng.shard_streams = self.rollout_component.shard_streams()
        path = self.store.write(self.components, self.manifest())
        self.store.prune(keep=self.config.checkpoint.keep)
        self._last_checkpoint_step = self.cumulative_env_steps
        self._last_checkpoint_seconds = time.perf_counter() - started
        self.printer(f"checkpoint    {path}")
        return path

    def _load(self, path: Path) -> None:
        """Resume from a checkpoint: refuse an identity that moved, then restore everything."""
        assert self.identity is not None
        manifest = msgspec.json.decode(
            (Path(path) / "manifest.json").read_bytes(), type=Manifest
        )
        drift = check_resume(
            manifest,
            self.identity,
            msgspec.json.decode(self.config_json.encode("utf-8")),
            allow_drift=self.allow_identity_drift,
        )
        del drift
        from .identity import identity_differences

        differences = identity_differences(manifest.identity, self.identity)
        if differences and self.allow_identity_drift:
            self.resumed_with_drift = True
            (self.run_dir / "drift.json").write_bytes(
                msgspec.json.encode(
                    {name: [str(was), str(now)] for name, (was, now) in differences.items()}
                )
            )
        self.store.read(
            Path(path), self.components, strict=self.config.checkpoint.strict_load
        )
        self.iteration = manifest.iteration
        self.cumulative_env_steps = manifest.cumulative_env_steps
        self.cumulative_timesteps = manifest.cumulative_timesteps
        self.wall_seconds = manifest.wall_seconds
        self._last_checkpoint_step = manifest.cumulative_env_steps
        self._last_candidate_step = manifest.cumulative_env_steps
        self._last_floor_step = manifest.cumulative_env_steps
        self.ratings = self.rater.table
        if self.ratings is not None:
            self.pool.note_refit(self.ratings)
        self.printer(
            f"resumed       iteration {self.iteration} at {self.cumulative_env_steps} env "
            f"steps, state {manifest.state_digest[:16]}"
        )

    def state_digest(self) -> str:
        """sha256 over everything the next iteration's numbers depend on.

        The weights, both optimizers' moments, the return scaler and the schedule positions:
        what it is for is that the iteration at which two runs first differ is a lookup rather
        than a bisection.
        """
        digest = hashlib.sha256()
        _hash_tensors(digest, self.model.state_dict())
        for optimizer in self.update.optimizers:
            state = optimizer.state_dict()["state"]
            for key in sorted(state):
                digest.update(str(key).encode("utf-8"))
                _hash_tensors(digest, state[key])
        digest.update(msgspec.json.encode(self.gae.scaler.state()))
        digest.update(msgspec.json.encode(self.schedules.backoff.state()))
        digest.update(str(self.update.model_updates).encode("utf-8"))
        return digest.hexdigest()

    # -- what a halt leaves behind -------------------------------------------

    def _halt_bundle(self, alarm: AlarmResult) -> str | None:
        """A checkpoint and a diagnostic bundle, in that order, before the halt is raised."""
        with contextlib.suppress(Exception):
            self._checkpoint()
        return self._dump_bundle(alarm)

    def _dump_bundle(self, alarm: AlarmResult | None = None, note: str = "") -> str | None:
        try:
            trace, divergences = self._replay_one()
        except Exception as exc:  # pragma: no cover - a bundle never fails on its own parts
            trace, divergences = None, [f"the replay raised {type(exc).__name__}: {exc}"]
        folder = write_bundle(
            self.run_dir,
            self.iteration,
            identity=self.identity,
            config_json=self.config_json,
            rows=self.rows,
            alarms=([alarm] if alarm is not None else []) + self.alarm_rows[-20:],
            episodes=self.recent_episodes,
            outliers=self._ratio_outliers(),
            state_digest=self.rows[-1].get("run/state_digest", "") if self.rows else "",
            trace=trace,
            divergences=divergences,
            note=note or (alarm.message if alarm is not None else ""),
        )
        self.printer(f"bundle        {folder}")
        return str(folder)

    def _replay_one(self) -> tuple[Any, list[str]]:
        """One of the last episodes, re-simulated from its own shard seed and reset ordinal."""
        if not self.recent_episodes:
            return None, []
        record = self.recent_episodes[-1]
        seed = self.planner.env_seed(
            record.worker, record.shard, self.source.generation[record.worker]
        )
        return replay_episode(self.config, record, seed=seed)

    def _ratio_outliers(self) -> list[RatioOutlier]:
        """The largest importance-ratio deviations this iteration reported.

        The update reports the maximum rather than the samples behind it, so what the bundle
        carries is that maximum with the iteration it belongs to. A per-sample list would need
        the update to keep one.
        """
        if not self.rows:
            return []
        deviation = float(self.rows[-1].get("ppo/ratio_max_abs_dev", 0.0) or 0.0)
        return [
            RatioOutlier(
                ratio=1.0 + deviation,
                deviation=deviation,
                cycle=-1,
                slot=-1,
                ordinal=self.iteration,
            )
        ]

    def _emergency(self, exc: BaseException) -> None:
        """The last thing a crashing run does: save, then say where."""
        self.printer(f"stopping after {type(exc).__name__}: {exc}")
        try:
            self._checkpoint()
        except Exception as inner:  # pragma: no cover - the save is best effort
            self.printer(f"the emergency checkpoint failed: {inner}")

    def _maybe_heatmap(self) -> None:
        every = self.config.metrics.image_every
        if every <= 0 or self.iteration % every:
            return
        folder = self.run_dir / "artifacts"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"play-heatmap-{self.iteration:06d}.json"
        path.write_bytes(msgspec.json.encode(self.probe.heatmap()))
        self.sinks.write_artifact("policy/play_heatmap", path)


# ---------------------------------------------------------------------------
# Small shared arithmetic
# ---------------------------------------------------------------------------

#: What an ``env/`` row carries when no episode finished this iteration. The keys exist because
#: the schema says a run emits them; the zeros say nothing finished, which is the truth.
_ENV_DEFAULTS: dict[str, MetricValue] = {
    "env/episode_steps_mean": 0.0,
    "env/episode_steps_p05": 0.0,
    "env/episode_steps_p50": 0.0,
    "env/episode_steps_p95": 0.0,
    "env/episode_steps_hist": "{}",
    "env/episode_steps_at_cap_frac": 0.0,
    "env/ticks_mean": 0.0,
    "env/crowns_for": 0.0,
    "env/crowns_against": 0.0,
    "env/crown_diff": 0.0,
    "env/tower_hp_frac_end_own": 0.0,
    "env/tower_hp_frac_end_enemy": 0.0,
    "env/draw_rate": 0.0,
    "env/win_rate_by_seat": 0.5,
    "env/win_rate_by_seat_ci95_lo": 0.0,
    "env/win_rate_by_seat_ci95_hi": 1.0,
    "env/elixir_leak_frac": 0.0,
    "env/elixir_count_exact_frac": 1.0,
    "env/reward_shaping_abs": 0.0,
    "env/reward_terminal_abs": 0.0,
}


def _opponent_index(plan: SlotPlan, assignment: Any) -> int:
    from .rollout.plan import opponent_index

    return opponent_index(plan, assignment)


def _check_pairs(episodes: Iterable[EpisodeRecord]) -> None:
    """Both seats of a battle report the episode, or neither does."""
    counts: dict[tuple[int, int, int, int], list[EpisodeRecord]] = {}
    for record in episodes:
        counts.setdefault(
            (record.worker, record.shard, record.battle, record.ordinal), []
        ).append(record)
    for key, group in counts.items():
        if len(group) != 2:
            raise EpisodesUnpaired(
                f"battle {key[2]} of worker {key[0]} shard {key[1]} reported "
                f"{len(group)} seat(s) for episode {key[3]}",
                slot=group[0].slot,
            )


def _fill_frac(buffer: Any, rounds: int, geo: Any) -> float:
    """How much of the rectangle this iteration wrote, as a share of what it holds.

    One round writes one shard's slots, and every cell of the rectangle is written exactly
    once, so a healthy iteration reads exactly one. It is reported rather than assumed because
    the number above one is the one that matters: a cycle published twice fills a row with the
    wrong transition and nothing downstream would notice.
    """
    capacity = geo.cycles * geo.n_slots
    if capacity == 0:
        return 0.0
    return float(rounds * geo.slots_per_shard * geo.workers) / capacity


def _discarded_frac(buffer: Any, trainable: np.ndarray) -> float:
    valid = buffer.valid[: buffer.cycles]
    total = int(valid.sum())
    if total == 0:
        return 0.0
    return 1.0 - int(trainable.sum()) / total


def _truncation_steps(config: RunConfig) -> int | None:
    for spec in config.env.truncation:
        steps = spec.kwargs.get("max_steps")
        if steps is not None:
            return int(steps)
    return None


def _nan_guard_trips(result: Any) -> int:
    """How many of the update's own numbers came back non-finite."""
    values = (
        result.policy_loss,
        result.value_loss,
        result.entropy,
        result.kl,
        result.grad_norm_actor,
        result.grad_norm_critic,
    )
    return int(sum(1 for value in values if not math.isfinite(float(value))))


def _hash_tensors(digest: Any, mapping: Mapping[str, Any]) -> None:
    """Fold a state dict into a hash, name by name, in a fixed order."""
    for name in sorted(mapping):
        value = mapping[name]
        digest.update(name.encode("utf-8"))
        array = getattr(value, "detach", None)
        if array is None:
            digest.update(repr(value).encode("utf-8"))
            continue
        tensor = value.detach().to("cpu").contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).numpy().tobytes())


def _gpu_util() -> float:  # pragma: no cover - there is no GPU in the suite
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.utilization()) / 100.0
    except Exception:
        return 0.0


def _vram_peak_mb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated()) / 1e6  # pragma: no cover
    except Exception:  # pragma: no cover - torch is optional for everything but a run
        return 0.0


def _rss_peak_mb() -> float:
    """The parent's peak resident memory, where the platform will say.

    Reported as zero rather than guessed where it will not: the RAM ledger is what the doctor
    projects a run from, and a fabricated measurement beside it would be worse than a gap.
    """
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(peak) / 1e6 if peak > 1 << 20 else float(peak) / 1e3
    except ImportError:
        try:
            import ctypes
            import ctypes.wintypes as wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            if not ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                handle, ctypes.byref(counters), counters.cb
            ):
                return 0.0
            return float(counters.PeakWorkingSetSize) / 1e6
        except Exception:
            return 0.0
