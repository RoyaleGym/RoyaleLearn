"""The loop: the only place in the package where the phases of an iteration are ordered.

Everything else in this repository is a piece that can be replaced -- a codec, an estimator, a
matchmaker, a sink. This file is the sentence they are read in. It holds no policy of its own
beyond that ordering and the invariants that go with it, and it is the only module that knows
that a critic pass comes before GAE, that a batch is judged before the update learns from it,
that a gate belongs at an env-step cadence rather than an iteration one, and that a checkpoint
is written after the alarms have had their look at the row.

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
import sys
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
from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    DirCheckpointStore,
    check_described,
    check_resume,
)
from .config import RunConfig, config_hash, dump_config, geometry, validate
from .errors import PreflightError, RoyaleLearnError
from .identity import RunIdentity, compute_identity, describe_device
from .identity import run_id as run_id_of
from .ladder.actors import EvalActors
from .ladder.evaluate import EvalRunner, SeedSet, eval_seed_set
from .ladder.eviction import HallOfFameEviction
from .ladder.gate import WilsonGate, floor_decision
from .ladder.matchmaker import MixMatchmaker
from .ladder.pool import LEARNER_ID, LadderPool, is_learner, learner_probe_id
from .ladder.rating import BradleyTerryDavidsonRater, EloReadout
from .ladder.results import KIND_PROBE, ResultLog, context_digest
from .ladder.snapshots import DiskSnapshotStore, SnapshotSpec
from .metrics.alarms import AlarmSet
from .metrics.bundle import RatioOutlier, replay_episode, write_bundle
from .metrics.records import (
    IterationMetrics,
    episode_fields,
    ladder_fields,
    rollout_policy_fields,
    schedule_fields,
    update_fields,
)
from .metrics.sinks import METRICS_NAME, build_sinks
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
        ordinals = payload.get("ordinals", {})
        self.matchmaker.restore_ordinals(ordinals)
        # The same counters reach the workers, where they decide which episode each battle
        # starts. The matchmaker's copy decides who that episode is played against; without
        # both, a resumed run either plays the right battles against the wrong opponents or
        # the wrong battles against the right ones.
        self.source.ordinals = tuple(
            int(ordinals.get(str(battle), ordinals.get(battle, 0)))
            for battle in range(self.source.geometry.n_battles)
        )


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


def _is_publisher(value: Any) -> bool:
    """Is this a viewer publisher rather than a yes-or-no?

    Duck-typed on ``publish`` rather than imported and isinstance-d, because importing
    ``royalegym.viser`` here would pull the viewer into every process that builds a player,
    including the ones that never publish.
    """
    return value is not True and value is not False and hasattr(value, "publish")


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
        self.card_legal: dict[int, int] = {}
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
        slot_legal: list[np.ndarray] = []
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
                # Per hand slot rather than per action: a slot is playable when ANY of its tiles
                # is. This is what separates "the policy does not choose this card" from "this
                # card is rarely affordable", and nothing measured it before.
                slots_wide = mask[:, 1:].reshape(mask.shape[0], self.hand.hand_size, self.tiles)
                slot_legal.append(slots_wide.any(dim=-1).to("cpu").numpy())
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

        fields.update(
            self._plays(
                taken,
                np.concatenate(onehots, axis=0),
                np.concatenate(slot_legal, axis=0),
            )
        )
        return fields

    def _plays(
        self, actions: np.ndarray, onehot: np.ndarray, slot_legal: np.ndarray
    ) -> dict[str, MetricValue]:
        """Where the cards went, and which cards they were.

        The action index says the hand slot and the tile; the card in that slot is read out of
        the observation's own one-hot, so the counters are per card rather than per hand
        position -- which is the only one of the two that means anything across a cycle.
        """
        fields: dict[str, MetricValue] = {}
        fields.update(self._availability(onehot, slot_legal))
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
            # The share of plays is a share of a policy AND of an elixir bar, and a reader
            # cannot tell which they are looking at. Divided by the decisions where the card
            # was affordable, it is the policy alone.
            affordable = self.card_legal.get(card, 0)
            if affordable:
                fields[f"policy/card_play_rate/{card}"] = played_count / affordable
        return fields

    def _availability(self, onehot: np.ndarray, slot_legal: np.ndarray) -> dict[str, MetricValue]:
        """How often each card was in the hand at all, and how often it was affordable.

        Without these, a per-card play share answers a question nobody asked. A three-cost card
        is legal at a decision far more often than a four-cost one when the bar averages about
        one and a half elixir, so a policy that chooses uniformly among what it can afford still
        plays cheap cards many times more often. Reading that as a preference is reading the
        elixir economy as a policy.
        """
        block = self.hand.onehot_width
        rows, hand = slot_legal.shape
        wide = onehot[:, : hand * block].reshape(rows, hand, block)
        cards = wide.argmax(axis=-1)
        self.card_legal = {}
        fields: dict[str, MetricValue] = {}
        for card in np.unique(cards):
            in_hand = cards == card
            legal = int(np.count_nonzero(in_hand & slot_legal))
            self.card_legal[int(card)] = legal
            fields[f"policy/card_in_hand_frac/{int(card)}"] = float(
                np.count_nonzero(in_hand.any(axis=-1)) / rows
            )
            fields[f"policy/card_legal_frac/{int(card)}"] = float(
                np.count_nonzero((in_hand & slot_legal).any(axis=-1)) / rows
            )
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
        viser: Any = False,
    ) -> None:
        self.spec = spec
        self.factory_spec = factory_spec
        self.actors = actors
        self.master_seed = int(master_seed)
        self.extra_modules = tuple(extra_modules)
        self.device = device
        self.release_mode = release_mode
        self.max_decisions = int(max_decisions)
        #: Whether this battle carries the viewer's state stream, or the publisher that carries
        #: it. One vec env in a process may: the publisher binds one fixed UDP port, so a second
        #: would be an OSError.
        #:
        #: A ``ViserPublisher`` here is used as given. That is how a caller outside this repo
        #: puts a publisher of its own in the path -- one that PACES, for instance. A battle is
        #: simulated far faster than it is played, and the viewer keeps only the newest datagram,
        #: so an unpaced stream is sampled and the battle jumps; the fix is a publisher whose
        #: publish sleeps to the next slot. Pacing is a property of watching rather than of
        #: playing, so it belongs to whoever is watching and not here. Anything else is read as
        #: a bool and means what it meant before: build the publisher from the environment.
        self.viser = viser if _is_publisher(viser) else bool(viser)
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
                given = self.viser if _is_publisher(self.viser) else "env"
                self._vec = self.factory_spec.build_vec(1, self.extra_modules, viser=given)
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
                agents[seat]: int(policies[seat](obs[agents[seat]], float(uniforms[seat]), rng))
                for seat in (0, 1)
            }
            obs, _reward, terminated, truncated, infos = env.step(actions)
            if terminated[agents[a_seat]] or truncated[agents[a_seat]]:
                outcome = int(infos[agents[a_seat]].get("outcome", 0) or 0)
                break
        self.battles += 1
        return 0.5 + 0.5 * float(np.sign(outcome))


# ---------------------------------------------------------------------------
# Interactive control
# ---------------------------------------------------------------------------


#: The encodings a reader's shell actually writes, in the order they are tried. `utf-8-sig`
#: FIRST because it is the plain-UTF-8 decoder plus a BOM it discards, so it covers both the
#: Python-written file and every PowerShell redirection: `>`, `Set-Content` and
#: `Out-File -Encoding utf8` all emit a BOM on 5.1, and `﻿` is not whitespace, so `.strip()`
#: left it and the first character of the file was the BOM rather than the letter.
#: `Out-File -Encoding unicode` is UTF-16, which is not UTF-8 at all.
CONTROL_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-16", "latin-1")


def _control_letter(raw: bytes) -> str:
    """The first letter of a control file, whichever documented shell wrote it.

    ``latin-1`` last as a decoder that cannot fail, so a file of arbitrary bytes gives a letter
    that is simply not one of the three rather than an exception. A control file is a
    convenience and must not be able to end a run: a UnicodeDecodeError is a ValueError with no
    OSError in it, so before 2026-09-23 a UTF-16 file escaped this method's guard, went through
    ``_collect`` and ``_iterate``, and reached ``learn``.
    """
    for encoding in CONTROL_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        return text.strip().lstrip("﻿").strip()[:1].lower()
    return ""  # pragma: no cover - latin-1 decodes every byte string


class _Control:
    """``c`` checkpoint, ``q`` checkpoint and quit, ``p`` pause, read once per round.

    A sentinel file and a signal handler, so that a run needs neither a terminal attached nor a
    thread spinning on one. The file is read and removed, so a letter acts once; the handler
    turns Ctrl-C into ``q``, which is a checkpoint rather than a lost afternoon.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        install_signal: bool = True,
        printer: Callable[[str], None] | None = None,
    ) -> None:
        self.path = run_dir / CONTROL_FILE
        self.pending: str = ""
        self.resume = threading.Event()
        self.printer = printer or (lambda _line: None)
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
            raw = self.path.read_bytes()
            self.path.unlink()
        except OSError:
            return ""
        letter = _control_letter(raw)
        if letter in ("c", "q", "p"):
            return letter
        if raw.strip():
            # Said rather than swallowed. The file is consumed either way -- a letter has to act
            # once -- and the difference is whether anybody learns why nothing happened. This is
            # the half that made the BOM invisible for as long as it was there.
            self.printer(
                f"control file held {raw[:16]!r}, which is not c, q or p; ignored. "
                "Write one letter, and Ctrl-C always works"
            )
        return ""

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


def _refuse_an_occupied_run_dir(run_dir: Path) -> None:
    """Refuse a FRESH start into a directory that already holds a run.

    The directory follows the config's identity, so launching the same config fresh again lands
    in the same place, and it used to start at iteration 1 and write over the checkpoints there:
    train-hog26-10's directory holds metric rows for 1-610, then 1-45, then 1-21, then 611-621,
    under a checkpoint index that mixes launches. There is no flag to allow it, because it cannot
    be made safe, only destructive; the two things a person could want are both named below.
    """
    rows = run_dir / METRICS_NAME
    checkpoints = run_dir / "checkpoints"
    holds_rows = rows.is_file() and rows.stat().st_size > 0
    holds_checkpoints = checkpoints.is_dir() and any(checkpoints.iterdir())
    if not (holds_rows or holds_checkpoints):
        return
    held = [
        what
        for what, present in (("metric rows", holds_rows), ("checkpoints", holds_checkpoints))
        if present
    ]
    raise PreflightError(
        f"{run_dir} already holds a run ({' and '.join(held)}). A fresh start here would begin "
        "at iteration 1 and write over its checkpoints.\n"
        f"  To carry that run on:  royalelearn resume --run {run_dir}\n"
        "  To start a separate run: give it another run_name or runs_dir."
    )


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
        run_dir: str | Path | None = None,
    ) -> None:
        self.config = validate(config)
        self.printer = printer or (lambda _line: None)
        self.geometry = geometry(self.config)
        self.codec_path = codec
        self.resume_from = Path(resume) if resume is not None else None
        #: A directory to open INSTEAD of the one this configuration's identity names. A run
        #: directory is ``<run_name>-<run_id>`` and the run id is a hash of the identity, which
        #: includes the commit of every repository. So a tool pointed at a run that started
        #: before the code moved computed a DIFFERENT id, made an empty directory beside the real
        #: one and evaluated nothing: `royalelearn eval --run <a live run>` could not reach the
        #: run it was given, which is why the train session reads episodes.jsonl with its own
        #: tool instead. When a caller names a directory, that directory is the answer. The
        #: identity is still checked, separately and by name: the directory says WHERE and the
        #: identity says WHETHER.
        self.given_run_dir = Path(run_dir) if run_dir is not None else None
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
        self.last_gate_seconds = 0.0
        self.gate_seconds_total = 0.0
        #: The last probe of the live policy, by rung. Emptied at the top of every iteration, so
        #: a row publishes a score only when that iteration measured one: a probe's numbers are
        #: about the weights it played, and the weights move every iteration.
        self.last_rungs: dict[str, Any] = {}
        self.rung_seconds_total = 0.0
        self._entered = False
        self._closed = False
        self._last_checkpoint_step = 0
        self._last_checkpoint_seconds = 0.0
        self._last_candidate_step = 0
        self._last_floor_step = 0
        #: True from the update's first change to the learner until the row that reports the
        #: update is written. Inside that window no metric row describes the learner in memory,
        #: and ``_emergency`` reads this to know it must not save it.
        self._learner_ahead_of_rows = False
        #: The env step and folder of the last checkpoint this process wrote or resumed from.
        self._saved_at: tuple[int, Path] | None = None

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

        self._vram_gate(torch)

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
        self.run_dir = self.given_run_dir or run_directory(config, self.run_id)
        if self.resume_from is None:
            _refuse_an_occupied_run_dir(self.run_dir)
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
            segment_name=f"{buffer_segment_name(self.run_id)}-{os.getpid():x}-{next(_SEGMENTS)}",
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
        self.sinks.open(identity=self.identity, config_json=self.config_json, run_dir=self.run_dir)
        self.alarms = AlarmSet(config.alarms, ratio_atol=self._ratio_atol(), printer=self.printer)
        self.store = DirCheckpointStore(self.run_dir, keep=config.checkpoint.keep)
        self.components = self._components()
        self.control = _Control(
            self.run_dir,
            install_signal=self.install_signal_handler,
            printer=self.printer,
        )
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
            getattr(self, "eval_player", None),
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

        # The workers load the engine themselves, so they are held to the binary the identity
        # names; the inline source runs in this process and loaded the file that was measured.
        return ProcessRolloutSource(
            self.config,
            self.spec,
            self.report.table,
            expected_engine_binary=self.identity.engine_build.binary_sha256,
            **kwargs,
        )

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
        from .identity import env_spec_digest_of

        self.context = context_digest(
            # By value, like the identity: a config that spells the same environment differently
            # must file its games under the same context, or a resumed run loses its own ladder.
            env_spec_digest_of(config),
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
        eval_env = (
            config.eval_env
            if config.eval_env is not None
            else msgspec.structs.replace(config.env, truncation=[])
        )
        self.eval_env_spec = eval_env
        # One object rather than three methods, so that a process which only wants to PLAY
        # evaluation battles can build it without a coordinator behind it. `model` is the live
        # learner itself and not a copy: the probe's question is what the policy does NOW.
        self.eval_actors = EvalActors(
            self.spec,
            config.net,
            self.snapshot_store,
            device=self.device,
            release_mode=config.ladder.release_mode,
            model=self.model,
        )
        self.player = EnvBattlePlayer(
            self.spec,
            eval_env,
            self.eval_actors,
            master_seed=config.master_seed,
            extra_modules=tuple(config.extra_component_modules),
            device=self.device,
            release_mode=config.ladder.release_mode,
        )
        # The player the GATE uses. `rollout.eval_workers` was accepted, recorded in the run
        # identity and read by nothing until 2026-09-23 -- the same defect `rollout.overlap` had.
        # A gate is 2,200 battles at a measured 8.02 s each, so this is the difference between a
        # gate that costs 4.9 hours and one that costs 4.9/N. The probe keeps the parent's player:
        # it plays the LIVE weights, which no worker has.
        self.eval_player: Any = self.player
        if config.rollout.eval_workers > 1:
            self.eval_player = self._eval_farm(config, eval_env)
        self.eval_runner = _RecordingEvalRunner(
            self.eval_player,
            self.seeds,
            master_seed=config.master_seed,
            context=self.context,
            run_id=self.run_id,
            log=self.results,
            bootstrap_resamples=config.ladder.gate.bootstrap_resamples,
            release_mode=config.ladder.release_mode,
            obs_digest=self._snapshot_obs_digest,
        )
        # A second runner over the same player, the same seeds and the same log. It is separate
        # for two reasons. Its results carry ``kind="probe"`` and the kind belongs to the
        # runner, so that one object cannot write two cuts of the log. And the gate's runner
        # keeps its last comparison, which is what ``ladder/paired_rho`` reports: a probe
        # writing into that would publish the probe's correlation under a key whose description
        # says it is the one setting the gate's effective sample size.
        self.probe_runner = EvalRunner(
            self.player,
            self.seeds,
            master_seed=config.master_seed,
            context=self.context,
            run_id=self.run_id,
            log=self.results,
            bootstrap_resamples=config.ladder.gate.bootstrap_resamples,
            release_mode=config.ladder.release_mode,
            obs_digest=self._snapshot_obs_digest,
            kind=KIND_PROBE,
        )
        self.gate = WilsonGate(
            config.ladder.gate,
            self.rater,
            gates_dir=ladder_dir / "gates",
        )

    def _eval_farm(self, config: Any, eval_env: Any) -> Any:
        """A farm for the gate's battles, or the parent's player if one cannot be built.

        A FAILED START FALLS BACK RATHER THAN ENDING THE RUN, loudly. The gate path has never
        executed in this project's life, so the first run to reach one should not lose hours
        because a worker could not spawn; it should take 4.9 hours instead of 4.9/N and say why.
        A worker that fails DURING a batch is a different matter and raises: half a comparison
        from the parent and half from a worker would be one measurement from two sources.
        """
        from .ladder.farm import EvalFarm, EvalWorkerConfig

        try:
            worker_config = EvalWorkerConfig(
                spec=self.spec,
                net=config.net,
                env=eval_env,
                extra_modules=tuple(config.extra_component_modules),
                snapshot_root=str(self.run_dir / "snapshots"),
                template=self.snapshot_template,
                master_seed=config.master_seed,
                release_mode=config.ladder.release_mode,
                max_decisions=self.player.max_decisions,
                max_resident=config.ladder.max_resident_opponents + 2,
            )
            return EvalFarm(
                worker_config,
                workers=config.rollout.eval_workers,
                fallback=self.player,
                printer=self.printer,
            )
        except Exception as exc:  # pragma: no cover - a farm that cannot even be described
            self.printer(
                f"eval farm     not built ({type(exc).__name__}: {exc}); gates will play their "
                "battles in this process, which is about 8 s each"
            )
            return self.player

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
        if member in self.pool.anchors or is_learner(member):
            return self.spec.obs_digest
        try:
            return self.snapshot_store.obs_digest(member)
        except KeyError:
            return self.spec.obs_digest

    def _next_legal_minibatch(self) -> int | None:
        """The largest legal ``minibatch_size`` below the configured one, or None.

        Legal means it divides ``batch_size``, which is the invariant that keeps a minibatch a
        pure memory knob. It matters that this is computed rather than suggested: at the shipped
        ``batch_size`` of 4096 there is no divisor between 256 and 512, so "lower it" has exactly
        one answer and a reader should not have to find that out by trying.
        """
        ppo = self.config.ppo
        below = [n for n in range(1, ppo.minibatch_size) if ppo.batch_size % n == 0]
        return max(below) if below else None

    def _vram_gate(self, torch: Any) -> None:
        """Measure one minibatch's device peak and refuse a run that will not fit beside it.

        A projection would be a guess, and a guess about this is what put ``minibatch_size`` at
        512 in the first place. So the gate runs the real backward pass the update runs, at the
        real minibatch size, and reads what it cost.
        """
        headroom = self.config.doctor.vram_headroom_mb
        # THIS run's device, not the machine's. A CPU run on a machine that happens to hold a
        # card would measure a card it never touches: the peak reads near zero, the gate passes
        # whatever the run needs, and every health/vram_* key afterwards describes somebody
        # else's process. `cuda.is_available()` answers a question about the machine.
        if not headroom or not str(self.device).startswith("cuda"):
            return
        free_before, _total = torch.cuda.mem_get_info()  # pragma: no cover - no GPU in the suite
        torch.cuda.reset_peak_memory_stats()  # pragma: no cover
        try:  # pragma: no cover
            self._probe_backward(torch)
        except Exception as exc:  # pragma: no cover - a failure here is the update's to report
            # Not silent. The gate not running is survivable; what is not is that vram_needed_mb
            # is then never set, so `vram_spilling` -- which compares against it -- stays quiet
            # for the whole run and the row looks healthy because nothing measured it.
            self.printer(
                f"the VRAM gate did not run: the probe raised {type(exc).__name__}: {exc}. "
                "health/vram_needed_mb will be absent and the vram_spilling alarm cannot fire "
                "for this run"
            )
            return
        peak = torch.cuda.max_memory_reserved()  # pragma: no cover
        torch.cuda.reset_peak_memory_stats()  # pragma: no cover
        margin = headroom * 1_000_000  # pragma: no cover
        # Kept so every later row can be checked against it. The preflight guards one instant;
        # what it cannot see is another process taking memory at hour three, which is the same
        # silent slowdown arriving later. This is the only threshold in the harness measured on
        # the machine it will be applied to, minutes before it is applied.
        self.vram_needed_mb = (peak + margin) / 1e6  # pragma: no cover
        if peak + margin <= free_before:  # pragma: no cover
            return
        lower = self._next_legal_minibatch()  # pragma: no cover
        advice = (  # pragma: no cover
            f"lower ppo.minibatch_size to {lower} -- the next legal value, since it must divide "
            f"ppo.batch_size {self.config.ppo.batch_size}"
            if lower
            else "lower ppo.batch_size, or run on a larger device"
        )
        raise PreflightError(  # pragma: no cover
            f"one minibatch of {self.config.ppo.minibatch_size} peaked at "
            f"{peak / 1e6:.0f} MB of device memory and only {free_before / 1e6:.0f} MB was free, "
            f"leaving less than the {headroom} MB doctor.vram_headroom_mb requires.\n"
            f"  This platform does not refuse an oversubscribed allocation -- it backs it with "
            f"host memory over PCIe -- so the run would not fail here, it would be several times "
            f"slower for its whole life and nothing would say so.\n"
            f"  {advice}, or set doctor.vram_headroom_mb to 0 to run anyway."
        )

    def _probe_backward(self, torch: Any) -> None:
        """One forward and backward at the update's minibatch size, on zeros.

        Zeros are the right input: the question is what the SHAPES cost, and a mask of zeros
        would make the softmax degenerate, so the no-op column is set legal the way a real row's
        always is.
        """
        from .api.policy import ObsBatch

        spec, rows = self.spec, self.config.ppo.minibatch_size
        planes = spec.obs_space["mask_planes"].shape
        mask = torch.zeros((rows, spec.n_actions), dtype=torch.bool, device=self.device)
        mask[:, 0] = True
        batch = ObsBatch(
            spatial=torch.zeros((rows, *spec.spatial_shape), device=self.device),
            mask_planes=torch.zeros((rows, *planes), device=self.device),
            vector=torch.zeros((rows, spec.vector_size), device=self.device),
            mask=mask,
            # The probe measures what a real minibatch costs, and with card identity on a real
            # minibatch carries the id planes and their embedding.
            card_ids=(
                torch.zeros(
                    (rows, *spec.obs_space["card_ids"].shape), dtype=torch.int64, device=self.device
                )
                if "card_ids" in spec.obs_space
                else None
            ),
        )
        actions = torch.zeros((rows,), dtype=torch.int64, device=self.device)
        result = self.model.backprop(batch, actions)
        result.log_probs.sum().backward()
        self.model.zero_grad(set_to_none=True)

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
        """One whole iteration, and whatever the interactive control asked for during it.

        The batch is judged before anything learns from it. Every invariant, the probe's
        legality check among them, reads only what collection wrote, so asking first costs
        nothing -- and asking after the update is how a real run trained a refused batch of
        22,627 rows for 18 optimizer steps and then saved the result. From the update's first
        change until the row reporting it is written, the learner in memory is one no row
        describes; ``_emergency`` is what that window is marked for.
        """
        began = time.perf_counter()
        sched = self.schedules.state(
            iteration=self.iteration,
            cumulative_env_steps=self.cumulative_env_steps,
            cumulative_timesteps=self.cumulative_timesteps,
        )
        self.rng.iteration = self.iteration
        self.rng.shard_streams = self.rollout_component.shard_streams()

        plan = self.matchmaker.plan(self.iteration, self.pool, self.ratings, self.geometry)
        self.buffer.begin_iteration(plan, self.geometry.cycles)
        self.source.begin_iteration(plan, self.buffer, self.iteration)
        self.inference.begin_iteration(plan)

        collection = self._collect(plan, sched)
        episodes = collection["episodes"]
        self.recent_episodes = (self.recent_episodes + episodes)[-200:]

        trainable = self.buffer.trainable()
        self._check_iteration(collection, trainable, episodes)
        probe = self.probe.measure(
            self.buffer, self.inference.gather, self.buffer.action[: self.buffer.cycles], trainable
        )

        self._learner_ahead_of_rows = True
        result = self.update.step(self.buffer, sched)
        # "The values the last iteration actually ran at": set once it has run, so a checkpoint
        # taken before then does not report the schedule of an iteration that never finished.
        self.schedule_component.last = sched

        self.cumulative_timesteps += int(trainable.sum())
        self.cumulative_env_steps += self.geometry.cycles * self.geometry.n_battles
        self.iteration += 1
        self.wall_seconds += time.perf_counter() - began

        self._record_training_results(episodes)
        gate_seconds = self._maybe_gate()
        self._maybe_refit()
        rung_seconds = self._maybe_probe_rungs()

        row = self._row(
            sched=sched,
            result=result,
            episodes=episodes,
            collection=collection,
            trainable=trainable,
            probe=probe,
            iteration_seconds=time.perf_counter() - began,
            gate_seconds=gate_seconds,
            rung_seconds=rung_seconds,
            wall=time.perf_counter() - run_started,
        )
        self.rows.append(row)
        # Written before the alarms read it. A halt raises out of that read after checkpointing
        # the learner this row describes, and the row has to be in the metric file by then: the
        # file, not this process's memory, is what a checkpoint answers to.
        self.sinks.write(row)
        self.sinks.write_episodes(episodes)
        self._learner_ahead_of_rows = False
        # Through on_fired rather than the return value: a halt is raised from inside evaluate,
        # so anything written from what it returns is written for every iteration EXCEPT the one
        # that halted the run.
        self.alarms.evaluate(
            row,
            on_halt=self._halt_bundle,
            on_dump=self._dump_bundle,
            on_fired=self._record_alarms,
        )
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
                check_round(round_, cycle=cycle, slots=self.planner.round_slots(round_.shard))
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
                        opponent_ix=(opponent[slots] if assigned else np.zeros(0, dtype=np.int8)),
                        learner_seat=(
                            seat[self.planner.slot_battle[slots][::2]]
                            if assigned
                            else np.zeros(0, dtype=np.int8)
                        ),
                    )
                )
                buffer.record_round(round_, answer.actions, answer.log_probs)
                self._record_finals(source, round_)
                rounds += 1
                command = self.control.poll() or command
                if command == "p":
                    command = self.control.wait_while_paused(self.printer)
        for round_ in source.finish_iteration():
            # Not bookkeeping: the trailing round is where the last cycle's reward, done flags
            # and deploy status are, and row T-1 has no other source for them. It is not
            # counted as a round -- nothing was acted on it -- so the arithmetic that sizes the
            # iteration still describes the cycles that were.
            buffer.record_round(round_, None, None)
            self._record_finals(source, round_)
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

    def _record_finals(self, source: Any, round_: RolloutRound) -> None:
        """Hand the update the observation each truncation of this round was cut off in.

        Against the row that was truncated, which is one below the round that reported it: the
        truncation flag and the value the bootstrap comes from have to be the same cell, or the
        estimator bootstraps a row from a state it never reached.
        """
        if round_.cycle <= 0:
            return
        final_slots, final_rows = source.final_rows(round_)
        self.update.record_final_observations(round_.cycle - 1, final_slots, final_rows)

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
            assignments_constant_within_episodes(buffer.group[: buffer.cycles], buffer.episode_end)
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
        candidate = self.pool.issue_candidate_id()
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

    def _maybe_probe_rungs(self) -> float:
        """Play the LIVE policy against the fixed rungs, on its own cadence in iterations.

        What it is: the one measurement in a run that is about the policy being trained. Every
        other evaluation is a frozen snapshot against something, because the gate hands
        ``snap:v{n}`` to the runner, so without this a run publishes no score for the thing it
        is changing.

        What it is not: the gate. Nothing here admits, promotes or evicts anything, and the
        games are written under ``kind="probe"`` so that the authoritative fit, the gate's
        anchor reference and ``ladder/eval_games_total`` do not see them. Two reasons for that,
        one statistical and one about the scale. The rungs are fixed and the learner is not, so
        a probe is a measurement against a ruler rather than a game between two rated players;
        and each probe is its own id, so pooling them would add a column per probe, every one
        of them with too few games to place, to the fit every ladder decision is made on.

        The cost is real and is why the default is off: a probe plays ``probe_games`` battles
        per rung, one at a time, in the parent, between the update and the metric row.
        """
        config = self.config.ladder
        self.last_rungs = {}
        every = config.probe_every_iterations
        if every <= 0 or self.iteration % every:
            return 0.0
        started = time.perf_counter()
        member = learner_probe_id(self.cumulative_env_steps)
        for opponent in config.probe_opponents:
            self.last_rungs[opponent] = self.probe_runner.compare(
                member, opponent, games=config.probe_games, iteration=self.iteration
            )
        seconds = time.perf_counter() - started
        self.rung_seconds_total += seconds
        scores = ", ".join(
            f"{opponent.split(':', 1)[-1]} {comparison.score_a:.3f}"
            for opponent, comparison in self.last_rungs.items()
        )
        self.printer(f"probe         {member}: {scores} in {seconds:.1f}s")
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
        (folder / f"{self.iteration:06d}.json").write_bytes(msgspec.json.encode(self.ratings))

    # -- the row -------------------------------------------------------------

    def _row(
        self,
        *,
        sched: ScheduleState,
        result: Any,
        episodes: Sequence[EpisodeRecord],
        collection: Mapping[str, Any],
        trainable: np.ndarray,
        probe: Mapping[str, MetricValue],
        iteration_seconds: float,
        gate_seconds: float,
        rung_seconds: float,
        wall: float,
    ) -> dict[str, MetricValue]:
        """One iteration as one flat row, merged from all three sources.

        ``probe`` is the policy probe's measurement, taken before the update because its
        legality check is one of the invariants that decide whether the update runs at all.
        """
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
            "time/codec": float(source_stats.get("codec_ms", 0.0)) * collection["rounds"] / 1000.0,
            "time/ipc": ipc_seconds,
            "time/critic_pass": float(result.critic_pass_seconds),
            "time/gae": float(result.gae_seconds),
            "time/update": update_seconds,
            "time/checkpoint": self._take_checkpoint_seconds(),
            "time/gate": gate_seconds,
            "time/probe": rung_seconds,
            # No "time/overlap_saved": it was a hardcoded 0.0, which reads as "overlap saved
            # nothing this iteration" rather than "there is no overlap". rollout.overlap is
            # refused until spec 14.1's driver exists, so nothing can fill it.
        }
        # The probe is in the attributed sum rather than in the residual. It plays battles one
        # at a time in the parent, so at the cadences worth running it is minutes, and a
        # residual carrying minutes nobody can name is how a run's wall clock stops adding up.
        attributed = sum(
            float(metrics.time[key])
            for key in (
                "time/collection",
                "time/update",
                "time/checkpoint",
                "time/gate",
                "time/probe",
            )
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
            **_optional("throughput/gpu_util_frac", _gpu_util()),
        }

        metrics.ppo = dict(update_fields(result))
        metrics.ppo["ppo/advantage_std_pre_norm"] = float(self.update.advantage_std_pre_norm)
        metrics.ppo["ppo/advantage_std_choice_pre_norm"] = float(
            getattr(self.update, "advantage_std_choice_pre_norm", 0.0)
        )
        metrics.ppo["ppo/advantage_mean_choice"] = float(
            getattr(self.update, "advantage_mean_choice", 0.0)
        )
        stats_of = self.update.advantage_stats
        metrics.ppo["ppo/return_running_mean"] = (
            float(stats_of.raw_return_mean) if stats_of else 0.0
        )
        metrics.ppo["ppo/return_running_std"] = float(stats_of.raw_return_std) if stats_of else 0.0
        metrics.ppo["ppo/reward_clip_frac"] = (
            float(stats_of.clipped_reward_frac) if stats_of else 0.0
        )
        metrics.ppo["ppo/lr_backoff_events"] = self.schedules.backoff.events

        aggregate = episode_fields(episodes, truncation_steps=_truncation_steps(self.config))
        metrics.env = {
            key: value for key, value in aggregate.fields.items() if key.startswith("env/")
        }
        metrics.policy = {
            key: value for key, value in aggregate.fields.items() if key.startswith("policy/")
        }
        # What the policy did while it was choosing, as opposed to what the optimizer saw. These
        # sums were accumulated at every rollout forward and read by nobody until 2026-09-22.
        metrics.policy.update(rollout_policy_fields(stats))
        for key, value in probe.items():
            (metrics.env if key.startswith("env/") else metrics.policy)[key] = value
        metrics.env.setdefault("env/illegal_action_rate", 0.0)
        # The per-episode group is filled ONLY when an episode finished. ``alarms.py`` states
        # the rule these two lines used to break: a missing key is never a firing, because a row
        # from an iteration in which no episode finished carries no ``env/`` group at all and an
        # alarm reading that absence as a zero would halt a healthy run. Filling the group with
        # defaults, and ``cards_per_match`` with 0.0, handed the alarms exactly the zeros the
        # rule exists to keep away from them.
        #
        # It had already happened. 74 of the 124 metric rows on disk completed zero episodes and
        # every one of them carries cards_per_match = 0.0; on one run ``noop_collapse`` fired at
        # iterations 7, 9 and 10 reporting a policy that played no cards in iterations where no
        # episode ended, and ``noop_collapse_severe`` -- a HALT with patience 5 -- held at
        # iterations 1 to 4 and broke at 5 only because the value happened to read exactly 3.0
        # against a strict ``< 3.0``. One iteration from halting a healthy run on a sentinel.
        if metrics.env.get("env/episodes_completed"):
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
                rungs=self.last_rungs or None,
                # Absent until a probe has run, rather than a zero that reads as "probing is
                # free" on a run that never probes.
                probe_seconds_frac=(
                    self.rung_seconds_total / max(1e-9, wall) if self.rung_seconds_total else None
                ),
            )
        )

        failures = getattr(self, "failures_by_kind", {})
        housekeeping = _housekeeping_counts(self)
        metrics.health = {
            "health/illegal_action_rate": float(metrics.env["env/illegal_action_rate"]),
            "health/mask_disagreements": 0,
            "health/worker_restarts": int(sum(getattr(self.source, "restarts", []) or [0])),
            "health/rows_dropped_dead_worker": int((~buffer.valid[: buffer.cycles]).sum()),
            "health/obs_codec_clipped": int(getattr(self.codec, "clipped", 0)),
            "health/samples_unused_frac": float(result.samples_unused_frac),
            "health/nan_guard_trips": _nan_guard_trips(result),
            "health/vram_peak_mb": _vram_peak_mb(),
            **_vram_regime(getattr(self, "vram_needed_mb", None), device=str(self.device)),
            **_optional("health/rss_peak_mb", _rss_peak_mb()),
            "health/buffer_fill_frac": _fill_frac(buffer, collection["rounds"], geo),
            "health/housekeeping_failures": sum(housekeeping.values()),
        }
        for kind, n in sorted(failures.items()):
            metrics.health[f"health/worker_failures/{kind}"] = int(n)
        # The total is always present, like its neighbours, so a zero is a measurement. The
        # breakdown appears only when there is something to break down, which is the same shape
        # ``worker_failures`` uses one line above.
        for kind, n in sorted(housekeeping.items()):
            if n:
                metrics.health[f"health/housekeeping/{kind}"] = int(n)
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
        return bool(every > 0 and self.cumulative_env_steps - self._last_checkpoint_step >= every)

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
        self._saved_at = (self.cumulative_env_steps, path)
        self.store.prune(keep=self.config.checkpoint.keep)
        self._last_checkpoint_step = self.cumulative_env_steps
        self._last_checkpoint_seconds = time.perf_counter() - started
        self.printer(f"checkpoint    {path}")
        return path

    def _load(self, path: Path) -> None:
        """Resume from a checkpoint: refuse an identity that moved, or a learner the run's own
        record does not contain, then restore everything."""
        assert self.identity is not None
        manifest = msgspec.json.decode((Path(path) / "manifest.json").read_bytes(), type=Manifest)
        drift = check_resume(
            manifest,
            self.identity,
            msgspec.json.decode(self.config_json.encode("utf-8")),
            allow_drift=self.allow_identity_drift,
        )
        del drift
        # The record is the one beside the checkpoint (``<run>/checkpoints/<step>``), not this
        # coordinator's: a resume under a drifted identity writes to a new run directory.
        check_described(manifest, Path(path).parent.parent / METRICS_NAME)
        from .identity import identity_differences

        differences = identity_differences(manifest.identity, self.identity)
        if differences and self.allow_identity_drift:
            self.resumed_with_drift = True
            (self.run_dir / "drift.json").write_bytes(
                msgspec.json.encode(
                    {name: [str(was), str(now)] for name, (was, now) in differences.items()}
                )
            )
        self.store.read(Path(path), self.components, strict=self.config.checkpoint.strict_load)
        self.iteration = manifest.iteration
        self.cumulative_env_steps = manifest.cumulative_env_steps
        self.cumulative_timesteps = manifest.cumulative_timesteps
        self.wall_seconds = manifest.wall_seconds
        self._last_checkpoint_step = manifest.cumulative_env_steps
        self._saved_at = (manifest.cumulative_env_steps, Path(path))
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

    def _record_alarms(self, fired: list[AlarmResult]) -> None:
        """Everything that fired this iteration, to the run's own file and to the bundle's list.

        Called from inside ``AlarmSet.evaluate`` and before the halt, because a halting
        iteration's alarms are the ones worth having.
        """
        self.alarm_rows.extend(fired)
        self.sinks.write_alarms(fired)

    def _halt_bundle(self, alarm: AlarmResult) -> str | None:
        """A checkpoint and a diagnostic bundle, in that order, before the halt is raised.

        THE CHECKPOINT WAS WRITTEN UNDER A BLANKET SUPPRESS. It is the save an operator is most
        likely to reach for, because it is the state the run stopped in, and swallowing its
        failure left them believing they had it: the bundle still appeared, the halt still raised
        with its own message, and nothing anywhere said the file was not written. It still must
        not raise -- that would replace the alarm with a disk error as the reason the run
        stopped -- so the failure rides in the bundle's note and in the run log instead.
        """
        note = ""
        try:
            self._checkpoint()
        except Exception as exc:
            note = (
                f"{alarm.message}\n\nTHE HALT CHECKPOINT DID NOT WRITE: "
                f"{type(exc).__name__}: {exc}. There is no save at this iteration; the newest "
                "one is whatever the run wrote before it."
            )
            self.printer(f"checkpoint    FAILED at the halt: {type(exc).__name__}: {exc}")
        return self._dump_bundle(alarm, note=note)

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
        """The last thing a crashing run does: save the learner its last row describes, or say
        why there is nothing honest to save.

        A checkpoint answers to the metric file: its weights, moments and counters are ones a
        row reports. From the update's first change until the row reporting it is written that
        stops being true -- partly trained if the update itself raised, trained and unreported
        if something after it did -- and a save would put weights no row describes under the
        previous row's counters. So inside that window nothing is written and the checkpoint to
        resume from is named instead. What the update had done since is lost, which is the
        trade the checkpoint cadence already sets.

        Anywhere else the learner is exactly the last row's, but collection has moved each
        battle's ordinal past the episodes that finished in it, and the save carries those.
        That is consistent for resume by the mechanism every checkpoint already relies on
        (``_RolloutComponent``): the ordinals say which battles are played next, so a resumed
        run plays new ones, none twice, and the refused iteration's episodes are skipped rather
        than trained on. The one other thing that moves while collecting is a restarted
        worker's generation, which the save carries the same way; this package draws from no
        global random stream, so no other stream has moved. When a checkpoint of this same
        learner is already on disk -- the periodic one, a halt's, the one the run resumed from --
        it is kept: it was written at the iteration boundary, so it can still replay the
        iteration that failed, and a copy whose battles have skipped ahead would add nothing and
        lose that.
        """
        self.printer(f"stopping after {type(exc).__name__}: {exc}")
        refusal = self._emergency_refusal()
        if refusal:
            self.printer(f"no emergency checkpoint: {refusal}")
            return
        try:
            self._checkpoint()
        except Exception as inner:  # pragma: no cover - the save is best effort
            self.printer(f"the emergency checkpoint failed: {inner}")

    def _emergency_refusal(self) -> str:
        """Why an emergency save now would misstate the run or only repeat it; empty if neither."""
        if self._learner_ahead_of_rows:
            resume = (
                f"the checkpoint at {self._saved_at[1]}"
                if self._saved_at is not None
                else "nowhere: this run has written no checkpoint yet"
            )
            return (
                "the update had started changing the learner and no metrics row describes it "
                f"yet; resume from {resume}"
            )
        if self._saved_at is not None and self._saved_at[0] == self.cumulative_env_steps:
            return f"the checkpoint at {self._saved_at[1]} already holds this learner"
        return ""

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
        counts.setdefault((record.worker, record.shard, record.battle, record.ordinal), []).append(
            record
        )
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


def _housekeeping_counts(run: Any) -> dict[str, int]:
    """Failures from the actions that retry silently, by source.

    THESE COUNTERS EXISTED AND NOTHING READ THEM. ``prune_failures``,
    ``compaction_failures`` and ``terminate_failures`` were each added with a test proving they
    increment, and each then lived on an object that is discarded at the end of the run. A counter
    whose only reader is the test that proves it counts is not a channel -- it is the same silence
    with a variable in it, and all three guard actions whose success and failure otherwise look
    identical: a directory that would not delete, a file that would not rename, a worker that
    would not die.

    Absent sources read zero rather than raising, because this is called while building a row and
    a health metric that can break the row it is reporting on is worse than no metric.
    """
    counts = {
        "prune": int(getattr(getattr(run, "store", None), "prune_failures", 0) or 0),
        "eval_shutdown": int(
            getattr(getattr(run, "eval_player", None), "terminate_failures", 0) or 0
        ),
    }
    sinks = getattr(run, "sinks", None)
    compaction = int(getattr(sinks, "compaction_failures", 0) or 0)
    for sink in getattr(sinks, "sinks", ()) or ():
        compaction += int(getattr(sink, "compaction_failures", 0) or 0)
    counts["compaction"] = compaction
    return counts


def _optional(key: str, value: float | None) -> dict[str, MetricValue]:
    """``{key: value}`` when there is a value, and nothing at all when there is not.

    The row's rule, kept in one place: a metric nobody measured is absent, because a neutral
    value is a claim and a reader cannot tell one from a measurement.
    """
    return {} if value is None else {key: value}


def _gpu_util() -> float | None:  # pragma: no cover - there is no GPU in the suite
    """How busy the device was, or None when nobody could say.

    It returned 0.0 on every path, including the two that are not measurements: no CUDA device,
    and the reading itself failing. A run on a card at 92% published "0.0" for the whole of its
    life because ``torch.cuda.utilization`` raises without pynvml installed, and an idle GPU is
    exactly what a reader looking for a bottleneck would conclude from it. The exception is also
    printed once rather than swallowed, because "pynvml is not installed" is a thing somebody can
    fix in a minute and a silent zero is not.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.utilization()) / 100.0
    except Exception as exc:
        global _GPU_UTIL_COMPLAINED
        if not _GPU_UTIL_COMPLAINED:
            _GPU_UTIL_COMPLAINED = True
            print(f"throughput/gpu_util_frac is not being recorded: {type(exc).__name__}: {exc}")
        return None


#: Said once per process. A line per iteration about a metric nobody can read is noise.
_GPU_UTIL_COMPLAINED = False


def _vram_regime(
    needed_mb: float | None = None, *, device: str | None = None
) -> dict[str, MetricValue]:
    """The three numbers that say which memory regime a slowing update is in, plus retries.

    Read together at the end of an iteration they separate reservation growth, fragmentation
    inside a fixed reservation, an outside holder, and no memory mechanism at all. Reported as
    zeros without a device, like every other health figure here, because a gap is better than a
    guess and the doctor projects a run from these.
    """
    # Before torch is even imported: a run on the CPU has no device readings to take, and
    # taking the machine's would publish another process's memory under this run's name.
    if device is not None and not str(device).startswith("cuda"):
        return {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        stats = torch.cuda.memory_stats()  # pragma: no cover - no GPU in the suite
        free, _total = torch.cuda.mem_get_info()  # pragma: no cover
        reserved = float(stats.get("reserved_bytes.all.current", 0))  # pragma: no cover
        fields = {  # pragma: no cover
            "health/vram_reserved_mb": reserved / 1e6,
            "health/vram_inactive_split_mb": (
                float(stats.get("inactive_split_bytes.all.current", 0)) / 1e6
            ),
            "health/vram_driver_free_mb": float(free) / 1e6,
            # What COULD be mine: free space plus what I am already holding. This is the
            # quantity the run's health turns on, and the raw free figure is not -- a caching
            # allocator that has finished growing sits at or near zero free as its normal steady
            # state, so a threshold on that alone fires on healthy runs and stays quiet on sick
            # ones. Measured 2026-09-22: driver free read 0 on every iteration of a run that was
            # entirely well.
            "health/vram_available_mb": (float(free) + reserved) / 1e6,
            "health/vram_alloc_retries": int(stats.get("num_alloc_retries", 0)),
        }
        if needed_mb is not None:  # pragma: no cover
            fields["health/vram_needed_mb"] = float(needed_mb)
        return fields  # pragma: no cover
    except Exception:  # pragma: no cover - torch is optional for everything but a run
        return {}


def _vram_peak_mb() -> float:
    """Peak device memory since the previous call, which is what the schema promises.

    The counter behind this is ``max_memory_allocated``, a high-water mark that torch never
    lowers on its own, so reading it without resetting reports the largest allocation the
    PROCESS ever made and repeats that number every iteration afterwards. Measured 2026-09-22:
    six runs of different shapes all reported 3930.72, identical to five decimals, because each
    had reached that mark once and the row could no longer fall. An iteration that halved the
    minibatch would have read unchanged, and the comparison that found minibatch 256 several
    times faster than 512 would have concluded memory was not involved.

    Resetting here is what makes the key's own description true. Note the unit: this is decimal
    MB against a card quoted in MiB, so 3930.72 is 91.5% of a 4096 MiB device, not 96%.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        peak = float(torch.cuda.max_memory_allocated()) / 1e6  # pragma: no cover
        torch.cuda.reset_peak_memory_stats()  # pragma: no cover
        return peak  # pragma: no cover
    except Exception:  # pragma: no cover - torch is optional for everything but a run
        return 0.0


#: The POSIX platforms whose ``ru_maxrss`` is in BYTES. Everywhere else it is kilobytes, which
#: is Linux and therefore this project's CI. The list is short and knowable; the magnitude test
#: it replaced was not, and it was wrong by a factor of 1000 for exactly the processes worth
#: measuring -- see ``_rss_megabytes``.
RSS_IN_BYTES: tuple[str, ...] = ("darwin",)


def _rss_megabytes(peak: int, platform: str) -> float:
    """``ru_maxrss`` in megabytes, given which POSIX system produced it.

    Kilobytes on Linux, bytes on macOS and the BSDs. Until 2026-09-23 this decided between them
    by MAGNITUDE -- bytes above 1 GiB, kilobytes below -- which is right for a small process on
    Linux and wrong for a big one: a 2 GB parent reports 2,097,152 KiB, which is above the
    threshold, and was published as 2 MB. One thousandth of the truth, in the only measured RSS
    the harness has, on the platform its CI runs, about the resource that ends runs here.

    The platform is knowable. The size was a guess that failed at the size that mattered.
    """
    if platform.startswith(RSS_IN_BYTES):
        return float(peak) / 1e6
    return float(peak) * 1024.0 / 1e6


def _rss_peak_mb() -> float | None:
    """The parent's peak resident memory, where the platform will say, and None where it will not.

    Absent rather than guessed: the RAM ledger is what the doctor projects a run from, and a
    fabricated measurement beside it would be worse than a gap. It used to answer 0.0 for "this
    platform cannot say", which is a measurement a planner would read.

    The Windows path needs its signatures DECLARED, which is why it read zero on this machine for
    the whole life of the project. ``GetCurrentProcess`` returns the pseudo-handle
    0xFFFFFFFFFFFFFFFF, and ctypes assumes a C ``int`` return and an ``int`` argument for
    functions nobody has told it about, so the handle arrived at ``GetProcessMemoryInfo``
    truncated to 32 bits and the call failed. Every failure then became 0.0. Measured after
    declaring them: 13.9 MB for a bare interpreter and 31.7 MB for one holding this package,
    against 0.0 before.

    Which line is load-bearing, since it is not the obvious one: passing the handle as an explicit
    ``c_void_p`` is what fixes it. Removing only the ``restype`` declaration leaves the test green,
    because ctypes reads the truncated -1 back and ``c_void_p(-1)`` is the same pseudo-handle
    again. The plant that goes red is the call written as it was, with neither the wrapper nor
    ``argtypes``.
    """
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return _rss_megabytes(peak, sys.platform)
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
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.GetCurrentProcess.argtypes = []
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_Counters),
                wintypes.DWORD,
            ]
            handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return None
            return float(counters.PeakWorkingSetSize) / 1e6
        except Exception:
            return None
