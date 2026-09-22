"""The run's rectangle: which battle and seat every slot is, who holds it, and the seed
streams that follow from that map.

A slot is one seat of one battle, held for the whole run. The map from slot to
``(worker, shard, battle, seat)`` is fixed at the first iteration and never moves again,
which is what lets the experience buffer be a plain two-dimensional array and the composition
of an inference batch be a function of the slot index rather than of which worker answered
first.

The map is the obvious one, and it is obvious on purpose: battle ``b`` is slots ``2b`` (blue)
and ``2b + 1`` (red), which is the order ``ClashSelfPlayVecEnv`` already batches its seats in,
and battles are numbered worker-major and then shard-major. One shard's slots are therefore a
contiguous ascending range, so a shard-round writes a contiguous block of the rectangle and
needs no scatter.

Every random draw a rollout makes is addressed from here, by name, through
``royalelearn.seeding``: the shard's one env seed, the scripted opponents' generators, the
matchmaker's per-episode draw and the uniforms that drive action sampling. Nothing in the
rollout path calls a global generator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..api.rollout import (
    GROUP_LEARNER,
    ROLE_MIRROR,
    Assignment,
    SlotPlan,
)
from ..seeding import (
    ACT_CYCLE,
    ENV_SHARD,
    MATCH_BATTLE,
    SCRIPTED_SLOT,
    derive_generator,
    derive_int,
    stream_path,
)
from .layout import PLAN
from .scripted import SCRIPTED_NAMES

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..config import Geometry

__all__ = [
    "PLAN_DTYPE",
    "SlotPlanner",
    "opponent_index",
    "read_plan",
    "write_plan",
]

#: The plan region as a structured array, so that a worker reads its whole opening table in one
#: view instead of unpacking a record per slot. The offsets are the record's; the itemsize is
#: checked against it below, so a change to either is a failure here rather than a silent
#: misread in a child process.
PLAN_DTYPE = np.dtype(
    {
        "names": [f.name for f in PLAN.fields],
        "formats": ["<u8", "i1", "i1", "i1"],
        "offsets": [f.offset for f in PLAN.fields],
        "itemsize": PLAN.size,
    }
)
if PLAN_DTYPE.itemsize != PLAN.size:  # pragma: no cover - a typo in the table above
    raise RuntimeError("PLAN_DTYPE does not match the PLAN record")

#: What ``opponent_ix`` says when a seat has no opponent table entry: the mirror case, where
#: both seats are the learner, and the dead-worker case, where nobody plays at all.
NO_OPPONENT = -1


class SlotPlanner:
    """The slot map of one run, and the seed paths addressed by it.

    Constructed from the geometry and the master seed, both of which are in the run identity,
    so two runs of one identity have the same map and draw the same numbers.
    """

    def __init__(self, geometry: Geometry, master_seed: int) -> None:
        self.geometry = geometry
        self.master_seed = master_seed
        self.n_battles = geometry.n_battles
        self.n_slots = geometry.n_slots
        self.workers = geometry.workers
        self.shards_per_worker = geometry.shards_per_worker
        self.games_per_shard = geometry.games_per_shard
        self.games_per_worker = geometry.games_per_worker
        self.slots_per_shard = geometry.slots_per_shard

        battles = np.arange(self.n_battles, dtype=np.int32)
        self.battle_worker = (battles // self.games_per_worker).astype(np.int32)
        self.battle_shard = (
            (battles % self.games_per_worker) // self.games_per_shard
        ).astype(np.int32)
        #: Where a battle sits inside its own shard's vec env.
        self.battle_index = (battles % self.games_per_shard).astype(np.int32)

        self.slot_battle = np.repeat(battles, 2).astype(np.int32)
        self.slot_seat = np.tile(np.array([0, 1], dtype=np.int8), self.n_battles)
        self.slot_worker = self.battle_worker[self.slot_battle]
        self.slot_shard = self.battle_shard[self.slot_battle]
        #: Where a slot sits inside its own shard's batch: the vec env's own flat index.
        self.slot_index = (2 * self.battle_index[self.slot_battle] + self.slot_seat).astype(
            np.int32
        )

    # -- the map ------------------------------------------------------------

    def shard_battles(self, worker: int, shard: int) -> np.ndarray:
        """The battles of one shard, ascending. Contiguous, by construction."""
        base = worker * self.games_per_worker + shard * self.games_per_shard
        return np.arange(base, base + self.games_per_shard, dtype=np.int32)

    def shard_slots(self, worker: int, shard: int) -> np.ndarray:
        """The slots of one shard, ascending: both seats of each of its battles."""
        base = 2 * (worker * self.games_per_worker + shard * self.games_per_shard)
        return np.arange(base, base + self.slots_per_shard, dtype=np.int32)

    def worker_slots(self, worker: int) -> np.ndarray:
        """Every slot one worker holds, ascending across its shards."""
        base = 2 * worker * self.games_per_worker
        return np.arange(base, base + 2 * self.games_per_worker, dtype=np.int32)

    def round_slots(self, shard: int) -> np.ndarray:
        """Every slot of one shard-round: that shard of every worker, in worker order.

        A round covers one shard across the whole farm, and the workers' blocks are already
        ascending because battles are numbered worker-major, so this is a concatenation and
        not a sort.
        """
        return np.concatenate(
            [self.shard_slots(worker, shard) for worker in range(self.workers)]
        )

    def slot_of(self, battle: int, seat: int) -> int:
        return 2 * battle + seat

    # -- the seed streams ---------------------------------------------------

    def env_seed_path(self, worker: int, shard: int, generation: int) -> str:
        return stream_path(ENV_SHARD, worker=worker, shard=shard, generation=generation)

    def env_seed(self, worker: int, shard: int, generation: int) -> int:
        """The one ``ClashSelfPlayVecEnv.reset(seed=...)`` of a shard, per respawn generation."""
        return derive_int(self.master_seed, self.env_seed_path(worker, shard, generation))

    def scripted_generator(
        self, worker: int, slot: int, generation: int
    ) -> np.random.Generator:
        """One worker-side scripted opponent's generator, addressed by the slot it plays."""
        return derive_generator(
            self.master_seed,
            stream_path(SCRIPTED_SLOT, worker=worker, slot=slot, generation=generation),
        )

    def match_path(self, battle: int, ordinal: int) -> str:
        return stream_path(MATCH_BATTLE, battle=battle, ordinal=ordinal)

    def match_seed(self, battle: int, ordinal: int) -> int:
        """The matchmaker's draw for one episode of one battle, as an integer.

        Addressed by the battle and its reset ordinal rather than by the iteration and the
        slot, so the same episode of the same battle meets the same opponent whichever
        iteration it falls in and whatever the worker count is.
        """
        return derive_int(self.master_seed, self.match_path(battle, ordinal))

    def uniforms(self, iteration: int, cycle: int) -> np.ndarray:
        """The ``(n_slots,)`` uniforms that drive action sampling at one cycle.

        Drawn once per cycle and indexed by slot, so a sampled action is a pure function of
        the master seed, the iteration, the cycle and the slot -- and therefore independent of
        how the round's inference batch happened to be composed.
        """
        generator = derive_generator(
            self.master_seed, stream_path(ACT_CYCLE, iteration=iteration, cycle=cycle)
        )
        return generator.random(self.n_slots, dtype=np.float32)

    # -- the opening table --------------------------------------------------

    def mirror_plan(self, iteration: int, ordinals: np.ndarray | None = None) -> SlotPlan:
        """The table in which every battle is the learner against itself.

        It is what a run collects before the ladder has anyone in it, and it is what the
        rollout tests assign, so that what they exercise is the rollout path rather than the
        matchmaker's mixture.
        """
        if ordinals is None:
            ordinals = np.zeros(self.n_battles, dtype=np.int64)
        return SlotPlan(
            iteration=iteration,
            n_battles=self.n_battles,
            n_slots=self.n_slots,
            assignment=tuple(
                Assignment(
                    battle=battle,
                    ordinal=int(ordinals[battle]),
                    role=ROLE_MIRROR,
                    opponent_id=None,
                    group=(GROUP_LEARNER, GROUP_LEARNER),
                    learner_seat=-1,
                )
                for battle in range(self.n_battles)
            ),
            resident_snapshots=(),
        )


def opponent_index(plan: SlotPlan, assignment: Assignment) -> int:
    """Where a battle's opponent sits in the table its ``group`` refers to.

    A pool opponent indexes the plan's resident snapshots and a scripted one indexes
    ``scripted.SCRIPTED_NAMES``; a mirror has neither. The index rather than the name crosses
    to the worker because the worker needs one byte per slot, and because the only thing it
    does with it is pick a scripted opponent out of a fixed tuple.
    """
    opponent = assignment.opponent_id
    if opponent is None:
        return NO_OPPONENT
    kind, _, name = opponent.partition(":")
    if kind == "scripted":
        try:
            return SCRIPTED_NAMES.index(name)
        except ValueError:
            raise KeyError(
                f"battle {assignment.battle} was assigned scripted opponent {name!r}, which is "
                f"not one of {', '.join(SCRIPTED_NAMES)}"
            ) from None
    if kind == "snap":
        try:
            return plan.resident_snapshots.index(opponent)
        except ValueError:
            raise KeyError(
                f"battle {assignment.battle} was assigned {opponent!r}, which is not resident "
                f"in the plan's {len(plan.resident_snapshots)} snapshot(s)"
            ) from None
    raise KeyError(f"opponent id {opponent!r} is neither a snapshot nor a scripted opponent")


def write_plan(view: memoryview, plan: SlotPlan, slots: np.ndarray, planner: SlotPlanner) -> None:
    """Write one worker's rows of the opening table into its plan region.

    ``slots`` are that worker's slots, ascending; the row order is theirs, so the worker reads
    row ``i`` for its ``i``-th slot without needing the global slot number.
    """
    table = np.frombuffer(view, dtype=PLAN_DTYPE, count=len(slots))
    for row, slot in enumerate(slots):
        battle = int(planner.slot_battle[slot])
        seat = int(planner.slot_seat[slot])
        assignment = plan.assignment[battle]
        table[row]["seed"] = np.uint64(planner.match_seed(battle, assignment.ordinal))
        table[row]["role"] = assignment.role
        table[row]["group"] = assignment.group[seat]
        table[row]["opponent_ix"] = opponent_index(plan, assignment)


def read_plan(view: memoryview, n_slots: int) -> np.ndarray:
    """One worker's rows of the opening table, as a structured array over the region."""
    return np.frombuffer(view, dtype=PLAN_DTYPE, count=n_slots)
