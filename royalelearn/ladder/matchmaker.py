"""Who each battle plays next.

Two separate decisions, and they are made in different places on purpose.

**Which role a battle plays** -- mirror, pool or scripted -- belongs to the battle slot, for the
whole run. The mixture is a count of slots: exactly 48 of 96 battles are mirror slots at the
laptop profile, not 96 slots each drawn mirror with probability a half. That is what makes a
cycle's learner rows a constant, and the iteration is sized from that constant. The layout is a
seeded permutation of the battle indices, so it is a pure function of the master seed and the
battle count -- a permutation rather than "the first 48", because battles map to workers and
shards in blocks and a role that always lands on low indices always lands on the same worker.

**Who a battle plays and from which seat** is still one draw per episode, addressed by the
battle and its reset ordinal. An iteration spans several episodes in every slot, so one draw
per slot per iteration would give every episode that starts inside that iteration the same
opponent -- a correlation across the mixture that nothing in the statistics accounts for. And
because the address does not mention the iteration, it does not move when the iteration length
changes: the same episode of the same battle meets the same opponent however the iteration
boundary falls across it.

The role cannot have that last property and be exact at the same time. Exactly half of 96
battles and exactly half of 192 battles are not the same set of indices, so a run replayed at a
different worker count now lays its roles out differently. Exactness across the rectangle is
worth more than invariance to it: the rectangle is part of the run's identity already, and a
sampled mixture was halting one laptop iteration in four.

A policy is bound to a battle for exactly one episode. The parent redraws at the battle's own
episode boundary, so a partially controlled trajectory cannot occur.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import numpy as np

from ..api.ladder import Matchmaker, RatingTable
from ..api.rollout import (
    GROUP_LEARNER,
    GROUP_SCRIPTED,
    ROLE_MIRROR,
    ROLE_POOL,
    ROLE_SCRIPTED,
    Assignment,
    EpisodeRecord,
    SlotPlan,
)
from ..config import role_counts
from ..seeding import MATCH_BATTLE, derive_generator, stream_path
from .pool import SCRIPTED_IDS

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..config import Geometry, LadderConfig
    from .pool import LadderPool

__all__ = ["RESIDENCY_BATTLE", "ROLE_LAYOUT_BATTLE", "MixMatchmaker", "pfsp_shape"]

#: The residency draw's address. It is not a battle index -- no battle is negative -- so the
#: draw cannot collide with any battle's own stream, and it is still a path out of ``STREAMS``
#: rather than a private string.
RESIDENCY_BATTLE = -1
#: The role layout's address, on the same pattern and a different sentinel. Its ``ordinal`` is
#: the battle count, because the layout is a permutation of exactly that many slots.
ROLE_LAYOUT_BATTLE = -2


def pfsp_shape(p: np.ndarray, weighting: str, power: float) -> np.ndarray:
    """The unnormalised PFSP weight of an opponent the learner scores ``p`` against.

    ``hard`` for training and ``variance`` for measurement, because the two want different
    things: training wants opponents that still beat the learner, and a measurement wants the
    most even matchup, which is where a game carries the most information.

    ``p`` is the **fitted model's** predicted score rather than the direct head-to-head record.
    On a pool of several dozen the learner has played few pairs directly, a Beta(1,1) prior
    reads exactly one half for the rest, and ``hard`` over those priors collapses to uniform --
    discarding what the fit knows perfectly well, that the learner crushes one member and that
    member crushes another.
    """
    if weighting == "hard":
        weights = (1.0 - p) ** power
    elif weighting == "variance":
        weights = p * (1.0 - p)
    elif weighting == "linear":
        weights = 1.0 - p
    elif weighting == "uniform":
        weights = np.ones_like(p)
    else:
        raise ValueError(f"unknown PFSP weighting {weighting!r}")
    if not np.any(weights > 0.0):
        weights = np.ones_like(p)
    return weights


class MixMatchmaker(Matchmaker):
    """The mixture of section 11.3: half mirror, a third pool, a sixth scripted.

    The shares are counts of battle slots. Of 96 battles, 48 are mirror slots, 34 pool and 14
    scripted, for the whole run; the learner sits in both seats of a mirror battle and one seat
    of the others, so a cycle collects 96 + 48 = 144 learner rows out of 192, three quarters,
    every cycle of every iteration whatever the master seed.

    Counts rather than a draw per episode because the iteration is sized from that number. A
    drawn mixture makes the mirror count Binomial(96, 0.5) -- a standard deviation of 4.9 rows
    per cycle -- against an iteration whose slack over the trainable-row floor is 0.2%, so
    about one master seed in four sized an iteration too small and halted the run in its first
    minutes. Counting also makes the discarded-row share an equality rather than a sample
    proportion compared against a constant, and those two quantities were not even the same
    one: a mirror episode runs about 9% longer than a scripted one, so a battle's share of the
    mixture and its share of the rows differ by about two points.

    What stays random is what does not change a seat count: which opponent a pool or scripted
    battle draws, and which seat the learner takes, both redrawn at every episode boundary from
    that battle's own stream.

    A scripted battle draws uniformly from ``scripted_ids``. A run passes
    ``ladder.scripted_opponents`` there, and the default is the two anchors. The list is only who
    the learner trains against: the anchors the pool and the gate measure against are set apart
    from it and stay the two.
    """

    #: 2 adds ``n_battles``. A format-1 checkpoint loads; see ``load_checkpoint``.
    FORMAT_VERSION = 2

    def __init__(
        self,
        master_seed: int,
        config: LadderConfig,
        *,
        n_battles: int | None = None,
        learner_id: str = "learner",
        scripted_ids: Sequence[str] = SCRIPTED_IDS,
    ) -> None:
        mix = tuple(float(share) for share in config.mix)
        if abs(sum(mix) - 1.0) > 1e-9:
            raise ValueError(f"the mixture {mix} does not sum to one")
        self.master_seed = int(master_seed)
        self.config = config
        self.mix = mix
        self.learner_id = learner_id
        self.scripted_ids = tuple(scripted_ids)
        if not self.scripted_ids and (mix[1] > 0.0 or mix[2] > 0.0):
            # A pool battle plays scripted until the first snapshot is admitted, so a pool share
            # needs the list as much as a scripted one does.
            raise ValueError(
                f"the mixture {mix} has pool or scripted battles and no scripted opponent to "
                "give them"
            )
        self._ordinal: dict[int, int] = {}
        self._residents: tuple[int, tuple[str, ...]] | None = None
        self._plan_epoch: int | None = None
        self._n_battles: int | None = None
        self._mirror_battles = 0
        self._layout: np.ndarray | None = None
        if n_battles is not None:
            self.set_rectangle(int(n_battles))

    # -- the mixture --------------------------------------------------------

    @property
    def n_battles(self) -> int | None:
        """The rectangle the roles are laid out over, or None before the first ``plan``."""
        return self._n_battles

    def set_rectangle(self, n_battles: int) -> None:
        """Lay the roles out over ``n_battles`` battle slots.

        Called by ``plan`` with the iteration's geometry, and by the constructor when a caller
        already knows the rectangle. Laying it out again for the same count is a no-op, so an
        iteration boundary does not move a single role.
        """
        if n_battles < 1:
            raise ValueError(f"a rectangle of {n_battles} battles has nothing to assign")
        if self._n_battles == n_battles:
            return
        mirror, pool, _scripted = role_counts(self.mix, n_battles)
        rng = derive_generator(
            self.master_seed,
            stream_path(MATCH_BATTLE, battle=ROLE_LAYOUT_BATTLE, ordinal=n_battles),
        )
        order = rng.permutation(n_battles)
        layout = np.full(n_battles, ROLE_SCRIPTED, dtype=np.int8)
        layout[order[:mirror]] = ROLE_MIRROR
        layout[order[mirror : mirror + pool]] = ROLE_POOL
        self._n_battles = n_battles
        self._mirror_battles = mirror
        self._layout = layout

    def role_of(self, battle: int) -> int:
        """What ``battle`` plays, every episode, for the whole run."""
        if self._layout is None or self._n_battles is None:
            raise RuntimeError(
                "the role layout is not set: build the matchmaker with n_battles, or call "
                "plan() with the iteration's geometry before drawing an assignment"
            )
        if not 0 <= battle < self._n_battles:
            raise IndexError(
                f"battle {battle} is outside a rectangle of {self._n_battles} battles"
            )
        return int(self._layout[battle])

    @property
    def learner_rows(self) -> int:
        """Learner rows one cycle of the rectangle produces: two per mirror battle, one per
        other battle, so ``n_battles + mirror_battles``."""
        if self._n_battles is None:
            raise RuntimeError("no rectangle has been laid out yet")
        return self._n_battles + self._mirror_battles

    @property
    def expected_learner_row_fraction(self) -> float:
        """The share of collected rows the learner sits in: both seats of a mirror, one of the
        rest.

        Exact once a rectangle is laid out. Before then there is no rectangle to be exact
        about, and this is the mixture's own share -- what the count converges to as the
        battles grow. Nothing in a run reads it there: ``plan`` lays the rectangle out at the
        top of every iteration, before a row is collected.
        """
        if self._n_battles is None:
            mirror = self.mix[0]
            return mirror + (1.0 - mirror) / 2.0
        return self.learner_rows / (2 * self._n_battles)

    @property
    def expected_discarded_rows_frac(self) -> float:
        """What ``throughput/discarded_rows_frac`` must come out at, by construction."""
        return 1.0 - self.expected_learner_row_fraction

    def ordinal(self, battle: int) -> int:
        """The episode ``battle`` is about to play. Counts from zero."""
        return self._ordinal.get(battle, 0)

    def restore_ordinals(self, ordinals: Mapping[int, int]) -> None:
        """Put every battle back on the episode it was about to play.

        An ordinal is not bookkeeping: it addresses the stream an episode's seed comes from
        (``match/battle/{b}/ordinal/{k}``), so a resume that left these at zero would draw the
        run's opening battles again under weights that have moved on, and the environment side
        of the run would diverge from the one being continued while the learner side matched
        perfectly.
        """
        self._ordinal.update({int(battle): int(value) for battle, value in ordinals.items()})

    def on_episode(self, record: EpisodeRecord) -> None:
        """Advance a battle's ordinal past the episode that just finished.

        Both seats of a battle report the same episode, so the advance is a maximum rather than
        an increment: two records must not move the ordinal twice.
        """
        self._ordinal[record.battle] = max(self.ordinal(record.battle), record.ordinal + 1)

    # -- residency ----------------------------------------------------------

    def residents(self, pool: LadderPool, ratings: RatingTable | None) -> tuple[str, ...]:
        """The frozen snapshots a pool battle may be given right now.

        At most ``max_resident_opponents`` of them, so a shard-round is a handful of batched
        forwards and the snapshot cache never thrashes. The set is drawn PFSP-weighted, without
        replacement, from a stream addressed by the pool's residency epoch -- which moves only
        when a snapshot is admitted or evicted or a refit lands, all of which happen at an
        iteration boundary. So the set is a pure function of the pool and the ratings, every
        member of the pool has a floored chance of being in it, and an assignment drawn in the
        middle of an iteration cannot name a snapshot the iteration's plan did not carry.
        """
        candidates = pool.sampler()
        epoch = pool.residency_epoch
        cached = self._residents
        if cached is not None and cached[0] == epoch and set(cached[1]) <= set(candidates):
            return cached[1]
        limit = max(1, int(self.config.max_resident_opponents))
        if len(candidates) <= limit:
            chosen = tuple(candidates)
        else:
            weights = self.weights(candidates, ratings, self.config.pfsp_weighting)
            rng = derive_generator(
                self.master_seed,
                stream_path(MATCH_BATTLE, battle=RESIDENCY_BATTLE, ordinal=epoch),
            )
            picked = rng.choice(len(candidates), size=limit, replace=False, p=weights)
            chosen = tuple(sorted(candidates[int(position)] for position in picked))
        self._residents = (epoch, chosen)
        return chosen

    def weights(
        self, candidates: Sequence[str], ratings: RatingTable | None, weighting: str
    ) -> np.ndarray:
        """The normalised draw weights over ``candidates``.

        ``0.8 * hard + 0.2 * uniform``, then every weight floored at
        ``weight_floor_scale / M``. Without the uniform share and the floor the population's
        tail becomes unreachable, which is the job AlphaStar's forgotten-players slice does;
        with them, the weakest member of a pool of fifty is still drawn about once in a
        thousand episodes and the fit keeps a number for it.
        """
        count = len(candidates)
        if count == 0:
            return np.zeros(0, dtype=np.float64)
        predicted = np.array(
            [self._predicted_score(candidate, ratings) for candidate in candidates],
            dtype=np.float64,
        )
        shaped = pfsp_shape(predicted, weighting, self.config.pfsp_power)
        shaped = shaped / shaped.sum()
        uniform = self.config.pfsp_uniform_floor
        weights = (1.0 - uniform) * shaped + uniform / count
        weights = np.maximum(weights, self.config.weight_floor_scale / count)
        return weights / weights.sum()

    def _predicted_score(self, opponent: str, ratings: RatingTable | None) -> float:
        """P(the learner scores against ``opponent``) under the current fit.

        Read off the fitted ratings rather than asked of the rater, so that a matchmaker holds
        a rating table and not a fitting object, and a plan built from a stored table is the
        plan that table implies.
        """
        if ratings is None:
            return 0.5
        learner = ratings.rating.get(self.learner_id)
        other = ratings.rating.get(opponent)
        if learner is None or other is None:
            return 0.5
        return 1.0 / (1.0 + 10.0 ** ((other - learner) / 400.0))

    # -- the draw -----------------------------------------------------------

    def assign(
        self, battle: int, ordinal: int, pool: LadderPool, ratings: RatingTable | None = None
    ) -> Assignment:
        """One battle's next episode.

        The role is the slot's, not the episode's: ``role_of(battle)`` answers it and nothing
        drawn here can change it. What is drawn is the opponent and the learner's seat.

        A pool assignment names its opponent by its position in the resident table, and the
        table the worker resolves that position against is the one the iteration's plan carried.
        So the pool must not move between the plan and the assignments drawn against it: it is
        checked here rather than left to discipline, because a move would not fail -- it would
        quietly file the episode under another snapshot's name.
        """
        if self._plan_epoch is not None and pool.residency_epoch != self._plan_epoch:
            raise RuntimeError(
                f"the pool is at residency epoch {pool.residency_epoch} and this iteration was "
                f"planned at {self._plan_epoch}; an admission, an eviction or a refit belongs "
                "at an iteration boundary, because an assignment's group is a position in the "
                "plan's resident table and every one of them moves with the epoch"
            )
        role = self.role_of(battle)
        rng = derive_generator(
            self.master_seed, stream_path(MATCH_BATTLE, battle=battle, ordinal=ordinal)
        )
        residents = self.residents(pool, ratings)
        if role == ROLE_POOL and not residents:
            # Before the first snapshot is admitted there is nothing to draw, and a scripted
            # opponent is the honest substitute: it keeps the seat count and the discarded-row
            # fraction exactly what the mixture says they are.
            role = ROLE_SCRIPTED

        if role == ROLE_MIRROR:
            return Assignment(
                battle=battle,
                ordinal=ordinal,
                role=ROLE_MIRROR,
                opponent_id=None,
                group=(GROUP_LEARNER, GROUP_LEARNER),
                learner_seat=0,
            )

        learner_seat = int(rng.integers(2))
        if role == ROLE_SCRIPTED:
            opponent = self.scripted_ids[int(rng.integers(len(self.scripted_ids)))]
            other = GROUP_SCRIPTED
        else:
            weights = self.weights(residents, ratings, self.config.pfsp_weighting)
            position = int(rng.choice(len(residents), p=weights))
            opponent = residents[position]
            other = position
        group = [0, 0]
        group[learner_seat] = GROUP_LEARNER
        group[1 - learner_seat] = other
        return Assignment(
            battle=battle,
            ordinal=ordinal,
            role=role,
            opponent_id=opponent,
            group=(group[0], group[1]),
            learner_seat=learner_seat,
        )

    def plan(
        self,
        iteration: int,
        pool: LadderPool,
        ratings: RatingTable | None,
        geometry: Geometry,
    ) -> SlotPlan:
        """The iteration's opening table: ``assign`` across the geometry at each battle's own
        current ordinal.

        The pool's residency epoch is recorded as the plan is built, and every assignment drawn
        until the next plan is checked against it.
        """
        self._plan_epoch = None
        self.set_rectangle(geometry.n_battles)
        assignments = tuple(
            self.assign(battle, self.ordinal(battle), pool, ratings)
            for battle in range(geometry.n_battles)
        )
        residents = self.residents(pool, ratings)
        self._plan_epoch = pool.residency_epoch
        return SlotPlan(
            iteration=iteration,
            n_battles=geometry.n_battles,
            n_slots=geometry.n_slots,
            assignment=assignments,
            resident_snapshots=residents,
        )

    # -- checkpoint ---------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        """The ordinals, and the rectangle the roles are laid out over.

        The layout itself is not stored: it is a pure function of the master seed, the mixture
        and the battle count, so storing the count reproduces it exactly and storing the array
        would only add a second copy that could disagree with the first.
        """
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "matchmaker.json").write_bytes(
            msgspec.json.encode(
                {
                    "ordinal": {str(k): v for k, v in sorted(self._ordinal.items())},
                    "n_battles": self._n_battles,
                }
            )
        )

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        """Put the ordinals and the rectangle back.

        A format-1 checkpoint has no ``n_battles``; it loads, and the rectangle comes back at
        the resumed run's first ``plan``, which is drawn before any assignment is. The roles
        are the same either way, because the same seed and the same battle count produce the
        same layout.
        """
        path = folder / "matchmaker.json"
        if not path.exists():
            if strict:
                raise FileNotFoundError(str(path))
            print(f"no matchmaker state at {path}; every battle restarts at ordinal zero")
            return
        state = msgspec.json.decode(path.read_bytes())
        self._ordinal = {int(k): int(v) for k, v in state["ordinal"].items()}
        self._residents = None
        self._plan_epoch = None
        self._n_battles = None
        self._mirror_battles = 0
        self._layout = None
        stored = state.get("n_battles")
        if stored is not None:
            self.set_rectangle(int(stored))
