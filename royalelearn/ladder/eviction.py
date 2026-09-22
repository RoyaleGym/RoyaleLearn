"""Which snapshots the sampler stops drawing.

Eviction here is about sampling cost and nothing else. Every snapshot stays in the archive, in
the result log and in the fit: at about a megabyte each, five hundred of them is under a
gigabyte, and a rating without the player it rated is not worth keeping. What is bounded is how
many the matchmaker considers per episode, because that draw is linear in the pool and happens
in Python.

What is kept is a stratified sample across the fitted rating range rather than the newest N.
A pool of the last fifty snapshots is a pool of one opponent measured fifty times; the weak
members are what stop a policy from forgetting how to beat them, which is exactly the failure
the anti-forgetting floor exists for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..api.ladder import EvictionPolicy, RatingTable

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .pool import LadderPool

__all__ = ["HallOfFameEviction"]


class HallOfFameEviction(EvictionPolicy):
    """Keep the anchors, the run's first snapshot, the champion chain and a spread of the rest.

    Snapshots flagged ``meta["cycle"]`` are preferred within their stratum: those are the ones
    that beat the champion and lost to the wider pool, which makes them the most diverse
    opponents in the archive and the ones a scalar rating describes worst.
    """

    def __init__(self, *, prefer_cycles: bool = True) -> None:
        self.prefer_cycles = prefer_cycles

    def select_for_eviction(
        self, *, pool: LadderPool, ratings: RatingTable, max_sampled: int
    ) -> list[str]:
        sampled = list(pool.sampler())
        if len(sampled) <= max_sampled:
            return []
        chain = set(pool.champion_chain)
        protected = [
            member
            for member in sampled
            if pool.is_anchor(member) or member == pool.v0 or member in chain
        ]
        optional = [member for member in sampled if member not in set(protected)]
        budget = max(0, max_sampled - len(protected))
        if budget >= len(optional):
            return []
        if budget == 0:
            return sorted(optional)

        # Rank by rating and cut into equal-count strata: equal-width strata over the range
        # would leave the budget unspent wherever the population is thin, and a pool's ratings
        # are always thin at the ends.
        ranked = sorted(optional, key=lambda member: (ratings.rating.get(member, 0.0), member))
        keep: list[str] = []
        for stratum in range(budget):
            low = stratum * len(ranked) // budget
            high = max(low + 1, (stratum + 1) * len(ranked) // budget)
            stratum_members = ranked[low:high]
            keep.append(
                max(stratum_members, key=lambda member: self._priority(member, pool, ratings))
            )
        evicted = sorted(set(optional) - set(keep))
        return evicted

    def _priority(
        self, member: str, pool: LadderPool, ratings: RatingTable
    ) -> tuple[int, int, int, str]:
        """What makes one member of a stratum worth keeping over another.

        A detected cycle first, then the one with the most evidence behind its rating, then the
        more recent snapshot: an interval nobody has narrowed is the least useful opponent to
        keep and the cheapest to re-measure later, because the archive still holds it.
        """
        cycle = 1 if (self.prefer_cycles and pool.meta(member).get("cycle")) else 0
        return (cycle, ratings.n_games.get(member, 0), pool.step_of(member), member)
