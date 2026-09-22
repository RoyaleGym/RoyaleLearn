"""Measuring two policies against each other, and the four separations that measurement needs.

1. **No experience is recorded.** The runner writes results and nothing else; the source it
   drives plans every row non-trainable. Evaluation that feeds the buffer makes the training
   distribution depend on the measurement, and then neither means what it says.
2. **A different episode definition.** A full match under real rules, built from its own env
   spec (``config.eval_env``) rather than by mutating the live one. A step cap does not make
   evaluation cheaper, it makes it uninformative: a cut match is scored on crowns that have
   mostly not been taken yet, so nearly every game is a draw and the rating barely moves. That
   holds for a smoke configuration too, where the number of battles is reduced and the episode
   definition is not.
3. **Separate streams and a separate table.** The seeds come from a set drawn once at run start
   and reused forever; results are tagged ``kind="eval"`` and are the only ones the fit reads.
4. **Paired, with common random numbers.** Each pairing plays each seed twice with the sides
   swapped, and the unit of analysis is the seed, not the battle. That removes the side bias
   exactly rather than averaging it away, and removes the start-state variance the two policies
   share. The interval is a bootstrap over seeds and never a binomial over battles: the two
   battles of one seed are correlated, and treating them as independent overstates the sample
   size by up to a factor of two.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol

import msgspec
import numpy as np

from ..errors import PreflightError
from ..seeding import EVAL_BOOTSTRAP, EVAL_MATCH, EVAL_SEED_SET, derive_generator, stream_path
from .rating import Z95
from .results import KIND_EVAL, GameResult, ResultLog

__all__ = [
    "SIDES",
    "BattlePlayer",
    "Comparison",
    "EvalRunner",
    "SeedScore",
    "SeedSet",
    "bootstrap_interval",
    "comparison_id",
    "eval_seed_set",
    "paired_rho",
    "score_interval",
]

#: Blue is seat 0 and red is seat 1, as everywhere else in the harness. A pairing plays each
#: seed once with ``a`` on each side.
SIDES: tuple[str, str] = ("blue", "red")


class SeedSet(msgspec.Struct, frozen=True):
    """The run's frozen evaluation seeds.

    Drawn once, written into the run directory, hashed into the run identity and reused for
    every comparison the run ever makes. A rating is only a common scale if every player was
    measured on the same positions.
    """

    seeds: tuple[int, ...]

    def sha(self) -> str:
        """The digest the identity and every ``GateDecision`` carry."""
        joined = ",".join(str(seed) for seed in self.seeds)
        return hashlib.sha256(joined.encode("ascii")).hexdigest()

    def __len__(self) -> int:
        return len(self.seeds)


def eval_seed_set(master_seed: int, count: int) -> SeedSet:
    """Draw the run's evaluation seeds from ``eval/seed_set``.

    63 bits, because the seed crosses into ``ClashSelfPlayVecEnv.reset(seed=...)`` and
    gymnasium's seeding refuses a value that does not fit a signed 64-bit integer.
    """
    rng = derive_generator(master_seed, stream_path(EVAL_SEED_SET))
    drawn = rng.integers(0, 2**63 - 1, size=int(count), dtype=np.int64)
    return SeedSet(seeds=tuple(int(seed) for seed in drawn))


class BattlePlayer(Protocol):
    """Playing one evaluation battle, whatever is driving the environment.

    The runner owns the statistics and the log; this is the seam where a rollout source, its
    frozen actors and the release sampling mode live. ``act_path`` names the stream the acting
    uniforms come from, so the same battle replays identically wherever it is run.
    """

    def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
        """``a``'s score: 1 for a win, 0.5 for a draw, 0 for a loss."""
        ...


class SeedScore(msgspec.Struct, frozen=True):
    """One seed, played from both sides. ``x`` is the unit of analysis."""

    seed_index: int
    as_blue: float
    as_red: float

    @property
    def x(self) -> float:
        """``(score as blue + score as red) / 2``, in {0, 0.25, 0.5, 0.75, 1}."""
        return 0.5 * (self.as_blue + self.as_red)


class Comparison(msgspec.Struct, frozen=True):
    """What a pairing measured.

    ``rho`` is the empirical correlation between the two sides of a seed. The effective sample
    size is about ``n / (1 + rho)``: at zero, pairing costs nothing, and above it pairing wins.
    It is worth knowing on its own, too -- it says how much of a battle's outcome the start
    state decides rather than the policies.
    """

    a: str
    b: str
    n_seeds: int
    n_games: int
    score_a: float
    lo: float
    hi: float
    rho: float
    draw_rate: float
    seed_scores: tuple[SeedScore, ...]

    @property
    def se(self) -> float:
        """The standard error of ``score_a`` over seeds, which is the paired one."""
        if self.n_seeds < 2:
            return 0.0
        values = np.array([score.x for score in self.seed_scores], dtype=np.float64)
        return float(values.std(ddof=1) / np.sqrt(self.n_seeds))


def bootstrap_interval(
    values: Sequence[float], rng: np.random.Generator, resamples: int, confidence: float = 0.95
) -> tuple[float, float]:
    """The percentile bootstrap interval of the mean, resampling seeds.

    Seeds and not battles: the two battles of one seed share a start state and are correlated,
    so a binomial over battles would report an interval up to a factor of the square root of
    two too narrow.
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0:
        return 0.0, 1.0
    if sample.size == 1:
        return float(sample[0]), float(sample[0])
    draws = rng.integers(0, sample.size, size=(int(resamples), sample.size))
    means = sample[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


def comparison_id(a: str, b: str) -> str:
    """One name for a pairing, whichever way round it was asked for.

    Order-independent so that comparing A with B and comparing B with A draw the same bootstrap
    resamples, and the two intervals are exact mirrors rather than two samples of one.
    """
    low, high = sorted((a, b))
    return hashlib.sha256(f"{low}\x1f{high}".encode()).hexdigest()[:16]


class EvalRunner:
    """The fixed seed set, the side swap, the bootstrap, and a line per battle in the log."""

    def __init__(
        self,
        player: BattlePlayer,
        seeds: SeedSet,
        *,
        master_seed: int,
        context: str = "",
        run_id: str = "",
        log: ResultLog | None = None,
        bootstrap_resamples: int = 10_000,
        release_mode: str = "stochastic",
        obs_digest: Callable[[str], str] | None = None,
    ) -> None:
        if release_mode not in ("stochastic", "argmax"):
            raise ValueError(f"unknown release mode {release_mode!r}")
        self.player = player
        self.seeds = seeds
        self.master_seed = int(master_seed)
        self.context = context
        self.run_id = run_id
        self.log = log
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.release_mode = release_mode
        self.obs_digest = obs_digest
        self.games_played = 0

    def compare(self, a: str, b: str, *, games: int, iteration: int = 0) -> Comparison:
        """Play ``games`` battles between ``a`` and ``b``: half the seeds, twice each."""
        self._check_pairing(a, b)
        n_seeds = int(games) // 2
        if n_seeds < 1:
            raise ValueError(f"a paired comparison needs at least two battles, not {games}")
        if n_seeds > len(self.seeds):
            raise ValueError(
                f"{games} battles need {n_seeds} seeds and the frozen set holds "
                f"{len(self.seeds)}; raise ladder.eval_seed_count for a longer comparison"
            )
        identity = comparison_id(a, b)
        scores: list[SeedScore] = []
        results: list[GameResult] = []
        draws = 0
        wall = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        for seed_index in range(n_seeds):
            seed = self.seeds.seeds[seed_index]
            sides: list[float] = []
            for a_seat, side in enumerate(SIDES):
                score = self.player.play(
                    a=a,
                    b=b,
                    seed=seed,
                    a_seat=a_seat,
                    act_path=stream_path(
                        EVAL_MATCH, comparison=identity, seed_index=seed_index, side=side
                    ),
                )
                score = _checked_score(score)
                sides.append(score)
                draws += score == 0.5
                results.append(
                    GameResult(
                        a=a,
                        b=b,
                        score_a=score,
                        seed_index=seed_index,
                        side_a=side,
                        context=self.context,
                        kind=KIND_EVAL,
                        run_id=self.run_id,
                        iteration=iteration,
                        wall=wall,
                    )
                )
            scores.append(
                SeedScore(seed_index=seed_index, as_blue=sides[0], as_red=sides[1])
            )
        if self.log is not None:
            self.log.extend(results)
        self.games_played += len(results)

        paired = np.array([score.x for score in scores], dtype=np.float64)
        rng = derive_generator(
            self.master_seed, stream_path(EVAL_BOOTSTRAP, comparison=identity)
        )
        lo, hi = bootstrap_interval(paired, rng, self.bootstrap_resamples)
        return Comparison(
            a=a,
            b=b,
            n_seeds=n_seeds,
            n_games=len(results),
            score_a=float(paired.mean()),
            lo=lo,
            hi=hi,
            rho=paired_rho(scores),
            draw_rate=draws / len(results),
            seed_scores=tuple(scores),
        )

    def _check_pairing(self, a: str, b: str) -> None:
        """Refuse a pairing whose two policies did not see the same observation.

        A policy that was shown the opponent's hand and one that was not are not the same kind
        of player, and a number comparing them would be about the information rather than about
        them. Both ids and both digests are named, because the interesting question when this
        fires is which of the two is the odd one.
        """
        if self.obs_digest is None:
            return
        digest_a, digest_b = self.obs_digest(a), self.obs_digest(b)
        if digest_a != digest_b:
            raise PreflightError(
                f"{a} observes {digest_a} and {b} observes {digest_b}; a rating over two "
                "different observations measures the observation, so this pairing is refused"
            )


def paired_rho(scores: Sequence[SeedScore]) -> float:
    """The correlation between a seed's two side assignments.

    Zero variance on either side -- every seed decided the same way -- is reported as zero
    rather than as a division by zero: nothing about the pairing was learned.
    """
    if len(scores) < 2:
        return 0.0
    blue = np.array([score.as_blue for score in scores], dtype=np.float64)
    red = np.array([score.as_red for score in scores], dtype=np.float64)
    if blue.std() == 0.0 or red.std() == 0.0:
        return 0.0
    return float(np.corrcoef(blue, red)[0, 1])


def _checked_score(score: float) -> float:
    if score not in (0.0, 0.5, 1.0):
        raise ValueError(f"a battle scores 0, 0.5 or 1, not {score!r}")
    return float(score)


def score_interval(comparison: Comparison, z: float = Z95) -> tuple[float, float]:
    """The normal interval around a comparison's paired mean, for a quick sanity check against
    the bootstrap. The bootstrap is what the gate reads; this is what says the two agree."""
    half = z * comparison.se
    return comparison.score_a - half, comparison.score_a + half
