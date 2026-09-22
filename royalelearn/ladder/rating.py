"""What a pile of results says about who is stronger.

Two layers, and they are not interchangeable.

``EloReadout`` is the dashboard number: RoyaleGym's zero-sum logistic Elo at ``k = 32``, updated
as results arrive. It is order dependent, which under parallel workers means it is not
reproducible, and a fixed ``k`` keeps moving a frozen player whose strength is constant. It goes
on a panel and nowhere near a decision.

``BradleyTerryDavidsonRater`` is authoritative: a MAP fit over the whole stored result matrix,
refitted from scratch whenever it is asked for. The property that makes it the layer a gate may
use is that **the rating is a pure function of the stored results** -- same games in, same
numbers out, in any order, on any machine -- which is the same promise the engine makes about a
battle. Standard errors come from the inverse observed Fisher information, and a difference
between two players uses the corresponding 2x2 block rather than the sum of two marginals, which
would double-count the gauge they share.

The model, with ``q_i = 10 ** (r_i / 400)``::

    P(i wins) = q_i                     / (q_i + q_j + nu * sqrt(q_i * q_j))
    P(draw)   = nu * sqrt(q_i * q_j)    / (q_i + q_j + nu * sqrt(q_i * q_j))
    P(j wins) = q_j                     / (q_i + q_j + nu * sqrt(q_i * q_j))

Written in ``theta = ln(10) / 400 * r`` the denominator is a sum of exponentials of affine
functions of the parameters, so its logarithm is convex and the log-posterior is concave in
``(theta, ln nu)`` jointly: Newton converges in a handful of steps from any start, and the
solution it reaches is the only one.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import numpy as np

from ..api.ladder import Rater, RatingTable

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .results import ResultView

__all__ = [
    "ELO_SCALE",
    "Z95",
    "BradleyTerryDavidsonRater",
    "EloReadout",
    "elo_of_score",
    "score_of_elo",
    "wilson_interval",
]

#: ``theta = ELO_SCALE * rating``. One Elo point is this much of a natural logit, and
#: ``dp/dElo`` at even strength is a quarter of it: 0.0014386, the number the statistics table
#: of the ladder's documentation is built on.
ELO_SCALE = math.log(10.0) / 400.0

#: The normal quantile every interval in this package uses. One number, one place.
Z95 = 1.959963984540054

#: The standard deviation of the weakly informative prior on ``ln nu``. It is here so that a
#: result set with too few draws to identify the draw term has a finite fit at all; at the two
#: per cent below which the draw model is switched off entirely it is already irrelevant.
_LOG_NU_PRIOR_SD = 2.0


def wilson_interval(p_hat: float, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for an observed rate.

    Wilson rather than the normal approximation because the gate's bound is evaluated near 0.55
    at n around a thousand, where the two differ by enough to change a decision, and because the
    normal interval is nonsense at the ends. An observed 55.2% over 1000 battles clears 0.52 and
    an observed 55.0% does not, which is the whole margin the promotion rule runs on.
    """
    if n <= 0:
        return 0.0, 1.0
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denominator
    half = z / denominator * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n))
    return centre - half, centre + half


def elo_of_score(p: float) -> float:
    """The Elo gap a score rate implies: ``400 * log10(p / (1 - p))``."""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return 400.0 * math.log10(p / (1.0 - p))


def score_of_elo(gap: float) -> float:
    """The score rate an Elo gap implies."""
    return 1.0 / (1.0 + 10.0 ** (-gap / 400.0))


class EloReadout:
    """The online dashboard rating, and nothing else.

    Zero-sum: what one player gains the other loses, so the population's total is conserved and
    the number is about relative strength. It is deliberately not checkpointed as authority --
    a resume refits the real ratings from the log -- but its state round-trips so that the
    dashboard does not jump at a resume.
    """

    FORMAT_VERSION = 1

    def __init__(self, k_factor: float = 32.0, initial: float = 1200.0) -> None:
        self.k = float(k_factor)
        self.initial = float(initial)
        self._rating: dict[str, float] = {}
        self._games: dict[str, int] = {}

    def rating(self, player: str) -> float:
        return self._rating.get(player, self.initial)

    def ratings(self) -> dict[str, float]:
        return dict(self._rating)

    def games(self, player: str) -> int:
        return self._games.get(player, 0)

    def update(self, a: str, b: str, score_a: float) -> tuple[float, float]:
        """Record one game and return the two new ratings."""
        from royalegym.selfplay import expected_score

        if score_a not in (0.0, 0.5, 1.0):
            raise ValueError(f"score_a must be 0, 0.5 or 1, not {score_a!r}")
        rating_a, rating_b = self.rating(a), self.rating(b)
        delta = self.k * (score_a - expected_score(rating_a, rating_b))
        self._rating[a] = rating_a + delta
        self._rating[b] = rating_b - delta
        self._games[a] = self.games(a) + 1
        self._games[b] = self.games(b) + 1
        return self._rating[a], self._rating[b]

    # -- checkpoint ---------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "elo.json").write_bytes(
            msgspec.json.encode(
                {
                    "k_factor": self.k,
                    "initial": self.initial,
                    "rating": self._rating,
                    "games": self._games,
                }
            )
        )

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        path = folder / "elo.json"
        if not path.exists():
            if strict:
                raise FileNotFoundError(str(path))
            print(f"no Elo readout at {path}; starting it from {self.initial:g}")
            return
        state = msgspec.json.decode(path.read_bytes())
        self.k = float(state["k_factor"])
        self.initial = float(state["initial"])
        self._rating = {str(k): float(v) for k, v in state["rating"].items()}
        self._games = {str(k): int(v) for k, v in state["games"].items()}


class BradleyTerryDavidsonRater(Rater):
    """The authoritative rating: a MAP fit over the whole result matrix.

    ``nu`` is fitted jointly with the ratings so that the draw rate is explained rather than
    absorbed into strength. Below ``min_draw_rate`` the draw term is switched off and draws are
    counted as half a win, and the table says which of the two happened: at a handful of draws
    the parameter is not identified and a number nobody could trust is worse than no number.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        prior_sd: float = 400.0,
        anchor: str = "scripted:noop",
        draws: str = "davidson",
        min_draw_rate: float = 0.02,
        max_iterations: int = 50,
        tolerance: float = 1e-10,
        min_pair_games: int = 30,
    ) -> None:
        if draws not in ("davidson", "half_win"):
            raise ValueError(f"draws must be 'davidson' or 'half_win', not {draws!r}")
        self.prior_sd = float(prior_sd)
        self.anchor = anchor
        self.draws = draws
        self.min_draw_rate = float(min_draw_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.min_pair_games = int(min_pair_games)
        self._table: RatingTable | None = None
        self._theta: dict[str, float] = {}
        self._nu: float = 0.0
        self._free: dict[str, int] = {}
        self._covariance: np.ndarray | None = None

    # -- the fit ------------------------------------------------------------

    @property
    def table(self) -> RatingTable | None:
        """The last fit, or None before one."""
        return self._table

    def fit(self, results: ResultView) -> RatingTable:
        """Refit every rating from the whole result matrix."""
        players = tuple(sorted(set(results.players()) | {self.anchor}))
        index = {player: position for position, player in enumerate(players)}
        free = {
            player: position
            for position, player in enumerate(p for p in players if p != self.anchor)
        }
        n_free = len(free)

        pairs = [(index[a], index[b], record) for a, b, record in results.pairs()]
        draw_rate = results.draw_rate()
        davidson = self.draws == "davidson" and draw_rate >= self.min_draw_rate

        left = np.array([pair[0] for pair in pairs], dtype=np.int64)
        right = np.array([pair[1] for pair in pairs], dtype=np.int64)
        wins_left = np.array([pair[2].wins_a for pair in pairs], dtype=np.float64)
        wins_right = np.array([pair[2].wins_b for pair in pairs], dtype=np.float64)
        drawn = np.array([pair[2].draws for pair in pairs], dtype=np.float64)
        total = wins_left + wins_right + drawn

        theta = np.zeros(len(players), dtype=np.float64)
        log_nu = math.log(max(2.0 * draw_rate / max(1.0 - draw_rate, 1e-9), 1e-3))
        dimension = n_free + (1 if davidson else 0)
        sigma_theta = ELO_SCALE * self.prior_sd

        # The free parameters' position in the Newton system: every player but the anchor, whose
        # rating is the gauge and is pinned at zero exactly, then ln(nu) if it is being fitted.
        slot = np.full(len(players), -1, dtype=np.int64)
        for player, position in free.items():
            slot[index[player]] = position
        slot_left, slot_right = slot[left], slot[right]
        nu_slot = n_free

        # Nothing to solve for: one player and no draw term is already the whole answer, and a
        # zero-by-zero Newton system is an exception rather than a fixed point.
        converged = dimension == 0
        iterations = 0
        gradient = np.zeros(dimension, dtype=np.float64)
        hessian = np.zeros((dimension, dimension), dtype=np.float64)
        newton_steps = self.max_iterations if dimension else 0
        while iterations < newton_steps:
            iterations += 1
            a = theta[left]
            b = theta[right]
            p_left, p_right, p_draw = _outcome_probabilities(a, b, log_nu, davidson)

            gradient[:] = 0.0
            hessian[:] = 0.0
            grad_left = wins_left + 0.5 * drawn - total * (p_left + 0.5 * p_draw)
            grad_right = wins_right + 0.5 * drawn - total * (p_right + 0.5 * p_draw)
            _accumulate(gradient, slot_left, grad_left)
            _accumulate(gradient, slot_right, grad_right)

            g_left = p_left + 0.5 * p_draw
            g_right = p_right + 0.5 * p_draw
            _accumulate2(
                hessian, slot_left, slot_left, -total * (p_left + 0.25 * p_draw - g_left**2)
            )
            _accumulate2(
                hessian, slot_right, slot_right, -total * (p_right + 0.25 * p_draw - g_right**2)
            )
            cross = -total * (0.25 * p_draw - g_left * g_right)
            _accumulate2(hessian, slot_left, slot_right, cross)
            _accumulate2(hessian, slot_right, slot_left, cross)

            if davidson:
                gradient[nu_slot] += float(np.sum(drawn - total * p_draw))
                gradient[nu_slot] -= log_nu / (_LOG_NU_PRIOR_SD**2)
                left_nu = -total * (0.5 * p_draw - g_left * p_draw)
                right_nu = -total * (0.5 * p_draw - g_right * p_draw)
                _accumulate2(hessian, slot_left, np.full_like(slot_left, nu_slot), left_nu)
                _accumulate2(hessian, np.full_like(slot_left, nu_slot), slot_left, left_nu)
                _accumulate2(hessian, slot_right, np.full_like(slot_right, nu_slot), right_nu)
                _accumulate2(hessian, np.full_like(slot_right, nu_slot), slot_right, right_nu)
                hessian[nu_slot, nu_slot] += float(np.sum(-total * (p_draw - p_draw**2)))
                hessian[nu_slot, nu_slot] -= 1.0 / (_LOG_NU_PRIOR_SD**2)

            for player, position in free.items():
                gradient[position] -= theta[index[player]] / (sigma_theta**2)
                hessian[position, position] -= 1.0 / (sigma_theta**2)

            step = np.linalg.solve(-hessian, gradient)
            for player, position in free.items():
                theta[index[player]] += step[position]
            if davidson:
                log_nu += step[nu_slot]
            if float(np.max(np.abs(step))) < self.tolerance:
                converged = True
                break

        covariance = np.linalg.inv(-hessian)
        se_theta = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

        self._theta = {player: float(theta[index[player]]) for player in players}
        self._nu = float(math.exp(log_nu)) if davidson else 0.0
        self._free = dict(free)
        self._covariance = covariance

        rating = {player: self._theta[player] / ELO_SCALE for player in players}
        se = {
            player: (float(se_theta[free[player]]) / ELO_SCALE if player in free else 0.0)
            for player in players
        }
        counts = results.n_games()
        table = RatingTable(
            rating=rating,
            se=se,
            anchor=self.anchor,
            draw_nu=self._nu if davidson else None,
            n_games={player: counts.get(player, 0) for player in players},
            transitivity_residual=self._residual(results),
            converged=converged,
            iterations=iterations,
        )
        self._table = table
        return table

    def predict(self, a: str, b: str) -> float:
        """P(a scores against b), draws counted as half a win.

        An id the fit has never seen reads at the prior's mean, which is the anchor's own
        rating: a player with no games is not assumed to be average, it is assumed to be what
        the prior says before any evidence, and every real opponent of a gate has a thousand
        games behind it.
        """
        davidson = self._nu > 0.0
        p_a, _p_b, p_draw = _outcome_probabilities(
            np.array([self._theta.get(a, 0.0)]),
            np.array([self._theta.get(b, 0.0)]),
            math.log(self._nu) if davidson else 0.0,
            davidson,
        )
        return float(p_a[0] + 0.5 * p_draw[0])

    def difference_se(self, a: str, b: str) -> float:
        """The standard error of ``rating[a] - rating[b]``, in Elo.

        From the 2x2 block of the covariance, never from the two marginals: the marginals share
        the gauge, and adding them in quadrature counts that shared uncertainty twice.
        """
        if self._covariance is None:
            raise RuntimeError("difference_se needs a fit; call fit() first")
        variance = 0.0
        index_a = self._free.get(a)
        index_b = self._free.get(b)
        if index_a is not None:
            variance += float(self._covariance[index_a, index_a])
        if index_b is not None:
            variance += float(self._covariance[index_b, index_b])
        if index_a is not None and index_b is not None:
            variance -= 2.0 * float(self._covariance[index_a, index_b])
        return math.sqrt(max(variance, 0.0)) / ELO_SCALE

    def transitivity_residual(self, results: ResultView) -> float:
        """The share of well-played pairs the fit cannot explain.

        A pair counts when it has at least ``min_pair_games`` games and its observed score is
        more than two standard errors from the fitted prediction. Near zero on transitive
        results; large on a population where A beats B beats C beats A, which is the case a
        single number per player cannot describe at all. Above about a tenth the scalar release
        metric is lying, and the ``Rater`` ABC is the seam a Nash averaging or alpha-rank
        implementation drops into.
        """
        if self._table is None:
            self.fit(results)
        return self._residual(results)

    def _residual(self, results: ResultView) -> float:
        eligible = 0
        contradicted = 0
        for a, b, record in results.pairs():
            if record.games < self.min_pair_games:
                continue
            eligible += 1
            predicted = self.predict(a, b)
            # The spread is the model's own, so it is never zero: a pair the fit calls even and
            # that went ten-nil contradicts it, and a binomial error taken at the observed rate
            # would have called that pair a perfect fit.
            se = math.sqrt(max(predicted * (1.0 - predicted), 1e-12) / record.games)
            if abs(record.score_a - predicted) > 2.0 * se:
                contradicted += 1
        return contradicted / eligible if eligible else 0.0

    # -- checkpoint ---------------------------------------------------------

    def save_checkpoint(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ratings.json").write_bytes(
            msgspec.json.encode({"table": self._table, "nu": self._nu, "theta": self._theta})
        )

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        path = folder / "ratings.json"
        if not path.exists():
            if strict:
                raise FileNotFoundError(str(path))
            print(f"no rating table at {path}; it will be refitted from the result log")
            return
        state = msgspec.json.decode(path.read_bytes())
        self._nu = float(state["nu"])
        self._theta = {str(k): float(v) for k, v in state["theta"].items()}
        stored = state["table"]
        self._table = (
            msgspec.convert(stored, type=RatingTable) if stored is not None else None
        )


def _outcome_probabilities(
    a: np.ndarray, b: np.ndarray, log_nu: float, davidson: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(P(left wins), P(right wins), P(draw))`` for every pair at once.

    Through the log-sum-exp rather than through ``exp`` of the ratings: a pool spanning a
    thousand Elo has ``q`` ratios of ten to the two and a half, and the shifted form costs one
    subtraction and never overflows.
    """
    draw_term = log_nu + 0.5 * (a + b) if davidson else np.full_like(a, -np.inf)
    largest = np.maximum(np.maximum(a, b), draw_term)
    exp_a = np.exp(a - largest)
    exp_b = np.exp(b - largest)
    exp_d = np.exp(draw_term - largest) if davidson else np.zeros_like(a)
    total = exp_a + exp_b + exp_d
    return exp_a / total, exp_b / total, exp_d / total


def _accumulate(target: np.ndarray, positions: np.ndarray, values: np.ndarray) -> None:
    """``target[positions] += values``, skipping the pinned anchor's -1."""
    keep = positions >= 0
    np.add.at(target, positions[keep], values[keep])


def _accumulate2(
    target: np.ndarray, rows: np.ndarray, columns: np.ndarray, values: np.ndarray
) -> None:
    keep = (rows >= 0) & (columns >= 0)
    np.add.at(target, (rows[keep], columns[keep]), values[keep])
