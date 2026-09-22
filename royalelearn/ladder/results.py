"""The result log: every evaluation battle ever played, as lines, and the cache derived from it.

The log is the source of truth for every rating this repository reports. It is append-only and
one line per battle rather than an aggregate per pair, for three reasons: a batch fit needs the
dense matrix and an aggregate cannot be re-cut by context or by kind; draws have to be counted
separately, because five wins and five losses carry a completely different variance from ten
draws and a single ``wins: float`` with a draw adding a half cannot tell them apart; and a
method that is not a scalar rating at all -- Nash averaging, alpha-rank -- needs the games
themselves.

``context`` is what separates results that must not be pooled. It hashes the env factory, the
observation builder's own spec, the deck protocol and the engine build, so a policy that was
shown the opponent's hand and one that was not are never pooled: rating them together would
measure the information rather than the player. Results from different contexts are kept in one
file and separated on the way out.

It is a separator rather than a guarantee, and the gap is worth naming because a digest invites
the stronger reading. The engine build it hashes covers the tables compiled into the engine, and
NOT the card table, which the engine reads at runtime. So two games played against different
card tables carry the same context and are pooled by ``eval_view``, and a rating fitted across a
card-table change is fitted across a change in the game itself with nothing recording that it
happened. This is not fixable here -- the missing stamp belongs to the engine's Python surface,
which exposes no card-table vintage to read -- so the honest statement is that a shared context
means the recorded inputs matched, not that the two policies played the same game.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import BinaryIO

import msgspec

__all__ = [
    "KIND_EVAL",
    "KIND_TRAIN",
    "Aggregate",
    "GameResult",
    "PairRecord",
    "ResultLog",
    "ResultView",
    "context_digest",
    "rebuild",
]

#: Separates the two ids in an aggregate key. A unit separator cannot occur in an id -- they are
#: snapshot digests, step counts and scripted names -- so a key splits back into its pair exactly.
PAIR_SEPARATOR = "\x1f"

#: What ``kind`` a result may carry. Training results are recorded and excluded from the fit by
#: default: they are PFSP-selected and therefore biased towards hard matchups.
KIND_EVAL = "eval"
KIND_TRAIN = "train"


class GameResult(msgspec.Struct, frozen=True):
    """One battle, from ``a``'s point of view.

    ``score_a`` is 1 for a win, 0.5 for a draw and 0 for a loss. ``seed_index`` is the position
    in the run's frozen evaluation seed set rather than the seed itself, so a line stays short
    and the seed set stays the one authority on what that seed was.
    """

    a: str
    b: str
    score_a: float
    seed_index: int
    side_a: str
    context: str
    kind: str
    run_id: str
    iteration: int
    wall: str


class PairRecord(msgspec.Struct):
    """One unordered pair's record, ``a`` being the lexicographically smaller id.

    Draws are their own count. That is the whole reason this repository keeps its own log: the
    variance of a score rate with a draw rate ``d`` is ``(p(1-p) - d/4)/n``, strictly less than
    the binomial, and a rater that cannot see ``d`` cannot use that.
    """

    wins_a: int = 0
    draws: int = 0
    wins_b: int = 0

    @property
    def games(self) -> int:
        return self.wins_a + self.draws + self.wins_b

    @property
    def score_a(self) -> float:
        """``a``'s score rate over the pair, draws counted as half a win."""
        return (self.wins_a + 0.5 * self.draws) / self.games if self.games else 0.5


class Aggregate(msgspec.Struct):
    """The derived cache: the log reduced to per-pair counts.

    It is a cache and nothing else. ``rebuild(games) == aggregate`` is a test, and a cache that
    disagrees with the log is discarded rather than trusted.
    """

    n_games: int = 0
    pairs: dict[str, PairRecord] = msgspec.field(default_factory=dict)
    by_player: dict[str, int] = msgspec.field(default_factory=dict)
    contexts: dict[str, int] = msgspec.field(default_factory=dict)
    kinds: dict[str, int] = msgspec.field(default_factory=dict)


def context_digest(
    env_factory_digest: str, obs_digest: str, deck_protocol: str, engine_build_digest: str
) -> str:
    """The sixteen hex characters that say two results may be pooled.

    The observation builder's digest is in here because it carries what the policy was shown;
    the deck protocol because strength in this game is a function of the deck pair and deck
    matchups are the main source of non-transitivity, so one declared protocol is one ladder.
    """
    joined = "\x00".join(
        (env_factory_digest, obs_digest, deck_protocol, engine_build_digest)
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _pair_key(a: str, b: str) -> str:
    return f"{a}{PAIR_SEPARATOR}{b}"


def rebuild(games: Iterable[GameResult]) -> Aggregate:
    """The aggregate a sequence of results implies, from nothing but the results."""
    agg = Aggregate()
    for game in games:
        agg.n_games += 1
        agg.by_player[game.a] = agg.by_player.get(game.a, 0) + 1
        agg.by_player[game.b] = agg.by_player.get(game.b, 0) + 1
        agg.contexts[game.context] = agg.contexts.get(game.context, 0) + 1
        agg.kinds[game.kind] = agg.kinds.get(game.kind, 0) + 1
        low, high = (game.a, game.b) if game.a <= game.b else (game.b, game.a)
        score_low = game.score_a if low == game.a else 1.0 - game.score_a
        record = agg.pairs.setdefault(_pair_key(low, high), PairRecord())
        if score_low == 1.0:
            record.wins_a += 1
        elif score_low == 0.0:
            record.wins_b += 1
        else:
            record.draws += 1
    return agg


class ResultView:
    """A read-only cut of the log: one context, one kind, and the pair counts over it.

    Everything a rater reads goes through here, and everything here is ordered -- players
    sorted, pairs sorted -- so that a fit is a pure function of the games and not of the order
    in which they happened to be written.
    """

    def __init__(self, games: Sequence[GameResult], *, context: str | None = None) -> None:
        self._games = tuple(games)
        self._context = context
        self._aggregate = rebuild(self._games)
        self._players = tuple(sorted(self._aggregate.by_player))

    @property
    def context(self) -> str | None:
        return self._context

    @property
    def games(self) -> tuple[GameResult, ...]:
        return self._games

    @property
    def aggregate(self) -> Aggregate:
        return self._aggregate

    def players(self) -> tuple[str, ...]:
        """Every id that played, sorted, so an index map is the same on any machine."""
        return self._players

    def pairs(self) -> Iterator[tuple[str, str, PairRecord]]:
        """Every pair that met, in sorted order, ``a`` the smaller id."""
        for key in sorted(self._aggregate.pairs):
            low, high = key.split(PAIR_SEPARATOR, 1)
            yield low, high, self._aggregate.pairs[key]

    def record(self, a: str, b: str) -> PairRecord:
        """One pair's record from ``a``'s side, empty when the two never met."""
        low, high = (a, b) if a <= b else (b, a)
        stored = self._aggregate.pairs.get(_pair_key(low, high))
        if stored is None:
            return PairRecord()
        if low == a:
            return msgspec.structs.replace(stored)
        return PairRecord(wins_a=stored.wins_b, draws=stored.draws, wins_b=stored.wins_a)

    def n_games(self) -> dict[str, int]:
        return dict(self._aggregate.by_player)

    def draw_rate(self) -> float:
        """The share of battles that ended level, which is what decides the rater's draw model."""
        total = self._aggregate.n_games
        if not total:
            return 0.0
        draws = sum(record.draws for record in self._aggregate.pairs.values())
        return draws / total

    def __len__(self) -> int:
        return len(self._games)


class ResultLog:
    """The append-only ``ladder/games.jsonl`` and the aggregate beside it.

    Writes are append, flush and ``os.fsync`` per batch, so a crash truncates at most the last
    line; the reader drops an unparsable final line and says nothing about it, because that is
    exactly what a crash looks like and there is nothing to decide.
    """

    FORMAT_VERSION = 1

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.aggregate_path = self.path.with_name("aggregate.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._encoder = msgspec.json.Encoder()
        self._decoder = msgspec.json.Decoder(GameResult)
        self._handle: BinaryIO | None = None

    # -- writing ------------------------------------------------------------

    def append(self, result: GameResult) -> None:
        self.extend((result,))

    def extend(self, results: Iterable[GameResult]) -> int:
        """Append a batch, flush it and fsync it. Returns how many lines were written."""
        batch = [self._checked(result) for result in results]
        if not batch:
            return 0
        handle = self._open()
        for result in batch:
            handle.write(self._encoder.encode(result))
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        return len(batch)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _open(self) -> BinaryIO:
        if self._handle is None:
            self._handle = self.path.open("ab")
        return self._handle

    @staticmethod
    def _checked(result: GameResult) -> GameResult:
        if result.score_a not in (0.0, 0.5, 1.0):
            raise ValueError(f"score_a must be 0, 0.5 or 1, not {result.score_a!r}")
        if result.a == result.b:
            raise ValueError(f"a battle needs two players, both are {result.a!r}")
        return result

    # -- reading ------------------------------------------------------------

    def read(self) -> list[GameResult]:
        """Every line of the log, tolerating a truncated last one."""
        if not self.path.exists():
            return []
        raw = self.path.read_bytes().split(b"\n")
        games: list[GameResult] = []
        for index, line in enumerate(raw):
            if not line.strip():
                continue
            try:
                games.append(self._decoder.decode(line))
            except msgspec.DecodeError:
                if index == len(raw) - 1:
                    break  # a crash truncates the last line and nothing before it
                raise
        return games

    def aggregate(self) -> Aggregate:
        """The cache, rebuilt from the log. Cheap enough to be exact rather than incremental."""
        return rebuild(self.read())

    def save_aggregate(self) -> Path:
        """Write the derived cache beside the log."""
        self.aggregate_path.write_bytes(msgspec.json.encode(self.aggregate()))
        return self.aggregate_path

    def load_aggregate(self) -> Aggregate | None:
        """The cache on disk, or None when there is none."""
        if not self.aggregate_path.exists():
            return None
        return msgspec.json.decode(self.aggregate_path.read_bytes(), type=Aggregate)

    def view(
        self,
        *,
        kind: str | None = KIND_EVAL,
        context: str | None = None,
        pool_contexts: bool = False,
    ) -> ResultView:
        """One cut of the log.

        A view over more than one context has to be asked for by name. Two contexts are two
        different games, and a rating that pooled them would be a number about neither.
        """
        games = [game for game in self.read() if kind is None or game.kind == kind]
        if context is not None:
            games = [game for game in games if game.context == context]
        elif not pool_contexts:
            present = {game.context for game in games}
            if len(present) > 1:
                raise ValueError(
                    f"the log holds results from {len(present)} contexts "
                    f"({', '.join(sorted(present))}); name one, or pass pool_contexts=True to "
                    "state that pooling them is meant"
                )
        return ResultView(games, context=context)

    def eval_view(self, context: str | None = None) -> ResultView:
        """The evaluation results of one context: what the authoritative rating is fitted to."""
        return self.view(kind=KIND_EVAL, context=context)
