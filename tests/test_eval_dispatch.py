"""A gate plays 2,200 battles one at a time in the parent, and nothing about them requires that.

Every battle a comparison plays is fully determined by ``(a, b, seed, a_seat, act_path)``:
``EnvBattlePlayer.play`` resets the environment with the seed and derives its acting uniforms from
the master seed and the stream path, so no battle can see another. The serial loop in
``EvalRunner.compare`` is therefore an implementation detail rather than part of the measurement.

These tests hold it to that. A player that plays the same battles in a DIFFERENT ORDER -- which is
what any farm does -- must produce the identical ``Comparison`` and the identical rows in the
result log. That is the property a parallel evaluator has to keep, and it is worth pinning before
one exists, because the failure it protects against is silent: an environment whose ``reset`` does
not fully reseed would give the same numbers serially every time and different ones in a farm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from royalelearn.ladder.evaluate import BattleRequest, EvalRunner, eval_seed_set
from royalelearn.ladder.results import ResultLog

GAMES = 12


def score_of(request: BattleRequest) -> float:
    """A fixed, order-independent answer, so a difference can only come from the dispatch.

    Deliberately NOT symmetric in the seat and not constant across seeds: a comparison assembled
    from the wrong request would land on a different number rather than on the same one.
    """
    return {0: 1.0, 1: 0.0, 2: 0.5}[(request.seed + request.a_seat) % 3]


class SerialPlayer:
    """What the runner has always had: one battle at a time, in the order asked."""

    def __init__(self) -> None:
        self.order: list[tuple[int, int]] = []

    def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
        request = BattleRequest(a=a, b=b, seed=seed, a_seat=a_seat, act_path=act_path)
        self.order.append((seed, a_seat))
        return score_of(request)


class ScrambledPlayer(SerialPlayer):
    """A farm, reduced to the only thing about it that can change an answer: the ORDER.

    It plays the requests back to front and returns the scores in REQUEST order, which is the
    contract a pool has to meet. If the runner assembled its comparison from the order things
    were played in rather than the order it asked for, this test would read a different score.
    """

    def play_many(self, requests: list[BattleRequest]) -> list[float]:
        scores = {}
        for request in reversed(requests):
            scores[request] = self.play(
                a=request.a,
                b=request.b,
                seed=request.seed,
                a_seat=request.a_seat,
                act_path=request.act_path,
            )
        return [scores[request] for request in requests]


class MiscountingPlayer(SerialPlayer):
    """A pool that loses a result. Silent otherwise: the comparison is simply shorter."""

    def play_many(self, requests: list[BattleRequest]) -> list[float]:
        return [score_of(request) for request in requests[:-1]]


def runner(player: Any, tmp_path: Path, name: str) -> EvalRunner:
    return EvalRunner(
        player,
        eval_seed_set(1234, count=64),
        master_seed=1234,
        run_id="test",
        log=ResultLog(tmp_path / f"{name}.jsonl"),
    )


def test_a_scrambled_dispatch_measures_the_same_thing(tmp_path: Path) -> None:
    """The property a farm has to keep, stated as an equality rather than as an intention."""
    serial = runner(SerialPlayer(), tmp_path, "serial")
    scrambled = runner(ScrambledPlayer(), tmp_path, "scrambled")

    first = serial.compare("snap:v0", "scripted:noop", games=GAMES, iteration=3)
    second = scrambled.compare("snap:v0", "scripted:noop", games=GAMES, iteration=3)

    assert second == first, "the same battles in a different order measured something else"
    assert second.seed_scores == first.seed_scores


def test_the_two_dispatches_write_the_same_log(tmp_path: Path) -> None:
    """The log is the run's record of which battle produced which result, so order is in it.

    A farm that returned the right aggregate from rows filed under the wrong seed would pass the
    test above and leave a result log nobody can re-derive the comparison from.
    """
    serial = runner(SerialPlayer(), tmp_path, "serial")
    scrambled = runner(ScrambledPlayer(), tmp_path, "scrambled")
    serial.compare("snap:v0", "scripted:noop", games=GAMES)
    scrambled.compare("snap:v0", "scripted:noop", games=GAMES)

    left = (tmp_path / "serial.jsonl").read_text(encoding="utf-8").splitlines()
    right = (tmp_path / "scrambled.jsonl").read_text(encoding="utf-8").splitlines()
    assert left == right


def test_the_scrambled_player_really_did_scramble(tmp_path: Path) -> None:
    """Otherwise the two tests above would pass against a play_many nobody ever called."""
    player = ScrambledPlayer()
    runner(player, tmp_path, "check").compare("snap:v0", "scripted:noop", games=GAMES)
    assert player.order, "play_many was never called"
    assert player.order == list(reversed(sorted(player.order))) or player.order != sorted(
        player.order
    ), "the order was not scrambled, so nothing about dispatch was tested"


def test_a_pool_that_returns_the_wrong_number_of_results_is_refused(tmp_path: Path) -> None:
    """A lost battle would otherwise shorten the comparison and nothing would say so."""
    with pytest.raises(ValueError, match="results"):
        runner(MiscountingPlayer(), tmp_path, "short").compare(
            "snap:v0", "scripted:noop", games=GAMES
        )
