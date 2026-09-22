"""The log is the source of truth, so what is tested is that nothing else can disagree with it.

The aggregate is a cache and is asserted to equal a rebuild; a crash-truncated last line is
tolerated and nothing before it is; and a view over two contexts has to be asked for by name,
because two contexts are two different games and a rating pooling them would be about neither.
"""

from __future__ import annotations

import re

import msgspec
import pytest

from royalelearn.ladder.results import (
    KIND_EVAL,
    KIND_TRAIN,
    GameResult,
    PairRecord,
    ResultLog,
    ResultView,
    context_digest,
    rebuild,
)


def _game(
    a: str, b: str, score_a: float, index: int = 0, context: str = "ctx", kind: str = KIND_EVAL
) -> GameResult:
    return GameResult(
        a=a,
        b=b,
        score_a=score_a,
        seed_index=index,
        side_a="blue" if index % 2 == 0 else "red",
        context=context,
        kind=kind,
        run_id="run",
        iteration=index,
        wall="2026-09-22T00:00:00Z",
    )


def _batch() -> list[GameResult]:
    return [
        _game("learner", "snap:1", 1.0, 0),
        _game("snap:1", "learner", 0.0, 1),
        _game("learner", "snap:1", 0.5, 2),
        _game("learner", "scripted:noop", 1.0, 3),
        _game("snap:1", "scripted:noop", 0.0, 4),
    ]


def test_the_log_round_trips(tmp_path) -> None:
    log = ResultLog(tmp_path / "ladder" / "games.jsonl")
    assert log.extend(_batch()) == 5
    log.close()
    assert log.read() == _batch()


def test_the_aggregate_equals_a_rebuild(tmp_path) -> None:
    log = ResultLog(tmp_path / "games.jsonl")
    log.extend(_batch())
    log.close()
    assert log.aggregate() == rebuild(log.read())
    log.save_aggregate()
    assert log.load_aggregate() == log.aggregate()


def test_a_pair_is_unordered_and_draws_are_their_own_count(tmp_path) -> None:
    log = ResultLog(tmp_path / "games.jsonl")
    log.extend(_batch())
    log.close()
    view = log.eval_view("ctx")
    record = view.record("learner", "snap:1")
    assert (record.wins_a, record.draws, record.wins_b) == (2, 1, 0)
    mirrored = view.record("snap:1", "learner")
    assert (mirrored.wins_a, mirrored.draws, mirrored.wins_b) == (0, 1, 2)
    assert record.score_a + mirrored.score_a == pytest.approx(1.0)
    # Ten draws and five wins with five losses are the same score rate and nothing else.
    assert PairRecord(wins_a=5, wins_b=5).score_a == PairRecord(draws=10).score_a
    assert PairRecord(wins_a=5, wins_b=5) != PairRecord(draws=10)


def test_a_truncated_last_line_is_tolerated(tmp_path) -> None:
    path = tmp_path / "games.jsonl"
    log = ResultLog(path)
    log.extend(_batch())
    log.close()
    raw = path.read_bytes()
    path.write_bytes(raw + msgspec.json.encode(_batch()[0])[:20])
    assert ResultLog(path).read() == _batch()


def test_a_corrupt_line_that_is_not_the_last_one_raises(tmp_path) -> None:
    path = tmp_path / "games.jsonl"
    log = ResultLog(path)
    log.extend(_batch())
    log.close()
    lines = path.read_bytes().split(b"\n")
    lines[1] = b'{"a": "broken"'
    path.write_bytes(b"\n".join(lines))
    with pytest.raises(msgspec.DecodeError):
        ResultLog(path).read()


def test_contexts_are_never_pooled_without_the_flag(tmp_path) -> None:
    log = ResultLog(tmp_path / "games.jsonl")
    log.extend([_game("learner", "snap:1", 1.0, 0, context="one")])
    log.extend([_game("learner", "snap:1", 1.0, 1, context="two")])
    log.close()
    with pytest.raises(ValueError, match="2 contexts"):
        log.eval_view()
    assert len(log.eval_view("one")) == 1
    assert len(log.view(kind=KIND_EVAL, pool_contexts=True)) == 2


def test_training_results_are_recorded_and_left_out(tmp_path) -> None:
    log = ResultLog(tmp_path / "games.jsonl")
    log.extend([*_batch(), _game("learner", "snap:1", 1.0, 9, kind=KIND_TRAIN)])
    log.close()
    assert len(log.eval_view("ctx")) == 5
    assert len(log.view(kind=KIND_TRAIN, context="ctx")) == 1
    assert log.aggregate().kinds == {KIND_EVAL: 5, KIND_TRAIN: 1}


def test_a_score_that_is_not_a_result_is_refused(tmp_path) -> None:
    log = ResultLog(tmp_path / "games.jsonl")
    with pytest.raises(ValueError, match=re.escape("0, 0.5 or 1")):
        log.append(_game("learner", "snap:1", 0.75))
    with pytest.raises(ValueError, match="two players"):
        log.append(_game("learner", "learner", 1.0))


def test_the_view_is_ordered_and_its_draw_rate_is_over_battles() -> None:
    view = ResultView(_batch())
    assert view.players() == tuple(sorted(view.players()))
    assert [pair[:2] for pair in view.pairs()] == sorted(pair[:2] for pair in view.pairs())
    assert view.draw_rate() == pytest.approx(1 / 5)
    assert view.n_games()["learner"] == 4


def test_the_context_digest_moves_with_every_part_of_it() -> None:
    base = context_digest("env", "obs", "deck", "engine")
    assert len(base) == 16
    assert base != context_digest("env2", "obs", "deck", "engine")
    assert base != context_digest("env", "obs2", "deck", "engine")
    assert base != context_digest("env", "obs", "deck2", "engine")
    assert base != context_digest("env", "obs", "deck", "engine2")
    # The parts are separated, so no rearrangement of them collides.
    assert context_digest("a", "bc", "d", "e") != context_digest("ab", "c", "d", "e")
