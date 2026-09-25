"""The scripted opponents a run can name, and what each one is for.

Two of them are the ladder's anchors and existed from the start: a bot that never plays and a bot
that plays at random. A learner that beats those has shown very little, so the other four are
RoyaleGym's one-sentence strategies, there so a run can be measured against something that plays.

``ladder.scripted_opponents`` is who the scripted share of the mixture trains against. It
defaults to the two anchors: noop never plays a card and random_legal passes nine decisions in
ten. The second half of this file holds the list to four things: the default draws exactly what
runs drew before the field existed, a bad list fails at load, the matchmaker draws exactly the
listed opponents, and a run's workers really play them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import config as cfg
from royalelearn.errors import PreflightError
from royalelearn.ladder.matchmaker import MixMatchmaker
from royalelearn.ladder.pool import SCRIPTED_IDS, LadderPool
from royalelearn.ladder.results import KIND_TRAIN, ResultLog
from royalelearn.rollout.scripted import (
    SCRIPTED_NAMES,
    build_opponent,
    scripted_id,
)

PUSH = scripted_id("push")
DEFEND = scripted_id("defend")


def test_every_name_builds_an_opponent_that_acts() -> None:
    """A name in the table that cannot be built is a run that dies at its first assignment."""
    mask = np.zeros(64, dtype=bool)
    mask[0] = True
    rng = np.random.default_rng(0)
    for name in SCRIPTED_NAMES:
        opponent: Any = build_opponent(name)
        action = opponent.act({}, mask, rng)
        assert isinstance(action, int | np.integer), name
        assert bool(mask[int(action)]), f"{name} chose an action its mask forbids"


def test_each_call_builds_a_fresh_opponent() -> None:
    """Several of these keep no state today and one of them may tomorrow.

    The same instance handed to eight environments is a shared mutable, which is the bug this
    package refuses elsewhere, and RoyaleGym's own ladder() is a function for the same reason.
    """
    for name in SCRIPTED_NAMES:
        assert build_opponent(name) is not build_opponent(name)


def test_an_unknown_name_says_what_the_known_ones_are() -> None:
    with pytest.raises(KeyError) as raised:
        build_opponent("agressive")
    message = str(raised.value)
    for name in SCRIPTED_NAMES:
        assert name in message


def test_the_order_of_the_table_is_part_of_the_protocol() -> None:
    """The parent sends an index and the worker looks it up, so a new name is APPENDED.

    Inserting one would silently change which opponent every recorded run had played, and every
    id in the result log with it.
    """
    assert SCRIPTED_NAMES[:2] == ("noop", "random_legal")
    assert len(set(SCRIPTED_NAMES)) == len(SCRIPTED_NAMES)
    assert scripted_id(SCRIPTED_NAMES[0]) == "scripted:noop"


# -- who the scripted share trains against -------------------------------------

SEED = 20260925


def _with_ladder(**ladder: Any) -> cfg.RunConfig:
    """The laptop profile with some ladder fields changed, and nothing else."""
    base = cfg.laptop()
    return msgspec.structs.replace(base, ladder=msgspec.structs.replace(base.ladder, **ladder))


def _problems(config: cfg.RunConfig) -> list[str]:
    return [problem for problem in cfg.check_consistency(config) if "scripted_opponents" in problem]


def test_the_default_is_the_two_anchors_everywhere_a_run_starts_from() -> None:
    """The field is new and every run before it met exactly these two, so the default keeps
    what every earlier run played."""
    assert cfg.LadderConfig().scripted_opponents == SCRIPTED_IDS
    for name in cfg.PROFILES:
        assert cfg.profile(name).ladder.scripted_opponents == SCRIPTED_IDS, name


def test_the_config_default_draws_what_the_matchmaker_default_draws(tmp_path: Path) -> None:
    """In order, not only as a set, because the draw is an index into the list.

    A run passes the config's list and most tests build the matchmaker without one, so those
    tests describe a run only while the two defaults agree. Both sides here run the same
    ``assign``, so this cannot see a change to the draw itself. The pin below does.
    """
    pool = LadderPool(ResultLog(tmp_path / "games.jsonl"))
    ladder = cfg.LadderConfig()
    constructor = MixMatchmaker(SEED, ladder, n_battles=96)
    configured = MixMatchmaker(SEED, ladder, n_battles=96, scripted_ids=ladder.scripted_opponents)
    scripted = 0
    for battle in range(96):
        for ordinal in range(10):
            expected = constructor.assign(battle, ordinal, pool, None)
            assert configured.assign(battle, ordinal, pool, None) == expected
            scripted += expected.opponent_id is not None
    assert scripted > 0, "an empty pool plays scripted in the pool slots too"


def test_the_default_draw_is_the_one_runs_made_before_the_field(tmp_path: Path) -> None:
    """Pinned, not compared with a second matchmaker running the same code.

    The digest was recorded from 919406c, the commit before ``scripted_opponents`` existed, with
    the same seed, rectangle and empty pool: 480 scripted draws, from the 14 scripted battles
    and the 34 pool battles standing in for them, ten episodes each. A change to the default list,
    its order or the way ``assign`` reads it hands resumed and repeated runs other opponents,
    and moves this digest.
    """
    pool = LadderPool(ResultLog(tmp_path / "games.jsonl"))
    ladder = cfg.LadderConfig()
    matchmaker = MixMatchmaker(SEED, ladder, n_battles=96, scripted_ids=ladder.scripted_opponents)
    drawn = []
    for battle in range(96):
        for ordinal in range(10):
            assignment = matchmaker.assign(battle, ordinal, pool, None)
            drawn.append(f"{assignment.opponent_id}/{assignment.learner_seat}")
    assert sum(entry.startswith("scripted:") for entry in drawn) == 480
    digest = hashlib.sha256("\n".join(drawn).encode()).hexdigest()
    assert digest == "a972e51f48dd2ba920b8b3950df63eb30ffb20d76a681cec437063ba5e26ad53"


def test_an_unknown_opponent_fails_at_load_naming_it_and_the_known_ones() -> None:
    """Not at the first plan, where ``plan.opponent_index`` would raise in the parent after
    preflight has already been paid for. A bare name is a typo too: the field takes full ids, as
    probe_opponents does."""
    for typo in ("scripted:agressive", "push"):
        loaded = cfg.load_config({"ladder": {"scripted_opponents": [typo]}}, say=None)
        with pytest.raises(PreflightError) as raised:
            cfg.validate(loaded)
        message = str(raised.value)
        assert f"ladder.scripted_opponents names {typo!r}" in message
        for name in SCRIPTED_NAMES:
            assert scripted_id(name) in message


def test_an_opponent_named_twice_is_refused() -> None:
    """The draw is uniform over the list, so a repeat is a weight nobody wrote down."""
    problems = _problems(_with_ladder(scripted_opponents=(PUSH, DEFEND, PUSH)))
    assert len(problems) == 1 and "more than once" in problems[0] and PUSH in problems[0]
    assert _problems(_with_ladder(scripted_opponents=(PUSH, DEFEND))) == []


def test_an_empty_list_is_refused_while_any_battle_would_draw_from_it() -> None:
    """A scripted battle draws from the list, and so does a pool battle until the first snapshot
    is admitted. Only a mixture with neither may leave it empty."""
    shipped = _problems(_with_ladder(scripted_opponents=()))
    assert len(shipped) == 1 and "scripted share" in shipped[0]
    before_the_first_snapshot = _problems(_with_ladder(scripted_opponents=(), mix=(0.5, 0.5, 0.0)))
    assert len(before_the_first_snapshot) == 1 and "pool" in before_the_first_snapshot[0]
    assert cfg.check_consistency(_with_ladder(scripted_opponents=(), mix=(1.0, 0.0, 0.0))) == []

    # The matchmaker refuses the same thing, for a caller that never went through a config: once
    # with only a scripted share, and once with a pool share and no scripted share.
    for mix in ((0.0, 0.0, 1.0), (0.5, 0.5, 0.0)):
        with pytest.raises(ValueError, match="no scripted opponent"):
            MixMatchmaker(SEED, cfg.LadderConfig(mix=mix), scripted_ids=())
    MixMatchmaker(SEED, cfg.LadderConfig(mix=(1.0, 0.0, 0.0)), scripted_ids=())


def test_the_scripted_role_draws_every_listed_opponent_and_nothing_else(tmp_path: Path) -> None:
    """Over a thousand assignments, from a scripted share and from pool slots standing in for
    one. Neither anchor is listed, so an anchor drawn here is the list being ignored."""
    empty = LadderPool(ResultLog(tmp_path / "games.jsonl"))
    listed = (PUSH, DEFEND)
    for mix in ((0.0, 0.0, 1.0), (0.5, 0.5, 0.0)):
        matchmaker = MixMatchmaker(
            SEED, cfg.LadderConfig(mix=mix), n_battles=40, scripted_ids=listed
        )
        drawn: dict[str, int] = {}
        for battle in range(40):
            for ordinal in range(25):
                opponent = matchmaker.assign(battle, ordinal, empty, None).opponent_id
                if opponent is not None:
                    drawn[opponent] = drawn.get(opponent, 0) + 1
        assert set(drawn) == set(listed), (mix, drawn)
        assert min(drawn.values()) > 0.35 * sum(drawn.values()), (mix, drawn)


def _spy_on_scripted_play(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """How often each scripted opponent's ``act`` ran, by name, without changing what it does.

    On the class, so it counts every instance, including the ones a worker built before the
    spy was set. Each name is its own class, which is what lets a count name the opponent.
    """
    classes = {name: type(build_opponent(name)) for name in SCRIPTED_NAMES}
    assert len(set(classes.values())) == len(SCRIPTED_NAMES)
    calls = dict.fromkeys(SCRIPTED_NAMES, 0)
    for name, kind in classes.items():
        original = kind.act

        def act(self: Any, *args: Any, _name: str = name, _original: Any = original) -> Any:
            calls[_name] += 1
            return _original(self, *args)

        monkeypatch.setattr(kind, "act", act)
    return calls


def test_a_run_plays_the_listed_opponent_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the config reaches the worker, the worker plays that opponent, and the result
    log files the games under its name.

    Three observables, because each is blind where another sees. The log names the opponent the
    parent assigned. The spy shows which opponent's code the worker ran: push, and neither
    anchor, which is what a run that ignored the list would have run. And push's own seats
    played cards, which the no-op anchor never does.
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from royalelearn.testing import coordinator, tiny_config

    calls = _spy_on_scripted_play(monkeypatch)
    config = tiny_config(
        tmp_path,
        ladder=cfg.LadderConfig(
            mix=(0.0, 0.0, 1.0),
            scripted_opponents=(PUSH,),
            candidate_every_env_steps=1_000_000,
            floor_admit_every_env_steps=1_000_000,
            eval_seed_count=4,
            refit_every_iterations=1_000_000,
        ),
    )
    with coordinator(config) as run:
        # Only training moves: the rating's and the gate's anchors are still the two, and the
        # rater still pins the first of them.
        assert run.pool.anchors == SCRIPTED_IDS
        assert run.gate.anchors == SCRIPTED_IDS
        assert run.rater.anchor == cfg.LadderConfig().rater.anchor == SCRIPTED_IDS[0]
        run.iterate()
        episodes = list(run.recent_episodes)
        games = [game for game in run.results.read() if game.kind == KIND_TRAIN]

    assert games, "the tiny config's step limit ends episodes inside an iteration"
    assert {game.b for game in games} == {PUSH}
    assert calls["push"] > 0
    assert {name for name, count in calls.items() if count} == {"push"}, calls
    pushed = [record for record in episodes if record.policy_id == PUSH]
    assert pushed, "push's own seats finished episodes"
    assert sum(record.cards_played for record in pushed) > 0
