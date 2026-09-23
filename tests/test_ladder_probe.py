"""The probe that measures the live policy, and the four things it must not disturb.

Every other evaluation in this harness is a frozen snapshot against something: the gate names
``snap:v{n}`` and hands it to the runner. So nothing in a run ever measured the policy that is
actually being trained, and the two keys that claim to report it -- the score rate against each
scripted anchor -- asked about a pair no evaluator could write.

The probe is the measurement that was missing, and it is deliberately not the gate:

1. It names the live learner **and the step it was taken at**. A policy at iteration 5 and the
   same policy at iteration 50 are different players, and a fit that pooled their games would
   average two players into one rating.
2. Its games carry their own ``kind`` and are excluded from the authoritative cut. The tests
   below hold that from both sides: the fit and the gate's inputs are byte-identical across a
   probe, and the probe really did write the games that would have moved them.
3. It plays the first seeds of the frozen set rather than a fresh draw, so two probes are
   measurements of two policies on the same start states.
4. It is off unless a run asks for it, because it costs battles.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import config as cfg
from royalelearn.errors import PreflightError
from royalelearn.ladder.evaluate import EvalRunner, eval_seed_set
from royalelearn.ladder.pool import (
    LEARNER_ID,
    SCRIPTED_NOOP,
    SCRIPTED_RANDOM_LEGAL,
    LadderPool,
    is_learner,
    learner_probe_id,
)
from royalelearn.ladder.rating import BradleyTerryDavidsonRater
from royalelearn.ladder.results import KIND_EVAL, KIND_PROBE, GameResult, ResultLog

sys.path.insert(0, str(Path(__file__).parent))

SEED = 4242
NOOP = SCRIPTED_NOOP
RANDOM_LEGAL = SCRIPTED_RANDOM_LEGAL


class CountingPlayer:
    """Scores ``a`` by a fixed rate and remembers every battle it was asked for.

    A sampled player would make every assertion here a statement about a seed. This one makes
    the seeds, the sides and the score the test's own numbers.
    """

    def __init__(self, rate: float = 1.0) -> None:
        self.rate = rate
        self.played: list[tuple[str, str, int, int]] = []

    def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
        self.played.append((a, b, seed, a_seat))
        return self.rate

    def seeds_played(self) -> tuple[int, ...]:
        return tuple(seed for _a, _b, seed, _seat in self.played)


def _game(a: str, b: str, score_a: float, kind: str, index: int = 0) -> GameResult:
    return GameResult(
        a=a,
        b=b,
        score_a=score_a,
        seed_index=index,
        side_a="blue",
        context="ctx",
        kind=kind,
        run_id="run",
        iteration=0,
        wall="",
    )


def _runner(player: Any, log: ResultLog | None = None, *, kind: str = KIND_EVAL) -> EvalRunner:
    return EvalRunner(
        player,
        eval_seed_set(SEED, 40),
        master_seed=SEED,
        context="ctx",
        run_id="run",
        log=log,
        bootstrap_resamples=200,
        kind=kind,
    )


# -- who the live learner is -------------------------------------------------


def test_a_probe_names_the_learner_and_the_step_it_was_taken_at() -> None:
    """Two probes of one run are two players, and the id has to say so.

    The result log is keyed on the id. If every probe wrote ``learner``, a run's probes would
    pool into one row of the matrix and the rating fitted to it would be the average of every
    policy the run has ever had.
    """
    assert learner_probe_id(12_500_000) == "learner@12500000"
    assert learner_probe_id(0) != learner_probe_id(4_000_000)
    assert is_learner(learner_probe_id(12_500_000))
    assert is_learner(LEARNER_ID)
    assert not is_learner("snap:v17")
    assert not is_learner("scripted:noop")


# -- the games do not reach the fit ------------------------------------------


def test_a_probe_game_is_not_in_the_cut_the_rating_is_fitted_to(tmp_path: Path) -> None:
    log = ResultLog(tmp_path / "ladder" / "games.jsonl")
    log.extend(
        [
            _game("snap:v1", NOOP, 1.0, KIND_EVAL),
            _game(learner_probe_id(500), NOOP, 1.0, KIND_PROBE),
        ]
    )
    view = log.eval_view("ctx")
    assert [game.kind for game in view.games] == [KIND_EVAL]
    assert learner_probe_id(500) not in view.players()
    assert log.aggregate().kinds == {KIND_EVAL: 1, KIND_PROBE: 1}


def test_a_probe_moves_neither_the_ratings_nor_the_gate_s_inputs(tmp_path: Path) -> None:
    """The authoritative fit and the pair records the gate reads, across a probe.

    The probe writes into the same log as the gate, so "excluded" has to be checked on the way
    out rather than assumed from the write. Both halves matter: the fit must not move, and the
    probe must have written the games that would have moved it -- a probe that quietly played
    nothing would pass the first half on its own.
    """
    log = ResultLog(tmp_path / "ladder" / "games.jsonl")
    pool = LadderPool(log, context="ctx")
    log.extend(
        [
            _game("snap:v1", NOOP, 1.0, KIND_EVAL, index=index % 4)
            for index in range(40)
        ]
        + [_game("snap:v1", NOOP, 0.0, KIND_EVAL, index=4)]
    )
    rater = BradleyTerryDavidsonRater(prior_sd=400.0, anchor=NOOP, draws="half_win")
    before = rater.fit(pool.eval_view())
    before_record = pool.eval_view().record("snap:v1", NOOP)

    player = CountingPlayer(rate=1.0)
    runner = _runner(player, log, kind=KIND_PROBE)
    comparison = runner.compare(learner_probe_id(500), NOOP, games=10, iteration=3)

    assert comparison.n_games == 10, "the probe has to have played the games it reports"
    written = [game for game in log.read() if game.kind == KIND_PROBE]
    assert len(written) == 10
    assert {game.a for game in written} == {learner_probe_id(500)}

    after = rater.fit(pool.eval_view())
    assert after.rating == before.rating
    assert after.n_games == before.n_games
    assert learner_probe_id(500) not in after.rating
    after_record = pool.eval_view().record("snap:v1", NOOP)
    assert msgspec.json.encode(after_record) == msgspec.json.encode(before_record)


def test_an_evaluation_runner_refuses_a_kind_that_is_not_one_of_the_two_it_writes() -> None:
    with pytest.raises(ValueError, match="kind"):
        _runner(CountingPlayer(), kind="train")


# -- the same seeds, every probe ---------------------------------------------


def test_two_probes_play_the_same_seeds(tmp_path: Path) -> None:
    """A probe takes the FIRST ``games // 2`` seeds of the frozen set, not a fresh draw.

    The point of a probe is the comparison between two of them. Two policies measured on two
    different sets of start states differ by the states as much as by the policies, and at the
    handful of seeds a probe can afford that difference is most of the number.
    """
    seeds = eval_seed_set(SEED, 40)
    log = ResultLog(tmp_path / "ladder" / "games.jsonl")
    first_player, second_player = CountingPlayer(), CountingPlayer()
    _runner(first_player, log, kind=KIND_PROBE).compare(
        learner_probe_id(500), NOOP, games=6, iteration=1
    )
    _runner(second_player, log, kind=KIND_PROBE).compare(
        learner_probe_id(9_000), NOOP, games=6, iteration=9
    )
    assert first_player.seeds_played() == second_player.seeds_played()
    assert set(first_player.seeds_played()) == set(seeds.seeds[:3])
    # Each seed is played from both sides, so the unit of the score is the seed and the side
    # bias is removed rather than averaged away.
    assert sorted(seat for _a, _b, _seed, seat in first_player.played) == [0, 0, 0, 1, 1, 1]


# -- the config --------------------------------------------------------------


def test_the_shipped_config_does_not_probe_at_all() -> None:
    """Off by default: the probe plays real battles and every existing run's numbers stand."""
    ladder = cfg.LadderConfig()
    assert ladder.probe_every_iterations == 0
    assert ladder.probe_opponents == (SCRIPTED_NOOP, SCRIPTED_RANDOM_LEGAL)
    assert cfg.check_consistency(cfg.laptop()) == []


def test_an_unknown_probe_opponent_is_refused_at_config_time(run_config: Any) -> None:
    """Not at the first probe, which on the shipped cadence is an hour into a run."""
    config = msgspec.structs.replace(
        run_config,
        ladder=cfg.LadderConfig(
            probe_every_iterations=5, probe_opponents=("scripted:agressive",)
        ),
    )
    problems = cfg.check_consistency(config)
    assert any("scripted:agressive" in problem for problem in problems), problems
    with pytest.raises(PreflightError, match="scripted:agressive"):
        cfg.validate(config)


def test_a_probe_the_frozen_seed_set_cannot_cover_is_refused_at_config_time(
    run_config: Any,
) -> None:
    config = msgspec.structs.replace(
        run_config,
        ladder=cfg.LadderConfig(
            probe_every_iterations=1, probe_games=40, eval_seed_count=4
        ),
    )
    problems = cfg.check_consistency(config)
    assert any("probe_games" in problem for problem in problems), problems


def test_a_probe_of_one_battle_is_refused_at_config_time(run_config: Any) -> None:
    """A paired comparison is two battles per seed; one battle is a side, not a measurement."""
    config = msgspec.structs.replace(
        run_config,
        ladder=cfg.LadderConfig(probe_every_iterations=1, probe_games=1),
    )
    assert any("probe_games" in problem for problem in cfg.check_consistency(config))


def test_a_config_with_the_probe_off_is_not_held_to_the_probe_s_bounds(
    run_config: Any,
) -> None:
    """The bounds are about a probe that will run. A run that never probes keeps today's
    behaviour whatever its unused probe fields say, which is what makes the new fields safe to
    add to a config tree that is hashed into every run identity."""
    config = msgspec.structs.replace(
        run_config,
        ladder=cfg.LadderConfig(probe_every_iterations=0, probe_games=400, eval_seed_count=4),
    )
    assert cfg.check_consistency(config) == []


# -- what the row carries ----------------------------------------------------


def test_the_row_carries_the_score_with_its_n_and_its_interval(tmp_path: Path) -> None:
    """A score rate on its own is unreadable: at ten seeds a 0.60 and a 0.40 are one sample."""
    from royalelearn.metrics.records import ladder_fields

    log = ResultLog(tmp_path / "ladder" / "games.jsonl")
    pool = LadderPool(log, context="ctx")
    runner = _runner(CountingPlayer(rate=1.0), log, kind=KIND_PROBE)
    rungs = {
        NOOP: runner.compare(learner_probe_id(500), NOOP, games=10, iteration=3),
        "scripted:push": runner.compare(
            learner_probe_id(500), "scripted:push", games=10, iteration=3
        ),
    }
    fields = ladder_fields(pool, rungs=rungs, probe_seconds_frac=0.25)

    assert fields["ladder/score_vs_noop"] == 1.0
    assert fields["ladder/score_vs/noop"] == 1.0
    assert fields["ladder/score_vs_n/noop"] == 5
    assert fields["ladder/score_vs_ci95_lo/noop"] <= 1.0 <= fields["ladder/score_vs_ci95_hi/noop"]
    assert fields["ladder/score_vs/push"] == 1.0
    assert fields["ladder/probe_seconds_frac"] == 0.25
    # A rung that was not probed is absent rather than neutral: 0.5 is what a genuine even
    # contest looks like, and the two must not be one number.
    assert "ladder/score_vs_random_legal" not in fields
    assert "ladder/score_vs/random_legal" not in fields


def test_a_row_from_an_iteration_that_did_not_probe_carries_no_score(tmp_path: Path) -> None:
    from royalelearn.metrics.records import ladder_fields

    pool = LadderPool(ResultLog(tmp_path / "ladder" / "games.jsonl"), context="ctx")
    fields = ladder_fields(pool)
    assert not [key for key in fields if key.startswith("ladder/score_vs")]
    assert "ladder/probe_seconds_frac" not in fields


def test_every_key_the_probe_publishes_is_one_the_schema_knows(tmp_path: Path) -> None:
    from royalelearn.metrics import schema
    from royalelearn.metrics.records import ladder_fields, unknown_keys

    log = ResultLog(tmp_path / "ladder" / "games.jsonl")
    pool = LadderPool(log, context="ctx")
    runner = _runner(CountingPlayer(rate=0.5), log, kind=KIND_PROBE)
    rungs = {
        name: runner.compare(learner_probe_id(500), name, games=4, iteration=1)
        for name in (NOOP, RANDOM_LEGAL, "scripted:first_affordable")
    }
    fields = ladder_fields(pool, rungs=rungs, probe_seconds_frac=0.01)
    assert unknown_keys(fields) == ()
    assert schema.CONDITIONAL["ladder/probe_seconds_frac"]


# -- the coordinator ---------------------------------------------------------

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from test_coordinator import coordinator, tiny_config  # noqa: E402


def _probing_config(tmp_path: Path, **ladder: Any) -> cfg.RunConfig:
    settings: dict[str, Any] = {
        "mix": (1.0, 0.0, 0.0),
        "candidate_every_env_steps": 1_000_000,
        "floor_admit_every_env_steps": 1_000_000,
        "eval_seed_count": 4,
        "refit_every_iterations": 1_000_000,
        "probe_every_iterations": 1,
        "probe_games": 2,
        "probe_opponents": (NOOP,),
    }
    settings.update(ladder)
    return tiny_config(tmp_path, ladder=cfg.LadderConfig(**settings))


class ScriptedScore:
    """A battle player that decides the outcome without an environment.

    The cadence, the ids, the log and the row are what these tests are about, and a real battle
    costs about half a second on MockEngine. The one test that has to see the live weights play
    uses the real player.
    """

    def __init__(self) -> None:
        self.played: list[tuple[str, str, int, int]] = []

    def play(self, *, a: str, b: str, seed: int, a_seat: int, act_path: str) -> float:
        self.played.append((a, b, seed, a_seat))
        return 1.0 if a_seat == 0 else 0.5


def test_the_probe_runs_on_its_interval_and_not_otherwise(tmp_path: Path) -> None:
    with coordinator(_probing_config(tmp_path, probe_every_iterations=2)) as run:
        player = ScriptedScore()
        run.probe_runner.player = player
        run.iterate()
        assert player.played == [], "iteration 1 is not a multiple of 2"
        assert "ladder/score_vs_noop" not in run.rows[-1]
        run.iterate()
        assert len(player.played) == 2
        assert run.rows[-1]["ladder/score_vs_noop"] == 0.75
        assert run.rows[-1]["ladder/score_vs_n/noop"] == 1


def test_a_probe_names_the_step_it_was_taken_at_in_the_result_log(tmp_path: Path) -> None:
    with coordinator(_probing_config(tmp_path)) as run:
        run.probe_runner.player = ScriptedScore()
        run.iterate()
        first = run.cumulative_env_steps
        run.iterate()
        second = run.cumulative_env_steps
    games = [game for game in run.results.read() if game.kind == KIND_PROBE]
    assert first != second
    assert {game.a for game in games} == {learner_probe_id(first), learner_probe_id(second)}
    assert {game.b for game in games} == {NOOP}


def test_a_run_s_probe_games_stay_out_of_its_own_fit(tmp_path: Path) -> None:
    with coordinator(_probing_config(tmp_path, refit_every_iterations=1)) as run:
        run.probe_runner.player = ScriptedScore()
        run.iterate()
        probed = [game for game in run.results.read() if game.kind == KIND_PROBE]
        assert probed, "the probe has to have played, or this proves nothing"
        assert len(run.pool.eval_view()) == 0
        assert run.ratings is None
        assert not [key for key in run.rows[-1] if key.startswith("ladder/rating/")]


def test_the_default_run_never_probes(tmp_path: Path) -> None:
    with coordinator(tiny_config(tmp_path)) as run:
        run.probe_runner.player = ScriptedScore()
        run.iterate()
        run.iterate()
        assert run.probe_runner.games_played == 0
        assert run.rows[-1]["time/probe"] == 0.0
        assert "ladder/probe_seconds_frac" not in run.rows[-1]
        assert "ladder/score_vs_noop" not in run.rows[-1]


def test_the_probe_plays_the_weights_the_learner_has_now(tmp_path: Path) -> None:
    """Not an archived snapshot, which is what every other evaluation in the run plays.

    Two things say so and neither is a restatement of the implementation. The snapshot archive
    is EMPTY at this point, so a policy that came back at all did not come from it. And with
    every live parameter zeroed the masked logits are all equal, which makes the action a known
    function of the uniform -- the inverse CDF of a uniform distribution over the legal set --
    and a frozen copy taken before the change would not follow it.
    """
    import torch

    with coordinator(_probing_config(tmp_path)) as run:
        assert run.snapshot_store.list() == []
        policy = run._eval_actor(learner_probe_id(run.cumulative_env_steps))
        with torch.no_grad():
            for parameter in run.model.parameters():
                parameter.zero_()
        legal = [0, 3, 7, 11]
        mask = np.zeros(run.spec.n_actions, dtype=bool)
        mask[legal] = True
        obs = {
            "spatial": np.zeros(run.spec.obs_space["spatial"].shape, dtype=np.float32),
            "vector": np.zeros(run.spec.vector_size, dtype=np.float32),
            "action_mask": mask,
        }
        drawn = [policy(obs, uniform, None) for uniform in (0.0, 0.3, 0.6, 0.9)]
    assert drawn == legal, "a uniform distribution over the legal set, read by inverse CDF"
