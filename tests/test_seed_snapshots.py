"""``ladder.seed_snapshots``: frozen actors a run trains against from its first iteration.

Asked for an exploiter measurement: a copy trained only against one frozen policy, to see whether
that policy has a blind spot a cheap opponent finds. What the ladder owes that run is that the
pool holds the frozen policy before the first battle is planned, that pool battles meet it and
nothing else, that it plays with its own weights, and that it stays: through eviction and through
a resume that no longer has the folder it came from.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import msgspec
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from royalelearn import config as cfg
from royalelearn.errors import IdentityMismatch, PreflightError
from royalelearn.ladder.snapshots import SPEC_NAME, weights_sha256
from royalelearn.testing import coordinator, tiny_config

#: Every battle a pool battle: no mirror, no scripted share.
POOL_ONLY = (0.0, 1.0, 0.0)


def frozen_actor(tmp_path: Path) -> tuple[Path, str]:
    """A snapshot folder as ``put`` writes one, and the sha256 of its weights.

    From a tiny run one iteration in, so its weights are not the ones the seeded run starts
    with: the same master seed builds the same initial actor.
    """
    with coordinator(tiny_config(tmp_path / "source")) as source:
        source.iterate()
        digest = source.snapshot_store.put("frozen", source.model, {"step": 1})
        folder = source.snapshot_store.root / digest
    return folder, weights_sha256(folder)


def seeded(tmp_path: Path, folder: Path, sha256: str, **ladder: Any) -> cfg.RunConfig:
    base = tiny_config(tmp_path / "run")
    seeds = (cfg.SeedSnapshot(name="frozen", path=str(folder), sha256=sha256),)
    return msgspec.structs.replace(
        base, ladder=msgspec.structs.replace(base.ladder, seed_snapshots=seeds, **ladder)
    )


def training_opponents(run_dir: Path) -> list[str]:
    """Who the learner met in every training battle that finished, from the result log."""
    text = (Path(run_dir) / "ladder" / "games.jsonl").read_text(encoding="utf-8")
    games = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [game["b"] for game in games if game["kind"] == "train"]


def test_pool_battles_meet_the_seed_and_nothing_else(tmp_path: Path) -> None:
    """With every battle a pool battle, each finished one was against the seed.

    Without the seed a pool battle plays a scripted opponent until a snapshot is admitted, and
    the tiny run's gate never admits one, so this would meet the anchors.
    """
    folder, sha256 = frozen_actor(tmp_path)
    with coordinator(seeded(tmp_path, folder, sha256, mix=POOL_ONLY)) as run:
        assert run.pool.sampler() == ("seed:frozen",)
        assert run.snapshot_store.digest("seed:frozen") == folder.name, (
            "the pool's seed is not the folder's weights"
        )
        run.iterate()
        run.iterate()
        run_dir = run.run_dir
    met = training_opponents(run_dir)
    assert met, "no training battle finished, so nothing says who the pool battles met"
    assert set(met) == {"seed:frozen"}, f"pool battles met {sorted(set(met))}"


def test_a_seed_is_never_evicted_and_is_not_the_runs_v0(tmp_path: Path) -> None:
    """A budget that evicts every member it may: the seed is not one of them.

    ``v0`` is the run's own first snapshot, which ``rating_above_v0`` is measured from; a seed
    admitted before it must not take its place.
    """
    from royalelearn.api.ladder import RatingTable
    from royalelearn.ladder.eviction import HallOfFameEviction
    from royalelearn.ladder.pool import SCRIPTED_NOOP, LadderPool
    from royalelearn.ladder.results import ResultLog

    pool = LadderPool(ResultLog(tmp_path / "games.jsonl"), context="ctx")
    pool.seed("seed:frozen")
    members = [f"snap:v{index}" for index in range(6)]
    for index, member in enumerate(members):
        pool.add(member, step=(index + 1) * 1000)
    assert pool.v0 == "snap:v0"
    everyone = ["seed:frozen", *members]
    ratings = RatingTable(
        rating={member: 0.0 for member in everyone},
        se=dict.fromkeys(everyone, 12.0),
        anchor=SCRIPTED_NOOP,
        draw_nu=None,
        n_games=dict.fromkeys(everyone, 100),
        transitivity_residual=0.0,
        converged=True,
        iterations=4,
    )
    # Two places: v0 and the seed are protected, so everything else goes.
    evicted = HallOfFameEviction().select_for_eviction(pool=pool, ratings=ratings, max_sampled=2)
    assert sorted(evicted) == members[1:], evicted


def test_a_folder_whose_weights_are_not_the_pinned_ones_is_refused(tmp_path: Path) -> None:
    folder, sha256 = frozen_actor(tmp_path)
    wrong = "0" * 64
    with pytest.raises(PreflightError) as refused, coordinator(seeded(tmp_path, folder, wrong)):
        pass
    message = str(refused.value)
    assert sha256 in message and wrong in message, message


def test_a_folder_this_run_cannot_load_is_refused_by_field(tmp_path: Path) -> None:
    folder, sha256 = frozen_actor(tmp_path)
    other = tmp_path / "other-architecture"
    shutil.copytree(folder, other)
    spec = json.loads((other / SPEC_NAME).read_text(encoding="utf-8"))
    spec["arch_digest"] = "another-network"
    (other / SPEC_NAME).write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(IdentityMismatch) as refused, coordinator(seeded(tmp_path, other, sha256)):
        pass
    assert "arch_digest" in str(refused.value)


def test_a_resume_keeps_the_seed_without_its_folder(tmp_path: Path) -> None:
    """The run holds its own copy, so the folder it came from can be gone by the resume."""
    folder, sha256 = frozen_actor(tmp_path)
    config = seeded(tmp_path, folder, sha256, mix=POOL_ONLY)
    with coordinator(config) as first:
        first.iterate()
        saved = first.checkpoint()
        run_dir = first.run_dir
    shutil.rmtree(folder)
    with coordinator(config, resume=saved, run_dir=run_dir) as again:
        assert again.pool.seeded == ("seed:frozen",)
        assert again.pool.sampler() == ("seed:frozen",)
        again.iterate()
    assert set(training_opponents(run_dir)) == {"seed:frozen"}


@pytest.mark.parametrize(
    ("seeds", "said"),
    [
        (
            (
                cfg.SeedSnapshot(name="a", path="x", sha256="a" * 64),
                cfg.SeedSnapshot(name="a", path="y", sha256="b" * 64),
            ),
            "more than once",
        ),
        ((cfg.SeedSnapshot(name="a:b", path="x", sha256="a" * 64),), "no ':'"),
        ((cfg.SeedSnapshot(name="a", path="x", sha256="ABC"),), "64 lowercase hex"),
    ],
)
def test_a_seed_list_the_config_can_see_is_wrong_is_refused(
    tmp_path: Path, seeds: tuple[cfg.SeedSnapshot, ...], said: str
) -> None:
    base = tiny_config(tmp_path)
    config = msgspec.structs.replace(
        base, ladder=msgspec.structs.replace(base.ladder, seed_snapshots=seeds)
    )
    problems = cfg.check_consistency(config)
    assert any(said in problem for problem in problems), problems


def test_the_rows_say_whether_pool_battles_played_the_pool(tmp_path: Path) -> None:
    """``ladder/role_counts``: with every battle a pool battle and nothing admitted, each finished
    battle played a scripted opponent and is counted as pool_fallback; with a seed in the pool,
    each played the pool and none fell back. Summed over three iterations, so some finish."""

    def counts(config: cfg.RunConfig) -> dict[str, int]:
        total: dict[str, int] = {}
        with coordinator(config) as run:
            for _ in range(3):
                run.iterate()
                for key, value in run.rows[-1].items():
                    if key.startswith("ladder/role_counts/"):
                        total[key.rsplit("/", 1)[1]] = total.get(key.rsplit("/", 1)[1], 0) + value
        return total

    base = tiny_config(tmp_path / "empty")
    empty = counts(
        msgspec.structs.replace(base, ladder=msgspec.structs.replace(base.ladder, mix=POOL_ONLY))
    )
    assert empty["scripted"] > 0, empty
    fallback = empty["scripted"]
    assert empty == {"mirror": 0, "pool": 0, "scripted": fallback, "pool_fallback": fallback}

    folder, sha256 = frozen_actor(tmp_path)
    full = counts(seeded(tmp_path, folder, sha256, mix=POOL_ONLY))
    assert full["pool"] > 0, full
    assert full == {"mirror": 0, "pool": full["pool"], "scripted": 0, "pool_fallback": 0}


# -- weight --------------------------------------------------------------------------------------


def _weighted_ladder(*weights: float, weighting: str = "uniform") -> cfg.LadderConfig:
    seeds = tuple(
        cfg.SeedSnapshot(name=f"s{i}", path=f"seeds/s{i}", sha256=f"{i}" * 64, weight=w)
        for i, w in enumerate(weights)
    )
    return cfg.LadderConfig(mix=POOL_ONLY, pfsp_weighting=weighting, seed_snapshots=seeds)


def _seeded_pool(tmp_path: Path, names: list[str]) -> Any:
    from royalelearn.ladder.pool import LadderPool
    from royalelearn.ladder.results import ResultLog

    pool = LadderPool(ResultLog(tmp_path / "games.jsonl"), context="ctx")
    for name in names:
        pool.seed(name)
    return pool


def test_a_seeds_weight_multiplies_its_draw_weight() -> None:
    from royalelearn.ladder.matchmaker import MixMatchmaker

    matchmaker = MixMatchmaker(1, _weighted_ladder(2.0, 3.0))
    weights = matchmaker.weights(("seed:s0", "seed:s1"), None, "uniform")
    assert weights.tolist() == pytest.approx([0.4, 0.6])
    # Beside a member that is not a seed, which keeps a weight of one.
    weights = matchmaker.weights(("seed:s0", "seed:s1", "snap:v3"), None, "uniform")
    assert weights.tolist() == pytest.approx([2 / 6, 3 / 6, 1 / 6])


def test_the_battles_meet_the_seeds_in_their_weights(tmp_path: Path) -> None:
    from royalelearn.ladder.matchmaker import MixMatchmaker

    matchmaker = MixMatchmaker(1, _weighted_ladder(2.0, 3.0), n_battles=1)
    pool = _seeded_pool(tmp_path, ["seed:s0", "seed:s1"])
    met = [matchmaker.assign(0, ordinal, pool).opponent_id for ordinal in range(4000)]
    assert met.count("seed:s1") / len(met) == pytest.approx(0.6, abs=0.03)


def test_a_weight_of_one_is_the_draw_and_the_config_hash_there_were_before() -> None:
    from royalelearn.ladder.matchmaker import MixMatchmaker

    plain = MixMatchmaker(1, _weighted_ladder(1.0, 1.0, weighting="hard"))
    weights = plain.weights(("seed:s0", "seed:s1", "snap:v3"), None, "hard")
    assert weights.tolist() == pytest.approx([1 / 3] * 3)
    config = cfg.laptop()
    seeds = (
        cfg.SeedSnapshot("bc", "artifacts/bc", "a" * 64),
        cfg.SeedSnapshot("rl", "artifacts/rl", "b" * 64),
    )
    config = msgspec.structs.replace(
        config, ladder=msgspec.structs.replace(config.ladder, seed_snapshots=seeds)
    )
    # The hash of this config before the weight existed: a seeded run carries on.
    assert cfg.config_hash(config).startswith("098a9113520b")
    heavier = msgspec.structs.replace(
        config,
        ladder=msgspec.structs.replace(
            config.ladder, seed_snapshots=(msgspec.structs.replace(seeds[0], weight=2.0), seeds[1])
        ),
    )
    assert cfg.config_hash(heavier) != cfg.config_hash(config)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("inf")])
def test_a_weight_that_is_not_a_positive_number_is_refused(weight: float) -> None:
    problems = cfg.check_consistency(cfg.RunConfig(ladder=_weighted_ladder(1.0, weight)))
    assert [p for p in problems if "weight" in p and "s1" in p], problems
