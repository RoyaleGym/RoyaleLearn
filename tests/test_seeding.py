"""Name-addressed streams: pinned, non-colliding, and the same in another process."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from royalelearn import seeding

MASTER = 20260921

#: Pinned so that a change to the derivation is a failing test rather than a run that silently
#: stops reproducing the one before it. Recomputing these values is a decision.
PINNED_INTS = {
    "env/worker/0/shard/0/gen/0": 4572602439375946369,
    "match/battle/3/ordinal/7": 7258887212566958618,
    "act/iteration/1/cycle/2": 840595216339540220,
    "ppo/minibatch/iteration/0/epoch/0": 7324861131128626404,
    "eval/seed_set": 2177012627665394896,
}

PINNED_DRAWS = {
    "env/worker/0/shard/0/gen/0": [588, 581, 717],
    "match/battle/3/ordinal/7": [237, 244, 38],
    "act/iteration/1/cycle/2": [297, 214, 152],
    "ppo/minibatch/iteration/0/epoch/0": [972, 878, 35],
    "eval/seed_set": [71, 548, 525],
}


@pytest.mark.parametrize(("path", "expected"), sorted(PINNED_INTS.items()))
def test_derive_int_is_pinned(path: str, expected: int) -> None:
    assert seeding.derive_int(MASTER, path) == expected


@pytest.mark.parametrize(("path", "expected"), sorted(PINNED_DRAWS.items()))
def test_derive_generator_is_pinned(path: str, expected: list[int]) -> None:
    drawn = seeding.derive_generator(MASTER, path).integers(0, 1000, 3).tolist()
    assert drawn == expected


def test_derive_int_fits_in_a_signed_64_bit_seed() -> None:
    # The value crosses into ClashSelfPlayVecEnv.reset(seed=...), and gymnasium's seeding
    # refuses anything that does not fit a signed 64-bit integer.
    for battle in range(200):
        value = seeding.derive_int(MASTER, f"match/battle/{battle}/ordinal/0")
        assert 0 <= value < 2**63


def test_the_same_values_in_another_process() -> None:
    """A stream is a function of its name, not of the process that asked for it."""
    program = (
        "from royalelearn import seeding;"
        f"print(seeding.derive_int({MASTER}, 'act/iteration/1/cycle/2'));"
        f"print(seeding.derive_generator({MASTER}, 'eval/seed_set').integers(0, 1000, 3).tolist())"
    )
    out = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert int(out[0]) == PINNED_INTS["act/iteration/1/cycle/2"]
    assert out[1] == str(PINNED_DRAWS["eval/seed_set"])


def test_a_new_stream_does_not_move_an_existing_one() -> None:
    """The property positional spawning does not have: adding a consumer perturbs nothing."""
    before = {path: seeding.derive_int(MASTER, path) for path in PINNED_INTS}
    seeding.derive_generator(MASTER, "some/new/consumer/added/later").random(1000)
    after = {path: seeding.derive_int(MASTER, path) for path in PINNED_INTS}
    assert after == before


def test_neighbouring_names_do_not_collide() -> None:
    """Names one character apart, and names of the same shape, draw different numbers."""
    names = [
        "act/iteration/1/cycle/2",
        "act/iteration/1/cycle/3",
        "act/iteration/2/cycle/2",
        "env/worker/0/shard/0/gen/0",
        "env/worker/0/shard/1/gen/0",
        "env/worker/1/shard/0/gen/0",
        "match/battle/0/ordinal/0",
        "match/battle/0/ordinal/1",
    ]
    first_draws = {
        name: tuple(seeding.derive_generator(MASTER, name).random(4).tolist()) for name in names
    }
    assert len(set(first_draws.values())) == len(names)


def test_a_different_master_seed_is_a_different_stream() -> None:
    path = "act/iteration/0/cycle/0"
    assert seeding.derive_int(MASTER, path) != seeding.derive_int(MASTER + 1, path)


def test_seedseq_entropy_is_the_master_seed() -> None:
    seq = seeding.derive_seedseq(MASTER, "eval/seed_set")
    assert seq.entropy == MASTER
    assert len(seq.spawn_key) == 4
    assert all(isinstance(word, int) for word in seq.spawn_key)


def test_generators_are_pcg64() -> None:
    generator = seeding.derive_generator(MASTER, "torch/init")
    assert isinstance(generator.bit_generator, np.random.PCG64)


def test_every_stream_in_the_namespace_formats() -> None:
    """Every row of STREAMS is a template ``stream_path`` can fill, and no two are the same."""
    fields = {
        "worker": 0,
        "shard": 1,
        "generation": 2,
        "battle": 3,
        "ordinal": 4,
        "iteration": 5,
        "cycle": 6,
        "slot": 7,
        "epoch": 8,
        "comparison": "cand-vs-champ",
        "seed_index": 9,
        "side": 0,
    }
    paths = [seeding.stream_path(stream.template, **fields) for stream in seeding.STREAMS]
    assert len(set(paths)) == len(paths)
    assert len({stream.template for stream in seeding.STREAMS}) == len(seeding.STREAMS)


def test_an_unknown_template_is_refused() -> None:
    with pytest.raises(KeyError, match="STREAMS"):
        seeding.stream_path("made/up/{worker}", worker=0)
