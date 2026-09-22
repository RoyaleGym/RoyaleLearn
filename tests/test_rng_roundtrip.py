"""Restoring an RNG restores the stream, not merely the parameters.

The assertion is the only one worth making: after a round trip, the next thousand draws are the
thousand that would have come next. Anything weaker -- the state dict compares equal, the seed
is the same -- is what the reference learners already do, and a run resumed at ten million steps
that draws the action noise of step zero satisfies both.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from royalelearn.api.checkpoint import RngState
from royalelearn.checkpoint import RngComponent


def test_the_numpy_generator_reproduces_its_next_thousand_draws(tmp_path) -> None:
    generator = np.random.Generator(np.random.PCG64(20260922))
    generator.random(37)  # somewhere in the middle of a stream, not at its start
    component = RngComponent(master_seed=20260922, generator=generator)
    component.iteration = 12
    component.shard_streams = [{"worker": 0, "shard": 1, "generation": 2, "ordinal": 9}]
    component.save_checkpoint(tmp_path)
    expected = generator.random(1000)

    restored = np.random.Generator(np.random.PCG64(1))
    loaded = RngComponent(master_seed=0, generator=restored)
    loaded.load_checkpoint(tmp_path, strict=True)
    assert np.array_equal(restored.random(1000), expected)
    assert loaded.iteration == 12
    assert loaded.master_seed == 20260922
    assert loaded.shard_streams == component.shard_streams


def test_python_random_reproduces_its_next_thousand_draws(tmp_path) -> None:
    random.seed(4242)
    random.random()
    component = RngComponent(master_seed=1)
    component.save_checkpoint(tmp_path)
    expected = [random.random() for _ in range(1000)]

    random.seed(1)  # somewhere else entirely
    RngComponent(master_seed=0).load_checkpoint(tmp_path, strict=True)
    assert [random.random() for _ in range(1000)] == expected


def test_torch_cpu_reproduces_its_next_thousand_draws(torch, tmp_path) -> None:
    torch.manual_seed(20260922)
    torch.rand(11)
    component = RngComponent(master_seed=20260922)
    component.save_checkpoint(tmp_path)
    expected = torch.rand(1000)

    torch.manual_seed(1)
    RngComponent(master_seed=0).load_checkpoint(tmp_path, strict=True)
    assert torch.equal(torch.rand(1000), expected)


def test_the_state_is_json_and_carries_no_pickle(tmp_path) -> None:
    component = RngComponent(
        master_seed=5, generator=np.random.default_rng(3), eval_seed_set_sha="deadbeef"
    )
    component.save_checkpoint(tmp_path)
    blob = (tmp_path / "rng.json").read_bytes()
    assert blob.startswith(b"{")
    assert b"\x80\x04" not in blob
    state = component.capture()
    assert isinstance(state, RngState)
    assert state.eval_seed_set_sha == "deadbeef"
    assert state.numpy_minibatch["bit_generator"] == "PCG64"


def test_a_missing_state_is_a_refusal_under_strict_and_a_notice_without_it(
    tmp_path, capsys
) -> None:
    from royalelearn.errors import CheckpointFormatError

    component = RngComponent(master_seed=5)
    with pytest.raises(CheckpointFormatError):
        component.load_checkpoint(tmp_path, strict=True)
    component.load_checkpoint(tmp_path, strict=False)
    assert "rng.json" in capsys.readouterr().out
