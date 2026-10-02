"""How a frozen policy picks its action: ``stochastic``, ``argmax``, or a decode of your own.

``ladder.opponent_mode`` and ``ladder.release_mode`` take ``stochastic`` (sample the masked
distribution), ``argmax`` (its most likely action) or ``plugin:<module>:<function>``: a function
of your own, called with the masked log-probabilities, the mask and the observation vector of a
batch of rows, which returns one action per row. A plugin's actions are checked against the mask.
Nothing else is built in, and a name that is not one of these is refused saying how to supply it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from royalelearn import config as cfg
from royalelearn.learn.decode import decode_actions, parse_mode, resolve_decoder
from royalelearn.testing import coordinator, tiny_config

torch = pytest.importorskip("torch")

PLUGIN = "plugin:test_decode:first_legal_play"
SEEN: dict[str, Any] = {}


def first_legal_play(log_probs: Any, mask: Any, vector: Any) -> Any:
    """A decode for the tests: each row's first legal action after the no-op, else the no-op."""
    SEEN["shapes"] = (tuple(log_probs.shape), tuple(mask.shape), tuple(vector.shape))
    plays = mask.clone()
    plays[:, 0] = False
    first = plays.int().argmax(-1)
    return torch.where(plays.any(-1), first, torch.zeros_like(first))


def illegal_choice(log_probs: Any, mask: Any, vector: Any) -> Any:
    """Plays the last action of every row, legal or not."""
    return torch.full((mask.shape[0],), mask.shape[1] - 1, dtype=torch.int64)


def _distribution(rows: int = 6, width: int = 12, seed: int = 0) -> Any:
    from royalelearn.learn.distribution import MaskedCategorical

    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, width), generator=generator) < 0.4
    mask[:, 0] = True
    mask[1, 1:] = False  # one row with nothing but the no-op
    logits = torch.randn((rows, width), generator=generator)
    return MaskedCategorical(logits, mask)


def test_the_built_ins_and_a_plugin_parse_and_anything_else_is_refused() -> None:
    assert parse_mode("stochastic") == ("stochastic", "")
    assert parse_mode("argmax") == ("argmax", "")
    assert parse_mode(PLUGIN) == ("plugin", "test_decode:first_legal_play")
    for bad in ("threshold:0.3", "threshold_plus:0.6", "greedy", "plugin:", "plugin:onlymodule"):
        with pytest.raises(ValueError, match="plugin:<module>:<function>") as refused:
            parse_mode(bad)
        assert "is not built in" in str(refused.value) or bad.startswith("plugin")
    with pytest.raises(ValueError, match="decode 'threshold' is not built in"):
        parse_mode("threshold:0.3")
    problems = cfg.check_consistency(
        cfg.RunConfig(ladder=cfg.LadderConfig(opponent_mode="threshold:0.6", release_mode=PLUGIN))
    )
    assert [p for p in problems if "opponent_mode" in p]
    assert not [p for p in problems if "release_mode" in p]


def test_a_plugin_decides_with_the_log_probs_the_mask_and_the_vector() -> None:
    dist = _distribution()
    vector = torch.zeros((6, 5))
    got = decode_actions(dist, PLUGIN, torch.zeros(6), vector=vector)
    assert got.tolist() == first_legal_play(dist.log_probs, dist.mask, vector).tolist()
    assert SEEN["shapes"] == ((6, 12), (6, 12), (6, 5))
    assert int(got[1]) == 0


def test_the_built_ins_are_the_distributions_own_sample_and_mode() -> None:
    dist = _distribution(seed=1)
    uniforms = torch.linspace(0.05, 0.95, 6)
    vector = torch.zeros((6, 5))
    assert decode_actions(dist, "stochastic", uniforms, vector=vector).tolist() == (
        dist.sample(uniforms).tolist()
    )
    assert decode_actions(dist, "argmax", uniforms, vector=vector).tolist() == (
        dist.mode().tolist()
    )


def test_a_plugin_that_plays_an_illegal_action_is_refused() -> None:
    with pytest.raises(ValueError, match="illegal"):
        decode_actions(
            _distribution(),
            "plugin:test_decode:illegal_choice",
            torch.zeros(6),
            vector=torch.zeros((6, 5)),
        )


def test_a_plugin_that_cannot_be_found_is_refused_by_name() -> None:
    for target in ("plugin:no_such_module_here:fn", "plugin:test_decode:no_such_function"):
        with pytest.raises(ValueError, match=target.split(":", 1)[1]):
            resolve_decoder(target)


def _obs(spec: Any, rows: int, seed: int) -> Any:
    from royalelearn.api.policy import ObsBatch

    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, spec.n_actions), generator=generator) < 0.02
    mask[:, 0] = True
    return ObsBatch(
        spatial=torch.rand(
            (rows, spec.frame_stack * spec.spatial_shape[0], *spec.tiles), generator=generator
        ),
        vector=torch.rand((rows, spec.vector_size), generator=generator),
        mask=mask,
        mask_planes=mask[:, 1:].float().view(rows, spec.hand_size, *spec.tiles),
    )


def test_a_runs_opponent_seats_are_decoded_by_the_plugin(tmp_path: Path) -> None:
    import msgspec

    from royalelearn.learn.distribution import MaskedCategorical

    ladder = msgspec.structs.replace(tiny_config(tmp_path).ladder, opponent_mode=PLUGIN)
    with coordinator(tiny_config(tmp_path, ladder=ladder)) as run:
        spec = run.spec
        actor = run.model.actor
        obs = _obs(spec, 16, seed=5)
        uniforms = torch.rand(16, generator=torch.Generator().manual_seed(6))
        with torch.no_grad():
            got = run.inference._act_frozen(actor, obs, uniforms)
            dist = MaskedCategorical(actor.logits(obs).float(), obs.mask)
            sampled = dist.sample(uniforms)
    want = first_legal_play(dist.log_probs, obs.mask, obs.vector)
    assert got.tolist() == want.tolist()
    assert got.tolist() != sampled.tolist(), "the plugin and sampling agree here: nothing tested"


def test_a_runs_release_seats_are_decoded_by_the_plugin(tmp_path: Path) -> None:
    import numpy as np

    with coordinator(tiny_config(tmp_path)) as run:
        actors = run.eval_actors
        actors.release_mode = PLUGIN
        env = run.spec.env_factory.factory()()
        observations, _ = env.reset(seed=3)
        seat = next(iter(observations.values()))
        mask = np.asarray(seat["action_mask"])
        assert mask[1:].any(), "the observation offers no play, so nothing is tested"
        policy = actors.policy(run.model.actor)
        assert policy(seat, 0.5, None) == int(np.flatnonzero(mask[1:])[0]) + 1


def test_a_run_refuses_a_plugin_it_cannot_find_before_it_starts(tmp_path: Path) -> None:
    import msgspec

    from royalelearn.errors import PreflightError

    ladder = msgspec.structs.replace(
        tiny_config(tmp_path).ladder, opponent_mode="plugin:no_such_module_here:fn"
    )
    with (
        pytest.raises((PreflightError, ValueError), match="no_such_module_here"),
        coordinator(tiny_config(tmp_path, ladder=ladder)),
    ):
        pass
