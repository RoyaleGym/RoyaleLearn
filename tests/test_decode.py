"""How a frozen policy picks its action: ``stochastic``, ``argmax``, ``gtau:<x>``, ``gtaucap:<x>``.

``gtau`` plays when ``1 - p(no-op) > x`` and then takes the group with the largest summed
probability -- a hand slot summed over its tiles, or one ability button -- and that slot's most
likely tile, or the press. An illegal action's probability is zero before anything is read. The
rule is checked on hand-built distributions, then on a run's own opponent seats. ``gtaucap`` is
``gtau`` that also plays whenever the seat's elixir is at the cap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from royalelearn import config as cfg
from royalelearn.learn.decode import decode_actions, elixir_at_cap, gtau_actions, parse_mode
from royalelearn.testing import coordinator, tiny_config

torch = pytest.importorskip("torch")

HAND, TILES, BUTTONS = 4, 6, 2
GRID = 1 + HAND * TILES


def _log_probs(masses: dict[int, float], legal: set[int] | None = None) -> torch.Tensor:
    """One row whose legal actions carry ``masses`` (normalised), everything else illegal."""
    width = GRID + BUTTONS
    p = torch.zeros(width)
    for action, mass in masses.items():
        p[action] = mass
    if legal is not None:
        for action in range(width):
            if action not in legal:
                p[action] = 0.0
    p = p / p.sum()
    return p.log()[None, :]


def _gtau(lp: torch.Tensor, x: float = 0.12, at_cap: bool | None = None) -> int:
    cap = None if at_cap is None else torch.tensor([at_cap])
    return int(gtau_actions(lp, threshold=x, hand_size=HAND, tiles=TILES, at_cap=cap)[0])


def _tile(slot: int, tile: int) -> int:
    return 1 + slot * TILES + tile


def test_it_waits_when_the_play_mass_is_under_the_threshold() -> None:
    assert _gtau(_log_probs({0: 0.90, _tile(1, 2): 0.10})) == 0
    assert _gtau(_log_probs({0: 0.885, _tile(1, 2): 0.115})) == 0
    assert _gtau(_log_probs({0: 0.87, _tile(1, 2): 0.13})) == _tile(1, 2)


def test_it_plays_the_slot_with_the_most_summed_mass_at_that_slots_best_tile() -> None:
    # Slot 2 holds the single most likely tile, but slot 0's tiles sum to more.
    masses = {
        0: 0.5,
        _tile(2, 5): 0.2,
        _tile(0, 1): 0.1,
        _tile(0, 3): 0.12,
        _tile(0, 4): 0.08,
    }
    assert _gtau(_log_probs(masses)) == _tile(0, 3)


def test_a_button_is_a_group_of_its_own() -> None:
    masses = {0: 0.5, _tile(1, 0): 0.1, _tile(1, 1): 0.1, GRID + 1: 0.3}
    assert _gtau(_log_probs(masses)) == GRID + 1
    masses[_tile(1, 2)] = 0.15
    assert _gtau(_log_probs(masses)) == _tile(1, 2)


def test_an_illegal_action_counts_as_nothing() -> None:
    masses = {0: 0.5, _tile(3, 0): 0.4, _tile(1, 0): 0.1}
    legal = {0, _tile(1, 0)}
    assert _gtau(_log_probs(masses, legal)) == _tile(1, 0)
    assert _gtau(_log_probs({0: 1.0}, {0}), x=0.0) == 0


def test_the_mode_names_parse_and_bad_ones_are_refused() -> None:
    assert parse_mode("stochastic") == ("stochastic", 0.0)
    assert parse_mode("argmax") == ("argmax", 0.0)
    assert parse_mode("gtau:0.12") == ("gtau", 0.12)
    assert parse_mode("gtaucap:0.3") == ("gtaucap", 0.3)
    for bad in ("gtau", "gtau:1", "gtau:-0.1", "gtau:x", "greedy", "gtaucap", "gtaucap:1.5"):
        with pytest.raises(ValueError):
            parse_mode(bad)
    problems = cfg.check_consistency(
        cfg.RunConfig(ladder=cfg.LadderConfig(opponent_mode="gtau:2", release_mode="gtaucap:0.2"))
    )
    assert [p for p in problems if "opponent_mode" in p] and not [
        p for p in problems if "release_mode" in p
    ]


def test_at_the_elixir_cap_it_plays_what_gtau_would_play() -> None:
    masses = {0: 0.95, _tile(2, 1): 0.03, _tile(2, 4): 0.01, GRID: 0.01}
    assert _gtau(_log_probs(masses)) == 0
    assert _gtau(_log_probs(masses), at_cap=False) == 0
    assert _gtau(_log_probs(masses), at_cap=True) == _tile(2, 1)
    # Over the threshold the cap changes nothing; with nothing legal to play it still waits.
    over = {0: 0.5, _tile(1, 0): 0.5}
    assert _gtau(_log_probs(over), at_cap=True) == _gtau(_log_probs(over)) == _tile(1, 0)
    assert _gtau(_log_probs({0: 1.0}, {0}), at_cap=True) == 0


def test_gtaucap_without_the_elixir_is_refused_not_decoded_as_gtau() -> None:
    from royalelearn.learn.distribution import MaskedCategorical

    lp = _log_probs({0: 0.95, _tile(2, 1): 0.05})
    dist = MaskedCategorical(lp, torch.isfinite(lp))
    with pytest.raises(ValueError, match="elixir"):
        decode_actions(dist, "gtaucap:0.12", torch.zeros(1), hand_size=HAND, tiles=TILES)
    # And plain gtau never reads one it is handed.
    cap = torch.tensor([True])
    for mode, want in (("gtau:0.12", 0), ("gtaucap:0.12", _tile(2, 1))):
        got = decode_actions(dist, mode, torch.zeros(1), hand_size=HAND, tiles=TILES, at_cap=cap)
        assert int(got[0]) == want, mode


def test_the_cap_is_read_from_the_own_elixir_field_by_name(tmp_path: Path) -> None:
    with coordinator(tiny_config(tmp_path)) as run:
        spec = run.spec
    (offset,) = [o for name, o, _ in spec.vector_layout if name == "own_elixir"]
    assert offset != 0 or [n for n, o, _ in spec.vector_layout if o == 0] == ["own_elixir"]
    vector = torch.zeros((4, spec.vector_size))
    vector[:, offset] = torch.tensor([1.0, 0.995, 0.99, 0.0])
    vector[:, [i for i in range(spec.vector_size) if i != offset]] = 1.0
    assert elixir_at_cap(spec, vector).tolist() == [True, True, False, False]


def test_a_runs_opponent_seats_are_decoded_by_opponent_mode(tmp_path: Path) -> None:
    import msgspec

    from royalelearn.api.policy import ObsBatch
    from royalelearn.learn.distribution import MaskedCategorical

    ladder = msgspec.structs.replace(tiny_config(tmp_path).ladder, opponent_mode="gtau:0.12")
    with coordinator(tiny_config(tmp_path, ladder=ladder)) as run:
        spec = run.spec
        actor = run.model.actor
        rows = 16
        generator = torch.Generator().manual_seed(5)
        mask = torch.rand((rows, spec.n_actions), generator=generator) < 0.02
        mask[:, 0] = True
        obs = ObsBatch(
            spatial=torch.rand(
                (rows, spec.frame_stack * spec.spatial_shape[0], *spec.tiles), generator=generator
            ),
            vector=torch.rand((rows, spec.vector_size), generator=generator),
            mask=mask,
            mask_planes=mask[:, 1:].float().view(rows, spec.hand_size, *spec.tiles),
        )
        uniforms = torch.rand(rows, generator=generator)
        with torch.no_grad():
            for p in actor.parameters():
                p.add_(0.05)
            got = run.inference._act_frozen(actor, obs, uniforms)
            dist = MaskedCategorical(actor.logits(obs).float(), obs.mask)
            want = decode_actions(
                dist,
                "gtau:0.12",
                uniforms,
                hand_size=spec.hand_size,
                tiles=spec.tiles[0] * spec.tiles[1],
            )
            sampled = dist.sample(uniforms)
    assert got.tolist() == want.tolist()
    assert got.tolist() != sampled.tolist(), "gtau and sampling agree here, so nothing is tested"


def _obs(spec, rows: int, generator) -> object:
    from royalelearn.api.policy import ObsBatch

    mask = torch.rand((rows, spec.n_actions), generator=generator) < 0.02
    mask[:, 0] = True
    return ObsBatch(
        spatial=torch.rand(
            (rows, spec.frame_stack * spec.spatial_shape[0], *spec.tiles), generator=generator
        ),
        vector=torch.rand((rows, spec.vector_size), generator=generator) * 0.9,
        mask=mask,
        mask_planes=mask[:, 1:].float().view(rows, spec.hand_size, *spec.tiles),
    )


def test_gtaucap_opponent_seats_play_at_the_cap(tmp_path: Path) -> None:
    import msgspec

    from royalelearn.learn.distribution import MaskedCategorical

    # A threshold no row clears: gtau waits everywhere, so every play below is the cap's.
    ladder = msgspec.structs.replace(tiny_config(tmp_path).ladder, opponent_mode="gtaucap:0.999")
    with coordinator(tiny_config(tmp_path, ladder=ladder)) as run:
        spec = run.spec
        actor = run.model.actor
        rows = 16
        obs = _obs(spec, rows, torch.Generator().manual_seed(7))
        (offset,) = [o for name, o, _ in spec.vector_layout if name == "own_elixir"]
        capped = torch.arange(rows) % 2 == 0
        obs.vector[capped, offset] = 1.0
        uniforms = torch.zeros(rows)
        with torch.no_grad():
            got = torch.as_tensor(run.inference._act_frozen(actor, obs, uniforms))
            dist = MaskedCategorical(actor.logits(obs).float(), obs.mask)
            tiles = spec.tiles[0] * spec.tiles[1]
            gtau = decode_actions(
                dist, "gtau:0.999", uniforms, hand_size=spec.hand_size, tiles=tiles
            )
            free = decode_actions(dist, "gtau:0.0", uniforms, hand_size=spec.hand_size, tiles=tiles)
    playable = obs.mask[:, 1:].any(-1)
    assert gtau.tolist() == [0] * rows
    assert got[~capped].tolist() == [0] * int((~capped).sum())
    assert got[capped].tolist() == free[capped].tolist()
    assert bool((got[capped & playable] != 0).all()) and bool((capped & playable).any())


def test_gtaucap_release_seats_play_at_the_cap(tmp_path: Path) -> None:
    import numpy as np

    with coordinator(tiny_config(tmp_path)) as run:
        spec = run.spec
        actors = run.eval_actors
        actors.release_mode = "gtaucap:0.999"
        (offset,) = [o for name, o, _ in spec.vector_layout if name == "own_elixir"]
        env = spec.env_factory.factory()()
        observations, _ = env.reset(seed=3)
        env_obs = next(iter(observations.values()))
        mask = np.asarray(env_obs["action_mask"]).copy()
        assert mask[1:].any(), "the sample observation offers no play, so nothing is tested"
        # Two legal plays beside the no-op: an untrained net's play mass is then far under the
        # threshold, so gtau alone waits.
        mask[1 + np.flatnonzero(mask[1:])[2:]] = False
        env_obs = dict(env_obs, action_mask=mask)
        policy = actors.policy(run.model.actor)
        low = dict(env_obs, vector=np.asarray(env_obs["vector"]).copy())
        low["vector"][offset] = 0.0
        full = dict(env_obs, vector=np.asarray(env_obs["vector"]).copy())
        full["vector"][offset] = 1.0
        assert policy(low, 0.0, None) == 0
        assert policy(full, 0.0, None) != 0
