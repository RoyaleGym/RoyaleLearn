"""How a frozen policy picks its action: ``stochastic``, ``argmax`` and ``gtau:<x>``.

``gtau`` plays when ``1 - p(no-op) > x`` and then takes the group with the largest summed
probability -- a hand slot summed over its tiles, or one ability button -- and that slot's most
likely tile, or the press. An illegal action's probability is zero before anything is read. The
rule is checked on hand-built distributions, then on a run's own opponent seats.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from royalelearn import config as cfg
from royalelearn.learn.decode import decode_actions, gtau_actions, parse_mode
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


def _gtau(lp: torch.Tensor, x: float = 0.12) -> int:
    return int(gtau_actions(lp, threshold=x, hand_size=HAND, tiles=TILES)[0])


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
    for bad in ("gtau", "gtau:1", "gtau:-0.1", "gtau:x", "greedy"):
        with pytest.raises(ValueError):
            parse_mode(bad)
    problems = cfg.check_consistency(
        cfg.RunConfig(ladder=cfg.LadderConfig(opponent_mode="gtau:2", release_mode="gtau:0.2"))
    )
    assert [p for p in problems if "opponent_mode" in p] and not [
        p for p in problems if "release_mode" in p
    ]


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
