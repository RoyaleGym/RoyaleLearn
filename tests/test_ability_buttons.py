"""Ability buttons: a hero's or a champion's press is an action, and its readiness is an input.

RoyaleGym's ``TileActionParser(ability_buttons=True)`` appends one action per ability button after
the tile actions: action ``1 + hand * tiles + k`` presses button ``k``, masked by the env. The
same bits go out on their own as the observation's ``ability_ready``. A learner that refused
either could not play a deck with a hero or a champion as one, and nearly every real deck of the
shipped archetype carries a hero.

What is checked here:

* a run with the buttons switched on builds, collects and updates (MockEngine: two buttons,
  never ready, so the plumbing is exercised without a hero);
* the head writes one logit per button, after the tile logits, in the parser's order, and the
  mask alone decides whether a press can be chosen;
* ``ability_ready`` reaches the network: the same state with a button ready gives other logits;
* a press is not counted as a card play by the behaviour metrics, and is counted as a press.

A network without buttons is unchanged: its architecture digest is pinned in
``tests/test_card_identity.py`` and its runs' state digests in ``tests/test_resume.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from royalelearn import config as cfg
from royalelearn.testing import coordinator, read_env_spec, tiny_config

torch = pytest.importorskip("torch")


def _button_env() -> cfg.EnvFactorySpec:
    env = cfg.default_env_spec(cfg.MOCK_ENGINE, max_steps=6)
    parser = msgspec.structs.replace(env.action_parser, kwargs={"ability_buttons": True})
    return msgspec.structs.replace(env, action_parser=parser)


@pytest.fixture(scope="module")
def spec() -> Any:
    return read_env_spec(_button_env())


def test_the_spec_names_the_buttons_and_the_grid(spec: Any) -> None:
    grid = 1 + spec.hand_size * spec.tiles[0] * spec.tiles[1]
    assert spec.n_grid_actions == grid
    assert spec.n_buttons == spec.obs_space["ability_ready"].shape[0] > 0
    assert spec.n_actions == grid + spec.n_buttons


def _obs(spec: Any, model: Any, *, ready: tuple[int, ...]) -> Any:
    from royalelearn.api.policy import ObsBatch

    grid = spec.n_grid_actions
    mask = torch.zeros((1, spec.n_actions), dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 1 : 1 + 5] = True
    for k in ready:
        mask[0, grid + k] = True
    planes = mask[:, 1:grid].float().view(1, spec.hand_size, *spec.tiles)
    return ObsBatch(
        spatial=torch.zeros((1, spec.frame_stack * spec.spatial_shape[0], *spec.tiles)),
        vector=torch.full((1, spec.vector_size), 0.25),
        mask=mask,
        mask_planes=planes,
    )


def test_the_head_writes_a_logit_per_button_after_the_tiles_and_the_mask_decides(
    spec: Any,
) -> None:
    from royalelearn.testing import build_model

    model = build_model(spec)
    obs = _obs(spec, model, ready=(1,))
    logits = model.actor.logits(obs)
    assert logits.shape == (1, spec.n_actions)
    dist = model.actor.distribution(obs)
    probs = dist.log_probs.detach().exp()[0]
    grid = spec.n_grid_actions
    assert float(probs[grid + 1]) > 0.0, "a ready button cannot be chosen"
    assert float(probs[grid + 0]) == 0.0, "a button the mask refuses can be chosen"
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)
    uniforms = torch.linspace(0.0005, 0.9995, 400)
    picked = {int(dist.sample(u.view(1))[0]) for u in uniforms}
    assert grid + 1 in picked, f"400 samples never pressed the ready button: {sorted(picked)[:10]}"


def test_ability_ready_reaches_the_network(spec: Any) -> None:
    from royalelearn.testing import build_model

    model = build_model(spec)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
        idle = model.actor.logits(_obs(spec, model, ready=()))
        ready = model.actor.logits(_obs(spec, model, ready=(0,)))
    tiles = slice(1, spec.n_grid_actions)
    assert not torch.equal(idle[:, tiles], ready[:, tiles]), (
        "the tile logits do not move when a button becomes ready: ability_ready is not an input"
    )
    with torch.no_grad():
        value_idle = model.value(_obs(spec, model, ready=()))
        value_ready = model.value(_obs(spec, model, ready=(0,)))
    assert not torch.equal(value_idle, value_ready), "the critic does not see ability_ready"


def test_a_press_is_not_a_card_play(spec: Any) -> None:
    from royalelearn.metrics.behaviour import RowBehaviour

    stats = RowBehaviour(spec)
    rows = 4
    mask = np.zeros((rows, spec.n_actions), dtype=bool)
    mask[:, 0] = True
    mask[:, 1:6] = True
    mask[:, spec.n_grid_actions] = True
    vector = np.full((rows, spec.vector_size), 0.5, dtype=np.float32)
    actions = np.array([0, 1, spec.n_grid_actions, spec.n_grid_actions])
    fields = stats.fields(vector, mask, actions)
    assert fields["policy/noop_rate"] == pytest.approx(0.25)
    assert fields["policy/button_press_rate"] == pytest.approx(0.5)
    assert sum(stats.card_plays.values()) == 1


def test_a_run_with_ability_buttons_collects_and_updates(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, env=_button_env())
    with coordinator(config) as run:
        for _ in range(2):
            run.iterate()
        rows = list(run.rows)
    assert rows[-1]["run/iteration"] == 2
    assert rows[-1]["policy/button_press_rate"] == 0.0  # MockEngine's buttons are never ready


def test_buttons_without_their_readiness_are_refused() -> None:
    """An action after the grid must be a button the observation says the readiness of: an
    action space with buttons and an observation without ``ability_ready`` is refused."""
    from royalelearn.errors import PreflightError
    from royalelearn.rollout.envspec import read_env_spec as read

    factory = _button_env()
    env = factory.build_vec(2)
    try:
        del env.single_observation_space.spaces["ability_ready"]
        with pytest.raises(PreflightError, match="ability buttons"):
            read(env, factory, frame_stack=1)
    finally:
        env.close()
