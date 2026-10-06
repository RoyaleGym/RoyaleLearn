"""``net.button_head = "card"``: each ability button scored from its card, not from its position.

The default head writes button ``k``'s logit from a weight row of its own, so what it learns is
about position ``k``, and a deck that puts its hero in another position gets the other row's
habits. The card head builds each button's input from the card it belongs to -- the hand slots'
card table -- its own status and the pooled board, through one scorer shared by every button. Swap
two buttons and their logits swap with them. Unset, the head is the one every save was made with.

The per-button fields come from RoyaleGym's ``SpatialObsBuilder(button_index=True)``; here they are
appended to a real button environment's layout, so the widths are what that builder declares.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from royalelearn import config as cfg
from royalelearn.errors import PreflightError
from royalelearn.testing import read_env_spec

torch = pytest.importorskip("torch")

STATUS = (
    "own_button_available_by_index",
    "own_button_spent_by_index",
    "own_button_cooldown_by_index",
)


def _button_env() -> cfg.EnvFactorySpec:
    env = cfg.default_env_spec(cfg.MOCK_ENGINE, max_steps=6)
    parser = msgspec.structs.replace(env.action_parser, kwargs={"ability_buttons": True})
    return msgspec.structs.replace(env, action_parser=parser)


def _with_button_fields(spec: Any) -> Any:
    """The spec as ``button_index=True`` declares it: four fields after the existing ones."""
    width = spec.num_cards + 1
    layout = list(spec.vector_layout)
    offset = spec.vector_size
    fields = [("own_button_cards", spec.n_buttons * width)]
    fields += [(name, spec.n_buttons) for name in STATUS]
    for name, size in fields:
        layout.append((name, offset, size))
        offset += size
    vector = msgspec.structs.replace(spec.obs_space["vector"], shape=(offset,))
    return msgspec.structs.replace(
        spec,
        obs_space={**spec.obs_space, "vector": vector},
        vector_layout=tuple(layout),
        vector_size=offset,
    )


@pytest.fixture(scope="module")
def spec() -> Any:
    return _with_button_fields(read_env_spec(_button_env()))


def _net(**overrides: Any) -> Any:
    return cfg.NetConfig(
        channels=8,
        blocks=1,
        norm_groups=4,
        card_embed=8,
        value_hidden=16,
        device="cpu",
        autocast_dtype="float32",
        **overrides,
    )


def _build(spec: Any, net: Any) -> Any:
    from royalelearn.learn.nets import DefaultNetworkFactory

    return DefaultNetworkFactory(4242).build(spec, net, "cpu")


def _obs(spec: Any, cards: list[int], ready: list[bool]) -> Any:
    """One row whose buttons hold ``cards`` (catalogue ids) with ``ready`` ready bits."""
    from royalelearn.api.policy import ObsBatch
    from royalelearn.obs_layout import field_slice

    generator = torch.Generator().manual_seed(3)
    grid = spec.n_grid_actions
    mask = torch.zeros((1, spec.n_actions), dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 1:6] = True
    for k, on in enumerate(ready):
        mask[0, grid + k] = on
    vector = torch.rand((1, spec.vector_size), generator=generator)
    onehot = field_slice(spec, "own_button_cards")
    vector[0, onehot] = 0.0
    width = spec.num_cards + 1
    for k, card in enumerate(cards):
        vector[0, onehot.start + k * width + card] = 1.0
    for name in STATUS:
        vector[0, field_slice(spec, name)] = torch.tensor([0.5] * spec.n_buttons)
    return ObsBatch(
        spatial=torch.rand(
            (1, spec.frame_stack * spec.spatial_shape[0], *spec.tiles), generator=generator
        ),
        mask_planes=mask[:, 1:grid].float().view(1, spec.hand_size, *spec.tiles),
        vector=vector,
        mask=mask,
    )


@pytest.mark.parametrize("policy_head", ["pointer", "factored"])
def test_swapping_two_heroes_swaps_their_logits_and_moves_nothing_else(
    spec: Any, policy_head: str
) -> None:
    """The same state with the two buttons' heroes swapped: each hero's press logit is the same,
    in its new position, and so are the value and every other logit. Plants: a per-position
    weight in the scorer, or the per-button fields reaching the trunk or the critic."""
    assert spec.n_buttons >= 2
    model = _build(spec, _net(button_head="card", policy_head=policy_head))
    first, swapped = _obs(spec, [3, 7], [True, True]), _obs(spec, [7, 3], [True, True])
    grid = spec.n_grid_actions
    with torch.no_grad():
        a, b = model.actor.logits(first)[0], model.actor.logits(swapped)[0]
        va, vb = model.value(first), model.value(swapped)
    assert torch.allclose(a[grid : grid + 2], b[grid : grid + 2].flip(0), atol=1e-6)
    assert torch.allclose(a[:grid], b[:grid], atol=1e-6), "the swap moved a tile or the no-op"
    assert torch.allclose(va, vb, atol=1e-6), "the swap moved the value"
    assert not torch.allclose(a[grid], a[grid + 1]), "two heroes scored alike: the card is unread"


def test_one_scorer_serves_every_button(spec: Any) -> None:
    head = _build(spec, _net(button_head="card")).actor.head
    assert head.buttons is None and head.button_score is not None
    assert head.button_score.out_features == 1


def test_unset_is_the_head_every_save_was_made_with(spec: Any) -> None:
    from royalelearn.learn.nets import DefaultNetworkFactory

    factory = DefaultNetworkFactory(4242)
    assert factory.arch_digest(spec, _net()) == factory.arch_digest(spec, _net(button_head="index"))
    assert factory.arch_digest(spec, _net()) != factory.arch_digest(spec, _net(button_head="card"))
    assert "button_head" not in cfg.dump_config(cfg.RunConfig())
    head = _build(spec, _net()).actor.head
    assert head.buttons is not None and head.button_score is None


def test_the_card_head_without_its_fields_or_buttons_is_refused() -> None:
    plain = read_env_spec(_button_env())
    with pytest.raises(PreflightError, match="own_button_cards"):
        _build(plain, _net(button_head="card"))
    no_buttons = read_env_spec(cfg.default_env_spec(cfg.MOCK_ENGINE, max_steps=6))
    with pytest.raises(PreflightError, match="no"):
        _build(no_buttons, _net(button_head="card"))
    with pytest.raises(PreflightError, match="button_head"):
        _build(plain, _net(button_head="slot"))
