"""``net.hand_slot_features``: per-slot vector fields added to that slot's query in the policy head.

The pointer head builds each hand slot's query from the card's embedding, its cost and whether it
is affordable. A field named here -- one value per slot, such as whether the card would be played
in its evolved form -- joins them, so the same card in two forms is two queries where the slot and
the tile are chosen. Each slot reads only its own value. Unset, nothing changes.
"""

from __future__ import annotations

from typing import Any

import pytest

from royalelearn import config as cfg
from royalelearn.errors import PreflightError
from royalelearn.obs_layout import field_slice
from test_nets import MASTER_SEED, _build, arch, fake_obs

FEATURES = ("own_hand_cost", "own_hand_affordable")


def _digest(spec: Any, net: Any) -> str:
    from royalelearn.learn.nets import DefaultNetworkFactory

    return DefaultNetworkFactory(MASTER_SEED).arch_digest(spec, net)


def test_the_query_takes_one_more_input_per_feature(env_spec: Any) -> None:
    plain = _build(env_spec, arch()).actor.head
    fed = _build(env_spec, arch(hand_slot_features=FEATURES)).actor.head
    assert fed.query.in_features == plain.query.in_features + len(FEATURES)
    assert fed.query_bias.in_features == plain.query_bias.in_features + len(FEATURES)
    assert fed.query.weight.abs().sum() > 0, "the new columns start initialised, not at zero"


def test_each_slot_reads_only_its_own_value(torch: Any, env_spec: Any) -> None:
    """Plant: read the field without the per-slot view and every slot moves together."""
    model = _build(env_spec, arch(hand_slot_features=("own_hand_affordable",)))
    head, trunk = model.actor.head, model.actor.trunk
    obs = fake_obs(torch, env_spec, 2, seed=5)
    part = field_slice(env_spec, "own_hand_affordable")
    moved = obs.vector.clone()
    moved[:, part.start + 1] += 1.0
    with torch.no_grad():
        features = trunk(obs)
        before = head.tile_logits(features, obs)
        after = head.tile_logits(features, obs._replace(vector=moved))
    changed = (before - after).abs().flatten(2).amax(-1)  # (B, slots)
    assert bool((changed[:, 1] > 0).all())
    assert bool((changed[:, [0, 2, 3]] == 0).all())


def test_unset_changes_nothing_and_set_is_another_architecture(env_spec: Any) -> None:
    assert "hand_slot_features" not in cfg.dump_config(cfg.RunConfig())
    assert _digest(env_spec, arch()) != _digest(env_spec, arch(hand_slot_features=FEATURES))


@pytest.mark.parametrize(
    ("name", "says"),
    [("own_hand_cards", "wide"), ("no_such_field", "no_such_field")],
)
def test_a_field_that_is_not_one_value_per_slot_is_refused(
    env_spec: Any, name: str, says: str
) -> None:
    with pytest.raises(PreflightError, match=says):
        _build(env_spec, arch(hand_slot_features=(name,)))
