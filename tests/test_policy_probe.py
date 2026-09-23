"""What a per-card play share is a share OF.

A policy plays a three-cost card far more often than a four-cost one whatever it prefers, because
the bar averages about one and a half elixir and the cheap card is affordable at many more
decisions. The play share reflects the policy and the elixir economy together, and a reader
cannot tell which they are looking at. These tests hold the two apart: a policy that chooses
uniformly among what it can afford has an even play RATE and a wildly uneven play SHARE.

The owner's reading on 2026-09-22 was that the bot plays its five cheap cards at about 20% each
and its three four-cost cards at 1.04% combined. That is a true measurement of the share. Whether
it is a statement about the policy is what these keys answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalelearn.coordinator import PolicyProbe


class _Hand:
    """The hand layout the probe reads: four slots, a one-hot of five cards plus empty."""

    onehot_width = 6
    hand_size = 4


def _probe(tiles: int = 3) -> PolicyProbe:
    probe = PolicyProbe.__new__(PolicyProbe)
    probe.hand = _Hand()
    probe.tiles = tiles
    probe.card_plays = {}
    probe.card_legal = {}
    probe.card_tile_plays = {}
    probe.tile_plays = {}
    return probe


def _onehot(rows: list[list[int]]) -> np.ndarray:
    """One row per decision, listing the card index in each of the four hand slots."""
    out = np.zeros((len(rows), _Hand.hand_size * _Hand.onehot_width), dtype=np.float32)
    for index, hand in enumerate(rows):
        for slot, card in enumerate(hand):
            out[index, slot * _Hand.onehot_width + card] = 1.0
    return out


def test_a_card_that_is_rarely_affordable_reads_as_rarely_played(tmp_path) -> None:
    """The measurement the behavioural read needs, and the one that was missing.

    Card 3 sits in the hand at every decision and is affordable at one in ten of them, and the
    policy plays it every single time it can. Its play SHARE is a ninth of card 1's, which reads
    as a strong preference against it. Its play RATE is HIGHER than card 1's, because it was
    taken on both of its two opportunities and card 1 was taken on 18 of its 20. The share and
    the rate point in opposite directions, and only one of them is about the policy.
    """
    probe = _probe()
    decisions = 20
    onehot = _onehot([[1, 2, 3, 4]] * decisions)
    legal = np.zeros((decisions, _Hand.hand_size), dtype=bool)
    legal[:, 0] = True  # card 1: affordable at every decision
    legal[::10, 2] = True  # card 3: affordable at one in ten

    # Play slot 0 whenever it is legal, and slot 2 whenever IT is legal.
    actions = np.zeros(decisions, dtype=np.int64)
    for row in range(decisions):
        actions[row] = 1 + (2 * probe.tiles if legal[row, 2] else 0)

    fields = probe._plays(actions, onehot, legal)

    assert fields["policy/card_legal_frac/1"] == pytest.approx(1.0)
    assert fields["policy/card_legal_frac/3"] == pytest.approx(0.1)
    assert fields["policy/card_play_frac/3"] == pytest.approx(2 / 20)
    assert fields["policy/card_play_frac/1"] == pytest.approx(18 / 20)
    # Nine times the share. The rates say the policy took each card on almost every decision
    # where it could: card 1 on 18 of its 20 opportunities, card 3 on both of its 2.
    assert fields["policy/card_play_rate/1"] == pytest.approx(18 / 20)
    assert fields["policy/card_play_rate/3"] == pytest.approx(1.0)


def test_a_card_the_policy_avoids_reads_as_avoided(tmp_path) -> None:
    """The other half: with affordability held equal, the rate falls and it means what it says."""
    probe = _probe()
    decisions = 20
    onehot = _onehot([[1, 2, 3, 4]] * decisions)
    legal = np.zeros((decisions, _Hand.hand_size), dtype=bool)
    legal[:, 0] = True
    legal[:, 2] = True  # both affordable at every decision

    actions = np.full(decisions, 1, dtype=np.int64)  # always slot 0
    actions[0] = 1 + 2 * probe.tiles  # slot 2 once

    fields = probe._plays(actions, onehot, legal)

    assert fields["policy/card_legal_frac/1"] == pytest.approx(1.0)
    assert fields["policy/card_legal_frac/3"] == pytest.approx(1.0)
    assert fields["policy/card_play_rate/1"] == pytest.approx(19 / 20)
    assert fields["policy/card_play_rate/3"] == pytest.approx(1 / 20)


def test_a_card_never_affordable_has_no_rate_rather_than_a_zero(tmp_path) -> None:
    """Dividing by no opportunities is not zero preference, and a zero would read as one."""
    probe = _probe()
    onehot = _onehot([[1, 2, 3, 4]] * 5)
    legal = np.zeros((5, _Hand.hand_size), dtype=bool)
    legal[:, 0] = True

    fields = probe._plays(np.ones(5, dtype=np.int64), onehot, legal)

    assert fields["policy/card_legal_frac/3"] == pytest.approx(0.0)
    assert "policy/card_play_rate/3" not in fields
    assert fields["policy/card_in_hand_frac/3"] == pytest.approx(1.0)


class _Publisher:
    """Stands in for a ViserPublisher: the player duck-types on ``publish``."""

    attached = False

    def __init__(self) -> None:
        self.published: list[object] = []

    def publish(self, *args: object, **kwargs: object) -> None:
        self.published.append(args)


def _player(viser: object):
    """Through the real constructor, because the coercion under test is IN it.

    The first version of this built the object with ``__new__`` and assigned the attribute, so
    it passed with the old ``bool(viser)`` still in place: it tested the assignment rather than
    the constructor. A plant proves nothing unless it is on the line the test depends on.
    """
    from royalelearn.coordinator import EnvBattlePlayer

    return EnvBattlePlayer(
        spec=None,
        factory_spec=None,
        actors=lambda _name: None,
        master_seed=0,
        viser=viser,
    )


def test_the_player_keeps_a_publisher_it_is_given() -> None:
    """A caller outside this repo puts its own publisher in the path, and pacing is why.

    A battle is simulated far faster than it is watched and the viewer keeps only the newest
    datagram, so an unpaced stream is sampled and the battle jumps. The fix is a publisher whose
    publish sleeps to the next slot, and that belongs to whoever is watching rather than to the
    learner. Before this, ``player.viser`` was coerced with ``bool()`` and there was nowhere to
    put one.
    """
    from royalelearn.coordinator import _is_publisher

    publisher = _Publisher()
    assert _is_publisher(publisher)
    player = _player(publisher)
    assert player.viser is publisher


def test_a_bool_still_means_what_it_meant() -> None:
    """True builds the publisher from the environment; False publishes nothing."""
    from royalelearn.coordinator import _is_publisher

    assert not _is_publisher(True)
    assert not _is_publisher(False)
    assert not _is_publisher(1), "a truthy number is a bool's business, not a publisher's"
    assert not _is_publisher(None)
