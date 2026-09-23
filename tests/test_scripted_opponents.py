"""The scripted opponents a run can name, and what each one is for.

Two of them are the ladder's anchors and existed from the start: a bot that never plays and a bot
that plays at random. A learner that beats those has shown very little, so the other four are
RoyaleGym's one-sentence strategies, there so a run can be measured against something that plays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from royalelearn.rollout.scripted import (
    SCRIPTED_NAMES,
    build_opponent,
    scripted_id,
)


def test_every_name_builds_an_opponent_that_acts() -> None:
    """A name in the table that cannot be built is a run that dies at its first assignment."""
    mask = np.zeros(64, dtype=bool)
    mask[0] = True
    rng = np.random.default_rng(0)
    for name in SCRIPTED_NAMES:
        opponent: Any = build_opponent(name)
        action = opponent.act({}, mask, rng)
        assert isinstance(action, int | np.integer), name
        assert bool(mask[int(action)]), f"{name} chose an action its mask forbids"


def test_each_call_builds_a_fresh_opponent() -> None:
    """Several of these keep no state today and one of them may tomorrow.

    The same instance handed to eight environments is a shared mutable, which is the bug this
    package refuses elsewhere, and RoyaleGym's own ladder() is a function for the same reason.
    """
    for name in SCRIPTED_NAMES:
        assert build_opponent(name) is not build_opponent(name)


def test_an_unknown_name_says_what_the_known_ones_are() -> None:
    with pytest.raises(KeyError) as raised:
        build_opponent("agressive")
    message = str(raised.value)
    for name in SCRIPTED_NAMES:
        assert name in message


def test_the_order_of_the_table_is_part_of_the_protocol() -> None:
    """The parent sends an index and the worker looks it up, so a new name is APPENDED.

    Inserting one would silently change which opponent every recorded run had played, and every
    id in the result log with it.
    """
    assert SCRIPTED_NAMES[:2] == ("noop", "random_legal")
    assert len(set(SCRIPTED_NAMES)) == len(SCRIPTED_NAMES)
    assert scripted_id(SCRIPTED_NAMES[0]) == "scripted:noop"
