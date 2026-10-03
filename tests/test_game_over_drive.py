"""The preflight's drive to a finished battle: room past the match clock for how a battle ends.

How a level overtime ends is the engine's rule. On RoyaleSim r31 the crown towers drain after
overtime, and a level battle ends some ticks past regulation and overtime, so a drive bounded
by the match clock alone refuses a battle that is ending normally.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from royalelearn.errors import PreflightError
from royalelearn.rollout.preflight import _drive_to_game_over
from royalelearn.testing import read_env_spec, tiny_config


def test_a_battle_that_ends_after_the_match_clock_still_reaches_game_over(tmp_path: Path) -> None:
    """A battle the match clock says is 400 ticks shorter than the engine runs it stands in for
    the drain; one that has still not ended well past the clock is refused."""
    config = tiny_config(tmp_path)
    spec = read_env_spec(config.env)
    shorter = msgspec.structs.replace(spec, overtime_ticks=spec.overtime_ticks - 400)
    state, _parser = _drive_to_game_over(config, shorter)
    assert state.game_over
    never = msgspec.structs.replace(spec, regular_ticks=100, overtime_ticks=0)
    with pytest.raises(PreflightError, match="did not end"):
        _drive_to_game_over(config, never)
