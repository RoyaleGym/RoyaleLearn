"""Behaviour statistics computed from rows, one function per statistic.

Here rather than inside the coordinator's probe because more than one caller needs the same
number: the run's own sample of its rows, and any other source of rows that has to be compared
with it -- demonstrations driven through the environment, say. Two implementations of one
statistic would make every difference between a policy and a demonstration partly a difference
between two pieces of code.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ELIXIR_TOLERANCE", "play_rate_by_elixir"]

#: How far below a whole elixir a bar may read and still count as that elixir. The vector
#: carries the bar in half precision, so a bar of exactly 3 reads 2.998 and would otherwise be
#: counted as 2.
ELIXIR_TOLERANCE = 0.005


def play_rate_by_elixir(
    elixir: np.ndarray, n_legal: np.ndarray, actions: np.ndarray, *, noop: int = 0
) -> dict[str, float]:
    """``env/play_rate_by_elixir/{k}``: at each whole elixir, the share of choice rows played.

    Choice rows only (more than one legal action): a row where the no-op was the only legal
    action says nothing about when the policy chooses to wait. A level with no choice row is
    absent rather than 0.0, which would read as a policy that never plays there.
    """
    elixir = np.asarray(elixir, dtype=np.float64).reshape(-1)
    choice = np.asarray(n_legal).reshape(-1) > 1
    played = np.asarray(actions).reshape(-1) != noop
    level = np.floor(elixir + ELIXIR_TOLERANCE).astype(np.int64)
    fields: dict[str, float] = {}
    for k in np.unique(level[choice]):
        rows = choice & (level == k)
        fields[f"env/play_rate_by_elixir/{int(k)}"] = float(played[rows].mean())
    return fields
