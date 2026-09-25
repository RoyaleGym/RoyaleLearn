"""Behaviour statistics computed from rows, one function per statistic.

Here rather than inside the coordinator's probe because more than one caller needs the same
number: the run's own sample of its rows, and any other source of rows that has to be compared
with it -- demonstrations driven through the environment, say. Two implementations of one
statistic would make every difference between a policy and a demonstration partly a difference
between two pieces of code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..obs_layout import HAND_CARD_ONEHOT, field_slice, hand_fields

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.metrics import MetricValue
    from ..api.rollout import EnvSpec

__all__ = [
    "ELIXIR_TOLERANCE",
    "RowBehaviour",
    "behaviour_fields",
    "max_mana",
    "play_rate_by_elixir",
]

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


def max_mana() -> float:
    """The full elixir bar, in elixir.

    The observation carries the bar as a fraction of it, and the metric is in elixir, so the
    number has to come from somewhere: it comes from the calibration the engine itself reads,
    never from a constant in this repository.
    """
    from royalegym.protocol import default_calibration

    return float(default_calibration().int("match.MAX_MANA"))


class RowBehaviour:
    """The ``policy/`` group and the elixir fields, from rows: each row's observation vector, its
    mask, and the action taken there.

    A class because the per-card counts behind the fields are kept for the heat map. The rows can
    come from anywhere -- a sample of a rollout buffer (``coordinator.PolicyProbe``), or a
    demonstration driven through the environment -- and are measured by this one piece of code.
    """

    def __init__(self, spec: EnvSpec) -> None:
        self.spec = spec
        self.hand = hand_fields(spec)
        self.elixir = field_slice(spec, "own_elixir")
        self.onehot = field_slice(spec, HAND_CARD_ONEHOT)
        self.tiles = spec.tiles[0] * spec.tiles[1]
        self.max_mana = max_mana()
        self.card_plays: dict[int, int] = {}
        self.card_legal: dict[int, int] = {}
        self.tile_plays: dict[int, int] = {}
        self.card_tile_plays: dict[tuple[int, int], int] = {}

    def fields(
        self, vector: np.ndarray, mask: np.ndarray, actions: np.ndarray
    ) -> dict[str, MetricValue]:
        """The fields from rows: each row's vector, its mask, and the action taken.

        Rows that never went through a rollout buffer -- a demonstration driven through the
        environment -- are measured by this same code, so a difference between a policy and a
        demonstration is never a difference between two implementations of one statistic.
        """
        fields: dict[str, MetricValue] = {}
        self.card_plays, self.tile_plays, self.card_tile_plays = {}, {}, {}
        mask = np.asarray(mask).astype(bool)
        actions = np.asarray(actions).astype(np.int64)
        legal = mask.sum(axis=-1).astype(np.float64)
        # Per hand slot rather than per action: a slot is playable when ANY of its tiles is.
        # This is what separates "the policy does not choose this card" from "this card is
        # rarely affordable", and nothing measured it before.
        slot_legal = mask[:, 1:].reshape(mask.shape[0], self.hand.hand_size, self.tiles).any(-1)
        fields["policy/legal_actions_mean"] = float(legal.mean())
        for percentile, name in ((5, "p05"), (50, "p50"), (95, "p95")):
            fields[f"policy/legal_actions_{name}"] = float(np.percentile(legal, percentile))
        fields["policy/forced_noop_frac"] = float(np.mean(legal <= 1))

        bar = vector[:, self.elixir].reshape(-1).astype(np.float64)
        fields["env/mean_elixir_at_decision"] = float(bar.mean() * self.max_mana)
        fields["env/frac_elixir_above_99"] = float(np.mean(bar >= 0.99))
        fields.update(play_rate_by_elixir(bar * self.max_mana, legal, actions))

        fields.update(self._plays(actions, vector[:, self.onehot], slot_legal))
        return fields

    def _plays(
        self, actions: np.ndarray, onehot: np.ndarray, slot_legal: np.ndarray
    ) -> dict[str, MetricValue]:
        """Where the cards went, and which cards they were.

        The action index says the hand slot and the tile; the card in that slot is read out of
        the observation's own one-hot, so the counters are per card rather than per hand
        position -- which is the only one of the two that means anything across a cycle.
        """
        fields: dict[str, MetricValue] = {}
        fields.update(self._availability(onehot, slot_legal))
        played = actions > 0
        fields["policy/noop_rate"] = float(np.mean(~played))
        if not played.any():
            fields["policy/tile_entropy"] = 0.0
            fields["policy/tile_top1_share"] = 0.0
            fields["policy/card_tile_top10_share"] = 0.0
            return fields
        index = actions[played] - 1
        slot = index // self.tiles
        tile = index % self.tiles
        block = self.hand.onehot_width
        rows = np.arange(onehot.shape[0])[played]
        cards = np.array(
            [
                int(np.argmax(onehot[row, s * block : (s + 1) * block]))
                for row, s in zip(rows, slot, strict=True)
            ],
            dtype=np.int64,
        )
        counts = np.bincount(tile, minlength=self.tiles).astype(np.float64)
        share = counts / counts.sum()
        positive = share[share > 0]
        fields["policy/tile_entropy"] = float(-(positive * np.log(positive)).sum())
        fields["policy/tile_top1_share"] = float(share.max())

        pairs: dict[tuple[int, int], int] = {}
        for card, position in zip(cards, tile, strict=True):
            pairs[(int(card), int(position))] = pairs.get((int(card), int(position)), 0) + 1
            self.card_plays[int(card)] = self.card_plays.get(int(card), 0) + 1
        self.card_tile_plays = pairs
        self.tile_plays = {int(t): int(c) for t, c in enumerate(counts) if c}
        top = sorted(pairs.values(), reverse=True)[:10]
        fields["policy/card_tile_top10_share"] = float(sum(top) / sum(pairs.values()))
        total = float(sum(self.card_plays.values()))
        for card, played_count in sorted(self.card_plays.items()):
            fields[f"policy/card_play_frac/{card}"] = played_count / total
            # The share of plays is a share of a policy AND of an elixir bar, and a reader
            # cannot tell which they are looking at. Divided by the decisions where the card
            # was affordable, it is the policy alone.
            affordable = self.card_legal.get(card, 0)
            if affordable:
                fields[f"policy/card_play_rate/{card}"] = played_count / affordable
        return fields

    def _availability(self, onehot: np.ndarray, slot_legal: np.ndarray) -> dict[str, MetricValue]:
        """How often each card was in the hand at all, and how often it was affordable.

        Without these, a per-card play share answers a question nobody asked. A three-cost card
        is legal at a decision far more often than a four-cost one when the bar averages about
        one and a half elixir, so a policy that chooses uniformly among what it can afford still
        plays cheap cards many times more often. Reading that as a preference is reading the
        elixir economy as a policy.
        """
        block = self.hand.onehot_width
        rows, hand = slot_legal.shape
        wide = onehot[:, : hand * block].reshape(rows, hand, block)
        cards = wide.argmax(axis=-1)
        self.card_legal = {}
        fields: dict[str, MetricValue] = {}
        for card in np.unique(cards):
            in_hand = cards == card
            legal = int(np.count_nonzero(in_hand & slot_legal))
            self.card_legal[int(card)] = legal
            fields[f"policy/card_in_hand_frac/{int(card)}"] = float(
                np.count_nonzero(in_hand.any(axis=-1)) / rows
            )
            fields[f"policy/card_legal_frac/{int(card)}"] = float(
                np.count_nonzero((in_hand & slot_legal).any(axis=-1)) / rows
            )
        return fields

    def heatmap(self) -> dict[str, Any]:
        """The per-card play counts over the tile grid, as the artifact a sink is handed."""
        tiles_y, tiles_x = self.spec.tiles
        return {
            "tiles": [tiles_y, tiles_x],
            "counts": {
                f"{card}/{tile}": n for (card, tile), n in sorted(self.card_tile_plays.items())
            },
        }


def behaviour_fields(
    spec: EnvSpec, vector: np.ndarray, mask: np.ndarray, actions: np.ndarray
) -> dict[str, MetricValue]:
    """``RowBehaviour(spec).fields(vector, mask, actions)``: the fields alone, for rows."""
    return RowBehaviour(spec).fields(vector, mask, actions)
