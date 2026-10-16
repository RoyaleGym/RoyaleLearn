"""Policy transforms for ``tests/test_policy_transform.py``.

Each is named in a config as ``plugin:transform_plugins:<name>``.
"""

from __future__ import annotations

from typing import Any


def sharpen(logits: Any, mask: Any, vector: Any, layout: Any) -> Any:
    """A temperature of one half."""
    return logits * 2.0


def gate(logits: Any, mask: Any, vector: Any, layout: Any) -> Any:
    """Moves the no-op against the rest, by the layout: a row's own numbers only."""
    import torch

    shift = torch.zeros_like(logits)
    shift[:, 0] = 1.5
    shift[:, 1 + layout.hand_size * layout.tiles :] = -0.5
    return logits + shift


def narrower(logits: Any, mask: Any, vector: Any, layout: Any) -> Any:
    return logits[:, :-1]


def float64(logits: Any, mask: Any, vector: Any, layout: Any) -> Any:
    return logits.double()


def detached(logits: Any, mask: Any, vector: Any, layout: Any) -> Any:
    return logits.detach() * 2.0


def not_a_number(logits: Any, mask: Any, vector: Any, layout: Any) -> Any:
    out = logits.clone()
    out[:, 0] = float("nan")
    return out


def mixes_rows(logits: Any, mask: Any, vector: Any, layout: Any) -> Any:
    return logits - logits.mean(dim=0, keepdim=True)
