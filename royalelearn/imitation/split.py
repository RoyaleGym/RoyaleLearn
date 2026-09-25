"""Which rows are held out: by group, never by row (sections 19.10 and 19.12).

Rows of one match are not independent -- a decision and the one after it share almost every
input -- so a validation split drawn by row would measure how well a model remembers the match
it was trained on. The split is by ``group`` (whatever the producer uses to name a match), and it
is a pure function of the group's value, so every tool that splits the same rows splits them the
same way and a row never changes side between two runs.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["VALIDATION_PERCENT", "is_validation"]

#: Groups whose hash lands in the first five of a hundred buckets are validation.
VALIDATION_PERCENT = 5


def _bucket(group: int) -> int:
    digest = hashlib.sha256(int(group).to_bytes(8, "little", signed=False)).digest()
    return int.from_bytes(digest[:8], "big") % 100


def is_validation(groups: np.ndarray, *, percent: int = VALIDATION_PERCENT) -> np.ndarray:
    """``(N,)`` bool: whether each row's group is held out.

    The first eight bytes of sha256 over the group's eight little-endian bytes, read as a
    big-endian integer, modulo 100, below ``percent``. Computed once per distinct group.
    """
    values = np.asarray(groups, dtype=np.uint64).reshape(-1)
    unique, inverse = np.unique(values, return_inverse=True)
    held = np.array([_bucket(int(group)) < percent for group in unique], dtype=bool)
    return held[inverse] if values.size else np.zeros(0, dtype=bool)
