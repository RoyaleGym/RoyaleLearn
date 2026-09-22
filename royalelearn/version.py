"""The package version, and the git description of the checkout it was run from.

``__version__`` is the released number and is what ``pyproject.toml`` declares.
``git_describe()`` is the working-tree detail that goes into a run's identity record and
into every checkpoint manifest, so that a result can be traced back to the exact source
that produced it -- including the ``-dirty`` suffix when it was produced from uncommitted
changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__version__ = "0.1.0"

UNKNOWN = "unknown"


def git_describe() -> str:
    """``git describe --always --dirty`` for this checkout, or ``"unknown"``.

    Never raises: an installed copy with no ``.git`` beside it, a machine without git and
    a git that fails for any other reason all give ``"unknown"``, because a missing
    provenance string must not stop a run.
    """
    repo = Path(__file__).resolve().parent.parent
    try:
        completed = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if completed.returncode != 0:
        return UNKNOWN
    described = completed.stdout.strip()
    return described or UNKNOWN
