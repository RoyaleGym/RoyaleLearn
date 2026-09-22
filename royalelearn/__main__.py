"""``python -m royalelearn``.

The module body is one call. Everything the entry point has to do before torch is imported --
the deterministic cuBLAS workspace and the BLAS thread counts -- ``cli`` does as it is imported,
which is why this file imports it and nothing else.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
