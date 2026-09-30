"""Regenerate ``metrics-row.json``: iteration 4 of the suite's tiny MockEngine run.

The alarm-plant baseline is a row the harness itself wrote, not one written by hand, so an
alarm gated on a key no row carries fails in ``tests/test_alarm_plants.py``. It comes from a CPU
run, so the CUDA-only keys (``health/vram_*``, ``throughput/gpu_util_frac``) are absent, as they
are from any CPU run.

    python tests/data/make_metrics_row.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from royalelearn.testing import coordinator, tiny_config


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp, coordinator(tiny_config(Path(tmp))) as run:
        for _ in range(4):
            run.iterate()
        row = dict(run.rows[-1])
    out = Path(__file__).with_name("metrics-row.json")
    out.write_text(json.dumps(row, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(row)} keys, iteration {row.get('run/iteration')})")


if __name__ == "__main__":
    main()
