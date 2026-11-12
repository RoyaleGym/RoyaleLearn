"""``time/update_*``: where the update's epochs spent their seconds.

On CUDA each phase is a pair of events read after the update's one synchronisation, so the split
costs nothing the update would not wait for; on the processor the work is synchronous and the
wall clock is the measure, which is what runs here. Each path reports the phases it runs: the
critic apart from the actor only where ``ppo.forced_rows`` skips, the terms only with terms.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from royalelearn.testing import coordinator, tiny_config


@pytest.mark.parametrize(
    "forced_rows,phases",
    [
        ("all", {"actor", "backward", "optimizer"}),
        ("critic_only", {"critic", "actor", "backward", "optimizer"}),
    ],
)
def test_the_update_says_where_its_seconds_went(
    tmp_path: Path, forced_rows: str, phases: set[str]
) -> None:
    """Plant: drop a phase's timing, and it is missing here; time one twice, and the split
    exceeds the update it is part of."""
    base = tiny_config(tmp_path)
    config = tiny_config(tmp_path, ppo=msgspec.structs.replace(base.ppo, forced_rows=forced_rows))
    with coordinator(config) as run:
        run.iterate()
        row = dict(run.rows[-1])
    found = {key.removeprefix("time/update_") for key in row if key.startswith("time/update_")}
    assert found == phases
    split = sum(float(row[f"time/update_{phase}"]) for phase in phases)
    assert all(float(row[f"time/update_{phase}"]) > 0.0 for phase in phases)
    assert split <= float(row["time/update"])
