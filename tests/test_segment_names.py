"""Shared-memory segment names fit the strictest platform.

macOS caps a POSIX shared-memory name at 31 bytes, the leading slash included (PSHMNAMLEN), and
refuses a longer one with ``OSError: File name too long`` when the segment is created. Linux
allows 255 and Windows does not use the name that way, so a long name failed on macOS alone.
Every name is held to 30 characters before the slash, and a longer one is refused on every
platform, so a Windows or Linux run finds it first.
"""

from __future__ import annotations

from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import pytest

from royalelearn.rollout.layout import (
    MAX_SEGMENT_NAME,
    buffer_segment_name,
    checked_segment_name,
    control_segment_name,
)
from royalelearn.testing import coordinator, tiny_config

#: A run id as the identity makes one.
RUN_ID = "03b1e4151d5a376d"


def test_the_limit_is_macos_less_its_slash() -> None:
    assert MAX_SEGMENT_NAME == 30


def test_the_longest_names_a_run_can_make_fit() -> None:
    # The buffer's name as the coordinator suffixes it: a process id and a counter.
    longest_buffer = f"{buffer_segment_name(RUN_ID)}-{0xFFFFFFFF:x}-{99_999}"
    # A worker's control segment, after a stale segment of the same name made it retry.
    longest_control = f"{control_segment_name(RUN_ID, 999)}-{99}"
    for name in (longest_buffer, longest_control):
        assert len(name) <= MAX_SEGMENT_NAME, name
        assert checked_segment_name(name) == name


def test_a_name_too_long_for_macos_is_refused_everywhere() -> None:
    with pytest.raises(ValueError, match="macOS"):
        checked_segment_name("x" * (MAX_SEGMENT_NAME + 1))


def _recording(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    created: list[str] = []
    original = shared_memory.SharedMemory.__init__

    def init(self: Any, name: Any = None, create: bool = False, size: int = 0, **kw: Any) -> None:
        if create and name is not None:
            created.append(str(name))
        original(self, name=name, create=create, size=size, **kw)

    monkeypatch.setattr(shared_memory.SharedMemory, "__init__", init)
    return created


def test_every_segment_a_run_creates_has_a_short_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _recording(monkeypatch)
    with coordinator(tiny_config(tmp_path)) as run:
        run.iterate()
    assert created, "the run created no named segment, so nothing was checked"
    too_long = [name for name in created if len(name) > MAX_SEGMENT_NAME]
    assert not too_long, f"segments named past {MAX_SEGMENT_NAME} characters: {too_long}"


def test_every_worker_control_segment_has_a_short_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from rollout_support import preflight, rollout_config
    from royalelearn.rollout.farm import ProcessRolloutSource
    from royalelearn.rollout.preflight import DEFAULT_CODEC

    created = _recording(monkeypatch)
    config = rollout_config(workers=2, games_per_worker=2, shards_per_worker=1)
    report = preflight(config, codec=DEFAULT_CODEC)
    source = ProcessRolloutSource(
        config, report.spec, report.table, run_id=RUN_ID, codec=DEFAULT_CODEC
    )
    try:
        # What spawning a worker does first; the processes themselves are not needed here.
        for index in range(2):
            source.workers.append(source._build_worker(index))
    finally:
        source.close()
    assert len(created) >= 2, f"expected a control segment per worker, saw {created}"
    too_long = [name for name in created if len(name) > MAX_SEGMENT_NAME]
    assert not too_long, f"segments named past {MAX_SEGMENT_NAME} characters: {too_long}"
