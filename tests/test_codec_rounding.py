"""A value rounded into a ``uint8`` plane is counted wherever it was packed, and reported.

Preflight decides each plane's storage from a sample. A plane whose values are fractions only in
battles the sample never reached (a feature it never showed) is admitted to a byte, and every
later value is stored rounded. That loss is counted by the codec that packed the row, in the
rollout worker as in the parent, carried home in the control word, summed into
``health/obs_codec_rounded`` and ``health/obs_codec_clipped``, named once in a line, and alarmed
on. ``rollout.codec_float16_planes`` stores a named plane as float16, which keeps it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import msgspec
import pytest

from rollout_support import PARENT_PID_ENV, SUPPORT_MODULE
from royalelearn import config as cfg
from royalelearn.rollout.envspec import ComponentSpec
from royalelearn.rollout.inline import round_timings

pytest.importorskip("torch")


def test_unset_nothing_is_encoded() -> None:
    encoded = msgspec.json.encode(cfg.RolloutConfig())
    assert b"codec_float16_planes" not in encoded and b"codec_unseen_float16" not in encoded
    assert cfg.codec_float16_planes_of(cfg.RolloutConfig()) == ()
    assert cfg.codec_unseen_float16_of(cfg.RolloutConfig()) is False


def test_the_counts_ride_the_control_words_home() -> None:
    """Summed over the workers; the first rounded plane is the lowest-numbered worker's.
    Plant: read one worker's word, and a second worker's rounding goes unseen."""
    words = {
        1: {"t_env_ns": 0, "codec_clipped": 1, "codec_rounded": 5, "codec_round_plane": 4},
        0: {"t_env_ns": 0, "codec_clipped": 0, "codec_rounded": 2, "codec_round_plane": 7},
        2: {"t_env_ns": 0, "codec_clipped": 0, "codec_rounded": 0, "codec_round_plane": 0},
    }
    timings = round_timings(words, {})
    assert (timings["codec_clipped"], timings["codec_rounded"]) == (1, 7)
    assert timings["codec_round_plane"] == 6
    assert round_timings({}, {})["codec_round_plane"] == -1


def test_a_rounded_value_raises_the_alarm() -> None:
    from royalelearn.metrics.alarms import default_alarms

    (alarm,) = [a for a in default_alarms(cfg.AlarmConfig()) if a.name == "obs_codec_rounded"]
    assert alarm.holds({"health/obs_codec_rounded": 3})
    assert not alarm.holds({"health/obs_codec_rounded": 0})
    assert "codec_float16_planes" in alarm.message({"health/obs_codec_rounded": 3})


def _first_byte_plane(tmp_path: Path) -> str:
    """A plane this environment's codec stores in a byte by default."""
    from royalelearn.rollout.codec import STORAGE_UINT8
    from royalelearn.testing import coordinator, tiny_config

    with coordinator(tiny_config(tmp_path / "probe")) as run:
        table = run.report.table
    return next(name for name, storage, _ in table.plane if storage == STORAGE_UINT8)


def _config(tmp_path: Path, plane: str, **rollout: Any) -> cfg.RunConfig:
    from royalelearn.testing import tiny_config

    base = tiny_config(tmp_path)
    env = msgspec.structs.replace(
        base.env,
        obs_builder=ComponentSpec(f"{SUPPORT_MODULE}.FractionalObsBuilder", {"plane": plane}),
    )
    return tiny_config(
        tmp_path,
        env=env,
        extra_component_modules=[SUPPORT_MODULE],
        rollout=msgspec.structs.replace(base.rollout, **rollout),
    )


def _rounded(config: cfg.RunConfig, *, fraction_from: str) -> tuple[int, int, list[str]]:
    """``(rounded, clipped)`` of a two-iteration run's last row, and its printed lines. The
    builder's fraction appears in every process but ``fraction_from``'s."""
    from royalelearn.testing import coordinator

    lines: list[str] = []
    if fraction_from == "parent":
        os.environ[PARENT_PID_ENV] = str(os.getpid())
    try:
        with coordinator(config, printer=lines.append) as run:
            if fraction_from == "nobody":
                # After preflight's sample: from here every observation is fractional.
                os.environ[PARENT_PID_ENV] = "0"
            run.iterate()
            run.iterate()
            row = run.rows[-1]
    finally:
        os.environ.pop(PARENT_PID_ENV, None)
    return int(row["health/obs_codec_rounded"]), int(row["health/obs_codec_clipped"]), lines


def test_an_inline_run_reports_the_values_its_shards_rounded(tmp_path: Path) -> None:
    """Plant: report the coordinator's own codec, which no shard packs through the counts of,
    and the row reads zero beside a rollout that rounded every observation."""
    plane = _first_byte_plane(tmp_path)
    rounded, clipped, lines = _rounded(_config(tmp_path / "run", plane), fraction_from="nobody")
    assert rounded > 0 and clipped == 0
    said = [line for line in lines if "values rounded" in line]
    assert len(said) == 1 and repr(plane) in said[0]


def test_a_plane_named_float16_keeps_its_fractions(tmp_path: Path) -> None:
    plane = _first_byte_plane(tmp_path)
    config = _config(tmp_path / "run", plane, codec_float16_planes=(plane,))
    rounded, _clipped, lines = _rounded(config, fraction_from="nobody")
    assert rounded == 0
    assert not [line for line in lines if "values rounded" in line]


@pytest.mark.slow
def test_a_process_run_reports_what_its_workers_rounded(tmp_path: Path) -> None:
    """Each worker has a codec of its own, and the parent's never packs a training row. Plant:
    leave the counts out of the control word, and this reads zero."""
    plane = _first_byte_plane(tmp_path)
    config = _config(
        tmp_path / "run", plane, source="process", workers=2, round_timeout_s=60.0
    )
    rounded, clipped, lines = _rounded(config, fraction_from="parent")
    assert rounded > 0 and clipped == 0
    assert any("values rounded" in line and repr(plane) in line for line in lines)
