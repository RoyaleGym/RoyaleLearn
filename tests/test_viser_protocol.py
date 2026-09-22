"""The learning-status wire protocol, checked against the other end of it.

``tests/test_viser_sink.py`` plays the viewer with a plain socket and asserts what this package
sends. That is the right test for the sink, and it cannot catch the one failure that matters
here: the protocol is implemented independently in this repository and in RoyaleViser, on
purpose -- the dependency direction is the viewer watching the learner, and pulling pygame into
a training run to share a constant would be the wrong trade -- so each side currently tests only
against its own copy of the constants. A drift between them renders as an empty panel, which is
indistinguishable from no learner attached.

So this file imports the viewer, under ``importorskip``, and asserts that the two copies are the
same. The ban is on the *package* importing ``royaleviser``; a test that skips cleanly where the
viewer is not installed is what keeps the two ends pinned together.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from royalelearn.metrics import viser_sink

model = pytest.importorskip("royaleviser.model")
sources = pytest.importorskip("royaleviser.sources")


def _learning_fields() -> tuple[str, ...]:
    """``royaleviser.model.Learning``'s own field order, less the extras map."""
    fields = tuple(model.Learning.__dataclass_fields__)
    return tuple(name for name in fields if name != "extra")


# -- the fields --------------------------------------------------------------


def test_the_panel_rows_are_the_viewers_own_fields_in_its_own_order() -> None:
    assert _learning_fields() == viser_sink.LEARNING_FIELDS


def test_every_field_the_sink_fills_is_a_field_the_viewer_draws() -> None:
    drawn = set(_learning_fields())
    assert set(viser_sink.FIELD_SOURCES) <= drawn
    assert "run" in drawn  # filled from the run directory rather than from a metric


def test_the_extras_map_exists_on_both_sides() -> None:
    assert "extra" in model.Learning.__dataclass_fields__
    assert {name for name, _key in viser_sink.EXTRA_SOURCES} == {"cards_per_match", "gate"}


# -- the envelope and the transport ------------------------------------------


def test_the_envelope_constants_agree() -> None:
    assert viser_sink.LEARNING_TAG == model.LEARNING_TAG
    assert viser_sink.LEARNING_PREFIX == model.LEARNING_PREFIX


def test_the_transport_constants_agree() -> None:
    assert viser_sink.VISER_HELLO == sources.STREAM_HELLO
    assert viser_sink.VISER_HEARTBEAT_S == sources.STREAM_HEARTBEAT_S
    assert viser_sink.VISER_ATTACH_TIMEOUT_S == sources.STREAM_ATTACH_TIMEOUT_S
    assert viser_sink.VISER_MAX_DATAGRAM == sources.STREAM_MAX_DATAGRAM
    assert viser_sink.VISER_REPEATS == sources.LEARNING_REPEATS
    assert viser_sink.VISER_LEARNING_PORT_OFFSET == sources.LEARNING_PORT_OFFSET
    assert viser_sink.VISER_HOST == sources.STREAM_HOST
    assert viser_sink.VISER_PORT == sources.STREAM_PORT


def test_the_default_endpoint_is_the_one_the_viewer_listens_on() -> None:
    assert viser_sink.learning_endpoint(None) == sources.learning_endpoint(
        sources.STREAM_HOST, sources.STREAM_PORT
    )
    assert viser_sink.learning_endpoint("10.0.0.4:7000") == sources.learning_endpoint(
        "10.0.0.4", 7000
    )


# -- a whole status, end to end ----------------------------------------------


def _row() -> dict[str, Any]:
    """One metric row wide enough to fill every panel row the sink knows how to fill."""
    row: dict[str, Any] = {key: 1.5 for key in viser_sink.FIELD_SOURCES.values()}
    row["run/iteration"] = 12
    row["ladder/pool_size"] = 7
    row["ladder/eval_games_total"] = 2200
    row["ladder/gate_failed_condition"] = "beats_champion"
    row["policy/cards_per_match"] = 21.5
    row["ladder/rating/learner"] = 1183.4
    row["ladder/rating_se/learner"] = 21.8
    return row


def test_a_status_this_package_encodes_decodes_as_the_viewers_own_type() -> None:
    sink = viser_sink.ViserSink(pump_thread=False)
    sink.run = "my-run"
    status = sink.status_of(_row())
    data = msgspec.msgpack.encode({viser_sink.LEARNING_TAG: status})
    assert model.is_learning(data)
    learning = model.decode_learning(data)
    assert learning.run == "my-run"
    assert learning.iteration == 12
    assert learning.pool_size == 7
    assert learning.games_vs_pool == 2200
    assert learning.extra["rating"] == "1183 ± 22"
    assert learning.extra["cards_per_match"] == 21.5
    assert learning.extra["gate"] == "beats_champion"
    for field in _learning_fields():
        assert getattr(learning, field) is not None, f"{field} arrived empty"


def test_an_omitted_field_stays_omitted_rather_than_becoming_a_zero() -> None:
    """The viewer draws an absent field as an em dash, and a zero would be a wrong number."""
    sink = viser_sink.ViserSink(pump_thread=False)
    status = sink.status_of({"run/iteration": 3})
    learning = model.decode_learning(
        msgspec.msgpack.encode({viser_sink.LEARNING_TAG: status})
    )
    assert learning.iteration == 3
    assert learning.kl is None
    assert learning.elo is None
