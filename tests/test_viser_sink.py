"""The learning status, played from the other end with a plain socket.

This file imports nothing from RoyaleViser on purpose. It states the protocol -- the hello, the
one-key msgpack map, the field names, the re-send on a fresh hello, the silence while nobody is
watching -- so that a change on either side fails here rather than in somebody's window.
"""

from __future__ import annotations

import contextlib
import socket

import msgspec
import pytest

from royalelearn.identity import EngineBuild, RunIdentity
from royalelearn.metrics.viser_sink import (
    EXTRA_SOURCES,
    FIELD_SOURCES,
    LEARNING_FIELDS,
    LEARNING_PREFIX,
    LEARNING_TAG,
    VISER_HELLO,
    VISER_LEARNING_PORT_OFFSET,
    ViserSink,
    learning_endpoint,
)

IDENTITY = RunIdentity(
    format_version=1,
    royalelearn_version="0.1.0",
    royalelearn_git="unknown",
    royalegym_version="0.1.0",
    royalegym_git="unknown",
    engine_build=EngineBuild(
        engine_class="royalegym.mock_engine.MockEngine",
        calibration_digest="c",
        build_digest="b",
        catalogue_sha256="a",
        path_search=None,
        stale_build_differences=[],
    ),
    env_spec_digest="e",
    obs_digest="o",
    action_digest="d",
    frame_stack=1,
    arch_digest="r",
    codec_version=1,
    codec_table_digest="t",
    algo_digest="g",
    rollout_digest="l",
    ladder_digest="p",
    master_seed=7,
    determinism_tier="run_exact",
    torch_version="none",
    device_kind="cpu",
)


def _row(iteration: int = 7) -> dict[str, object]:
    """A row carrying every metric the panel reads, and a few it does not."""
    row: dict[str, object] = {key: 0.25 for key in FIELD_SOURCES.values()}
    row["run/iteration"] = iteration
    row["ladder/pool_size"] = 12
    row["ladder/eval_games_total"] = 4400
    row["policy/cards_per_match"] = 21.5
    row["ladder/gate_failed_condition"] = "pool_collapse"
    row["ladder/rating/learner"] = 1183.4
    row["ladder/rating_se/learner"] = 21.7
    row["health/rss_peak_mb"] = 900.0  # a key the panel has no row for
    return row


@pytest.fixture
def viewer():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1.0)
    yield sock
    sock.close()


@pytest.fixture
def sink(tmp_path):
    published = ViserSink(host="127.0.0.1", port=0, pump_thread=False)
    published.open(identity=IDENTITY, config_json="{}", run_dir=tmp_path / "run-0007")
    yield published
    published.close()


def _receive(viewer: socket.socket) -> dict:
    data, _address = viewer.recvfrom(65535)
    assert data.startswith(LEARNING_PREFIX)
    message = msgspec.msgpack.decode(data)
    assert set(message) == {LEARNING_TAG}
    return message[LEARNING_TAG]


def test_nothing_is_sent_before_a_hello(sink, viewer) -> None:
    for iteration in range(50):
        sink.write(_row(iteration))
    assert sink.sent == 0
    assert not sink.attached
    with pytest.raises(TimeoutError):
        viewer.recvfrom(65535)


def test_a_hello_brings_the_status(sink, viewer) -> None:
    viewer.sendto(VISER_HELLO, sink.address)
    sink.write(_row(7))
    status = _receive(viewer)
    assert sink.sent == 1
    assert status["run"] == "run-0007"
    assert status["iteration"] == 7
    for field in LEARNING_FIELDS:
        assert field in status, field
    assert isinstance(status["iteration"], int)
    assert isinstance(status["pool_size"], int)
    assert isinstance(status["games_vs_pool"], int)
    assert isinstance(status["policy_loss"], float)
    assert isinstance(status["run"], str)


def test_the_extras_are_the_three_worth_the_space(sink, viewer) -> None:
    viewer.sendto(VISER_HELLO, sink.address)
    sink.write(_row())
    extra = _receive(viewer)["extra"]
    assert len(extra) <= 3
    assert extra["rating"] == "1183 ± 22"
    assert extra["cards / match"] == 21.5
    assert extra["gate"] == "pool_collapse"
    assert set(extra) == {"rating", *(name for name, _key in EXTRA_SOURCES)}


def test_a_fresh_hello_re_sends_the_last_status_unchanged(sink, viewer) -> None:
    viewer.sendto(VISER_HELLO, sink.address)
    sink.write(_row(11))
    first = _receive(viewer)
    assert sink.pump() is True
    assert _receive(viewer) == first
    assert sink.sent == 2


def test_one_message_is_the_whole_status(sink, viewer) -> None:
    """A row missing a metric leaves that field out rather than sending a zero for it."""
    viewer.sendto(VISER_HELLO, sink.address)
    partial = _row(3)
    del partial["ppo/kl"]
    del partial["ladder/rating_se/learner"]
    sink.write(partial)
    status = _receive(viewer)
    assert "kl" not in status
    assert status["extra"]["rating"] == "1183"
    assert status["iteration"] == 3


def test_a_row_the_panel_has_no_row_for_is_simply_not_sent(sink, viewer) -> None:
    viewer.sendto(VISER_HELLO, sink.address)
    sink.write(_row())
    status = _receive(viewer)
    assert set(status) <= {*LEARNING_FIELDS, "extra"}


def test_a_viewer_that_goes_away_costs_nothing(sink, viewer) -> None:
    viewer.sendto(VISER_HELLO, sink.address)
    sink.write(_row(1))
    assert sink.sent == 1
    viewer.close()
    # The heartbeat is what keeps the sender awake; without one it falls silent by itself.
    sink._last_hello -= 10.0
    assert not sink.attached
    before = sink.sent
    for iteration in range(50):
        sink.write(_row(iteration))
    assert sink.sent == before


def test_the_endpoint_is_the_state_streams_port_plus_one(monkeypatch) -> None:
    monkeypatch.delenv("ROYALEVISER", raising=False)
    host, port = learning_endpoint()
    assert (host, port) == ("127.0.0.1", 9870 + VISER_LEARNING_PORT_OFFSET)
    assert learning_endpoint("10.0.0.4:7000") == ("10.0.0.4", 7000 + VISER_LEARNING_PORT_OFFSET)
    monkeypatch.setenv("ROYALEVISER", "127.0.0.1:9000")
    assert learning_endpoint() == ("127.0.0.1", 9000 + VISER_LEARNING_PORT_OFFSET)


def test_a_port_that_cannot_be_bound_is_not_a_failed_run(tmp_path, capsys) -> None:
    holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    holder.bind(("127.0.0.1", 0))
    taken = holder.getsockname()[1]
    try:
        sink = ViserSink(host="127.0.0.1", port=taken, pump_thread=False)
        sink.open(identity=IDENTITY, config_json="{}", run_dir=tmp_path / "run")
        assert sink.unavailable
        assert "learning status port" in capsys.readouterr().out
        sink.write(_row())
        assert sink.sent == 0
        sink.close()
    finally:
        holder.close()


def test_the_sink_does_not_import_the_viewer() -> None:
    """The sink speaks the viewer's datagram; it does not depend on the viewer.

    Asked of a fresh interpreter rather than of this one. ``royaleviser`` is installed in the
    same environment, so whether it is in ``sys.modules`` here says what the rest of the session
    imported, not what this module does.
    """
    import subprocess
    import sys
    from pathlib import Path

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import royalelearn.metrics.viser_sink, sys; "
            "print('royaleviser' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


def test_a_taken_port_is_retried_rather_than_given_up_on() -> None:
    """The usual holder of the learning port is another training run, and that run ends.

    A sink that gave up for good would leave a fourteen-hour run with a dark panel because
    of a collision in its first second, and the viewer cannot tell that from no learner.
    """
    import socket as _socket

    held = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    held.bind(("127.0.0.1", 0))
    host, port = held.getsockname()[:2]
    sink = ViserSink(host=host, port=port, pump_thread=False)
    try:
        sink._bind()
        assert sink.unavailable is not None, "the port was free after all"
        assert sink._retry_at > 0.0

        sink._bind()  # too soon: the clock holds it off without touching the socket
        assert sink._socket is None

        held.close()
        sink._retry_at = 0.0
        sink._bind()
        assert sink._socket is not None, "the port was free and the sink stayed dark"
        assert sink.unavailable is None
    finally:
        sink.close()
        with contextlib.suppress(OSError):
            held.close()
