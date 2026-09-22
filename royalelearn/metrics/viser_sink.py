"""The learning status the viewer's panel reads.

RoyaleViser draws a learning panel beside the battle it is already showing, fed by a second
datagram sender on the state stream's port plus one. This module is that sender, in about a
hundred lines of ``socket`` and msgpack. It does **not** import ``royaleviser``: that package
pulls in pygame, and the dependency direction is the viewer watching the learner and never the
reverse. What it does instead is pin the protocol's constants here, beside the code that
speaks it, so a drift between the two ends is caught by this package's own test rather than in
somebody's window.

The protocol, in full:

* the learner binds ``host:port``, taken from the same ``ROYALEVISER`` setting the environment's
  state stream uses, with the learning port defaulting to the state port plus one -- two
  processes cannot bind one port;
* it sends nothing at all until a ``royaleviser 1`` hello arrives. The viewer sends one every
  second and counts as attached while one has arrived within three seconds, so a detached run
  costs one monotonic clock read per iteration and no traffic;
* while attached it sends one msgpack map ``{"learning": {...}}`` per iteration, and re-sends
  the last status on a fresh hello so a viewer that attaches mid-iteration fills immediately;
* **one message is the whole status.** The viewer replaces rather than merges and draws an
  omitted field as an em dash, so a partial update would blank the panel rather than leave it
  stale, and a field that was never sent is never a zero;
* nothing acknowledges a datagram, so one status is sent to one viewer several times, a
  heartbeat apart.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from ..api.metrics import MetricRow, MetricsSink

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..identity import RunIdentity

__all__ = [
    "LEARNING_FIELDS",
    "LEARNING_PREFIX",
    "LEARNING_TAG",
    "VISER_ATTACH_TIMEOUT_S",
    "VISER_ENV_VAR",
    "VISER_HEARTBEAT_S",
    "VISER_HELLO",
    "VISER_HOST",
    "VISER_LEARNING_PORT_OFFSET",
    "VISER_MAX_DATAGRAM",
    "VISER_PORT",
    "VISER_REPEATS",
    "ViserSink",
    "learning_endpoint",
    "status_fields",
]

#: The viewer's heartbeat, sent to the learner's port once a second while its window is open.
VISER_HELLO = b"royaleviser 1"
#: How often it arrives, and how often to look for it.
VISER_HEARTBEAT_S = 1.0
#: No hello for this long: detached, send nothing.
VISER_ATTACH_TIMEOUT_S = 3.0
#: One datagram, UDP over IPv4.
VISER_MAX_DATAGRAM = 65507
#: Copies of one status per viewer, a heartbeat apart, because nothing is acknowledged.
VISER_REPEATS = 3
#: The one-key envelope. Its first bytes are msgpack for a one-key map named ``learning``,
#: which is what lets the viewer tell a status from a frame with one comparison and no decode.
LEARNING_TAG = "learning"
LEARNING_PREFIX = b"\x81\xa8learning"
#: The learning port is the state stream's plus one.
VISER_LEARNING_PORT_OFFSET = 1
VISER_HOST = "127.0.0.1"
VISER_PORT = 9870
VISER_ENV_VAR = "ROYALEVISER"

#: The panel's fixed rows, in the order the viewer lays them out. A key that is neither one of
#: these nor ``extra`` is ignored at the other end, so a misspelling leaves an em dash rather
#: than showing up as a wrong number somewhere else.
LEARNING_FIELDS: tuple[str, ...] = (
    "run",
    "iteration",
    "policy_loss",
    "value_loss",
    "entropy",
    "kl",
    "clip_frac",
    "explained_var",
    "grad_norm",
    "learning_rate",
    "env_steps_per_s",
    "engine_ticks_per_s",
    "episode_ticks",
    "crowns_per_episode",
    "towers_per_episode",
    "illegal_rate",
    "elixir_wasted",
    "elo",
    "win_rate",
    "pool_size",
    "games_vs_pool",
)

#: Which metric fills each panel row. One place, so that a renamed metric is one edit and a
#: row the harness does not measure is visibly absent rather than silently zero.
FIELD_SOURCES: dict[str, str] = {
    "iteration": "run/iteration",
    "policy_loss": "ppo/policy_loss",
    "value_loss": "ppo/value_loss",
    "entropy": "ppo/entropy",
    "kl": "ppo/kl",
    "clip_frac": "ppo/clip_fraction",
    "explained_var": "ppo/explained_variance",
    "grad_norm": "ppo/grad_norm_actor",
    "learning_rate": "run/lr_actor",
    "env_steps_per_s": "throughput/overall_steps_per_second",
    "engine_ticks_per_s": "throughput/engine_ticks_per_second",
    "episode_ticks": "env/ticks_mean",
    "crowns_per_episode": "env/crown_diff",
    "towers_per_episode": "env/crowns_for",
    "illegal_rate": "env/illegal_action_rate",
    # The viewer's row is named for elixir per episode; what a row of this harness carries is
    # the share of decisions taken at a full bar, which is what the environment measures.
    "elixir_wasted": "env/elixir_leak_frac",
    "elo": "ladder/elo_readout",
    "win_rate": "ladder/gate_observed_rate",
    "pool_size": "ladder/pool_size",
    "games_vs_pool": "ladder/eval_games_total",
}

#: The panel fits about three rows under the fixed ones, so these are the three worth having:
#: the fitted rating with its interval, the no-op collapse metric, and how the last gate went.
EXTRA_SOURCES: tuple[tuple[str, str], ...] = (
    ("cards_per_match", "policy/cards_per_match"),
    ("gate", "ladder/gate_failed_condition"),
)


#: The three rows the viewer's model types as integers.
_INTEGER_FIELDS = frozenset({"iteration", "pool_size", "games_vs_pool"})


def learning_endpoint(setting: str | None = None) -> tuple[str, int]:
    """``host:port`` for the status stream, from the same setting as the state stream.

    Unset, it is the default state port plus one, so attaching a viewer to a training run stays
    one environment variable rather than two.
    """
    raw = setting if setting is not None else os.environ.get(VISER_ENV_VAR, "")
    host, port = VISER_HOST, VISER_PORT
    if raw:
        head, _, tail = raw.rpartition(":")
        if tail.isdigit():
            host, port = (head or VISER_HOST), int(tail)
        else:
            host = raw
    return host, port + VISER_LEARNING_PORT_OFFSET


class ViserSink(MetricsSink):
    """One status datagram per iteration, to a viewer that says it is there."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        host: str | None = None,
        port: int | None = None,
        pump_thread: bool = True,
        learner_id: str = "learner",
    ) -> None:
        resolved_host, resolved_port = learning_endpoint(endpoint)
        self.host = host if host is not None else resolved_host
        self.port = int(port) if port is not None else resolved_port
        self.learner_id = learner_id
        self.sent = 0
        self.dropped = 0
        self.unavailable: str | None = None
        self.run = ""
        self._status: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._socket: socket.socket | None = None
        self._peer: tuple[str, int] | None = None
        self._sent_to: tuple[str, int] | None = None
        self._repeats = 0
        self._last_hello = 0.0
        self._last_poll = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._want_thread = pump_thread

    # -- the sink -----------------------------------------------------------

    def open(self, *, identity: RunIdentity, config_json: str, run_dir: Path) -> None:
        self.run = Path(run_dir).name
        self._bind()

    def write(self, row: MetricRow) -> None:
        status = self.status_of(row)
        with self._lock:
            self._status = status
            self._sent_to, self._repeats = None, VISER_REPEATS
            if self._attached(time.monotonic()):
                self._send()

    def pump(self) -> bool:
        """Look at the socket now and send the standing status to a viewer still owed it.

        Either this or the daemon thread has to be called about once a second: a learner inside
        an optimisation step calls nothing for a long time, and a viewer that attached in the
        meantime would sit on an empty panel until the iteration ended.
        """
        now = time.monotonic()
        with self._lock:
            self._poll(now)
            if not self._fresh(now):
                return False
            if self._sent_to == self._peer and self._repeats <= 0:
                return False
            return self._send()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2 * VISER_HEARTBEAT_S)
        self._thread = None

    def save_checkpoint(self, folder: Path) -> None:
        return

    def load_checkpoint(self, folder: Path, *, strict: bool = True) -> None:
        return

    @property
    def address(self) -> tuple[str, int]:
        """Where a viewer should say hello. Bound lazily, so read it after ``open``."""
        self._bind()
        if self._socket is None:
            raise RuntimeError(f"the learning status port is not bound: {self.unavailable}")
        return self._socket.getsockname()[:2]

    @property
    def attached(self) -> bool:
        """Whether a viewer said hello recently. One clock read; the socket at most once a
        second."""
        now = time.monotonic()
        with self._lock:
            return self._attached(now)

    # -- the status ---------------------------------------------------------

    def status_of(self, row: MetricRow) -> dict[str, Any]:
        """One metric row as one whole status.

        Only the fields this row actually carries: the viewer replaces what it holds rather
        than merging, and it draws a field nobody sent as an em dash, which is the truth. A
        zero would not be.
        """
        status: dict[str, Any] = {"run": self.run}
        for field, key in FIELD_SOURCES.items():
            value = row.get(key)
            if value is None:
                continue
            status[field] = (
                int(value) if field in _INTEGER_FIELDS else _number(value)
            )
        extra: dict[str, Any] = {}
        rating = self._rating_text(row)
        if rating is not None:
            extra["rating"] = rating
        for name, key in EXTRA_SOURCES:
            value = row.get(key)
            if value is not None:
                extra[name] = value if isinstance(value, str) else _number(value)
        if extra:
            status["extra"] = extra
        return status

    def _rating_text(self, row: MetricRow) -> str | None:
        """The fitted rating and its standard error as one line, ``"1183 ± 22"``.

        As text rather than as two numbers because the panel has one row for it, and because a
        rating without its interval is the number this whole ladder exists to stop anybody
        reading on its own.
        """
        rating = row.get(f"ladder/rating/{self.learner_id}")
        se = row.get(f"ladder/rating_se/{self.learner_id}")
        if rating is None:
            return None
        if se is None:
            return f"{float(rating):.0f}"
        return f"{float(rating):.0f} ± {float(se):.0f}"

    # -- the socket ---------------------------------------------------------

    def _bind(self) -> None:
        """Bind the learning port, or give up on the panel and say so once.

        A panel nobody can see is a nuisance; a training run that will not start because
        something else holds a UDP port is not a trade worth making.
        """
        with self._lock:
            if self._socket is not None or self.unavailable:
                return
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind((self.host, self.port))
            except OSError as exc:
                sock.close()
                self.unavailable = str(exc)
                print(
                    f"the learning status port {self.host}:{self.port} is not available "
                    f"({exc}); this run publishes no learning panel"
                )
                return
            sock.setblocking(False)
            self._socket = sock
            if self._want_thread and self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="royalelearn-viser", daemon=True
                )
                self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(VISER_HEARTBEAT_S):
            # One bad status costs that status and never the thread: a learner whose panel
            # stops updating is a nuisance, a learner that dies in this thread is not.
            with contextlib.suppress(Exception):
                self.pump()

    def _attached(self, now: float) -> bool:
        if self._socket is None:
            return False
        if now - self._last_poll >= VISER_HEARTBEAT_S:
            self._poll(now)
        return self._fresh(now)

    def _fresh(self, now: float) -> bool:
        return self._peer is not None and now - self._last_hello < VISER_ATTACH_TIMEOUT_S

    def _poll(self, now: float) -> None:
        """Drain the heartbeats waiting on the socket; the viewer's address is in them."""
        self._last_poll = now
        sock = self._socket
        while sock is not None:
            try:
                data, address = sock.recvfrom(64)
            except (BlockingIOError, ConnectionResetError, OSError):
                return
            if data != VISER_HELLO:
                continue
            peer = (address[0], address[1])
            if peer != self._peer or now - self._last_hello >= VISER_ATTACH_TIMEOUT_S:
                # Another viewer, or one away long enough to have been restarted: either way
                # it holds no status, so it is owed the standing one.
                self._sent_to, self._repeats = None, VISER_REPEATS
            self._peer, self._last_hello = peer, now

    def _send(self) -> bool:
        if self._socket is None or self._peer is None or self._status is None:
            return False
        try:
            data = msgspec.msgpack.encode({LEARNING_TAG: self._status})
        except (TypeError, ValueError):
            self.dropped += 1
            self._repeats = 0
            return False
        if len(data) > VISER_MAX_DATAGRAM:  # a status this big is an ``extra`` gone wrong
            self.dropped += 1
            self._repeats = 0
            return False
        try:
            self._socket.sendto(data, self._peer)
        except OSError:
            return False  # the viewer went away between its hello and this send
        self._sent_to = self._peer
        self._repeats = max(0, self._repeats - 1)
        self.sent += 1
        return True


def _number(value: Any) -> Any:
    """A metric value as something msgpack has a type for."""
    if isinstance(value, bool | int | float | str):
        return value
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def status_fields(status: Mapping[str, Any]) -> tuple[str, ...]:
    """The fixed rows a status carries, in the panel's own order. For tests and for the CLI."""
    return tuple(field for field in LEARNING_FIELDS if field in status)
