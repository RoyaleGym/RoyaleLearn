"""The control file, as the shells a reader actually uses write it.

`docs/running.md` tells the reader to write `c`, `q` or `p` into a file to checkpoint, quit or
pause a run. Every redirection form in Windows PowerShell 5.1 -- the default shell on Windows 10
and 11, and the one the README writes every command block for -- puts a BYTE ORDER MARK at the
front of the file. A UTF-8 BOM is `\ufeff`, which is not whitespace, so `.strip()` leaves it and
the first character of the file is the BOM rather than the letter.

The consequence was silent in both directions. The read UNLINKS the file before the letter is
validated, so an unrecognised first character consumed the command and printed nothing: every
documented interactive control did nothing at all for a PowerShell user, with no error and no
retry. And `Out-File -Encoding unicode` writes UTF-16, which raises UnicodeDecodeError -- not an
OSError, so it escaped the guard and took the run down.

Reported by the train session on 2026-09-23 from a platform-portability hunt, measured on this
machine rather than argued. The suite could not have found it: `tests/test_coordinator.py` writes
the file with `write_text("c", encoding="utf-8")` from Python, which emits no BOM, so the
two-OS matrix was green on both.
"""

from __future__ import annotations

import codecs
from pathlib import Path

import pytest

from royalelearn.coordinator import _Control

CONTROL = "control"


def control(tmp_path: Path) -> _Control:
    return _Control(tmp_path, install_signal=False)


def write(control_obj: _Control, payload: bytes) -> None:
    control_obj.path.parent.mkdir(parents=True, exist_ok=True)
    control_obj.path.write_bytes(payload)


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("plain utf-8", b"c"),
        # `>` and `Set-Content` in PowerShell 5.1
        ("utf-8 with a BOM", codecs.BOM_UTF8 + b"c"),
        # `Out-File -Encoding unicode`
        ("utf-16-le with a BOM", codecs.BOM_UTF16_LE + "c".encode("utf-16-le")),
        ("utf-16-be with a BOM", codecs.BOM_UTF16_BE + "c".encode("utf-16-be")),
        ("a trailing newline, which every shell adds", codecs.BOM_UTF8 + b"c\r\n"),
        ("upper case, because a reader may type it", codecs.BOM_UTF8 + b"C"),
    ],
)
def test_a_letter_written_by_a_readers_shell_is_read(
    tmp_path: Path, name: str, payload: bytes
) -> None:
    """Every encoding a documented shell produces, not only the one Python writes."""
    obj = control(tmp_path)
    write(obj, payload)
    assert obj.poll() == "c", f"a control file written as {name} did not reach the run"


def test_a_file_that_cannot_be_decoded_does_not_take_the_run_down(tmp_path: Path) -> None:
    """The guard caught OSError, and a UnicodeDecodeError is a ValueError with no OSError in it.

    So a mis-encoded control file escaped poll(), through _collect and _iterate, into learn().
    A control file is a convenience; it must not be able to end a run.
    """
    obj = control(tmp_path)
    write(obj, b"\xff\xfe\x00\x00bad")
    assert obj.poll() == ""


def test_an_unrecognised_letter_says_so_rather_than_vanishing(tmp_path: Path) -> None:
    """The file is consumed either way; the difference is whether anybody learns why.

    This is the half that made the BOM invisible: the letter was taken, the file was removed,
    and nothing was printed, so a reader saw a control that did nothing and no reason for it.
    """
    said: list[str] = []
    obj = _Control(tmp_path, install_signal=False, printer=said.append)
    write(obj, b"z")
    assert obj.poll() == ""
    assert said and "z" in said[0], said


def test_the_file_is_consumed_so_a_letter_acts_once(tmp_path: Path) -> None:
    obj = control(tmp_path)
    write(obj, codecs.BOM_UTF8 + b"q")
    assert obj.poll() == "q"
    assert obj.poll() == ""
    assert not obj.path.exists()
