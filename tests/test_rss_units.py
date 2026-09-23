"""What ``ru_maxrss`` is measured in, which is not the same on two POSIX systems.

``resource.getrusage().ru_maxrss`` is KILOBYTES on Linux and BYTES on macOS and the BSDs. The
harness decided between them by MAGNITUDE -- bytes if the number was above 1 GiB, kilobytes
otherwise -- which is right for a small process on Linux and wrong for a big one.

A 2 GB parent on Linux reports 2,097,152 KiB, which is above the threshold, so it was read as
bytes and published as **2 MB**: one thousandth of the truth, in the only measured RSS the harness
has, on the platform its CI runs. It read 0.0 on Windows until 2026-09-22 and this on Linux after.

Found by the train session on 2026-09-23, in a hunt for exactly this class. The rule is the
platform, which is knowable, rather than the size, which is a guess that fails at the size that
matters.
"""

from __future__ import annotations

import pytest

from royalelearn.coordinator import _rss_megabytes

GIGABYTE_ON_LINUX = 2 * 1024 * 1024  # KiB
GIGABYTE_ON_MACOS = 2 * 1000 * 1000 * 1000  # bytes


@pytest.mark.parametrize("platform", ["linux", "linux2"])
def test_a_two_gigabyte_process_on_linux_reads_as_two_thousand_megabytes(platform: str) -> None:
    """The case the magnitude rule got wrong, and it is the case that matters.

    Memory is what ends runs on this project's machine. An RSS reading that is a thousand times
    too small is worse than none, because it is the number somebody checks before deciding a run
    fits.
    """
    assert _rss_megabytes(GIGABYTE_ON_LINUX, platform) == pytest.approx(2147.48, abs=0.1)


def test_a_two_gigabyte_process_on_macos_reads_the_same() -> None:
    assert _rss_megabytes(GIGABYTE_ON_MACOS, "darwin") == pytest.approx(2000.0)


def test_a_small_process_is_right_on_both_and_always_was() -> None:
    """100 MB, which is the size at which the magnitude rule happened to work."""
    assert _rss_megabytes(100 * 1024, "linux") == pytest.approx(104.86, abs=0.1)
    assert _rss_megabytes(100 * 1000 * 1000, "darwin") == pytest.approx(100.0)


def test_an_unknown_posix_platform_is_read_as_linux() -> None:
    """Linux's kilobytes is the POSIX majority and the one CI runs.

    Named rather than left to a magnitude test: a wrong guess here is a factor of 1000, and a
    reader of an odd platform's number should be able to find this sentence.
    """
    assert _rss_megabytes(GIGABYTE_ON_LINUX, "freebsd14") == pytest.approx(2147.48, abs=0.1)


def test_the_magnitude_rule_is_gone() -> None:
    """The same number means two different things and only the platform separates them."""
    assert _rss_megabytes(GIGABYTE_ON_LINUX, "linux") != pytest.approx(
        _rss_megabytes(GIGABYTE_ON_LINUX, "darwin")
    )
