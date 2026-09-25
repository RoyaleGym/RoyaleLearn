"""Would a reader's shell actually run the commands on this repo's pages?

WHY THIS EXISTS
    On 2026-09-22 the published install block in these repos ran in NO shell a newcomer
    would use, and had not since the repos were split. Four readers hit it from four
    entry points. The shapes, each confirmed on a real machine rather than argued: ``&&``
    is a parse error in Windows PowerShell 5.1 before anything executes; a trailing ``#``
    comment is not a comment in ``cmd``, so ``python -m venv .venv  # Python 3.12``
    silently creates four directories and no environment; and a Windows backslash path
    inside a ```bash fence loses its backslashes.

    The second is the dangerous one, because it does not fail.

ROYALELEARN WAS THE LAST REPO WITHOUT IT, AND IT FOUND SOMETHING IMMEDIATELY
    Sim, gym and viser all carried this guard; this repo did not, which the documentation
    session reported on 2026-09-23. Run against these pages for the first time it refused
    one block: ``docs/running.md`` opened with two commands using Windows backslash paths
    under no named shell, on a page that does not say which shell it is for until line
    397. A macOS reader pasting it gets ``examplesconfigslaptop.json``. The page says so
    at the top now.

WHAT IT CHECKS, AND WHAT IT DELIBERATELY DOES NOT
    Only those shapes, on the pages a newcomer actually lands on. It does not read the
    commands and cannot tell you the instructions are correct: the real check is a person
    running the recipe verbatim in PowerShell from an empty folder, which is how the
    original defect was found. This is the guard that stops it coming back through
    ordinary editing.

    ``tests/_shell_fences.py`` is a vendored copy, taken from RoyaleGym's on 2026-09-23.
    Its own ``--selftest`` plants all four shapes plus three that must NOT fire, and that
    self-test is run here too: a guard whose self-test is never run is a guard nobody has
    seen work.

A GUARD WITH FALSE POSITIVES IS ONE SOMEBODY SWITCHES OFF
    These pages pass today, so if this test ever goes red on a page a person has just
    verified by hand, suspect the guard before the page.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from _shell_fences import check_text, selftest

REPO = Path(__file__).resolve().parents[1]
CHECKER = Path(__file__).resolve().parent / "_shell_fences.py"

#: The pages a reader lands on and pastes from. Not every .md in the repo: the guard is
#: about what a newcomer runs, and a reference page that shows a POSIX command to a
#: contributor is not the same promise. ``running.md`` is here because it is the page the
#: README sends somebody to the first time a run misbehaves, and the block it opens with
#: is the second thing they will paste. ``extensions.md`` tells a reader to install a
#: package of their own, and they paste that command too.
READER_PAGES = ("README.md", "docs/running.md", "docs/extensions.md")


@pytest.mark.parametrize("page", READER_PAGES)
def test_a_reader_could_paste_this_page_into_their_shell(page: str) -> None:
    path = REPO / page
    assert path.exists(), f"{page} is named here as a reader-facing page and does not exist"
    problems = check_text(path.read_text(encoding="utf-8"), page)
    assert not problems, (
        f"{page} has {len(problems)} block(s) a reader's shell would not run:\n  "
        + "\n  ".join(problems)
        + "\n\nThese are shapes, not opinions: && is a parse error in Windows PowerShell, "
        "a trailing # is not a comment in cmd, and bash eats backslashes. If the page is "
        "right and this is wrong, fix the guard rather than silencing it."
    )


def test_the_guards_own_self_test_passes() -> None:
    """It plants all four shapes and three that must not fire. Run it, do not assume it."""
    assert selftest() == 0, (
        "the shell-fence guard's own self-test does not behave. Its printed lines say "
        "which planted shape went unrefused, or which safe page was refused; run "
        "`python tests/_shell_fences.py --selftest` to see them."
    )


def test_the_guard_runs_standalone_on_a_fresh_clone() -> None:
    """No pytest, no package, no dependencies: ``python tests/_shell_fences.py --selftest``.

    That is the whole point of vendoring it. A contributor who has cloned the repo and
    not yet built anything can still check a page they are editing -- and in this repo
    "not yet built anything" is four clones and a data-generation stage away from a
    working suite, so a guard that needed the package would be unusable exactly when it
    is most wanted.
    """
    done = subprocess.run(
        [sys.executable, str(CHECKER), "--selftest"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
    )
    assert done.returncode == 0, (
        f"the vendored checker does not run standalone (exit {done.returncode}):\n"
        f"{(done.stdout + done.stderr)[-1500:]}"
    )


def test_the_vendored_checker_still_has_the_shape_of_a_copy() -> None:
    """It is a copy on purpose, and a copy that has drifted is worth knowing about.

    This cannot compare against the original, which lives outside this repo and is not
    present in a clone. What it can do is hold the copy to the shape that makes it a
    copy: no imports beyond the standard library, so it runs anywhere, and the self-test
    and entry point still present so a reader can run it.

    THE LIMIT IS NOT HYPOTHETICAL, and it is the reason this test claims only what it
    proves. On 2026-09-23 the three existing copies were found 49 lines behind the
    original: all three were identical to EACH OTHER and all three were stale, so no
    amount of comparing copies could have seen it. Three copies agreeing is not
    confirmation when the thing they agree with is each other. This is the fourth copy,
    and it is worth the next person knowing that its being byte-identical to the other
    three says nothing about whether any of them is current.
    """
    text = CHECKER.read_text(encoding="utf-8")
    assert "def selftest(" in text, "the vendored checker has lost its self-test"
    assert "__main__" in text, "the vendored checker is no longer runnable on its own"
    imports = {
        line.split()[1].split(".")[0]
        for line in text.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    }
    outside = imports - sys.stdlib_module_names - {"__future__"}
    assert not outside, (
        f"the vendored checker imports {sorted(outside)}, which is not in the standard "
        "library, so it no longer runs on a fresh clone with nothing installed"
    )
