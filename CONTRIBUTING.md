# Contributing

Thanks for helping. Bug reports, fixes, examples and docs are all welcome.

## Set up

Clone RoyaleLearn beside RoyaleSim, RoyaleGym and RoyaleViser, and install them into one
virtual environment. The full steps are in the guide's
[Build from source](docs/guide.md#build-from-source-for-contributors).

## Before you open a pull request

    python -m pytest -q -p no:randomly
    python -m ruff check royalelearn tests

- Add a test that fails without your change and passes with it.
- A skipped test is not a passing one. If a test skips on your machine, say which and why.
- Keep public text plain: short sentences, no internal names.

## Reporting a bug

Open an issue with the template. Include the commands you ran, the full error, your OS and
Python version, and `python -c "import royalelearn; print(royalelearn.__version__)"`.

## Licence

By contributing you agree your work is released under this repository's [licence](LICENSE).
