# AGENTS.md

Notes for AI coding agents working in this repo. People should start at the [README](README.md).

## What this is

RoyaleLearn trains Clash Royale bots with PPO on RoyaleGym's environments. RoyaleSim is the
engine, RoyaleGym the environments, RoyaleViser the viewer, and RoyaleImitate an optional
imitation-learning add-on. Dependencies run one way: RoyaleLearn to RoyaleGym to RoyaleSim.

## Layout

| Path | What it holds |
|---|---|
| `royalelearn/learner.py` | `Learner`, the one-call trainer with named settings |
| `royalelearn/coordinator.py` | `LearningCoordinator`: one run, from preflight to checkpoints |
| `royalelearn/config.py` | `RunConfig` and its sections; `validate` |
| `royalelearn/api/` | The interfaces each part implements (buffer, policy, rollout, ladder, metrics) |
| `royalelearn/rollout/` | Collecting battles: the row codec, the env spec, preflight, workers |
| `royalelearn/learn/` | The network, the action distribution, decode, GAE and the PPO update |
| `royalelearn/ladder/` | Snapshots, evaluation, ratings and the gate into the opponent pool |
| `royalelearn/metrics/` | Metric rows, alarms and sinks (console, jsonl, W&B, viewer) |
| `royalelearn/extensions.py` | The surface add-ons may use; anything else may change |
| `royalelearn/cli.py` | The `royalelearn` command |
| `docs/` | The guide and the deeper pages |

## Build and test

Clone this repo beside RoyaleSim, RoyaleGym and RoyaleViser and install all four into one
virtual environment (the guide's "Build from source"). CI's own commands, from this folder:

    python -m pytest -q -p no:randomly
    python -m ruff check royalelearn tests

Tests marked `slow` or `engine` are deselected by default. CI runs them in jobs of their own.

## Rules that tests enforce

- A feature that is off changes nothing: rows, codec tables and architecture digests of a run
  that does not use it stay byte-identical, so saved checkpoints keep loading. Golden digests
  pin this.
- An observation key the codec does not store is refused at start-up, never dropped.
- Card ids are positions in the catalogue. Tests name cards, never bare ids.
- No global random state: every stream is derived from the run's seed.
- Shell blocks on reader pages run as pasted in the shell they name.

## Public text

This repo is public. Describe the library and its defaults. Never add training results,
recipes, tuned settings, run names, or paths and names from private repos or machines.
