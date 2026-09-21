# RoyaleLearn

The training harness of the Royale family, in RLGym-PPO's role: vectorised self-play rollouts
over `royalegym` environments, a PPO learner, a frozen-pool ladder with confidence intervals,
checkpoints and a metrics sink. Python today; the rollout workers move to Rust when they
become the bottleneck, as RoyaleSim's tick loop already is.

The family (five sibling repos under the GitHub organization
[RoyaleGym](https://github.com/RoyaleGym), one workspace, one venv):

| Repo | Analog | Role | Package |
|---|---|---|---|
| [RoyaleSim](https://github.com/RoyaleGym/RoyaleSim) | RocketSim | deterministic integer tick engine (Rust + PyO3) | `royalesim` |
| [RoyaleGym](https://github.com/RoyaleGym/RoyaleGym) | RLGym | environment API: obs, actions, rewards, state setters, terminals | `royalegym` |
| **RoyaleLearn** | RLGym-PPO | this repo: the learner | `royalelearn` |
| [RoyaleViser](https://github.com/RoyaleGym/RoyaleViser) | rlviser | out-of-process viewer for traces, captures and a running env | `royaleviser` |
| RoyaleLive (private) | - | the client instrument that records ground-truth traces from the real game | scripts |

Dependency direction is strictly `RoyaleLearn -> RoyaleGym -> RoyaleSim`. This repo imports
`royalegym` (environments, the `ViserPublisher`) and nothing from RoyaleSim directly.

## What exists today

- `royalelearn/__init__.py`: the package, importable without torch or rlgym_ppo. It names
  the seed classes (`royalelearn.SEED_CLASSES`) and resolves them lazily; without torch,
  `royalelearn.PPOLearner` raises an ImportError that says which package is missing.
- `royalelearn/continuous_policy.py`, `discrete_policy.py`, `multi_discrete_policy.py`,
  `value_estimator.py`, `experience_buffer.py`, `ppo_learner.py`: rlgym_ppo's `ppo/`
  subpackage, taken as a seed (copied verbatim and unmodified from
  [rlgym-ppo](https://github.com/AechPro/rlgym-ppo), Copyright Matthew Allen, licensed
  under the Apache License 2.0; see `NOTICE` and `LICENSE-APACHE-2.0`). They
  import `torch` and `rlgym_ppo`, are bound to Rocket League's action layout
  (`MultiDiscreteFF` hardcodes its 8 bins) and do not run here. They are excluded from ruff
  (`pyproject.toml`) and will be replaced, not cleaned.
- `tests/test_package.py`: the import contract above and that both layers below
  (`royalegym`, `royalesim`) import from the workspace venv.

Nothing is wired to an environment. There is no rollout worker, no ladder, no checkpoint
format, no metrics sink.

## The plan

1. Rollout workers over `royalegym`'s self-play vec env (`ClashSelfPlayVecEnv`), Rust-backed
   default obs and actions so Python stays off the per-tick path (measured 2026-09: Python
   obs building capped throughput at ~520 env-steps/s against ~25k engine ticks/s).
2. A PPO learner in torch for `royalegym`'s discrete card-and-tile action space, replacing
   the seed's Rocket League policies.
3. A frozen-pool ladder: periodic policy snapshots, ELO with confidence intervals against
   the pool, the win rate that gates a snapshot into the pool.
4. Checkpoints (policy, critic, optimizer, the pool index, the env config that produced them)
   and a metrics sink fed by the simulator (ticks/s, env-steps/s, episode length,
   crown/tower stats, illegal-action rate) and the learner (loss, KL, entropy, ELO),
   streamed to Weights & Biases.
5. Evaluation (`ladder.py`): the frozen-pool ladder with confidence intervals is the release
   gate; a bot is judged on its ranking against the pool, not on agreement with the simulator.

The plan is executed once, completely: the training design is treated as a permanent fixture
rather than a first cut. No baselines, no throwaway trainers, no "simple version first";
every piece above lands in its final shape with its design written down. `docs/design.md`
gives the reasoning, and the conventions the harness is held to.

## Workspace setup

The five repos are cloned as siblings into one folder and share one venv at that folder's
root (Python 3.12; Rust 1.80+ with cargo for the engine):

```
mkdir Royale && cd Royale
git clone https://github.com/RoyaleGym/RoyaleSim.git
git clone https://github.com/RoyaleGym/RoyaleGym.git
git clone https://github.com/RoyaleGym/RoyaleViser.git
git clone https://github.com/RoyaleGym/RoyaleLearn.git
python -m venv .venv
.venv\Scripts\python -m pip install maturin pytest hypothesis ruff
cd RoyaleSim && ..\.venv\Scripts\python tools\extract_arena.py && ..\.venv\Scripts\python tools\extract_cards.py && ..\.venv\Scripts\python tools\extract_globals.py && cd ..   # RoyaleSim/data/derived/ (gitignored; the crate compiles arena.json in)
cd RoyaleSim   && ..\.venv\Scripts\maturin develop --release && cd ..   # royalesim into the venv (~1 min, ~1.5 GB RAM)
.venv\Scripts\python -m pip install -e RoyaleGym                          # numpy, gymnasium, pettingzoo, msgspec
.venv\Scripts\python -m pip install -e RoyaleViser                        # pygame
.venv\Scripts\python -m pip install -e RoyaleLearn                        # this repo; add [torch] for the seed modules
                                                                          # RoyaleLive (private): scripts, no package; see its README
```

The order matters: `royalelearn` declares `royalegym` as a dependency and pip resolves it
from the venv, never from PyPI. `pip install -e "RoyaleLearn[torch]"` adds torch; the seed
modules additionally need `rlgym_ppo`, which is not a dependency and will not become one.

## Tests

```
cd RoyaleLearn
..\.venv\Scripts\python -m pytest -q      # 9 passed
..\.venv\Scripts\ruff check .             # clean; the six seed files are excluded
```

## Status

The package imports and its tests pass in the workspace venv; **the harness itself is not
started**. There is no rollout worker, no learner wired to an environment, no ladder, no
checkpoint format and no metrics sink — only the seed modules described above and the import
contract that keeps them optional.

When the harness is written it is built once and completely, as the permanent training
design: no baselines, no interim versions. `docs/design.md` says what that means and why.

## Licence

MIT (`LICENSE`), except the six files listed in `NOTICE`, which are copied verbatim
from [rlgym-ppo](https://github.com/AechPro/rlgym-ppo) (Copyright Matthew Allen) and
remain under the Apache License 2.0 (`LICENSE-APACHE-2.0`).

## Community

Training runs, ladder results and harness design are discussed in the project's Discord:
[**https://discord.gg/4D2BS5JBHP**](https://discord.gg/4D2BS5JBHP)

Issues and pull requests on this repo are welcome too.
