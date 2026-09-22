# RoyaleLearn

Training for Clash Royale bots: self-play rollouts over
[RoyaleGym](https://github.com/RoyaleGym/RoyaleGym) environments, a PPO learner for the
game's discrete card-and-tile action space, and a frozen-pool ladder that decides whether a
new policy is actually stronger than the one before it.

> **Status: the design is settled, the harness is not written.** The package imports and its
> 9 tests pass, and it carries six seed modules that do not run. There is no rollout worker,
> no learner wired to an environment, no ladder, no checkpoint format and no metrics sink.
> What this repo offers today is the design below, written against interfaces that do exist —
> a specification a contributor can implement against.

## What the harness will be

Five pieces. Each is an ABC with a default implementation and none of the defaults is
special, so a rating scheme, a sampler or a sink can be swapped without forking the loop.

1. **Rollout workers** over `royalegym`'s self-play vectorised env (`ClashSelfPlayVecEnv`),
   with the Rust-backed default observations and actions so Python stays off the per-tick
   path. Measured 2026-09: a Python observation builder capped throughput at roughly 520
   env-steps/s against the engine's ~25 000 ticks/s. The workers are Python first and move to
   Rust when they become the bottleneck, as the engine's tick loop already did.
2. **A PPO learner** in torch over `Discrete(2305)` — a masked categorical head, one policy
   for both seats — replacing the seed modules' Rocket League policies.
3. **A frozen-pool ladder**: periodic policy snapshots, Elo with confidence intervals against
   the pool, and the win rate that gates a snapshot into it. The pool bookkeeping already
   exists one layer down (`royalegym.selfplay.OpponentPool`: snapshots, uniform / latest /
   PFSP sampling, Elo, head-to-head records, save and load); this repo drives it, runs the
   evaluation matches and owns the gate (`ladder.py`).
4. **Checkpoints** holding the policy, the critic, the optimizer, the pool index, and the env
   configuration and seeds that produced them — enough to resume a run or reproduce a result
   rather than approximate it. The engine is deterministic and seedable and a recorded trace
   re-verifies bit for bit (`royalegym.replay`), so "reproduce" can mean exactly that.
5. **A metrics sink** fed from both sides: the simulator (ticks/s, env-steps/s, episode
   length, crown and tower statistics, illegal-action rate) and the learner (loss, KL,
   entropy, Elo against the pool), streamed to Weights & Biases.

**The metric.** A bot is judged on its ranking against the frozen pool, with confidence
intervals, and that ranking is the release gate. Agreement with the simulator is not the
metric — that is RoyaleSim's concern, measured against recordings of the real game and
tracked in that repo's calibration ledger. A bot that exploited a simulator artefact would
score well on parity and badly on the thing anyone cares about.

**Built once, completely.** There is deliberately no baseline learner, no throwaway trainer
and no "simple version first". The layers below are already finished to that standard, and a
learner that is 80% right produces a bot that loses for reasons nobody can attribute: a wrong
ladder makes a good policy look bad, a wrong checkpoint makes a good run unreproducible, a
rollout worker that quietly drops episodes shows up as a plateau rather than as an error.
Each costs more to diagnose later than to design now, so a partial harness is not a
milestone. `docs/design.md` carries the reasoning and the conventions the harness is held to.

## The surface it is built against

Everything below this repo runs today, so the shapes the learner has to handle are not
hypothetical:

```python
from royalegym import ClashParallelEnv, ClashSelfPlayVecEnv
from royalegym.rust_engine import RustEngine

venv = ClashSelfPlayVecEnv(num_games=4, env_fn=lambda: ClashParallelEnv(RustEngine()))
obs, info = venv.reset(seed=0)

venv.num_envs             # 8  -- both seats of 4 games in one batch
venv.single_action_space  # Discrete(2305)
{k: v.shape for k, v in obs.items()}
# {'action_mask': (8, 2305), 'spatial': (8, 21, 32, 18), 'vector': (8, 355)}
```

What that implies for the learner:

- **One head serves both players.** Observations and actions are in the acting player's own
  frame, so blue and red feed a single batch and a single policy.
- **Action 0 is no-op; the other 2304 are (hand slot, tile)** — 4 slots over an 18 x 32 tile
  grid. `action_mask` is per *pair*, not per factor: elixir is per card and the legal
  territory depends on the card, so the mask says "Giant here, not Fireball" exactly. The
  learner is expected to apply it to the logits, and illegal-action rate is a tracked metric.
- **One step is one decision**, 500 ms of game time by default (`decision_ms`); timing is
  expressed by choosing no-op on a step rather than by a separate sub-action.
- **Autoreset is same-step**: when a game ends the returned observation is already the next
  game's first, with the final observation and info in `infos["final_obs"]` /
  `infos["final_info"]`. The rollout worker has to split episodes on that, not on the
  observation.
- **Runs are seedable end to end**, which is what makes checkpoint-level reproduction
  meaningful.

A viewer can attach to any of it out of process (`ROYALEVISER=host:port`, handled by
`royalegym`), never through this repo.

## What is in the repo today

- `royalelearn/__init__.py` — the package, importable in the workspace venv without torch or
  `rlgym_ppo`. It names the seed classes (`royalelearn.SEED_CLASSES`) and resolves them
  lazily, so a missing dependency is a clear error rather than an import failure at startup:

  ```
  $ python -c "import royalelearn; royalelearn.PPOLearner"
  ImportError: royalelearn.PPOLearner lives in the vendored rlgym_ppo seed
  royalelearn/ppo_learner.py, which needs 'torch' (not installed).
  The seed is not wired to RoyaleGym; see README.md.
  ```

- `royalelearn/continuous_policy.py`, `discrete_policy.py`, `multi_discrete_policy.py`,
  `value_estimator.py`, `experience_buffer.py`, `ppo_learner.py` — six modules taken as a
  seed: copied verbatim and unmodified from
  [rlgym-ppo](https://github.com/AechPro/rlgym-ppo) (Copyright Matthew Allen, Apache License
  2.0; see `NOTICE` and `LICENSE-APACHE-2.0`). They import `torch` and `rlgym_ppo`, are bound
  to Rocket League's action layout (`MultiDiscreteFF` hardcodes its 8 bins) and do not run
  here. They are a reference for what a working PPO harness looks like, not a foundation:
  they are excluded from ruff (`pyproject.toml`) and will be replaced, not cleaned.
- `tests/test_package.py` — the import contract above, and that both layers below
  (`royalegym`, `royalesim`) import from the workspace venv.

Nothing is wired to an environment yet.

## Where this fits

The five repos are siblings in one workspace sharing one venv; four are public under the
GitHub organization [RoyaleGym](https://github.com/RoyaleGym).

| Repo | What it does | Package |
|---|---|---|
| [RoyaleSim](https://github.com/RoyaleGym/RoyaleSim) | deterministic integer-tick battle engine (Rust + PyO3): pathfinding, targeting, combat, spells, elixir, win conditions | `royalesim` |
| [RoyaleGym](https://github.com/RoyaleGym/RoyaleGym) | environment API over the engine: observations, actions and masks, rewards, state mutators, done conditions; Gymnasium, PettingZoo and self-play vec envs | `royalegym` |
| **RoyaleLearn** | this repo: the training harness | `royalelearn` |
| [RoyaleViser](https://github.com/RoyaleGym/RoyaleViser) | out-of-process viewer for traces, captures and a running env | `royaleviser` |
| RoyaleLive (private) | the client instrument that records ground-truth traces from the real game | scripts |

Dependency direction is strictly `RoyaleLearn -> RoyaleGym -> RoyaleSim`. This repo imports
`royalegym` (environments, the viser publisher) and nothing from RoyaleSim directly, and it
never touches calibration data.

The layering will look familiar if you know RLGym and RocketSim — environment API over
engine, learner on top, viewer out of process — which is good prior art for a project shape.

## Workspace setup

The repos are cloned as siblings into one folder and share one venv at that folder's root
(Python 3.12; Rust 1.80+ with cargo for the engine):

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
                                                                          # RoyaleLive (private): not needed for anything here
```

The order matters: `royalelearn` declares `royalegym` as a dependency and pip resolves it
from the venv, never from PyPI. `pip install -e "RoyaleLearn[torch]"` adds torch; the seed
modules additionally need `rlgym_ppo`, which is not a dependency and will not become one.

## Tests

```
cd RoyaleLearn
..\.venv\Scripts\python -m pytest -q      # 9 passed
..\.venv\Scripts\ruff check .             # All checks passed!  (the six seed files are excluded)
```

## Licence

MIT (`LICENSE`), except the six files listed in `NOTICE`, which are copied verbatim
from [rlgym-ppo](https://github.com/AechPro/rlgym-ppo) (Copyright Matthew Allen) and
remain under the Apache License 2.0 (`LICENSE-APACHE-2.0`).

## Community

Training runs, ladder results and harness design are discussed in the project's Discord:
[**https://discord.gg/4D2BS5JBHP**](https://discord.gg/4D2BS5JBHP)

Issues and pull requests on this repo are welcome too.
