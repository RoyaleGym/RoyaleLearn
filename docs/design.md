# Design of the training harness

What RoyaleLearn is meant to be, and the decisions that shape it. The README says what
exists today; this page says where it is going and why.

## The harness is designed once, completely

The training design is treated as a permanent fixture, not a first cut: rollout workers,
learner, ladder, checkpoints and metrics are designed as the harness the project keeps. This
repo deliberately has **no baseline learner, no throwaway trainer and no "simple version
first"**.

The reason is specific to this project rather than a general principle. The layers below are
already finished to that standard — an integer-only deterministic engine, an environment API
whose defaults are held to two independent engine implementations — and a throwaway trainer
on top of them would not tell us anything we could act on. A learner that is 80% right
produces a bot that loses for reasons nobody can attribute: a wrong ladder makes a good
policy look bad, a wrong checkpoint makes a good run unreproducible, and a rollout worker
that quietly drops episodes shows up as a plateau rather than as an error. Each of those
costs more to diagnose later than it costs to design now.

The practical consequence: a partial harness is not a milestone. Design decisions are
written down with their reasons, in this folder, before the code lands.

## What the harness consists of

1. **Rollout workers** over `royalegym`'s self-play vectorised env (`ClashSelfPlayVecEnv`),
   with Rust-backed default observations and actions so Python stays off the per-tick path.
2. **A PPO learner** in torch for `royalegym`'s discrete card-and-tile action space,
   replacing the seed's Rocket League policies.
3. **A frozen-pool ladder**: periodic policy snapshots, ELO with confidence intervals
   against the pool, and the win rate that gates a snapshot into the pool.
4. **Checkpoints** holding the policy, the critic, the optimizer, the pool index and the env
   configuration that produced them — enough to resume a run or reproduce a result.
5. **A metrics sink** fed from both sides: the simulator (ticks/s, env-steps/s, episode
   length, crown and tower statistics, illegal-action rate) and the learner (loss, KL,
   entropy, ELO against the pool).

## The metric

A bot is judged on its ranking against the frozen pool, with confidence intervals, and that
is the release gate. Agreement with the simulator is not the metric — that is RoyaleSim's
concern, measured against recordings of the real game, and it is tracked in that repo's
calibration ledger. Keeping the two apart matters: a bot that exploits a simulator artefact
would score well on parity and badly on the thing anyone cares about.

## Conventions

These are RLGym's conventions, adopted across the family:

- **Every user-facing behaviour is an ABC with swappable implementations.** In this repo that
  means the rollout worker, the learner, the ladder's matchmaking and rating, the checkpoint
  store and the metrics sink. Defaults are provided and none of them is special.
- **Python is never on the per-tick path in the common case.** Measured: a Python
  observation builder capped throughput at about 520 env-steps/s against the engine's
  ~25 000 ticks/s. Rollout workers move to Rust when they become the bottleneck.
- **Rendering, logging and monitoring are out of process or off the hot path.** A viewer
  attaches through `royalegym.viser.ViserPublisher` (`ROYALEVISER=host:port`), never through
  this repo.
- **Determinism is preserved end to end.** The engine is deterministic and seedable, and a
  trace recorded from an env re-verifies bit for bit (`royalegym.replay`). Seeds go into
  checkpoints, so a run can be replayed rather than approximated.
- **Layering.** This repo imports `royalegym` and nothing from `royalesim` directly, and it
  never touches calibration data. The direction is `RoyaleLearn -> RoyaleGym -> RoyaleSim`.

## The rlgym-ppo seed modules

Six modules in `royalelearn/` are rlgym-ppo's `ppo/` subpackage, copied verbatim and
unmodified (Copyright Matthew Allen, Apache License 2.0; see `NOTICE` and
`LICENSE-APACHE-2.0`). They import `torch` and `rlgym_ppo`, are bound to Rocket League's
action layout — `MultiDiscreteFF` hardcodes its 8 bins — and do not run here.

They are a reference for what a working RLGym-PPO learner looks like, not a foundation. They
will be **replaced** by the harness above, not cleaned up or extended, and `rlgym_ppo` is not
a dependency of this package and will not become one. Their attribution headers and their
listing in `NOTICE` stay with them for as long as they are in the tree; they are excluded
from ruff in `pyproject.toml` so that lint noise never tempts anyone to edit them.
