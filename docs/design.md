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

The conventions the harness is held to are the family's, and will be familiar from RLGym:

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
  checkpoints, so a run can be replayed rather than approximated. Two runs of one identity
  agree row for row; a checkpoint restores the whole learner byte for byte in a fresh process;
  and a resume from an episode boundary continues the original's rows field for field, because
  an episode is addressed by its battle and its ordinal rather than by how many came before it.
  A checkpoint written mid-episode does not replay the episode in flight -- those transitions
  were already counted -- so the battles are the right ones and their phase is not.
  `docs/harness-spec.md` section 12.4 states exactly where the line falls, and two tests draw
  it from either side. Every checkpoint's learner is one a metrics row describes: the loop judges
  a batch before the update trains on it, an emergency save writes nothing while the learner is
  ahead of its rows, and a resume refuses a checkpoint whose state digest no row of its iteration
  carries (0839758).
- **Layering.** This repo imports `royalegym` and nothing from `royalesim` directly, and it
  never touches calibration data. The direction is `RoyaleLearn -> RoyaleGym -> RoyaleSim`.

## What the references contributed

rlgym-ppo's `ppo/` subpackage sat in `royalelearn/` while the harness was designed, as a
reference for what a working RLGym-PPO learner looks like, and was removed when the harness
landed. It was never a foundation: its modules are bound to Rocket League's action layout,
and the decisions worth keeping from them are recorded with their reasons in
`docs/harness-spec.md`, where they are named beside the ones that were not kept. `rlgym_ppo`
is not a dependency of this package and will not become one.

## What is open

The harness runs: `royalelearn train` collects rollouts, updates, rates, checkpoints, and
`royalelearn resume` continues a run from what it wrote. What is not settled is written here
rather than left to be rediscovered.

**Rewards reach the buffer one cycle late.** A worker publishes the result of stepping cycle `c`
as cycle `c + 1`, and `RectBuffer.record_round` files the reward and the terminated and truncated
flags at the cycle the round carries: beside the next action, not the one that earned them. GAE
reads row `t` as the result of action `t`. So each action is credited with the previous step's
reward and sees its own only through the lambda-weighted carry; an episode's last reward and end
flag land on the first row of the episode after it; and the reward of every iteration's last step
is dropped, because `record_round` skips the trailing round as the bootstrap cycle. Seen
directly on `MockEngine`: a probe reward that pays exactly `gamma - 1` on every step reads 0.0 in
row 0 of every iteration and `gamma - 1` in every other row. The fix is to file those scalars, and
the final observations, one cycle earlier, the trailing round included. It changes what every run
so far has optimised, so it comes before any training result is read.

**The forced-row decision, which is the one that changes what the next version is.** In the
first real iterations, 93% of collected decisions had exactly one legal action: the elixir bar
could afford nothing, so the mask left only the no-op. Those rows carry no policy gradient, and
the update that chews through them three times is 97.7% of the iteration's wall clock.
`docs/harness-spec.md` section 18 records the measurement and the three responses -- drop the
rows at collection, keep them for the critic but exclude them from the policy loss, or raise
`decision_ms` -- with what each does to the value function and to the wall clock. None is
obviously right and the evidence for choosing is two iterations.

**Two regions guarded by their ordering rather than by a flag.** A worker's actions region and
the parent's finals region are both written before the control word that announces them, which
is what makes the word the guard. That reasoning is sound and it is not checked. The two
failures this design has actually had were both a region read in its unwritten state -- an
observation cell nobody had filled, and a control word still idle -- so these two deserve the
same treatment as `valid`: a guard rather than an argument.

**The alarms have been validated against two iterations of one profile, which is not
validation.** Two were found wrong that way and repaired — both were means over a population
that is 93% structural zeros — and the rule that found them is in `harness-spec.md` section
13.3, along with what a validated threshold looks like. The systemic repair is not built: a
metric's row population belongs in its identity (`ppo/kl@choice` against `@all`) rather than in
its implementation, so that a threshold cannot be set against the wrong population by accident.

Two alarms and the halt path have narrower gaps. `vram_spilling` (29a05a4) has been seen only
staying silent: 42 rows from seven runs at minibatch 256 on the 4 GB card read 2995 to 3372 MB
available against 2349 needed, and nobody has watched it fire on a card. The suite trips it on a
synthetic row, and a test now fails for any alarm with no row to trip it (4ab3cf5).
`shaping_dominates` compared two structural zeros on every row recorded until 1065b78 and c38dc82
fixed its inputs (`harness-spec.md` section 10), so it has not been seen on a real run either. And a
halting iteration's alarms never reach `alarms.jsonl`, because `AlarmSet.evaluate` raises before it
returns them. The halting alarm is in the bundle, and the warnings beside it are only printed.

**Three workers spun through the update, because their clock ticked every 15.6 ms.** Sampled
during an iteration's update phase, each of three rollout workers burned 92% of a core waiting with
nothing to do, in a phase that is 97.7% of an iteration. The cause was the clock, not a leftover
semaphore count. The spin before a worker's sleep was timed with `time.monotonic()`, which on
Windows before Python 3.13 is `GetTickCount64()`, with a resolution of 15.625 ms. Measured on the
development machine, a spin meant to last 100 microseconds lasted 15.4 ms on average. The sleep
after it was `acquire(timeout=0.001)`, which returns in about 1.6 ms when the process's system
timer is at 1 ms and in about 16 ms when it is not. So an idle worker spun about 94% of its time
with a 1 ms timer and about half with the default one. The parent's wait had the same clock.

Since 98ff62f both sides time the spin with `perf_counter`. A wait that sees the command takes the
token that announced it, so a semaphore holds one token per command not yet answered. A worker
sleeps up to `SLEEP_S` (50 ms), and only on the shard whose command is due next; every command a
run sends arrives in that order. It glances at its other shard once per pass, so a command sent out
of turn, which only tests and tools do, is taken at most one sleep late. Three workers of two
shards on MockEngine, idle for 5 s after an iteration, went from 31-44% of a core each to 0.0-0.9%,
measured side by side on a machine at full load. `tests/test_worker_hygiene.py` holds an idle
worker under 10% of a core and covers the out-of-turn case and both token rules. What is still
open is the same check on the laptop profile's real update phase, on a quiet machine: per-worker
CPU time sampled 20 s apart should rise by under 2 s.

**Anything that makes a worker's first episode special is a thing a resumed worker must not
re-do**, because on a resume there is no first episode. That rule cost one real defect before it
was written down: the warm-up that staggers first-episode phases apart was still running after a
resume, moving every battle off the episode it was continuing. Two others were already guarded
and are worth knowing about, since both look like the same trap and are not — the codec table is
computed from a sample drawn from a named stream rather than from whatever the environment
happened to produce, and the frame-stack history strip is checkpointed with an explicit guard
against carrying an empty restored rectangle over it. A new piece of per-run warm-up is the thing
to check this rule against.

**Two behaviours that work but are not pinned by a test.** The ratio invariant under a
deliberately corrupted mask, and an episode replayed from its shard seed against
`royalegym.replay.verify_trace`. Both have been driven by hand and both passed, which is not the
same thing. The smoke configuration and the resume now have tests -- `tests/test_engine_contract.py`
runs an iteration end to end on the real engine, and `tests/test_resume.py` holds the resume
guarantee from both sides of the episode boundary. Writing that one is what found the unread
ordinal, the environment gap, and the crash in `verify-resume`.

**Two checkpoints written before 0839758 describe no metric row, and resume refuses both.** Until
then the update ran before the batch was judged, so a refused batch was trained first and refused
after, and the emergency save wrote those weights under the previous row's counters. Running
`checkpoint.check_described` over every checkpoint under `runs/` (34 on 2026-09-22) lets 32
through and refuses two: `integrator-rerun-fa93b56e506d5d0d/checkpoints/000000043776` holds 72
updates against its row's 54, and `train-diag0-hog26k-1ad6a480b7666090/checkpoints/000000038304`
holds 69 against 60. Neither can be used to check a real resume, which now needs a fresh run. The
fix has a price: a crash inside the update now loses the progress since the last periodic
checkpoint, where it used to save weights no row describes. An in-memory copy of the pre-update
learner would win that back, and it is not built.

**Runs before 23d971c trained a different objective.** Until then every shipped config named
RoyaleGym's `default_reward`, whose elixir-trade term pays a player who never commits a card. They
now name `royalelearn.rewards.default_potential_reward`, the composition `harness-spec.md` section
10 was written around. The reward is part of the environment spec's digest, which is in the run
identity and is the first input of the ladder's context digest, so the older runs have a different
identity and context and never pool with newer ones. That is the separator doing its job. One
question about the new reward is open there: its potential terms pay `gamma * Phi` on the
terminating step, which a potential-based shaping should not, and removing it changes the
objective again.
