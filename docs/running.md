# running.md: running a job, and what to do when something looks wrong

This page is for the person sitting in front of a run. It covers starting one, reading what
scrolls past, the 28 alarms and what to do about each, the two ways a shared machine eats an
afternoon, and the one memory setting that is worth understanding before you touch it.

Every command block on this page is written for Windows PowerShell, the shell that opens by
default on Windows 10 and 11, which is why the paths use backslashes. On macOS or Linux the same
commands work with forward slashes and `.venv/bin/python` in place of `.venv\Scripts\python`.

It is not the specification. [harness-spec.md](harness-spec.md) is the dense version.
[metrics.md](metrics.md) describes every number in a row. [checkpoints.md](checkpoints.md)
covers saving and resuming. [throughput.md](throughput.md) covers how fast it goes. Every claim
here was checked against the code on 2026-09-22 and names the file it came from. Where the code
and the spec disagree, the code wins.

**What has not been shown.** A run completes its loop. Nobody has trained a good bot with this
harness yet. So this page tells you what each number *would* mean, and you should not read any of
them as a result.

A few words used throughout:

- **Iteration.** One lap: collect a pile of battles, then learn from them. One row of numbers
  per lap.
- **Timestep.** One decision by one bot in one battle.
- **The update.** The part of an iteration where the network changes. The long quiet part.
- **Alarm.** A yes-or-no question asked about one row of numbers. 28 of them, in
  `royalelearn/metrics/alarms.py`.
- **Patience.** How many iterations in a row the answer must be yes before the alarm speaks.

---

## 1. Starting a run, and what the first minute looks like

**What you do.** Two commands, in this order:

```
python -m royalelearn doctor  --config examples\configs\laptop.json
python -m royalelearn train   --config examples\configs\laptop.json
```

`doctor` runs the start-up gates on their own and exits. It takes seconds to a minute and
catches most first-run failures without spawning a worker or allocating a buffer
(`royalelearn/cli.py`, `_doctor`). Run it first. It is free.

`train` takes `--config` (required), and optionally `--run-name`, `--until-timesteps`,
`--inline` (one process, no worker farm, useful for debugging), `--device cuda|cpu` and
`--allow-dirty` (below) (`cli.py`, `build_parser`). Everything a run writes lands in
`runs/<run_name>-<run_id>/` (`coordinator.py`, `run_directory`).

**What happens before any battle is played.** `LearningCoordinator.__enter__`
(`coordinator.py`, around line 855) runs a fixed order: determinism tier, seed torch, resolve the
device, preflight, build the network, VRAM gate, compute the run identity, create the run
directory, then the buffer, the worker farm, the ladder and the metrics sinks. Nothing is
collected until all of that has passed.

**What refuses a run before it starts.** Each of these raises and the process exits with code
2, because every one of them is a `RoyaleLearnError` and `cli.main` turns those into one line
rather than a traceback (`errors.py`, `cli.py`).

| Refusal | Where | What it means |
| --- | --- | --- |
| Inconsistent config | `config.validate` | A number contradicts another one, a component in the `env` block is given a setting it does not take, a shaping weight is not a finite number of zero or more, or the `imitation` block is malformed. The message names every problem at once, not just the first. `ppo.batch_size` not being a multiple of `ppo.minibatch_size` is the common one. |
| Stale engine build | `preflight._construct` | One environment is constructed first. A calibration or build mismatch dies here rather than at cycle 0, and the message names the fix: `maturin develop --release, in the RoyaleSim checkout` (`preflight.REBUILD_COMMAND`). |
| Mask disagreement | `preflight._mask_disagreement_gate` | For both teams, exhaustively, the action mask and the engine are asked about every action. A bot trained against a wrong mask is worthless, so this refuses rather than warns. Off with `doctor.run_mask_disagreement_gate = false`. |
| No legal no-op | `preflight._noop_gate` | Checked on 1000 sampled states and on a finished battle. A state with no legal action is a distribution over nothing, and what that produces is a NaN in the first backward pass. |
| Action layout mismatch | `preflight._action_layout_gate` | The head writes a logit per (hand slot, tile); the environment reads an integer. A transposition between them is a bot that plays a different card in a different place from the one it learned. Exhaustive over every non-no-op action. |
| Memory projection over budget | `preflight.run_preflight` | The run projects more resident memory than `doctor.ram_budget_mb`, 6,500 MB by default (`config.DoctorConfig`). Lower `rollout.workers`, `rollout.games_per_worker` or `ppo.timesteps_per_iteration`. |
| One minibatch will not fit | `coordinator._vram_gate` | Section 5. This is the one that matters most on a small card. |
| A worker did not come up | `rollout/farm.py`, `_await_report` | The child's traceback is printed after `rollout worker N did not come up:`. |
| Identity drift, on resume | `coordinator._load` | The checkpoint was made by a different run. It names every differing field. `--allow-identity-drift` overrides it. |
| A run is already in that folder | `coordinator._refuse_an_occupied_run_dir` | The same config on the same code always gets the same `runs/<run_name>-<run_id>/`. A fresh start refuses that folder once it holds metric rows or checkpoints, before writing anything. `resume` the run that is there, or give the new one another `--run-name`. There is no flag to overwrite it. |
| Uncommitted code | `cli.refuse_dirty_sources` | `train` refuses when the royalelearn, royalegym or royaleviser package, the engine's data, the config file, or a package your own components come from has uncommitted or untracked files. `resume` and `verify-resume` check the same, apart from the config file. `--allow-dirty` runs anyway. |
| A worker on another engine build | `rollout/farm.py`, `check_worker_binaries` | Every worker reports the compiled engine file it loaded, and it must be the one the run's identity names. At start this means the engine changed while the run was starting. A worker restarted after a rebuild is refused the same way, so do not rebuild RoyaleSim under a running job. |
| An imitation file is not the one the config names | `imitation/init.py`, `verify_imitation_files` | A run with an `imitation` block names each folder it reads by path and digest. At every start, fresh or resumed, each folder is hashed again, and one whose content has changed is refused before preflight runs. `royalelearn artifact-digest <folder>` prints a folder's digest. |

One thing is a warning and not a refusal: if the projection exceeds what is free on the machine
*right now*, preflight says so and continues (`preflight.run_preflight`, around line 204). The
reasoning is in the code: something else on the machine may well stop before the run needs the
memory, and refusing over a neighbour who is busy for a minute is the more expensive mistake.

**Why refusing early is a kindness.** A useful run is many hours. Every gate above catches a
fault that would otherwise show up as a run that appears to work and produces nothing, which you
would discover much later and would then have to bisect. The alternative to a refusal at second
five is not success; it is the same failure, expensive and hard to attribute.

**What the first minute actually looks like.** Preflight prints a labelled block: engine and card
count, decision timing, the codec table, hand field offsets, the three gate results, the engine
build digests, the memory ledger, the geometry, the credit horizon, the viewer line, and finally
`run id`. Then the run directory is created and collection starts.

**The first round is the slow one.** Workers are spawning and nothing is warm. Measured on the
maintainer's laptop, 2026-09-22 (README): first rounds of 46, 79 and 179 seconds, a spread of
3.9x, against later rounds of 12.6, 13 and 19 seconds, a spread of 1.5x. Wait for the second
round before you believe any timing. A complete iteration on that machine was measured at 518
and 544 seconds, so about nine minutes idle.

---

## 2. What the console shows while it runs, and where the numbers are written

**What you get:** three kinds of line, so that a long silence always has a label on it.

1. `collected     228 cycles, 32832 timesteps in 46.0s (951 env steps/s); updating`, from
   `coordinator._say_collected`. It exists because the update that follows is minutes of
   silence, and a silent run is indistinguishable from a stopped one. The code notes that two
   runs on this project were killed by the person watching for exactly that reason.
2. `updating      epoch 1/3, 32832 samples, 0s in`, one line per epoch, from `learn/ppo.py`
   around line 375. Same purpose, and it lets you size the phase from outside.
3. The per-iteration block, printed by `ConsoleSink` in `metrics/sinks.py`. About 145 numbers,
   grouped by the part of the key before the slash, in the schema's own order.

Alarm lines appear as `alarm [warn] <name>: <key>=<value> -- <meaning>` (`metrics/alarms.py`,
`AlarmSet.evaluate`). Checkpoints print `checkpoint    <path>`, bundles print
`bundle        <path>`.

**Where the numbers are written.** Everything goes to `runs/<run_name>-<run_id>/`:
`metrics.jsonl` (one JSON object per iteration, flushed every row, safe to tail),
`episodes.jsonl` (one record per finished battle, gzipped in place after 200 iterations),
`alarms.jsonl` (only written when an alarm fires, so an untroubled run has no such file),
plus `config.json`, `identity.json`, `checkpoints/`, `snapshots/`, `ladder/`, `artifacts/` and
`bundles/`. The names are in `metrics/sinks.py` and `metrics/bundle.py`.

**Read [metrics.md](metrics.md) for what each number means**, which handful are worth watching,
and which rows are empty or unreliable today. It is not repeated here.

**Controlling a running job.** Write a single letter into `runs/<...>/control`
(`coordinator._Control`, around line 713):

- `c` write a checkpoint now,
- `q` write a checkpoint and stop,
- `p` pause until you write another letter.

The letter is read once per round and then the file is deleted. Ctrl-C is turned into `q`, so
an interrupt is a checkpoint rather than a lost afternoon.

---

## 3. The alarms, one group at a time

There are exactly 28. You can print the table without starting a run:

```
../.venv/Scripts/python -c "
from royalelearn.metrics.alarms import default_alarms
from royalelearn.config import AlarmConfig
for a in default_alarms(AlarmConfig()):
    print(f'{a.severity:5} patience={a.patience:<3} {a.name}')
"
```

Three rules govern all of them, from `metrics/alarms.py`:

- **`warn`** (21 alarms) prints a line and appends to `alarms.jsonl`. The run continues.
- **`halt`** (7 alarms) does that, then writes a checkpoint and a diagnostic bundle into
  `bundles/<iteration>/`, then stops the run with exit code 2.
- **A missing key never fires.** An iteration in which no battle finished carries no `env/`
  numbers at all, and an alarm that read that absence as a zero would stop a healthy run.

Thresholds live in the `alarms` block of your config (`config.AlarmConfig`). Any alarm can be
made louder, quieter or turned off by name through `patience_overrides`, `severity_overrides` and
`disabled`. Changing them cannot change any number the run produces, which is why they are left
out of the run identity. Patience restarts from zero on a resume, which is the honest reading:
the iterations that would have tripped an alarm were not observed by this process.

### 3.1 Something is broken (six alarms)

These say a piece of the harness is not doing what it claims. None of them is a tuning problem
and none is fixed by changing a hyperparameter. Four of the six stop the run.

| Alarm | Reads | Fires when | Severity, patience |
| --- | --- | --- | --- |
| `illegal_actions` | `env/illegal_action_rate` | above 0.0 | halt, 1 |
| `ratio_invariant` | `ppo/ratio_max_abs_dev` | above `alarms.ratio_invariant_multiple` (5.0) times `ppo.ratio_atol` | halt, 1 |
| `nonfinite` | `health/nan_guard_trips` | above 0 | halt, 1 |
| `buffer_overflow` | `health/buffer_fill_frac` | above `max(1.0, alarms.buffer_fill_frac)` | halt, 1 |
| `elixir_count_inexact` | `env/elixir_count_exact_frac` | below 0.99 | warn, 3 |
| `seat_bias` | `env/win_rate_by_seat_ci95_lo/_hi` | the interval clears 0.45 or 0.55 | warn, 3 |

**`illegal_actions`.** The engine refused a command the mask allowed. Under a correct mask this
is exactly zero, so any positive value is a bug in the mask, in the action encoding, or in an
unmasked policy. Do not retune anything. Read the bundle, then re-run
`python -m royalelearn doctor`, which exercises the mask and layout gates.

**`ratio_invariant`.** At the first minibatch of the first epoch the stored log-probabilities
must be the ones the current weights produce on the stored bytes, so the ratio must be 1 to
tolerance. It is the cheapest detector there is for a mask, codec or weight-version drift.
Shipped tolerances are 1e-4 for float32 and 2e-2 for bfloat16 (`config.PPOConfig.ratio_atol`), so
the alarm fires above 5e-4 or 0.1 depending on your run's precision. The metric is only computed
on some iterations (`ppo.check_ratio_invariant_every`, 50, and `ppo.debug_assert_iterations`, 10
from a start), so the alarm is only asked on those. Do nothing about learning rates: this is a
code or version mismatch, and the bundle carries the outliers.

**`nonfinite`.** A loss, a gradient or a logit was not finite. Usually an entirely masked
distribution, a bad reward, or a learning rate high enough to blow up in one step. Check the
reward function first if you wrote your own.

**`buffer_overflow`.** A rectangle cannot be more than full: every cell is written exactly once
by the worker that owns it. Above one, a cycle was published twice or the geometry moved under
the buffer, and the rows of that iteration are not the rows they say they are. The configured
threshold (0.98) is a floor on the bound rather than the bound itself, so lowering it can only
make the alarm stricter. A rollout bug, not a setting.

**`elixir_count_inexact`.** The observation's opponent-elixir slot is documented as exact, and on
some episodes it was an estimate. A value near zero and a value just under one are different
stories, so check the engine build first: the digests are in the run's `identity.json` and in the
`engine build` line preflight prints. It warns because a slightly imperfect count is worth
knowing about and not worth losing hours over.

**`seat_bias`.** The 95% interval around win rate by seat has cleared the 0.45 to 0.55 band.
Causes, in order of likelihood: an unseeded reset, a reward asymmetry, an observation mirror bug.
A few points can legitimately be the engine's own seat asymmetry, which is why this warns.

### 3.2 The machine is the problem (four alarms)

| Alarm | Reads | Fires when | Severity, patience |
| --- | --- | --- | --- |
| `worker_failures` | `health/worker_restarts` | the counter rose this iteration | warn, 1 |
| `worker_failures_persistent` | `health/worker_restarts` | it rose on three iterations in a row | halt, 3 |
| `vram_spilling` | `time/update`, `health/vram_driver_free_mb` | update at least 2x this run's best while the driver reports under 128 MB free | warn, 3 |
| `capacity_ratio` | `throughput/rollout_capacity_ratio` | below `alarms.capacity_ratio` (1.5) | warn, 3 |

**`worker_failures`.** One rollout worker was restarted. The counter is cumulative over the run,
so this alarm watches that it *moved*, not its level (`RoseAlarm` in `alarms.py`). One restart is
survivable: the dead worker's slots arrive marked invalid, the round continues without them, and
the iteration ends with fewer transitions. On a laptop the usual cause is memory pressure. Check
what else is running.

**`worker_failures_persistent`.** Workers have failed on three consecutive iterations. No longer
transient, so it halts. Separately, `rollout.max_restarts_per_worker` (3) raises if one worker
keeps dying. Read the failure message printed at the time: it names the worker, the cycle, the
kind and the shard, and says what share of the iteration's rows went with it.

**`vram_spilling`.** Section 5 is about this one. In short: the update has gone at least twice as
slow as this run's own best while the driver says the card is full, three iterations running.

**`capacity_ratio`.** Rollout capacity divided by update capacity. The design invariant is that
the harness is never the bottleneck, at least 2 on every shipped profile. Below 1.5 says the data
supply is becoming the limit rather than the learner. Add workers or battles per worker if you
have the memory and the cores. See [throughput.md](throughput.md).

### 3.3 The update is going wrong (four alarms)

| Alarm | Reads | Fires when | Severity, patience |
| --- | --- | --- | --- |
| `clip_pinned` | `ppo/clip_fraction` | above `alarms.clip_fraction` (0.5) | halt, 3 |
| `kl_high` | `ppo/kl` | above `alarms.kl_high` (0.05) | warn, 3 |
| `kl_dead` | `ppo/kl` | below `alarms.kl_dead` (1e-5) | warn, 10 |
| `ev_negative` | `ppo/explained_variance` | below `alarms.explained_variance` (0.0) | warn, 50 |

**`clip_pinned`.** More than half the samples hit the clip, three iterations running. Pinned near
1.0 is the signature of a rollout-versus-update mask disagreement, or of a learning rate far too
high. Check the mask first: that explanation is a bug, the other is a setting. It halts because
training otherwise runs on happily while producing nothing usable.

**`kl_high`.** The policy moved further in one update than the band allows. The learning-rate
backoff (`ppo.lr_backoff`, threshold 0.02) should already be acting, so read this as a prompt to
check `ppo/lr_backoff_events`. If it persists, raise `ppo.batch_size`.

**`kl_dead`.** Ten iterations with essentially no movement at all. Dead entropy, a learning rate
too low, or a frozen head. Check `ppo/entropy` and the learning rate first.

**`ev_negative`.** The critic explains none of the return. Patience is 50 because a critic being
bad early is completely normal, and the alarm is about a plateau rather than a start. This is
the most likely explanation of a run that has stopped changing. Nothing on this project has
established a healthy value here yet, so treat a firing as a prompt to investigate rather than
as a verdict.

### 3.4 The bot's behaviour is going wrong (eight alarms)

These are about what the policy is actually doing in battles. Seven warn; one stops the run.

| Alarm | Reads | Fires when | Severity, patience |
| --- | --- | --- | --- |
| `noop_collapse` | `policy/cards_per_match` | below `alarms.cards_per_match_warn` (8.0) | warn, 5 |
| `noop_collapse_severe` | `policy/cards_per_match` | below `alarms.cards_per_match_halt` (3.0) | halt, 5 |
| `noop_entropy_floor` | `ppo/noop_entropy` | below `alarms.noop_entropy_floor` (0.02) | warn, 5 |
| `tile_spam` | `policy/tile_top1_share` | above `alarms.tile_top1_share` (0.25) | warn, 5 |
| `artefact_exploit` | `policy/card_tile_top10_share` | above `alarms.card_tile_top10_share` (0.5) | warn, 5 |
| `draw_equilibrium` | `env/draw_rate`, `env/episode_steps_at_cap_frac` | draws above 0.5 **and** episodes ending at the cap above 0.8 | warn, 5 |
| `reward_clipped` | `ppo/reward_clip_frac` | any reward was clipped this iteration | warn, 1 |
| `shaping_dominates` | `env/reward_shaping_abs`, `env/reward_terminal_abs` | the shaping terms' per-episode sums exceed the terminal term's: a term has stopped telescoping | warn, 5 |

**`noop_collapse` and `noop_collapse_severe`.** Cards played per finished battle, not the no-op
rate, is the collapse metric. A healthy policy is about 94% no-op simply because most ticks you
cannot afford anything, so the no-op rate is a bad instrument; about 22 cards a match is the
healthy figure the schema records. Under 8 warns. Under 3 means the policy has stopped playing
cards at all, and that halts. The usual lever is `ppo.ent_coef_noop`, a warm-start bonus that
anneals to zero by design, and the alarms deliberately outlive the schedule.

**`noop_entropy_floor`.** The leading indicator of the above, and it keeps watching after
`ent_coef_noop` has annealed away, which is when it matters most. It measures the binary
entropy of play-against-wait over only the rows where the mask offered more than the no-op,
because a decision the elixir bar cannot afford has zero entropy by construction. Seeing this
before `noop_collapse` is the normal order.

**`tile_spam`.** A quarter of every card played is going on one tile. That is either a
degenerate policy or an engine quirk being exploited.

**`artefact_exploit`.** Half of all plays are in the ten most-played (card, tile) pairs. A real
meta is not that concentrated, so this is probably an artefact of the simulator rather than a
strategy. This is the only warn-severity alarm that writes a diagnostic bundle anyway
(`dump_bundle=True` in `alarms.py`), because the bundle carries a trace you can watch.

**`draw_equilibrium`.** The turtle equilibrium: nobody attacks, and the step limit ends every
game. It needs both conditions, which is what distinguishes it from a run that happens to be
drawing. Usually a reward shaping problem: there is no gradient pushing anyone to commit.

**`reward_clipped`.** A reward hit `advantage.reward_clip` this iteration. That matters more
than the size of the number suggests. Potential shaping leaves the best policy unchanged only
while every step's reward reaches the return intact, and a clipped step is one where it does not.
If the step that was cut was the one carrying the win, a win is worth less than a win. Raise
`advantage.reward_clip`, or lower the shaping weights you raised.

**`shaping_dominates`.** One of your shaping terms has stopped telescoping. Under a potential
reward each term's per-episode sum is small by construction, whatever its weight: the steps cancel.
A sum that climbs towards the terminal term means something in the reward is not a difference of a
potential, for example a term that pays on the final step or reads something other than the state.
Changing a weight will not fix it; find the term. To see how loud each term actually is, read
`env/reward_terms_step_abs/<term>` beside `env/reward_terms_step_abs/terminal`. Note from [metrics.md](metrics.md) that `env/reward_terminal_abs`
read 0.0 in every row on disk before a fix on 2026-09-22, so on older runs this alarm could not
fire for a reason that had nothing to do with the policy.

### 3.5 The ladder is not telling you anything (two alarms)

| Alarm | Reads | Fires when | Severity, patience |
| --- | --- | --- | --- |
| `transitivity` | `ladder/transitivity_residual` | above `alarms.transitivity_residual` (0.10) | warn, 3 |
| `gate_starved` | `ladder/consecutive_gate_failures` | at or above `alarms.gate_failures` (5) | warn, 1 |

**`transitivity`.** One number per player fails to explain the result log. In plain words the
rating is lying: A beats B, B beats C, C beats A, and no single scalar can describe that. Any
rating you read while this is firing is not measuring what you think it is.

**`gate_starved`.** Five gates failed in a row, so no new snapshot has joined the pool. This is
the plateau signal stated as an event rather than as a curve. It is a warning because a plateau
is information, not a fault. See [ladder.md](ladder.md).

### 3.6 Learning from demonstrations (four alarms)

These fire only on a run with a `warm_start` or `imitation` section: a run that started from a
cloned policy, or that is held near a reference policy while it learns. Section 19 of
[harness-spec.md](harness-spec.md) describes the sections. All four warn and none stops the run,
because a run stopped early would drop out of any comparison it is part of.

| Alarm | Reads | Fires when | Severity, patience |
| --- | --- | --- | --- |
| `imitation_ref_kl_high` | `imitation/<name>/kl` | a regulariser's KL above `imitation.alarms.ref_kl_warn` (1.0 nats) | warn, 1 |
| `imitation_lambda_saturated` | `imitation/<name>/lambda_at_max` | the anchor's coefficient sat at its ceiling | warn, `imitation.alarms.lambda_saturated_patience` (10) |
| `actor_handoff` | `ppo/kl`, `ppo/clip_fraction`, `ppo/iterations_since_unfreeze` | in the first `warm_start.alarms.handoff_window` (20) iterations after a frozen actor starts to move, KL above 0.05 or clip fraction above 0.3 | warn, 1 |
| `critic_unready` | `ppo/ev_at_unfreeze` | the critic's explained variance when the actor was unfrozen, below 0.3 | warn, 1 |

**`imitation_ref_kl_high`.** The policy has moved far from the reference it is anchored to. That
can be the anchor letting go on schedule, or the reward pulling the policy somewhere the reference
never goes. Read `imitation/<name>/kl_noop`, `kl_card` and `kl_tile` to see which part moved.

**`imitation_lambda_saturated`.** The coefficient has been at `coef.max` for ten iterations and
the KL is still above its budget. The anchor is pulling as hard as it is allowed to and losing.
That is a statement about the reward, not about the anchor.

**`actor_handoff`.** The first iterations after the freeze moved the policy fast. The
learning-rate backoff acts on its own; this says why it acted.

**`critic_unready`.** The critic was trained on the frozen policy's battles and still
explained little of the return when the actor was let go, so the first policy updates run on a
poor baseline. A longer freeze is the usual answer.

`kl_dead` cannot fire on a frozen iteration. Its key, `ppo/kl`, is left out of a frozen row
rather than written as 0.0, because nothing was measured.

---

## 4. Sharing one machine

This is where real time gets lost, and neither hazard announces itself clearly.

### 4.1 The viewer's ports are fixed

RoyaleViser uses two fixed UDP ports on 127.0.0.1: **9870** for battle frames
(`STREAM_PORT` in `RoyaleViser/royaleviser/sources.py`) and **9871** for the learner's status
panel (`STREAM_LEARNING_PORT`, defined as the frame port plus one). The learner side pins the
same constants in `RoyaleLearn/royalelearn/metrics/viser_sink.py` (`VISER_PORT = 9870`,
`VISER_LEARNING_PORT_OFFSET = 1`). Nothing sets `SO_REUSEADDR` anywhere, so a second bind to
either port fails.

Two consequences, and they behave differently.

**Frames, port 9870.** Only bound when `ROYALEVISER` is set in the environment, and only by one
vec env in the run (`coordinator._build_source`, `rollout/inline.py` around line 1439: worker 0,
shard 0, battle 0). A second run that also sets `ROYALEVISER` cannot bind it, the environment
raises at construction (`rollout/envspec.py`, `build_vec`), and the farm reports
`rollout worker 0 did not come up:` with the traceback. Loud, and it refuses to start. That is
the good case.

**Status, port 9871.** This one is bound by **every** run, whether or not you asked for a
viewer, because the `viser` sink is in the default sink list (`config.MetricsConfig`) and
`ViserSink.open` binds unconditionally through `learning_endpoint()`, which falls back to
127.0.0.1:9871 when `ROYALEVISER` is unset. The second run to start prints one line:

```
the learning status port 127.0.0.1:9871 is not available (...); this run publishes no
learning panel until it frees, which it retries every 10s. Another training run is the
usual holder, and a run that was killed rather than asked to stop can still be holding it.
```

**How you spot it.** That line, in the first second of the run, and then nothing. It prints
once, not every retry (`viser_sink._bind`, the `first` flag). If the port later frees, the run
prints `the learning status port ... is free; panel live` and the panel starts working. So
scroll back to the start of the log before concluding the panel is broken.

**The silent case.** If run A holds 9871 and run B holds 9870, the viewer draws run B's battle
under run A's learning panel. RoyaleViser has machinery for exactly this: `render.py` around
line 1058 compares the run name carried in the frame against the run name in the status and
draws a `frames X / learner Y` warning. But that check needs **both** names, and the frame's
name comes from the `ROYALEVISER_RUN` environment variable (`RoyaleGym/royalegym/viser.py`,
`from_env`), which RoyaleLearn never sets. So today the frame name is empty, the warning is
suppressed, and the two runs look like one. If you are going to run two jobs with a viewer
attached, set `ROYALEVISER_RUN` yourself to something distinct.

**What to do.** Run one training job per machine when you want the viewer. If you need two, give
the second one a different `ROYALEVISER` host:port, which moves both ports together, and point
the viewer at whichever you mean.

### 4.2 A killed run can leave workers behind

Workers are spawned processes named `royalelearn-worker-<index>` (`rollout/farm.py`, `_spawn`),
started with the `spawn` method on every platform and marked `daemon=True`. Three things
normally clean them up:

- the coordinator's `close()`, called from `__exit__` and from `learn`'s `finally`;
- `ProcessRolloutSource.close`, additionally registered with `atexit` (`farm.py` around line
  118), which sends each shard a close command, joins, terminates, then kills;
- the worker's own parent check. Each worker calls `multiprocessing.parent_process().is_alive()`
  every `PARENT_CHECK_S`, 2.0 seconds (`rollout/worker.py`), and exits when the answer is no.
  The comment there names the exact harm: a worker that outlives its parent holds a viewer port
  and a share of the memory, which makes the *next* run more likely to lose a worker to memory
  pressure and harder to attribute when it does.

So an orphan should clear within a couple of seconds. It does not always, because the parent
check only runs between rounds, and a worker inside a long engine step is not looking.

**How you spot it.** Three signs, in the order you will meet them:

1. The new run says the learning status port is not available, and nothing else is obviously
   running.
2. `python -m royalelearn doctor` reports far less free memory than the machine should have.
3. On Windows, `Get-Process python` lists more interpreters than your current jobs need.
   Workers are Python processes; the process name does not carry the worker name, so count
   rather than look for a label.

**What to do.** End the leftover `python` processes yourself before starting the next run. One
related symptom: shared-memory segments can outlive their process too. The farm handles that by
taking the next free name rather than attaching to somebody else's memory, and raises after 16
stale segments, which it documents as a symptom rather than a case to handle (`farm.py`,
`_build_worker`).

---

## 5. The minibatch and VRAM lesson

This is the most expensive thing learned on this project, and it is worth the four paragraphs.

**What happened.** `ppo.minibatch_size` was 512, chosen as the largest value that fits a 4 GB
card. It does not fit. Measured 2026-09-22 on that card (`config.DoctorConfig`,
`config.PPOConfig` and `metrics/alarms.py` all record it): minibatch 512 reserved **4,243 MB of
a 4,294 MB card** and the update took **180 to 233 seconds**, against **47 to 49 seconds** at
256. That is about 4.5 times slower. It did not fail. Windows does not refuse an oversubscribed
allocation, it backs it with host memory over PCIe and reports success, so the allocation is
served, the counters look ordinary, and the run is several times slower for the rest of its
life with nothing saying why.

For scale: machine contention alone moved the same update by 1.4 to 1.7 on that laptop, so a
factor of two sits between ordinary contention and a spill. That gap is what makes the alarm
below possible.

**What is shipped.** `ppo.minibatch_size = 256` is the laptop default (`config.PPOConfig`), and
it held 48 seconds within 2.3%. The larger profiles set their own: 2,048 on `workstation`, 8,192
on `many_core`. The cliff belongs to the footprint against the card, not to the number itself.
Minibatch size is a pure memory knob: gradients accumulate weighted by n over `batch_size` with
one optimizer step per batch, and a test proves the accumulated gradient equals the full-batch
one. Changing it changes what the run costs, not what it computes.

**What the gate does today.** `coordinator._vram_gate` (around line 1230) does not project. At
start-up it runs one real forward and backward pass at your actual `ppo.minibatch_size`, on
zeros with the no-op column set legal (`_probe_backward`), and reads `max_memory_reserved()`.
If that peak plus `doctor.vram_headroom_mb` (256 by default) exceeds what the driver reported
free *before* the probe, the run is refused with a message that states the measured peak, the
free memory, why the platform would not have failed on its own, and the next legal minibatch
size. "Next legal" is computed, not suggested: it must divide `ppo.batch_size`, and at the
shipped `batch_size` of 4,096 there is no divisor between 256 and 512, so the advice has exactly
one answer (`_next_legal_minibatch`). Set `doctor.vram_headroom_mb = 0` to skip the gate.

The gate is compared against what the driver says is **free**, not against the card's capacity,
and that choice is load-bearing: a cap on total capacity would have passed the 512
configuration, since 4,243 is under 4,294, and it spilled anyway on the other processes' share.

**What the alarm catches that the gate cannot.** The gate guards one instant. Another process
taking memory at hour three produces the same silent slowdown, later. `vram_spilling`
(`metrics/alarms.py`, `SpillAlarm`) watches the harm and uses the memory reading as evidence
that the harm is this one: the update at least 2.0 times this run's own best while the driver
reports under 128 MB free, for 3 consecutive iterations. The best is a running minimum, so a
slow iteration cannot raise the bar it is judged against and the first iteration's warm-up
cannot lower it.

The class docstring records why the obvious version was thrown away. The first attempt compared
free memory plus what this process already held against the startup peak. Tested on the card, a
second process took 2 GB and left 35 MB free for five iterations, and the alarm stayed silent by
645 MB, because an outsider can only consume the free part, so that sum bottoms out at what this
process holds, which the gate already guaranteed is enough. It was least sensitive in exactly the
case its own text named. Worth remembering if you write your own check.

It warns rather than halting, deliberately: preflight can afford to refuse because nothing is
lost at second five, and a nine-hour run cannot.

---

## 6. Stopping, resuming, and what survives

**Stopping.** Write `q` to `runs/<...>/control`, or press Ctrl-C, which the coordinator turns
into `q`. Either way the run writes a checkpoint on the way out. A crash or an alarm halt also
attempts an emergency checkpoint, and it *refuses* to make one while the learner in memory is
one no metric row describes, which is the window from the update's first change until the row
reporting it is written (`coordinator._emergency`, `_emergency_refusal`). When it refuses it
tells you which existing checkpoint to resume from instead.

**Resuming.** `python -m royalelearn resume --run runs\royalelearn-<id>`. With no `--checkpoint`
it takes the newest from the run index and prints which one it chose. A checkpoint whose identity
does not match is refused, with every differing field named; `--allow-identity-drift` overrides
that. Resume also refuses while code the run executes has uncommitted changes, your own reward
package included; `--allow-dirty` overrides that. A checkpoint written by an older version, from
before runs recorded the engine file and your own code, prints a `not checked on resume:` line
for each and carries on.

**What survives, what does not.** Read [checkpoints.md](checkpoints.md). It covers what is in the
file, how big it is, how often one is written, how to tell a resume is really continuing rather
than quietly restarting, and its own list of rough edges. The one thing worth repeating here: a
battle that was half played when the file was written is not finished, and the next battle starts
in its place.

---

## 7. What this page deliberately does not claim

No number here is offered as a result. The alarms describe conditions worth looking at, and what
each metric *would* mean about a run that was working. Treat any rating or explained-variance
figure you see as an instrument reading, not as evidence about a bot.

**Read next:** [metrics.md](metrics.md) for the numbers, [checkpoints.md](checkpoints.md) for
saving and resuming, [throughput.md](throughput.md) for speed and contention,
[ladder.md](ladder.md) for the pool and the gate, and [harness-spec.md](harness-spec.md) for all
of it in detail.
