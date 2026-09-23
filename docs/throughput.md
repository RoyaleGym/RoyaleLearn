# Throughput: how fast this goes, and what the slow part is

What you get from this page: a realistic idea of how long one training step takes, a command that
measures **your** machine instead of quoting someone else's, and a way to tell which part of your
computer is holding you up.

Every measured number here carries a date and says which run shape produced it. Where nobody has
measured a thing, this page says so rather than guessing. The gaps are real and there are several.

Last checked against the code on 2026-09-22.

## The short version

Training runs as a loop of **iterations**. Each iteration has two halves.

1. **Collection.** The bots play a batch of battles and write down every decision.
2. **The update.** PPO reads that batch and changes the bot's weights. PPO is the learning
   algorithm used here. It nudges the bot toward the choices that scored well, in steps small
   enough that it does not throw away what it already knows.

The two halves take turns. PPO needs data produced by the current weights, so it cannot keep
learning from a batch collected several weight changes ago. That is what people mean by
**on-policy**. Collect, learn, collect again.

On the laptop this harness was built on, a 4-core machine with a 4 GB RTX 3050, the update is the
slow half by a wide margin. Measured on 2026-09-22 at 2 workers by 24 battles and 8,192 timesteps
an iteration, an iteration took 30 to 35 seconds. The update was 25 seconds of it and collection
was 5 to 8 seconds. Across seven iterations the update was 71 to 83 percent of the whole thing.

**So on a small graphics card, training time is graphics card time, not battle engine time.** That
single fact decides most of what follows. It will not be true on every machine, which is why the
next section is a command and not a table.

## Measure your own machine

```
python -m royalelearn bench --profile laptop
```

`bench` is a real training iteration, not a simulation of one. It builds the environments, runs
the same start-up checks `train` runs, collects a batch, and performs a real PPO update. It needs
torch, like `train` and `doctor` do.

Two consequences. It takes as long as one iteration of the config you pointed it at, which on the
`laptop` profile is minutes rather than seconds. And it leaves a run folder behind at
`runs/<run_name>-<run_id>/`, with `config.json`, `identity.json` and `metrics.jsonl` in it, because
it goes through the same code path a run does (`royalelearn/cli.py`, `_bench`).

If you want a quick answer rather than a real one, point it at a smaller config:

```
python -m royalelearn bench --config examples/configs/train-diag1-selfplay.json
```

That file is 2 workers by 24 battles with 8,192 timesteps an iteration, which is the shape the
measurements at the top of this page came from.

### What bench prints

Ten lines, name and value. Here is what each one is.

| Line | What it means |
|---|---|
| `iterations` | How many whole iterations it timed. One to three. |
| `env_ms_per_game_step` | Milliseconds spent inside the battle engine per battle-step, as the workers report it. |
| `codec_us_per_row` | Microseconds to pack one observation into the shared buffer. |
| `boundary_mb_per_second` | Megabytes a second crossing between the worker processes and the learner. |
| `inference_ms_per_round` | Milliseconds the learner spends choosing actions for one round of one shard. |
| `update_timesteps_per_second` | How many transitions a second the PPO update works through. See the defects below. |
| `rollout_capacity_ratio` | Update seconds divided by collection seconds. Above 1 means collection is the cheaper half. |
| `vram_peak_mb` | Peak graphics memory this iteration, in decimal MB. A 4096 MiB card is 4,295 decimal MB, so read the percentage carefully. |
| `ratio_max_abs_dev`, `ratio_atol` | Not throughput. They are a correctness check on the stored log probabilities, printed here because this is the command that has them. |

One caveat on `codec_us_per_row`. It is measured in the parent process, by packing a zero-filled
observation 200 times, not in the workers on real battle data (`_codec_microseconds` in
`royalelearn/cli.py`). Treat it as an order of magnitude, not as your run's real packing cost.

### What bench does not tell you

It does not print collection seconds or update seconds, which is the split you most want. Those
are in the run folder it just wrote. `metrics.jsonl` holds one JSON object per iteration:

```
python -c "import json;
rows=[json.loads(l) for l in open('runs/royalelearn-<run_id>/metrics.jsonl')];
print([(r['run/iteration'], round(r['time/collection'],1), round(r['time/update'],1)) for r in rows])"
```

It also does not try any configuration but the one you gave it. It will not tell you whether more
workers, a second shard per worker or a different minibatch would be faster. Running it twice with
one field changed will.

And it does not warn you that the first iteration of any run is the slow one. Worker processes are
spawning, nothing is warm. The README records the first collection round of one `laptop` run at 46
seconds and the second at 19 seconds, on 2026-09-22. Wait for the second iteration before you
believe any timing.

### What was wrong with bench, and what is left

The train session reported four defects on 2026-09-22. Two are fixed, and two were true statements
about a command that had promised more than it did, so the promise went instead.

1. **It writes no page, and now it does not claim to.** The spec said `bench` writes the "measured
   on" block here. It never did, and it should not: this page is kept by hand, and a command that
   overwrote it would lose the prose around the numbers. `bench` prints the block and says to paste
   it under a dated heading. The spec says that now too.
2. **There is still no shard verdict, and the docstring no longer promises one.** Nothing in
   `bench` separates the shards. To answer whether a second shard is worth its thread, run `bench`
   twice on a quiet machine with `rollout.shards_per_worker` at 1 and at 2 and compare collection
   seconds. Under about 10%, set it to 1 and spend the thread on a worker.
3. **FIXED.** `update_timesteps_per_second` divided every transition collected since start-up by
   the last iteration's update seconds, so three iterations reported about three times the truth
   while one iteration was right, which is what kept it hidden. It is now that iteration's own
   transitions over its own update seconds. `tests/test_bench_report.py` grades the arithmetic.
4. **Documented rather than changed.** `--seconds` is a lower bound, checked between iterations and
   never inside one. The loop now stops at `--iterations` (default 3) instead of a hard-coded three,
   so a longer measurement is a flag rather than an edit. On the `laptop` profile one iteration is
   already past the 60-second default, so the default is one iteration.

## The two halves, in the numbers the harness records

Every iteration writes a row of metrics. The timing keys are defined in
`royalelearn/metrics/schema.py` and filled in `royalelearn/coordinator.py`.

| Key | What it holds |
|---|---|
| `time/iteration` | The whole thing. |
| `time/collection` | Playing the battles, including the learner choosing actions. |
| `time/inference` | The part of collection that is the learner choosing actions. |
| `time/env` | Seconds inside the battle engine, as the workers report them. |
| `time/codec` | Packing observations in the workers. |
| `time/ipc` | Waiting on round signals and reading scalars. |
| `time/update` | The PPO update. |
| `time/checkpoint`, `time/gate` | Saving, and judging a candidate against the frozen pool. |
| `time/residual` | Iteration seconds not attributed to any phase above. |

Three keys in that schema are written as `0.0` on every iteration today:
`time/critic_pass`, `time/gae` and `time/overlap_saved` (`coordinator.py`, lines 1769 to 1774).
The critic pass and GAE are not free, they simply happen inside the update, so their cost is
already inside `time/update` and is not broken out. Do not read those zeros as "this part is
instant".

`time/residual` is the honest one to watch. Anything the harness spends that is not collection,
update, checkpoint or gate shows up there as itself rather than being quietly absorbed.

## The three profiles

A profile is a set of numbers for a class of machine. They live in `royalelearn/config.py` as
functions, and `python -m royalelearn config --profile laptop -o run.json` writes any of them out
as a file you can edit.

These are the shipped values, and the derived shape follows from them. You can print the same
table yourself:

```
python -c "from royalelearn.config import profile, geometry;
[print(n, geometry(profile(n))) for n in ('laptop','workstation','many_core')]"
```

| | `laptop` | `workstation` | `many_core` |
|---|---|---|---|
| Target machine | 8 GB RAM, 4 GB VRAM | 32 GB RAM, 12 GB VRAM | 64 GB RAM, 24 GB VRAM |
| Worker processes | 3 | 12 | 24 |
| Battles at once | 96 | 768 | 1,536 |
| Decisions per battle per iteration | 228 | 114 | 114 |
| Learner transitions per iteration | 32,768 asked, 32,832 collected | 131,072 | 262,144 |
| Minibatch | 256 | 2,048 | 8,192 |
| Network width and depth | 64 channels, 4 blocks | 96 channels, 8 blocks | 128 channels, 12 blocks |

**Use `laptop` unless you have measured otherwise.** It is what `config`, `doctor` and `bench`
default to, and it is the only profile anyone has run. `workstation.json` and the `many_core` profile have never been
run, so their minibatch values are a guess about a card nobody here owns. If the guess is wrong for
your card, the start-up check refuses the run and names a smaller value to use, which is the
failure you want.

One `laptop` iteration plays about three hours of Clash Royale. That is 96 battles advancing 228
decisions each, at the 500 ms per decision the shipped configs ask for. The harness reads the real
decision length off the running environment at start-up and prints it, so if your environment says
something else, that is the number that counts.

## Why the update dominates on a small card

The update does not walk the batch once. On the `laptop` profile it walks it three times
(`ppo.n_epochs` is 3), and each pass is split twice over:

- 32,832 transitions, in batches of 4,096, is 9 optimizer steps per pass.
- Each batch of 4,096 is fed to the card in minibatches of 256, so 16 forward and backward passes
  per optimizer step.
- Three passes over the batch: roughly 390 forward and backward passes in one iteration.

Collection, meanwhile, is 228 rounds of work spread over three separate processes that run while
the learner is mostly idle.

**The minibatch is a memory knob, not a learning knob.** Gradients accumulate across the
minibatches of a batch, weighted, with one optimizer step per batch, so the result is identical
whatever the minibatch is. `royalelearn/learn/ppo.py` says this at the top and a test holds it
there by comparing the accumulated gradient against the whole-batch one. The only thing the
minibatch changes is how much graphics memory is live at once.

That is why the value matters so much on a small card, and why guessing it went wrong once here.
Measured on 2026-09-22 on the 4 GB card: a minibatch of 512 reserved 4,243 MB of a 4,294 MB card
and did not fail. It ran about 4.5 times slower instead, because Windows backs an oversubscribed
allocation with system memory over the PCIe bus rather than refusing it (the note beside
`vram_headroom_mb` in `royalelearn/config.py` records this). A minibatch of 256 is stable on the
same card. 256 is now the shipped default in all three example configs and in the code.

The lesson for your own machine: set the minibatch to the largest value that **fits**, not the
largest the card's advertised size suggests. And when the update suddenly gets several times
slower without anything else changing, suspect a spill before you suspect anything else.

## What this changes about your choices

**Do not add workers to fix a slow iteration on a small card.** Check `rollout_capacity_ratio`
first. It is update seconds over collection seconds. The 2026-09-22 diagnostic figures, 25 seconds
of update against 5 to 8 of collection, work out at roughly 3 to 5. Collection was already three to
five times cheaper than the update. Adding workers there buys nothing and costs memory.

**The levers that move the update** are the network size (`net.channels`, `net.blocks`), the number
of passes over the batch (`ppo.n_epochs`), and the batch size itself
(`ppo.timesteps_per_iteration`). Halving the timesteps per iteration roughly halves the update and
roughly halves the collection with it, so it makes iterations shorter without making training
faster per transition. Nobody has measured the learning cost of any of these changes here.

**The lever that moves collection** is workers times battles per worker, bounded by your memory
and your cores. There are three workers in the `laptop` profile on a machine with eight threads,
not thirty-two, because the learner is the bottleneck there.

**`rollout.overlap` does nothing yet.** It is on in the `workstation` and `many_core` profiles and
it is meant to hide collection under the update. The loop in `coordinator.py` collects and then
updates, with no branch on that setting, and `time/overlap_saved` is written as 0.0 every
iteration. The memory projection does reserve room for a second batch when it is on. So today the
setting costs projected memory and saves nothing. Leave it off on a small machine.

## Which resource is biting

Four candidates, and the metrics row tells them apart. All of these keys are in `metrics.jsonl` and
on the console at the end of every iteration.

### The graphics card

- `throughput/rollout_capacity_ratio` well above 1 means the update is the slow half. That is the
  expected state on a small card. An alarm called `capacity_ratio` fires the other way, after three
  iterations below 1.5, because that means the harness has become the bottleneck rather than the
  learner.
- `health/vram_peak_mb` is this iteration's peak, reset on every read, so it falls when you lower
  the minibatch. If it were never reset it would report the largest allocation the process ever
  made and repeat that figure forever, which is a mistake this code made once and fixed.
- `health/vram_available_mb` against `health/vram_needed_mb` is the spill test. `needed` is
  measured at start-up by running one real minibatch backward pass on your card. `available` is
  driver-free memory plus what this process already holds. When available drops below needed, the
  `vram_spilling` alarm fires after three iterations, and it means the update is being backed by
  system memory over PCIe and is several times slower than it should be.
- `throughput/gpu_util_frac` is device utilisation over the iteration.

Before any of that, there is a refusal. At start-up the harness runs one real minibatch backward
pass, measures what it cost, and refuses to start if that peak plus `doctor.vram_headroom_mb`
(256 MB by default) does not fit in the free memory the driver reports. The error names the next
legal smaller minibatch. This is the only threshold in the harness measured on the machine it is
applied to, minutes before it is applied.

### Memory

- `python -m royalelearn doctor --config run.json` prints the memory projection for your settings
  beside how much is free on your machine right now, and refuses a run whose projection exceeds
  `doctor.ram_budget_mb`, 6,500 MB by default. Going over what is free right now is a warning and
  not a refusal, deliberately: something else on your machine may well stop before the run needs
  the memory.
- Be honest about that projection. The per-process figures inside it come from the design document,
  not from a measured run. Only the experience buffer term is arithmetic on your actual settings.
- When you really run out, the symptom is a worker being killed mid-round rather than a clean
  error. `health/worker_restarts` and `health/rows_dropped_dead_worker` count it. A run survives
  it, with fewer transitions that iteration.

### Processor cores

- `throughput/parent_wait_frac` is the share of a round the learner spent blocked, waiting for
  workers that had not published yet. The schema expects it under 0.3. Rising means the workers
  cannot keep up, which on a laptop usually means too many workers for the cores you have, or
  something else on the machine.
- `time/env` against `time/collection` says how much of collection was really inside the battle
  engine rather than waiting, packing or choosing actions.

### Something else on the machine

This is the most common one on a laptop and the hardest to see in the metrics. The README records
one `laptop` iteration that shared eight processors with a second training run and had still not
finished after 46 minutes. If your numbers are several times worse than the ones on this page,
count what else is running before you change any setting.

## The honest state of the figures

What has actually been measured, all on 2026-09-22, on one 4-core laptop with an RTX 3050 4 GB and
7.8 GB of system memory, with up to six other jobs sharing it:

- **The diagnostic shape** (2 workers by 24 battles, 8,192 timesteps an iteration, minibatch 256,
  which is `examples/configs/train-diag1-selfplay.json`): 30 to 35 seconds an iteration, of which
  the update was 25 seconds and collection 5 to 8 seconds. Over seven iterations the update was 71
  to 83 percent of every iteration. Those 8,256 transitions an iteration are 48 battles advancing
  43 seconds of game time each, so collection ran roughly 260 to 410 times faster than watching
  those battles in real time, and the whole iteration including the update ran roughly 60 to 70
  times faster.
- **Minibatch 512 does not fit the 4 GB card.** It reserves 4,243 MB of 4,294 MB and spills into
  system memory instead of failing. 256 is stable. The two complete `laptop` profile iterations
  anyone has timed took 518 and 544 seconds, and both were run at minibatch 512, the setting that
  spills.
- **Attaching the viewer to a running job cost nothing detectable.** 18 iterations alternating
  three attached and three detached, three times over: 28.13 plus or minus 1.25 seconds attached
  against 28.33 plus or minus 1.65 seconds detached. This figure is reported by the session that
  ran it and has not been independently checked.

What has **not** been measured, and is worth knowing before you plan around anything here:

- **No full `laptop` profile iteration at the shipped minibatch of 256 has been timed.** The 518
  and 544 second figures were taken at 512. A run at 256 should be faster, and by how much is
  unknown.
- **No `workstation` or `many_core` run exists.** Every number in those two profile columns is a
  choice, not a measurement.
- **Nobody has measured the value of a second shard per worker.** The setting exists, the comment
  in `config.py` estimates it at about 25 percent of round time, and `bench` does not test it.
- **Nobody has trained a bot with this yet.** All of the above is about how fast the loop turns,
  not about whether turning it produces a good player. Speed and learning are different questions
  and this page only answers the first.
- **No figure here was collected on an idle machine.** Six other jobs were running. Expect your
  own numbers to differ, in either direction.

## Read next

- [`docs/design.md`](design.md) for what the pieces are and why the harness is shaped this way.
- [`docs/harness-spec.md`](harness-spec.md) for the full specification, including section 2 on the
  throughput budget and section 13 on the metric schema and the alarms. It is dense, and where it
  disagrees with the code, the code is what runs.
- The README for the install, the first run, and the reward function you will actually be writing.
