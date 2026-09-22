# Saving a run, resuming it, and what comes back

Training a bot takes hours or days. Your laptop will sleep, Windows will want to update, and you
will want your GPU back for something else. This page is about stopping a run and starting it
again without losing the work.

The short version: a checkpoint is a folder of files. Resuming from one puts the learner back
exactly as it was, byte for byte, in a fresh process. What it cannot put back is a battle that was
halfway through when the file was written.

For the full specification, see [harness-spec.md](harness-spec.md). For why the pieces are shaped
this way, see [design.md](design.md). This page is the practical version.

## The commands

| What you want | Command |
| --- | --- |
| Save right now, keep training | write `c` into `runs/<your-run>/control` |
| Save and stop | write `q` into that same file, or press Ctrl-C |
| Continue the newest checkpoint | `royalelearn resume --run runs/<your-run>` |
| Continue a specific one | `royalelearn resume --run runs/<your-run> --checkpoint runs/<your-run>/checkpoints/000002000000` |
| Prove a resume works on your machine | `royalelearn verify-resume --config my.json` |
| Watch a saved bot play one battle | `royalelearn play --checkpoint runs/<your-run>/checkpoints/000002000000` |

The control file is read once per collection round, so a letter you write takes effect at the end
of the iteration that is running. Ctrl-C is wired to `q`, which means an interrupt is a save and
not a lost afternoon. That handler is installed in `royalelearn/coordinator.py`, class `_Control`.

## Where the files go

Everything one run writes lives in one directory, named `<runs_dir>/<run_name>-<run_id>`. The
default `runs_dir` is `runs`, relative to wherever you started the command
(`royalelearn/config.py`, `runs_dir: str = "runs"`).

```
runs/my-run-f4a99b1ce0c23888/
  config.json          the resolved config, written once
  identity.json        what this run is, hashed
  metrics.jsonl        one line per iteration
  episodes.jsonl       one line per finished battle
  alarms.jsonl         anything that tripped
  checkpoints/
    index.json
    000000041280/      <- a checkpoint, named by cumulative environment steps
  snapshots/           frozen opponents, a different thing (see below)
  ladder/              the record of who beat whom
```

The twelve digits are the cumulative environment step count, zero padded so the folders sort. The
`index.json` beside them is how `resume` finds the newest one. It reads the index rather than
parsing folder names, because parsing names crashes on any stray file a human leaves in there, and
this code path also runs from the crash handler.

### How big is one

Measured on 2026-09-22, from `runs/train-diag1-selfplay-f4a99b1ce0c23888/checkpoints/000000041280`,
at the default network (64 channels, 4 blocks):

| Part | Size |
| --- | --- |
| Whole checkpoint | 10.5 MB, 19 files |
| `optimizers/` | 6.8 MB |
| `actor_critic/` | 3.4 MB |
| everything else | under 30 KB combined |

Measure your own:

```bash
du -sk runs/<your-run>/checkpoints/*/
```

The default keeps the newest 10 and deletes the rest after each write
(`CheckpointConfig.keep = 10` in `royalelearn/config.py`). So the disk cost of a long run at the
default network is about 105 MB, not 10.5 MB times however many hours you ran.

One measured write cost 0.35 seconds, in run `train-deckA-live` at iteration 26 on 2026-09-22, read
off the `time/checkpoint` column of `metrics.jsonl`. That is a single sample and not an average.
The column is reported on the row after the save, because the save happens after the row it belongs
to is written.

## What is inside

Ten folders, one per component, plus a `manifest.json`. Each component writes its own files, so
nothing about a component's state is described in two places.

| Folder | What it holds | Why you would miss it |
| --- | --- | --- |
| `actor_critic/` | the two networks' weights, and a description of the architecture | this is the bot |
| `optimizers/` | both Adam optimizers' momentum terms, and the update counter | without it the first few updates after a resume are wrong, and the curve dips |
| `schedules/` | the learning rate backoff, and the values the last iteration ran at | the rates would jump back to their configured start |
| `advantage/` | the running mean and variance used to scale returns | the scale would restart and the critic's loss would spike |
| `rollout/` | which battle each worker is on, and which random stream each one draws from | the run would replay its opening battles under weights that have moved on |
| `matchmaker/` | which battle plays which episode next | same reason |
| `ladder/` | the pool of frozen opponents, who the champion is, their ratings | the opponent ladder would start empty |
| `rating/` | the Bradley-Terry fit and the Elo readout | the numbers you have been watching |
| `metrics/` | how many rows have been written | the row counter |
| `rng/` | every random number stream's position | see below |

"Advantage" is the reinforcement learning word for how much better an action turned out than the
bot expected. The scaler in `advantage/` keeps those numbers in a sane range. "Adam" is the
optimizer, the thing that decides how far each weight moves; it keeps a running memory of recent
gradients, which is why it has state worth saving.

`manifest.json` sits at the top and records the run id, the identity, the whole resolved config
verbatim, the iteration, the step counters, a `state_digest`, and a SHA-256 for every other file in
the checkpoint. The manifest is not in its own file list, which is why a checkpoint has 19 files and
18 hashes.

```bash
python -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['iteration'], m['cumulative_env_steps'], m['state_digest'][:16]); print(sorted(m['files']))" runs/<your-run>/checkpoints/000000041280/manifest.json
```

### Nothing here can run code when you load it

Network weights are safetensors. Everything else is JSON. The two optimizer files are `torch.save`
output, read back with `weights_only=True`, which can only decode tensors
(`royalelearn/learn/ppo.py`, `save_checkpoint` and `load_checkpoint`). No `pickle` on any path this
code writes. `tests/test_checkpoint.py::test_nothing_written_carries_a_pickle` scans the written
bytes for pickle markers rather than trusting the intent.

This matters because you will load a checkpoint six weeks later from a folder you have not looked
at since, possibly one somebody sent you.

### The write is atomic

Files go into `<name>.partial/`, each one is fsynced, and then the directory is renamed into place.
If the machine dies mid-write you keep the previous checkpoint intact and you get an obviously named
`.partial` folder that the next prune removes. The directory handle itself is not fsynced, because
that call raises on Windows, which is where the default profile runs
(`royalelearn/checkpoint.py`, `DirCheckpointStore.write`).

## What is not inside

- **The rollout buffer.** The iteration boundary is the resume point, so there is no mid-iteration
  state to keep. There is an `include_buffer` field in the config, default `false`. Nothing in the
  code reads it today. Grep for it and you will find one definition and no use.
- **The frozen opponent snapshots.** Those live in `runs/<your-run>/snapshots/` and are never
  pruned. The `ladder/` folder inside a checkpoint holds the *index* of the pool, not the weights.
  See the section below.
- **A half-played battle.** This is the real limitation and it has its own section.

## How often a checkpoint is written

Every `checkpoint.every_env_steps` environment steps, default 2,000,000
(`royalelearn/config.py`, `CheckpointConfig`). An "environment step" is one game tick that one
battle advanced, summed over every battle running.

On the laptop profile that works out to about 91 iterations between saves. You can check the
arithmetic for your own config without running anything:

```bash
python -c "
from royalelearn.config import profile, geometry
c = profile('laptop'); g = geometry(c)
per_iter = g.cycles * g.n_battles
print(g.workers, 'workers,', g.n_battles, 'battles,', per_iter, 'env steps an iteration')
print(c.checkpoint.every_env_steps / per_iter, 'iterations between checkpoints')
"
```

On 2026-09-22 that prints 3 workers, 96 battles, 21,888 env steps an iteration, and 91.37
iterations between checkpoints.

Ninety-one iterations is a long time on a 4 GB laptop. If you are experimenting, lower
`checkpoint.every_env_steps` in your config file. Set it to `0` and periodic saves are off
entirely; the manual `c`, the quit save, and the crash save all still work.

A checkpoint is also written when:

- you write `c` or `q` to the control file, or press Ctrl-C;
- an alarm halts the run (it checkpoints, then writes a diagnostic bundle, then raises);
- the run crashes, with one important exception described next.

## The save a crashing run refuses to make

This landed on 2026-09-22, in commit `0839758`.

The rule is that a checkpoint's weights must be weights that some row of `metrics.jsonl` reports.
There is a window in every iteration where that is not true: from the moment the optimizer first
changes a weight until the row describing that update is written. Inside that window the learner in
memory is one that no row describes.

If a run crashes inside that window, it now writes nothing and prints where to resume from instead:

```
no emergency checkpoint: the update had started changing the learner and no metrics row
describes it yet; resume from runs/my-run-.../checkpoints/000002000000
```

You lose whatever the interrupted update had done. That is the same trade the checkpoint interval
already makes, and it is better than the alternative, which is a file holding weights trained past
the counters saved beside them. Two such files were found on disk before this change, one holding
72 optimizer updates against its row's 54.

There is a second refusal, on the same principle, at load time. `resume` checks that the
checkpoint's `state_digest` appears in some row of that iteration in `metrics.jsonl`. A digest no
row carries is refused by name:

```
the checkpoint of iteration 12 holds learner 9f3c... and no metrics row describes it:
runs/my-run/metrics.jsonl records 2086e4f7dd8105a9 for that iteration. Its weights were
trained past the row its counters belong to; resume from an earlier checkpoint with --checkpoint
```

If there is nothing to compare against (no metrics file, no row of that iteration, a last line torn
by a crash) the resume goes ahead and says so, because an absence tells you nothing about the
weights. Iteration zero has no row by construction and is silent. The logic is
`check_described` in `royalelearn/checkpoint.py`, tested in `tests/test_checkpoint.py`.

## Resuming

```bash
royalelearn resume --run runs/my-run-f4a99b1ce0c23888
```

You do not pass a config. The run directory describes itself: `resume` reads its `config.json` and
points `runs_dir` and `run_name` back at where the directory actually is, so a run you moved or
renamed still resumes (`_run_config` in `royalelearn/cli.py`).

### What you should see

Three things, in this order:

```
resuming from runs/my-run-f4a99b1ce0c23888/checkpoints/000000041280
```

Then, if your config has changed since the checkpoint, one line per changed setting, by dotted
path. These are printed and continued past, not refused:

```
config differs from the checkpoint's: checkpoint.every_env_steps: 2000000 -> 500000
```

Then the line that tells you it worked:

```
resumed       iteration 10 at 41280 env steps, state 2086e4f7dd8105a9
```

After that the usual iteration output starts again, at iteration 11.

The resumed process writes into the **same** run directory. `metrics.jsonl` is opened in append
mode, so your curve keeps going in the same file rather than starting a second one.

## How to tell it is really continuing and not quietly restarting

Three checks, in increasing order of paranoia.

**1. The counters continue.** The first row after a resume carries `run/iteration` and
`run/cumulative_env_steps` one step past the checkpoint's, not zero.

```bash
tail -2 runs/<your-run>/metrics.jsonl | python -c "
import json,sys
for l in sys.stdin: r=json.loads(l); print(r['run/iteration'], r['run/cumulative_env_steps'], r['run/state_digest'][:16])
"
```

**2. The state digest matches.** This is the strong one. `state_digest` is a SHA-256 over the
weights, both optimizers' moment buffers, the return scaler and the schedule positions, hashed in a
fixed order (`state_digest` in `royalelearn/coordinator.py`). The digest the resume prints must
equal the digest in the checkpoint's manifest, and that digest must appear in `metrics.jsonl`.

```bash
python -c "
import json
m = json.load(open('runs/<your-run>/checkpoints/000000041280/manifest.json'))
rows = [json.loads(l) for l in open('runs/<your-run>/metrics.jsonl')]
match = [r for r in rows if r['run/iteration'] == m['iteration'] and r['run/state_digest'] == m['state_digest']]
print(m['iteration'], m['state_digest'][:16], 'described by', len(match), 'row(s)')
"
```

On the run measured for this page that printed `10 2086e4f7dd8105a9 described by 1 row(s)`.

**3. Prove it on your own machine.** `verify-resume` runs a few iterations, checkpoints, then
starts a genuinely separate process to resume it and compares the learner state across the process
boundary:

```bash
royalelearn verify-resume --config my.json --iterations 6 --split 3
```

It prints `the resumed process loaded the same learner state, byte for byte` or it tells you the two
digests that disagreed. This takes real time, since it runs six iterations of your config twice
over.

## What does not come back

The learner comes back exactly. The environment comes back deterministically but not mid-battle.

A checkpoint records, per battle, which episode that battle is about to play, and which seeded
stream that episode's randomness comes from. On resume each battle starts an episode from the
beginning. A battle that was halfway through when the file was written does not continue from
halfway through; nothing offers that today.

You do not lose training data. The transitions already collected from that battle were in the
buffer of the iteration that finished, and they were already trained on. What you lose is the
*phase*: after a resume the battles are the right battles, but they are all starting fresh rather
than spread across the middle of their episodes the way they were before.

The practical effect:

| If the checkpoint fell on | You get |
| --- | --- |
| an episode boundary for every battle | the following iterations reproduce the original run field for field, including the state digest |
| the middle of some battles | the same battles against the same opponents, but the episode counts and episode length averages in the first rows after the resume differ from an uninterrupted run |

Both of those are tested rather than asserted in prose.
`tests/test_resume.py::test_a_resume_continues_the_original_row_for_row` runs six iterations
straight through, then three plus three, and compares every metric field.
`tests/test_resume.py::test_an_episode_in_flight_is_not_replayed` measures the gap, and is written
so that it fails if resuming inside an episode ever becomes possible.

```bash
python -m pytest tests/test_resume.py --collect-only -q -m slow
```

Those five tests are marked `slow` and are excluded from the default suite, so you have to ask for
them by name.

Two smaller things that also do not come back: the wall clock columns (`time/`, `throughput/`) are
about this process and not about the run, and a paused run's pause does not survive a restart.

## When a resume refuses

It refuses rather than continuing something subtly different. Four ways:

- **The identity moved.** The identity is what the run is: the engine build, the card catalogue,
  the observation layout, the network architecture, the master seed and a few more. If any of it
  differs, the resume stops and names every field that changed. A policy trained against one
  catalogue's observation vector cannot load into another's, and the failure you would otherwise get
  is a tensor shape error deep inside a forward pass. `--allow-identity-drift` records the
  differences to `drift.json` and continues, and marks every subsequent row with
  `run/resumed_with_drift`. Use it knowing the curve is no longer one experiment.
- **A file does not hash to what the manifest says.** Every file is checked before anything loads.
  A write truncated by a crash six weeks ago becomes an error at the moment of the resume, naming
  the file, instead of a run that continues from something else.
- **The learner is not described by any metrics row.** Described above.
- **A component folder is missing.** Under the default `checkpoint.strict_load = true` this raises
  and names the folder. Set it to `false` and each missing piece prints the exact path it wanted
  and carries on with a default. Tolerate-everything is right for a research toy and wrong for a
  harness that promises the curve continues, so the default is strict.

## Checkpoints and pool snapshots are different things

These are easy to confuse because both are files of weights in your run directory.

| | Checkpoint | Pool snapshot |
| --- | --- | --- |
| Where | `runs/<run>/checkpoints/<step>/` | `runs/<run>/snapshots/<digest>/` |
| Holds | the whole run: both networks, optimizers, ladder, streams, counters | one frozen actor network, in half precision, plus a description |
| For | continuing the run | being an opponent, and being rated |
| Written | every 2,000,000 env steps by default | every 4,000,000 env steps, as a gate candidate (`LadderConfig.candidate_every_env_steps`) |
| Deleted | all but the newest 10 | never |
| Restores training | yes | no, you cannot resume from one |

Your bot plays against its own past selves. A snapshot is one of those past selves: frozen, given an
id, auditioned by the gate, and admitted to the pool if it is good enough. Its rating would mean
nothing without the player that earned it, which is why snapshots are kept forever while
checkpoints are pruned. At the default network a snapshot is roughly half the size of the
`actor.safetensors` in a checkpoint, because it is stored in half precision. That has not been
measured at the default network here; the only snapshots on this machine on 2026-09-22 came from
tiny-network probe runs, at 17 KB each, which tells you nothing useful about a real one.

The `ladder/` folder *inside* a checkpoint is the pool's index: which snapshot ids are in it, which
one is champion, what their ratings are. Not the weights. So a checkpoint is only resumable
alongside the `snapshots/` folder of the same run. Do not copy `checkpoints/` somewhere on its own
and expect the opponents to come with it.

## Rough edges, honestly

- There is no way to resume inside an episode, and no plan to add one written down here.
- `checkpoint.include_buffer` exists in the config and nothing reads it.
- Checkpoint sizes on this page are one measurement of one run at the default network on
  2026-09-22. Bigger profiles use bigger networks, so scale accordingly rather than trusting 10.5 MB.
- The write cost above is a single sample, not a distribution.
- Moving a run directory between machines is untested here. The identity check includes the engine
  build and the device kind, so a resume on different hardware is likely to refuse before it gets
  anywhere interesting.
