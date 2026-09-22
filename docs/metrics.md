# The numbers a run prints, and which ones to watch

A training run prints about 145 numbers every iteration. You do not need most of them. This page
tells you where they land, how to read one, which handful actually tell you whether the bot is
learning, and which ones are broken or empty today so you do not spend an evening chasing them.

The deep version of all of this lives in [harness-spec.md](harness-spec.md) and
[design.md](design.md). The single source of truth for what any one number means is the code:
`royalelearn/metrics/schema.py`, which lists every key with a unit, a sentence, and a healthy
range. Every claim on this page was checked against that file or measured on 2026-09-22.

A few words first, because the rest of the page uses them.

- **Iteration.** One lap of the training loop. The bot plays a pile of battles, then learns from
  them, then starts again. One line of numbers per lap.
- **Timestep.** One decision by one bot in one battle. An iteration collects tens of thousands.
- **PPO.** The learning algorithm the harness uses. It nudges the bot toward things that worked
  and away from things that did not, in small steps, and refuses to take a big step even when the
  data suggests one. The "update" is the part of an iteration where that happens.
- **Critic.** A second small network that guesses how the battle is going to end from the current
  board. It is not the bot that plays. It exists so the bot can tell a good move from a lucky one.

---

## Where the numbers go

Everything one run writes ends up in one folder: `runs/<run_name>-<run_id>`, from
`run_directory` in `royalelearn/coordinator.py`. The defaults are `runs_dir = "runs"` and
`run_name = "royalelearn"` (`royalelearn/config.py`), so a plain `royalelearn train` writes into
`runs/royalelearn-<id>/`.

Inside it:

| File | What it is |
| --- | --- |
| `metrics.jsonl` | One JSON object per iteration. The main thing. |
| `episodes.jsonl` | One record per finished battle. Big. Gzipped in place after 200 iterations. |
| `alarms.jsonl` | Only written when an alarm fires. An empty run has no such file. |
| `config.json` | The full resolved config, written once at the start. |
| `identity.json` | What this run *is*: seeds, versions, the things two runs must share to be comparable. |
| `checkpoints/` | Saved learners you can resume from. |
| `snapshots/`, `ladder/` | Past versions of the bot and the record of who beat whom. |
| `artifacts/` | Per-card heatmaps, every 50 iterations (`metrics.image_every`). |
| `bundles/` | Written only when a run halts on an alarm. See the alarms section. |

Four destinations are configured by default, and you can see them yourself:

```
../.venv/Scripts/python -c "from royalelearn.config import profile; print([(s.kind, s.enabled) for s in profile('laptop').metrics.sinks])"
```

which prints `[('jsonl', True), ('console', True), ('viser', True), ('wandb', False)]`.

- **jsonl** is the file. It is installed whether or not your config asks for it, and `validate`
  refuses a config that disables it (`config.py`, around line 706). It flushes after every row,
  so you can tail the file while the run is going.
- **console** is the block printed to your terminal each iteration.
- **viser** feeds the live viewer, if you have one attached. It sends nothing at all until a
  viewer says hello, so a run with no viewer costs one clock read per iteration.
- **wandb** is Weights and Biases, a hosted dashboard. It is **off in every shipped config**:
  all three profiles (`laptop`, `workstation`, `many_core`) and all eight files in
  `examples/configs/` have `"kind": "wandb", "enabled": false`. It needs an account and the
  `wandb` package, which is not installed here. If you turn it on without installing the package,
  the run refuses to start with a message telling you to install it
  (`royalelearn/metrics/wandb_sink.py`).

So out of the box: a file, and your terminal. Nothing leaves the machine.

---

## How to read one row

A row is flat. Every key is `group/name`, and there are eight groups, in this order: `run`,
`throughput`, `time`, `ppo`, `policy`, `env`, `ladder`, `health`.

The console prints exactly the row it was handed, grouped by the part before the slash. Here is a
real block, from `runs/train-diag0-hog26k-1ad6a480b7666090` on 2026-09-22, trimmed to twelve keys
so it fits on a page (the real one has 145):

```
-- iteration 7 ----------------------------------------
  run
    iteration                          7
    wall_seconds                       226.1
  throughput
    overall_steps_per_second           251.7
  time
    iteration                          34.76
    update                             26.65
  ppo
    kl                                 2.001e-06
    clip_fraction                      0
    explained_variance                 0.3017
  policy
    cards_per_match                    42.87
  env
    episodes_completed                 23
    draw_rate                          0
  health
    vram_peak_mb                       2009
```

To pull the same numbers out of the file afterwards:

```
../.venv/Scripts/python -c "
import json, sys
for line in open(sys.argv[1], encoding='utf-8'):
    r = json.loads(line)
    print(r['run/iteration'], r.get('ppo/kl'), r.get('ppo/explained_variance'), r.get('policy/cards_per_match'))
" runs/train-diag0-hog26k-1ad6a480b7666090/metrics.jsonl
```

which on that run prints:

```
1 7.141222340578679e-06 -0.09380161762237549 None
2 8.427956004197767e-07 0.0865025520324707 None
3 3.0327626632242755e-07 0.14469212293624878 29.227272727272727
...
```

Note the `None` on iterations 1 and 2, and note that the command uses `.get()` rather than `[...]`.
**A missing key is not a zero.** Those two iterations finished no battles, so there is no `env/`
group and no `policy/cards_per_match` in the row at all. The harness does this on purpose: a zero
would read as "no cards played", which is a different and much worse thing than "no battle
finished yet". Write your plotting code to skip missing keys, not to fill them with zeros.

To look up what any key means and what range is healthy:

```
../.venv/Scripts/python -c "
from royalelearn.metrics.schema import lookup
s = lookup('ppo/explained_variance')
print(s.unit, '|', s.low, 'to', s.high)
print(s.description)
"
```

---

## The rows to actually watch

Nine numbers. The ranges come from `low` and `high` in `royalelearn/metrics/schema.py`.

| Key | Healthy | What it means, plainly |
| --- | --- | --- |
| `policy/cards_per_match` | 10 to 30 | How many cards the bot plays in a battle. This is the single best health check. A bot that has quietly learned to do nothing shows up here first. The schema says about 22 is right. |
| `ppo/explained_variance` | 0.5 to 0.9 | How well the critic predicts how a battle ends. 0 means it is guessing. Below zero means it is worse than guessing the average. Still negative after fifty iterations is the most likely reason a run stops improving. |
| `ppo/kl` | 0.003 to 0.02 | How far the update moved the bot's behaviour. Think of it as step size, measured in the bot's own behaviour rather than in weights. Too small and nothing is happening. Too large and the update is thrashing. |
| `ppo/clip_fraction` | 0.05 to 0.20 | The share of samples where PPO refused a big step and clamped it. A little is normal and is the algorithm working. Pinned near 1.0 means something is wrong, usually a mismatch between what the bot did during the battles and what the update thinks it did. |
| `ppo/entropy_normalised` | 0.3 to 0.8 | How spread out the bot's choices are, as a fraction of maximum spread. 1.0 is a coin flip between every legal move. Near 0 means the bot always does the same thing. Early it should be high, and it should come down slowly. |
| `env/win_rate_by_seat` | 0.45 to 0.55 | How often the blue side wins. Both sides see a mirrored copy of the same board, so anything far from a half is a bug in the harness, not a discovery about the game. |
| `env/draw_rate` | under 0.5 | Share of battles where nobody won. Two bots that have both learned to sit still produce a lot of these. |
| `env/illegal_action_rate` | exactly 0 | Commands the engine refused. Any non-zero value at all is a bug, and it halts the run. |
| `throughput/rollout_capacity_ratio` | 2.0 or more | Time spent learning divided by time spent playing. Above 1 means the learner is the slow part, which is the intended state. Below 1.5 warns. |

Two more worth a glance, without a healthy range attached:

- `time/update` against `time/iteration`. This tells you where your evening is going. On the
  diagnostic run above (2 workers, 24 battles each, 8,192 timesteps an iteration, minibatch 256,
  measured 2026-09-22 on a 4-core laptop with an RTX 3050 4 GB and 7.8 GB RAM, sharing the machine
  with six other jobs) an iteration took 30 to 35 seconds, of which the update was 25 seconds and
  collecting the battles was 5 to 8. The update was 71 to 83 percent of every iteration across all
  7 iterations. On that machine, training time is GPU time. Buying more simulator speed would have
  bought nothing. That is one geometry on one machine and it is **not** the shipped laptop profile,
  which is 3 workers, 32 battles each and 32,768 timesteps an iteration.
- `ladder/rating_above_v0`. How much stronger the bot is than its own first snapshot, in Elo
  points (the chess-style rating scale). It is 0.0 until enough evaluation battles have been played
  for a fit, so a short run shows 0.0 and that is not a bug.

If you only ever look at two, look at `policy/cards_per_match` and `ppo/explained_variance`.

---

## Rows that are zero or unreliable today

These are real gaps. They are listed here rather than hidden because a number that is always zero
looks like an answer.

You can reproduce the whole list yourself by scanning every run on disk for keys that are zero in
every row they appear in. As of 2026-09-22 there were 56 metric files and 249 rows under `runs/`.

| Key | State | Why |
| --- | --- | --- |
| `throughput/gpu_util_frac` | Always 0.0, in all 249 rows | `_gpu_util` in `coordinator.py` calls `torch.cuda.utilization()`, which needs NVIDIA's NVML Python bindings. They are not installed in this environment, the call raises, and the function catches everything and returns 0.0. Check with `../.venv/Scripts/python -c "import importlib.util as u; print(u.find_spec('pynvml'))"`. Use `nvidia-smi` in another terminal instead. |
| `health/rss_peak_mb` | Always 0.0, in all 249 rows | Peak system memory of the parent process. The Unix path uses the `resource` module, which does not exist on Windows. The Windows fallback calls `GetProcessMemoryInfo` through `ctypes.windll.psapi`, and that call returns failure on this box, so the function returns 0.0. The code chooses to report zero rather than guess, which is the right call for a number the planner reads, but it does mean the row is empty on Windows. Use Task Manager. |
| `ladder/eval_games_total` | Always 0, in all 249 rows | Evaluation battles played so far. The evaluation runner appends straight to the shared result log rather than through the pool's counter, so the counter never moves. `records.py` documents this at the line that emits it. A run can fit a real rating off 24 real evaluation battles while this row still reads 0. |
| `ladder/score_vs_noop`, `ladder/score_vs_random_legal` | Absent in new rows; exactly 0.5 in all 172 older rows that have them | Score against the two scripted benchmark opponents. The bot is never evaluated under its own name, so the pair these keys ask about has no games, and the lookup used to answer 0.5, which is also what a genuine dead heat looks like. Today the harness omits the key instead, which at least makes the gap visible. Making them real needs the live bot evaluated against the anchors every iteration, which costs time and has not been decided. |
| `time/critic_pass`, `time/gae`, `time/overlap_saved` | Hardcoded 0.0 | Three timing slots that were reserved and never wired up. See `coordinator.py` lines 1769, 1770 and 1774: they are literal zeros. Whatever time those phases take is currently swept into `time/collection`, `time/update` and `time/residual`. |
| `time/codec` | 0.0 in every row on disk | Time spent packing observations in the workers. It reads `codec_ms` out of the rollout source's stats, and no shipped source reports it yet. |
| `env/reward_terminal_abs` | 0.0 in every row on disk | How much of the reward came from actually winning. A fix landed on 2026-09-22 (commit `c38dc82`) that files the win/loss term under the fixed name `terminal` rather than under its Python class name. Every row currently on disk predates that fix, so they all read 0.0. New runs should carry a real number. |

Some other rows are zero simply because nothing has happened yet, and those are fine:
`health/worker_restarts`, `health/nan_guard_trips`, `health/obs_codec_clipped`,
`ppo/lr_backoff_events`, `ladder/evictions`, `ladder/champion_step`. Zero there is the healthy
reading. `env/illegal_action_rate` and `health/samples_unused_frac` are zero **by construction**:
the schema gives both a healthy range of exactly 0 to 0, and a non-zero value is a bug report.

---

## How the alarms work

The harness watches 23 conditions, and you do not have to know them to start a run. They are
defined in `royalelearn/metrics/alarms.py`.

Each alarm is one yes-or-no question about one row, plus a patience and a severity.

- **Patience** is how many iterations in a row the condition has to hold before the alarm says
  anything. Most are 3 or 5. `ev_negative` is 50, because a critic being bad early is normal.
- **Severity `warn`** (16 of them) prints a line and appends to `alarms.jsonl`. The run keeps going.
- **Severity `halt`** (7 of them) does all of that, then writes a checkpoint and a diagnostic
  bundle into `bundles/`, then stops the run. `royalelearn train` exits with code 2.

The seven that stop a run are `illegal_actions`, `ratio_invariant`, `nonfinite`,
`buffer_overflow`, `clip_pinned`, `noop_collapse_severe` and `worker_failures_persistent`.

Two design choices are worth knowing about:

1. **A missing key never fires an alarm.** If a row does not carry a key an alarm reads, the alarm
   is simply not asked. An iteration where no battle finished cannot trip the draw-rate alarm.
2. **The batch is judged before the update learns from it.** As of 2026-09-22, the invariant checks
   run before the PPO update, not after (commit `0839758`). A batch that fails them is refused
   before it can train anything.

You can see the whole table, with names and severities, without starting a run:

```
../.venv/Scripts/python -c "
from royalelearn.metrics.alarms import default_alarms
from royalelearn.config import AlarmConfig
for a in default_alarms(AlarmConfig()):
    print(f'{a.severity:5} patience={a.patience:<3} {a.name}')
"
```

Thresholds, patiences and severities all live in the `alarms` block of your config, and any alarm
can be turned off by name via `alarms.disabled`. Changing them cannot change any number the run
produces, which is why they are deliberately left out of the run identity.

**What each individual alarm means and what to do when it fires is [running.md](running.md), not
this page.**

---

## Sending the numbers somewhere else

Two ways, and the first one is almost always the right one.

### Read the file from another process

`metrics.jsonl` is flushed after every row, so a second terminal can follow a live run without
touching the harness at all. No code change, no risk of your plotting bug taking down a nine-hour
training run.

```
../.venv/Scripts/python -c "
import json, time, sys
path = sys.argv[1]
with open(path, encoding='utf-8') as handle:
    while True:
        line = handle.readline()
        if not line:
            time.sleep(2); continue
        r = json.loads(line)
        print(r['run/iteration'], r.get('policy/cards_per_match'), r.get('ppo/explained_variance'))
" runs/royalelearn-<id>/metrics.jsonl
```

### Write your own sink

A sink is a small class. The base is `MetricsSink` in `royalelearn/api/metrics.py`, and it has
three methods you must write and four you can ignore:

| Method | Required | Called with |
| --- | --- | --- |
| `open(identity, config_json, run_dir)` | yes | Once, at the start of the run. |
| `write(row)` | yes | One flat dict per iteration. |
| `close()` | yes | Once, at the end. |
| `write_episodes(rows)` | no | Finished battles. Default: ignore. |
| `write_alarms(alarms)` | no | Alarms that fired. Default: ignore. |
| `write_artifact(name, path)` | no | A file produced this iteration. Default: ignore. |
| `save_checkpoint` / `load_checkpoint` | no for a stateless sink | Resume support. |

Two rules the base class states outright: do not change the row you were handed, and do not raise
on a key you do not recognise. Iterate the row rather than reaching for a fixed list of keys, and a
metric added tomorrow will flow through your sink without an edit.

Be aware of the trade: a sink that raises takes the iteration down with it
(`CompositeSink` in `sinks.py` does not swallow exceptions). That is deliberate, because a metrics
sink that silently stops writing is how a run becomes unattributable. It does mean your HTTP
request to a flaky dashboard needs its own try block.

Wiring it in currently means a small edit. `build_sinks` in `royalelearn/metrics/sinks.py` matches
on the sink `kind` string and raises `ValueError` on one it does not know, so a config cannot name
a class the package has never heard of. Adding a branch there is three lines. There is no plugin
system for this today.

### The live viewer

`ViserSink` is the third default sink. It sends a fixed 21-row status panel to RoyaleViser over a
UDP socket, and it sends nothing until a viewer sends a hello, so an unattached run pays nothing.
The mapping from metric key to panel row is one dict, `FIELD_SOURCES` in
`royalelearn/metrics/viser_sink.py`, so you can see exactly which number each panel row is.

The train session measured the cost of attaching a viewer to a running job on 2026-09-22: 18
iterations alternating 3 attached and 3 detached, three times over, giving 28.13 plus or minus
1.25 seconds attached against 28.33 plus or minus 1.65 seconds detached. That is no detectable
cost. This figure was reported by the train session and has not been independently checked, so
treat it as a strong hint rather than a settled result.

---

## One measured thing about memory

If you have a 4 GB graphics card, leave `ppo.minibatch_size` at 256. Measured on 2026-09-22 on the
RTX 3050 4 GB: minibatch 512 does not fit and spills into system memory, and an iteration at
minibatch 512 on the laptop profile took 518 and 544 seconds, against roughly half a minute at
256 on the smaller diagnostic geometry. The run does not crash when this happens. It just gets
several times slower and says nothing, which is why there is now a check before the run starts and
a `vram_spilling` warning during it. The relevant rows are `health/vram_available_mb` and
`health/vram_needed_mb`.

---

## Where to go next

- [running.md](running.md) for each alarm and what to do about it.
- [harness-spec.md](harness-spec.md) for the full specification, including the alarm table this
  page summarises.
- [design.md](design.md) for why the harness is shaped the way it is.
- `royalelearn/metrics/schema.py` for all 129 fixed keys and 10 key families, each with a unit, a
  sentence and a healthy range. If this page and that file ever disagree, the file is right.
