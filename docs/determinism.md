# Will I get the same run twice?

Usually yes, and this page tells you exactly when.

Reproducibility is the difference between "my change helped" and "my change might have
helped". If two runs of the same settings can end up anywhere, you cannot tell a real
improvement from luck. So a run here is a function of its settings, and it records enough
about itself that you can prove two runs were the same experiment.

This page covers what is reproducible, what is not, and the commands to check.

---

## The short answer

| you do this | do you get the same run? |
|---|---|
| Run the same config twice on the same machine | Yes, every number, iteration by iteration |
| Run it on a friend's machine with a different GPU | The same battles start. The learning can drift |
| Stop it and resume from a checkpoint | Yes from an episode boundary, not from mid-battle |
| Change the card list, the engine, or the worker count | No, and the harness tells you it is a new experiment |
| Set `determinism.tier = "throughput"` | No, and that is the point of the setting |

The default is the reproducible one. You do not have to turn anything on.

---

## Two different questions

People say "the same run" and mean one of two things. They come apart, so it is worth
separating them before anything else.

**The same battles.** Every battle starts from the same opening position. Battle 7 plays its
third match against the same opponent in both runs, from the same seed. Nothing about the
game itself moved.

**The same learning curve.** Every number the learner produced is identical. Same gradients,
same weights after iteration 12, same win rate on the chart.

The first is cheap and nearly unbreakable. The second needs the floating point arithmetic on
your GPU to come out the same way twice, which is a stronger ask.

The gap between them matters in practice. If you move a run to a different GPU, the battles
still start from the same positions, but the network's forward pass can produce very slightly
different numbers, the policy picks a different card once, and from that point the two runs
are playing different games. Nothing is broken. They are just two different runs now.

---

## The two tiers

A tier is how much reproducibility you are asking for. It lives in your config under
`determinism.tier`, and it is one of two values, defined in
[`royalelearn/determinism.py`](../royalelearn/determinism.py).

**`run_exact`** is the default (`royalelearn/config.py`, `DeterminismConfig.tier`). It pins
the GPU to deterministic kernels, turns off the autotuner that picks a different algorithm
depending on how busy the card is, turns off tf32 (a faster, lower precision matrix mode that
is not bit reproducible across differently shaped inputs), and pins torch to one CPU thread.
With this, two runs of the same config on the same machine agree number for number.

**`throughput`** drops all of that. The environment side stays reproducible. The learner is
allowed to use whatever kernel is fastest, so the curve will be close but not identical.

The tier is recorded in the run's identity, so a run can never quietly be one tier while you
believe it was the other.

### What run_exact costs

The spec estimates 10 to 20 percent of throughput, and that estimate has not been measured
here. What has been measured is where the time goes, which tells you what the estimate is a
percentage of.

Measured 2026-09-22, on a 4 core laptop with an RTX 3050 4 GB and 7.8 GB of RAM, sharing the
machine with six other jobs. This was a small diagnostic geometry, 2 workers with 24 battles
each and 8,192 timesteps an iteration, not the shipped laptop profile:

- 30 to 35 seconds an iteration
- of which the PPO update took 25 seconds and collecting the battles took 5 to 8 seconds
- the update was 71 to 83 percent of every iteration, across 7 iterations

("PPO" is the learning algorithm, Proximal Policy Optimization. The "update" is the part
where it adjusts the network's weights from the battles it just collected. "Collection" is
playing the battles.)

So on this machine the slow part is the GPU, not the simulator. If you turn determinism off
to go faster, the saving comes off the part that is already most of the clock. Whether that
is worth losing reproducibility is your call, but at least you can see what you are trading.

---

## How a seed reaches a battle

There is one seed in your config, `master_seed`. Everything else is derived from it: take the
name of what wants random numbers, for example `env/battle/7/ordinal/3`, hash the name with
blake2b, and use your seed plus that hash to build a PCG64 generator. Read the whole thing in
[`royalelearn/seeding.py`](../royalelearn/seeding.py). It is about 150 lines, most of it
comments.

The unusual part is that streams are addressed **by name**, not by order. The common way to
do this is to make a root generator and spawn children from it as the code asks. That works
until you add a diagnostic that draws one number, at which point every stream after it
shifts, and your policy plays different games for no reason you can point at.

Here a stream is a pure function of its name. Adding a new consumer somewhere else in the
code cannot move an existing one. There is a test that asserts exactly this
(`tests/test_seeding.py::test_a_new_stream_does_not_move_an_existing_one`).

### The whole namespace

Fifteen names, counted 2026-09-22 with
`python -c "from royalelearn import seeding; print(len(seeding.STREAMS))"`. This is the list
in the order `seeding.STREAMS` holds it.

| name | what draws from it |
|---|---|
| `env/worker/{w}/shard/{s}/gen/{g}` | the one environment reset of a shard, once per respawn |
| `env/battle/{b}/ordinal/{k}` | one episode of one battle, so a resume can start the next one |
| `env/stagger/worker/{w}/shard/{s}/gen/{g}` | the warm-up that spreads a shard's episode endings apart |
| `match/battle/{b}/ordinal/{k}` | who this battle plays this episode |
| `act/iteration/{i}/cycle/{t}` | the random numbers that pick an action |
| `scripted/worker/{w}/slot/{r}/gen/{g}` | one scripted (hand written, not learned) opponent |
| `ppo/minibatch/iteration/{i}/epoch/{e}` | the shuffle before a weight update |
| `eval/seed_set` | the fixed set of seeds every evaluation uses |
| `eval/match/{comparison}/{seed_index}/{side}` | one evaluation battle |
| `eval/bootstrap/{comparison}` | the resampling behind an evaluation's error bars |
| `preflight/env` | the environment the start-up checks build and throw away |
| `preflight/sample` | the random legal plays the start-up checks make |
| `torch/init` | the network's starting weights |
| `torch/global` | `torch.manual_seed` at start-up |
| `torch/cuda` | `torch.cuda.manual_seed_all` at start-up |

Two details worth knowing:

**An episode is addressed by battle and ordinal, not by how many came before it.** Battle 7's
fourth match has a name of its own. That is what lets a stopped run pick up at the right
match instead of replaying three matches to get there. See `_episode_seed` in
`royalelearn/rollout/inline.py`.

**Action sampling does not use the GPU's random numbers.** Each cycle draws one vector of
uniform random numbers with numpy, and each seat indexes into it by its slot. So which action
a policy takes depends on the run seed, the cycle and the slot, and not on how many rows
happened to share a batch on the GPU. See `Inference.uniforms` in
`royalelearn/learn/inference.py` and the module comment in `royalelearn/learn/distribution.py`.

There is a test that scans every source file in the package for a call to a global random
number generator and fails if it finds one (`tests/test_no_global_rng.py`). That is a source
scan rather than a runtime check, on the reasoning that a bad call that fires once every
thousand iterations would never show up in a fixture.

---

## What breaks reproducibility

Each of these genuinely changes the numbers. Most of them are caught for you.

**A different card list or a different engine build.** The card table's hash, the engine's own
build and calibration hashes, and a hash of the compiled engine file are part of the run's
identity. Rebuild the engine, from different data or from different code, and the identity
changes, so you have a different experiment. Do not rebuild it under a running job: a worker that
restarts on the new build is refused.

**A different number of workers or battles per worker.** The rectangle of seats is laid out
worker by worker, and battle 7 is a different battle when the layout changes. Worker count,
battles per worker, shards per worker, cycles and total seats are all in the identity
(`rollout_digest` in `royalelearn/identity.py`).

**A different GPU, or a different torch or CUDA build.** `run_exact` promises bit identical
learning **within a device class**. Both the device description and the torch version are
recorded in the identity so that you can see when they moved.

**`tier = "throughput"`.** Stated above. It is recorded, so it is never a surprise.

**A worker that died and restarted.** If a worker process crashes, the harness can replace it
(`rollout.restart_failed_workers`). The replacement resets its battles from a new seed, at
the next "generation", so from that point its battles are different ones. Check
`health/worker_restarts` in your metrics before you compare two runs. It counts how many
times this happened.

**Resuming from a checkpoint taken mid-battle.** Covered in its own section below.

**`rollout.overlap`.** This one is a trap. With overlap on, a cycle's actions come from the
previous iteration's weights, so a run collects different data. But `overlap` is deliberately
**not** in the identity (`EXCLUDED_FROM_IDENTITY` in `royalelearn/identity.py`), on the
reasoning that it schedules the collection rather than configures it. A resume with it
flipped prints the difference and continues. If you flip it, you are comparing two things
that are not the same.

**Changing the minibatch size for memory reasons.** The minibatch is how many battle steps
the GPU chews at once. It looks like a pure memory knob and it is in the identity anyway,
because it changes the arithmetic. The default is 256 (`royalelearn/config.py`,
`PPOConfig.minibatch_size`). Changing it to fit your card is a real fix. It is also a new
experiment.

### What does NOT break it

**Running in one process instead of three.** `python -m royalelearn train --inline` runs the
battles in the learner's own process. `rollout.source` is not in the identity, and
`tests/test_rollout_farm.py` collects one iteration both ways and asserts the two rectangles
of observations are identical byte for byte, along with the rewards, flags, ticks and episode
records. That test runs 3 workers with 2 battles each on the MockEngine, so it is evidence
for the mechanism rather than for your exact configuration.

**Checkpoint interval, metric sinks, alarm thresholds, the run's name.** All recorded, none
in the identity. An alarm can stop a run. It cannot change a number.

---

## Resuming: what comes back and what does not

A resume restores the learner completely. In a fresh process, the weights, both optimizers'
internal state, the return scaler and every schedule position come back byte for byte. That
is checked by hashing all of it and comparing the hash across the process boundary, which you
can run yourself (see below).

Each battle is also put back on the episode it was about to play, not on episode zero. That
is what the `env/battle/{b}/ordinal/{k}` name buys.

What does **not** come back is a match that was half played when the checkpoint was written.
Those transitions were already in the original run's buffer, and replaying them would count
them twice, so the resumed run starts the next match instead. Resuming inside an episode is
not offered.

The practical consequence: if your checkpoint landed mid-battle, the resumed run plays the
same battles as the original but at a different phase. Episode counts per iteration will
differ. The tests draw this line explicitly, one for each side of it, in
`tests/test_resume.py`:

- `test_a_resume_continues_the_original_row_for_row`, with episodes arranged to end on
  iteration boundaries
- `test_an_episode_in_flight_is_not_replayed`, which asserts the phase difference is still
  there, and says in its failure message to widen the guarantee if it ever stops being true

Both run on the MockEngine on CPU at a tiny size. They prove the machinery, not your GPU.

---

## The run's fingerprint

`royalelearn/identity.py` computes a `RunIdentity` at start-up and reduces it to sixteen hex
characters. Two runs with the same sixteen characters are the same experiment. Two runs with
different ones are not, whatever their curves look like.

It is written to `identity.json` in the run directory the moment the run is created, next to
`config.json`. Every metric row also carries the determinism tier.

What goes into it, and roughly why:

| in the identity | because |
|---|---|
| `master_seed` | it changes every byte after it |
| `determinism_tier` | it decides whether the numbers repeat at all |
| `device_kind`, `torch_version` | different kernels, different arithmetic |
| engine class, build hash, calibration hash, card list hash, compiled engine file hash | the game itself |
| observation, action and env spec hashes | what the policy sees and can do |
| `frame_stack` | how many past frames the network gets |
| network architecture hash | the model |
| PPO and advantage settings hash | the learning algorithm's knobs |
| rollout geometry hash | workers, battles, shards, cycles, seats |
| ladder hash | who the learner plays |
| both package versions and their git descriptions | the code |
| the Python source of any package your own components come from, by hash | your code, such as a reward in `mybot` |
| each config section from an installed package the run uses, with every file it names by content, and the package's version and commit | what the run learns from besides its own battles |

Recorded but deliberately **not** in it: the run name, the output directory, the timestep
limit, metric sinks, checkpoint settings, alarm thresholds, and the collection scheduling
knobs. The full list is `EXCLUDED_FROM_IDENTITY` in `royalelearn/identity.py`.

**Why this matters when you compare two runs later.** In three weeks you will have a folder
of runs and a chart where one line is above another. The identity is how you find out whether
those two lines are answering the same question. If they differ in a field, you are looking
at two experiments, and the difference in the chart may be that field rather than the change
you were testing.

A resume compares identities field by field and **refuses** when one moved, naming the field.
You can override that with `--allow-identity-drift`, which writes a `drift.json` listing every
difference and marks every later metric row as drifted. That flag exists so the refusal can be
absolute by default.

---

## Check it yourself

### What is this config, exactly

```
python -m royalelearn config --profile laptop -o laptop.json
python -m royalelearn identity --config laptop.json
```

Run 2026-09-22 on the laptop this was written on. Your git hashes and your GPU line will
differ:

```
royalelearn   0.1.0 (c6c2028-dirty)
royalegym     0.1.0 (6299ade-dirty)
torch         2.11.0+cu128 on cuda:NVIDIA GeForce RTX 3050 Laptop GPU:sm_86
master seed   20260921
determinism   run_exact
env           royalegym.rust_engine.RustEngine, dc1b8cb199176805
geometry      3 x 32 battles, 192 slots, 228 cycles
algo          41d1a2ba44bedb1b
ladder        f8256cfe45ba817b
config hash   81169890e7f1e8d8a098f28c8234314c1ab154da08607f2331a3fe3544a3296f
```

This is fast, needs no GPU work, and does not build an environment. It is the thing to paste
into an issue. The full run id needs a built environment and is printed by
`python -m royalelearn doctor`.

Note the `-dirty` suffix. It means the checkout had uncommitted changes, and it is in the
identity, so a run from a dirty tree is honestly marked as not reproducible from any commit.
`train` and `resume` refuse to start from one unless you pass `--allow-dirty`.

Your own code is in the identity too. If a component in your config comes from your own package
(a reward in `mybot.rewards`, say), the run records a hash of that package's Python source. Edit
the reward, or a helper it imports, and the next run is a different run with a different id, even
if you committed the edit. A resume of the old run is then refused by name, because it would carry
on training against an objective the checkpoint never saw.

### Do my two runs match

Every metric row carries `run/state_digest`, a hash over the weights, both optimizers'
internal state, the return scaler and the schedule positions. Comparing those tells you the
iteration at which two runs first diverged, instead of making you squint at a chart.

```
python -c "
import json, sys
def digests(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line)['run/state_digest'] for line in fh if line.strip()]
a, b = (digests(p + '/metrics.jsonl') for p in sys.argv[1:3])
for i, (x, y) in enumerate(zip(a, b), 1):
    if x != y:
        print('first difference at iteration', i); break
else:
    print('identical for', min(len(a), len(b)), 'iterations')
" runs/first runs/second
```

Two identical runs will still differ in their wall clock columns, the ones starting `time/`
and `throughput/` and `run/wall_seconds`. That is expected. The test that compares two runs
row for row excludes exactly those columns and nothing else.

### Does a resume really resume

```
python -m royalelearn verify-resume --config laptop.json --iterations 6 --split 3
```

This runs three iterations, checkpoints, starts a **second process** that loads the
checkpoint, and compares the learner's whole state hash across the process boundary. The second
process is a `resume`, so uncommitted code is refused before the first half starts; pass
`--allow-dirty` if the tree is dirty on purpose. A second
process is the only version of this worth running. One that works inside the interpreter that
wrote the checkpoint skips the machinery being tested.

It does real training, so it takes real time. That time has not been measured here.

### Is this battle the battle that happened

```
python -m royalelearn replay --run runs/mine --episode 0/0/7/3
```

That is worker 0, shard 0, battle 7, episode 3. It re-simulates the episode from its seed and
verifies the trace against a fresh engine. An empty divergence list is the whole promise.

### The cheap tests

```
python -m pytest tests/test_seeding.py tests/test_no_global_rng.py -q
```

19 tests in `tests/test_seeding.py`, collected 2026-09-22. Several of them pin exact drawn
values, so a change to how seeds are derived is a failing test rather than a run that quietly
stops matching the one before it.

`tests/test_resume.py` has 5 tests and is marked slow, so the default `pytest` run skips it.
Run it with `-m ""` when you want it, and expect it to take minutes.

---

## Where this is rough

Said plainly, because the rest of the page is only worth anything if this part is honest.

- **The row for row guarantees are tested on a toy engine on CPU.** `tests/test_resume.py`
  uses the MockEngine, one worker, two battles, a tiny network, float32, on CPU. It proves the
  plumbing. It does not prove that your GPU run repeats, and nobody has run that check on the
  real engine here yet.
- **The 10 to 20 percent cost of `run_exact` is an estimate, not a measurement.** Nothing on
  this machine has compared the two tiers head to head.
- **Cross machine reproducibility has not been tested.** The design says the environment side
  is bit identical anywhere, because seeds are hashes of names and the engine is integer
  arithmetic. Nobody has run the same config on two machines and diffed the result.
- **A worker restart is recorded, not compensated.** You get a counter telling you it
  happened. You do not get the original battles back.
- **Mid-episode resume is not offered and is not planned here.** If your checkpoints keep
  landing mid-battle, the fix today is to compare from a later iteration, not to expect the
  phase to match.

---

## Going deeper

- [`docs/harness-spec.md`](harness-spec.md) section 5 is the full specification, including the
  three tiers and the enforcement rules. Where it disagrees with the code, the code is right.
- [`docs/design.md`](design.md) is the design rationale.
- [`royalelearn/seeding.py`](../royalelearn/seeding.py) is the seeding namespace, and it is
  readable in one sitting.
- [`royalelearn/identity.py`](../royalelearn/identity.py) is what a run IS, in one file.
