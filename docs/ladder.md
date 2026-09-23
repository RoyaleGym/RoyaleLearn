# The ladder: how the bot is measured against past versions of itself

You get one honest answer out of this page: whether today's bot is actually better than
last hour's bot, and how much of that answer you should believe.

Everything here lives under `royalelearn/ladder/`. You do not have to configure any of it
to start a run. You do have to read it before you trust a number.

The deep version is [`docs/harness-spec.md`](harness-spec.md) section 11, and the reasoning
behind the whole harness is in [`docs/design.md`](design.md). This page is the short one.

## Why you cannot judge the bot by its reward

Your reward function is a small piece of Python that scores what just happened in a battle.
The training loop tries to make the reward go up. So the reward going up tells you the loop
is working. It does not tell you the bot is good.

Two reasons, and both bite in practice.

**The reward is zero sum.** The bot plays itself. Both seats of a battle are in the same
batch of training data. Every shipped reward gives one seat exactly what it takes from the
other, so the average reward across the batch is zero by construction and stays zero forever.
It is a flat line no matter how good or bad the bot gets. The comment in
`royalelearn/metrics/records.py` around line 170 explains why two numbers are published per
reward term instead of one, for exactly this reason.

**The opponent moves.** Even with a reward that was not zero sum, the bot's win rate in
training is a win rate against itself. A bot that gets worse in a way that its own copy
cannot punish still wins half its battles. Half is what self-play always reports, at every
skill level, including none.

So the only way to ask "is it better" is to make it play something that does not move. That
is what the ladder is.

## The pieces

| File | What it holds |
| --- | --- |
| `ladder/snapshots.py` | The archive of frozen copies of the bot, on disk |
| `ladder/pool.py` | Who is in the ladder, who the champion is |
| `ladder/matchmaker.py` | Who each battle plays |
| `ladder/evaluate.py` | Playing a measured set of battles between two of them |
| `ladder/gate.py` | Whether a new copy is allowed to join |
| `ladder/rating.py` | Turning a pile of results into a number per player |
| `ladder/eviction.py` | Which copies the matchmaker stops drawing, to keep it cheap |
| `ladder/results.py` | The append-only log of every measured battle |

In your run folder they write to `ladder/games.jsonl`, `ladder/gates/*.json`,
`ladder/ratings/*.json` and `ladder/eval_seeds.json`.

## The frozen pool, and why old versions are kept

A **snapshot** is a frozen copy of the bot's decision-making network, saved to disk and never
trained again. The **pool** is the set of snapshots the bot still plays against.

The code takes a snapshot at a fixed cadence measured in game steps, not in training
iterations. `royalelearn/coordinator.py` does this in `_maybe_gate`. The shipped values are
in `examples/configs/`:

| Profile | `candidate_every_env_steps` |
| --- | --- |
| `laptop.json` | 4,000,000 |
| `workstation.json` | 2,000,000 |
| `smoke.json` | 32 |

Steps rather than iterations, because an iteration changes size when you change the batch
size or the number of workers, and then a cadence in iterations quietly means something else.

Old snapshots are kept rather than replaced for one reason. A bot that only ever plays its
most recent self forgets how to beat the things it already beat. It drifts into a corner of
the game where its recent ancestors are weak, wins there, and loses to something it crushed
an hour ago. Keeping the weak old versions in the pool is what stops that.

Snapshots are kept forever in the archive. What is bounded is how many the matchmaker draws
from, because that draw runs in Python once per battle. `pool_working_size` in
`royalelearn/config.py` defaults to 48. When the sampler grows past that,
`ladder/eviction.py` drops members, and it does not drop the oldest. It keeps a spread across
the rating range, plus the two scripted anchors, the run's very first snapshot, and every
snapshot that was ever champion. Dropping the oldest would throw away exactly the weak
opponents the pool exists to preserve. Evicted snapshots leave the sampler but stay in the
archive and in the result log, so their results still count toward everyone's rating.

## Who the bot plays in a given battle

Every battle draws its opponent from a three-way mixture. `LadderConfig.mix` in
`royalelearn/config.py` defaults to `(0.50, 0.35, 0.15)`, and all three shipped configs use
it unchanged:

- **50% mirror.** The bot plays its live self. Both seats are trainable, which is where most
  of the training data comes from.
- **35% pool.** The bot plays one frozen snapshot.
- **15% scripted.** The bot plays one of the two simple hard-coded opponents below.

Before the first snapshot exists there is nothing in the pool, so a pool draw falls back to
scripted (`matchmaker.py`, in `assign`). That keeps the number of trainable rows per battle
exactly what the mixture says it is, which the loop checks every iteration.

At most `max_resident_opponents` snapshots are loaded on the GPU at once. That defaults to
**2**. Those two are redrawn only when the pool changes, never in the middle of an iteration.

### One thing the config says and the code does not do yet

The config asks for **PFSP** weighting. PFSP means "prioritised fictitious self-play", and in
plain words it means picking opponents that still beat you rather than picking uniformly at
random, because those are the ones there is something to learn from. `pfsp_weighting`
defaults to `"hard"`.

Right now that weighting comes out uniform. The weight of an opponent depends on how well
the *live* bot is predicted to score against it, and the live bot has no fitted rating,
because it is never measured under its own name (see the broken rows section below). With no
rating for the live bot, `matchmaker._predicted_score` returns 0.5 for every candidate, and
a constant prediction gives every candidate the same weight.

You can see it in one command:

```
python -c "from royalelearn.config import LadderConfig; from royalelearn.api.ladder import RatingTable; from royalelearn.ladder.matchmaker import MixMatchmaker; m=MixMatchmaker(1,LadderConfig()); t=RatingTable(rating={'snap:v0':0.0,'snap:v1':200.0,'snap:v2':400.0,'snap:v3':-200.0},se={},anchor='scripted:noop',draw_nu=None,n_games={},transitivity_residual=0.0,converged=True,iterations=3); print(m.weights(('snap:v0','snap:v1','snap:v2','snap:v3'),t,'hard'))"
```

Measured 2026-09-22, that prints `[0.25 0.25 0.25 0.25]`. Add a `'learner'` entry to the
rating table and it prints `[0.082 0.233 0.630 0.054]`, which is the PFSP behaviour the config
describes. So the opponent draw is currently uniform over the resident snapshots. That is not
wrong, it is just not what the name suggests, and it has not been fixed yet.

## The two scripted anchors

An **anchor** is an opponent whose strength never changes. They matter because everything
else in the ladder is improving, so a rating measured only against other snapshots is a
rating on a scale that is itself sliding. `royalelearn/rollout/scripted.py` ships two:

- **`scripted:noop`** does nothing at all. It never plays a card.
- **`scripted:random_legal`** plays a legal random move, but only 10% of the time. The other
  90% it does nothing. `RANDOM_LEGAL_NOOP_PROB = 0.9` in that file. Uniform over every legal
  move would play a card the instant one was affordable, which is a strange thing to train
  against.

They exist before the first snapshot does, so the rating scale has a fixed point from the
first battle. `scripted:noop` is pinned at rating 0 and every other number is relative to it.

They are deliberately weak. If your bot is not beating `scripted:noop` comfortably, nothing
else on this page is worth reading yet.

## The gate: what a new snapshot has to clear to join

The **champion** is the current best snapshot. A **candidate** is the fresh snapshot just
taken. The **gate** is the test that decides whether the candidate joins the pool and whether
it takes the champion's place. It is in `ladder/gate.py`, class `WilsonGate`.

Three conditions. Defaults from `GateConfig` in `royalelearn/config.py`.

**1. It beats the champion.** 1,000 battles against the champion. A **score rate** is wins
plus half the draws, over battles played. The gate does not use the raw score rate. It uses
the bottom of a 95% confidence interval around it, which is the lowest score rate still
consistent with what was observed. That bottom has to reach 0.52. Check the threshold
yourself:

```
python -c "from royalelearn.ladder.rating import wilson_interval, elo_of_score; [print(p, round(wilson_interval(p,1000)[0],5)) for p in (0.550,0.551,0.552)]; print('elo at 0.552:', round(elo_of_score(0.552),1))"
```

Measured 2026-09-22: an observed 55.0% does not clear it, 55.1% does, and 55.2% is about
36 Elo of improvement. A threshold at 0.50 would admit anything merely not worse, and the
pool would fill with sideways moves.

**2. No regression against the anchors.** 200 battles against each anchor. The candidate has
to score within 2 percentage points of what the champion scores against the same anchor.
This catches the bot that found a blind spot its recent ancestors share, wins by exploiting
it, and has quietly forgotten how to play.

**3. No collapse against the wider pool.** 100 battles against each of 8 snapshots chosen to
span the rating range. The candidate's average score has to reach what the fitted model
predicts the champion would score against those same 8, less one standard error.

That is 1,000 + 400 + 800 = **2,200 battles per gate**.

The verdict is not a single yes or no:

| Conditions 1 and 2 | Condition 3 | Result |
| --- | --- | --- |
| pass | pass | Admitted to the pool, and becomes champion |
| pass | fail | Admitted to the pool, flagged `cycle`, does **not** become champion |
| fail | anything | Discarded, never retried |

The middle row is the interesting one. A candidate that beats the champion but falls apart
against the rest of the pool has found a rock-paper-scissors loop, not an improvement. It is
a useful opponent to keep, and `ladder/eviction.py` protects flagged snapshots when it prunes,
because they are the ones a single rating number describes worst.

A failed candidate is thrown away and not retried. Consecutive failures are a signal, not an
error: the `gate_starved` alarm fires at 5 in a row (`AlarmConfig.gate_failures`). Separately,
`floor_admit_every_env_steps` (50,000,000 in all three configs) admits a snapshot
unconditionally if nothing has passed in that long, so a plateau cannot starve the pool of
fresh opponents. A floor admission never promotes.

Every decision is written to `ladder/gates/<candidate>.json` with the numbers behind it. A
real one, trimmed, from a smoke-sized run on 2026-09-22:

```json
{"candidate":"snap:v2","champion":"snap:v0","admit":false,"conditions":{
  "beats_champion":{"passed":false,"n":2,"observed":0.5,"bound":0.0945,"reference":0.52},
  "anchors:scripted:noop":{"passed":true,"n":2,"observed":1.0,"bound":0.6119},
  "anchors:scripted:random_legal":{"passed":false,"n":2,"observed":0.0,"bound":0.1498}}}
```

You can re-run a gate from stored snapshots without touching the training run:

```
python -m royalelearn gate --run runs/<your-run> --candidate snap:v3
```

### How the battles are counted

Evaluation battles are kept separate from training battles on purpose
(`ladder/evaluate.py`). They record no training data. They use a full match with no step cap,
because a cut-short match is scored on crowns nobody has taken yet and almost every game comes
out a draw. And they use a set of seeds drawn once at the start of the run and reused forever,
so every player is measured on the same starting positions.

Each pairing plays each seed **twice, with the sides swapped**. Blue and red are not symmetric,
so playing both sides removes the side advantage exactly instead of averaging over it. The
unit of measurement is the seed, not the battle, which is why `champion_games: 1000` means 500
seeds and `eval_seed_count` defaults to 500. `ladder/paired_rho` in your metrics file says how
correlated a seed's two sides were, which is how much of a battle the starting position decided
rather than the two players.

## The rating, and what it is not

Two different numbers in the metrics file, and they are not interchangeable.

**`ladder/elo_readout`** is the dashboard figure. Elo is the chess-style rating where a
higher number means stronger and a 400-point gap means roughly a 10-to-1 favourite. This one
updates live as training battles finish, starting from 1200 (`EloReadout` in
`ladder/rating.py`, `k_factor = 32`). It depends on the order the results arrive in, which
under parallel workers is not reproducible. It is for watching. No decision reads it.

**The fitted rating** is the real one. It refits from scratch over the entire stored result
log every `refit_every_iterations` iterations, which defaults to 10. Same games in, same
numbers out, in any order, on any machine. It is a Bradley-Terry-Davidson fit, which means it
finds the one rating per player that best explains all the recorded wins, losses and draws at
once, with draws modelled explicitly rather than counted as half a win.

You can run it yourself on any finished run. It only reads the log, so it needs no GPU and no
environment:

```
python -m royalelearn rate --run runs/<your-run>
```

Real output, 2026-09-22, from a 10-iteration smoke-sized run with 24 evaluation battles in
total:

```
member                              rating      se           95% interval   games
scripted:random_legal                493.4   177.7   [   145.2,    841.6]       8
snap:v3                              303.9   180.8   [   -50.5,    658.3]       6
snap:v4                              303.9   180.8   [   -50.5,    658.3]       6
snap:v1                              167.0   170.0   [  -166.2,    500.1]       6
snap:v2                              167.0   170.0   [  -166.2,    500.1]       6
snap:v0                              129.6   160.2   [  -184.4,    443.7]       8
scripted:noop                          0.0     0.0   [     0.0,      0.0]       8
anchor scripted:noop at 0; transitivity residual 0.0000
```

Read that table as a warning, not as a result. The intervals are roughly 700 Elo wide,
because each player has 6 or 8 battles behind it. The `random_legal` anchor sitting on top is
noise. This is what the output looks like when there is not nearly enough evidence, and it is
the shape you will see from any short run.

**What the rating means:** one player is stronger than another *inside this pool, under this
deck protocol, on these seeds*.

**What it does not mean:** anything about real players, any in-game trophy count, or any
ladder rank. There is no bridge in this repository between these numbers and a human
opponent. A run that climbs 400 Elo has climbed 400 Elo against its own history. That is
genuine progress and it is also the only claim the number supports.

**When even the within-pool reading breaks:** `ladder/transitivity_residual` is the share of
well-measured pairs that a single number per player fails to explain. It is near zero on a
population where strength is a straight line. It gets large when A beats B beats C beats A,
which one number per player cannot describe at all. The `transitivity` alarm fires above
0.10 (`AlarmConfig.transitivity_residual`). Above that, treat the whole rating column as
unreliable rather than as a slightly noisy truth.

## Reading the ladder rows in your metrics file

Every iteration writes one flat row to `runs/<your-run>/metrics.jsonl`. To see just the
ladder part of the last row:

```
python -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]; print(json.dumps({k:v for k,v in rows[-1].items() if k.startswith('ladder/')},indent=1))" runs/<your-run>/metrics.jsonl
```

The rows that are worth your attention, built in `royalelearn/metrics/records.py`,
`ladder_fields`:

| Key | What it tells you |
| --- | --- |
| `ladder/champion_id` | Which snapshot the next candidate must beat |
| `ladder/pool_size` | Everything in the archive, including the 2 anchors |
| `ladder/sampler_size` | Snapshots the matchmaker can actually draw. Anchors are not counted here |
| `ladder/gate_attempts`, `ladder/gate_passes` | How many gates ran, how many admitted |
| `ladder/consecutive_gate_failures` | The plateau signal. 5 trips the `gate_starved` alarm |
| `ladder/gate_failed_condition` | Which of the three conditions failed, or `none` |
| `ladder/gate_observed_rate` | The candidate's raw score rate against the champion. Appears once a gate decision carries its champion condition |
| `ladder/gate_lower_bound` | The bottom of its interval, which is what the gate compares to 0.52. Same condition as the row above |
| `ladder/rating/<member>` | One pool member's fitted rating, only after the first refit |
| `ladder/rating_ci95_lo/<member>` and `..._hi/<member>` | That rating's interval. A wide one means you do not know yet |
| `ladder/transitivity_residual` | Above 0.10, stop trusting the rating column. Appears once a rating fit exists |
| `ladder/paired_rho` | How much the starting position decides, rather than the players. Appears once the paired-seed correlation has been computed |
| `ladder/draw_rate_eval` | Draw rate in evaluation battles |
| `ladder/gate_seconds_frac` | Share of wall clock spent gating rather than training. Appears on an iteration where a gate ran |
| `ladder/elo_readout` | The live dashboard Elo. Never a decision. Appears once a training game has been scored this run |

A row that is missing a key is not a bug. The harness uses absence to say "nothing to report"
rather than publishing a zero that reads like a measurement. So do not read the first rows of a
run and conclude the ladder is broken: most of the table above only starts once a gate has run or
a rating has been fitted, which takes a while. `royalelearn/metrics/schema.py` lists every key
that may be absent in `CONDITIONAL`, each with the condition that makes it appear. Print them for
yourself:

```
python -c "from royalelearn.metrics import schema; [print(k, '->', v) for k, v in schema.CONDITIONAL.items()]"
```

## Rows you should not trust today

Three of the ladder numbers do not mean what their names say. All three come from the same
root cause: **the live bot is never measured under its own name.** Every evaluation battle is
a snapshot against something else, because the gate hands snapshot ids to the evaluator. So
the id `learner` never appears in the evaluation results, and anything looked up under that
id comes back empty.

**`ladder/score_vs_noop` and `ladder/score_vs_random_legal` appear on probe iterations, and
nowhere else.** Until commit `b106aa1` on 2026-09-22 they were published as 0.5 in every row,
which was indistinguishable from a genuine even contest; then they were omitted, because the
pair they asked about had no games and never could. Measured 2026-09-22 across the metrics rows
in `runs/`: 127 rows written before that fix carry a flat 0.5. If you are reading an older run's
file, ignore those two columns entirely.

The decision that was open is now taken. Set `ladder.probe_every_iterations` and every
`ladder.probe_every_iterations` iterations the run plays the LIVE bot against each rung in
`ladder.probe_opponents` and publishes, for each of them:

| key | what it is |
|---|---|
| `ladder/score_vs/{rung}` | the live bot's score against that rung, 1.0 for a win |
| `ladder/score_vs_n/{rung}` | how many SEEDS it was measured over, each played from both sides |
| `ladder/score_vs_ci95_lo/{rung}`, `..._hi/{rung}` | the interval around it |

`ladder/score_vs_noop` and `ladder/score_vs_random_legal` are the same numbers under their old
names, for the two rungs that have always been the anchors. It is off by default, because it
costs battles nobody was paying for: at the shipped 40 battles a rung it is under 4% of one
gate. Read the interval before the score. At 20 seeds the instrument's own repeat spread against
`scripted:random_legal` is about 18 points with the policy unchanged, so a smaller move than that
is the instrument. `scripted:noop` does not act, so it does not have that noise.

**`ladder/rating_above_v0` was minus the first snapshot's rating, and is now absent instead.**
It is meant to be the live bot's rating above the run's first snapshot. The live bot has no
fitted rating, because every evaluation game is a snapshot against something, so the lookup
returned a default of 0.0 and the row published `0 - rating[v0]`. On one run that read -93.9,
-146.2, -191.7 and -129.6, which looks exactly like a bot falling behind its own opening
snapshot and is nothing of the kind. Since 2026-09-22 the key is published only when the fit
holds both sides (`royalelearn/metrics/records.py`, the guard above
`fields["ladder/rating_above_v0"]`), so it is simply absent until the live bot is evaluated
under its own id. If you are reading a run's file from before that, ignore the column. To see
the old shape for yourself:

```
python -c "import json,glob; [print(r.get('ladder/rating_above_v0'), r['ladder/rating/snap:v0']) for p in glob.glob('runs/*/metrics.jsonl') for r in map(json.loads, open(p,encoding='utf-8')) if 'ladder/rating/snap:v0' in r]"
```

Read `ladder/rating/<member>` directly instead. Those columns are correct.

**`ladder/eval_games_total` was structurally zero and has just been fixed.** It used to read a
counter that nothing ever incremented, because the only caller of the counting method hard
codes the training label. The same commit changed it to count the evaluation log directly. It
is still 0 in every metrics row on disk, including runs made after the fix, but that is now
the truthful answer: those runs never played an evaluation battle. Count the log yourself:

```
python -c "import json,sys; print(sum(1 for l in open(sys.argv[1],encoding='utf-8') if l.strip() and json.loads(l)['kind']=='eval'))" runs/<your-run>/ladder/games.jsonl
```

## What has actually been run, and what has not

This matters more than any of the above.

**No run on this machine has ever executed the shipped gate.** Measured 2026-09-22: of the 50
run folders under `runs/` with a metrics file, 24 ran at least one gate, and every one used
`champion_games: 2`, which is the smoke setting. The 1,000-battle gate in `laptop.json` and
`workstation.json` has never fired. The code path is the same and the suite covers it, but
the numbers it produces on a real run have not been seen.

**The ladder has never had more than a handful of members.** The largest pool in any run on
disk is 3, which is 2 anchors plus 1 snapshot. Eviction, the stratified sample in condition 3,
and the anti-forgetting argument for keeping old snapshots are all untested at the pool sizes
they were designed for.

**Nobody has measured what a gate costs on the laptop profile.** `ladder/gate_seconds_frac`
exists to tell you, and the schema flags it above 0.1. That is 2,200 full matches with no step
cap, so it will not be free, but there is no measured figure to quote.

What is measured, on a 4-core laptop with an RTX 3050 4 GB and 7.8 GB of RAM shared with six
other jobs, on 2026-09-22: in a diagnostic run at 2 workers by 24 battles and 8,192 timesteps
per iteration with minibatch 256, an iteration took 30 to 35 seconds. About 25 seconds of that
was the learning update and 5 to 8 seconds was collecting battles, with the update taking 71
to 83% of every iteration across 7 iterations. That geometry is not the laptop profile, which
uses 3 workers by 32 battles and 32,768 timesteps. It does say that on this machine training
time is GPU time rather than simulator time.

## Where to go next

- The dense version of all of this: [`docs/harness-spec.md`](harness-spec.md), section 11.
- Why the harness is shaped this way: [`docs/design.md`](design.md).
- A single head-to-head between two members, outside the training loop:
  `python -m royalelearn eval --run runs/<your-run> --a snap:v3 --b snap:v0 --seeds 50`
- The raw evidence behind every rating: `runs/<your-run>/ladder/games.jsonl`, one JSON line
  per battle.
