# What this harness needs from RoyaleGym, and why

This page is for two readers.

If you are trying to understand the project, it shows you where the seam between RoyaleLearn and
RoyaleGym is. You get a list of everything the trainer reaches across that seam for, in plain
words, plus the things it deliberately does itself.

If you are about to change something in RoyaleGym, it tells you what would break here and how
loudly. Some of what follows is a contract. Break it and this harness stops working, or worse,
keeps working on wrong numbers. The rest is convenience, and you can move it freely.

Neither reader needs the spec. [`docs/harness-spec.md`](harness-spec.md) is the dense version and
[`docs/design.md`](design.md) is the reasoning. This page only says what crosses the line.

## The one-sentence version

RoyaleLearn asks RoyaleGym for battles, and for an honest description of those battles. Everything
after that is the harness's own problem.

## See it for yourself

Nothing below is a claim you have to take on faith. This prints the whole surface the harness
reads its layout from, using the pure-Python engine so it works even if the Rust engine is not
built:

```
cd RoyaleLearn
../.venv/Scripts/python -c "
from royalegym import ClashParallelEnv, MockEngine
e = ClashParallelEnv(engine=MockEngine())
obs, info = e.reset(seed=0)
sp = e.observation_space('blue')
for k in sorted(sp.spaces): print(k, sp.spaces[k].shape, sp.spaces[k].dtype)
print('actions', e.action_space('blue').n)
print('vector fields', [f.key for f in e.obs_builder.vector_layout()])
print('planes', [n for n, static in e.obs_builder.spatial_layout()])
print('static planes', [n for n, static in e.obs_builder.spatial_layout() if static])
print('config keys', sorted(e.config()))
"
```

On 2026-09-22 that printed an observation of four keys: `spatial` at `(20, 32, 18)` float32,
`vector` at `(229,)` float32, `action_mask` at `(2305,)` int8, and `mask_planes` at
`(4, 32, 18)` int8. The vector is 20 named fields adding up to 229 numbers. Two of the 20 board
planes are declared static, `water` and `no_deploy`. That engine's card table has 16 cards, and
1235 of the 2305 actions were legal on the first step, which is the figure RoyaleGym's own
`docs/architecture.md` quotes.

Those numbers are properties of that engine and that observation builder, not of this harness.
That is the point of the next section.

## The harness reads the layout, it does not know it

Not one of the numbers above is written down in RoyaleLearn. There is no `229`, no `20`, no
`2305` in the source. The harness reads them off a freshly built environment at start-up, in
`royalelearn/rollout/envspec.py`, and everything downstream is computed from what it read.

That is why the same code runs against a 16-card catalogue and a 65-card one without a change,
and why a plane you add to the observation reaches the network without anyone editing this repo.

It is also why the surfaces below matter. If a layout method stops telling the truth, the harness
has no second source to catch it with.

## The surfaces, one at a time

### 1. Building an environment

The harness never constructs an env with a Python closure. It carries a JSON description of one
(`EnvFactorySpec`) and hands it to RoyaleGym's own `EnvFactory`, because a description survives a
process boundary and a checkpoint file, and a closure does not.

What it uses:

- `royalegym.env.EnvFactory(engine=, obs_builder=, action_parser=, reward_fn=, state_mutator=,
  decision_ms=, termination_cond=, truncation_cond=)`, each component given as a
  `(class, kwargs)` pair.
- `royalegym.env.ClashSelfPlayVecEnv(num_games, factory, viser=, autoreset_seed_fn=)`.
- `royalegym.done_condition.AnyCondition`, only when a run names more than one condition in a
  slot.

The default components a shipped run names are in `royalelearn/config.py` around line 467:
`royalegym.obs.SpatialObsBuilder`, `royalegym.action.TileActionParser`,
`royalegym.state_mutator.DefaultStateMutator`, `royalegym.done_condition.GameOverCondition` to
terminate and `royalegym.done_condition.StepLimitCondition` to truncate. The reward is the one
exception and is covered in section 5.

Only classes under `royalegym.` and `royalelearn.` may be named, unless a config adds a module to
`extra_component_modules`. That is a security rule, not a style rule: a component path is a string
from a file, and importing an arbitrary string is code execution.

Two arguments on the vec env earn their own mention.

`autoreset_seed_fn(game, ordinal) -> int` is what makes a resumed run possible at all. Without it
the battle a game plays next depends on how many battles it has already played, so a fresh worker
cannot arrive at episode 400 without playing 399 first. With it, an episode has a name. The
harness checkpoints `episode_ordinals` and restores it with `set_episode_ordinals` before
`reset()`.

`viser` decides which single game publishes frames for the viewer. It has to be decided once,
because the viewer listens on one fixed UDP port and a second env binding it raises. The harness
passes `viser="env"` for exactly one shard of one worker and `None` everywhere else
(`royalelearn/rollout/worker.py:144`).

### 2. The observation and its layout

The harness requires four keys, and refuses the run by name if any is missing
(`envspec.py:251`): `spatial`, `vector`, `action_mask`, `mask_planes`.

Beyond the keys it reads two declarations from the observation builder:

- `vector_layout()`, a list of named fields with sizes. The harness turns it into offsets and
  looks its own fields up by name. The policy head needs three of them: `own_hand_cards`,
  `own_hand_cost` and `own_hand_affordable` (`royalelearn/obs_layout.py:35-39`). Rename one and
  the run refuses at start-up with a message listing what the layout does declare. That is by
  design, and it is far better than the alternative, which is quietly slicing somebody else's
  numbers.
- `spatial_layout()`, one entry per board plane with a static flag. A static plane is stored once
  per run instead of once per transition. The harness trusts the declaration and never infers it.
  Guessing here would be a real bug: the tower planes are constant for as long as no tower falls,
  so a rule that promoted "unchanged on a thousand states" to "static" would freeze a tower at
  full health for the rest of the run and the policy would never see one die
  (`royalelearn/rollout/codec.py:26-29`).

It also reads the declared per-channel bounds on `observation_space["spatial"]`. A plane whose
declared bounds fit in a byte, and whose sampled values are whole numbers, is stored in one byte.
Everything else is stored as half precision. The `vector` is stored as half precision because the
builder bounds it in [0, 1], which the command above confirms.

### 3. The action space and the legality mask

This is the most load-bearing agreement in the project, because a transposition here is a policy
that plays a different card in a different place from the one it learned to.

The harness assumes the action space is `1 + hand_size * tiles_y * tiles_x`, index 0 is the no-op,
and index `1 + slot * tiles_y * tiles_x + ty * tiles_x + tx` plays that hand slot on that tile.
It does not take this on trust. At start-up it calls `parser.encode(slot, x, y)` for every single
non-no-op action and compares it against its own arithmetic
(`royalelearn/rollout/preflight.py:402`). On the shipped parser that is 2304 comparisons.

Three more mask facts the harness depends on:

- `mask[0]` is always set. RoyaleGym sets the no-op unconditionally, including after game over.
  The harness asserts it on every batch during the first iterations of a run, and its failure
  message names the RoyaleGym file it came from (`royalelearn/learn/distribution.py:51`). A row
  that arrives without the no-op did not come from the environment.
- `mask_planes` is exactly `action_mask[1:]` reshaped. The harness never stores the planes. It
  stores the bit-packed mask once and reshapes it back on the GPU, which saves about a sixth of
  every stored row. Start-up checks the equality on sampled states.
- The mask agrees with the engine. RoyaleGym ships
  `royalegym.action.mask_disagreements(engine, parser, state, team)` and this harness runs it
  exhaustively for both seats at start-up. The check already existed in RoyaleGym and nobody ran
  it during training, which is why it is a gate here. A policy trained against a wrong mask is
  worthless.

`vec.action_masks()` is never called. The mask is already in the observation, and restacking the
whole batch a second time costs a copy for nothing.

### 4. The step protocol and the episode statistics

The harness is written against SAME_STEP autoreset. When a battle ends, the observation that comes
back is already the next battle's first one, and the ended battle's final observation and info are
under `infos["final_obs"]` and `infos["final_info"]`, masked by `infos["_final_obs"]`.

The eight terminal scalars under `EPISODE_STAT_KEYS` are copied straight out of `final_info` into
the harness's own `EpisodeRecord` (`royalelearn/api/rollout.py:237-247`): `episode_steps`,
`episode_ticks`, `own_crowns`, `enemy_crowns`, `own_tower_hp_frac`, `enemy_tower_hp_frac`,
`elixir_leak_steps` and `elixir_count_exact`. `outcome` and `winner` come from the same place
(`royalelearn/rollout/inline.py:674`).

They are copied rather than recomputed on purpose. The environment already knows how the episode
went, and a second implementation of the same summary is a second answer that can disagree.

`info["tick"]` and `info["deploy_status"]` are used per step. `deploy_status` is how the harness
counts commands the engine refused, which should be zero when the mask is right.

### 5. The reward base class

RoyaleLearn uses RoyaleGym's `RewardFunction` and `CombinedReward` as base classes and ships four
reward terms of its own on top (`royalelearn/rewards.py`). Nothing in RoyaleGym is modified.

What it needs from the base class is small: `get_reward(team, prev, state, results)`,
`bind(engine)`, `reset(state)`, `config()`, and `CombinedReward.terms_for(team)` for the per-term
breakdown the metrics use.

Every shipped config now names `royalelearn.rewards.default_potential_reward`, not RoyaleGym's
`default_reward`. That changed in commit `23d971c` on 2026-09-22. Before it, every real iteration
trained a different objective from the one the design specified, which is exactly the kind of
mismatch this page exists to make visible. `rewards.py` gives the full argument. The short version
is that RoyaleGym's elixir-trade term pays a player who never commits a card, and its tower term
is one discount factor away from being policy-invariant.

This is a place where the harness disagrees with the library and says so, rather than a place
where it asks the library to change.

### 6. The state mutators

A mutator describes the starting position. The harness names one in its config and otherwise does
not touch this surface. The shipped runs use `DefaultStateMutator` with no arguments, which deals
each side a random eight cards per episode.

Curriculum work lives here rather than in the trainer. RoyaleGym's `WeightedStateMutator` mixes
fresh battles, damaged mid-games and saved snapshots by weight. Nothing in RoyaleLearn anneals
those weights today. If you are looking for an obvious unfinished edge, that is one.

### 7. The engine config stamps

`ClashParallelEnv.config()` returns a JSON-able description of the environment. On 2026-09-22 it
has 14 keys. The harness stores it in the run identity and prints part of it at start-up.

Three of those keys carry the most weight:

| Key | What it pins |
|---|---|
| `calibration_digest` | RoyaleSim's calibration file as it is on disk |
| `build_digest` | the copy of it compiled into the engine, when there is one |
| `reveal` | whether the observation was built with hidden information opened |

`reveal` is the one a reader should care about most. RoyaleGym's default observation hides the
opponent's hand and counts their elixir the way a person would. Turning a `Reveal` on adds slots
and channels rather than filling zeroed ones, so a fair observation and a cheating one are not
even the same width. Because `config()` records it, a checkpoint can say which one produced the
policy. Nothing about a weights file alone would show that.

The harness adds a digest of its own over the card catalogue's names, ids, elixir costs and
placement kinds (`royalelearn/identity.py:150`), so a run identity notices a card table change.

### 8. The trace, and the frames the viewer draws

Both are RoyaleGym's formats. The harness produces and consumes them and defines neither.

- `royalegym.replay.ReplayRecorder`, assigned to `env.recorder`, records a battle. `save_trace`
  writes it. `verify_trace` re-simulates it on a fresh engine and returns the list of divergences,
  which should be empty. The harness uses this to bundle one offending episode beside a failing
  iteration's metrics (`royalelearn/metrics/bundle.py`).
- `royalegym.viser.ViserPublisher` sends one UDP frame per env step, and only while a viewer's
  heartbeat is fresh. An env nobody is watching pays one `if` per step.

On the cost of watching: the train session reported on 2026-09-22 that attaching the viewer to a
running job cost nothing they could detect, 28.13 plus or minus 1.25 seconds an iteration attached
against 28.33 plus or minus 1.65 seconds detached, over 18 iterations alternating three attached
and three detached. That is their measurement, reported here, and nobody has checked it
independently.

### 9. The opponent pool and the rating arithmetic

The harness uses `royalegym.selfplay` for the parts that are pure bookkeeping over ids: the
`OpponentPool` registry and its persistence, `expected_score` for the Elo update, and the
`NoopOpponent` and `RandomLegalOpponent` scripted seats that a run trains against before it has
any snapshots of its own.

It deliberately does not use two of the pool's methods. Both reasons are in
`royalelearn/ladder/pool.py:7-13` and both are about measurement, not taste.

## Contract or convenience

| Surface | Change it and | Kind |
|---|---|---|
| Action index arithmetic and `Discrete(n)` width | start-up refuses, loudly, with the mismatching index | contract |
| `mask[0]` set unconditionally | the update asserts and names the rows | contract |
| `mask_planes == action_mask[1:]` reshaped | start-up refuses; the codec stores the mask once | contract |
| The four observation keys | start-up refuses by name | contract |
| `vector_layout()` field names for the hand | start-up refuses and lists what you do declare | contract |
| `spatial_layout()` static flags | silently wrong stored rows if the flag lies | contract, and the scariest one |
| SAME_STEP autoreset with `final_obs` and `final_info` | episodes stop being recorded | contract |
| `EPISODE_STAT_KEYS` | the terminal scalars stop arriving | contract |
| `config()` keys used by the identity | runs stop being comparable to older ones | contract |
| Number of planes, vector width, card count | nothing. All read at start-up | convenience |
| Which planes are static | nothing, as long as the flag is honest | convenience |
| Reward term set and weights | nothing. The harness ships its own composition | convenience |
| `OpponentPool` result and eviction behaviour | nothing. The harness does not call those two methods | convenience |
| Engine choice, mutator, done conditions | nothing. They are config strings | convenience |

The row to read twice is the static flag. Every other contract fails at start-up, in the first
five seconds, with a message naming the thing that moved. A dishonest static flag fails silently,
for nine hours, and produces a policy trained on a board it never saw change.

## What the harness does itself, and why

Not everything here is an ask. Several things RoyaleGym could plausibly own are held on this side
on purpose.

**Its own reward composition.** Section 5. The shipped terms are potentials, so shaping cannot
change which policy is optimal.

**Its own result log and rating.** RoyaleGym's pool stores `wins: float` with a draw adding a
half, which makes five wins and five losses indistinguishable from ten draws even though the two
carry completely different variance. `royalelearn/ladder/results.py` logs one line per battle and
counts draws separately.

**Its own eviction.** RoyaleGym's pool drops the oldest snapshot when it is full. The oldest
snapshots are exactly the weak, diverse opponents that stop a policy forgetting how to beat them.
`royalelearn/ladder/eviction.py` keeps a stratified sample across the rating range instead.

**Its own byte layout across the process boundary.** Nothing is pickled on the hot path. Both
sides compute the same offsets from the same environment description, so a disagreement about a
row is a refused attach instead of two processes reading different bytes
(`royalelearn/rollout/layout.py`).

**Its own start-up gates.** Several of them check RoyaleGym rather than RoyaleLearn. That is not a
comment on RoyaleGym's tests. It is that a check which runs in a test suite does not run on your
machine, with your card table, on the morning of your run.

## The asks that are open

Filed against RoyaleGym, each with what it costs here today. None of them blocks a run. The
harness works against the surface it has.

**1. Default observations and the mask computed in Rust.** This is RoyaleGym's own open item and
it is already on their list. Today, building both players' observations and legal-move lists in
Python is most of the time an env step takes. RoyaleGym measured 826 env steps a second on the
Rust engine on 2026-09-21 against roughly 32,000 engine ticks a second when the engine is stepped
directly, and their `docs/architecture.md` names Python as the gap.

Worth saying plainly: on the machine this harness was developed on, that is not currently the
bottleneck. A diagnostic run on 2026-09-22 at 2 workers by 24 battles, 8,192 timesteps an
iteration and minibatch 256 took 30 to 35 seconds an iteration, of which the PPO update was about
25 seconds and collecting the battles was 5 to 8. The update was 71 to 83 percent of every
iteration across 7 iterations. That geometry is smaller than the shipped laptop profile, which is
3 workers by 32 battles and 32,768 timesteps an iteration, so do not carry those seconds across.
The conclusion that survives is that this ask helps machines with more GPU than that one, and
frees cores on the ones without.

**2. `ClashSelfPlayVecEnv(..., copy=False)`.** The vec env deep-copies the whole batched
observation on every reset and every step (`royalegym/env.py:884` and `:914`). The harness packs
each observation into its own shared-memory row immediately afterwards, so the copy is thrown
away. This is a one-line ask.

**3. A richer `OpponentPool`.** Separate wins, draws and losses; a way to mark a result as an
evaluation and to tag it with a context; an atomic save; an eviction policy that is not
oldest-first. All four are still open as of today: `_Record` still carries a single `wins: float`
with draws at a half (`royalegym/selfplay.py:95`), `record_result(a, b, score_a)` takes no context,
`save` is a plain `write_bytes`, and `_evict` deletes from `snapshots` without touching `records`.
The harness ships its own versions of all of it, so this is now about whether the next person to
build a trainer has to write them again.

**4. `call` / `get_attr` / `set_attr` on the vec env.** There are none today. The harness reaches
into `vec.envs[i]` directly, and in one place it replaces `env.reward_fn` with a wrapper so it can
catch the per-term reward breakdown before the autoreset clears it
(`royalelearn/rollout/inline.py:400`). That works and is tested, but it is undocumented reaching
into somebody else's object.

**5. A counter for entities dropped past `max_entities`.** `EntityListObsBuilder` drops entities
past its cap and says so in its docstring, silently at runtime
(`royalegym/obs.py:1080-1081`). A silent truncation during training is the exact class of bug
`docs/design.md` warns about. This one is currently theoretical here, for the reason in the next
paragraph.

**Blocked, not open: a network for `EntityListObsBuilder`.** RoyaleGym asked on 2026-09-22 which
RoyaleLearn network pairs with the entity-list observation so the two builders can be compared
like for like. The answer is none. The only trunk in this repo, `ClashTrunk`
(`royalelearn/learn/nets.py`), reads `spatial`, `mask_planes` and `vector`. Until an entity-set
trunk exists here, that comparison cannot be run.

**Landed since the spec was written: grid reuse.** Ask 3 in `harness-spec.md` section 16 asked
RoyaleGym to reuse `PlacementOracle` grids between the mask and the observation. It has landed.
`royalegym/action.py` now has a `grid_key` memo with hit and miss counters, and RoyaleGym measured
2.66 `point_grid` calls per env step. Treat that row of the spec table as closed.

**Not RoyaleGym's, though the spec implies it.** `royalelearn/ladder/results.py` says the ladder's
context digest cannot pin the card table because the engine's Python surface exposes no card-table
vintage. That is no longer true. `RustEngine.config()` now reports `cards_json_fnv1a64` and
`cards_vintage`, and `ClashParallelEnv.config()` passes the engine's own config through. The
remaining work is here: `context_digest` in `royalelearn/coordinator.py:1049` hashes the env spec,
the observation digest, the release mode and the build digest, and does not include the catalogue.
So two games played against different card tables still share a context and are still pooled.
That is a real gap, it is ours, and the fix is available.

## If you are changing RoyaleGym, read this part

The fastest way to find out whether you broke this harness is to run its start-up gates. They
build one environment, check everything on this page, and exit. Nothing trains and no checkpoint
is needed. It does import torch to name the network architecture, so you need the `torch` extra
installed:

```
cd RoyaleLearn
../.venv/Scripts/python -m royalelearn doctor
```

That is the whole contract, executable. If it passes, the surfaces this page describes are intact.
If it fails, the message names the thing that moved and the file it came from.
