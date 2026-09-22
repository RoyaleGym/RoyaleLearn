# The harness, specified

`docs/design.md` says what the harness is for and which commitments it is held to. This page says
what is built, field by field, so that the code can be written from it without further design
decisions. Every number is either **[M]** measured with its source named, or **[A]** derived by
arithmetic from measured numbers, with the arithmetic shown. Nothing here is a menu: where two
defensible choices existed, one was taken and the reason is given.

Where this page says *the references*, it means the three learners this family already has:
[rlgym-ppo](https://github.com/AechPro/rlgym-ppo), rlgym-learn with rlgym-learn-algos, and
[rocket-learn](https://github.com/Rolv-Arild/rocket-learn). Their decisions are copied where they are
right, and named where they are not.

Terms, fixed once and used throughout:

| term | meaning |
|---|---|
| tick | 50 ms of game time (`calibration.json` `time.TICK_MS`, status `measured`) |
| decision / game-step | one `ClashParallelEnv.step`, `decision_ms` of game time rounded up to whole ticks; 500 ms = 10 ticks by default |
| timestep | one stored learner transition: one seat's decision |
| slot | one seat of one battle, held for the whole run; `R` slots in total |
| cycle | one advance of every slot by one decision; the first index of the experience buffer |
| iteration | `T` cycles, then one PPO update |
| env step | a game-step, counted for cadences (snapshots, checkpoints) |

---

## 1. The decisions everything else follows from

**D1. The iteration is a rectangle: `T` cycles × `R` slots.** Slot → (worker, shard, battle, seat)
is fixed for the run. The buffer is a plain two-dimensional array with no ring arithmetic, GAE is one
vectorised backward scan, and — the reason that matters most — the composition and the row order of
every inference batch are functions of the slot index rather than of which worker answered first.
Determinism does not depend on arrival order.

**D2. A worker owns a vec env of many battles; the boundary carries one packed slab per worker per
shard-round.** `K` worker processes each holding `ClashSelfPlayVecEnv(num_games=M)`, not `K×M`
processes holding one battle each. RoyaleGym already batches battles into one `step` and the engine
under it is Rust, so there is no reason to pay `K×M` interpreters to use three cores.

**D3. The process boundary is a byte layout, not a Python protocol.** `rollout/layout.py` defines
offsets, dtypes and a `LAYOUT_VERSION`; both sides compute the same offsets from the same `EnvSpec`.
`docs/design.md` commits to rollout workers moving to Rust when they become the bottleneck. A Rust
worker that writes these bytes and flips these events is a drop-in; the Python worker becomes the
reference implementation and the differential test against it is the acceptance criterion.

**D4. Observations are quantised in the worker and written straight into their final resting place in
the experience buffer.** 14 163 B per row against 55 397 B as float32 **[A]**, section 7.2. The learner never
copies an observation on the CPU except into a small pinned staging ring.

**D5. The policy acts on `decode(encode(obs))`, never on the raw float32.** The bytes that are stored
are exactly the input that produced the stored log-prob, which makes `ratio == 1` at the first
minibatch of the first epoch an invariant the update can assert (section 9.6). It is the cheapest detector
there is for a mask, codec or weight-version drift, and every one of those failures is otherwise
silent.

**D6. The mask travels with the transition, bit-packed, and the same mask is applied at update.**
289 bytes, 2.2% of a row. A rollout/update mask disagreement pins the clip fraction at 1.0 in one
direction and produces `log π = -inf` in the other; storing the mask makes both impossible.

**D7. Inference runs in the parent, one model, one CUDA context, one batched forward per distinct
policy per round.** A 4 GB GPU holds one set of weights comfortably and `K` copies badly, and a
worker that never imports torch costs ~300 MB less resident memory and three seconds less start-up.
A test asserts `"torch" not in sys.modules` inside a worker.

**D8. Every random stream is name-addressed.** `derive_seedseq(master_seed, "ppo/minibatch/epoch/1")`,
never positional spawning. Adding a new consumer of randomness never perturbs an existing stream, so
a config change that ought to be irrelevant is irrelevant. Action sampling takes caller-supplied
uniforms (section 8.3), so a sampled action is a pure function of `(master_seed, iteration, cycle, slot)`
and is independent of batch composition.

**D9. Learner determinism is on by default.** `determinism.tier = "run_exact"`. It costs 10-20% of
throughput **[A]**, and a result that cannot be reproduced costs more than that. The tier is one
config field, it is part of the run identity, and `"throughput"` is the other value.

**D10. Evaluation is structurally separate from training.** Its own env spec, its own worker farm
built at gate time and closed afterwards, its own fixed seed set, its own result table, no
transitions recorded. It is not a percentage of the training mixture.

**D11. The authoritative rating is a batch fit over an append-only result log, not an online filter.**
Bradley-Terry with a Davidson draw term, MAP with a Gaussian prior, standard errors from the inverse
observed Fisher information. Same games in, same numbers out, in any order, on any machine. Online
Elo stays as the dashboard readout and is never the gate.

**D12. Every user-facing behaviour is an ABC in `royalelearn/api/` that imports numpy and msgspec and
never torch.** Concrete implementations live outside `api/` and may import torch. `import
royalelearn` and `royalelearn --help` work in an environment without torch, which is the property the
package has today and keeps.

---

## 2. Geometry and the throughput budget

### 2.1 Unit costs

| quantity | value | source |
|---|---|---|
| engine tick | 50 ms game time | **[M]** `calibration.json` `time.TICK_MS` |
| engine throughput | ~32 000 ticks/s, 20 ticks per call | **[M]** RoyaleGym `docs/architecture.md`, 2026-09-21 |
| `ClashParallelEnv.step` through the full Python stack | 826 game-steps/s = 1.21 ms | **[M]** RoyaleGym `tests/test_rust_engine.py` throughput report, 2026-09-21 |
| split of that 1.21 ms | 0.31 ms engine (26%), 0.90 ms Python obs + mask for both seats (74%) | **[A]** from the two rows above |
| one network forward | 378 MFLOP/sample at C=64, N=4, 32×18 board | **[A]** section 8.1 |
| RTX 3050 Laptop, achieved on convs | ~3 TFLOP/s fp32, ~9 TFLOP/s bf16 | **[A]** 40% of the 6 / 24 peak |
| observation row, as the environment hands it over | 55 397 B | **[M]** shapes from `single_observation_space` |
| observation row, this codec | 14 163 B | section 7.2 |

Both row figures are arithmetic on the observation space rather than constants, so the next layout
change is a recomputation and not a rewrite. With `S` spatial planes of `H·W` cells of which `s` are
static and `f` are stored as `float16`, a vector of width `V` and an action space of `A`:

```
row_bytes  = H*W * ( (S - s - f) * 1 + f * 2 )  +  2 * V  +  ceil(A / 8)
float_row  = 4 * ( H*W * S + V )  +  A  +  H*W * hand_size
```

On the catalogue the engine ships today, 95 cards, that is `S = 20`, `s = 2` static planes held once
per seat, `f = 2` hit-point planes, `H·W = 576`, `V = 1177`, `A = 2305`, `hand_size = 4`, giving
`576*(16 + 2*2) + 2*1177 + 289 = 9 216 + 2 304 + 2 354 + 289 = 14 163 B` against
`4*(576*20 + 1177) + 2 305 + 2 304 = 55 397 B`, 3.91 times smaller **[A]**. The final term of
`float_row` is the four mask planes, which the environment emits and the codec never stores: they are
the bit-packed mask reshaped, and the learner reshapes it back at unpack.

### 2.2 Profiles

One config block per machine class; only numbers change.

| field | laptop (default) | workstation | many-core |
|---|---|---|---|
| RAM / VRAM / threads | 8 GB / 4 GB / 8 | 32 GB / 12 GB / 16 | 64 GB+ / 24 GB+ / 64 |
| `rollout.workers` K | 3 | 12 | 24 |
| `rollout.games_per_worker` M | 32 | 64 | 64 |
| `rollout.shards_per_worker` | 2 | 2 | 2 |
| battles G = K·M | 96 | 768 | 1 536 |
| slots R = 2G | 192 | 1 536 | 3 072 |
| learner rows (0.75·R) | 144 | 1 152 | 2 304 |
| `ppo.timesteps_per_iteration` | 32 768 | 131 072 | 262 144 |
| cycles T = ceil(ts / learner rows) | 228 | 114 | 114 |
| `net.channels` C / `net.blocks` N | 64 / 4 | 96 / 8 | 128 / 12 |
| `ppo.batch_size` | 4 096 | 16 384 | 32 768 |
| `ppo.minibatch_size` | 512 | 2 048 | 8 192 |
| buffer bytes (observations) | 623 MB | 5.0 GB | 10.0 GB |
| `rollout.overlap` | false | true | true |
| `ladder.candidate_every_env_steps` | 4 000 000 | 2 000 000 | 2 000 000 |

`shards_per_worker` is the intra-worker overlap: each worker holds two independent vec envs of `M/2`
battles and alternates, so it is stepping one shard while the parent runs inference on the other. A
**cycle** is `shards_per_worker` consecutive shard-rounds, so that every slot advances exactly once;
the shard interleaving is a scheduling detail of the parent and never appears in the buffer's index.

What a second shard is worth follows from one measured fact: **environment time and inference time
are additive, with no overlap of their own.** Standing a fixed 5 ms cost after every vec step adds
5.1 to 5.7 ms to the step at every batch size, on both engines **[M]**. So the second shard hides
`min(env, inference)` and nothing else, and its value is set by the ratio of the two per shard-round:

| env ms per shard ÷ inference ms per round | what a second shard buys |
|---|---|
| well under 1 (few battles per worker, inference dominates) | close to double |
| about 1 | most of the inference |
| well over 1 (many battles per worker, the environment dominates) | little; at a ratio near 7, about 14% **[M]** |

That inverts the obvious intuition: sharding is worth most where battles per worker is *small*,
because a worker with many battles has already amortised inference by batching. At the laptop
profile a shard is 16 battles against three batched forwards, which sits in the middle of that table.
Two things keep the number honest rather than assumed: a fixed sleep releases the interpreter cleanly
while a real forward competes for cores, so the table is the optimistic case; and `royalelearn bench`
measures the ratio against the network actually configured and prints what a second shard is worth on
that machine. If it prints under about 10%, set `shards_per_worker = 1` and spend the thread on a
worker instead.

The buffer row is `(T + obs.frame_stack) × R × row_bytes`: `T+1` cycles for the bootstrap row plus
`frame_stack - 1` history cycles carried over from the previous iteration (section 9.1). At the laptop
profile that is `229 × 192 × 14 163 B = 623 MB` **[A]**. The two larger profiles run
`rollout.overlap = true`, which costs a second rectangle, so their figures are twice
`115 × 1 536 × 14 163 B` and twice `115 × 3 072 × 14 163 B`.

### 2.3 The laptop budget

Per iteration at the laptop profile: `T=228` cycles × `R=192` slots, of which `144` are learner rows,
so `228 × 144 = 32 832` timesteps and `228 × 96 = 21 888` game-steps.

| phase | seconds | basis |
|---|---|---|
| engine ticks + Python obs and mask | 8.8 | 21 888 game-steps ÷ (3 × 826 game-steps/s) **[M]** |
| rollout inference (actor only) | ~4 | 456 shard-rounds × ~3 forwards × ~1.5 ms launch-bound, plus 16.5 TFLOP **[A]** |
| critic pass, chunks of 1024 | 1.4 | 32 976 rows × 378 MFLOP ÷ 9 TFLOP/s **[A]** |
| PPO update: 3 epochs × (fwd+bwd ≈ 3×fwd) × 2 nets | 24.8 | 223 TFLOP ÷ 9 TFLOP/s **[A]** |
| GAE, advantage standardisation, H2D copies | ~2 | **[A]** |
| **total, serial** | **~41** | **≈ 800 timesteps/s** |

At `determinism.tier = "run_exact"` subtract 10-20%: **~680-720 timesteps/s**, so 100 M timesteps is
**39-41 hours**. With `rollout.overlap = true` the rollout hides under the update and the figure is
~1 100 timesteps/s, 25 hours; that costs a second buffer (623 MB) and is off by default on 8 GB.

Two consequences, stated so nobody re-derives them:

- **The learner is the bottleneck on this machine, by about 2.5×.** Rollout capacity is ~3 300
  timesteps/s against an update capacity of ~1 330. That inverts the usual RLGym situation and it is
  why `K = 3` and not 32, and why the network is 430 k parameters and not 4 M.
- **"The harness is never the bottleneck" is an invariant with a number:** rollout capacity ≥ 2 ×
  update capacity on every shipped profile. `royalelearn bench` prints both, the run logs
  `throughput/rollout_capacity_ratio` every iteration, and start-up warns below 1.5.

RoyaleGym's planned Rust-backed default observation and mask would take the rollout side from
298 µs/transition to about 110 µs **[A]**, i.e. 9 000 timesteps/s. It is not a prerequisite: it is
what later lets `K` drop to one or two and frees cores.

### 2.4 The RAM ledger, printed by `royalelearn doctor`

```
experience buffer (pinned)      229 x 192 x 14 163 B      623 MB
parent torch + CUDA host allocations                    ~1 400 MB
3 workers x (32 RustEngine battles, no torch)      3 x ~250 =  750 MB
3 workers x shard staging and scalars                3 x ~40 =  120 MB
python interpreters, numpy, msgspec                       ~400 MB
                                                        ----------
                                                         ~3.3 GB of 7.8 GB
at gate time, additionally: 2 eval workers x 24 battles   ~380 MB
```

`doctor` refuses to start a run whose projection exceeds `doctor.ram_budget_mb` (default 6 500).

---

## 3. The file tree

```
RoyaleLearn/
  README.md
  LICENSE                                  MIT
  pyproject.toml                           deps: royalegym, numpy, msgspec; extras torch, wandb, dev
  docs/
    design.md                              the commitments; unchanged in intent
    harness-spec.md                        this file
    throughput.md                          the budget of section 2 and how to re-measure it
    determinism.md                         the contract of section 5
    ladder.md                              rating, matchmaking, the gate, the statistics
    checkpoints.md                         the format, versioning, what resume proves
    metrics.md                             every metric, its range, its alarm
    running.md                             first run, reading the dashboard, one section per alarm
    royalegym-asks.md                      section 16, with costs
    media/                                 unchanged
  examples/
    train_1v1.py                           the file a user runs and edits
    configs/laptop.json                    the shipped default, written out in full
    configs/workstation.json               the scale-up profile
    configs/smoke.json                     MockEngine, 2 workers x 2 battles, 3 iterations
    custom_reward.py                       how to swap a RewardFunction and keep the identity honest
  royalelearn/
    __init__.py                            version and lazy exports; imports nothing that needs torch
    version.py                             __version__, git describe helper
    errors.py                              PreflightError, IdentityMismatch, CheckpointFormatError,
                                           WorkerTimeout, AlarmHalt, StaleEngineBuild
    config.py                              the whole msgspec config tree, hashing, JSON I/O
    seeding.py                             derive_seedseq / derive_generator / derive_int, the namespace
    determinism.py                         apply(tier), the env-var preconditions, the assertions
    identity.py                            RunIdentity, EngineBuild, compute_identity(config)
    obs_layout.py                          vector fields resolved by name from the env's vector_layout()
    coordinator.py                         LearningCoordinator: the loop, the only place phases are ordered
    cli.py                                 argument parsing for python -m royalelearn
    __main__.py                            entry point -> cli.main()
    rewards.py                             the potential-based default reward composition
    checkpoint.py                          DirCheckpointStore: manifest, per-component folders, pruning
    api/
      __init__.py                          re-exports every ABC and struct
      rollout.py                           EnvSpec, SlotPlan, RolloutRound, WorkerCommand, WorkerFailure,
                                           EpisodeRecord, RolloutSource
      policy.py                            ActionDistribution, Actor, Critic, ActorCritic, NetworkFactory
      buffer.py                            ObsCodec, ExperienceBuffer
      advantage.py                         AdvantageEstimator
      update.py                            Update, UpdateResult
      ladder.py                            Matchmaker, Rater, PromotionGate, SnapshotStore, EvictionPolicy
      metrics.py                           MetricsSink, MetricRow, Alarm, AlarmResult
      checkpoint.py                        Checkpointable, CheckpointStore, Manifest
      schedule.py                          Schedule
    rollout/
      __init__.py
      layout.py                            the shared-memory byte layout; the cross-language contract
      codec.py                             SpatialObsCodec: pack, unpack_to_device, the quantisation table
      envspec.py                           EnvFactorySpec, ComponentSpec, the class allow-list
      preflight.py                         engine construction, mask_disagreements, digests, the RAM ledger
      plan.py                              SlotPlanner: geometry, slot maps, the per-slot seed paths
      scripted.py                          worker-side NoopOpponent / RandomLegalOpponent adapters
      worker.py                            child process main loop; numpy only, torch is a test failure
      farm.py                              ProcessRolloutSource: spawn, handshake, rounds, failure, restart
      inline.py                            InlineRolloutSource: same semantics, one process, no shm
    learn/
      __init__.py
      nets.py                              ClashTrunk, PointerPolicyHead, ValueHead, DefaultNetworkFactory
      distribution.py                      MaskedCategorical
      actor_critic.py                      SeparateActorCritic, BehaviourSnapshot
      buffer.py                            RectBuffer: the (T+1, R) shared-memory-backed rectangle
      gae.py                               GAE (torch, vectorised) and reference_gae (pure python)
      returns.py                           WelfordReturnScaler
      ppo.py                               PPOUpdate: the losses, the minibatch loop, the diagnostics
      inference.py                         BatchedInference: per-round forwards, streams, staging ring
      schedules.py                         Constant, Linear, Geometric, PiecewiseConstant, the lr backoff
    ladder/
      __init__.py
      snapshots.py                         DiskSnapshotStore: fp16 actor + spec.json + digest, LRU
      matchmaker.py                        MixMatchmaker: the mixture, PFSP from the fitted ratings
      rating.py                            BradleyTerryDavidsonRater, EloReadout, transitivity_residual
      results.py                           ResultLog: append-only games.jsonl plus the aggregate cache
      evaluate.py                          EvalRunner: fixed seed set, paired sides, no experience
      gate.py                              WilsonGate: the three conditions and the decision record
      eviction.py                          HallOfFameEviction
      pool.py                              LadderPool: royalegym OpponentPool + ResultLog + snapshots
    metrics/
      __init__.py
      schema.py                            every metric key, unit, description and healthy range
      records.py                           IterationMetrics and the per-source dataclasses
      sinks.py                             JsonlSink (always on), ConsoleSink, CompositeSink
      viser_sink.py                        ViserSink: the learning-status datagram the viewer reads
      wandb_sink.py                        WandbSink: a decorator, run-id persistence, optional import
      alarms.py                            the alarm table of section 13.3 and its evaluator
      bundle.py                            write_bundle: the diagnostic directory written on any halt
  tests/                                   section 15
```

Deleted in the first commit: `continuous_policy.py`, `discrete_policy.py`, `multi_discrete_policy.py`,
`value_estimator.py`, `experience_buffer.py`, `ppo_learner.py`, the `SEED_CLASSES` machinery in
`__init__.py`, `NOTICE`, `LICENSE-APACHE-2.0`, the `[tool.ruff] extend-exclude` list, and the seed
assertions in `tests/test_package.py`. `pyproject.toml`'s `license` becomes `{ text = "MIT" }`.
`docs/design.md` loses its "The rlgym-ppo seed modules" section and gains one sentence recording that
they were a reference and were removed when the harness landed.

---

## 4. The ABCs

All in `royalelearn/api/`. They import `numpy`, `typing` and `msgspec` only. `torch` appears in type
annotations behind `if TYPE_CHECKING`. Every ABC that owns state also implements
`api.checkpoint.Checkpointable`.

### 4.1 `api/rollout.py`

```python
class ObsKeySpec(msgspec.Struct, frozen=True):
    """One key of the environment's observation space, read from the space itself."""
    shape: tuple[int, ...]
    dtype: str                              # numpy dtype name
    low: tuple[float, ...]                  # per channel for a 3-D key, length 1 otherwise
    high: tuple[float, ...]                 # per channel for a 3-D key, length 1 otherwise

class EnvSpec(msgspec.Struct, frozen=True):
    """Everything the learner must know about the environment before it builds anything.

    Every field here is READ from the running environment at preflight. The harness holds no
    layout constant of its own: a width, a plane count and a field offset are all environment
    facts, and typing one into this repo would make a RoyaleGym change a silent wrong answer
    instead of a loud one."""
    num_cards: int                          # the card catalogue this run is built on
    obs_space: dict[str, ObsKeySpec]        # every key of single_observation_space, verbatim
    vector_layout: tuple[tuple[str, int, int], ...]   # (name, offset, size), from ObsBuilder
    spatial_layout: tuple[tuple[str, bool], ...]      # (name, static), from ObsBuilder
    frame_stack: int                        # obs.frame_stack; how many cycles the trunk sees
    vector_size: int                        # obs_space["vector"].shape[0]
    spatial_shape: tuple[int, int, int]     # obs_space["spatial"].shape
    n_actions: int                          # single_action_space.n
    hand_size: int
    tiles: tuple[int, int]                  # (tiles_y, tiles_x)
    decision_ms: int                        # 500
    tick_ms: int                            # 50
    decision_ticks: int                     # 10
    regular_ticks: int                      # 3600
    overtime_ticks: int                     # 1200
    engine_build_digest: str                # sha256, section 5.3
    obs_digest: str                         # sha256 over obs_space, vector_layout, spatial_layout,
                                            # frame_stack, n_actions, num_cards and the obs-builder
                                            # ComponentSpec -- so a Reveal change changes it
    env_factory: "EnvFactorySpec"
```

`vector_size` and `spatial_shape` are conveniences derived from `obs_space` and are there so that the
common expressions read well; the space is the authority and a mismatch between the two is an
assertion at construction. The observation carries no velocity, acceleration or difference channel:
motion reaches the policy by frame stacking (`obs.frame_stack`, section 9.1), which is the harness's
own affair and costs the environment nothing.

Superhuman information — anything a human player at the same moment could not see — is enabled per
field through the obs builder's frozen `Reveal` struct, which is part of `ClashParallelEnv.config()`.
An enabled field adds vector slots, and `enemy_spell_aim` adds a spatial plane, so both widths and the
plane count move with it. Every one of those changes arrives through `vector_layout`,
`spatial_layout` and the observation space like any other layout fact, and through `obs_digest` like
any other component setting, so a policy that sees revealed information can never be rated against
one that does not (section 11.6).

```python
class Assignment(msgspec.Struct, frozen=True):
    """What one battle's next episode is. Drawn by the Matchmaker at the battle's own episode
    boundary, from match/battle/{b}/ordinal/{k}, and therefore a pure function of the master
    seed, the battle index and the reset ordinal."""
    battle: int
    ordinal: int                            # reset ordinal this assignment takes effect on
    role: int                               # 0 mirror, 1 pool, 2 scripted
    opponent_id: str | None                 # "snap:<id>", "scripted:<name>", or None for mirror
    group: tuple[int, int]                  # per seat: -1 learner, 0..P-1 resident snapshot,
                                            #           -2 scripted (worker-side), -3 dead
    learner_seat: int                       # 0 blue, 1 red; -1 for mirror (both)

class SlotPlan(msgspec.Struct, frozen=True):
    """The iteration's opening assignment table: what every battle is playing at the moment the
    iteration begins. Assignments change only at a battle's own episode boundary, where the
    parent draws a fresh one; nothing in an iteration's span reassigns a battle mid-episode, so
    a partially controlled trajectory cannot occur."""
    iteration: int
    n_battles: int
    n_slots: int                            # 2 * n_battles
    assignment: tuple[Assignment, ...]      # one per battle, in battle order
    resident_snapshots: tuple[str, ...]     # len <= ladder.max_resident_opponents

class RolloutRound:
    """One shard-round of observations. Holds zero-copy views; not a Struct."""
    cycle: int                              # buffer row index
    shard: int
    slots: np.ndarray                       # int32[n], the slot indices this round covers, ascending
    obs_rows: np.ndarray                    # int32[n], where in the buffer each slot's row was written
    group: np.ndarray                       # int8[n], as reported by the worker for THIS round
    reward: np.ndarray                      # float32[n]
    terminated: np.ndarray                  # bool[n]
    truncated: np.ndarray                   # bool[n]
    valid: np.ndarray                       # bool[n]; False where that slot's worker is dead
    deploy_status: np.ndarray               # int8[n]; -1 no command, 0 OK, 1..11 = a MASK BUG
    tick: np.ndarray                        # int32[n]
    episode_end: np.ndarray                 # int8[n]; 0 none, 1 win, 2 loss, 3 draw
    episodes: list["EpisodeRecord"]         # one per episode that ended on this round
    timings: dict[str, float]

class EpisodeRecord(msgspec.Struct, frozen=True):
    slot: int
    worker: int
    shard: int
    battle: int
    seat: int                               # 0 blue, 1 red
    ordinal: int                            # reset ordinal of this episode within its battle
    episode_seed_path: str
    policy_id: str                          # "learner" | "snap:<id>" | "scripted:<name>"
    opponent_id: str
    bucket: str                             # "mirror" | "pool" | "scripted"
    # the terminal scalars, copied straight out of the env's final_info; the environment's
    # EPISODE_STAT_KEYS is the authority on the set, and a key it adds arrives here unchanged
    episode_steps: int
    episode_ticks: int
    own_crowns: int
    enemy_crowns: int
    own_tower_hp_frac: float                # mean of the three towers' hp/max
    enemy_tower_hp_frac: float              # mean of the three towers' hp/max
    elixir_leak_steps: int                  # decisions spent at full elixir
    elixir_count_exact: bool                # False if this seat's count of the opponent's elixir was
                                            # ever caught disagreeing with the bar it models, so the
                                            # policy read an estimate in a slot documented as exact
    # counted by the worker as the episode runs
    winner: int                             # -1 none, 0 blue, 1 red, 2 draw
    outcome: int                            # +1 win, 0 draw, -1 loss, from this seat's view
    cards_played: int
    illegal_commands: int                   # deploy_status in 1..11; must be 0
    undiscounted_return: float
    reward_terms: dict[str, float]          # per-term episode sums for this seat

class WorkerCommand(msgspec.Struct, tag_field="kind", tag=str.upper):
    """Tagged union; six variants, one per round."""
class Step(WorkerCommand):
    actions: np.ndarray                     # int16[n_slots_this_shard]
    gamma: float                            # the current schedule value, threaded into the reward
    group: np.ndarray                       # int8[n_slots_this_shard]
    opponent_ix: np.ndarray                 # int8[n_slots_this_shard], index into the plan's
                                            # resident_snapshots / scripted table
    learner_seat: np.ndarray                # int8[n_battles_this_shard]
                                            # the three arrays carry the parent's assignment for
                                            # every battle whose episode started on the previous
                                            # round, and are empty otherwise; the worker needs them
                                            # only to know which slots it fills with a scripted
                                            # action
class Plan(WorkerCommand):
    plan: SlotPlan                          # the iteration's opening table, once per iteration
class SetState(WorkerCommand):
    snapshots: tuple[bytes | None, ...]     # per battle; start from a recorded position
class Spaces(WorkerCommand):
    pass                                    # re-read the spaces without stepping
class Defer(WorkerCommand):
    pass                                    # nothing this round; hand back the same data
class Close(WorkerCommand):
    pass

class WorkerFailure(msgspec.Struct, frozen=True):
    worker: int
    shard: int
    cycle: int
    kind: str                               # "exception" | "crash" | "timeout" | "protocol"
    message: str                            # the child's traceback, verbatim

class RolloutSource(ABC):
    """Where experience comes from. The one seam between the learner and the world.

    Implementers may assume: `begin_iteration` precedes any `next_round`; exactly one `submit`
    per `next_round`; `Step.actions` are legal under the mask that was handed out; the caller
    does not retain a RolloutRound's views past the next `next_round` for the same shard.
    Implementers MUST guarantee: `slots` is ascending; every slot appears exactly once per
    cycle; a dead worker's slots arrive with `valid=False` rather than not arriving."""
    @abstractmethod
    def spec(self) -> EnvSpec: ...
    @abstractmethod
    def begin_iteration(self, plan: SlotPlan, buffer: "ExperienceBuffer", iteration: int) -> None: ...
    @abstractmethod
    def next_round(self, timeout_s: float = 30.0) -> RolloutRound: ...
    @abstractmethod
    def submit(self, command: WorkerCommand) -> None: ...
    @abstractmethod
    def drain_failures(self) -> list[WorkerFailure]: ...
    @abstractmethod
    def restart(self, worker: int) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    def stats(self) -> dict[str, float]:
        """Per-round timing: env_ms, wait_ms, codec_ms, parent_wait_frac, bytes_out."""
        return {}
```

Shipped: `rollout.farm.ProcessRolloutSource` (default) and `rollout.inline.InlineRolloutSource`
(identical semantics in one process, no shared memory — the reference implementation and what the
fast tests run). A future `RustRolloutSource` implements the same ABC and writes the same bytes.

### 4.2 `api/policy.py`

```python
ObsBatch = tuple  # a NamedTuple of device tensors, shaped from EnvSpec.obs_space:
                  #   spatial     (B, k*S, H, W) f32    k = frame_stack, S spatial planes
                  #   mask_planes (B, k*A_s, H, W) f32   A_s per-slot action planes, derived
                  #   vector      (B, V) f32
                  #   mask        (B, n_actions) bool

class ActionDistribution(ABC):
    @abstractmethod
    def sample(self, uniforms: Tensor) -> Tensor: ...
        """(B,) int64. Driven by CALLER-SUPPLIED uniforms in [0,1) so that the sampled action is
        reproducible independently of batch composition and of torch's global RNG."""
    @abstractmethod
    def mode(self) -> Tensor: ...             # (B,) int64, mask-respecting argmax
    @abstractmethod
    def log_prob(self, actions: Tensor) -> Tensor: ...   # (B,) float32
    @abstractmethod
    def entropy(self) -> Tensor: ...          # (B,) float32
    @abstractmethod
    def noop_entropy(self) -> Tensor: ...     # (B,) float32, H2 of p(no-op) against p(play)
    @abstractmethod
    def n_legal(self) -> Tensor: ...          # (B,) int64

class Actor(ABC, nn.Module):
    @abstractmethod
    def logits(self, obs: ObsBatch) -> Tensor: ...
        """(B, spec.n_actions) float32 ALWAYS, even inside autocast. Raw logits: no softmax, no
        clamp, no mask."""
    def distribution(self, obs: ObsBatch) -> ActionDistribution:
        return MaskedCategorical(self.logits(obs), obs.mask)

class Critic(ABC, nn.Module):
    @abstractmethod
    def value(self, obs: ObsBatch) -> Tensor: ...        # (B,) float32

class ActorCritic(ABC, nn.Module):
    """Implementers may assume obs tensors are on the device and already dequantised, and that
    obs.mask[:, 0] is True on every row (RoyaleGym sets mask[NOOP]=1 unconditionally,
    action.py:284, including after game over). They MUST assert it."""
    arch: "ArchSpec"
    @abstractmethod
    def act(self, obs: ObsBatch, uniforms: Tensor) -> "ActResult": ...
    @abstractmethod
    def value(self, obs: ObsBatch) -> Tensor: ...
    @abstractmethod
    def backprop(self, obs: ObsBatch, actions: Tensor) -> "BackpropResult": ...
        """Recompute under the current parameters, applying exactly the mask read from the
        buffer -- never a recomputed one."""
    @abstractmethod
    def actor_parameters(self) -> Iterator[nn.Parameter]: ...
    @abstractmethod
    def critic_parameters(self) -> Iterator[nn.Parameter]: ...
    @abstractmethod
    def actor_state_dict_fp16(self) -> dict[str, Tensor]: ...
        """What a pool snapshot stores: actor only, fp16, no optimizer."""

class ActResult(msgspec.Struct):
    actions: Tensor      # (B,) int64
    log_probs: Tensor    # (B,) float32
    entropy: Tensor      # (B,) float32, diagnostics only
    p_noop: Tensor       # (B,) float32, diagnostics
    n_legal: Tensor      # (B,) int64, diagnostics

class BackpropResult(msgspec.Struct):
    log_probs: Tensor    # (B,) float32
    entropy: Tensor      # (B,) float32
    noop_entropy: Tensor # (B,) float32
    values: Tensor       # (B,) float32
    n_legal: Tensor      # (B,) int64

class NetworkFactory(ABC):
    @abstractmethod
    def build(self, spec: EnvSpec, arch: "ArchSpec", device, dtype) -> ActorCritic: ...
    @abstractmethod
    def arch_digest(self, spec: EnvSpec, arch: "ArchSpec") -> str: ...
        """sha256 of the canonical JSON of (arch, obs_space, frame_stack, num_cards, n_actions).
        A snapshot or checkpoint built from a different architecture is refused, not
        shape-errored."""
```

Shipped: `learn.nets.DefaultNetworkFactory` building `SeparateActorCritic(ClashTrunk +
PointerPolicyHead, ClashTrunk + ValueHead)`. `SharedTrunkActorCritic` ships too, because
`docs/design.md` requires a swappable alternative and because it is the escape hatch if rollout
inference turns out to cost more than a quarter of the iteration.

### 4.3 `api/buffer.py`

```python
class CodecTable(msgspec.Struct, frozen=True):
    """How each observation key is stored, decided from EnvSpec.obs_space at preflight rather
    than from a list of plane indices. Logged, hashed and written into every snapshot."""
    plane: tuple[tuple[str, str, float], ...]   # per spatial plane: (name, storage, divisor);
                                                # storage in {"uint8", "float16", "static",
                                                #             "derived"}
    vector: str                                 # "float16"
    mask: str                                   # "bitpack"
    def digest(self) -> str: ...                # sha256 of the canonical JSON

class ObsCodec(ABC):
    """Quantisation of one observation row. The worker packs; the learner unpacks on the GPU.
    Implementers MUST be exact round-trips for the integer-valued channels and MUST declare
    row_bytes as a constant given an EnvSpec and its CodecTable."""
    @abstractmethod
    def table(self, spec: EnvSpec, sample: Sequence[dict[str, np.ndarray]]) -> CodecTable: ...
        """Decide storage per key from the declared bounds and a sample of real observations."""
    @abstractmethod
    def row_bytes(self, spec: EnvSpec) -> int: ...
    @abstractmethod
    def pack(self, obs: dict[str, np.ndarray], out: memoryview, row: int) -> None: ...
    @abstractmethod
    def static_planes(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...
        """The planes EnvSpec.spatial_layout declares static; stored once per seat, never per
        row. Declared, never inferred: see section 7.2."""
    @abstractmethod
    def unpack_to_device(self, raw: Tensor, statics: Tensor, out: ObsBatch) -> None: ...
        """Dequantise, scatter the static planes in, reshape the stored mask into the mask
        planes, and gather the frame-stack history."""
    @property
    @abstractmethod
    def codec_version(self) -> int: ...

class ExperienceBuffer(ABC, Checkpointable):
    """A rectangle of (T + frame_stack) cycles x R slots: T collected cycles, one bootstrap
    row, and frame_stack - 1 history rows carried over from the previous iteration. Owns the
    shared-memory block the workers write into. Implementers may assume each (cycle, slot) cell
    is written exactly once, by the worker that owns that slot; the learner only reads."""
    @abstractmethod
    def shared_handle(self) -> "BufferHandle": ...      # name, size, offsets; picklable
    @abstractmethod
    def begin_iteration(self, plan: SlotPlan, cycles: int) -> None: ...
    @abstractmethod
    def record_round(self, r: RolloutRound, actions: np.ndarray,
                     log_probs: np.ndarray) -> None: ...
        """Scalars only: observations are already in place. O(n), no observation copy."""
    @abstractmethod
    def set_values(self, values: Tensor) -> None: ...           # (T+1, R) float32
    @abstractmethod
    def set_final_values(self, cells: np.ndarray, values: Tensor) -> None: ...
        """V(final_obs) for truncated cells; `cells` is int64[(k, 2)] of (cycle, slot)."""
    @abstractmethod
    def set_advantages(self, adv: Tensor, ret: Tensor) -> None: ...   # (T, R) float32
    @abstractmethod
    def trainable_mask(self) -> Tensor: ...                     # (T, R) bool
    @abstractmethod
    def batches(self, batch_size: int, minibatch_size: int, epochs: int,
                rng_for_epoch: Callable[[int], np.random.Generator]) -> Iterator["Batch"]: ...
        """Yields batches; each Batch knows its true sample count and iterates device-resident
        minibatches. A batch never straddles an epoch boundary: the epoch's remainder is its own
        smaller batch, correctly weighted. Gathers per MINIBATCH, never per batch."""
```

Shipped: `learn.buffer.RectBuffer`.

### 4.4 `api/advantage.py`, `api/update.py`, `api/schedule.py`

```python
class AdvantageEstimator(ABC, Checkpointable):
    @abstractmethod
    def compute(self, *, rewards, values, final_values, terminated, truncated, trainable,
                gamma: float, lam: float) -> tuple[Tensor, Tensor, "AdvantageStats"]: ...
        """rewards/terminated/truncated/trainable are (T, R); values is (T+1, R);
        final_values is (T, R) and is read only where truncated.
        Returns (advantages (T,R), returns (T,R), stats).
        Implementers MUST bootstrap a terminated cell from 0 and a truncated cell from
        final_values, and MUST NOT carry the recursion across an episode boundary."""

class AdvantageStats(msgspec.Struct):
    raw_return_mean: float; raw_return_std: float
    reward_scale: float; clipped_reward_frac: float

class Update(ABC, Checkpointable):
    @abstractmethod
    def step(self, buffer: ExperienceBuffer, sched: "ScheduleState") -> "UpdateResult": ...

class UpdateResult(msgspec.Struct):
    policy_loss: float; value_loss: float; entropy: float; noop_entropy: float
    entropy_normalised: float; kl: float; clip_fraction: float; dual_clip_fraction: float
    explained_variance: float; ratio_max_abs_dev: float
    grad_norm_actor: float; grad_norm_critic: float
    update_magnitude_actor: float; update_magnitude_critic: float
    kl_by_epoch: list[float]; clip_fraction_by_epoch: list[float]
    n_minibatches: int; n_optimizer_steps: int; n_samples: int
    samples_unused_frac: float; seconds: float

class Schedule(ABC):
    @abstractmethod
    def value(self, env_steps: int) -> float: ...
```

Shipped estimators: `learn.gae.GAE` (torch, vectorised over slots) and `learn.gae.reference_gae`
(pure python, per slot, used only by the test that proves the vectorised one). Shipped schedules:
`Constant(v)`, `Linear(a, b, over_env_steps)`, `Geometric(a, b, over_env_steps)` (used for gamma, so
that `1-gamma` decays geometrically), `PiecewiseConstant(points)`.

### 4.5 `api/ladder.py`

```python
class Matchmaker(ABC, Checkpointable):
    @abstractmethod
    def plan(self, iteration: int, pool: "LadderPool", ratings: "RatingTable",
             geometry: "Geometry") -> SlotPlan: ...
        """The iteration's opening table. One Assignment per battle, each drawn at that
        battle's current ordinal, so plan() is assign() applied across the geometry."""
    @abstractmethod
    def assign(self, battle: int, ordinal: int, pool: "LadderPool",
               ratings: "RatingTable") -> Assignment: ...
        """The assignment for one battle's next episode. Draws from
        match/battle/{battle}/ordinal/{ordinal}, so it is a pure function of the master seed and
        its two arguments: the same episode of the same battle always meets the same opponent,
        whichever iteration it happens to fall in and whichever worker holds it."""
    @abstractmethod
    def on_episode(self, record: EpisodeRecord) -> None: ...

class Rater(ABC, Checkpointable):
    @abstractmethod
    def fit(self, results: "ResultView") -> "RatingTable": ...
        """id -> (rating in Elo units, standard error). MUST be a pure function of `results`:
        same games in, same numbers out, in any order, on any machine."""
    @abstractmethod
    def predict(self, a: str, b: str) -> float: ...       # P(a scores against b), draws as 0.5
    @abstractmethod
    def transitivity_residual(self, results: "ResultView") -> float: ...

class RatingTable(msgspec.Struct, frozen=True):
    rating: dict[str, float]; se: dict[str, float]
    anchor: str; draw_nu: float | None
    n_games: dict[str, int]
    transitivity_residual: float; converged: bool; iterations: int

class PromotionGate(ABC):
    @abstractmethod
    def evaluate(self, candidate: str, pool: "LadderPool",
                 runner: "EvalRunner") -> "GateDecision": ...

class GateDecision(msgspec.Struct):
    candidate: str; champion: str
    admit: bool; promote: bool; cycle: bool
    conditions: dict[str, "ConditionResult"]
    eval_seed_set_sha: str; wall_seconds: float

class ConditionResult(msgspec.Struct):
    passed: bool; n: int; observed: float; bound: float; reference: float

class SnapshotStore(ABC):
    @abstractmethod
    def put(self, snapshot_id: str, ac: ActorCritic, meta: dict) -> str: ...
    @abstractmethod
    def get(self, snapshot_id: str, device) -> Actor: ...     # LRU-cached
    @abstractmethod
    def digest(self, snapshot_id: str) -> str: ...
    @abstractmethod
    def list(self) -> list[str]: ...

class EvictionPolicy(ABC):
    @abstractmethod
    def select_for_eviction(self, *, pool: "LadderPool", ratings: RatingTable,
                            max_sampled: int) -> list[str]: ...
        """Ids to remove FROM THE SAMPLER. The archive and the result log are never touched:
        eviction is about sampling cost, not about forgetting evidence."""
```

Shipped: `MixMatchmaker`, `BradleyTerryDavidsonRater` (authoritative) and `EloReadout` (dashboard),
`WilsonGate`, `DiskSnapshotStore`, `HallOfFameEviction`.

### 4.6 `api/metrics.py` and `api/checkpoint.py`

```python
class MetricsSink(ABC, Checkpointable):
    @abstractmethod
    def open(self, *, identity: "RunIdentity", config_json: str, run_dir: Path) -> None: ...
    @abstractmethod
    def write(self, row: "MetricRow") -> None: ...
        """One flat dict per iteration. Implementers MUST NOT mutate the row and MUST NOT raise
        on an unknown key."""
    def write_episodes(self, rows: Sequence[EpisodeRecord]) -> None: ...   # default: ignore
    def write_alarms(self, alarms: Sequence["AlarmResult"]) -> None: ...   # default: ignore
    def write_artifact(self, name: str, path: Path) -> None: ...           # default: ignore
    @abstractmethod
    def close(self) -> None: ...

class Checkpointable(Protocol):
    FORMAT_VERSION: int
    def save_checkpoint(self, folder: Path) -> None: ...
    def load_checkpoint(self, folder: Path, *, strict: bool) -> None: ...
        """With strict=False, a missing file prints the exact path it wanted and continues with a
        default. With strict=True it raises. strict defaults to True on resume."""

class CheckpointStore(ABC):
    @abstractmethod
    def write(self, components: Mapping[str, Checkpointable], manifest: "Manifest") -> Path: ...
    @abstractmethod
    def read(self, path: Path, components: Mapping[str, Checkpointable], *,
             strict: bool) -> "Manifest": ...
    @abstractmethod
    def latest(self, run_dir: Path) -> Path | None: ...
    @abstractmethod
    def prune(self, run_dir: Path, keep: int) -> list[Path]: ...
```

Shipped sinks: `JsonlSink` (always installed, never optional — it is the file the resume test
compares), `ConsoleSink`, `CompositeSink`, `WandbSink` (a decorator over any sink). Shipped store:
`DirCheckpointStore`.

---

## 5. Determinism

### 5.1 Three tiers, named and recorded

| tier | guarantee | cost | when |
|---|---|---|---|
| **T1 env-exact** | given the run identity, every episode's engine state-hash sequence, every observation, every mask and every reward are bit-identical, on any machine, forever | free | always, unconditionally |
| **T2 run-exact** | T1, plus every gradient, every parameter and every metric row bit-identical on the same device class and torch/CUDA build | 10-20% throughput **[A]** | **default**, `determinism.tier = "run_exact"` |
| **T3 throughput** | T1 only; the learner may use nondeterministic kernels and curves agree in distribution | fastest | opt-in, `tier = "throughput"`, stamped on every metric row and every checkpoint |

`tier` is part of the run identity. A T3 run may not be resumed into a T2 run without
`--allow-identity-drift`, which writes `drift.json` naming every differing field and marks every
later metric row `resumed_with_drift=true`.

T2 keeps bf16 autocast. Deterministic kernels are bit-reproducible run to run at any precision, so
the tier costs nothing in precision and buys exactly what it says: two runs of the same identity agree
row for row. It is not a claim that two differently shaped forwards *within* one run agree, which is a
separate matter handled in sections 8.2 and 9.6.

`determinism.apply(tier)` for T2:

```python
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False     # tf32 is not bit-reproducible across shapes
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(config.determinism.torch_threads)      # default 1
```

`CUBLAS_WORKSPACE_CONFIG=:4096:8` must be set before torch initialises CUDA, so `cli.py` sets it from
the config **before importing torch** and `determinism.apply` asserts it is already set, raising a
message that names the entry point if not. Every worker sets `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 before importing numpy: BLAS thread count changes
float reduction order, and the workers are engine-bound rather than BLAS-bound, so it costs nothing.

### 5.2 Name-addressed seeding

`royalelearn/seeding.py`:

```python
def derive_seedseq(master_seed: int, path: str) -> np.random.SeedSequence:
    digest = hashlib.blake2b(path.encode("utf-8"), digest_size=16).digest()
    return np.random.SeedSequence(entropy=master_seed, spawn_key=struct.unpack("<4I", digest))

def derive_generator(master_seed: int, path: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seedseq(master_seed, path)))

def derive_int(master_seed: int, path: str) -> int:
    """A reproducible 63-bit integer, for env seeds."""
    w = derive_seedseq(master_seed, path).generate_state(2, dtype=np.uint32).astype(np.uint64)
    return int((w[0] | (w[1] << np.uint64(32))) >> np.uint64(1))
```

The namespace is fixed and documented in `docs/determinism.md`:

| path | consumer |
|---|---|
| `env/worker/{w}/shard/{s}/gen/{g}` | the one `ClashSelfPlayVecEnv.reset(seed=...)` for that shard; `g` is the respawn generation |
| `match/battle/{b}/ordinal/{k}` | the matchmaker's draw for episode `k` of battle `b` |
| `act/iteration/{i}/cycle/{t}` | the `(R,)` uniform vector that drives action sampling at cycle `t` |
| `scripted/worker/{w}/slot/{r}/gen/{g}` | a worker-side scripted opponent's generator |
| `ppo/minibatch/iteration/{i}/epoch/{e}` | the minibatch permutation |
| `eval/seed_set` | the fixed evaluation seed set, drawn once at run start |
| `eval/match/{comparison_id}/{seed_index}/{side}` | one evaluation battle |
| `eval/bootstrap/{comparison_id}` | the bootstrap resampling |
| `torch/init` | network initialisation |
| `torch/global`, `torch/cuda` | `torch.manual_seed` / `torch.cuda.manual_seed_all` at start-up |

Two rules follow, both enforced by tests:

- **No component may call a global RNG.** `tests/test_no_global_rng.py` scans the package source and
  fails on `np.random.` not followed by `Generator|PCG64|SeedSequence|default_rng`, on a bare
  `random.`, and on unseeded `torch.rand*`.
- **The matchmaker draws from `match/...`, never from an env's generator.** Otherwise changing the
  opponent mixture changes the battle. Because the path is the battle and its ordinal rather than the
  iteration and the slot, an episode meets the same opponent however the iteration boundary happens to
  fall across it.

`ClashSelfPlayVecEnv` autoresets without a seed (`env.py:509-510`), so an episode's identity is a
function of its shard seed and its reset ordinal rather than of a seed of its own. That is
deterministic and sufficient: the worker records the shard seed path and the reset ordinal on every
`EpisodeRecord`, which is what `royalegym.replay` needs to re-simulate one battle. What it is not is
episode-addressable without replaying the battles before it, which is what the
`autoreset_seed_fn(game_index, ordinal)` hook gives it: an episode's seed is
`derive_int(master_seed, "env/battle/{b}/ordinal/{k}")`, so it is addressed by name rather than
by how many episodes came before it.

### 5.3 Run identity

`royalelearn/identity.py`:

```python
class EngineBuild(msgspec.Struct, frozen=True):
    engine_class: str                 # "RustEngine" | "MockEngine"
    calibration_digest: str           # config()["calibration_digest"]: 16 hex over the values loaded
    build_digest: str                 # config()["build_digest"]: the same hash over the data
                                      # compiled into the extension; equal to the above on a fresh
                                      # build, which is the whole point of having both
    catalogue_sha256: str             # over [(card.name, card.card_id, card.elixir, card.kind), ...]
    path_search: str | None
    stale_build_differences: list[str] # MUST be empty; recorded so a failure is auditable

class RunIdentity(msgspec.Struct, frozen=True):
    format_version: int               # of this struct; currently 1
    royalelearn_version: str; royalelearn_git: str
    royalegym_version: str;  royalegym_git: str
    engine_build: EngineBuild
    env_spec_digest: str              # sha256 of the canonical EnvFactorySpec JSON
    obs_digest: str; action_digest: str
    frame_stack: int                  # obs.frame_stack; it changes the trunk's input width
    arch_digest: str
    codec_version: int; codec_table_digest: str   # the version and the table it produced
    algo_digest: str                  # sha256 over the PPO, GAE and schedule config
    rollout_digest: str               # sha256 over {workers, games_per_worker, shards, T, R}
    ladder_digest: str                # sha256 over {mixture, gate params, eval seed set sha}
    master_seed: int
    determinism_tier: str
    torch_version: str
    device_kind: str                  # "cuda:NVIDIA GeForce RTX 3050 Laptop GPU:sm_86" | "cpu:x86_64"

run_id = sha256(msgspec.json.encode(identity, order="deterministic")).hexdigest()[:16]
```

Included even though they look cosmetic, because each changes the byte stream: `master_seed`,
`workers`, `games_per_worker`, `shards_per_worker`, `timesteps_per_iteration`, `determinism_tier`,
`device_kind`. Excluded and merely recorded, because none of them can change a number: `run_name`,
`runs_dir`, wandb settings, console verbosity, `checkpoint.keep`, alarm thresholds and severities
(an alarm can halt a run or warn, never alter a value). A resume diffs the excluded fields and prints
every difference; a difference in an included field is refused by name.

Both digests come from `ClashParallelEnv.config()`, which reports `calibration_digest` over the
calibration values as loaded and `build_digest` over the copies compiled into the extension —
`RustEngine.build_digest()` is the same number read directly off the engine. The harness hashes no
data file of its own: a digest computed here from `royalegym.protocol.data_dir()` would be a second
opinion about what the engine is running on, and a second opinion is exactly what a stale build looks
like. When they differ, `stale_build_differences` carries the engine's own list verbatim and the run
refuses to start (section 7.7).

### 5.4 What a resume reproduces, operationally

After loading a checkpoint written at the end of iteration `k`:

1. The checkpoint's `RunIdentity` equals the identity computed from the current config and
   environment, field for field, or the load is refused with every differing field named.
2. Every RNG state is restored rather than re-seeded: torch CPU, torch CUDA on every device, the
   minibatch generator's `bit_generator.state`, python `random`, and each shard's seed path, reset
   ordinal and respawn generation.
3. Schedule positions are restored: current `gamma`, `ent_coef`, `ent_coef_noop`, both learning
   rates, the lr-backoff counter, `cumulative_env_steps`, `cumulative_timesteps`, `iteration`,
   `cumulative_updates`.
4. The learner is byte-identical to what the checkpoint recorded: the `state_digest` reported on
   load equals the one written at iteration `k`. That is the weights, both optimizers' moments,
   the return scaler and the schedule positions, so it is the whole learner rather than a curve
   that resembles one.

5. Iteration `k+1` reproduces the original's metric row, field for field, **when the checkpoint
   was written at an episode boundary**.

**Where that last clause stops, and why it stops there.** An episode is addressed by its battle
and its ordinal: `ClashSelfPlayVecEnv` takes an `autoreset_seed_fn`, the harness gives it
`derive_int(master_seed, "env/battle/{b}/ordinal/{k}")`, and a resumed worker is put back on the
ordinal the original was about to play through `set_episode_ordinals`. Both counters are restored
— the environment's, which decides which battle is played, and the matchmaker's, which decides who
plays it — because restoring one without the other gives the right battles against the wrong
opponents or the reverse.

What is not restored is an episode that was *half finished* when the checkpoint was written. Its
transitions were already in the original run's buffer, and replaying it would count them twice, so
a resumed run starts the next episode instead. The battles it then plays are the right ones and
their phase is not: the rows differ in how many episodes completed, never in which were played.
Two tests draw that line — one asserts the row-for-row continuation from a boundary, the other
measures the phase difference from inside an episode, and the second fails if resuming mid-episode
is ever offered. The warm-up that spreads first-episode phases apart is skipped on a resume for
the same reason: it would move every battle off the episode it is continuing.

`state_digest`, defined once in `royalelearn/checkpoint.py`, is a sha256 over, in this fixed order:
the actor `state_dict` tensors (name-sorted, `.cpu().numpy()` bytes), the critic `state_dict`, each
optimizer's numeric `state_dict` leaves, the return-Welford triple, and the schedule positions. It is
logged every iteration, so when two runs do diverge the iteration at which they first differ is a
lookup rather than an investigation.

---

## 6. Configuration

`royalelearn/config.py`. `msgspec.Struct` throughout, every model `forbid_unknown_fields=True` so a
typo is an error rather than a silently ignored key, one JSON file, hashed with sha256 over the
canonical encoding. msgspec rather than pydantic because it is already a RoyaleGym dependency, it
gives forbidden-unknown-fields, tagged unions and a canonical JSON round-trip, and the package's
dependency list stays `royalegym, numpy, msgspec`.

There is one source of truth. The config object is it; there is no keyword-argument facade that
duplicates its leaves, because two sources of truth for the same number is the kind of variable this
harness exists to remove.

```python
class RunConfig(Struct, forbid_unknown_fields=True):
    format_version: int = 1
    run_name: str = "royalelearn"                # cosmetic, not in the identity
    runs_dir: str = "runs"                       # cosmetic
    master_seed: int = 20260921                  # IN the identity
    profile: str = "laptop"                      # selects the shipped defaults; all still overridable
    timestep_limit: int = 100_000_000            # learner transitions
    env: EnvFactorySpec
    eval_env: EnvFactorySpec | None = None       # None -> env with truncation removed
    extra_component_modules: list[str] = []
    obs: ObsConfig = ObsConfig()
    rollout: RolloutConfig = RolloutConfig()
    net: NetConfig = NetConfig()
    ppo: PPOConfig = PPOConfig()
    advantage: AdvantageConfig = AdvantageConfig()
    ladder: LadderConfig = LadderConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    metrics: MetricsConfig = MetricsConfig()
    alarms: AlarmConfig = AlarmConfig()
    determinism: DeterminismConfig = DeterminismConfig()
    doctor: DoctorConfig = DoctorConfig()
```

| field | default | unit | why |
|---|---|---|---|
| **`obs`** | | | |
| `frame_stack` k | 1 | cycles | the observation carries positions and no motion, so a policy that needs to tell an advancing unit from a retreating one needs more than one frame. Stacking is a gather of buffer rows (section 9.1), costs no extra storage and adds `S + hand_size` input channels per extra frame. 1 by default because the extra channels cost stem compute and the first run measures whether they buy anything |
| **`rollout`** | | | |
| `source` | `"process"` | `"process"` / `"inline"` | inline is the semantic reference and runs every fast test |
| `workers` | 3 | processes | 6-8 threads; the parent needs one for the CUDA driver and one for the loop, and the learner is the bottleneck so more workers buy little |
| `games_per_worker` | 32 | battles | with `workers=3`, 96 battles and 192 slots |
| `shards_per_worker` | 2 | vec envs | a worker steps one shard while the parent infers on the other. Worth `min(env, inference)` per round and no more, because the two are additive (section 2.2); `bench` prints what it is worth here, and 1 is the right value when that is under about 10% |
| `spin_us` | 100 | microseconds | spin before blocking on the round events; a Windows Event round trip measures ~40-80 microseconds **[A]**, a spin about 2 |
| `round_timeout_s` | 30.0 | seconds | **every** wait has a timeout; a timeout is a typed `WorkerFailure`, never a block |
| `restart_failed_workers` | `true` | | |
| `max_restarts_per_worker` | 3 | | three restarts of one worker in a run raises rather than silently degrading throughput |
| `launch_delay_s` | 0.5 | seconds | three `RustEngine` constructions at once each decode `calibration.json` and `arena.json` |
| `stagger_first_reset` | `true` | | desynchronise episode phase across battles at run start, so episode ends spread across cycles |
| `overlap` | `false` | | lag-1 collection under the update: about 1.37x wall clock for a second 623 MB buffer. Off on 8 GB; on in the workstation profile |
| `eval_workers` | 2 | processes | the gate's farm, built at gate time and closed after |
| `eval_games_per_worker` | 24 | battles | |
| **`net`** (`ArchSpec`) | | | |
| `channels` C | 64 | | about 430 k parameters per network; 223 TFLOP per iteration is about 25 s bf16 on a 3050 |
| `blocks` N | 4 | residual blocks | |
| `norm_groups` | 8 | | GroupNorm, never BatchNorm: BatchNorm computes a different function at rollout than at update, which breaks the stored-log-prob contract |
| `vector_embed` | 32 | channels | the scalar vector is broadcast into the map so elixir can modulate every tile |
| `value_hidden` | 256 | | |
| `card_embed` | 64 | must equal `channels` | the pointer inner product |
| `coord_conv` | `true` | | the arena is not translation-invariant: own half, enemy half, the river, the bridges, the tower rects |
| `separate_trunks` | `true` | | makes "the mask and the entropy bonus never touch the critic" structural |
| `policy_head` | `"pointer"` | `"pointer"` / `"slot_conv"` / `"flat_mlp"` | the latter two exist as ablations |
| `logit_scale` | `"rsqrt_c"` | | one over the square root of C on the pointer inner product |
| `init` | `"orthogonal"` | | gain sqrt(2) hidden, **0.01 policy head**, 1.0 value head |
| `noop_bias` | 0.0 | logits | self-limiting at init; the formula for when it is needed is in section 8.4 |
| `autocast_dtype` | `"bfloat16"` | | Ampere bf16 tensor cores; bf16 keeps fp32's exponent range so `finfo.min` masking is safe and no `GradScaler` is needed |
| `device` | `"cuda"` | | |
| **`ppo`** | | | |
| `n_epochs` | 3 | | the family disagrees by three orders of magnitude (1, 10, about 1000 gradient steps per 50 k), so none of them is evidence; 3 balances reuse against the update being the wall-clock bottleneck. Watch clip fraction across epochs |
| `timesteps_per_iteration` | 32 768 | timesteps | the compute budget and the episode arithmetic land on the same number independently |
| `batch_size` | 4 096 | samples per optimizer step | 24 optimizer steps per iteration; about nine episodes' worth of terminal signal per step, so the advantage mean is not dominated by one battle |
| `minibatch_size` | 512 | samples per forward | the largest that fits 4 GB under bf16 with cuDNN workspace. A **pure memory knob**: gradients accumulate weighted by `n/batch_size` with one `optimizer.step()` per batch, and a test proves the accumulated gradient equals the full-batch one |
| `clip_range` | 0.2 | | the PPO paper; all three references |
| `dual_clip_c` | 3.0 | | for a negative advantage the standard minimum does not bound the loss below, and in a 2305-way masked space a rarely-sampled action's ratio can be enormous |
| `vf_coef` | 1.0 | | separate networks and separate optimizers, so it only scales the critic's effective learning rate |
| `value_clipping` | `false` | | |
| `ent_coef` | `Linear(0.01 -> 0.003 over 30e6 env steps)` | | the community range across the three references; exploration matters less as the pool strengthens |
| `ent_coef_noop` | `Linear(0.02 -> 0.0 over 10e6 env steps)` | | a warm-start guard against no-op collapse, applied to a binary entropy bounded by 0.693 nats, so while it is on it needs a larger coefficient than the joint term. It anneals to zero because `H2(p_noop)` is maximised at `p_noop = 0.5` while a healthy policy sits near 0.94: a constant coefficient would bias every converged policy toward overplaying. The alarms outlive the schedule |
| `max_grad_norm` | 0.5 | | applied to the actor and the critic parameter sets **separately** |
| `lr_actor`, `lr_critic` | 2e-4 | | the community moved down from 3e-4 for long runs |
| `adam_eps` | 1e-5 | | the PPO paper's epsilon, not torch's 1e-8 |
| `lr_backoff.kl_threshold` | 0.02 | | |
| `lr_backoff.patience` | 3 | iterations | |
| `lr_backoff.factor` | 0.5 | | |
| `lr_backoff.lr_min` | 2e-5 | | |
| `advantage_standardization` | `true` | | once per **iteration**, over all trainable cells, before any split |
| `critic_chunk` | 1024 | rows | the whole-iteration critic pass is a 2.6 GB spike at this row size if it is not chunked |
| `discard_opponent_rows` | `true` | | the frozen seat's log-probs came from other weights; V-trace is the wrong tool against a multi-generation-old snapshot |
| `keep_previous_iterations` | 0 | iterations | the buffer is cleared every iteration: each timestep is trained on exactly `n_epochs` times and discarded. A rolling window is available at a linear memory cost |
| `debug_assert_iterations` | 10 | iterations | the two mask asserts and the ratio invariant run for this many iterations from a run's start |
| `check_ratio_invariant_every` | 50 | iterations | thereafter |
| `ratio_atol` | `{"fp32": 1e-4, "bfloat16": 2e-2}` | | selected by `net.autocast_dtype`. 0.0 is not claimed and neither is 1e-4 under autocast: the rollout forward and the update forward are different batch shapes, cuDNN picks a kernel per shape, and bf16 carries three significant digits, so the fp32 logits at the end of the two forwards differ at the 1e-2 level. Section 8.2 says what the check does and does not detect at that tolerance |
| **`advantage`** | | | |
| `gamma` | `Geometric(0.997 -> 0.999 over 20e6 env steps)` | | 0.99 (both reference learners' default) discounts the win to 0.027 by the end of regulation, wrong by about 25 times in terminal reach. The anneal follows OpenAI Five's 0.998 to 0.9997, with an endpoint scaled to a four-minute match |
| `gae_lambda` | **0.99** | | one over one minus gamma times lambda is 91 steps, which is **45.5 s**, matching the deploy-push-tower causal chain. The references' 0.95 gives 9.8 s and cannot connect a deploy to the tower it takes. The single highest-leverage number in the config, which is why `credit_horizon_seconds` is logged every iteration |
| `standardize_rewards` | `true` | | gamma near 1 inflates return magnitudes roughly tenfold over gamma = 0.99 |
| `reward_clip` | 10.0 | standard deviations | independent of the standardisation |
| **`ladder`** | | | see section 11 |
| `mix` | `(0.50, 0.35, 0.15)` | mirror / pool / scripted | |
| `max_resident_opponents` | 2 | snapshots | at most four batched forwards per shard-round, and an LRU that never thrashes |
| `pfsp_weighting` | `"hard"` | | training wants opponents that still beat the learner; evaluation uses `"variance"` |
| `pfsp_power` | 2.0 | | |
| `pfsp_uniform_floor` | 0.2 | | the anti-forgetting guard AlphaStar's forgotten-players slice does |
| `weight_floor_scale` | 0.01 | | every snapshot's weight floored at this over the pool size |
| `candidate_every_env_steps` | 4 000 000 | game-steps | about two hours on the laptop, so the gate is about 5% of wall clock |
| `floor_admit_every_env_steps` | 50 000 000 | game-steps | a plateau cannot starve the pool |
| `pool_working_size` | 48 | snapshots | sampling is linear in the pool per episode, in python |
| `eval_seed_count` | 500 | seeds | |
| `release_mode` | `"stochastic"` | | rating both the sampled and the argmax variant doubles the pool and the cost for no decision |
| `refit_every_iterations` | 10 | | |
| `gate.champion_games` | 1000 | battles | 500 seeds times 2 side assignments |
| `gate.champion_lower_bound` | 0.52 | score rate | requires an observed 55.2% or better, about 35 Elo |
| `gate.anchor_games` | 200 | battles per anchor | |
| `gate.anchor_tolerance_pp` | 2.0 | percentage points | |
| `gate.stratified_snapshots` | 8 | | |
| `gate.stratified_games` | 100 | battles each | |
| `gate.bootstrap_resamples` | 10 000 | | |
| `rater.prior_sd` | 400.0 | Elo | weakly informative; keeps the fit finite for a player with no losses |
| `rater.anchor` | `"scripted:noop"` | | the gauge exists from the first game, before any snapshot |
| `rater.draws` | `"davidson"` | `"davidson"` / `"half_win"` | fall back if the measured draw rate is under 2% |
| **`checkpoint`** | | | |
| `every_env_steps` | 2 000 000 | game-steps | |
| `keep` | 10 | checkpoints | |
| `include_buffer` | `false` | | the iteration boundary is the resume point; there is no mid-iteration state to preserve |
| `strict_load` | `true` | | tolerate-everything is right for a research tool and wrong for a harness that promises the curve continues |
| **`metrics`** | | | |
| `sinks` | `[Jsonl, Console, Viser, Wandb(enable=false)]` | | `JsonlSink` is always installed: it is the file the resume test compares. `ViserSink` costs one clock read per iteration while no viewer is attached |
| `image_every` | 50 | iterations | per-card tile heatmaps |
| `keep_episode_log_iterations` | 200 | iterations | after which `episodes.jsonl` is gzipped in place |
| **`determinism`** | | | |
| `tier` | `"run_exact"` | | section 5.1 |
| `torch_threads` | 1 | | thread count changes CPU reduction order |
| **`doctor`** | | | |
| `ram_budget_mb` | 6500 | MB | of 7800 |
| `run_mask_disagreement_gate` | `true` | | exhaustive over all 2304 non-no-op actions |

`EnvFactorySpec` (`rollout/envspec.py`) is the JSON-able env description, simultaneously the thing
sent to workers, the thing recorded in the checkpoint and the thing the ladder's `context` string is
derived from:

```python
class ComponentSpec(Struct, frozen=True):
    cls: str                                 # "royalegym.obs.SpatialObsBuilder"
    kwargs: dict[str, JsonValue] = {}

class EnvFactorySpec(Struct, frozen=True):
    engine: ComponentSpec
    obs_builder: ComponentSpec
    action_parser: ComponentSpec
    reward_fn: ComponentSpec
    state_mutator: ComponentSpec
    termination: list[ComponentSpec]
    truncation: list[ComponentSpec] = []
    decision_ms: int = 500
    def build_vec(self, num_games: int) -> "ClashSelfPlayVecEnv": ...
    def digest(self) -> str: ...             # sha256 of the canonical JSON
```

`build_vec` resolves each `ComponentSpec` to a class and hands the result to RoyaleGym's picklable
`EnvFactory(engine=..., obs_builder=(cls, kwargs), decision_ms=...)` recipe rather than constructing a
`ClashParallelEnv` itself. One place knows how a ClashParallelEnv is assembled, and it is the place
that also has to keep it picklable for `spawn`; a second assembly here would be a copy that drifts.
`EnvFactorySpec` stays the JSON layer above it — the thing sent to workers, recorded in the checkpoint
and hashed into the ladder's `context` — and `ClashParallelEnv.config()` read back off the built env
is what proves the two agree.

Only classes on an allow-list (`royalegym.*`, `royalelearn.*`, plus anything named in
`config.extra_component_modules`) may be instantiated. A spec is data that arrives from a config file
and later from a checkpoint, and `importlib` on an arbitrary string is a code-execution surface.

---

## 7. The rollout path

### 7.1 Process model

```
parent (learner: torch, CUDA, buffer, ladder, metrics, checkpoints)
  |-- worker 0  (python, numpy, royalegym, royalesim; NO torch)
  |     |-- shard 0: ClashSelfPlayVecEnv(num_games=M/2, env_fn=EnvFactorySpec.build)
  |     '-- shard 1: ClashSelfPlayVecEnv(num_games=M/2, ...)
  |-- worker 1 ...
  '-- worker K-1 ...
```

- Start method **`spawn`** on every platform. Windows has no `fork`, CUDA is already initialised in
  the parent so `fork` would be wrong anyway, and forcing `spawn` everywhere means the worker's
  import-time environment is identical on Linux.
- The child is handed an `EnvFactorySpec`, a msgspec Struct, never a pickled closure. That removes
  the reference learner's undocumented 4096-byte factory limit and gives the checkpoint its env
  description for free.
- The child's preamble, in this order, before numpy is imported:

```python
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl-C is the parent's business
```

- `launch_delay_s` between starts: `K` `RustEngine` constructions at once each decode
  `calibration.json` and `arena.json`.
- The child holds no policy. `tests/test_worker_hygiene.py` asserts `"torch" not in sys.modules`
  after start-up.

### 7.2 The observation codec

`rollout/codec.py`, `SpatialObsCodec`, `codec_version = 1`. The codec version is the *rule*; the
*table* it produces is computed at preflight from `EnvSpec.obs_space` and a sample of real
observations, so a plane added, removed or rescaled upstream reaches the right storage without a line
changing here.

The rule, applied per key:

| key | rule | storage |
|---|---|---|
| `spatial`, per plane | declared static by `ObsBuilder.spatial_layout()` | **not stored**; held once per seat in `statics` |
| `spatial`, per plane | declared `high <= 255` and integer-valued on 1000 sampled states | `uint8`, scale 1, **exact** |
| `spatial`, per plane | anything else | `float16` |
| `mask_planes` | equal to `action_mask[1:]` reshaped, by construction | **never stored**; the learner reshapes the stored mask at unpack |
| `vector` | bounded in [0, 1] by the builder | `float16`, 2 B per element |
| `action_mask` | | bit-packed `uint8`, LSB first, `ceil(n_actions / 8)` B, **exact** |

On today's 95-card catalogue with `Reveal` off, `S = 20` and that rule splits them 16 / 2 / 2: sixteen
`uint8` planes, the two hit-point planes as `float16`, and the two static planes. The hit-point planes
are the only ones whose declared `high` reaches 64 — every other plane is a small integer count or an
indicator in [0, 1] — and they are also the only ones that fail the integer test, because they carry
a fraction of full health; it is the second fact and not the first that sends them to `float16`. The
vector is 1 177 wide. Turning on `Reveal.enemy_spell_aim` adds a spatial plane, and with it one more
`uint8` row; the table below is then recomputed from the rule rather than patched.

| part | stored as | bytes | exact? |
|---|---|---|---|
| 16 count and indicator planes | `uint8`, scale 1 | 9 216 | **exact**: the declared bound is under 255 and the sampled values are integers |
| 2 hit-point planes | `float16` | 2 304 | exact to fp16; the values are hit points scaled into a small range, so the granularity is below the engine's own |
| 2 static planes | **not stored** | 0 | exact |
| 4 mask planes | **not stored**, derived from the mask | 0 | exact |
| vector, V = 1177 | `float16` | 2 354 | exact to fp16; the builder clips the whole vector to [0, 1] (`obs.py:181`) |
| action mask | bit-packed `uint8` | 289 | exact |
| **row total** | | **14 163 B** | 3.91x smaller than what the env hands over |

The same rule on `MockEngine`'s 16-card catalogue gives `9 216 + 2 304 + 2*229 + 289 = 12 267 B`, and
the whole test suite runs there precisely because its widths are not the Rust catalogue's.

Notes an implementer needs:

- The table is **printed at preflight**, written into the metric stream once, hashed into
  `codec_table_digest` and stored in the `spec.json` of every snapshot. Two runs whose tables differ
  are not comparable and the digest is what says so.
- **`mask_planes` is never stored.** RoyaleGym guarantees
  `mask_planes == action_mask[1:].reshape(hand_size, tiles_y, tiles_x)` exactly, with its own test;
  the learner does `mask[1:].view(...)` at unpack. Storing them would be 2 304 B of a 14 163 B row
  spent on a reshape.
- **A static plane is one the layout declares static, and nothing else.** On this catalogue those are
  the water and no-deploy masks. There is no sampling fallback and there must not be: the tower planes
  are constant across any sample in which no tower falls, so a rule that promoted "unchanged on a
  thousand states" to "static" would freeze a tower at full health for the rest of the run and the
  policy would never see one die. Storage is decided from a sample; *existence* is decided from the
  declaration.
- The static planes are asserted equal between the blue and the red row at start-up. They are, because
  the observation is in the acting player's own frame and the arena is symmetric under the 180-degree
  seat rotation; if they ever differ the codec stores one pair per seat and says so in a warning.
- A `uint8` plane whose value exceeds 255 is clipped and counted in `health/obs_codec_clipped`. A
  non-zero value is a bug report, not a tolerance: the plane was admitted to `uint8` on a declared
  bound, so a clip means the bound was wrong.
- `unpack_to_device` runs on the GPU: `uint8 -> float32` times a per-channel constant divisor,
  `float16 -> float32`, bit-unpack the mask with shifts, reshape it into the mask planes, scatter the
  static planes in, and gather the frame-stack history rows. Its FLOP count is nothing against
  378 MFLOP per sample.
- The per-channel divisors are **fixed constants from the declared bounds**, not running statistics.
  Observations are bounded by construction, Welford standardisation would divide a wide one-hot by a
  near-zero standard deviation and amplify noise that was not there, it adds cross-worker mutable
  state that a resume has to restore, and it makes the seat-mirror bit-identity unverifiable. The
  adaptive part is done by the stem's GroupNorm, which has no cross-worker state.

### 7.3 The boundary is a byte layout

`rollout/layout.py` is the cross-language contract. It defines, as module constants plus
`LAYOUT_VERSION: int = 1`:

```
Segment A   "royalelearn-buf-<run_id>"          the experience buffer: learner-owned, worker-written
    header    128 B    magic, LAYOUT_VERSION, cycles, n_slots, row_bytes, obs offsets, codec_version
    obs       (T+1) * R * row_bytes             the rectangle; index(t, r) = t * R + r

Segment B   "royalelearn-ctl-<run_id>-<w>"      one per worker, covering all its shards
    per shard, per parity p in {0, 1}:
        control  64 B    state u32 | cycle u64 | n_slots u32 | err_code u32 | err_len u32 | t_env_ns u64
        error   512 B    the child's traceback, utf-8
        scalars  n_slots_shard * 32 B
                         reward f32, group i8, flags u8 (terminated, truncated, valid),
                         deploy_status i8, tick i32, episode_end i8, episode_steps i32,
                         cards_played i16, elixir_leak i16, pad
        finals   n_trunc_max * row_bytes        final_obs of truncated rows, for the bootstrap
        actions  n_slots_shard * 2 B            int16, parent -> child
        assign   n_slots_shard * 4 B            group i8, opponent_ix i8, learner_seat i8, pad;
                                                parent -> child, written only for the slots whose
                                                episode started on the previous round
    plan       n_slots * 24 B                   the iteration's opening table: role i8, group i8,
                                                opponent_ix i8, seed hash u64
```

Signalling: two `multiprocessing.Event`s per (worker, shard), `obs_ready` and `actions_ready`, with a
spin-then-block wait of `rollout.spin_us` on both sides. There is no UDP, no magic float header, no
pickling on the hot path and no length-prefixed stream to desynchronise.

Error signalling: `err_code` 0 normal, 1 python exception with the message in `error`, 2 hard crash.
The parent turns either into a `WorkerFailure` delivered to the coordinator, which restarts the
worker. Both reference learners treat a dead worker as a permanent silent hang.

Because the contract is bytes, a Rust worker is a drop-in: it writes the same header, the same packed
rows and the same scalars, and flips the same events. `tests/test_rollout_farm.py` is the
acceptance criterion — the process farm and the inline source must produce byte-identical buffers,
scalars and episode records from the same seed over 30 cycles.

### 7.4 The worker main loop

```python
def worker_main(w: int, cfg: WorkerConfig, handles: Handles) -> None:
    _set_thread_env()                             # before numpy
    import numpy as np
    shards = [build_shard(cfg, w, s) for s in range(cfg.shards_per_worker)]
    _startup_gates(shards)                        # section 7.7
    for sh in shards:
        seed = derive_int(cfg.master_seed, f"env/worker/{w}/shard/{sh.index}/gen/{cfg.generation}")
        sh.obs = sh.vec.reset(seed=seed)
        publish(sh, cycle=-1)                     # hand the parent the first observations
    while True:
        for sh in shards:                         # strict alternation
            if not wait_actions(sh, cfg.spin_us):
                continue
            cmd = read_command(sh)                # STEP | PLAN | SET_STATE | SPACES | DEFER | CLOSE
            if cmd is CLOSE:
                return
            if cmd is PLAN:
                apply_plan(sh, read_plan(sh)); continue      # the iteration's opening table
            apply_assignments(sh, read_assignments(sh))      # the parent's draw for the battles
                                                             # that just started a new episode
            actions = read_actions(sh)                       # int16[n_slots_shard]
            actions = apply_scripted(sh, actions)            # worker-side opponents, numpy only
            obs, rew, term, trunc, info = sh.vec.step(actions)
            pack_round(sh, obs, rew, term, trunc, info)      # codec -> the buffer rectangle
            publish(sh)
```

- **The scripted opponents run here.** `RandomLegalOpponent(noop_prob=0.9)` is about 5 microseconds
  of numpy per row; routing 15% of battles through the GPU would cost a forward pass and a boundary
  crossing for nothing. They draw from a per-slot generator seeded from
  `scripted/worker/{w}/slot/{r}/gen/{g}`, so they stay reproducible.
- **A battle's assignment changes only at that battle's episode boundary, and the parent decides
  it.** The worker holds no pending plan and makes no draw: when a round reports `episode_end`, the
  parent draws the next assignment and hands it back on the same round's `Step`, before the first
  action of the new episode is taken. Partial control makes a trajectory unusable, and one place that
  knows what a battle is playing removes the whole class of bug rather than detecting it.
- **`vec.action_masks()` is never called** — it restacks the whole batch; the mask is already in
  `obs["action_mask"]`. `info["action_mask"]` is ignored for the same reason: it is a second copy of
  a key the observation already carries.
- **The terminal statistics are read, not reconstructed.** RoyaleGym puts seven flat scalars in
  `info` on the step an episode ends — `episode_steps`, `episode_ticks`, `own_crowns`,
  `enemy_crowns`, `own_tower_hp_frac`, `enemy_tower_hp_frac`, `elixir_leak_steps` — and the vec env
  nests them under `infos["final_info"]` as batched arrays with `_`-prefixed validity masks, so the
  worker reads `infos["final_info"]["_own_crowns"]` to find which rows have one and copies the seven
  values for those rows into their `EpisodeRecord`. The hit-point fractions are already the mean over
  a player's three towers. What the worker still counts itself is what no one else can: cards played,
  illegal commands, the undiscounted return and the per-term reward sums. A per-step counter in numpy
  costs nothing and one fixed record per finished episode crosses the boundary, rather than a per-step
  metrics array reassembled in the parent.

### 7.5 The round protocol

Per shard-round the parent does:

```
round  = source.next_round(timeout_s)                # spin/Event wait; views into shared memory
assign:                                              # the parent owns assignments
    for b in battles_that_ended(round.episode_end):   # ascending battle order; ends come in pairs
        a = matchmaker.assign(b, ordinal[b] + 1, pool, ratings)   # match/battle/{b}/ordinal/{k}
        apply(a)                                     # updates group for b's two slots, in place
group  = round.group                                 # the worker's report, with this round's
                                                     # new assignments already folded in
route:
    for gid in sorted(set(group)):                   # sorted: determinism, not arrival order
        rows = np.flatnonzero(group == gid)          # ascending slot order within the group
        if gid == SCRIPTED: continue                 # the worker fills these in
        obs    = codec.unpack_to_device(buffer_rows(round, rows), statics)
        logits = policy_for(gid).logits(obs).float()
        dist   = MaskedCategorical(logits, obs.mask)
        u      = uniforms[round.cycle][round.slots[rows]]     # act/iteration/{i}/cycle/{t}
        a      = dist.sample(u)
        actions[rows] = a
        if gid == LEARNER:
            log_probs[rows] = dist.log_prob(a)
source.submit(Step(actions=actions, gamma=sched.gamma,
                   group=new_group, opponent_ix=new_opponent_ix,
                   learner_seat=new_learner_seat))   # only the battles that just reset
buffer.record_round(round, actions, log_probs)       # scalars only; observations already in place
```

The assignment step comes **before** routing, in the same round, so the first observation of a new
episode is already routed by that episode's own assignment and no transition is ever produced under a
stale one. The worker is told the result because it fills the scripted seats itself; it is not asked
to decide anything.

Row order inside an inference batch is ascending slot index, never arrival order. Group sizes vary
from round to round because assignments change at episode boundaries; that costs a little kernel-shape
churn and nothing else, and it is the price of binding a policy to a battle for a whole episode, which
is the thing that must not be given up.

`uniforms[t]` is an `(R,)` float32 vector drawn once per cycle from `act/iteration/{i}/cycle/{t}` and
indexed by slot, so a sampled action is a pure function of `(master_seed, iteration, cycle, slot)` and
the logits. That is what makes the trajectory reproducible whatever the batch composition and whatever
`rollout.overlap` is set to.

Frozen snapshot policies run under `torch.inference_mode()` in fp16, from a `LoadedPolicyCache` of at
most `ladder.max_resident_opponents + 2` resident modules.

### 7.6 Failure, timeouts and restart

```python
def next_round(self, timeout_s: float) -> RolloutRound:
    deadline = time.monotonic() + timeout_s
    while not self._shard_ready(shard):
        if time.monotonic() > deadline:
            self._reap()                       # is_alive() on every child; exitcode into the log
            raise WorkerTimeout(self._diagnose())
        self._spin_or_block(shard)
```

Four properties, none of which either reference learner has:

- **Every wait has a timeout.** `round_timeout_s = 30`, `join_timeout_s = 10`.
- **`is_alive()` is checked on every timeout tick** and a dead child's exit code is logged.
- **A dead worker is restarted deterministically**: the new worker gets
  `env/worker/{w}/shard/{s}/gen/{g+1}`, so the run remains reproducible from the master seed and the
  restart count, both of which are checkpointed.
- **`health/worker_restarts` is a metric**, and `max_restarts_per_worker` restarts of one worker
  raises rather than silently degrading throughput.

The rectangle handles a mid-iteration death without a special case. If worker `w` dies at cycle `t*`,
its slots have no observation at `t*`, so for those slots `valid[t, r] = (t <= t* - 2)` and the
transition at `t* - 2` is marked `truncated` and bootstrapped from `values[t* - 1]`, which exists.
`health/rows_dropped_dead_worker` counts the lost cells. The restarted worker rejoins at the next
iteration boundary.

On shutdown: send `Close`, `join(timeout=10)`, then `terminate()`, then `kill()`. `close()` is
idempotent and is called from a `finally` in the coordinator and from an `atexit` hook.

### 7.7 Start-up gates

`rollout/preflight.py`, run by `royalelearn doctor` and by the coordinator before the first cycle.
Their results go into the run identity and the checkpoint.

1. **Construct one engine and reset it once.** `RustEngine()` raises `RuntimeError` listing every
   stale calibration key. Re-raise as `StaleEngineBuild` with the original text verbatim plus the
   rebuild command. A run must die here, not at cycle 0. The reset comes before anything is read: a
   freshly constructed env reports `decision_ticks = 1` until its first `reset()`, so reading the
   configuration earlier would record a number that is about to change.
2. **Read `ClashParallelEnv.config()`.** One JSON-able dict — `env`, `decision_ms`, `decision_ticks`,
   `reveal`, `engine`, `obs_builder`, `action_parser`, `reward_fn`, `termination_cond`,
   `truncation_cond`, `state_mutator`, `calibration_digest`, `build_digest`, each component as
   `{"class": ..., "params": component.config()}` — and it is the single source for the timing fields
   of `EnvSpec`, the `EngineBuild` digests of section 5.3 and the ladder's `context` of section 11.6.
   All three read that one dict, so a component whose parameters change moves the identity, the
   context and the printed summary together or not at all.
3. **Read the layout off the environment.** Every key of `single_observation_space` with its shape,
   dtype and per-channel bounds; `ObsBuilder.vector_layout()`; `ObsBuilder.spatial_layout()`. Build
   `EnvSpec.obs_space`, `vector_layout` and `spatial_layout`, compute the codec table (section 7.2)
   and `obs_digest`, and **print the table**: one line per spatial plane with its storage and divisor,
   the vector width and its named fields, the mask width and the row total. Nothing in the harness
   holds a width, a plane count or a field offset of its own, which is why the whole test suite runs
   on `MockEngine`, whose widths are not the Rust catalogue's.
4. **Resolve the vector fields the pointer head needs** — `hand_card_onehot`, `hand_cost`,
   `hand_affordable` — through `vector_layout`. A missing name is a `PreflightError` naming it.
5. **`royalegym.action.mask_disagreements(engine, parser, state, team)` for both teams**, exhaustive
   over all 2304 non-no-op actions. Non-empty is a `PreflightError`, not a warning: a policy trained
   against a wrong mask is worthless, the check is already written and nobody runs it.
6. **Assert `mask[NOOP]`** on 1000 sampled states including one with `game_over` set. This is the one
   precondition the whole masking scheme rests on.
7. **Assert the action-layout identity** exhaustively: for all 2304 non-no-op actions, a one-hot
   `(hand_size, tiles_y, tiles_x)` tensor reshaped in C order has its argmax at
   `parser.encode(slot, x, y)`; and, where the observation carries mask planes, that
   `mask_planes == action_mask[1:].reshape(hand_size, tiles_y, tiles_x)` on the sampled states.
8. **Record the `EngineBuild`** from step 2's digests and the card catalogue.
9. **Print the RAM ledger** and refuse if the projection exceeds `doctor.ram_budget_mb`.
10. **Print** `credit_horizon_seconds`, `timesteps_per_iteration`, `T`, `R`, the learner-row count and
   the `run_id`.
11. **Settle the viewer's state stream.** The publisher belongs to the vec env, so exactly one vec env
   in the whole run carries it: worker 0's first shard is built with `viser="env"` and every other
   shard with `viser=None`, and within that shard the index-0 battle is the one published. Where the
   vec env does not take the argument, the workers unset `ROYALEVISER` in their own environment
   instead and preflight prints that the state stream is off, because a per-process publisher binds
   the same UDP port once per env and raises `OSError` at construction. The learning-status stream of
   section 13.1 is a separate socket and is always available.

---

## 8. The networks and the masked distribution

### 8.1 Exact shapes

Every shape is written in the environment's own terms, because that is how the code computes them.
`B` batch; `C = net.channels`; `E = net.vector_embed`; `S`, `V` and `A` the spatial plane count,
vector width and action count from `EnvSpec.obs_space`; `P = spec.hand_size` hand slots;
`K = spec.num_cards + 1` card ids including "empty"; `k = spec.frame_stack`; board
`H x W = spec.tiles`.

```
ObsBatch (device tensors, from ObsCodec.unpack_to_device):
    spatial      (B, k*S, H, W)  float32   (bf16 inside autocast)
    mask_planes  (B, k*P, H, W)  float32   derived from mask, never stored
    vector       (B, V)          float32   the current frame only
    mask         (B, A)          bool

trunk input assembly:
    coords   (2, H, W)        constant buffer, y/(H-1) and x/(W-1) in [0,1]
    vemb     Linear(V, E)(vector) -> (B, E, 1, 1) -> expand -> (B, E, H, W)
    x        cat[spatial, mask_planes, coords, vemb]   -> (B, k*(S+P) + 2 + E, H, W)

stem     Conv2d(k*(S+P) + 2 + E, C, 3, padding=1) -> GroupNorm(norm_groups, C) -> ReLU
body     N x ResBlock: Conv3x3(C,C) GN ReLU Conv3x3(C,C) GN (+skip) ReLU   -> (B, C, H, W)

--- actor head ---------------------------------------------------------------
Fp       Conv3x3(C, C) GN ReLU                                    -> (B, C, H, W)
hand     onehot  = vector[:, f("hand_card_onehot")].view(B, P, K)
         card_id = onehot.argmax(-1)                              (B, P) int64
         cost    = vector[:, f("hand_cost")]                      (B, P)
         afford  = vector[:, f("hand_affordable")]                (B, P)
Ecard    Embedding(K, C)                                          (B, P, C)
q_in     cat([Ecard[card_id], cost[..., None], afford[..., None]]) (B, P, C+2)
q        Linear(C+2, C)(q_in)                                     (B, P, C)
qb       Linear(C+2, 1)(q_in)                                     (B, P, 1)
tiles    einsum('bcyx,bkc->bkyx', Fp, q) * C**-0.5 + qb[..., None] (B, P, H, W)
pooled   cat([Fp.mean((2,3)), Fp.amax((2,3))])                    (B, 2C)
noop     Linear(2C, 1)(pooled) + arch.noop_bias                   (B, 1)
logits   cat([noop, tiles.reshape(B, P*H*W)], dim=-1).float()      (B, A) float32

--- critic (its own trunk, identical shape, separate weights) -----------------
pooled_c cat([body_c.mean((2,3)), body_c.amax((2,3))])            (B, 2C)
vfeat    cat([pooled_c, Linear(V, E)(vector), legal_frac])        (B, 2C + E + P)
value    Linear(2C+E+P, value_hidden) ReLU Linear(value_hidden, 1) -> squeeze   (B,)
```

*Worked example, the shipped defaults on today's 95-card catalogue:* `k = 1`, `S = 20`, `P = 4`, `E = 32`,
`C = 64`, `H x W = 32 x 18`, `V = 1177`, `A = 2305`, so `spatial` is `(B, 20, 32, 18)`, the trunk takes
`1*(20 + 4) + 2 + 32 = 58` input channels and the stem is `Conv2d(58, 64, 3)`. Those numbers are an
illustration of the expressions above and never appear in the code: on `MockEngine`'s 16-card
catalogue the same expressions give a 229-wide vector and the same 58 channels, and a plane added
upstream changes the stem without anything here being edited.

`f(name)` is `obs_layout.py`'s lookup into `EnvSpec.vector_layout`: it returns the `(offset, size)`
slice the environment declared for that field. The head asks for `hand_card_onehot`, `hand_cost` and
`hand_affordable` by those names and computes no offset of its own. A layout that does not declare a
name the head needs is a `PreflightError` naming the missing field, raised before the first engine
step rather than discovered as a wrong slice halfway through a run.

The four mask planes are part of the trunk's input because they are a state feature the policy would
otherwise have to infer: each is the deploy legality of one hand slot over the whole board, which is
exactly "where could I place this card", and it is already computed for the mask. It costs four input
channels and nothing else, because it is a reshape of bytes the row already carries.

The head's shape is not a choice. `royalegym/action.py:272` is
`encode(slot, x_idx, y_idx) = 1 + slot*576 + y_idx*18 + x_idx`, so the non-no-op actions **are** a
`(P, H, W)` C-order tensor aligned pixel for pixel with the spatial planes, and index 0 is the no-op.
A flat `Linear(512, A)` head is 1 180 160 parameters against the pointer head's ~10.5 k, and it is the
structural cause of tile spam, because a shared-weight head cannot memorise one output unit the way a
flat head can. Conditioning on a **card embedding** rather than the slot index is the other half:
slot 0 holds a different card every cycle, so a slot-indexed head has to learn a card-agnostic
detector, and the embedding also buys deck transfer.

`legal_frac` is `P` scalars, the mean of each hand slot's slice of the mask. The mask is a function of
state, so feeding a summary of it to the critic is legitimate and mildly helpful; the critic never
sees the full mask and never receives the entropy gradient, which separate trunks make structural
rather than disciplinary.

Parameters at the worked example's values, convolution and linear weights only:

| component | parameters |
|---|---|
| stem 58 -> 64, 3x3 | 33 408 |
| 4 residual blocks (8 convs 64 -> 64, 3x3) | 294 912 |
| vector embedding `Linear(1177, 32)` | 37 664 |
| policy conv + card table 96x64 + query MLPs | 47 400 |
| value head, with its second `Linear(1177, 32)` | 79 120 |
| **actor** | **~413 k** |
| **critic** | **~445 k** |
| **actor + critic** | **~858 k** |

Weights 3.4 MB fp32, Adam moments 6.9 MB. Activations dominate: about 2.36 MB per sample fp32 for the
eight body convolutions, so minibatch 512 is 1.18 GB fp32 and **0.59 GB bf16** **[A]**. That is why
the minibatch is 512 and the precision is bf16.

`C = 96, N = 8` roughly triples the compute and is the single change to make on a bigger box.
`arch_digest` covers `C`, `N`, the observation space, `frame_stack`, `num_cards`, `n_actions`, the
head name and the trunk class, and a load with a different digest is refused with a message rather
than a shape error.

### 8.2 Precision rules

These are correctness rules, not performance rules.

- Trunk and heads run under `torch.autocast("cuda", dtype=torch.bfloat16)`. bf16 and not fp16: bf16
  has fp32's exponent range, so `finfo.min` masking and `log_softmax` behave and no `GradScaler` is
  needed.
- **Logits are cast to float32 before masking and before the distribution**, at rollout and at update
  alike. That makes the masking and the `log_softmax` exact, which is what keeps `finfo.min` from
  leaking probability onto an illegal action and keeps the entropy sum finite.
- **The two forwards do not agree bit for bit, and the spec does not claim they do.** The rollout
  forward runs one policy group of roughly fifty to a hundred and fifty rows; the update forward runs
  a minibatch of 512. cuDNN selects an algorithm per shape and bf16 carries about three significant
  digits, so the fp32 logits at the end of the two paths differ at the 1e-2 level and the log-probs
  with them. The fp32 cast fixes the masking, not the convolutions underneath it. Section 9.6 sets the
  tolerance accordingly and says what the check still catches.
- **GroupNorm, never BatchNorm.** BatchNorm computes a different function at rollout (a small batch,
  running statistics) than at update (batch 512, `train()` mode). That difference is not a rounding
  difference — it is a different function of the same weights, so the stored log-prob would stop being
  the log-prob of the action that was taken.

### 8.3 `MaskedCategorical`

`learn/distribution.py` is the only place in the package where a mask meets a logit.

```python
class MaskedCategorical(ActionDistribution):
    def __init__(self, logits: Tensor, mask: Tensor) -> None:      # (B, A) f32, (B, A) bool
        assert logits.dtype == torch.float32
        assert mask.dtype == torch.bool and mask.shape == logits.shape
        assert bool(mask[:, 0].all()), "mask[NOOP] must be True (royalegym action.py:284)"
        self._mask = mask
        self._logp = torch.log_softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), -1)

    def sample(self, u: Tensor) -> Tensor:                          # inverse CDF
        cdf = self._logp.exp().cumsum(-1)
        cdf = cdf / cdf[..., -1:].clamp_min(1e-30)
        idx = torch.searchsorted(cdf, u.unsqueeze(-1).clamp(0.0, 1.0 - 1e-7), right=True)
        return idx.squeeze(-1).clamp_(max=cdf.shape[-1] - 1)

    def mode(self) -> Tensor:
        return self._logp.argmax(-1)                                # post-mask, mask-respecting

    def log_prob(self, a: Tensor) -> Tensor:
        return self._logp.gather(-1, a.unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> Tensor:
        p = self._logp.exp()
        return -(p * torch.where(self._mask, self._logp, torch.zeros_like(self._logp))).sum(-1)

    def noop_entropy(self) -> Tensor:
        p = self._logp[:, 0].exp().clamp(1e-7, 1 - 1e-7)
        return -(p * p.log() + (1 - p) * (1 - p).log())

    def n_legal(self) -> Tensor:
        return self._mask.sum(-1)
```

Every detail has a failure it prevents, and every reference implementation gets at least one wrong:

1. **Mask before the softmax, never after.** The two are gradient-identical — a masked logit receives
   exactly zero gradient either way — so the difference is numerical and it is decisive. Post-softmax
   masking combined with the references' `clamp(probs, min=1e-11)` *resurrects every illegal action at
   p = 1e-11* with a finite log-prob, and `multinomial` will eventually draw one.
2. **`torch.finfo(dtype).min`, never `float("-inf")`.** `-inf * 0` is NaN in the entropy sum; one
   reference survives only because `torch.distributions.Categorical.entropy()` clamps internally. A
   library's clamp is not a place to keep a correctness property.
3. **No row can be fully masked**, because `action.py:284` sets `mask[NOOP] = 1` unconditionally,
   including after game over. NaN is structurally impossible and a test says so over 10 000 states.
4. **`mode()` takes the argmax of the normalised masked log-probs.** One reference takes
   `probs.argmax` on an *unmasked* softmax, which here would emit illegal actions in every evaluation
   game.
5. **`sample` takes uniforms.** With `right=True`, an illegal action's zero-width CDF interval can
   never be selected, and the sampled action is independent of batch composition and of torch's global
   RNG. A test compares it against a brute-force inverse CDF.

The mask travels with the transition (289 bit-packed bytes, 2.2% of a row) and the *same* mask is
applied in `backprop`. Two asserts run during `ppo.debug_assert_iterations`:

```python
assert stored_mask.gather(-1, actions.unsqueeze(-1)).all()   # every action taken was legal
assert torch.isfinite(log_probs).all()                       # no -inf reached the loss
```

A rollout/update mask mismatch is silent and slow-acting: unmasked at update pins the clip fraction at
1.0, masked at update gives `log pi = -inf` and NaN. Both are in the alarm table.

### 8.4 Initialisation and the no-op bias

Orthogonal, gain `sqrt(2)` on every hidden convolution and linear, **gain 0.01** on the pointer head's
`q` and `qb` linears and on the no-op linear, gain 1.0 on the value head's last linear, zero bias
everywhere, `Embedding.normal_(0, 0.02)`. Every draw comes from the torch generator seeded from
`torch/init`, so the initial weights are a function of `master_seed` alone.

With a near-zero head the opening policy is near-uniform over the roughly 691 legal actions, so the
agent spends its elixir immediately at random tiles and the mask then forces it to wait. That
oscillation is a good exploratory start and it is self-limiting, so `noop_bias` defaults to 0.0. If it
is ever needed, the bias that produces a target `p` on the no-op with `n` legal actions is

```
b = log( p / (1 - p) * (n - 1) )        # p = 0.9 at n = 691 gives b = 8.7
```

---

## 9. The experience buffer, GAE and the PPO update

### 9.1 Buffer layout

One shared-memory block of `(T + k) x R` rows of `row_bytes`, cycle-major, with `k = obs.frame_stack`:

```
index(t, r) = (t + k - 1) * R + r          t in [-(k-1), T], r in [0, R)
```

Cycle `T` holds observations only — it is the bootstrap row, and its scalars are unused. The `k - 1`
rows below cycle 0 are history: at the end of every iteration the last `k - 1` cycles are copied down
into them, so cycle 0 of the next iteration has a full stack and an iteration boundary is not a
discontinuity in what the policy sees. At `k = 1` there are none and the block is `(T+1) x R`. At the
laptop profile that is `229 x 192 x 14 163 B = 623 MB` **[A]**.

Every row of the rectangle is stored, including the seats a frozen or scripted opponent played. That
costs 25% more memory than storing only learner rows and it buys a buffer index that is the slot
index, with no separate `buffer_row` map to get wrong and no scratch area for discarded rows. What
decides whether a cell reaches the update is `trainable`, not where it was written.

**Frame stacking is a gather, not a second copy.** Because the buffer is a rectangle indexed by cycle
and a slot's rows sit at a fixed stride of `R` in that index, the stack for cell `(t, r)` is rows
`t, t-1, ..., t-k+1` of the same slot, assembled at unpack time in the same kernel that dequantises.
Nothing extra is stored and nothing is written twice. Where a previous row belongs to an earlier
episode — the worker's `episode_end` flag at `t-1` says so — the stack is zero-filled from that point
back, so the first frame of an episode has a zero history and the policy is never shown the tail of
the battle before it. Only `spatial` and `mask_planes` are stacked; `vector` is the current frame's,
because elixir, hand and clock are already the present state and a stale copy of them is noise. A test
at `k = 2` asserts that the ticks of two stacked frames are consecutive, reading `info["tick"]` off
the round, and that an episode's first cell stacks a zero.

The scalar columns live in the parent's own numpy arrays. All of GAE runs on them, on the GPU.

| column | dtype | shape | bytes at the laptop profile |
|---|---|---|---|
| `action` | int16 | (T, R) | 88 K |
| `log_prob`, `reward`, `advantage`, `ret`, `final_value` | float32 | (T, R) | 5 x 175 K |
| `value` | float32 | (T+1, R) | 176 K |
| `terminated`, `truncated`, `valid`, `trainable` | bool | (T, R) | 4 x 44 K |
| `group`, `deploy_status`, `episode_end` | int8 | (T, R) | 3 x 44 K |
| `tick` | int32 | (T, R) | 176 K |
| | | **total** | **under 2 MB** |

The buffer is cleared every iteration: each timestep is trained on exactly `n_epochs` times and then
discarded, which is what makes the KL and clip-fraction diagnostics mean what they say. One reference
learner ships a rolling window that is never cleared, so each timestep is in fact trained on about
three times across three consecutive updates with increasingly stale `old_probs`. That is a
deliberate mildly-off-policy design in its setting and would be an unexamined one here;
`ppo.keep_previous_iterations` exposes the behaviour at a linear memory cost for anyone who wants it.

### 9.2 Minibatching

```python
def batches(self, batch_size, minibatch_size, epochs, rng_for_epoch):
    flat = np.flatnonzero((self.trainable & self.valid).reshape(-1))
    for e in range(epochs):
        perm = flat[rng_for_epoch(e).permutation(flat.size)]     # epochs never straddle batches
        for start in range(0, perm.size, batch_size):
            yield Batch(perm[start:start + batch_size], minibatch_size, self._gather)

def _gather(self, idx):                                          # one minibatch
    idx = np.sort(idx)                                           # sorted: locality
    stage = self._pinned.next()                                  # a ring of 4 pinned slabs
    np.take(self._obs_view, idx, axis=0, out=stage.obs)          # one gather, shared -> pinned
    stage.to_device(non_blocking=True)
    return self._codec.unpack_to_device(stage, self._statics)
```

Three rules, each a fix to something the references do:

1. **Gather per minibatch, never per batch.** Fancy-indexing a whole batch materialises a 2.6 GB
   temporary at this row size.
2. **Pinned staging with non-blocking copies**, so the transfer of minibatch `i+1` overlaps the
   compute of minibatch `i`. The ring is four slabs of `minibatch_size x row_bytes`, 27 MB at the
   laptop profile, allocated once.
3. **Nothing is dropped.** Both references do `while start + batch_size <= total` and silently
   discard the remainder. Here the remainder is a smaller final batch, weighted by its true sample
   count; `health/samples_unused_frac` is in the metric stream and reads 0.

At the laptop profile: 32 832 trainable samples, `batch_size = 4096`, `n_epochs = 3`, so 8 batches
per epoch and **24 optimizer steps per iteration**, each from 8 accumulated minibatches of 512.

### 9.3 GAE

`learn/gae.py`, vectorised over slots, one backward loop of `T` steps over `(R,)` vectors, on the GPU:

```python
ended = terminated | truncated                            # the episode ended AT t
boot  = torch.where(terminated, zeros,
        torch.where(truncated,  final_values, values[1:]))    # (T, R)
adv   = torch.zeros(R); out = torch.empty(T, R)
for t in reversed(range(T)):
    delta  = rew_scaled[t] + gamma * boot[t] - values[t]
    adv    = delta + gamma * lam * (~ended[t]) * adv
    out[t] = adv
returns = out + values[:T]
```

- **A terminated cell bootstraps from 0. A truncated cell bootstraps from `final_value`.** A cell that
  is neither — including the last cycle of the iteration for a row still mid-episode — bootstraps from
  `values[t+1]`, which is the value of the **true** next observation, because SAME_STEP autoreset only
  replaces a row when its episode ended and an episode that ended is flagged. This is the correct
  treatment and it is where both older references go wrong: one bootstraps a truncation off an
  unrelated episode's first state, and one self-bootstraps from `V(s_T)`.
- **`~ended[t]` breaks the recursion at every episode end**, terminated or truncated alike. Using the
  terminated flag alone would leak the next episode's advantage backwards across a truncation.
- **Truncated cells need `final_obs`.** The worker packs `infos["final_obs"]` for those rows into the
  `finals` area of its control segment and the parent runs the critic on them into `final_value`. With
  the shipped config there is no `TruncationCondition`, so the path fires only for dead-worker rows;
  it is exercised and tested regardless.
- **A dead worker's rows fall out without a special case.** For a worker that died at cycle `t*`:
  `valid[t, r] = (t <= t* - 2)`, `truncated[t* - 2, r] = True`,
  `final_value[t* - 2, r] = values[t* - 1, r]`, and for `t > t* - 2` the rewards and values are zeroed
  and `ended` is set, so the loop runs over them inertly.
- `learn/gae.reference_gae` is the per-slot python implementation. `tests/test_gae.py` asserts the two
  agree to 1e-6 over random inputs including every boundary case. The reference learner ships the same
  pair and no test; here the test is the point of the pair.

`credit_horizon_seconds = decision_ms / 1000 / (1 - gamma * gae_lambda)` is computed, printed at
start-up and logged every iteration. At the defaults it is 45.5 s. It is the single highest-leverage
number in the configuration and it must never be set to ten seconds by accident.

### 9.4 Reward scaling

`learn/returns.py`, `WelfordReturnScaler`. Divide rewards by the running standard deviation of
**unstandardised, unclipped** discounted returns, then clip:

```python
raw   = discounted_cumsum_per_episode(rewards, gamma)      # before any scaling
self.welford.update(raw[trainable & valid])                # {mean, count, m2}, n-1 variance
scale = self.welford.std if (standardize and self.welford.count > 1) else 1.0
rew_scaled = (rewards / scale).clamp(-reward_clip, reward_clip)
```

The mean is never subtracted: subtracting it changes the sign structure of a zero-sum reward. The
Welford triple is checkpointed. This matters more here than in the references because gamma
approaching 0.999 inflates return magnitudes roughly tenfold over gamma = 0.99.

Observations are **not** normalised. They are bounded by construction, Welford would amplify one-hot
noise, and it would add cross-worker mutable state that a resume has to restore exactly. Fixed
per-channel divisors plus the stem's GroupNorm do the same job with no state.

### 9.5 The update

`learn/ppo.py`, `PPOUpdate.step`:

```python
def step(self, buffer, sched) -> UpdateResult:
    values = chunked_critic_pass(buffer, chunk=cfg.critic_chunk)       # (T+1, R), bf16, no_grad
    buffer.set_values(values)
    buffer.set_final_values(*critic_on_final_obs(buffer))              # only where truncated
    adv, ret, stats = self.gae.compute(..., gamma=sched.gamma, lam=cfg.gae_lambda)
    m = buffer.trainable_mask() & buffer.valid_mask()
    if cfg.advantage_standardization:                                  # ONCE per iteration
        adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-8)
    buffer.set_advantages(adv, ret)
    before_a, before_c = params_to_vector(actor), params_to_vector(critic)

    for batch in buffer.batches(cfg.batch_size, cfg.minibatch_size, cfg.n_epochs, rng_for_epoch):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        for mb in batch:
            w = mb.size / batch.size                                   # accumulation weight
            bp = actor_critic.backprop(mb.obs, mb.action)
            logp  = bp.log_probs                                       # fp32
            ratio = torch.exp(logp - mb.log_prob)
            surr  = torch.min(ratio * mb.adv,
                              ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * mb.adv)
            dual  = torch.where(mb.adv < 0,
                                torch.max(surr, cfg.dual_clip_c * mb.adv), surr)
            policy_loss  = -dual.mean() * w
            value_loss   = cfg.vf_coef * F.mse_loss(bp.values, mb.ret) * w
            entropy_loss = -(sched.ent_coef * bp.entropy.mean()
                             + sched.ent_coef_noop * bp.noop_entropy.mean()) * w
            (policy_loss + value_loss + entropy_loss).backward()
            self._accumulate_diagnostics(ratio, logp, bp, mb, w)       # under no_grad
        self._record_grad_norms()                                      # BEFORE clipping
        clip_grad_norm_(actor.parameters(),  cfg.max_grad_norm)        # separately
        clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)        # separately
        for opt in self.optimizers:
            opt.step()
    torch.cuda.synchronize(device)                                     # ONCE, then every .item()
    return UpdateResult(...)
```

The loss, written out, for one minibatch of `n` samples in a batch of `N`:

```
r_i            = exp( log pi_new(a_i | s_i) - log pi_old(a_i | s_i) )
surr_i         = min( r_i * A_i , clip(r_i, 1 - eps, 1 + eps) * A_i )
L_i            = surr_i                       if A_i >= 0
                 max( surr_i , c * A_i )      if A_i <  0          # dual clip, c = 3.0
L_policy       = - (1/n) sum_i L_i                        * (n/N)
L_value        = vf_coef * (1/n) sum_i (V(s_i) - G_i)^2   * (n/N)
L_entropy      = - ( ent_coef * (1/n) sum_i H_i
                   + ent_coef_noop * (1/n) sum_i H2_i )   * (n/N)
L              = L_policy + L_value + L_entropy
```

with `H_i` the entropy of the masked categorical and `H2_i` the binary entropy of `p(no-op)` against
`p(play)`, read off the same normalised log-probs at no cost. The actor and the critic have disjoint
parameters, so summing one loss and calling `backward()` once is safe and the entropy term's gradient
into critic parameters is exactly zero — which has its own test.

`ent_coef_noop` is a **warm-start guard and not a permanent term of the objective**. `H2(p_noop)` is
maximised at `p_noop = 0.5` while a healthy Clash policy plays about 22 cards in a match and sits near
`p_noop = 0.94`, so a coefficient that never decays pulls every converged policy toward overplaying —
it is a term that pays for a behaviour the objective does not want. It therefore anneals to zero over
its schedule; the schedule is in the run identity and `run/ent_coef_noop` is logged every iteration so
the two regimes are never confused in a plot. The `noop_collapse*` and `noop_entropy_floor` alarms are
untouched by this and keep watching after the coefficient reaches zero, and whether the guard was
needed at all is one of the things the first run measures (section 18).

The `n/N` weighting with one `optimizer.step()` per **batch** is what makes `minibatch_size` a pure
memory knob: the optimisation is identical whatever it is set to, which `tests/test_ppo.py` proves by
comparing the accumulated gradient against a full-batch one. On a 4 GB GPU that is not a nicety; it
is the mechanism that makes the run possible.

Diagnostics, all under `no_grad`:

```
kl         = mean( (exp(log r) - 1) - log r )          # Schulman k3: non-negative, low variance
clip_frac  = mean( |r - 1| > eps )
dual_frac  = mean( (A < 0) & (c*A > surr) )
ev         = 1 - Var(G - V) / max(Var(G), 1e-8)
ent_norm   = mean( H / log(max(n_legal, 2)) )
grad_norm_actor, grad_norm_critic                       # pre-clip
update_magnitude_actor  = || theta_a_before - theta_a_after ||
update_magnitude_critic = || theta_c_before - theta_c_after ||
```

`kl` and `clip_frac` are recorded **per epoch** as well as per iteration, because "epoch three's clip
fraction is more than twice epoch one's" is the concrete signal to lower `n_epochs`.

**The learning-rate backoff.** If `kl > lr_backoff.kl_threshold` for `patience` consecutive
iterations, multiply both learning rates by `factor`, floor at `lr_min`, reset the counter and log an
`lr_backoff` event. The counter and the current rates are checkpointed. One reference computes the KL
and acts on nothing; four lines prevent the class of blow-up that costs a whole run.

**Explicitly not included:** value-function clipping (the published studies find it
neutral-to-harmful, and one reference omits it with a comment saying so), observation normalisation
(section 9.4), any learning-rate schedule beyond the backoff, and `torch.compile` (measure it first,
on a machine where it is not a sixty-second warm-up on every run).

Checkpoint loading uses `torch.load(..., map_location=device, weights_only=True)`. Without
`map_location` a CPU-only resume fails; without `weights_only` every checkpoint is an execution
surface. The optimizer factory's keyword arguments are restored **over** a loaded state dict, so
changing the learning rate on a resume actually takes effect.

### 9.6 The ratio invariant

At epoch 0, minibatch 0, before any optimizer step, `ratio` should be 1.0 for every sample: the
parameters are unchanged since the rollout and the input bytes are identical, because the policy acted
on `decode(encode(obs))` (D5).

```python
if self._check_ratio_now(iteration):
    dev = (ratio - 1).abs().max().item()
    assert dev <= cfg.ratio_atol[precision], _ratio_message(dev, ratio, mb)
```

**The tolerance is a property of the precision, not a fudge factor.** In fp32 the two forwards agree
to `1e-4`. Under bf16 autocast they do not: the rollout forward is one policy group of roughly fifty
to a hundred and fifty rows and the update forward is a minibatch of 512, cuDNN picks an algorithm per
shape, and bf16's three significant digits put the resulting fp32 logits about 1e-2 apart (section
8.2). `ratio_atol` is therefore `1e-4` with autocast off and `2e-2` under bf16, and
`royalelearn bench` measures the actual `ratio_max_abs_dev` over ten rounds on the machine it is run
on and prints it beside the tolerance, so a user can see the margin rather than trust it.

What `2e-2` still detects is the whole reason the check exists, because every failure it is aimed at
produces a deviation of order **one**, not of order 1e-2:

- a rollout/update **mask** mismatch makes a masked-out action's log-prob `finfo.min`, so the ratio is
  either zero or astronomically large;
- a **codec** or codec-table mismatch feeds the update a different observation, and the log-prob of a
  2305-way categorical moves by far more than a percent;
- a **weight-version** mismatch — the policy that acted is not the policy being updated — moves
  log-probs by the size of a PPO update, which is what `ppo/update_magnitude_actor` measures and is
  orders above the tolerance.

Kernel-level nondeterminism is not what this check is for, and at `2e-2` it will not see it. That is
the correct division of labour: reproducibility is `determinism.tier`'s job, and tier T2 keeps bf16
precisely because deterministic kernels are bit-reproducible run to run at any precision — T2 promises
that two runs agree, never that two differently shaped forwards within one run do.

The check runs for the first `ppo.debug_assert_iterations` iterations of a run and every
`check_ratio_invariant_every` thereafter, and `ppo/ratio_max_abs_dev` is logged always. A violation's
message names the worst ten samples with their slot, cycle and episode ordinal, and lists the three
causes above in the order they are worth checking. No reference implementation has this check, and it
is the cheapest possible detector for the failure mode that is otherwise silent and slow-acting.

---

## 10. Rewards

`royalelearn/rewards.py` ships four `RewardFunction` implementations built on RoyaleGym's ABC and
composed with RoyaleGym's `CombinedReward`. Nothing in RoyaleGym is modified; these are additions
that live here until they are absorbed upstream (section 16, ask 4).

```python
def default_potential_reward() -> CombinedReward:
    return CombinedReward([
        (WinLossReward(draw=0.0),               1.0),    # royalegym's; the objective
        (PotentialCrownReward(),                0.2),
        (PotentialTowerHPReward(),              0.1),
        (CommittedElixirPotential(scale=10.0),  0.05),
    ])
```

Each potential term returns `gamma * Phi(s') - Phi(s)`, with `Phi` computed from the state:

| term | potential |
|---|---|
| `PotentialCrownReward` | `(own_crowns - foe_crowns) / 3` |
| `PotentialTowerHPReward` | `(sum_s own_hp[s]/own_max[s] - sum_s foe_hp[s]/foe_max[s]) / 3` |
| `CommittedElixirPotential` | `((own_bar + own_board) - (foe_bar + foe_board)) / scale`, where `bar` is `elixir_milli / 1000` and `board` is the sum of `Fraction(card.elixir, card.count)` over that player's live non-tower entities |

`gamma` comes from the schedule: `CombinedReward.set_gamma(g)` is called once per iteration by the
coordinator and the value rides the `Step` command to every worker, so the reward the worker applies
and the discount the learner uses are the same number. A test asserts they agree.

The reason this composition and not RoyaleGym's shipped `default_reward()`:

- The terminal win/loss term is the only one that is the objective. Everything else is shaping, and
  shaping that is not a difference of a potential can change which policy is optimal (Ng, Harada and
  Russell, 1999). RoyaleGym's `reward.py` docstring already states this principle.
- `TowerHPReward` computes `Phi(s') - Phi(s)` rather than `gamma*Phi(s') - Phi(s)`. It is one `gamma`
  away from being exactly policy-invariant, and fixing it means the term never needs annealing.
- `ElixirTradeReward` is **not** a potential and it rewards turtling: an agent that never plays a card
  never incurs the negative term while enemy units still die to its towers and earn the positive one.
  At weight 0.02 over about forty trades a match that is 0.8, comparable to the plus-or-minus-one
  terminal reward — the shaping can outweigh the objective.
- `ElixirLeakPenalty` is not zero-sum: both players can leak at once. That is the signature of a term
  standing in for a missing potential.
- The committed-elixir potential replaces both. Playing a card moves elixir from the bar to the board
  and is net zero; losing a unit costs; killing gains; and sitting at ten elixir is penalised
  automatically, because the opponent's potential keeps rising while yours does not. There is **no
  coefficient to re-tune and no annealing schedule**, which is what
  `royalegym/reward.py`'s own house rule — weights should settle, not drift — asks for.

Unit values use exact `Fraction(card.elixir, card.count)` arithmetic, as RoyaleGym's own elixir term
does, because the seat-mirror antisymmetry test depends on it: a running float sum of the same values
returns plus and minus 1.1e-19 on a perfect mirror instead of zero, which is harmless to a gradient
and makes the property uncheckable.

Every weighted term's per-episode sum, per seat, is on the `EpisodeRecord` and in the metric stream as
`env/reward_terms/<name>`. The acceptance criterion, checked by the `shaping_dominates` alarm, is that
the sum of the absolute shaping terms stays below the terminal term's magnitude.

---

## 11. The ladder

### 11.1 What lives where

`royalegym.selfplay` keeps what is bookkeeping over ids and results — `Opponent`, `PolicySnapshot`,
`OpponentPool`, the sampling strategies, `pfsp_weights`, `save`/`load`, and the streaming Elo — because
it needs neither torch nor an env. RoyaleLearn owns the weights, the routing, the result log, the fit,
the evaluation runner and the gate. `ladder/pool.py`'s `LadderPool` wraps an `OpponentPool`, a
`ResultLog` and a `SnapshotStore`.

`OpponentPool.record_result` and `_evict` are not called from this repo. The pool's aggregate record
cannot represent draws separately — `wins: float` with a draw adding 0.5 makes five wins and five
losses indistinguishable from ten draws, and those carry completely different variance — and `_evict`
drops the *oldest* snapshot, which is exactly the diversity the anti-forgetting floor exists to
protect. `ladder/results.py` owns the result log and `ladder/eviction.py` owns eviction until those
are fixed upstream (section 16, ask 4).

### 11.2 Rating

**Layer 1, the readout.** `EloReadout`, RoyaleGym's zero-sum logistic Elo at `k_factor = 32`. It goes
on the dashboard between refits and nowhere else. It is order-dependent, which under parallel workers
is nondeterministic, and a fixed `k` keeps moving a *frozen* player whose strength is constant.

**Layer 2, authoritative.** `BradleyTerryDavidsonRater`, a MAP fit over the whole stored evaluation
result matrix:

```
q_i        = 10 ** (r_i / 400)
P(i wins)  = q_i / (q_i + q_j + nu * sqrt(q_i * q_j))
P(draw)    = nu * sqrt(q_i * q_j) / (q_i + q_j + nu * sqrt(q_i * q_j))
P(j wins)  = q_j / (q_i + q_j + nu * sqrt(q_i * q_j))
prior      = r_i ~ Normal(0, prior_sd^2),  prior_sd = 400 Elo
gauge      = r["scripted:noop"] pinned at 0
```

The log-posterior is concave, so Newton/IRLS converges in fewer than ten steps on a pool of a few
hundred, in milliseconds. Standard errors come from the diagonal of the inverse observed Fisher
information — the Hessian of the negative log-posterior. A **difference** between two players uses the
corresponding 2x2 block, never the sum of the two marginals. `nu` is fitted jointly; if the measured
draw rate is under 2% the config falls back to `draws = "half_win"` and the run's `ladder.json`
records that it did.

The property that makes this the authoritative layer: **the rating is a pure function of the stored
result matrix.** Same games in, same numbers out, in any order, on any machine. That is the same
property RoyaleSim gives for battles and it is what `docs/design.md`'s determinism commitment requires
of this layer.

TrueSkill is rejected explicitly. It is online only, so it cannot refit history; its sigma shrinks
monotonically and needs an artificial floor to stop the rating freezing; and at the floor the
converged interval is about **plus or minus 92 Elo**, worse than a hundred games of direct
head-to-head. Every one of those compromises exists because a Rocket League evaluation game is a real
match in a real client. Here an evaluation battle costs a fraction of a second and is seed
reproducible, so the compromises should not be inherited along with the mechanism.

`rating_above_v0` is reported alongside the anchored rating, so the headline curve reads as "Elo above
the initial random policy" while the gauge itself is a permanent scripted anchor that exists before
any snapshot does.

### 11.3 Matchmaking

`MixMatchmaker.assign(battle, ordinal, pool, ratings)` is the unit of matchmaking: it answers what one
battle's next episode is, and it is called by the parent at that battle's own episode boundary, from
`match/battle/{b}/ordinal/{k}`. `plan()` is the same function applied across the geometry to produce
the iteration's opening table. A policy is therefore bound to a battle for exactly one episode, and a
partially controlled trajectory cannot occur.

Drawing per episode rather than once per iteration matters for two reasons. An iteration spans several
episodes in every slot, so one draw per slot per iteration would give every episode that starts inside
that iteration the same opponent — a correlation across the mixture that nothing in the statistics
accounts for. And because the draw is addressed by the battle and its ordinal rather than by the
iteration, it does not move when the iteration length or the worker count changes: the same episode of
the same battle meets the same opponent on any geometry, which is what makes a rerun a rerun.

| bucket | share | opponent | trainable seats |
|---|---|---|---|
| mirror | 0.50 | the live policy on both seats | 2 |
| pool | 0.35 | one of at most `max_resident_opponents` frozen snapshots, PFSP-weighted | 1 |
| scripted | 0.15 | uniform over `NoopOpponent` and `RandomLegalOpponent(0.9)`, run in the worker | 1 |

Expected trainable rows per battle is `0.5*2 + 0.5*1 = 1.5`, so **75% of collected rows are kept** and
`ppo.timesteps_per_iteration` counts kept rows. `throughput/discarded_rows_frac` is logged and its
expected value is asserted within a tolerance; a drift there means the matchmaker is not doing what
the config says.

- **PFSP weighting is `hard`, power 2**: `w_k` proportional to `(1 - p_k)^2`, with `p_k` the
  **fitted model's** predicted score probability rather than the direct head-to-head record. On a pool
  of 48 the learner has played few pairs directly and `win_rate`'s Beta(1,1) prior reads exactly 0.5
  for the rest, which collapses `hard` to uniform and discards the information that the learner
  crushes v40 and v40 crushes v55.
- **Mixed with a uniform floor**: `0.8 * hard + 0.2 * uniform`, and every weight floored at
  `weight_floor_scale / M` so no snapshot becomes unreachable. Without a floor the population's tail
  is forgotten, which is the job AlphaStar's forgotten-players slice does.
- **`hard` for training, `variance` for evaluation.** They serve different objectives: training wants
  opponents that still beat the learner, measurement wants the most even matchup because that is where a game
  carries the most information. Both weightings are already implemented in
  `royalegym.selfplay.pfsp_weights`.
- **The learner's seat is drawn uniformly** between blue and red for every pool and scripted battle,
  redrawn per episode like the rest of the assignment, so `env/win_rate_by_seat` measures the seat
  advantage the learner actually experiences. The shipped engine is not a 180-degree rotation between
  seats: multi-unit ground deploys land differently on the two sides, measured on the real game, so
  a seat advantage of a few points can be the game's own and is a warning, not a leak.
- **Evaluation is not in this mixture at all** (D10). The studied alternative folds evaluation into the
  rollout worker at a small probability and swaps the live match object in place; that makes "win rate"
  depend on the training curriculum and, in the reference, leaves a worker permanently misconfigured if
  anything raises between the swap and the restore.

**Deck protocol.** Strength in Clash is a function of the deck pair, and deck matchups are the main
source of non-transitivity. The v1 ladder is defined on one declared deck protocol, recorded in the
`context` string on every result, and results from different contexts are **never pooled**. Deck
generalisation is a later axis, not a v1 confound.

### 11.4 Snapshot cadence and the gate

Cadence is measured in **environment steps, not learner iterations**: iteration size changes when the
batch size or the worker count changes, and a cadence in iterations silently changes meaning. A
candidate every `candidate_every_env_steps` (4 000 000 game-steps at the laptop profile, about two
hours).

Every candidate is evaluated; only those that pass enter the pool. A failed candidate is discarded,
not retried, and its results stay in the log — a run of consecutive failures is exactly the plateau
signal worth having, and `gate_starved` alarms on five in a row. Independently of the gate a
**checkpoint** is written at every candidate: checkpoints are for resuming, pool membership is for
rating, and conflating the two is what makes a pool unbounded.

`WilsonGate` admits candidate `C` if and only if all three conditions hold:

1. **It beats the champion.** Over `n = 1000` paired battles (500 seeds x 2 side assignments), the
   Wilson 95% lower bound on `C`'s score rate is at least **0.52**. In practice that requires an
   observed rate of at least **55.2%**, about **35 Elo**. A bound at 0.50 would admit anything merely
   *not worse* and the pool would fill with lateral moves.
2. **No regression against the anchors.** Score rate against `NoopOpponent` and
   `RandomLegalOpponent(0.9)`, 200 battles each, is not more than
   `gate.anchor_tolerance_pp = 2` percentage points below the champion's. This catches the specific
   failure where a policy beats its recent ancestors by exploiting a shared blind spot and loses basic
   competence.
3. **No pool-wide collapse.** The candidate's **observed** mean score against a `variance`-weighted
   stratified sample of 8 pool snapshots, 100 battles each, is at least the champion's **fitted**
   mean predicted score against the same eight, less one standard error of the candidate's observed
   mean. The champion's side is the fit and not a record because the champion has not necessarily
   played those eight on those seeds, and a condition that silently skipped the snapshots it had no
   games against would be weakest exactly where the pool is most diverse. The rater's prediction is
   defined for every pair from the whole result matrix, and its use here is the same one the PFSP
   weights make of it (section 11.3). The cost stays at 800 battles.

Outcomes: all three pass, admit and promote to champion. (1) and (2) pass and (3) fails, **admit to the
pool but leave the champion unchanged**, with `meta["cycle"] = true` — it is a useful diverse opponent
and a detected cycle, not progress. (1) fails, discard the candidate and keep the results.

A floor admits unconditionally every `floor_admit_every_env_steps`, so a plateau cannot starve the
pool.

Total 2200 battles per gate, all of which are rating evidence. Cost on the laptop profile: 2200
battles of about 420 decisions is 924 000 game-steps, which at 2 eval workers is about **6 to 7
minutes** **[A]**, against a candidate cadence of about two hours — **roughly 5% of wall clock, which
is exactly why the gate can afford to be statistically honest.** `ladder/gate_seconds_frac` is logged.
If it exceeds 8%, raise `candidate_every_env_steps` first: a pool does not need more than twenty
members a day. Only if that is not acceptable, lower `n` to 600 and raise the required observed rate.
Never drop the interval.

A `GateDecision` record — the candidate, each condition's `n`, observed value, bound and verdict, the
eval seed set hash, the champion id and the wall time — is written to
`ladder/gates/<candidate>.json` and pushed to the metrics sink as an artifact.

### 11.5 Evaluation

`ladder/evaluate.py`, `EvalRunner`. Four separations, each easy to get wrong:

1. **No experience is recorded.** The runner writes results and nothing else. It drives a
   `RolloutSource` whose plan marks every row non-trainable.
2. **A different episode definition.** A full match under real rules: `GameOverCondition`, no
   truncation, the real match-start mutator. Training may later use randomised mid-game start states;
   evaluation never does. It is built from a **separate env spec** (`config.eval_env`), never by
   mutating the live env. A step cap does not make evaluation cheaper, it makes it uninformative: a
   cut match is scored on crowns that have mostly not been taken yet, so nearly every game is a draw
   and the rating barely moves (measured on `MockEngine`: eight of twelve games drawn at a 120-step
   cap **[M]**). That holds for the smoke configuration too, where the gate's `n` is reduced but its
   episode definition is not.
3. **Separate RNG streams and a separate result table.** The evaluation seeds come from a fixed set of
   `eval_seed_count` seeds drawn once at run start from `eval/seed_set`, written into the run
   directory, hashed into `ladder_digest`, and reused forever. Training results are recorded but tagged
   `kind="train"` and excluded from the fit by default: they are PFSP-selected and therefore biased
   toward hard matchups.
4. **Paired, with common random numbers.** Each pairing plays each seed **twice with the sides
   swapped**. The unit of analysis is the *seed*, scored
   `x_s = (score as blue + score as red) / 2`, taking values in `{0, 0.25, 0.5, 0.75, 1}`. This removes
   side bias exactly rather than averaging it away, and removes the start-state variance the two
   policies share. The interval is a **bootstrap over seeds** (`gate.bootstrap_resamples = 10 000`,
   seeded from `eval/bootstrap/{comparison_id}`), not a binomial over games: the two games of one seed
   are correlated and treating them as independent overstates `n` by up to a factor of two.

The empirical correlation between the two side assignments on a seed, `rho`, is measured and logged as
`ladder/paired_rho`. The effective sample size is about `n / (1 + rho)`; at `rho = 0` pairing costs
nothing and above it pairing wins. It is also worth knowing on its own: it says how much of a battle's
outcome the start state decides rather than the policies.

Evaluation uses the **release sampling mode**, one config field (`ladder.release_mode`, default
`"stochastic"`). Rating both the sampled and the argmax variant doubles the pool and the cost for no
decision value.

### 11.6 The result log, the snapshot archive and eviction

`ladder/results.py`. An append-only `ladder/games.jsonl`, one line per battle:

```json
{"a": "learner@12500000", "b": "snap:v17", "score_a": 1.0, "seed_index": 311, "side_a": "blue",
 "context": "9f3c1a2b4d5e6f70", "kind": "eval", "run_id": "3f9a1c...", "iteration": 381,
 "wall": "2026-09-22T04:11:07Z"}
```

`context` is
`sha256(env_factory.digest() + obs_builder_spec.digest() + deck_protocol + engine_build_digest)[:16]`.
The obs builder's own `ComponentSpec` digest is in there because it carries `Reveal`: a policy that
was shown the opponent's hand and a policy that was not are not the same kind of player, and pooling
their results would make the ladder measure the information rather than the policy. For the same
reason `EvalRunner` refuses a pairing whose two `spec.json` `obs_digest`s differ, naming both ids and
both digests rather than producing a number nobody can interpret.

The aggregate table is a derived cache and a test asserts `rebuild(log) == aggregate`. Writes are
append, flush and `os.fsync` per batch, so a crash truncates at most the last line and the reader
tolerates it.

`ladder/snapshots.py`, `DiskSnapshotStore`: content-addressed under `snapshots/<sha256[:16]>/` holding
`actor.safetensors` (fp16, about 0.8 MB), and `spec.json` carrying `arch_digest`, `obs_digest`,
`action_digest`, `codec_version`, `codec_table_digest`, `frame_stack`, `num_cards`, `vector_size`, the
training step, the `context` and the run id. **No pickled `nn.Module`**: unpickling a module is
arbitrary code execution on load and one class rename invalidates the whole pool. A load whose
`arch_digest`, `obs_digest` or `codec_table_digest` disagrees with the current run is refused with a
message naming the field. A `LoadedPolicyCache` keeps at most
`max_resident_opponents + 2` modules resident in VRAM.

Archive everything and never delete: at a megabyte each, five hundred snapshots is under a gigabyte.
`HallOfFameEviction` removes snapshots from the **sampler** only — never a scripted anchor, never v0,
never a member of the champion chain; among the rest it keeps a stratified sample across the fitted
rating range and prefers keeping snapshots flagged `meta["cycle"]`, because those are the diverse ones.
Evicted snapshots stay in the archive and in the result log and their ratings stay in the fit:
eviction is about sampling cost, not about forgetting evidence.

### 11.7 The statistics

Unpaired binomial half-widths at the worst case `p = 0.5`, with the Elo equivalent at
`dp/dElo = ln(10)/1600 = 0.0014386`:

| n battles | 95% half-width on score rate | equivalent Elo half-width |
|---|---|---|
| 100 | 9.8 pp | 68 |
| 200 | 6.9 pp | 48 |
| 400 | 4.9 pp | 34 |
| **1000** | **3.1 pp** | **22** |
| 2000 | 2.2 pp | 15 |
| 10000 | 1.0 pp | 6.8 |

Score rate to Elo, `d = 400 * log10(p / (1 - p))`:

| score rate | 0.525 | 0.55 | 0.575 | 0.60 | 0.65 | 0.75 |
|---|---|---|---|---|---|---|
| Elo gap | 17 | 35 | 52 | 70 | 108 | 191 |

The Wilson score interval, `z = 1.96`:

```
centre = (p_hat + z^2/(2n)) / (1 + z^2/n)
half   = z / (1 + z^2/n) * sqrt( p_hat*(1 - p_hat)/n + z^2/(4 n^2) )
lower  = centre - half
```

| observed | n | Wilson lower bound | clears 0.52? |
|---|---|---|---|
| 55.0% | 1000 | 0.519 | just short |
| **55.2%** | **1000** | **0.521** | **yes** |
| 57.5% | 1000 | 0.544 | yes |
| 55.0% | 400 | 0.501 | no |

A candidate must be about 35 Elo stronger to pass reliably: a true 35-Elo improvement passes with
roughly even odds per attempt and a true 70-Elo improvement passes essentially always, which is the
right shape for a gate applied every few million steps. With a non-trivial draw rate `d` the variance
of the score rate is `(p(1-p) - d/4)/n`, strictly less than the binomial, so the table is conservative
once draws are counted separately — one more reason to store them separately.

### 11.8 What the ladder logs

The fitted rating and its 95% interval for the learner and every pool member, with the anchor named;
`rating_above_v0`; the score rate against each scripted anchor, which is the one scale that never
drifts; every gate outcome with the condition that failed; the draw rate; `paired_rho`; the pool and
sampler sizes; eviction events; per-snapshot evaluation game counts; and the **transitivity
residual** — the fraction of head-to-head pairs with at least 30 games whose observed score
contradicts the fit by more than two standard errors.

The residual is the number that says whether a scalar rating is meaningful for this population. Above
about 10% the scalar release metric is lying and the `Rater` ABC is exactly the seam a Nash-averaging
or alpha-rank implementation drops into — the result log stores every game rather than an aggregate,
so it already holds the dense matrix such a method needs. Publishing the number is how
`royalegym/selfplay.py`'s honest docstring caveat about non-transitivity becomes a measurement instead
of a warning, and it is far better discovered by a logged diagnostic in month one than by a confusing
ladder in month six.

---

## 12. Checkpoints

### 12.1 Layout

```
<runs_dir>/<run_name>-<run_id>/
  identity.json                 RunIdentity, written once at run start
  config.json                   the full resolved config, canonical JSON, written once
  drift.json                    present only if --allow-identity-drift was used
  metrics.jsonl                 one row per iteration
  episodes.jsonl[.gz]           one row per episode
  alarms.jsonl                  one row per alarm firing
  bundles/<iteration>/          the diagnostic bundle written on any halt
  ladder/
    games.jsonl                 the append-only result log; the source of truth for ratings
    aggregate.json              derived cache
    eval_seeds.json             the frozen seed set
    ratings/<iteration>.json    each refit's RatingTable
    gates/<candidate>.json      each GateDecision
  snapshots/<sha>/              actor.safetensors + spec.json; outside checkpoints, they outlive them
  checkpoints/
    <cumulative_env_steps:012d>/
      manifest.json
      actor_critic/   actor.safetensors, critic.safetensors, arch.json
      optimizers/     actor_adam.pt, critic_adam.pt, misc.json (cumulative_model_updates)
      schedules/      state.json   gamma, ent_coef, ent_coef_noop, lr_actor, lr_critic,
                                   backoff counter, positions
      advantage/      welford.json {"mean": f64, "count": int, "m2": f64}
      rollout/        state.json   per (worker, shard): seed path, reset ordinal, respawn generation
      ladder/         pool.json, champion.json, ratings.json
      metrics/        wandb.json {"run_id": ...}, nested per decorated sink
      rng/            rng.json    section 12.3
      buffer/         buffer.npz  optional (checkpoint.include_buffer, default false)
```

Every component owns its folder and its own `save_checkpoint` / `load_checkpoint`. Adding a component
to a checkpoint is adding a folder constant and a dict entry; there is no central serialiser to edit.

**Atomicity.** Write into `<name>.partial/`, `fsync` **each file**, then
`os.replace(partial, final)`. The directory itself is not fsynced: `os.fsync` on a directory handle is
a POSIX guarantee and raises on Windows, where this harness's default profile runs, so the durability
step is per file and the atomic step is the rename. `os.replace` is atomic on both platforms, which is
the property the recovery actually needs — a crash mid-write leaves the previous checkpoint intact and
the partial directory obviously named. `latest()` reads the run index, never
`int(x) for x in os.listdir(...)` — that form crashes the save on any stray file, and the save is
called from the crash handler, which is exactly where the user most needs it not to.

**Pruning** keeps `checkpoint.keep` checkpoints by the manifest index and survives a stray file in the
directory.

**No pickle anywhere on a path this repo writes.** safetensors for weights, `torch.save`/`torch.load`
with `weights_only=True` for optimizer state, msgspec JSON for everything else, `.npz` with
`allow_pickle=False` for the optional buffer. `tests/test_snapshots.py` scans written files for the
pickle protocol marker.

### 12.2 `manifest.json`

```python
class Manifest(msgspec.Struct):
    format_version: int                  # refuse a higher one by name; migrate a lower one
    run_id: str; run_name: str
    identity: RunIdentity                # verbatim
    config: dict                         # verbatim, so a checkpoint is self-describing
    config_hash: str
    iteration: int
    cumulative_env_steps: int; cumulative_timesteps: int; cumulative_model_updates: int
    wall_seconds: float; created_unix_ns: int
    state_digest: str
    component_versions: dict[str, int]
    files: dict[str, str]                # every file in this checkpoint -> its sha256
```

`read()` verifies every hash and raises `CheckpointFormatError` naming the first mismatch. That turns
a truncated write from a crash six weeks ago from a silent wrong resume into an error.

On resume the store diffs the loaded config against the current one and **prints every difference**
before continuing. A difference in an identity-hashed field is a refusal, not a warning: a policy
trained on `MockEngine`'s 229-wide vector cannot load into the 1 177-wide one of the full catalogue, and
today nothing announces that mismatch.

`load_checkpoint(folder, strict)` defaults to `strict=True` for a resume and is only `False` for
`load-weights-only`, which starts a new run from an old policy. Tolerate-everything is right for a
research tool and wrong for a harness that promises the curve continues; with `strict=False` a missing
file prints the exact path it wanted and continues with a default.

### 12.3 RNG state

```python
class RngState(msgspec.Struct):
    master_seed: int
    torch_cpu: str                  # hex of torch.get_rng_state()
    torch_cuda: list[str]           # one per device
    python_random: list
    numpy_minibatch: dict           # Generator.bit_generator.state
    iteration: int                  # the counter that addresses act/... and match/... streams
    shard_streams: list[dict]       # per (worker, shard): seed path, reset ordinal, generation
    eval_seed_set_sha: str
```

One reference learner saves no RNG state at all; the other re-seeds all three generators from the
run's *initial* seed on load, so a run resumed at ten million steps draws the same action noise it drew
at step zero. Because every stream here is name-addressed, restoring the iteration counter and the
shard positions restores the *stream*, not merely the parameters.

The buffer is not stored by default. The iteration boundary is the resume point and a fresh iteration
starts from a clean rectangle, so there is no mid-iteration state to preserve; `include_buffer` exists
for anyone who changes the collection rule.

### 12.4 How a resume is proved

`tests/test_resume.py::test_resume_reproduces_metric_stream`, marked slow, on CPU, with `MockEngine`,
a tiny network, `InlineRolloutSource` and `determinism.tier = "run_exact"`:

1. Run six iterations, checkpoint after three, record the full metric stream and every `state_digest`.
2. From the checkpoint in a **fresh process**, run three more.
3. Assert iterations four to six produce **byte-identical** metric rows and `state_digest`s.

`royalelearn verify-resume --config <cfg>` runs the same procedure against a real config on the user's
own machine, which is the only way the promise means anything on hardware this project has not seen.

`docs/checkpoints.md` states plainly what is and is not bit-exact: the *environment* always (the
engine is integer-only and seedable, and `royalegym.replay` re-verifies a trace bit for bit); the
*learner* under `tier = "run_exact"` on the same device class and torch/CUDA build. A driver change is
a documented identity change, not a broken promise.

---

## 13. Metrics and alarms

### 13.1 The sink

`MetricsSink` per section 4.6. Shipped:

- **`JsonlSink`** — always installed, never optional. One msgspec-encoded line per iteration in
  `metrics.jsonl`, plus `episodes.jsonl` and `alarms.jsonl`. No service, no account, replayable, and it
  is the file the resume test compares.
- **`ConsoleSink`** — the per-iteration grouped block, printed by iterating the metric dict rather than
  by indexing a fixed key list, so a new metric cannot crash the console.
- **`ViserSink`** — the learning status the viewer's panel reads, described below.
- **`WandbSink`** — a **decorator** over any sink. It flattens nested keys to `group/name`, persists
  `run_id` in the checkpoint and passes it to `wandb.init(id=..., resume="allow")` so a resumed run
  continues the same wandb run, carries an `enable` flag that makes it a pure pass-through, populates
  the wandb config from the resolved config **and the `RunIdentity`**, and imports `wandb` lazily
  behind an optional extra.
- **`CompositeSink`** — fans out; the default is
  `Composite([Jsonl, Console, Viser, Wandb(enable=False)])`.

**The learning panel is a sink, not a hook.** RoyaleViser draws a learning panel beside the battle it
is already showing, fed by a second datagram sender on the state stream's port + 1. `ViserSink`
implements that sender itself, in about eighty lines of `socket` and `msgpack`: RoyaleLearn does not
import `royaleviser`, both because that package pulls in pygame and because the dependency direction
is the viewer watching the learner and never the reverse. The protocol, pinned in RoyaleViser's
`docs/internals.md` under "The learning status", is:

- the learner binds `host:port` taken from the same `ROYALEVISER` setting as the state stream, with
  the learning port defaulting to the state port + 1;
- it sends nothing at all until a `b"royaleviser 1"` hello arrives. A viewer sends one every second
  and counts as attached while a hello has arrived within three seconds, so a detached run costs one
  monotonic clock read per iteration and no socket traffic;
- while attached it sends one msgpack map `{"learning": {...}}` per iteration, and re-sends the last
  status on a fresh hello so a viewer that starts mid-iteration fills immediately. Either a daemon
  thread or a `pump()` called once a second serves the hellos;
- **one message is the whole status.** The viewer replaces rather than merges, and renders an omitted
  field as an em dash, so a partial update would blank the panel rather than leave it stale.

The field names are fixed by `royaleviser.model.Learning`: `run`, `iteration`, `policy_loss`,
`value_loss`, `entropy`, `kl`, `clip_frac`, `explained_var`, `grad_norm`, `learning_rate`,
`env_steps_per_s`, `engine_ticks_per_s`, `episode_ticks`, `crowns_per_episode`, `towers_per_episode`,
`illegal_rate`, `elixir_wasted`, `elo`, `win_rate`, `pool_size`, `games_vs_pool`, plus an
`"extra": {name: number | str}` map drawn underneath, of which about three rows fit on screen.
`ViserSink` maps the metric row onto those names and spends the three extras on `rating`, formatted
as `"1183 ± 22"` from the fitted rating and its standard error, `cards_per_match`, and the last gate
verdict. `tests/test_viser_sink.py` plays the viewer with a plain socket — send the hello, receive the
datagram, decode it, assert the fields — and imports nothing from RoyaleViser, so the test states the
protocol rather than inheriting it.

Environment-side metrics reach the sink on the round scalars and the `EpisodeRecord`s the worker emits;
learner-side metrics on `UpdateResult`; ladder metrics on the `RatingTable` and `GateDecision`. One row
per iteration merges all three. That is `docs/design.md`'s "one metrics sink fed from both sides".

**Aggregation happens in the worker.** Counters accumulate per step in numpy and one fixed record per
finished episode crosses the boundary. A per-step metric that is always reduced to a mean should be
reduced where it is produced; the reference learner collects an array on every one of fifty thousand
steps and reassembles it in a double python loop in the parent, which costs seconds per iteration
hidden inside a timing residual.

`metrics/schema.py` is the single source of truth: a dict of key to (unit, description, healthy
range). `tests/test_metrics.py` asserts that every key a run emits exists in the schema and that every
schema key is emitted, so the documentation and the code cannot drift.

### 13.2 The metric list

**`run/`** — `iteration`, `cumulative_timesteps`, `cumulative_env_steps`, `cumulative_updates`,
`wall_seconds`, `gamma`, `gae_lambda`, `credit_horizon_seconds`, `ent_coef`, `ent_coef_noop`,
`lr_actor`, `lr_critic`, `determinism_tier`, `resumed_with_drift`, `state_digest`.

**`throughput/`** — `overall_steps_per_second`, `collected_steps_per_second`,
`engine_ticks_per_second`, `rollout_capacity_ratio` (rollout timesteps/s over update timesteps/s; the
invariant of section 2.3, warn under 1.5), `boundary_mb_per_second`, `parent_wait_frac` (the parent blocked on workers that have not
published, which rises when the workers cannot keep up — not a number about workers idling),
`inference_ms_per_round`, `discarded_rows_frac`, `gpu_util_frac`.

**`time/`** — `iteration`, `collection`, `inference`, `env`, `codec`, `ipc`, `critic_pass`, `gae`,
`update`, `checkpoint`, `gate`, `overlap_saved`. The ratio of collection to iteration is how you see
whether a run is environment-bound or learner-bound, and the residual is broken out because the
reference's residual silently absorbs the four places it is actually slow.

**`ppo/`** — `policy_loss`, `value_loss`, `entropy`, `entropy_normalised`, `noop_entropy`, `kl`,
`kl_epoch{0,1,2}`, `clip_fraction`, `clip_fraction_epoch{0,1,2}`, `dual_clip_fraction`,
`explained_variance`, `grad_norm_actor`, `grad_norm_critic`, `update_magnitude_actor`,
`update_magnitude_critic`, `ratio_max_abs_dev`, `advantage_std_pre_norm`, `return_running_mean`,
`return_running_std`, `reward_clip_frac`, `n_minibatches`, `n_optimizer_steps`,
`samples_unused_frac`, `lr_backoff_events`.

| metric | healthy | what it diagnoses |
|---|---|---|
| `explained_variance` | rising to 0.5-0.9 | the critic's health. No reference logs it, and value loss is uninterpretable while returns are normalised by a moving standard deviation. Still negative after fifty iterations is the most likely cause of a plateau |
| `entropy_normalised` | 0.3-0.8 | raw entropy falling is ambiguous — a confident policy or a tighter mask. Only the normalised form separates them |
| `noop_entropy` | above 0.05 nats | the leading indicator of no-op collapse, before `cards_per_match` bottoms out. Taken over the rows whose mask offered more than the no-op, because a decision the elixir bar cannot afford has a binary entropy of zero by construction and most decisions on this environment are that one (section 18, item 8). An unconditioned mean measures the elixir curve: it reads near zero on a healthy run, so a floor on it fires permanently, and a gate that had really collapsed would move it by a fraction of what it moves on the rows that had a choice |
| `clip_fraction` | 0.05-0.20 | pinned near 1.0 is the signature of a rollout/update mask disagreement, or a learning rate far too high |
| `kl` | 0.003-0.02 | below the band, lower `batch_size`; above it, raise `batch_size` or let the backoff act |
| `grad_norm_*` | below `max_grad_norm` most steps | pinned at 0.5 every step means the clip is the binding constraint and the effective learning rate is unknown |
| `credit_horizon_seconds` | 40-50 | logged every run so it can never be ten seconds by accident |

**`policy/`** — `cards_per_match` (healthy about 22; **this and not the no-op rate is the
no-op-collapse metric**, because a healthy policy is about 94% no-op and 99.5% no-op is 1.8 cards a
match and dead), `noop_rate`, `legal_actions_mean`, `legal_actions_p05/p50/p95`, `forced_noop_frac`
(the share with exactly one legal action, which carries zero policy gradient), `tile_entropy`,
`tile_top1_share`, `card_tile_top10_share`, per-card play frequency, and a 32x18 play heatmap per card
as an artifact every `metrics.image_every` iterations.

The healthy figure for `cards_per_match` is elixir arithmetic, not a guess: a full regulation match
generates 85.7 elixir (120 s at 0.357/s plus 60 s at 0.714/s) plus 5 at the start, and at an average
four-elixir card that is about 22 cards.

**`env/`** — `episode_steps_mean` and the **histogram** (a spike at the 480-step cap is the draw and
turtle equilibrium), `episode_steps_p05/p50/p95`, `episodes_completed`, `ticks_mean`, `crowns_for`,
`crowns_against`, `crown_diff`, `tower_hp_frac_end_own/enemy`, `draw_rate`, `win_rate_by_seat` (near
50%; the observation layer guarantees bit-identical mirrored observations, but the shipped engine's
multi-unit ground deploys are not seat-symmetric, so a small persistent deviation can be the game's
own and a large one is a learner-side leak), `elixir_leak_frac`, `elixir_count_exact_frac` (the share
of episode-seats whose count of the opponent's elixir stayed exact; below 1.0 the observation's
enemy-elixir field was an estimate on some episodes and the alarm below says so),
`mean_elixir_at_decision`,
`frac_elixir_above_99`, `illegal_action_rate`, and `reward_terms/<name>` per weighted term.

**`ladder/`** — `rating`, `rating_se`, `rating_ci95_lo/hi` per member, `rating_above_v0`,
`elo_readout`, `champion_id`, `champion_step`, `pool_size`, `sampler_size`, `gate_attempts`,
`gate_passes`, `gate_observed_rate`, `gate_lower_bound`, `gate_failed_condition`, `gate_seconds_frac`,
`score_vs_noop`, `score_vs_random_legal`, `transitivity_residual`, `paired_rho`, `draw_rate_eval`,
`eval_games_total`, `evictions`.

**`health/`** — `illegal_action_rate` (**exactly zero by construction; this is an alert, not a plot**),
`mask_disagreements` (from the start-up gate), `worker_restarts`, `worker_failures_by_kind`,
`rows_dropped_dead_worker`, `obs_codec_clipped`, `samples_unused_frac`, `nan_guard_trips`,
`vram_peak_mb`, `rss_peak_mb`, `buffer_fill_frac`.

### 13.3 Alarms

`metrics/alarms.py`. Each alarm is `(name, predicate, severity, patience)` and fires when its predicate
holds for `patience` consecutive iterations. `severity="warn"` logs and writes an `alarms.jsonl` row;
`severity="halt"` additionally writes a checkpoint and a diagnostic bundle, then raises `AlarmHalt` so
the process exits non-zero. Thresholds live in `config.alarms`, are recorded, and are excluded from the
identity hash because an alarm can stop a run but never alter a number.

**A halt is a clean stop, never a death, and it explains itself.** The checkpoint is written *before*
the exception, so a halted run is always resumed rather than restarted; the bundle carries the last
fifty metric rows beside it. The final console line, and a `halt.json` in the bundle, name the alarm,
the threshold it crossed, the last five values of every metric the alarm reads, the checkpoint path
and the resume command verbatim. The reason a halt states its own evidence is that the first question
anyone asks on finding a stopped run is whether the stop was real, and a line reading only that a
metric crossed a threshold costs the same morning as no run at all.

This is also why no halting alarm has `patience = 1` on a *learning* quantity. The three that do —
`illegal_actions`, `ratio_invariant`, `nonfinite`, `buffer_overflow` — are correctness assertions
whose first occurrence is already a defect, and continuing past one wastes the compute that follows.
Everything that measures how training is *going* waits several consecutive iterations, because a
policy that is still near-random moves these quantities around for reasons that are not the failure
being hunted, and a false halt costs everything the run was for.

| alarm | predicate | patience | severity | what it means |
|---|---|---|---|---|
| `illegal_actions` | `env/illegal_action_rate > 0` | 1 | **halt** | a mask bug, an unmasked policy, or a wrong action encoding. With a correct mask it is exactly zero |
| `ratio_invariant` | `ppo/ratio_max_abs_dev > 5 * ratio_atol` | 1 | **halt** | a mask, codec or weight-version mismatch (section 9.6). The multiple of the configured tolerance is what keeps the alarm meaningful in fp32 and under bf16 alike; all three failures it is aimed at produce a deviation of order one |
| `nonfinite` | any loss, gradient or logit non-finite | 1 | **halt** | |
| `buffer_overflow` | `health/buffer_fill_frac > 1.0` | 1 | **halt** | an invariant is broken. A healthy rectangle reads **exactly one**: every cell is written once per iteration, so anything under one is a cell nobody filled and anything over it is impossible. The threshold is above one and not below it for that reason — a bound of 0.98 would halt every healthy run on its first iteration |
| `worker_failures` | `health/worker_restarts` rose | 1 | warn | |
| `worker_failures_persistent` | rose on three consecutive iterations | 3 | **halt** | |
| `clip_pinned` | `ppo/clip_fraction > 0.5` | 3 | **halt** | mask disagreement, or a learning rate far too high |
| `kl_high` | `ppo/kl > 0.05` | 3 | warn | the backoff should already be acting |
| `kl_dead` | `ppo/kl < 1e-5` | 10 | warn | nothing is moving: dead entropy, learning rate too low, or a frozen head |
| `ev_negative` | `ppo/explained_variance < 0` | 50 | warn | the most likely cause of a plateau |
| `noop_collapse` | `policy/cards_per_match < 8` | 5 | warn | about 22 is healthy |
| `noop_collapse_severe` | `policy/cards_per_match < 3` | 5 | **halt** | |
| `noop_entropy_floor` | `ppo/noop_entropy < 0.02` | 5 | warn | the leading indicator; it watches after `ent_coef_noop` has annealed to zero, which is when it matters most |
| `tile_spam` | `policy/tile_top1_share > 0.25` | 5 | warn | |
| `artefact_exploit` | `policy/card_tile_top10_share > 0.5` | 5 | warn, and dump five winning traces | a real meta is not that concentrated |
| `draw_equilibrium` | `env/draw_rate > 0.5` and `env/episode_steps_at_cap_frac > 0.8` | 5 | warn | the turtle equilibrium |
| `seat_bias` | `env/win_rate_by_seat`'s 95% interval excludes 0.45-0.55 | 3 | warn | an unseeded reset, a reward asymmetry, or an observation mirror bug; a few points inside that band can be the shipped engine's own seat asymmetry, which is why this warns rather than halts and why the ladder's paired evaluation swaps sides on every seed |
| `elixir_count_inexact` | `env/elixir_count_exact_frac < 0.99` | 3 | warn | the observation's opponent-elixir field is an estimate on some episodes: a repeated card in a deck, or an engine whose elixir law is not the calibration's. The policy is reading a documented-exact slot that is not. A value near zero rather than slightly under one is the second cause and not a broken counter: it says the engine build and the card data disagree about elixir, so read `run/engine_build_digest` before anything else |
| `shaping_dominates` | `sum of absolute shaping terms > absolute terminal term` | 5 | warn | shaping has taken over the objective |
| `transitivity` | `ladder/transitivity_residual > 0.10` | 3 | warn | the scalar rating is lying |
| `gate_starved` | five consecutive gate failures | 1 | warn | the plateau signal, stated as an event |
| `capacity_ratio` | `throughput/rollout_capacity_ratio < 1.5` | 3 | warn | the harness is becoming the bottleneck |

**An alarm nobody has seen stay silent is unvalidated.** Watching one fire proves only that it
can fire; what validates a threshold is a healthy iteration where the metric sits clearly on the
right side of it with room to move. An alarm that holds on every healthy row is worse than no
alarm, because a channel that always speaks teaches its reader to stop listening, and the event
it was built for then arrives invisibly. That is not hypothetical here: `noop_entropy_floor` held
on both iterations of the first real run and was read twice, by two people, as a policy warming
up.

Measured against those two iterations — the only healthy rows this harness has produced — four
alarms held and two of them are a threshold problem rather than a finding:

| alarm | held | reading |
|---|---|---|
| `noop_entropy_floor` | both rows | **the threshold was wrong**, and the metric is now conditioned on the rows that had a choice (section 18, item 8). The number those rows report is the unconditioned one |
| `kl_dead` | both rows, at 4.8e-06 and 3.6e-07 | **the same fault, and the same repair.** A forced row's ratio is exactly one — the distribution is a point mass at the same action before and after the update — so it contributes zero KL structurally. `ppo/kl`, `ppo/clip_fraction` and `ppo/dual_clip_fraction` are now means over the rows that had a choice, and the threshold stands. This reached past the dashboard: `lr_backoff` reads this KL, so a diluted one put the brake that stops a blow-up out of reach by the same factor |
| `ev_negative` | first row only | healthy: the critic had seen one batch, and explained variance was +0.25 by the second. Patience is 50 |
| `noop_collapse` | first row only | healthy: `cards_per_match` was 6.98 on a freshly initialised policy and 22.08 by the second row. Patience is 5 |

So two alarms are fixed and two behaved. The rule that produced
that table is worth more than the table: before shipping a threshold, measure the quantity on the
population it will really be averaged over, and if that population is mostly structural zeros,
**condition the metric rather than
lowering the number** — because the share of rows that are structural zeros is itself a property
of the game rather than of the learner. Here it is the elixir economy, and it moves as the policy
learns to hold elixir, as the deck changes, and in overtime at double rate. A threshold re-tuned
against today's share is a number with an expiry date nobody will notice passing.

What a validated threshold looks like, from the two that behaved: held on the first iteration,
cleared by the second, with patience long enough to absorb the start. A new alarm can be checked
against that shape in a minute.

The other alarms have been checked against two iterations of one profile, which by the rule above
is not validation. A metric's row population belongs in its identity rather than in its
implementation — `ppo/kl@choice` against `@all`, declared in the schema — so a threshold cannot
be set against the wrong population by accident and a reviewer sees the fault without running
anything. That is the systemic form of both repairs, and it is not built.

### 13.4 The diagnostic bundle

On any halt, `metrics/bundle.py::write_bundle` produces `<run>/bundles/<iteration>/` containing: the
last fifty metric rows, the alarm rows, the resolved config, the `RunIdentity`, the last twenty
`EpisodeRecord`s, the ten worst `ratio` outliers with their slot, cycle and episode ordinal, a
`royalegym.replay.Trace` of one offending episode re-simulated from its shard seed and reset ordinal
(so it is exact and self-verifying with `verify_trace`), and the current `state_digest`. One directory,
attachable to an issue. This is the concrete form of `docs/design.md`'s "a learner that is 80% right
produces a bot that loses for reasons nobody can attribute".

---

## 14. The loop, the entry point and the CLI

### 14.1 `LearningCoordinator`

`royalelearn/coordinator.py`. The only place in the package where the phases of an iteration are
ordered.

```python
class LearningCoordinator:
    def __init__(self, cfg: RunConfig) -> None: ...
    def __enter__(self) -> "LearningCoordinator":
        """Runs preflight (section 7.7) and raises PreflightError with the RustEngine stale-build
        text verbatim if the engine build disagrees with the data on disk."""
    def __exit__(self, *exc) -> None:
        """Idempotent close: CLOSE to every worker, join with a timeout, terminate the stragglers."""
    def learn(self, until_timesteps: int | None = None) -> None: ...
```

```
while cumulative_timesteps < limit:
    plan = matchmaker.plan(iteration, pool, ratings, geometry)   # the opening table only
    source.begin_iteration(plan, buffer, iteration)
    for t in range(T):                                   # one cycle = shards_per_worker rounds
        for _ in range(shards_per_worker):
            round = source.next_round(cfg.rollout.round_timeout_s)
            assigned = assign_ended_battles(round, matchmaker, pool, ratings)   # section 7.5
            actions, log_probs = inference.act(round, uniforms[t])
            source.submit(Step(actions=actions, gamma=sched.gamma, **assigned))
            buffer.record_round(round, actions, log_probs)
            episodes.extend(round.episodes)
    handle_failures(source.drain_failures())             # restart, mark rows invalid, count
    result   = update.step(buffer, sched)                # critic pass, GAE, PPO; section 9.5
    ladder.record_training_results(episodes)             # kind="train"; excluded from the fit
    if crossed(cfg.ladder.candidate_every_env_steps):
        candidate = snapshots.put(actor_critic, cumulative_env_steps)
        decision  = gate.evaluate(candidate, pool, eval_runner)   # builds and closes its own farm
        ladder.apply(decision)
    if iteration % cfg.ladder.refit_every_iterations == 0:
        ratings = rater.fit(results.eval_view())
    row = merge(run_fields, throughput, time, result, episode_stats(episodes), ladder_fields)
    alarms.evaluate(row)                                  # may raise AlarmHalt
    sinks.write(row); sinks.write_episodes(episodes)
    if crossed(cfg.checkpoint.every_env_steps):
        store.write(components, manifest())
    sched.advance(cumulative_env_steps)
    iteration += 1
```

With `rollout.overlap = true` the collection of iteration `i` runs on a second thread and a second
buffer while the update of iteration `i-1` runs on the default CUDA stream, against a
`BehaviourSnapshot` of the actor taken at the iteration boundary. The lag is exactly one iteration by
construction and the stored log-probs are the snapshot's, so the PPO ratio measures the true
off-policyness and the clip bounds it. `ppo/behaviour_lag_iterations` is logged as 0 or 1 so the
regime is visible in the run's config panel, and because sampling is driven by name-addressed uniforms
the trajectory is identical either way.

Invariants asserted once per iteration, each raising a named exception carrying the offending slot and
cycle:

```python
assert (round_counter == T * shards_per_worker)
assert buffer.trainable_mask().sum() >= cfg.ppo.timesteps_per_iteration * 0.98
assert np.isfinite(buffer.log_prob[buffer.trainable_mask()]).all()
assert mask_bit(buffer.mask_bits, buffer.action)[buffer.trainable_mask()].all()
assert (buffer.deploy_status[buffer.trainable_mask()] <= 0).all()
assert episodes_end_in_pairs(episodes)
assert abs(discarded_rows_frac - expected_from_mix) < 0.05
assert assignments_constant_within_episodes(buffer.group, buffer.episode_end)
```

The last of those is what makes "one policy for one whole episode" checkable rather than merely
intended: a battle's `group` entry may change only on the cycle after its `episode_end`, anywhere in
the rectangle. It is one vectorised comparison over `(T, R)` and it catches every way an assignment
could reach the middle of a trajectory.

### 14.2 The file a user runs

`examples/train_1v1.py` — about fifteen lines, all of it the things a bot creator actually changes.

```python
from pathlib import Path
from royalelearn import LearningCoordinator, load_config

def main() -> None:
    cfg = load_config(Path(__file__).parent / "configs" / "laptop.json")
    cfg.run_name = "my-first-run"
    cfg.advantage.gae_lambda = 0.99      # the number to think about; see docs/harness-spec.md
    cfg.metrics.sinks[3].enable = True   # weights and biases
    with LearningCoordinator(cfg) as run:
        run.learn(until_timesteps=100_000_000)

if __name__ == "__main__":
    main()
```

`examples/custom_reward.py` shows the other thing a bot creator changes: a `RewardFunction` subclass,
registered through `config.extra_component_modules` so that `EnvFactorySpec` may instantiate it, and
therefore recorded in the checkpoint and in the ladder's `context` like any other component.

### 14.3 The CLI

`python -m royalelearn <command>`, also installed as the `royalelearn` console script.

| command | what it does |
|---|---|
| `config [--profile laptop\|workstation\|many-core] [-o run.json]` | write a fully populated default config |
| `doctor [--config F]` | the start-up gates of section 7.7 on their own: build one env, print the engine build digests, the observation space and the codec table, run `mask_disagreements` over all 2304 non-no-op actions, check the action-layout identity, print the RAM ledger, the credit horizon, the geometry and the `run_id`. Seconds, and it catches most first-run failures |
| `bench [--config F] [--seconds 60]` | measure and print section 2.3's table for **this** machine: env milliseconds per game-step, codec microseconds per row, boundary microseconds per round, inference milliseconds per round, update timesteps per second, peak VRAM, the rollout/update capacity ratio, and `ratio_max_abs_dev` over ten rounds beside the configured `ratio_atol`. Writes the "measured on" block in `docs/throughput.md` |
| `train --config F [--run-name N] [--until-timesteps T] [--inline] [--device cuda\|cpu]` | a new run |
| `resume --run DIR [--checkpoint PATH] [--until-timesteps T] [--allow-identity-drift]` | continue; refuses on an identity mismatch by default and names every differing field |
| `verify-resume --config F [--iterations 6] [--split 3]` | section 12.4's proof, on the user's own machine |
| `eval --run DIR --a ID --b ID [--seeds N]` | a paired evaluation between two members, with its interval, outside the training loop |
| `gate --run DIR --candidate ID` | re-run a gate decision from stored snapshots and print the verdict |
| `rate --run DIR [--output F]` | refit ratings from `games.jsonl` and print the table with intervals and the transitivity residual |
| `identity --config F` | print the `RunIdentity` and the `run_id`: the thing to paste into an issue |
| `replay --run DIR --episode W/S/B/O [--out trace.msgpack]` | re-simulate one episode from its shard seed and reset ordinal and verify it with `royalegym.replay.verify_trace` |
| `play --checkpoint DIR [--opponent random\|noop\|<snapshot>] [--viser]` | watch one battle; `--viser` builds the single-battle vec env with `viser="env"`, which is the same path worker 0's first shard takes during a run |

`cli.py` sets `CUBLAS_WORKSPACE_CONFIG` and the BLAS thread variables **before importing torch**, and
`royalelearn/__init__.py` imports nothing that needs torch, so `royalelearn --help`, `royalelearn
config` and `royalelearn identity` work in an environment without it.

`--resume latest` resolves by reading the run index and **printing what it chose**. There is no
`"latest"` string inside the config: one reference resolves it by string-munging the save path, so a
run named `myrun` and one named `myrun-variant` collide.

### 14.4 Interactive control and crash discipline

Checked once per **round**, not once per iteration, via a sentinel file in the run directory and a
`SIGINT` handler, so neither a tty nor a busy-wait is needed: `c` checkpoint now, `q` checkpoint and
quit, `p` pause (a blocking wait on an Event).

The whole loop is wrapped in `try/except (Exception, KeyboardInterrupt)` — `KeyboardInterrupt`
explicitly, because it is not an `Exception` and a bare `except Exception` therefore skips its own
emergency save on Ctrl-C — with a nested `try` around the emergency checkpoint, then a `finally` that
closes every worker, joins **with a timeout** and terminates the stragglers.

---

## 15. The test plan

Every test runs on **CPU**, uses **`MockEngine`**, and never invokes `cargo` or `maturin`. MockEngine's
16-card catalogue gives a vector nowhere near the width of the full one, which doubles as the standing
check that no width is written down anywhere: a test that passes on both catalogues cannot contain a
literal from either. Networks in tests are `channels=8, blocks=1`. Markers: `slow` (over five
seconds) and `engine` (needs a fresh `royalesim` build). `addopts` runs everything not marked `slow`
or `engine`; the repo gate runs `pytest -q` and then `pytest -q -m "slow and not engine"`.

Neither reference learner has a single test file. That is the clearest place where the reference must
not be imitated: the layers below this one are held to full suites and this one carries the release
metric.

| file | asserts | speed |
|---|---|---|
| `test_package.py` | imports without torch; the public names resolve; the six seed modules are **gone**; asking for a torch-dependent name without torch raises an `ImportError` that names the missing package | fast |
| `test_config.py` | JSON round-trip is canonical and idempotent; a typo is rejected at every nesting level; the hash is stable under key reordering; each shipped profile is internally consistent (`T * learner_rows >= timesteps_per_iteration`, `batch_size % minibatch_size == 0`) | fast |
| `test_seeding.py` | `derive_*` is stable across processes and platforms against pinned values; adding a new stream name does not change an existing stream; two names do not collide on their first four draws | fast |
| `test_identity.py` | the identity is order-independent JSON; table-driven over every field, each included field changes `run_id` and each excluded field does not, `frame_stack`, `codec_table_digest` and a changed `Reveal` among the included ones | fast |
| `test_obs_layout.py` | `hand_card_onehot`, `hand_cost` and `hand_affordable` are resolved **by name** from `vector_layout()` and the resolved slices match the environment's own, on `MockEngine` and on a synthetic full-catalogue spec whose widths differ from it; a layout missing a required name raises `PreflightError` naming that name; no offset is computed from `V` anywhere in the module | fast |
| `test_codec.py` | the codec table is decided from the space: a plane whose declared `high` exceeds 255 lands in `float16` and one under it lands in `uint8`, a plane the layout declares static is not stored and one it does not declare static **is** stored even when it is constant on the sample; pack then unpack is **exact** on every `uint8` plane and on the mask; the fp16 parts round-trip within 2^-10 relative on 10 000 real observations; `mask_planes` reconstructed at unpack equals the environment's, bit for bit; `row_bytes` matches the formula of section 2.1 on both catalogues; the clipping counter fires on a synthetic count above a declared bound; `codec_version` and `codec_table_digest` are in the identity | fast |
| `test_layout.py` | the shared-memory offsets and sizes are self-consistent and stable against a golden record, so a change to `LAYOUT_VERSION` is deliberate; a short block raises at construction, not at first write; the error-byte protocol round-trips a traceback | fast |
| `test_action_layout.py` | for all 2304 non-no-op actions, a one-hot `(hand_size, tiles_y, tiles_x)` tensor reshaped in C order has its argmax at `parser.encode(slot, x, y)`; index 0 is the no-op. Exhaustive, because this is the most load-bearing index identity in the harness | fast |
| `test_distribution.py` | illegal actions have `p == 0` exactly and `log_prob == -inf`; entropy is finite with all but one action masked; `mode()` respects the mask; sampling from fixed uniforms matches a brute-force inverse CDF and never returns an illegal action; `mask[NOOP]` is set on 10 000 random RoyaleGym states including a `game_over` one | fast |
| `test_nets.py` | every documented shape and dtype at B = 1, 7, 512, on both catalogues and at `frame_stack` 1 and 2; the stem's input width equals `k*(S+P) + 2 + E` computed from the spec; logits are float32 under autocast; the gradient into a masked logit is exactly zero; the entropy term's gradient into critic parameters is exactly zero for both trunk variants; orthogonal gains are as specified; parameter counts within 5% of the documented figures; `arch_digest` round-trips, covers `frame_stack`, and a mismatching digest is refused with a message rather than a shape error | fast |
| `test_gae.py` | the vectorised implementation equals `reference_gae` to 1e-6 over random inputs; **terminated bootstraps from 0 and truncated bootstraps from `final_value`** (named as a regression test for the reference's bug); the recursion breaks at a truncation as well as at a termination; a one-cycle episode; a dead-worker prefix; all-terminated and none-terminated; the reported credit horizon matches `1/(1 - gamma*lambda)` | fast |
| `test_returns.py` | Welford matches numpy's mean and `n-1` variance over 10^5 samples; the state round-trips through JSON; the mean is never subtracted | fast |
| `test_buffer.py` | `record_round` writes exactly the right cells; minibatches cover every trainable valid cell exactly `n_epochs` times with nothing dropped; a batch never straddles an epoch; the pinned staging ring does not alias; the measured footprint matches the `row_bytes` formula of section 2.1 to the byte | fast |
| `test_frame_stack.py` | at `k = 2` the two frames of a stacked cell have consecutive `info["tick"]` values and belong to the same episode; the first cell of an episode stacks a zero history; cycle 0 of an iteration stacks the history rows carried over from the previous one; `vector` is the current frame's and is not stacked; `frame_stack` changes `arch_digest` and `obs_digest`; at `k = 1` the stacked observation is byte-identical to the unstacked one | fast |
| `test_ppo.py` | gradient accumulation over `k` minibatches gives the **same** gradient as one full batch to 1e-5 — the property that makes `minibatch_size` a pure memory knob; clip fraction and KL match hand-computed values on a synthetic batch; dual clip binds only for a negative advantage; `ratio == 1` gives exactly `-mean(A)`; the two mask asserts fire when fed a deliberately mismatched mask; the backoff fires after exactly `patience` consecutive breaches and floors at `lr_min` | fast |
| `test_schedules.py` | the gamma and entropy anneals hit their endpoints exactly at the stated env-step count; the whole schedule state round-trips | fast |
| `test_rewards.py` | each potential term equals `gamma*Phi(s') - Phi(s)` on a hand-built transition; the composition is antisymmetric between seats on a mirrored transition; `set_gamma` reaches every term and matches the schedule; the committed-elixir potential is zero for a card played and negative for elixir left in the bar while the opponent's rises | fast |
| `test_rollout_inline.py` | an `InlineRolloutSource` run of 20 cycles: slots map to the right battles, rewards are antisymmetric on mirror battles, episode ends arrive in pairs, `deploy_status` is never in 1..11, the seven terminal scalars are read out of `final_info` and match what the worker counted, and a battle's assignment changes only on the cycle after its `episode_end` | fast |
| `test_rollout_farm.py` | **the differential test**: `ProcessRolloutSource` and `InlineRolloutSource` with the same seed produce byte-identical buffers, scalars and episode records over 30 cycles. This is the acceptance criterion for any future Rust worker | slow |
| `test_worker_hygiene.py` | the worker has no `torch` in `sys.modules`; the thread variables are set before numpy; a deliberately raised exception arrives as `WorkerFailure(kind="exception")` with the traceback, the farm restarts the worker with the next generation, the run continues, and `health/worker_restarts` rises; a hung worker produces `WorkerTimeout` within the timeout rather than hanging | slow |
| `test_rollout_invariants.py` | the section 14.1 assertions fire on deliberately corrupted rounds: a dropped cycle, a mismatched slot count, an action illegal under its stored mask, a truncated cell with no `final_value` | fast |
| `test_rating.py` | the Bradley-Terry-Davidson fit recovers known strengths from synthetic results within its own standard errors over 100 seeds and to within 5 Elo at n = 2000; it is invariant to result order and to a permutation of player ids; standard errors shrink as one over the square root of n; the anchor is pinned exactly; the standard error of a difference uses the 2x2 block; the Davidson term recovers a known draw rate; the residual is near zero on transitive data and large on synthetic rock-paper-scissors; `wilson_interval` matches the published table | fast |
| `test_results_log.py` | the aggregate cache equals a rebuild from `games.jsonl`; a truncated last line is tolerated; contexts are never pooled without the flag | fast |
| `test_matchmaker.py` | the empirical mixture matches the configured shares over 10^5 draws; PFSP weights are floored; `assign(battle, ordinal, ...)` is a pure function of its arguments and the master seed, and gives the same answer at a different iteration length and a different worker count; `plan()` equals `assign()` applied across the geometry; at most `max_resident_opponents` snapshots appear; an assignment is binding for a whole episode | fast |
| `test_gate.py` | each of the three conditions fails independently and the decision names which; an observed 55.2% at n = 1000 passes and 55.0% fails; condition 3 compares the candidate's observed mean against the champion's **fitted** predicted mean and passes on a champion with no games against the stratified eight; the cycle case admits without promoting; paired scoring collapses sides correctly and is symmetric under swapping A and B; the bootstrap interval covers a known rate at the nominal level over 200 synthetic replications | fast |
| `test_eviction.py` | anchors, v0 and the champion chain are never evicted; the stratified sample spans the rating range; `cycle`-flagged snapshots are preferred; the archive and the log are untouched | fast |
| `test_snapshots.py` | save and load round-trip; a mismatching `arch_digest`, `obs_digest` or `codec_table_digest` is refused with the field named; `EvalRunner` refuses a pairing whose two `obs_digest`s differ and names both; the LRU evicts; a byte scan finds no pickle protocol marker in any written file | fast |
| `test_metrics.py` | every key a run emits exists in `schema.py` and every schema key is emitted; nested keys flatten to `a/b`; the wandb sink is a pure pass-through when disabled; the decorator's checkpoint nests | fast |
| `test_viser_sink.py` | a plain `socket` plays the viewer: nothing is sent before a hello, a hello produces one msgpack map within a second, every fixed field of the learning status is present with the right type, the three extras carry the formatted rating, `cards_per_match` and the last gate verdict, a second hello re-sends the last status unchanged, and a detached sink sends nothing over fifty iterations. Imports nothing from RoyaleViser | fast |
| `test_alarms.py` | every alarm fires on a synthetic row and does not fire on a healthy one; patience is honoured; a halt alarm raises `AlarmHalt` and writes a bundle | fast |
| `test_checkpoint.py` | every component round-trips; with `strict=False` a missing file prints its path and continues, with `strict=True` it raises; a flipped byte is detected by the manifest; a higher `format_version` is refused by name; pruning keeps exactly `keep` and survives a stray file and a `.partial` directory; the write is atomic under a simulated crash, on a platform where a directory handle cannot be fsynced as well as on one where it can | fast |
| `test_rng_roundtrip.py` | torch CPU, the numpy generator and python `random` round-trip and reproduce their next 1000 draws | fast |
| `test_no_global_rng.py` | a source scan finds no module-level `np.random.<func>`, no bare `random.`, and no unseeded `torch.rand*` in `royalelearn/` | fast |
| `test_env_contract.py` | every layout fact the harness relies on, **each assertion naming the RoyaleGym file it depends on**: the action space is `Discrete(1 + hand_size*tiles_y*tiles_x)` and index 0 is the no-op; `encode` matches the head's arithmetic; `mask_planes == action_mask[1:].reshape(hand_size, tiles_y, tiles_x)` wherever the key exists; `vector_layout()` is contiguous, covers the whole vector and declares the three hand fields; `spatial_layout()` declares one entry per plane and marks the static ones; `observation_space["spatial"].high` is per channel; SAME_STEP autoreset with `final_obs`; the seven terminal scalars under `final_info` with their validity masks; `info["tick"]` present and surviving into `final_info`; `mask[NOOP]` always set; episodes end in pairs; `deploy_status` present; `config()` returns every documented key after a reset. A RoyaleGym change then breaks this file loudly rather than the learner silently | fast |
| `test_resume.py` | section 12.4's identical-metric-stream and identical-`state_digest` proof, in-process and via a subprocess | slow |
| `test_resume_identity_guard.py` | changing `master_seed`, the arch, `codec_version`, `decision_ms` or `workers` each refuses the resume and names that field; `--allow-identity-drift` writes `drift.json` and marks the rows | slow |
| `test_ratio_invariant.py` | on CPU in fp32, a two-iteration run has `ratio_max_abs_dev` under 1e-6; perturbing the stored mask, the codec table or the worker's weight version each trips the assertion with the right message and a deviation of order one. There is no GPU test of the bf16 tolerance: what that tolerance is worth is measured by `bench` on the machine that will run, not asserted here | slow |
| `test_replay_episode.py` | an episode replayed from its shard seed and reset ordinal returns an empty divergence list from `royalegym.replay.verify_trace` | slow |
| `test_smoke_train.py` | `examples/configs/smoke.json` runs three iterations end to end: no NaN, no illegal actions, a checkpoint written and reloaded, ratings refitted, one gate at reduced `n`, and a metric row with every schema key present and finite | slow |
| `test_bench_report.py` | `bench` runs and produces every field; asserts nothing about the values, which are machine-dependent, but prints them, exactly as RoyaleGym's throughput test does | slow |

Target: about 52 fast tests under 30 seconds, about 9 slow tests under five minutes. None needs a GPU,
and the torch-dependent tests skip cleanly when torch is absent so that `pytest` still passes in a
virtual environment that has not finished installing it.

---

## 16. What the harness asks of RoyaleGym

Filed as issues there, each with the measured cost. **None blocks the first run**: the harness works
against the surface it is given, and each of these removes a workaround or is a pure speed-up.

The surface the harness reads its layout, its identity and its episode statistics from is in place,
which is why sections 4.1, 7.2 and 7.7 describe reading rather than inferring:
`ClashParallelEnv.config()` and `RustEngine.build_digest()` for the identity and the ladder context;
`ObsBuilder.vector_layout()` and `ObsBuilder.spatial_layout()` for the field offsets and the static
planes; `mask_planes` and per-channel observation bounds for the codec table; the seven terminal
scalars under `final_info` for the `EpisodeRecord`; per-team reward terms and float32 rewards for the
metric stream; and `ClashSelfPlayVecEnv(..., viser=...)` for the state stream. What remains open:

| # | ask | cost today | size |
|---|---|---|---|
| 1 | **Rust-backed default observation and mask** (RoyaleGym's own job 7) | 200 of 298 microseconds per transition, 67% of the rollout side; landing it takes rollout capacity from about 3 300 to about 9 000 timesteps/s and frees two cores | large, already planned |
| 2 | `ClashSelfPlayVecEnv(..., copy=False)` — drop the per-step `copy.deepcopy` of the whole batch | 14 microseconds per transition, 387 KB per step | one line |
| 3 | `PlacementOracle` grid reuse between the mask's grids and the observation's | up to 7 of 14 `point_grid` calls per step, about 80 microseconds per game-step | small |
| 4 | `selfplay._Record` gains separate `wins`/`draws`/`losses`; `record_result(..., eval=False, context=...)`; atomic `save`; `fit_ratings` and `is_stronger`; an `EvictionPolicy` ABC whose default is not oldest-first; `_evict` stops leaking `records`. Plus the potential-based reward terms of section 10 | draws are lossy, which breaks every binomial interval; `_evict` deletes exactly the diverse opponents the uniform floor protects; `ElixirTradeReward` rewards turtling at a magnitude comparable to the terminal reward | medium; the harness ships its own until then |
| 6 | `call`/`get_attr`/`set_attr` on the vec env; a counter for entities dropped past `max_entities` | reaching into `vec.envs[i]` is undocumented; a silent truncation during training is the class of bug `docs/design.md` warns about | small |

Ask 1 lands as a **drop-in speed-up, not a rewrite**, because nothing in this design touches the
Python builders' internals — only `single_observation_space`, the `Dict` key names, the two layout
methods and the mask contract. That is the property to protect in review.

---

## 17. Landing order

Seven stages. Each ends green and none leaves a half-wired module in the tree. Nothing in any stage
runs `cargo` or `maturin`: the whole plan runs against the already-built `royalesim` and against
`MockEngine`, which is deliberate, because it is what lets the middle stages proceed in parallel on a
small machine.

**Stage 0 — clear the seed.** Delete the six vendored modules, `SEED_CLASSES`, `NOTICE`,
`LICENSE-APACHE-2.0` and the ruff exclusion list; update `pyproject.toml` (licence, dependencies,
extras, markers); rewrite `royalelearn/__init__.py` and `tests/test_package.py`. This lands first
because everything else touches `pyproject.toml`.

**Stage 1 — the spine.** `config.py`, `seeding.py`, `determinism.py`, `identity.py`, `errors.py`,
`obs_layout.py`, `version.py`, the whole of `api/`, `rollout/layout.py`, `rollout/envspec.py`,
`metrics/schema.py`, and their tests. Nothing downstream can start until the ABCs and the byte layout
are frozen, because they are the interfaces everything else is written against.

**Stage 2 — four independent tracks.** Each owns disjoint files and depends only on stage 1.

- *Networks*: `learn/nets.py`, `learn/distribution.py`, `learn/actor_critic.py` and their tests.
- *Data path*: `rollout/codec.py`, `learn/buffer.py`, `learn/gae.py`, `learn/returns.py`,
  `learn/schedules.py` and their tests.
- *Rollout*: `rollout/plan.py`, `rollout/scripted.py`, `rollout/worker.py`, `rollout/farm.py`,
  `rollout/inline.py`, `rollout/preflight.py` and their tests. `InlineRolloutSource` lands **first**
  and is the semantic reference; the farm is accepted only when the differential test shows
  byte-identical buffers against it.
- *Ladder, metrics and checkpoints*: `ladder/**`, `metrics/**` except `schema.py`, `checkpoint.py` and
  their tests. This track is pure numpy and statistics with no torch and no env, so it is the cleanest
  one to start first if effort is scarce.

**Stage 3 — the update.** `learn/ppo.py`, `learn/inference.py`, `rewards.py` and their tests. Depends
on the networks and the data path; `ppo.py` and `buffer.py` are coupled through the minibatch path, so
splitting them costs more in interface churn than it saves.

**Stage 4 — the run.** `coordinator.py`, `cli.py`, `__main__.py`, `metrics/alarms.py`,
`metrics/bundle.py`, `examples/**`, and the slow tests: resume, the identity guard, the ratio
invariant, replay and the smoke run.

**Stage 5 — verification.** The full suite once, plus `royalelearn doctor` and `royalelearn bench`,
with the measured numbers pasted into `docs/throughput.md` and the README. This is the only stage that
runs the whole sweep; earlier stages scope `pytest` to the files they touch and report what they
changed rather than proving it with a full gate. Then the documentation pages of section 3, written
against the code as it actually landed, because every number in them is measured here.

**Stage 6 — the first real run.** Twenty-four hours on `MockEngine` first (no rebuild, and it is
faster), then `RustEngine`. Acceptance: `health/illegal_action_rate` exactly zero,
`ppo/explained_variance` positive by iteration 50, `policy/cards_per_match` above 10,
`env/win_rate_by_seat` inside 0.45-0.55, `throughput/rollout_capacity_ratio` above 1.5, and one gate
passed.

---

## 18. What the first run measures

Recorded here so that they are measurements rather than arguments. Each has a default that is safe to
run with, and each is a number the harness already logs.

1. **Achieved bf16 throughput and peak VRAM on the 3050.** Everything in section 2 scales off the
   assumed 9 TFLOP/s and 1.18 MB of activations per sample. `bench` answers it before a single gradient
   step. If the achieved rate is nearer 4 TFLOP/s the run is 2.2 times slower than budgeted, and the
   three pre-designed escapes, in order of preference, are `net.channels` 64 to 48 with `blocks` 4 to
   3, then `ppo.n_epochs` 3 to 2, then `rollout.overlap` on. The decision is made from the logged
   timing split, and the chosen values go into the identity so the two regimes are never confused in a
   plot.
2. **Worker resident memory with 32 battles.** If it exceeds about 250 MB, lower `games_per_worker`
   before lowering `workers`: the inference batch matters more than the worker count.
3. **The draw rate in self-play.** It decides Davidson against half-a-win in the rater and it changes
   the gate's variance. Measurable today with `RandomLegalOpponent` self-play, cheaply.
4. **The paired-seed correlation `rho`.** It sets the real effective sample size for the gate and it
   says how much of a battle the start state decides.
5. **The transitivity residual.** If self-play here produces genuine rock-paper-scissors between
   snapshots, the scalar release metric needs replacing, and the `Rater` ABC is the seam for it.
6. **Whether the no-op guard was needed at all.** `ent_coef_noop` anneals from a small positive
   coefficient to zero over the first ten million env steps, and `run/ent_coef_noop`,
   `ppo/noop_entropy` and `policy/cards_per_match` are all logged, so the run says plainly whether
   `cards_per_match` held up on its own as the coefficient fell, whether it needed the guard for
   longer, or whether it never came near collapsing. If the last, the term's default becomes zero and
   the alarms carry the job alone.
7. **Whether frame stacking buys anything.** The observation carries no motion, so `obs.frame_stack`
   is the knob that gives the policy a direction of travel. `k = 2` costs `S + hand_size` extra stem
   channels and no storage; the comparison is two runs to the same env-step count with everything else
   held, read off `policy/tile_top1_share` and the win rate against a fixed anchor.
8. **`n_epochs` and `batch_size`.** Both are marked measure-first and both have a stated rule:
   if epoch three's clip fraction is more than twice epoch one's, lower `n_epochs`; if the per-iteration
   KL sits below 0.003 lower `batch_size` and if it sits above 0.02 raise it.
7. **`decision_ms`.** 500 ms is the environment's default and the highest-leverage cost knob: 1000
   halves the cost and loses the timing granularity that decides Clash fights, 250 costs four times as
   much. Ablate it in the second run, not the first, because it is the parameter most likely to be
   blamed for a plateau that is really something else.
8. **How much of a batch can carry a gradient at all — measured, and it is 6%.** The first two
   real iterations on the laptop profile reported `policy/forced_noop_frac` at **0.937 and
   0.924** **[M]**, exactly equal to `policy/noop_rate` in both, with
   `policy/legal_actions_mean` at 29 of 2305. So in 93% of collected decisions the mask leaves
   exactly one action — the no-op — because the elixir bar cannot afford anything. Those rows
   are not a policy choosing to wait; they are a policy with nothing to choose, and they
   contribute exactly zero policy gradient while occupying a full row of the rectangle, a full
   share of the boundary's bandwidth and a full share of every epoch.

   It is visible in everything downstream and explains all of it: `ppo/kl` at 4.8e-06 then
   3.6e-07, `ppo/clip_fraction` at 1e-04 then exactly 0, `ppo/grad_norm_actor` at 0.003,
   `ppo/entropy_normalised` at 0.07 against a healthy band of 0.3-0.8 — an average over rows
   whose legal set has one member and whose entropy is therefore zero. A reader who saw only
   the KL would conclude the update was broken. The update is fine; the batch is 94% padding.

   **And the padding is where the wall clock goes.** The same iteration measured
   `time/collection` at 12.6 s against `time/update` at 531.1 s — **97.7% of the iteration is
   the update** **[M]**, three epochs over a batch that is 93% rows the policy cannot learn
   from. (The first iteration reads 84% because it pays for cuDNN's first look at each shape:
   `time/inference` 76.6 s then 10.2 s.)

   This is a measurement and not yet a decision, and the decision it feeds is the one to make
   deliberately: whether a forced row should be collected at all, whether it should be stored
   but excluded from the policy loss while still feeding the critic and the reward chain that
   GAE walks, or whether `decision_ms` should rise until a decision is usually a decision.
   The three differ on two axes and not one. On the value function: dropping a row removes it
   from the critic's targets as well as the policy's, and removes a link from the chain GAE
   walks backwards. On the wall clock: dropping at collection saves almost the whole update,
   because the rows never reach the trunk; excluding them from the policy loss alone saves
   nothing, because they still go through it; and raising `decision_ms` saves both while being
   the only one of the three that changes what the agent *is* rather than what the learner does
   with it. Choosing between the first two on value-function grounds alone would be choosing
   between "several times faster" and "no faster" without knowing it. That is why
   `forced_noop_frac` and the `time/` group are in the metric list rather than constants in the
   code.
9. **The trunk's discrimination, against its input's.** If the harness ever alarms on representation
   collapse — the encoder producing nearly the same embedding for boards that differ — the threshold
   must be **relative, never absolute**, and it must be built to three rules that a naive version of
   it breaks.

   *Relative, because the observation is mostly constant.* Across eight boards differing in one
   unit's position, a destroyed tower and the elixir, only 51 of the observation's numbers move —
   0.4% — and the greatest pairwise cosine is 0.9998 with nothing wrong **[M]**. The rest is static
   arena, standing towers and an unchanged hand. An embedding cosine of 0.99 on that input would mean
   the trunk was *increasing* discrimination, not losing it.

   *On the same states, not on a sample.* Because so few cells move, the input baseline is dominated
   by which boards were chosen: pairs differing only in elixir barely move it, a pair with a tower
   down moves it a great deal. Two people measuring one encoder against baselines drawn from
   different state sets will disagree about that encoder and both will be right about what they
   measured. The baseline is computed on the states the encoding was computed on, or the two numbers
   are not a pair. `royalegym.measure_variability(builder, states, action_masks)` returns the input
   side — cells, varying, fraction, raw cosine and varying-cell cosine — and excludes the action
   mask, which is legality rather than representation.

   *Report both cosines, raw and varying-cell.* Under a planted total collapse — the board removed
   from the observation entirely — the raw cosine moves by less than 0.01 while the varying-cell
   cosine goes to 1.0 **[M]**. The number a naive detector would watch is the number that does not
   move.

   One thing the measurement is not for: ranking two different observation builders against each
   other. It compares a representation against itself, and across representations of different
   sparsity it is not measuring the same property twice.
