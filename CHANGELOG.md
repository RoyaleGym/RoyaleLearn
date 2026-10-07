# Changelog

## 0.5.11

- AMD cards on torch's ROCm build, which shows them as "cuda" devices: a run's identity names
  the card by its AMD architecture (`rocm:<name>:gfx1100`); the throughput tier leaves cuDNN's
  benchmark search off there; the start-up memory check and the memory health fields carry on
  when the driver cannot say how much memory is free; and the GPU utilisation hint names AMD's
  package. When the card refuses the GPU memory cap (reported for AMD's APUs), the run goes on
  without one.
- A card reached through ZLUDA is named as such and noted at start-up, and cuDNN is switched off
  there. It is refused on a torch build without PTX, as torch's builds for CUDA 12 and later are,
  with a `net.autocast_dtype` other than float32, and under `determinism.tier` run_exact.
- The looser float32 ratio tolerance applies only on cards that have TF32 (NVIDIA Ampere and
  newer); elsewhere float32 is checked at float32's tolerance.
- `net.button_head = "card"`: each ability button is scored from its card's embedding (the
  hand slots' table), its own status and its ready bit beside the pooled board, by one scorer
  shared by every button, so which position a card's button sits in does not change its logit.
  It reads RoyaleGym's per-button fields (`SpatialObsBuilder(button_index=True)`), which then
  reach only the button scorer. Unset, the head and every digest are as before.

## 0.5.10

- The start-up memory projection counts a process worker's fixed cost as 200 MB, measured on
  Windows; it counted 40. A run whose rollout workers came up with torch imported is warned at
  start-up, since that costs some 850 MB a worker the projection cannot see.
- `net.hand_slot_features`: observation vector fields with one value per hand slot (for example
  `own_hand_evolved`) join that slot's card embedding where the policy head builds the slot's
  query. Unset, the network, its weights and its architecture digest are as before.
- `unit_ids`, each tile's own unit type (RoyaleGym's `SpatialObsBuilder(unit_identity=True)`),
  is stored exactly in a codec region of its own (one byte a value, two once the unit list passes
  255 types), carried in `ObsBatch.unit_ids` and embedded in a table of its own beside the card
  ids. Without it, every row, table and digest is as before.

## 0.5.9

- `doctor.vram_fraction` sets the GPU memory cap from a config: a share of the card in (0, 1],
  or null for no cap. Unset, the constructor's `vram_fraction` decides, as before.
- The ratio check's tolerance follows the precision the forwards ran at: float32 on a GPU allowed
  TF32 (the default outside `determinism.tier` run_exact) is checked at 5e-3, or at
  `ppo.ratio_atol["tf32"]` when a config names it, instead of float32's 1e-4.
- When the start-up memory probe runs out of memory under the cap, the message says so and names
  what to change.

## 0.5.8

- A CUDA run on Windows starts again: the memory cap named the device as `cuda` with no index,
  which torch refuses, so every such run stopped at start-up since 0.5.5. It now names the
  device's index, or the current device's.

## 0.5.7

- `rollout.overlap` runs: the next iteration is collected on a second thread while the update
  trains on a copy of this one. Its learner seats sample from a snapshot of the actor taken just
  before the update, so each batch after the first is one update behind the learner it trains,
  and the ratio check is made against that snapshot. Rows carry
  `ppo/behaviour_lag_iterations` and `time/overlap_saved`. It needs a second rectangle of memory,
  is off by default, and is refused under `determinism.tier` run_exact.

## 0.5.6

- `net.trunk_stride`, an opt-in stride for the residual trunk (1 or 2, default 1). At 2 a
  strided convolution after the stem runs the residual blocks at half the board's resolution in
  each direction, and their output is upsampled and added to the stem's, so the heads still read
  one feature per tile. At 1 nothing changes: the network, its weights and its architecture
  digest are as before.

## 0.5.5

- On Windows a GPU run caps its memory at 80% of the card by default. Windows backs a full card
  with system RAM instead of failing, so without a cap torch's memory cache could take gigabytes
  of it over a long run. Set it with `Learner(vram_fraction=...)`: a share of the card, or None
  for no cap. Elsewhere there is no cap unless you set one. The start-up check that one
  minibatch fits the card counts the cap.

## 0.5.4

- The spells' ids reach the network. RoyaleGym's `SpatialObsBuilder(spell_identity=True)` adds a
  `spell_ids` observation (four planes in the `card_ids` vocabulary). It is stored exactly beside
  `card_ids`, carried in `ObsBatch.card_ids` after the card planes, and read through the same card
  embedding into the trunk. It needs `card_identity`; a `spell_ids` without `card_ids`, or in
  another vocabulary, is refused at start-up. Without it, every row, table and digest is as before.

## 0.5.3

- The start-up check that plays a battle of no-ops to its end allows half a match past the
  clock, so it passes on an engine whose level overtime ends after the clock (RoyaleSim's tower
  drain).

## 0.5.2

- A run that uses an add-on installed from a wheel (RoyaleImitate's `warm_start` or `imitation`,
  installed with `pip install "royalegym[all]"`) starts. The run's identity names the add-on by
  the content of its files, `content:<sha256>`, where there is no commit to name; it used to
  refuse the run.

## 0.5.1

- `royalelearn.extensions.write_policy_record(folder, env_spec, net)` writes the `policy.json` that
  lets `Learner.load_policy` load an actor folder another tool wrote, such as a clone.

## 0.5.0

- `ladder.opponent_mode` and `ladder.release_mode` take `stochastic`, `argmax`, or a decode of your
  own: `plugin:<module>:<function>`, called with a batch's masked log-probabilities, mask and
  observation vector, returning one legal action per row. No other decode is built in, and a
  config that names another is refused.
- `royalelearn.extensions` adds `LearningCoordinator`, `SCRIPTED_NAMES` and `build_opponent`, for
  add-ons such as RoyaleImitate's recording and cloning.

## 0.4.3

- A run carried on from `save_dir` with another environment from `build_env` (the reward, the
  decks, the observation, the actions, when a battle ends) says so in one line, naming what
  changed. The run's folder keeps `environment.json` for it.

## 0.4.2

- Runs on macOS: shared-memory segment names are at most 30 characters (macOS refuses longer
  ones), and a longer name is refused on every platform.

## 0.4.1

- A `Learner` made again on the same `save_dir` with other training settings (learning rates,
  epochs, batch sizes, the entropy bonus, the discount) carries the run on and prints what
  changed. Another network, environment or opponent setup is refused, saying so in words.
- Ctrl-C during `learner.learn()` stops after the current update, writes a checkpoint and returns,
  so the script carries on (to `learner.save`, say). A second Ctrl-C stops at once.
- `Learner.load_policy` loads a run's folder (its newest checkpoint) or one checkpoint, as well as
  a folder `learner.save` wrote.

## 0.4.0

- `Learner` takes its settings by name, each with a default: the network
  (`trunk_channels`, `trunk_blocks`, `critic_hidden`), the update (`ppo_epochs`, `ppo_batch_size`,
  `ppo_minibatch_size`, `policy_lr`, `critic_lr`, `ppo_ent_coef`, `ppo_clip_range`, `gae_gamma`,
  `gae_lambda`, `standardize_returns`), `timestep_limit`, `n_checkpoints_to_keep`, and
  `log_to_wandb` with `wandb_project_name`, `wandb_group_name` and `wandb_run_name`. The defaults
  build the same run as before.
- `learner.learn()` with no `total_steps` stops at `timestep_limit`.
- A `build_env` in code run by `exec` (a docs checker, some IDE runners) is found by the run.

## 0.3.0

- `Learner` trains on the GPU by default: `device="auto"` is CUDA when torch can see a GPU, and
  otherwise the CPU, with one printed line saying so. On a CPU it uses half the machine's cores
  for the network (`threads=`).
- `net.policy_head = "factored"`: the policy as three stages (wait or act, then which card or button,
  then which tile) over the same action space, with `net.factored_act_init` for the gate's start.
- `ppo.entropy_coef_stages`: one entropy coefficient for each of those three stages.
- `SeedSnapshot.weight`: how often the pool draws a seed, as a multiplier on its draw weight.
- `Learner(..., extensions={...})` sets an add-on's config sections, such as RoyaleImitate's
  `warm_start`, and `learner.run` keeps the finished run for an add-on to read.

## 0.2.0

- `Learner(build_env).learn(total_steps=...)`: train a bot in one call, with defaults that run on a
  CPU, one progress line per update, and checkpoints that carry a run on from the same `save_dir`.
  `Learner.load_policy(path)` loads a saved bot that plays one seat.
- Ability buttons (a hero's or a champion's) are actions the network can choose.
- `ladder.seat_decks` trains one deck where the learner sits, against a field of others.
- `ladder.opponent_mode` sets how a frozen opponent picks its moves in training battles.

## 0.1.0

- The first release: the PPO learner, the opponent ladder, checkpoints and resume.
