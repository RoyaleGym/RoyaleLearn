# Changelog

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

- `Learner` names its settings the way rlgym-ppo does, each with a default: the network
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
