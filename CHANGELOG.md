# Changelog

## 0.3.0

- `Learner` trains on the GPU by default: `device="auto"` is CUDA when torch can see a GPU, and
  otherwise the CPU, with one printed line saying so. On a CPU it uses half the machine's cores
  for the network (`threads=`).
- `gtaucap:<x>`, a decode mode for `ladder.opponent_mode` and `ladder.release_mode`: `gtau:<x>`, and
  it also plays whenever the seat's elixir is full.
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
