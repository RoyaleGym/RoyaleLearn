# README media

Every file the README embeds. A *placeholder* is a generated SVG whose caption says what the
real image or screen recording must show; replace the file, keep the name.

| File | Kind | Shows / must show |
|---|---|---|
| `checkpoints.svg` | image placeholder | Reproducible checkpoints (planned) - A checkpoint's contents listed (policy, critic, optimizer, pool index, env config, seeds) and a resumed run's curve lying exactly on the original's |
| `family.svg` | diagram | the five repos and how they depend on each other; final |
| `frozen-pool-ladder.svg` | image placeholder | Frozen-pool ladder (planned) - Every pool snapshot's Elo with its confidence bar, the learner's current rating, and the win-rate gate line a snapshot must clear to enter the pool |
| `metrics-sink.svg` | image placeholder | Metrics from both sides (planned) - One Weights & Biases run: ticks/s, env-steps/s, episode length, crowns, illegal-action rate from the engine; loss, KL, entropy, Elo from the learner |
| `ppo-learner.png` | drawn figure | One masked head over 2305 actions - The mask of one decision drawn as one board per hand card, illegal tiles dark, and the card the elixir bar cannot pay for masked whole. Drawn by RoyaleGym's `docs/media/_shot_action_mask.py`; the step is searched for, not the first |
| `rollout-workers.svg` | image placeholder | Rollout workers (planned) - A throughput panel: N games stepping as one batch of 2N player slots, env-steps/s and engine ticks/s rising as workers are added |
| `self-play-env.png` | real still | already final |
| `training-run.svg` | video placeholder | A training run, watched live (planned) - The learner's Elo against the frozen pool with confidence bands, beside loss, entropy and env-steps/s, over one run from the first snapshot to the gate opening |
