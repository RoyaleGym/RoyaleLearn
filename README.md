# RoyaleLearn

[![suite](https://github.com/RoyaleGym/RoyaleLearn/actions/workflows/suite.yml/badge.svg)](https://github.com/RoyaleGym/RoyaleLearn/actions/workflows/suite.yml)

Train a Clash Royale bot in Python, on your own machine: you write the reward, it learns to win.
It is the trainer for the battles RoyaleGym runs, and it runs on a CPU.

## Install

    pip install "royalegym[all]"

This is the `[learn]` part, for Python 3.12. Until it is on PyPI, see the [guide's Install](docs/guide.md#install).

## Try it

```python
from royalegym import TowerHPReward, make_env
from royalelearn import Learner

def build_env():
    return make_env(reward=TowerHPReward())  # what the bot is paid for

if __name__ == "__main__":
    learner = Learner(build_env, save_dir="runs/my_bot")  # run again to carry on
    learner.learn(total_steps=20_000)
    learner.save("runs/my_bot/bot")
```

It prints one line per update. The Quick Start below also plays a battle with the bot to watch.

## Next

- Quick Start: [the first bot, step by step](https://github.com/RoyaleGym/RoyaleGym/blob/main/docs/site/pages/quick-start.md)
- Guides: [RoyaleGym's docs](https://github.com/RoyaleGym/RoyaleGym/tree/main/docs/site/pages/guides)
- Every setting, and how it works inside (Advanced): [docs/guide.md](docs/guide.md)
- Questions: [Discord](https://discord.gg/4D2BS5JBHP)

MIT licensed. See [LICENSE](LICENSE).
