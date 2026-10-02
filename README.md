# RoyaleLearn

<p align="center"><a href="https://github.com/RoyaleGym/RoyaleLearn/actions/workflows/suite.yml"><img alt="CI" src="https://github.com/RoyaleGym/RoyaleLearn/actions/workflows/suite.yml/badge.svg"></a> <img alt="License" src="https://img.shields.io/github/license/RoyaleGym/RoyaleLearn?style=flat-square&color=555"> <img alt="Python" src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white"> <a href="https://royalegym.github.io/RoyaleGym/"><img alt="Docs" src="https://img.shields.io/badge/docs-royalegym.github.io-8957e5?style=flat-square&logo=readthedocs&logoColor=white"></a> <a href="https://discord.gg/4D2BS5JBHP"><img alt="Discord" src="https://img.shields.io/discord/1551699576304705647?style=flat-square&logo=discord&logoColor=white&label=discord&color=5865F2"></a> <img alt="Last commit" src="https://img.shields.io/github/last-commit/RoyaleGym/RoyaleLearn?style=flat-square&color=555"></p>

Train a Clash Royale bot in Python, on your own machine: you write the reward, it learns to win.
It is the trainer for the battles RoyaleGym runs, and it uses your NVIDIA graphics card.

## Install

    pip install "royalegym[all]"

This is the `[learn]` part, for Python 3.12 or newer. Until it is on PyPI, see [Install](https://royalegym.github.io/RoyaleGym/install/)
for the exact line.

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

- The docs: [royalegym.github.io/RoyaleGym](https://royalegym.github.io/RoyaleGym/)
- Quick Start: [the first bot, step by step](https://royalegym.github.io/RoyaleGym/quickstart/)
- Every setting: [RoyaleLearn](https://royalegym.github.io/RoyaleGym/resources/royalelearn/). How it works inside (Advanced): [the guide](https://royalegym.github.io/RoyaleGym/repos/royalelearn/guide/)
- Questions: [Discord](https://discord.gg/4D2BS5JBHP)

MIT licensed. See [LICENSE](LICENSE).
