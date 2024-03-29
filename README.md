# **abcdRL** (Implement a RL algorithm in four simple steps)

English | [简体中文](./README.cn.md)

[![license](https://img.shields.io/pypi/l/abcdrl)](https://github.com/sdpkjc/abcdrl)
[![pytest](https://github.com/sdpkjc/abcdrl/actions/workflows/test.yml/badge.svg)](https://github.com/sdpkjc/abcdrl/actions/workflows/test.yml)
[![pre-commit](https://github.com/sdpkjc/abcdrl/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/sdpkjc/abcdrl/actions/workflows/pre-commit.yml)
[![pypi](https://img.shields.io/pypi/v/abcdrl)](https://pypi.org/project/abcdrl)
[![docker autobuild](https://img.shields.io/docker/cloud/build/sdpkjc/abcdrl)](https://hub.docker.com/r/sdpkjc/abcdrl/)
[![docs](https://img.shields.io/github/deployments/sdpkjc/abcdrl/Production?label=docs&logo=vercel)](https://docs.abcdrl.xyz/)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)
[![benchmark](https://img.shields.io/badge/Weights%20&%20Biases-benchmark-FFBE00?logo=weightsandbiases)](https://report.abcdrl.xyz/)
[![mirror repo](https://img.shields.io/badge/Gitee-mirror%20repo-black?style=flat&labelColor=C71D23&logo=gitee)](https://gitee.com/sdpkjc/abcdrl/)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![python versions](https://img.shields.io/pypi/pyversions/abcdrl)](https://pypi.org/project/abcdrl)

abcdRL is a **Modular Single-file Reinforcement Learning Algorithms Library** that provides modular design without strict and clean single-file implementation.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://docs.abcdrl.xyz/imgs/adam.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://docs.abcdrl.xyz/imgs/adam_dark.svg">
  <img alt="adam" src="https://docs.abcdrl.xyz/imgs/adam.svg" width="300">
</picture>

*Understand the full implementation details of the algorithm in a single file quickly when reading the code;  Benefit from a lightweight modular design, only need to focus on a small number of modules when modifying the algorithm.*

> abcdRL mainly references the single-file design philosophy of [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl/) and the module design of [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL/).

***Documentation ➡️ [docs.abcdrl.xyz](https://docs.abcdrl.xyz/)***

***Roadmap🗺️ [#57](https://github.com/sdpkjc/abcdrl/issues/57)***

## 🚀 Quickstart

Open the project in Gitpod🌐 and start coding immediately.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)

Using Docker📦:

```shell
# 0. Prerequisites: Docker & Nvidia Drive & NVIDIA Container Toolkit
# 1. Run DQN algorithm
docker run --rm --gpus all sdpkjc/abcdrl python abcdrl/dqn_torch.py
```

***[For detailed installation instructions 👀](https://docs.abcdrl.xyz/install/)***

## 🐼 Features

- 👨‍👩‍👧‍👦 Unified code structure
- 📄 Single-file implementation
- 🐷 Low code reuse
- 📐 Minimizing code differences
- 📈 Tensorboard & Wandb integration
- 🛤 PEP8(code style) & PEP526(type hint) compliant

## 🗽 Design Philosophy

- "Copy📋", ~~not "Inheritance🧬"~~
- "Single-file📜", ~~not "Multi-file📚"~~
- "Features reuse🛠", ~~not "Algorithms reuse🖨"~~
- "Unified logic🤖", ~~not "Unified interface🔌"~~

## ✅ Implemented Algorithms

***Weights & Biases Benchmark Report ➡️ [report.abcdrl.xyz](https://report.abcdrl.xyz)***

- [Deep Q Network (DQN)](https://doi.org/10.1038/nature14236) <sub>`dqn_torch.py`, `dqn_tf.py`, `dqn_atari_torch.py`, `dqn_atari_tf.py`</sub>
- [Deep Deterministic Policy Gradient (DDPG)](http://arxiv.org/abs/1509.02971) <sub>`ddpg_torch.py`</sub>
- [Twin Delayed Deep Deterministic Policy Gradient (TD3)](http://arxiv.org/abs/1802.09477) <sub>`td3_torch.py`</sub>
- [Soft Actor-Critic (SAC)](http://arxiv.org/abs/1801.01290) <sub>`sac_torch.py`</sub>
- [Proximal Policy Optimization (PPO)](http://arxiv.org/abs/1802.09477) <sub>`ppo_torch.py`</sub>

---

- [Double Deep Q Network (DDQN)](http://arxiv.org/abs/1509.06461) <sub>`ddqn_torch.py`, `ddqn_tf.py`</sub>
- [Prioritized Deep Q Network (PDQN)](http://arxiv.org/abs/1511.05952) <sub>`pdqn_torch.py`, `pdqn_tf.py`</sub>

## Citing abcdRL

```bibtex
@misc{zhao_abcdrl_2022,
    author = {Yanxiao, Zhao},
    month = {12},
    title = {{abcdRL: Modular Single-file Reinforcement Learning Algorithms Library}},
    url = {https://github.com/sdpkjc/abcdrl},
    year = {2022}
}
```
