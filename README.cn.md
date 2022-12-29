# **abcdRL** (简单四步实现一个强化学习算法)

[English](./README.md) | 简体中文

[![license](https://img.shields.io/badge/license-MIT-green)](https://github.com/sdpkjc/abcdrl)
[![pytest](https://github.com/sdpkjc/abcdrl/actions/workflows/test.yml/badge.svg)](https://github.com/sdpkjc/abcdrl/actions/workflows/test.yml)
[![pre-commit](https://github.com/sdpkjc/abcdrl/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/sdpkjc/abcdrl/actions/workflows/pre-commit.yml)
[![docker autobuild](https://img.shields.io/docker/cloud/build/sdpkjc/abcdrl)](https://hub.docker.com/r/sdpkjc/abcdrl/)
[![docs](https://img.shields.io/github/deployments/sdpkjc/abcdrl/Production?label=docs&logo=vercel)](https://docs.abcdrl.xyz/)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)
[![benchmark](https://img.shields.io/badge/Weights%20&%20Biases-benchmark-FFBE00?logo=weightsandbiases)](https://report.abcdrl.xyz/)
[![mirror repo](https://img.shields.io/badge/Gitee-mirror%20repo-black?style=flat&labelColor=C71D23&logo=gitee)](https://gitee.com/sdpkjc/abcdrl/)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![python versions](https://img.shields.io/badge/python-3.8|3.9|3.10-blue)](https://www.python.org/)

abcdRL 是一个**模块化单文件强化学习代码库**，提供“有但不严格”的模块化设计，和清晰的单文件算法实现。

<img src="https://abcdrl.xyz/logo/adam.svg" width="300"/>

*阅读代码时，在单文件代码中，快速了解算法的完整实现细节；改进算法时，得益于轻量的模块化设计，只需专注于少量的模块。*

> abcdRL 主要参考了 [vwxyzjn/cleanRL](https://github.com/vwxyzjn/cleanrl/) 的单文件设计哲学和 [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL/) 的模块设计。

***使用文档 ➡️ [docs.abcdrl.xyz](https://docs.abcdrl.xyz/zh/)***

***路线图🗺️ [#57](https://github.com/sdpkjc/abcdrl/issues/57)***

## 🚀 快速开始

在 Gitpod🌐 中打开项目，并立即开始编码。

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)

使用 Docker📦：

```bash
# 0. 安装 Docker & Nvidia Drive & NVIDIA Container Toolkit
# 1. 运行 DQN 算法
docker run --rm --gpus all sdpkjc/abcdrl python abcdrl/dqn.py
```

***[详细安装说明 👀](https://docs.abcdrl.xyz/zh/install/)***

## 🐼 特点

- 👨‍👩‍👧‍👦 统一的代码结构
- 📄 单文件实现
- 🐷 低代码复用
- 📐 最小化代码差异
- 📈 Tensorboard & Wandb 支持
- 🛤 符合 PEP8 & PEP526 规范

## 🗽 设计哲学

- 要“拷贝📋”，~~不要“继承🧬”~~
- 要“单文件📜”，~~不要“多文件📚”~~
- 要“功能复用🛠”，~~不要“算法复用🖨”~~
- 要“一致的逻辑🤖”，~~不要“一致的接口🔌”~~

## ✅ 已实现算法

***Weights & Biases 性能报告 ➡️ [report.abcdrl.xyz](https://report.abcdrl.xyz)***

- [Deep Q Network (DQN)](https://doi.org/10.1038/nature14236)
- [Deep Deterministic Policy Gradient (DDPG)](http://arxiv.org/abs/1509.02971)
- [Twin Delayed Deep Deterministic Policy Gradient (TD3)](http://arxiv.org/abs/1802.09477)
- [Soft Actor-Critic (SAC)](http://arxiv.org/abs/1801.01290)
- [Proximal Policy Optimization (PPO)](http://arxiv.org/abs/1802.09477)

---

- [Double Deep Q Network (DDQN)](http://arxiv.org/abs/1509.06461)
- [Prioritized Deep Q Network (PDQN)](http://arxiv.org/abs/1511.05952)

## 引用 abcdRL

```bibtex
@misc{zhao_abcdrl_2022,
    author = {Yanxiao, Zhao},
    month = {12},
    title = {{abcdRL: Modular Single-file Reinforcement Learning Algorithms Library}},
    url = {https://github.com/sdpkjc/abcdrl},
    year = {2022}
}
```
