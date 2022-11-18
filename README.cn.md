# **abcdRL** (简单四步实现一个强化学习算法)

[English](./README.md) | 简体中文

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://sdpkjc.coding.net/public/abcdrl/abcdrl/git/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.abcdrl.xyz)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://docs.abcdrl.xyz/zh)

abcdRL 是一个**模块化单文件强化学习代码库🗄**，提供“有但不严格🚥”的模块化设计🏗，和清晰的单文件算法实现📜。

<img src="docs/imgs/adam.svg" width="300"/>

*📖阅读代码时，在📜单文件代码中，快速了解算法的完整实现细节；🖌改进算法时，得益于🍃轻量的模块化设计，只需专注于少量的模块。*

> abcdRL 主要参考了 [vwxyzjn/cleanRL](https://github.com/vwxyzjn/cleanrl/) 的单文件设计哲学和 [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL/) 的模块设计。

***使用文档 ➡️ [docs.abcdrl.xyz](https://docs.abcdrl.xyz)***

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)

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

- [Deep Q Network (DQN)](https://doi.org/10.1038/nature14236)
- [Deep Deterministic Policy Gradient (DDPG)](http://arxiv.org/abs/1509.02971)
- [Twin Delayed Deep Deterministic Policy Gradient (TD3)](http://arxiv.org/abs/1802.09477)
- [Soft Actor-Critic (SAC)](http://arxiv.org/abs/1801.01290)
- [Proximal Policy Optimization (PPO)](http://arxiv.org/abs/1802.09477)

---

- [Double Deep Q Network (DDQN)](http://arxiv.org/abs/1509.06461)
- [Prioritized Deep Q Network (PDQN)](http://arxiv.org/abs/1511.05952)
