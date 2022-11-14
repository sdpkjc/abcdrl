# abcdRL

**模块化单文件强化学习代码库🗄**

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://sdpkjc.coding.net/public/abcdrl/abcdrl/git/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

***使用文档 ➡️ [abcdrl.xyz](https://abcdrl.xyz)***

强化学习领域有许多高质量的代码库（🚂[PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL/), [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3), [thu-ml/tianshou](https://github.com/thu-ml/tianshou), ...），它们大多为了高代码复用率和可扩展性，采用*多文件、多层抽象、模块化*的设计。但这样的设计不利于研究者快速了解算法实现细节，且在进行改进时需要仔细阅读文档，去了解大量的接口信息。(🏃[vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl/), [tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL)) 是一类*单文件*强化学习代码库，它们提供了清晰极简的算法实现。但其代码采用基于过程的实现方式，使得改进代码时，无法快速定位和限制需要改进的代码范围。

> abcdRL 主要参考了 [CleanRL](https://github.com/vwxyzjn/cleanrl/) 的单文件设计哲学和 [PARL](https://github.com/PaddlePaddle/PARL/) 的模块设计。

**🚴abcdRL 提供“有但不多”的模块化设计，和清晰的单文件算法实现。我们希望在上述两种类型的代码库之间，做出更适合强化学习算法👨‍🎨研究者的平衡。**

*📖阅读代码时，在📄单文件代码中，快速了解算法的完整实现细节；🖌改进算法时，得益于🍃轻量的模块化设计，只需专注于少量的模块。*

## ✅ 已实现算法

- [DQN](https://doi.org/10.1038/nature14236)
- [DDPG](http://arxiv.org/abs/1509.02971)
- [TD3](http://arxiv.org/abs/1802.09477)
- [SAC](http://arxiv.org/abs/1801.01290)
- [PPO](http://arxiv.org/abs/1802.09477)

---

- [DDQN](http://arxiv.org/abs/1509.06461)
- [PDQN](http://arxiv.org/abs/1511.05952)

## ⛳ 目标

- 提供高可读性的代码
- 实现尽可能多的前沿算法
