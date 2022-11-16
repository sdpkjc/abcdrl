# **abcdRL** (简单四步实现一个强化学习算法)

abcdRL 是一个**模块化单文件强化学习代码库🗄**，提供“有但不严格🚥”的模块化🏗设计，和清晰的单文件📜算法实现。

*阅读📖代码时，在单文件📜代码中，快速了解算法的完整实现细节；改进🖌算法时，得益于轻量🍃的模块化设计，只需专注于少量的模块。*

!!! note
    abcdRL 主要参考了 [vwxyzjn/cleanRL](https://github.com/vwxyzjn/cleanrl/) 的单文件设计哲学和 [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL/) 的模块设计。

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
