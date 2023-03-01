# **abcdRL:adam:** (简单四步实现一个强化学习算法)

[![docs](https://img.shields.io/github/deployments/sdpkjc/abcdrl/Production?label=docs&logo=vercel)](https://docs.abcdrl.xyz/)
[![pypi](https://img.shields.io/pypi/v/abcdrl)](https://pypi.org/project/abcdrl)
[![docker autobuild](https://img.shields.io/docker/cloud/build/sdpkjc/abcdrl)](https://hub.docker.com/r/sdpkjc/abcdrl/)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)
[![benchmark](https://img.shields.io/badge/Weights%20&%20Biases-benchmark-FFBE00?logo=weightsandbiases)](https://report.abcdrl.xyz/)
[![mirror repo](https://img.shields.io/badge/Gitee-mirror%20repo-black?style=flat&labelColor=C71D23&logo=gitee)](https://gitee.com/sdpkjc/abcdrl/)

abcdRL 是一个**模块化单文件强化学习代码库**，提供“有但不严格”的模块化设计，和清晰的单文件算法实现。

<figure markdown>
  ![Adam](imgs/adam.svg#only-light){ width="300" }
  ![Adam](imgs/adam_white.svg#only-dark){ width="300" }
  <figcaption>Adam</figcaption>
</figure>

*阅读代码时，在单文件代码中，快速了解算法的完整实现细节；改进算法时，得益于轻量的模块化设计，只需专注于少量的模块。*

!!! quote "Ref"
    abcdRL 主要参考了 [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl/) 的单文件设计哲学和 [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL/) 的模块设计。

***路线图🗺️ [#57](https://github.com/sdpkjc/abcdrl/issues/57)***

## 🗽 设计哲学

- 要“拷贝📋”，~~不要“继承🧬”~~
- 要“单文件📜”，~~不要“多文件📚”~~
- 要“功能复用🛠”，~~不要“算法复用🖨”~~
- 要“一致的逻辑🤖”，~~不要“一致的接口🔌”~~

## ✅ 已实现算法

***Weights & Biases 性能报告 ➡️ [report.abcdrl.xyz](https://report.abcdrl.xyz)***

- [Deep Q Network (DQN)](https://doi.org/10.1038/nature14236) <sub>`dqn_torch.py`, `dqn_tf.py`, `dqn_atari_torch.py`, `dqn_atari_tf.py`</sub>
- [Deep Deterministic Policy Gradient (DDPG)](http://arxiv.org/abs/1509.02971) <sub>`ddpg_torch.py`</sub>
- [Twin Delayed Deep Deterministic Policy Gradient (TD3)](http://arxiv.org/abs/1802.09477) <sub>`td3_torch.py`</sub>
- [Soft Actor-Critic (SAC)](http://arxiv.org/abs/1801.01290) <sub>`sac_torch.py`</sub>
- [Proximal Policy Optimization (PPO)](http://arxiv.org/abs/1802.09477) <sub>`ppo_torch.py`</sub>

---

- [Double Deep Q Network (DDQN)](http://arxiv.org/abs/1509.06461) <sub>`ddqn_torch.py`, `ddqn_tf.py`</sub>
- [Prioritized Deep Q Network (PDQN)](http://arxiv.org/abs/1511.05952) <sub>`pdqn_torch.py`, `pdqn_tf.py`</sub>

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
