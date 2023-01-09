# **abcdRL:adam:** (Implement a RL algorithm in four simple steps)

[![docs](https://img.shields.io/github/deployments/sdpkjc/abcdrl/Production?label=docs&logo=vercel)](https://docs.abcdrl.xyz/)
[![docker autobuild](https://img.shields.io/docker/cloud/build/sdpkjc/abcdrl)](https://hub.docker.com/r/sdpkjc/abcdrl/)
[![pypi](https://img.shields.io/pypi/v/abcdrl)](https://pypi.org/project/abcdrl)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)
[![benchmark](https://img.shields.io/badge/Weights%20&%20Biases-benchmark-FFBE00?logo=weightsandbiases)](https://report.abcdrl.xyz/)
[![mirror repo](https://img.shields.io/badge/Gitee-mirror%20repo-black?style=flat&labelColor=C71D23&logo=gitee)](https://gitee.com/sdpkjc/abcdrl/)

abcdRL is a **Modular Single-file Reinforcement Learning Algorithms Library** that provides modular design without strict and clean single-file implementation.

<figure markdown>
  ![Adam](imgs/adam.svg){ width="300" }
  <figcaption>Adam</figcaption>
</figure>

*Understand the full implementation details of the algorithm in a single file quickly when reading the code;  Benefit from a lightweight modular design, only need to focus on a small number of modules when modifying the algorithm.*

!!! quote "Ref"
    abcdRL mainly references the single-file design philosophy of [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl/) and the module design of [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL/).

***Roadmap🗺️ [#57](https://github.com/sdpkjc/abcdrl/issues/57)***

## 🗽 Design Philosophy

- "Copy📋", ~~not "Inheritance🧬"~~
- "Single-file📜", ~~not "Multi-file📚"~~
- "Features reuse🛠", ~~not "Algorithms reuse🖨"~~
- "Unified logic🤖", ~~not "Unified interface🔌"~~

## ✅ Implemented Algorithms

***Weights & Biases Benchmark Report ➡️ [report.abcdrl.xyz](https://report.abcdrl.xyz)***

- [Deep Q Network (DQN)](https://doi.org/10.1038/nature14236)
- [Deep Deterministic Policy Gradient (DDPG)](http://arxiv.org/abs/1509.02971)
- [Twin Delayed Deep Deterministic Policy Gradient (TD3)](http://arxiv.org/abs/1802.09477)
- [Soft Actor-Critic (SAC)](http://arxiv.org/abs/1801.01290)
- [Proximal Policy Optimization (PPO)](http://arxiv.org/abs/1802.09477)

---

- [Double Deep Q Network (DDQN)](http://arxiv.org/abs/1509.06461)
- [Prioritized Deep Q Network (PDQN)](http://arxiv.org/abs/1511.05952)

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
