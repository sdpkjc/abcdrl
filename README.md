# abcdRL

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://sdpkjc.coding.net/public/abcdrl/abcdrl/git/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

abcdRL is a **Modular single-file RL algorithms library🗄** that provides modular design without strict and clean single-file implementation.

*When 📖reading the code, understand the full implementation details of the algorithm in the 📜single file quickly; When 🖌modifying the algorithm, benefiting from a 🍃lightweight modular design, only need to focus on a small number of modules.*

***Documentation ➡️ [docs.abcdrl.xyz](https://abcdrl.xyz)***

!!! note
    abcdRL mainly references the single-file design philosophy of [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl/) and the module design of [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL/).

## 🐼 Features

- 👨‍👩‍👧‍👦 Unified code structure
- 📄 Single-file implementation
- 🐷 Low code reuse
- 📐 Minimizing code differences
- 📈 Tensorboard & Wandb support
- 🛤 PEP8(code style) & PEP526(type hint) compliant

## 🗽 Design Philosophy

- "Copy📋", not "Inheritance🧬"
- "Single-file📜", not "Multi-file📚"
- "Features reuse🛠", not "Algorithms reuse🖨"
- "Unified logic🤖", not "Unified interface🔌"

## ✅ Implemented Algorithms

- [Deep Q Network (DQN)](https://doi.org/10.1038/nature14236)
- [Deep Deterministic Policy Gradient (DDPG)](http://arxiv.org/abs/1509.02971)
- [Twin Delayed Deep Deterministic Policy Gradient (TD3)](http://arxiv.org/abs/1802.09477)
- [Soft Actor-Critic (SAC)](http://arxiv.org/abs/1801.01290)
- [Proximal Policy Optimization (PPO)](http://arxiv.org/abs/1802.09477)

---

- [Double Deep Q Network (DDQN)](http://arxiv.org/abs/1509.06461)
- [Prioritized Deep Q Network (PDQN)](http://arxiv.org/abs/1511.05952)
