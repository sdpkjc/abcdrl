# abcdRL

abcdRL is a **Modular single-file RL algorithms library🗄** that provides modular design without strict and clean single-file implementation.

*When 📖reading the code, understand the full implementation details of the algorithm in the 📜single file quickly; When 🖌modifying the algorithm, benefiting from a 🍃lightweight modular design, only need to focus on a small number of modules.*

!!! note
    abcdRL mainly references the single-file design philosophy of [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl/) and the module design of [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL/).

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
