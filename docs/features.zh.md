# 特点 🤖

- 👨‍👩‍👧‍👦 统一的代码结构
- 📄 单文件实现
- 🐷 低代码复用
- 📐 最小化代码差异
- 📈 集成 Tensorboard & Wandb
- 🛤 符合 PEP8 & PEP526 规范

!!! note "📐 最小化代码差异"
    为了便于比较不同算法之间的差异和统一代码风格，我们的代码将按照下述的关系图，尽力做到连线的代码文件差异的最小化。
    ``` mermaid
    graph LR
    A[dqn_torch.py] -->B[ddpg_torch.py];
    B -->C[td3_torch.py];
    C -->D[sac_torch.py];
    B -->E[ppo_torch.py];
    A -->F[ddqn_torch.py];
    A -->G[pdqn_torch.py];
    A -->H[dqn_atari_torch.py];
    A -->I[dqn_tf.py];
    F -->J[ddqn_tf.py];
    G -->K[pdqn_tf.py];
    H -->L[dqn_atari_tf.py];
    ```
