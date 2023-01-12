# Feature 🤖

- 👨‍👩‍👧‍👦 Unified code structure
- 📄 Single-file implementation
- 🐷 Low code reuse
- 📐 Minimizing code differences
- 📈 Tensorboard & Wandb integration
- 🛤 PEP8(code style) & PEP526(type hint) compliant

!!! note "📐 Minimizing code differences"
    In order to facilitate the comparison of the differences between algorithms and to unify the code style, the code will try to minimize the differences between the wired code files as shown in the diagram below.
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
