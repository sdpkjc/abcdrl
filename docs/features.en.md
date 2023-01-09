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
    A[dqn.py] -->B[ddpg.py];
    B -->C[td3.py];
    C -->D[sac.py];
    B -->E[ppo.py];
    A -->F[ddqn.py];
    A -->G[pdqn.py];
    A -->H[dqn_atari.py];
    ```
