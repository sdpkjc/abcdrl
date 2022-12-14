# Feature ๐ค

- ๐จโ๐ฉโ๐งโ๐ฆ Unified code structure
- ๐ Single-file implementation
- ๐ท Low code reuse
- ๐ Minimizing code differences
- ๐ Tensorboard & Wandb integration
- ๐ค PEP8(code style) & PEP526(type hint) compliant

!!! note "๐ Minimizing code differences"
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
