# Feature

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://sdpkjc.coding.net/public/abcdrl/abcdrl/git/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

- 👨‍👩‍👧‍👦 Unified code structure
- 📄 Single-file implementation
- 🐷 Low code reuse
- 📐 Minimizing code differences
- 📈 Tensorboard & Wandb support
- 🛤 PEP8(code style) & PEP526(type hint) compliant

!!! note "📐 Minimizing code differences"
    In order to facilitate the comparison of the differences between different algorithms and to unify the code style, our code will try to minimize the differences between the wired code files as shown in the diagram below.
    ``` mermaid
    graph LR
    A[dqn.py] -->B[ddpg.py];
    B -->C[td3.py];
    C -->D[sac.py];
    B -->E[ppo.py];
    A -->F[ddqn.py];
    A -->G[pdqn.py];
    ```
