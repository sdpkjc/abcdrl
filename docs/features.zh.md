# 特点 🤖

[<img src="https://img.shields.io/badge/license-MIT-green">](https://github.com/sdpkjc/abcdrl)
[![pytest](https://github.com/sdpkjc/abcdrl/actions/workflows/test.yml/badge.svg)](https://github.com/sdpkjc/abcdrl/actions/workflows/test.yml)
[![pre-commit](https://github.com/sdpkjc/abcdrl/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/sdpkjc/abcdrl/actions/workflows/pre-commit.yml)
[![docs](https://img.shields.io/github/deployments/sdpkjc/abcdrl/Production?label=docs&logo=vercel)](https://docs.abcdrl.xyz/)
[![benchmark](https://img.shields.io/badge/Weights%20&%20Biases-benchmark-FFBE00?logo=weightsandbiases)](https://benchmark.abcdrl.xyz/)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![python versions](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)

- 👨‍👩‍👧‍👦 统一的代码结构
- 📄 单文件实现
- 🐷 低代码复用
- 📐 最小化代码差异
- 📈 Tensorboard & Wandb 支持
- 🛤 符合 PEP8 & PEP526 规范

!!! note "📐 最小化代码差异"
    为了便于比较不同算法之间的差异和统一代码风格，我们的代码将按照下述的关系图，尽力做到连线的代码文件差异的最小化。
    ``` mermaid
    graph LR
    A[dqn.py] -->B[ddpg.py];
    B -->C[td3.py];
    C -->D[sac.py];
    B -->E[ppo.py];
    A -->F[ddqn.py];
    A -->G[pdqn.py];
    ```
