# 特点 🤖

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://sdpkjc.coding.net/public/abcdrl/abcdrl/git/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.abcdrl.xyz)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://docs.abcdrl.xyz/zh)

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
