# Feature 🤖

[![license](https://img.shields.io/badge/license-MIT-green)](https://github.com/sdpkjc/abcdrl)
[![pytest](https://github.com/sdpkjc/abcdrl/actions/workflows/test.yml/badge.svg)](https://github.com/sdpkjc/abcdrl/actions/workflows/test.yml)
[![pre-commit](https://github.com/sdpkjc/abcdrl/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/sdpkjc/abcdrl/actions/workflows/pre-commit.yml)
[![docker autobuild](https://img.shields.io/docker/cloud/build/sdpkjc/abcdrl)](https://hub.docker.com/r/sdpkjc/abcdrl/)
[![docs](https://img.shields.io/github/deployments/sdpkjc/abcdrl/Production?label=docs&logo=vercel)](https://docs.abcdrl.xyz/)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)
[![benchmark](https://img.shields.io/badge/Weights%20&%20Biases-benchmark-FFBE00?logo=weightsandbiases)](https://report.abcdrl.xyz/)
[![mirror repo](https://img.shields.io/badge/Gitee-mirror%20repo-black?style=flat&labelColor=C71D23&logo=gitee)](https://gitee.com/sdpkjc/abcdrl/)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![python versions](https://img.shields.io/badge/python-3.8|3.9|3.10-blue)](https://www.python.org/)

- 👨‍👩‍👧‍👦 Unified code structure
- 📄 Single-file implementation
- 🐷 Low code reuse
- 📐 Minimizing code differences
- 📈 Tensorboard & Wandb support
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
    ```
