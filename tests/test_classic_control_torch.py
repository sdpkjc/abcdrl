from __future__ import annotations

import subprocess


def test_dqn_torch() -> None:
    subprocess.run(
        "python abcdrl/dqn_torch.py"
        + " --trainer.env-id CartPole-v1"
        + " --trainer.num-envs 2"
        + " --trainer.learning-starts 8"
        + " --trainer.total-timesteps 32"
        + " --trainer.buffer-size 10"
        + " --trainer.batch-size 4",
        shell=True,
        check=True,
        timeout=100,
    )


def test_ddqn_torch() -> None:
    subprocess.run(
        "python abcdrl/ddqn_torch.py"
        + " --env-id CartPole-v1"
        + " --num-envs 2"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4",
        shell=True,
        check=True,
        timeout=100,
    )


def test_pdqn_torch() -> None:
    subprocess.run(
        "python abcdrl/pdqn_torch.py"
        + " --env-id CartPole-v1"
        + " --num-envs 2"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4",
        shell=True,
        check=True,
        timeout=100,
    )
