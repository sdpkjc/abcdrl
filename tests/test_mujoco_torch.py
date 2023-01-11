from __future__ import annotations

import subprocess


def test_ddpg() -> None:
    subprocess.run(
        "python abcdrl/ddpg_torch.py"
        + " --env-id Hopper-v4"
        + " --device auto"
        + " --num-envs 2"
        + " --learning-starts 64"
        + " --total-timesteps 256"
        + " --buffer-size 32"
        + " --batch-size 4",
        shell=True,
        check=True,
        timeout=300,
    )


def test_td3() -> None:
    subprocess.run(
        "python abcdrl/td3_torch.py"
        + " --env-id Hopper-v4"
        + " --device auto"
        + " --num-envs 2"
        + " --learning-starts 64"
        + " --total-timesteps 256"
        + " --buffer-size 32"
        + " --batch-size 4",
        shell=True,
        check=True,
        timeout=300,
    )


def test_sac() -> None:
    subprocess.run(
        "python abcdrl/sac_torch.py"
        + " --env-id Hopper-v4"
        + " --device auto"
        + " --num-envs 2"
        + " --learning-starts 64"
        + " --total-timesteps 256"
        + " --buffer-size 32"
        + " --batch-size 4",
        shell=True,
        check=True,
        timeout=300,
    )


def test_ppo() -> None:
    subprocess.run(
        "python abcdrl/ppo_torch.py"
        + " --env-id Hopper-v4"
        + " --device auto"
        + " --num-envs 2"
        + " --num-steps 64"
        + " --num-minibatches 16"
        + " --total-timesteps 256",
        shell=True,
        check=True,
        timeout=300,
    )
