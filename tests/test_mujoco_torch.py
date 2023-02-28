from __future__ import annotations

import subprocess


def test_ddpg_torch() -> None:
    subprocess.run(
        "python abcdrl/ddpg_torch.py"
        + " --trainer.env-id Hopper-v4"
        + " --trainer.num-envs 2"
        + " --trainer.learning-starts 64"
        + " --trainer.total-timesteps 256"
        + " --trainer.buffer-size 32"
        + " --trainer.batch-size 4",
        shell=True,
        check=True,
        timeout=300,
    )


def test_td3_torch() -> None:
    subprocess.run(
        "python abcdrl/td3_torch.py"
        + " --trainer.env-id Hopper-v4"
        + " --trainer.num-envs 2"
        + " --trainer.learning-starts 64"
        + " --trainer.total-timesteps 256"
        + " --trainer.buffer-size 32"
        + " --trainer.batch-size 4",
        shell=True,
        check=True,
        timeout=300,
    )


def test_sac_torch() -> None:
    subprocess.run(
        "python abcdrl/sac_torch.py"
        + " --trainer.env-id Hopper-v4"
        + " --trainer.num-envs 2"
        + " --trainer.learning-starts 64"
        + " --trainer.total-timesteps 256"
        + " --trainer.buffer-size 32"
        + " --trainer.batch-size 4",
        shell=True,
        check=True,
        timeout=300,
    )


def test_ppo_torch() -> None:
    subprocess.run(
        "python abcdrl/ppo_torch.py"
        + " --env-id Hopper-v4"
        + " --num-envs 2"
        + " --num-steps 64"
        + " --num-minibatches 16"
        + " --total-timesteps 256",
        shell=True,
        check=True,
        timeout=300,
    )
