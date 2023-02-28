from __future__ import annotations

import subprocess


def test_dqn_tf() -> None:
    subprocess.run(
        "python abcdrl/dqn_tf.py"
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


def test_ddqn_tf() -> None:
    subprocess.run(
        "python abcdrl/ddqn_tf.py"
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


def test_pdqn_tf() -> None:
    subprocess.run(
        "python abcdrl/pdqn_tf.py"
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
