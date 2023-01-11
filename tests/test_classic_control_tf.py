from __future__ import annotations

import subprocess


def test_dqn_tf() -> None:
    subprocess.run(
        "python abcdrl/dqn_tf.py"
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
