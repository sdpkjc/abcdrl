from __future__ import annotations

import subprocess


def test_dqn_atari_torch() -> None:
    subprocess.run(
        "python abcdrl/dqn_atari_torch.py"
        + " --env-id BreakoutNoFrameskip-v4"
        + " --num-envs 2"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4",
        shell=True,
        check=True,
        timeout=100,
    )
