from __future__ import annotations

import subprocess


def test_dqn_atari_torch() -> None:
    subprocess.run(
        "python abcdrl/dqn_atari_torch.py"
        + " --trainer.env-id BreakoutNoFrameskip-v4"
        + " --trainer.num-envs 2"
        + " --trainer.learning-starts 8"
        + " --trainer.total-timesteps 32"
        + " --trainer.buffer-size 10"
        + " --trainer.batch-size 4",
        shell=True,
        check=True,
        timeout=100,
    )
