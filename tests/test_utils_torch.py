from __future__ import annotations

import subprocess


def test_benchmark_torch() -> None:
    try:
        subprocess.run(
            "python benchmark.py --algs dqn --env-ids CartPole-v1 --frameworks torch",
            shell=True,
            check=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        pass


def test_capture_video_torch() -> None:
    subprocess.run(
        "python abcdrl/dqn_torch.py"
        + " --trainer.env-id CartPole-v1"
        + " --trainer.num-envs 2"
        + " --trainer.learning-starts 8"
        + " --trainer.total-timesteps 32"
        + " --trainer.buffer-size 10"
        + " --trainer.batch-size 4"
        + " --trainer.capture-video",
        shell=True,
        check=True,
        timeout=100,
    )


def test_wandb_track_torch() -> None:
    online_flag = False
    re = subprocess.run("wandb status", shell=True, check=True, capture_output=True, timeout=10)
    if "disabled" not in str(re.stdout):
        subprocess.run("wandb offline", shell=True, check=True, timeout=10)
        online_flag = True

    try:
        subprocess.run(
            "python abcdrl/dqn_torch.py"
            + " --trainer.env-id CartPole-v1"
            + " --trainer.num-envs 2"
            + " --trainer.learning-starts 8"
            + " --trainer.total-timesteps 32"
            + " --trainer.buffer-size 10"
            + " --trainer.batch-size 4"
            + " --trainer.capture-video"
            + " --logger.track",
            shell=True,
            check=True,
            timeout=100,
        )
    finally:
        if online_flag:
            subprocess.run("wandb online", shell=True, check=True, timeout=10)
