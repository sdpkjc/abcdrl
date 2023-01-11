from __future__ import annotations

import subprocess


def test_benchmark() -> None:
    try:
        subprocess.run(
            "python benchmark.py",
            shell=True,
            check=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        pass


def test_capture_video() -> None:
    subprocess.run(
        "python abcdrl/dqn_torch.py"
        + " --env-id CartPole-v1"
        + " --num-envs 2"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4"
        + " --capture-video True",
        shell=True,
        check=True,
        timeout=100,
    )


def test_wandb_track() -> None:
    online_flag = False
    re = subprocess.run("wandb status", shell=True, check=True, capture_output=True, timeout=10)
    if "disabled" not in str(re.stdout):
        subprocess.run("wandb offline", shell=True, check=True, timeout=10)
        online_flag = True

    try:
        subprocess.run(
            "python abcdrl/dqn_torch.py"
            + " --env-id CartPole-v1"
            + " --num-envs 2"
            + " --learning-starts 8"
            + " --total-timesteps 32"
            + " --buffer-size 10"
            + " --batch-size 4"
            + " --capture-video True",
            shell=True,
            check=True,
            timeout=100,
        )
    finally:
        if online_flag:
            subprocess.run("wandb online", shell=True, check=True, timeout=10)
