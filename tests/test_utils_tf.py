from __future__ import annotations

import subprocess


def test_benchmark_tf() -> None:
    try:
        subprocess.run(
            "python benchmark.py --algs dqn --env-id CartPole-v1 --frameworks tf",
            shell=True,
            check=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        pass


def test_capture_video_tf() -> None:
    subprocess.run(
        "python abcdrl/dqn_tf.py"
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


def test_wandb_track_tf() -> None:
    online_flag = False
    re = subprocess.run("wandb status", shell=True, check=True, capture_output=True, timeout=10)
    if "disabled" not in str(re.stdout):
        subprocess.run("wandb offline", shell=True, check=True, timeout=10)
        online_flag = True

    try:
        subprocess.run(
            "python abcdrl/dqn_tf.py"
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
