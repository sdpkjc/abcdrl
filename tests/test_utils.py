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


def test_example_all_wrapper() -> None:
    subprocess.run(
        "python abcdrl_copy_from/dqn_all_wrappers.py"
        + " --env-id CartPole-v1"
        + " --device auto"
        + " --num-envs 2"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4"
        + " --eval-frequency 5"
        + " --num-steps-eval 1"
        + " --save-frequency 16",
        shell=True,
        check=True,
        timeout=100,
    )


def test_example_eval_model() -> None:
    subprocess.run(
        "python abcdrl_copy_from/dqn_all_wrappers.py"
        + " --exp_name test_eval_dqn"
        + " --env-id CartPole-v1"
        + " --device auto"
        + " --num-envs 1"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 64"
        + " --batch-size 4"
        + " --save-frequency 16",
        shell=True,
        check=True,
        timeout=100,
    )

    subprocess.run(
        "python example_eval_model.py" + " --model-path models/test_eval_dqn/s16.agent" + " --total_timesteps 100",
        shell=True,
        check=True,
        timeout=100,
    )


def test_capture_video() -> None:
    subprocess.run(
        "python abcdrl/dqn.py"
        + " --env-id CartPole-v1"
        + " --device auto"
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
            "python abcdrl/dqn.py"
            + " --env-id CartPole-v1"
            + " --device auto"
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
