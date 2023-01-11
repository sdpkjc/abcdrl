from __future__ import annotations

import subprocess


def test_example_dqn_all_wrapper_torch() -> None:
    subprocess.run(
        "python abcdrl/utils/dqn_all_wrappers_torch.py"
        + " --env-id CartPole-v1"
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


def test_example_eval_model_torch() -> None:
    subprocess.run(
        "python abcdrl/utils/dqn_all_wrappers_torch.py"
        + " --exp_name test_eval_dqn"
        + " --env-id CartPole-v1"
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