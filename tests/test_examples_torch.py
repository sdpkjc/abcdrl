from __future__ import annotations

import subprocess


def test_example_dqn_all_wrapper_torch() -> None:
    subprocess.run(
        "python abcdrl/utils/dqn_all_wrappers_torch.py"
        + " --trainer.env-id CartPole-v1"
        + " --trainer.num-envs 2"
        + " --trainer.learning-starts 8"
        + " --trainer.total-timesteps 32"
        + " --trainer.buffer-size 10"
        + " --trainer.batch-size 4"
        + " --trainer.eval-frequency 5"
        + " --trainer.num-steps-eval 1"
        + " --saver.save-frequency 16",
        shell=True,
        check=True,
        timeout=100,
    )


def test_example_eval_model_torch() -> None:
    subprocess.run(
        "python abcdrl/utils/dqn_all_wrappers_torch.py"
        + " --trainer.exp_name test_eval_dqn"
        + " --trainer.env-id CartPole-v1"
        + " --trainer.num-envs 1"
        + " --trainer.learning-starts 8"
        + " --trainer.total-timesteps 32"
        + " --trainer.buffer-size 64"
        + " --trainer.batch-size 4"
        + " --saver.save-frequency 16",
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
