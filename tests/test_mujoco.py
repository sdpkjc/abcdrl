import subprocess


def test_ddpg() -> None:
    subprocess.run(
        "python abcdrl/ddpg.py"
        + " --env-id Hopper-v2"
        + " --device auto"
        + " --num-envs 1"
        + " --eval-frequency 5"
        + " --num-steps-eval 1"
        + " --learning-starts 64"
        + " --total-timesteps 256"
        + " --buffer-size 32"
        + " --batch-size 4"
        + " --save-frequency 128",
        shell=True,
        check=True,
    )


def test_td3() -> None:
    subprocess.run(
        "python abcdrl/td3.py"
        + " --env-id Hopper-v2"
        + " --device auto"
        + " --num-envs 2"
        + " --eval-frequency 5"
        + " --num-steps-eval 1"
        + " --learning-starts 64"
        + " --total-timesteps 256"
        + " --buffer-size 32"
        + " --batch-size 4"
        + " --save-frequency 128",
        shell=True,
        check=True,
    )


def test_sac() -> None:
    subprocess.run(
        "python abcdrl/sac.py"
        + " --env-id Hopper-v2"
        + " --device auto"
        + " --num-envs 2"
        + " --eval-frequency 5"
        + " --num-steps-eval 1"
        + " --learning-starts 64"
        + " --total-timesteps 256"
        + " --buffer-size 32"
        + " --batch-size 4"
        + " --save-frequency 128",
        shell=True,
        check=True,
    )


def test_ppo() -> None:
    subprocess.run(
        "python abcdrl/ppo.py"
        + " --env-id Hopper-v2"
        + " --device auto"
        + " --num-envs 2"
        + " --eval-frequency 5"
        + " --num-steps-eval 1"
        + " --num-steps 64"
        + " --num-minibatches 16"
        + " --total-timesteps 256"
        + " --save-frequency 128",
        shell=True,
        check=True,
    )
