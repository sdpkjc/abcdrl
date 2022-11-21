import subprocess


def test_dqn() -> None:
    subprocess.run(
        "python abcdrl/dqn.py"
        + " --env-id CartPole-v1"
        + " --device auto"
        + " --num-envs 2"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4",
        shell=True,
        check=True,
    )


def test_ddqn() -> None:
    subprocess.run(
        "python abcdrl/ddqn.py"
        + " --env-id CartPole-v1"
        + " --device auto"
        + " --num-envs 2"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4",
        shell=True,
        check=True,
    )


def test_pdqn() -> None:
    subprocess.run(
        "python abcdrl/pdqn.py"
        + " --env-id CartPole-v1"
        + " --device auto"
        + " --num-envs 2"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4",
        shell=True,
        check=True,
    )
