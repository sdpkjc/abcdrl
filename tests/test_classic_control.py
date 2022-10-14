import subprocess


def test_dqn():
    subprocess.run(
        "python abcdrl/dqn.py"
        + " --env-id CartPole-v1"
        + " --cuda True"
        + " --num-envs 1"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4",
        shell=True,
        check=True,
    )


def test_ddqn():
    subprocess.run(
        "python abcdrl/ddqn.py"
        + " --env-id CartPole-v1"
        + " --cuda True"
        + " --num-envs 1"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4",
        shell=True,
        check=True,
    )


def test_pdqn():
    subprocess.run(
        "python abcdrl/pdqn.py"
        + " --env-id CartPole-v1"
        + " --cuda True"
        + " --num-envs 1"
        + " --learning-starts 8"
        + " --total-timesteps 32"
        + " --buffer-size 10"
        + " --batch-size 4",
        shell=True,
        check=True,
    )
