import subprocess


def test_benchmark():
    try:
        subprocess.run(
            "python benchmark.py",
            shell=True,
            check=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        pass


def test_eval():
    subprocess.run(
        "python abcdrl/dqn.py"
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
    )
    subprocess.run(
        "python evaluate_model_example.py" + " --model-path models/test_eval_dqn/s16.agent" + " --total_timesteps 100",
        shell=True,
        check=True,
    )
