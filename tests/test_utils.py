from __future__ import annotations

import subprocess
from typing import Any, Callable, Generator

import abcdrl
import abcdrl_utils


def test_benchmark() -> None:
    try:
        subprocess.run(
            "python abcdrl_utils/benchmark.py",
            shell=True,
            check=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        pass


def test_example_all_wrapper() -> None:
    subprocess.run(
        "python abcdrl_utils/example_all_wrappers.py"
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
    )


def test_example_eval_model() -> None:
    subprocess.run(
        "python abcdrl_utils/example_all_wrappers.py"
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
        "python abcdrl_utils/example_eval_model.py"
        + " --model-path models/test_eval_dqn/s16.agent"
        + " --total_timesteps 100",
        shell=True,
        check=True,
    )


def set_all_wrappers(
    func: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    func = abcdrl_utils.wrapper_eval_step(func)  # type: ignore[assignment]
    func = abcdrl_utils.wrapper_logger(func)  # type: ignore[assignment]
    func = abcdrl_utils.wrapper_save_model(func)  # type: ignore[assignment]
    func = abcdrl_utils.wrapper_print_filter(func)  # type: ignore[assignment]
    return func


def test_dqn_wrappers() -> None:
    Trainer = abcdrl.dqn.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="CartPole-v1",
        device="auto",
        num_envs=2,
        learning_starts=8,
        total_timesteps=32,
        buffer_size=10,
        batch_size=4,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass


def test_ddqn_wrappers() -> None:
    Trainer = abcdrl.ddqn.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="CartPole-v1",
        device="auto",
        num_envs=2,
        learning_starts=8,
        total_timesteps=32,
        buffer_size=10,
        batch_size=4,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass


def test_pdqn_wrappers() -> None:
    Trainer = abcdrl.pdqn.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="CartPole-v1",
        device="auto",
        num_envs=2,
        learning_starts=8,
        total_timesteps=32,
        buffer_size=10,
        batch_size=4,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass


def test_ddpg_wrappers() -> None:
    Trainer = abcdrl.ddpg.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="Hopper-v4",
        device="auto",
        num_envs=2,
        learning_starts=64,
        total_timesteps=256,
        buffer_size=32,
        batch_size=4,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass


def test_td3_wrappers() -> None:
    Trainer = abcdrl.td3.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="Hopper-v4",
        device="auto",
        num_envs=2,
        learning_starts=64,
        total_timesteps=256,
        buffer_size=32,
        batch_size=4,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass


def test_sac_wrappers() -> None:
    Trainer = abcdrl.sac.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="Hopper-v4",
        device="auto",
        num_envs=2,
        learning_starts=64,
        total_timesteps=256,
        buffer_size=32,
        batch_size=4,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass


def test_ppo_wrappers() -> None:
    Trainer = abcdrl.ppo.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="Hopper-v4",
        device="auto",
        num_envs=2,
        num_steps=64,
        num_minibatches=16,
        total_timesteps=256,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass
