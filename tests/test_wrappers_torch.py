from __future__ import annotations

from typing import Any, Callable, Generator

from abcdrl import (
    ddpg_torch,
    ddqn_torch,
    dqn_atari_torch,
    dqn_torch,
    pdqn_torch,
    ppo_torch,
    sac_torch,
    td3_torch,
)
from abcdrl.utils import (
    wrapper_eval_step,
    wrapper_logger_torch,
    wrapper_print_filter,
    wrapper_save_model,
)


def set_all_wrappers(
    func: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    func = wrapper_eval_step.wrapper_eval_step(func)  # type: ignore[assignment]
    func = wrapper_logger_torch.wrapper_logger_torch(func)  # type: ignore[assignment]
    func = wrapper_save_model.wrapper_save_model(func)  # type: ignore[assignment]
    func = wrapper_print_filter.wrapper_print_filter(func)  # type: ignore[assignment]
    return func


def test_dqn_wrappers() -> None:
    Trainer = dqn_torch.Trainer
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


def test_dqn_atari_wrappers() -> None:
    Trainer = dqn_atari_torch.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="BreakoutNoFrameskip-v4",
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
    Trainer = ddqn_torch.Trainer
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
    Trainer = pdqn_torch.Trainer
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
    Trainer = ddpg_torch.Trainer
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
    Trainer = td3_torch.Trainer
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
    Trainer = sac_torch.Trainer
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
    Trainer = ppo_torch.Trainer
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
