from __future__ import annotations

from typing import Any, Callable, Generator

from abcdrl import ddqn_tf, dqn_tf, pdqn_tf
from abcdrl.utils import (
    wrapper_eval_step,
    wrapper_logger_tf,
    wrapper_print_filter,
    wrapper_save_model,
)


def set_all_wrappers(
    func: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    func = wrapper_eval_step.wrapper_eval_step(func)  # type: ignore[assignment]
    func = wrapper_logger_tf.wrapper_logger_tf(func)  # type: ignore[assignment]
    func = wrapper_save_model.wrapper_save_model(func)  # type: ignore[assignment]
    func = wrapper_print_filter.wrapper_print_filter(func)  # type: ignore[assignment]
    return func


def test_dqn_tf_wrappers() -> None:
    Trainer = dqn_tf.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="CartPole-v1",
        num_envs=2,
        learning_starts=8,
        total_timesteps=32,
        buffer_size=10,
        batch_size=4,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass


def test_ddqn_tf_wrappers() -> None:
    Trainer = ddqn_tf.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="CartPole-v1",
        num_envs=2,
        learning_starts=8,
        total_timesteps=32,
        buffer_size=10,
        batch_size=4,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass


def test_pdqn_tf_wrappers() -> None:
    Trainer = pdqn_tf.Trainer
    Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
    trainer = Trainer(
        env_id="CartPole-v1",
        num_envs=2,
        learning_starts=8,
        total_timesteps=32,
        buffer_size=10,
        batch_size=4,
    )
    for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
        pass
