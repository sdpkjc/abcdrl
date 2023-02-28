from __future__ import annotations

from typing import Any, Callable, Generator

# from abcdrl import (  # dqn_torch,; ddqn_torch,; dqn_atari_torch,; pdqn_torch,; ddpg_torch,; sac_torch,; td3_torch,
#     # ppo_torch,
# )
from abcdrl.utils import eval_step, logger_torch, model_saver, print_filter


def set_all_wrappers(
    func: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    func = eval_step.Evaluator.decorator()(func)  # type: ignore[assignment]
    func = model_saver.Saver.decorator()(func)  # type: ignore[assignment]
    func = logger_torch.Logger.decorator()(func)  # type: ignore[assignment]
    func = print_filter.Filter.decorator()(func)  # type: ignore[assignment]
    return func


# def test_dqn_torch_wrappers() -> None:
#     Trainer = dqn_torch.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         env_id="CartPole-v1",
#         num_envs=2,
#         learning_starts=8,
#         total_timesteps=32,
#         buffer_size=10,
#         batch_size=4,
#     )
#     for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
#         pass


# def test_dqn_atari_torch_wrappers() -> None:
#     Trainer = dqn_atari_torch.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         env_id="BreakoutNoFrameskip-v4",
#         num_envs=2,
#         learning_starts=8,
#         total_timesteps=32,
#         buffer_size=10,
#         batch_size=4,
#     )
#     for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
#         pass


# def test_ddqn_torch_wrappers() -> None:
#     Trainer = ddqn_torch.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         env_id="CartPole-v1",
#         num_envs=2,
#         learning_starts=8,
#         total_timesteps=32,
#         buffer_size=10,
#         batch_size=4,
#     )
#     for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
#         pass


# def test_pdqn_torch_wrappers() -> None:
#     Trainer = pdqn_torch.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         env_id="CartPole-v1",
#         num_envs=2,
#         learning_starts=8,
#         total_timesteps=32,
#         buffer_size=10,
#         batch_size=4,
#     )
#     for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
#         pass


# def test_ddpg_torch_wrappers() -> None:
#     Trainer = ddpg_torch.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         env_id="Hopper-v4",
#         num_envs=2,
#         learning_starts=64,
#         total_timesteps=256,
#         buffer_size=32,
#         batch_size=4,
#     )
#     for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
#         pass


# def test_td3_torch_wrappers() -> None:
#     Trainer = td3_torch.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         env_id="Hopper-v4",
#         num_envs=2,
#         learning_starts=64,
#         total_timesteps=256,
#         buffer_size=32,
#         batch_size=4,
#     )
#     for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
#         pass


# def test_sac_torch_wrappers() -> None:
#     Trainer = sac_torch.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         env_id="Hopper-v4",
#         num_envs=2,
#         learning_starts=64,
#         total_timesteps=256,
#         buffer_size=32,
#         batch_size=4,
#     )
#     for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
#         pass


# def test_ppo_torch_wrappers() -> None:
#     Trainer = ppo_torch.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         env_id="Hopper-v4",
#         num_envs=2,
#         num_steps=64,
#         num_minibatches=16,
#         total_timesteps=256,
#     )
#     for _ in trainer(eval_frequency=5, num_steps_eval=1, save_frequency=16):  # type: ignore[call-arg]
#         pass
