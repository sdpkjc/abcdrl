from __future__ import annotations

# from typing import Any, Callable, Generator

# from abcdrl import ddqn_tf, dqn_tf, pdqn_tf
# from abcdrl.utils import eval_step, model_saver, print_filter, wrapper_logger_tf


# def set_all_wrappers(
#     func: Callable[..., Generator[dict[str, Any], None, None]]
# ) -> Callable[..., Generator[dict[str, Any], None, None]]:
#     func = eval_step.Evaluator.decorator(eval_step.Evaluator.Config(eval_env_seed=5, num_steps_eval=1))(func)  # type: ignore[assignment]
#     func = model_saver.Saver.decorator(model_saver.Saver.Config(save_frequency=16))(func)  # type: ignore[assignment]
#     func = wrapper_logger_tf.wrapper_logger_tf(func)  # type: ignore[assignment]
#     # func = logger_torch.Logger.decorator()(func)  # type: ignore[assignment]
#     func = print_filter.Filter.decorator()(func)  # type: ignore[assignment]

#     return func


# def test_dqn_tf_wrappers() -> None:
#     Trainer = dqn_tf.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         Trainer.Config(
#             env_id="CartPole-v1", num_envs=2, learning_starts=8, total_timesteps=32, buffer_size=10, batch_size=4
#         )
#     )
#     for _ in trainer():  # type: ignore[call-arg]
#         pass


# def test_ddqn_tf_wrappers() -> None:
#     Trainer = ddqn_tf.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         Trainer.Config(
#             env_id="CartPole-v1", num_envs=2, learning_starts=8, total_timesteps=32, buffer_size=10, batch_size=4
#         )
#     )
#     for _ in trainer():  # type: ignore[call-arg]
#         pass


# def test_pdqn_tf_wrappers() -> None:
#     Trainer = pdqn_tf.Trainer
#     Trainer.__call__ = set_all_wrappers(Trainer.__call__)  # type: ignore[assignment]
#     trainer = Trainer(
#         Trainer.Config(
#             env_id="CartPole-v1", num_envs=2, learning_starts=8, total_timesteps=32, buffer_size=10, batch_size=4
#         )
#     )
#     for _ in trainer():  # type: ignore[call-arg]
#         pass
