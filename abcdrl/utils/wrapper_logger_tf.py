from __future__ import annotations

import time
from typing import Any, Callable, Generator

import gymnasium as gym
import tensorflow as tf
from combine_signatures.combine_signatures import combine_signatures


def wrapper_logger_tf(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    import wandb

    def setup_video_monitor() -> None:
        vcr = gym.wrappers.monitoring.video_recorder.VideoRecorder
        vcr.close_ = vcr.close  # type: ignore[attr-defined]

        def close(self):
            vcr.close_(self)
            if self.path:
                wandb.log({"videos": wandb.Video(self.path)})
                self.path = None

        vcr.close = close  # type: ignore[assignment]

    @combine_signatures(wrapped)
    def _wrapper(
        *args,
        track: bool = False,
        wandb_project_name: str = "abcdrl",
        wandb_tags: list[str] = [],
        wandb_entity: str | None = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        instance = args[0]
        exp_name_ = f"{instance.kwargs['exp_name']}__{instance.kwargs['seed']}__{int(time.time())}"
        if track:
            wandb.init(
                project=wandb_project_name,
                tags=wandb_tags,
                entity=wandb_entity,
                sync_tensorboard=True,
                config=instance.kwargs,
                name=exp_name_,
                save_code=True,
            )
            setup_video_monitor()

        writer = tf.summary.create_file_writer(f"runs/{exp_name_}")
        with writer.as_default():
            tf.summary.text(
                "hyperparameters",
                "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in instance.kwargs.items()]),
                0,
            )

            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if "logs" in log_data:
                    for log_item in log_data["logs"].items():
                        tf.summary.scalar(f"{log_data['log_type']}/{log_item[0]}", log_item[1], log_data["sample_step"])
                yield log_data

    return _wrapper
