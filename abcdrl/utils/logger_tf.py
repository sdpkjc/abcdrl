from __future__ import annotations

import dataclasses
from typing import Any, Callable, Generator, List, Optional

import tensorflow as tf
import wrapt


class Logger:
    @dataclasses.dataclass
    class Config:
        track: bool = False
        wandb_project_name: str = "abcdrl"
        wandb_tags: List[str] = dataclasses.field(default_factory=lambda: [])
        wandb_entity: Optional[str] = None

    @classmethod
    def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
        import wandb

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            if config.track:
                wandb.init(
                    project=config.wandb_project_name,
                    tags=config.wandb_tags,
                    entity=config.wandb_entity,
                    sync_tensorboard=True,
                    config=instance.config,
                    name=instance.config["run_name"],
                    monitor_gym=True,
                    save_code=True,
                )

            writer = tf.summary.create_file_writer(f"runs/{instance.config['run_name']}")
            with writer.as_default():
                tf.summary.text(
                    "hyperparameters",
                    "|param|value|\n|-|-|\n"
                    + "\n".join([f"|{key}|{value}|" for key, value in instance.config.items()]),
                    0,
                )

                gen = wrapped(*args, **kwargs)
                for log_data in gen:
                    if "logs" in log_data:
                        for log_item in log_data["logs"].items():
                            tf.summary.scalar(
                                f"{log_data['log_type']}/{log_item[0]}", log_item[1], log_data["sample_step"]
                            )
                    yield log_data

        return wrapper
