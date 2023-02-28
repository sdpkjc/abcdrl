from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Generator

import wrapt


class Saver:
    @dataclasses.dataclass
    class Config:
        save_frequency: int = 10_000

    @classmethod
    def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
        import dill

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            instance = args[0]
            save_frequency = max(config.save_frequency // instance.config["num_envs"] * instance.config["num_envs"], 1)

            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if not log_data["sample_step"] % save_frequency:
                    if not os.path.exists(f"models/{instance.config['exp_name']}"):
                        os.makedirs(f"models/{instance.config['exp_name']}")
                    with open(
                        f"models/{instance.config['exp_name']}/s{instance.agent.sample_step}.agent", "wb+"
                    ) as file:
                        dill.dump(instance.agent, file)
                yield log_data

        return wrapper
