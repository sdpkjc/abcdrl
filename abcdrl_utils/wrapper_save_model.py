from __future__ import annotations

import os
from typing import Any, Callable, Generator

import dill


def wrapper_save_model(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    def _wrapper(*args, save_frequency: int = 1_000_0, **kwargs) -> Generator[dict[str, Any], None, None]:
        save_frequency = max(save_frequency // args[0].kwargs["num_envs"] * args[0].kwargs["num_envs"], 1)

        gen = wrapped(*args, **kwargs)
        for log_data in gen:
            if not log_data["sample_step"] % save_frequency:
                if not os.path.exists(f"models/{args[0].kwargs['exp_name']}"):
                    os.makedirs(f"models/{args[0].kwargs['exp_name']}")
                with open(f"models/{args[0].kwargs['exp_name']}/s{args[0].agent.sample_step}.agent", "ab+") as file:
                    dill.dump(args[0].agent, file)
            yield log_data

    return _wrapper
