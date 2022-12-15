from __future__ import annotations

import os
from typing import Any, Callable, Generator

import dill
from combine_signatures.combine_signatures import combine_signatures


def wrapper_save_model(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    @combine_signatures(wrapped)
    def _wrapper(*args, save_frequency: int = 1_000_0, **kwargs) -> Generator[dict[str, Any], None, None]:
        instance = args[0]
        save_frequency = max(save_frequency // instance.kwargs["num_envs"] * instance.kwargs["num_envs"], 1)

        gen = wrapped(*args, **kwargs)
        for log_data in gen:
            if not log_data["sample_step"] % save_frequency:
                if not os.path.exists(f"models/{instance.kwargs['exp_name']}"):
                    os.makedirs(f"models/{instance.kwargs['exp_name']}")
                with open(f"models/{instance.kwargs['exp_name']}/s{instance.agent.sample_step}.agent", "wb+") as file:
                    dill.dump(instance.agent, file)
            yield log_data

    return _wrapper
