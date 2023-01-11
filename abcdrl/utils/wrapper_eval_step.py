from __future__ import annotations

from typing import Any, Callable, Generator

import gymnasium as gym
from combine_signatures.combine_signatures import combine_signatures


def wrapper_eval_step(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    @combine_signatures(wrapped)
    def _wrapper(
        *args,
        eval_frequency: int = 5_000,
        num_steps_eval: int = 500,
        eval_env_seed: int = 1,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        instance = args[0]
        eval_frequency = max(eval_frequency // instance.kwargs["num_envs"] * instance.kwargs["num_envs"], 1)
        eval_env = gym.vector.SyncVectorEnv([instance._make_env(eval_env_seed)])  # type: ignore[arg-type]
        eval_obs, _ = eval_env.reset(seed=1)

        gen = wrapped(*args, **kwargs)
        for log_data in gen:
            if not log_data["sample_step"] % eval_frequency and log_data["log_type"] == "collect":
                el_list, er_list = [], []
                for _ in range(num_steps_eval):
                    act = instance.agent.predict(eval_obs)
                    eval_obs, _, _, _, infos = eval_env.step(act)
                    if "final_info" in infos.keys():
                        final_info = next(item for item in infos["final_info"] if item is not None)
                        el_list.append(final_info["episode"]["l"][0])
                        er_list.append(final_info["episode"]["r"][0])
                eval_log_data = {"log_type": "evaluate", "sample_step": log_data["sample_step"]}
                if el_list and er_list:
                    eval_log_data["logs"] = {
                        "mean_episodic_length": sum(el_list) / len(el_list),
                        "mean_episodic_return": sum(er_list) / len(er_list),
                    }
                yield eval_log_data
            yield log_data

    return _wrapper
