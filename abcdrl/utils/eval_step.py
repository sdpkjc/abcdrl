from __future__ import annotations

import dataclasses
from typing import Any, Callable, Generator

import gymnasium as gym
import wrapt


class Evaluator:
    @dataclasses.dataclass
    class Config:
        eval_frequency: int = 5_000
        num_steps_eval: int = 500
        eval_env_seed: int = 1

    @classmethod
    def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            eval_frequency = max(config.eval_frequency // instance.config["num_envs"] * instance.config["num_envs"], 1)
            eval_env = gym.vector.SyncVectorEnv([instance._make_env(config.eval_env_seed)])  # type: ignore[arg-type]
            eval_obs, _ = eval_env.reset(seed=1)

            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if not log_data["sample_step"] % eval_frequency and log_data["log_type"] == "collect":
                    el_list, er_list = [], []
                    for _ in range(config.num_steps_eval):
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

        return wrapper
