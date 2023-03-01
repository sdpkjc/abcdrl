from __future__ import annotations

import dataclasses
import random
import time
from typing import Any, Callable, Generator, Optional

import dill
import gymnasium as gym
import numpy as np
import torch
import tyro


class Evaluater:
    @dataclasses.dataclass
    class Config:
        model_path: Optional[str] = None
        env_id: str = "CartPole-v1"
        total_timesteps: int = 10_000
        cuda: bool = True
        seed: int = 1
        capture_video: bool = False

    def __init__(self, config: Config = Config()) -> None:
        self.config = dataclasses.asdict(config)
        self.config["device"] = "cuda" if self.config["cuda"] and torch.cuda.is_available() else "cpu"

        self.eval_env = gym.vector.SyncVectorEnv([self._make_env(0)])  # type: ignore[arg-type]
        self.eval_obs, _ = self.eval_env.reset(seed=0)

        if config.model_path is None:
            raise ValueError("model_path is not specified.")
        with open(self.config["model_path"], "rb") as file:
            self.agent = dill.load(file)

    def __call__(self) -> Generator[dict[str, Any], None, None]:
        for evaluate_step in range(self.config["total_timesteps"]):
            act = self.agent.predict(self.eval_obs)
            self.eval_obs, _, _, _, infos = self.eval_env.step(act)

            if "final_info" in infos.keys():
                final_info = next(item for item in infos["final_info"] if item is not None)
                yield {
                    "log_type": "evaluate",
                    "evaluate_step": evaluate_step,
                    "logs": {
                        "episodic_length": final_info["episode"]["l"][0],
                        "episodic_return": final_info["episode"]["r"][0],
                    },
                }

    # edit point
    def _make_env(self, idx: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(self.config["env_id"])
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.config["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(
                        env, f"videos/{self.config['env_id']}__{self.config['seed']}__{int(time.time())}"
                    )
            env.action_space.seed(self.config["seed"])
            env.observation_space.seed(self.config["seed"])
            return env

        return thunk


if __name__ == "__main__":
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(1234)

    def main(evaluater: Evaluater.Config) -> None:
        for log_data in Evaluater(evaluater)():
            if "logs" in log_data and log_data["log_type"] != "train":
                print(log_data)

    tyro.cli(main)
