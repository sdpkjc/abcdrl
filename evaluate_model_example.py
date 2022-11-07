import random
import time
from typing import Callable, Generator

import dill
import fire
import gym
import numpy as np
import torch


class Evaluater:
    def __init__(
        self,
        model_path: str,
        env_id: str = "CartPole-v1",
        total_timesteps: int = 1_000_0,
        device: str = "auto",
        seed: int = 1,
        capture_video: bool = False,
    ) -> None:
        self.kwargs = locals()
        self.kwargs.pop("self")
        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.eval_env = gym.vector.SyncVectorEnv([self._make_env(0)])
        self.eval_obs = self.eval_env.reset()

        with open(self.kwargs["model_path"], "rb") as file:
            self.agent = dill.load(file)

    def __call__(self) -> Generator:
        for evaluate_step in range(self.kwargs["total_timesteps"]):
            act = self.agent.predict(self.eval_obs)
            self.eval_obs, _, _, infos = self.eval_env.step(act)

            for info in infos:
                if "episode" in info.keys():
                    yield {
                        "log_type": "evaluate",
                        "evaluate_step": evaluate_step,
                        "logs": {
                            "episodic_length": info["episode"]["l"],
                            "episodic_return": info["episode"]["r"],
                        },
                    }

    # edit point
    def _make_env(self, idx: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(self.kwargs["env_id"])
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.kwargs["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(
                        env, f"videos/{self.kwargs['env_id']}__{self.kwargs['seed']}__{int(time.time())}"
                    )
            env.seed(self.kwargs["seed"])
            env.action_space.seed(self.kwargs["seed"])
            env.observation_space.seed(self.kwargs["seed"])
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

    fire.Fire(Evaluater)
