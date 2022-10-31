import copy
import os
import random
import time
from typing import Callable, Dict, Optional, Generator

import fire
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch.utils.tensorboard import SummaryWriter

import wandb


class Network(nn.Module):
    def __init__(self, in_n: int, out_n: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_n, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

        self.network = Network(
            int(np.prod(self.kwargs["envs_single_observation_space"].shape)),
            self.kwargs["envs_single_action_space"].n,
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class Algorithm:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        self.model = Model(**self.kwargs).to(self.kwargs["device"])
        self.model_t = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs["learning_rate"])

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        val = self.model.value(obs)
        return val

    def learn(self, data: ReplayBufferSamples) -> Dict:
        with torch.no_grad():
            target_max, _ = self.model_t.value(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.kwargs["gamma"] * target_max * (1 - data.dones.flatten())

        old_val = self.model.value(data.observations).gather(1, data.actions).squeeze()
        td_loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        log_data = {"td_loss": td_loss, "q_value": old_val.mean()}
        return log_data

    def sync_target(self) -> None:
        self.model_t.load_state_dict(self.model.state_dict())


class Agent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        self.alg = Algorithm(**self.kwargs)
        self.sample_step = 0
        self.learn_step = 0

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # 评估
        obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
        with torch.no_grad():
            _, act = self.alg.predict(obs).max(dim=1)
        act = act.cpu().numpy()
        return act

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 训练
        if random.random() < self._get_epsilon():
            act = np.array([self.kwargs["envs_single_action_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
            with torch.no_grad():
                _, act = self.alg.predict(obs).max(dim=1)
            act = act.cpu().numpy()
        self.sample_step += self.kwargs["num_envs"]
        return act

    def learn(self, data: ReplayBufferSamples) -> Dict:
        # 数据预处理 & 目标网络同步
        log_data = self.alg.learn(data)
        if self.sample_step % self.kwargs["target_network_frequency"] == 0:
            self.alg.sync_target()
        self.learn_step += 1
        log_data["epsilon"] = self._get_epsilon()
        return log_data

    def _get_epsilon(self) -> float:
        slope = (self.kwargs["end_epsilon"] - self.kwargs["start_epsilon"]) * (
            self.sample_step / (self.kwargs["exploration_fraction"] * self.kwargs["total_timesteps"])
        ) + self.kwargs["start_epsilon"]
        return max(slope, self.kwargs["end_epsilon"])


class Trainer:
    def __init__(
        self,
        exp_name: str = os.path.basename(__file__).rstrip(".py"),
        seed: int = 1,
        device: str = "auto",
        capture_video: bool = False,
        env_id: str = "CartPole-v1",
        num_envs: int = 1,
        eval_frequency: int = 1_000,
        num_steps_eval: int = 100,
        total_timesteps: int = 5_000_00,
        gamma: float = 0.99,
        # Collect
        buffer_size: int = 1_000_0,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.05,
        exploration_fraction: float = 0.5,
        # Learn
        batch_size: int = 128,
        learning_rate: float = 2.5e-4,
        # Train
        learning_starts: int = 1_000_0,
        target_network_frequency: int = 500,
        train_frequency: int = 10,
    ) -> None:
        self.kwargs = locals()
        self.kwargs.pop("self")

        self.kwargs["exp_name"] = (
            f"{self.kwargs['env_id']}__{self.kwargs['exp_name']}__" + f"{self.kwargs['seed']}__{int(time.time())}"
        )
        self.kwargs["eval_frequency"] = max(
            self.kwargs["eval_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        self.kwargs["target_network_frequency"] = max(
            self.kwargs["target_network_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.kwargs["num_envs"])])
        self.eval_env = gym.vector.SyncVectorEnv([self._make_env(1)])

        self.kwargs["envs_single_observation_space"] = self.envs.single_observation_space
        self.kwargs["envs_single_action_space"] = self.envs.single_action_space

        self.buffer = ReplayBuffer(
            self.kwargs["buffer_size"],
            self.kwargs["envs_single_observation_space"],
            self.kwargs["envs_single_action_space"],
            self.kwargs["device"],
            n_envs=self.kwargs["num_envs"],
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        self.agent = Agent(**self.kwargs)

    def __call__(self) -> Generator:
        self.obs, self.eval_obs = self.envs.reset(), self.eval_env.reset()

        for _ in range(self.kwargs["learning_starts"]):
            yield self._run_collect()
        while self.agent.sample_step < self.kwargs["total_timesteps"]:
            for _ in range(self.kwargs["train_frequency"]):
                yield self._run_collect()
                if self.agent.sample_step % self.kwargs["eval_frequency"] == 0:
                    yield self._run_evaluate(n_steps=self.kwargs["num_steps_eval"])
            yield self._run_train()

    def _run_collect(self) -> Dict:
        act = self.agent.sample(self.obs)
        next_obs, reward, done, infos = self.envs.step(act)
        real_next_obs = next_obs.copy()

        for idx, done_i in enumerate(done):
            if done_i and infos[idx].get("terminal_observation") is not None:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        self.buffer.add(self.obs, real_next_obs, act, reward, done, infos)
        self.obs = next_obs

        for info in infos:
            if "episode" in info.keys():
                return {
                    "log_type": "collect",
                    "sample_step": self.agent.sample_step,
                    "logs": {
                        "episodic_length": info["episode"]["l"],
                        "episodic_return": info["episode"]["r"],
                    },
                }
        return {"log_type": "collect"}

    def _run_train(self) -> Dict:
        data = self.buffer.sample(self.kwargs["batch_size"])
        log_data = self.agent.learn(data)

        return {"log_type": "train", "sample_step": self.agent.sample_step, "logs": log_data}

    def _run_evaluate(self, n_steps: int = 1) -> Dict:
        el_list, er_list = [], []
        for _ in range(n_steps):
            act = self.agent.predict(self.eval_obs)
            self.eval_obs, _, _, infos = self.eval_env.step(act)

            for info in infos:
                if "episode" in info.keys():
                    el_list.append(info["episode"]["l"])
                    er_list.append(info["episode"]["r"])
                    break

        if el_list:
            return {
                "log_type": "evaluate",
                "sample_step": self.agent.sample_step,
                "logs": {
                    "mean_episodic_length": sum(el_list) / len(el_list),
                    "mean_episodic_return": sum(er_list) / len(er_list),
                },
            }

        return {"log_type": "evaluate"}

    def _make_env(self, idx: int) -> Callable:
        def thunk():
            env = gym.make(self.kwargs["env_id"])
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.kwargs["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.kwargs['exp_name']}")
            env.seed(self.kwargs["seed"])
            env.action_space.seed(self.kwargs["seed"])
            env.observation_space.seed(self.kwargs["seed"])
            return env

        return thunk


def logger(wrapped) -> Callable:
    def _wrapper(
        *args, track: bool = False, wandb_project_name: str = "abcdrl", wandb_entity: Optional[str] = None, **kwargs
    ) -> Generator:
        if track:
            wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                sync_tensorboard=True,
                config=args[0].kwargs,
                name=args[0].kwargs["exp_name"],
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{args[0].kwargs['exp_name']}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in args[0].kwargs.items()])),
        )

        gen = wrapped(*args, **kwargs)
        for log_data in gen:
            if "logs" in log_data:
                for log_item in log_data["logs"].items():
                    writer.add_scalar(f"{log_data['log_type']}/{log_item[0]}", log_item[1], log_data["sample_step"])
                if log_data["log_type"] != "train":
                    yield log_data

    return _wrapper


if __name__ == "__main__":
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(1234)

    Trainer.__call__ = logger(Trainer.__call__)
    fire.Fire(Trainer)
