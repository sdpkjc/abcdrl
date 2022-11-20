from __future__ import annotations

import copy
import os
import random
import time
from typing import Any, Callable, Generator, NamedTuple, Optional, Union

import dill
import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter


def get_space_shape(env_space: gym.Space) -> tuple[Any]:
    if isinstance(env_space, gym.spaces.Box):
        return env_space.shape
    elif isinstance(env_space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(env_space, gym.spaces.MultiDiscrete):
        return (int(len(env_space.nvec)),)
    elif isinstance(env_space, gym.spaces.MultiBinary):
        return (int(env_space.n),)
    else:
        raise NotImplementedError(f"{env_space} observation space is not supported")


class ReplayBuffer:
    class Samples(NamedTuple):
        observations: Union[torch.Tensor, np.ndarray]
        actions: Union[torch.Tensor, np.ndarray]
        next_observations: Union[torch.Tensor, np.ndarray]
        dones: Union[torch.Tensor, np.ndarray]
        rewards: Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        buffer_size: int = 1_000_0,
    ) -> None:
        self.obs_buf = np.zeros((buffer_size,) + get_space_shape(obs_space), dtype=obs_space.dtype)
        self.next_obs_buf = np.zeros((buffer_size,) + get_space_shape(obs_space), dtype=obs_space.dtype)
        self.acts_buf = np.zeros((buffer_size,) + get_space_shape(act_space), dtype=act_space.dtype)
        self.rews_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.dones_buf = np.zeros((buffer_size,), dtype=np.float32)

        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        infos: dict[str, Any],
    ) -> None:
        for obs_i, next_obs_i, act_i, rew_i, done_i in zip(obs, next_obs, act, rew, done):
            self.obs_buf[self.ptr] = np.array(obs_i).copy()
            self.next_obs_buf[self.ptr] = np.array(next_obs_i).copy()
            self.acts_buf[self.ptr] = np.array(act_i).copy()
            self.rews_buf[self.ptr] = np.array(rew_i).copy()
            self.dones_buf[self.ptr] = np.array(done_i).copy()
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int = 1) -> Samples:
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        return ReplayBuffer.Samples(
            observations=self.obs_buf[idxs],
            next_observations=self.next_obs_buf[idxs],
            actions=self.acts_buf[idxs],
            rewards=self.rews_buf[idxs],
            dones=self.dones_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


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
            np.prod(get_space_shape(self.kwargs["obs_space"])),
            self.kwargs["act_space"].n,
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

    def learn(self, data: ReplayBuffer.Samples) -> dict[str, Any]:
        with torch.no_grad():
            _, target_argmax = self.model.value(data.next_observations).max(dim=1)
            # double dqn
            target_max = self.model_t.value(data.next_observations).gather(1, target_argmax.unsqueeze(1)).squeeze()
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
            act = np.array([self.kwargs["act_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
            with torch.no_grad():
                _, act = self.alg.predict(obs).max(dim=1)
            act = act.cpu().numpy()
        self.sample_step += self.kwargs["num_envs"]
        return act

    def learn(self, data: ReplayBuffer.Samples) -> dict[str, Any]:
        # 数据预处理 & 目标网络同步
        data = data._replace(
            **{
                item[0]: torch.as_tensor(item[1], device=self.kwargs["device"])
                for item in data._asdict().items()
                if isinstance(item[1], np.ndarray)
            }
        )

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
        exp_name: Optional[str] = None,
        seed: int = 1,
        device: Union[str, torch.device] = "auto",
        capture_video: bool = False,
        env_id: str = "CartPole-v1",
        num_envs: int = 1,
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

        if self.kwargs["exp_name"] is None:
            self.kwargs["exp_name"] = (
                f"{self.kwargs['env_id']}__{os.path.basename(__file__).rstrip('.py')}__"
                + f"{self.kwargs['seed']}__{int(time.time())}"
            )
        self.kwargs["target_network_frequency"] = max(
            self.kwargs["target_network_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.kwargs["num_envs"])])
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete)

        self.kwargs["obs_space"] = self.envs.single_observation_space
        self.kwargs["act_space"] = self.envs.single_action_space

        self.buffer = ReplayBuffer(
            self.kwargs["obs_space"],
            self.kwargs["act_space"],
            buffer_size=self.kwargs["buffer_size"],
        )

        self.obs, _ = self.envs.reset(seed=[seed for seed in range(self.kwargs["num_envs"])])
        self.agent = Agent(**self.kwargs)

    def __call__(self) -> Generator[dict[str, Any], None, None]:
        for _ in range(self.kwargs["learning_starts"]):
            yield self._run_collect()
        while self.agent.sample_step < self.kwargs["total_timesteps"]:
            for _ in range(self.kwargs["train_frequency"]):
                yield self._run_collect()
            yield self._run_train()

    def _run_collect(self) -> dict[str, Any]:
        act = self.agent.sample(self.obs)
        next_obs, reward, terminated, truncated, infos = self.envs.step(act)
        done = terminated | truncated

        real_next_obs = next_obs.copy()
        if "final_observation" in infos.keys():
            for idx, final_obs in enumerate(infos["final_observation"]):
                real_next_obs[idx] = real_next_obs[idx] if final_obs is None else final_obs

        self.buffer.add(self.obs, real_next_obs, act, reward, done, infos)
        self.obs = next_obs

        if "final_info" in infos.keys():
            final_info = next(item for item in infos["final_info"] if item is not None)
            return {
                "log_type": "collect",
                "sample_step": self.agent.sample_step,
                "logs": {
                    "episodic_length": final_info["episode"]["l"][0],
                    "episodic_return": final_info["episode"]["r"][0],
                },
            }
        return {"log_type": "collect", "sample_step": self.agent.sample_step}

    def _run_train(self) -> dict[str, Any]:
        data = self.buffer.sample(batch_size=self.kwargs["batch_size"])
        log_data = self.agent.learn(data)

        return {"log_type": "train", "sample_step": self.agent.sample_step, "logs": log_data}

    def _make_env(self, idx: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(self.kwargs["env_id"], render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.kwargs["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.kwargs['exp_name']}")
            env.action_space.seed(self.kwargs["seed"])
            env.observation_space.seed(self.kwargs["seed"])
            return env

        return thunk


def eval_step_wrapper(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    def _wrapper(
        *args,
        eval_frequency: int = 5_000,
        num_steps_eval: int = 500,
        eval_env_seed: int = 1,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        eval_frequency = max(eval_frequency // args[0].kwargs["num_envs"] * args[0].kwargs["num_envs"], 1)
        eval_env = gym.vector.SyncVectorEnv([args[0]._make_env(eval_env_seed)])
        eval_obs, _ = eval_env.reset(seed=1)

        gen = wrapped(*args, **kwargs)
        for log_data in gen:
            if not log_data["sample_step"] % eval_frequency and log_data["log_type"] == "collect":
                el_list, er_list = [], []
                for _ in range(num_steps_eval):
                    act = args[0].agent.predict(eval_obs)
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


def logger_wrapper(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    def _wrapper(
        *args,
        track: bool = False,
        wandb_project_name: str = "abcdrl",
        wandb_tags: list[str] = [],
        wandb_entity: Optional[str] = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        if track:
            wandb.init(
                project=wandb_project_name,
                tags=wandb_tags,
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
            yield log_data

    return _wrapper


def save_model_wrapper(
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


def filter_wrapper(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    def _wrapper(*args, **kwargs) -> Generator[dict[str, Any], None, None]:
        gen = wrapped(*args, **kwargs)
        for log_data in gen:
            if "logs" in log_data and log_data["log_type"] != "train":
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

    Trainer.__call__ = eval_step_wrapper(Trainer.__call__)
    Trainer.__call__ = logger_wrapper(Trainer.__call__)
    Trainer.__call__ = save_model_wrapper(Trainer.__call__)
    Trainer.__call__ = filter_wrapper(Trainer.__call__)
    fire.Fire(Trainer)
