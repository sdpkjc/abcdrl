from __future__ import annotations

import copy
import dataclasses
import os
import random
import time
from typing import Any, Callable, Generator, Generic, Optional, TypeVar, Union

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter

SamplesItemType = TypeVar("SamplesItemType", torch.Tensor, np.ndarray)


def get_space_shape(env_space: gym.Space) -> tuple[int, ...]:
    if isinstance(env_space, gym.spaces.Box):
        return env_space.shape
    elif isinstance(env_space, gym.spaces.Discrete):
        return (1,)
    else:
        raise NotImplementedError(f"{env_space} observation space is not supported")


class ReplayBuffer:
    @dataclasses.dataclass
    class Samples(Generic[SamplesItemType]):
        observations: SamplesItemType
        actions: SamplesItemType
        next_observations: SamplesItemType
        dones: SamplesItemType
        rewards: SamplesItemType

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
        for obs_i, next_obs_i, act_i, rew_i, done_i in zip(obs, next_obs, act, rew, done):  # type: ignore[call-overload]
            self.obs_buf[self.ptr] = np.array(obs_i).copy()
            self.next_obs_buf[self.ptr] = np.array(next_obs_i).copy()
            self.acts_buf[self.ptr] = np.array(act_i).copy()
            self.rews_buf[self.ptr] = np.array(rew_i).copy()
            self.dones_buf[self.ptr] = np.array(done_i).copy()
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int = 1) -> Samples[np.ndarray]:
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        return ReplayBuffer.Samples[np.ndarray](
            observations=self.obs_buf[idxs],
            next_observations=self.next_obs_buf[idxs],
            actions=self.acts_buf[idxs],
            rewards=self.rews_buf[idxs],
            dones=self.dones_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class ActorNetwork(nn.Module):
    def __init__(self, in_n: int, out_n: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_n, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_n),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CriticNetwork(nn.Module):
    def __init__(self, in_n: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_n, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([x, a], 1))


class Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

        self.actor_nn = ActorNetwork(
            int(np.prod(get_space_shape(self.kwargs["obs_space"]))),
            int(np.prod(get_space_shape(self.kwargs["act_space"]))),
        )
        self.critic_nn_0 = CriticNetwork(
            int(np.prod(get_space_shape(self.kwargs["obs_space"])) + np.prod(get_space_shape(self.kwargs["act_space"])))
        )
        self.critic_nn_1 = CriticNetwork(
            int(np.prod(get_space_shape(self.kwargs["obs_space"])) + np.prod(get_space_shape(self.kwargs["act_space"])))
        )

        self.register_buffer(
            "action_scale",
            torch.FloatTensor((self.kwargs["act_space"].high - self.kwargs["act_space"].low) / 2.0),
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor((self.kwargs["act_space"].high + self.kwargs["act_space"].low) / 2.0),
        )

    def value(self, obs: torch.Tensor, act: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if act is None:
            act = self.action(obs)
        return self.critic_nn_0(obs, act), self.critic_nn_1(obs, act)

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        act = self.actor_nn(obs) * self.action_scale + self.action_bias
        return act


class Algorithm:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        self.model = Model(**self.kwargs).to(self.kwargs["device"])
        self.model_t = copy.deepcopy(self.model)
        self.optimizer_actor = optim.Adam(self.model.actor_nn.parameters(), lr=self.kwargs["learning_rate"])
        self.optimizer_critic = optim.Adam(
            list(self.model.critic_nn_0.parameters()) + list(self.model.critic_nn_1.parameters()),
            lr=self.kwargs["learning_rate"],
        )

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        act = self.model.action(obs)
        return act

    def learn(self, data: ReplayBuffer.Samples, update_actor: bool) -> dict[str, Any]:
        with torch.no_grad():
            clipped_noise = (torch.randn_like(torch.Tensor(data.actions[0])) * self.kwargs["policy_noise"]).clamp(
                -self.kwargs["noise_clip"], self.kwargs["noise_clip"]
            )
            next_state_action = self.model_t.action(data.next_observations)
            next_state_action = (next_state_action + clipped_noise.to(next(self.model_t.parameters()).device)).clamp(
                self.kwargs["act_space"].low[0], self.kwargs["act_space"].high[0]
            )
            next_q_value_0, next_q_value_1 = self.model_t.value(data.next_observations, next_state_action)
            next_q_value = torch.min(next_q_value_0, next_q_value_1)
            td_target = data.rewards.flatten() + self.kwargs["gamma"] * next_q_value.view(-1) * (
                1 - data.dones.flatten()
            )

        old_val_0, old_val_1 = self.model.value(data.observations, data.actions)
        old_val_0, old_val_1 = old_val_0.view(-1), old_val_1.view(-1)

        critic_loss_0 = F.mse_loss(td_target, old_val_0)
        critic_loss_1 = F.mse_loss(td_target, old_val_1)
        critic_loss = critic_loss_0 + critic_loss_1
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        log_data = {
            "td_loss": critic_loss / 2,
            "td_loss_0": critic_loss_0,
            "td_loss_1": critic_loss_1,
            "q_value": ((old_val_0 + old_val_1) / 2).mean(),
            "q_value_0": old_val_0.mean(),
            "q_value_1": old_val_1.mean(),
        }

        if update_actor:
            actor_loss, _ = self.model.value(data.observations)
            actor_loss = -actor_loss.mean()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            log_data["actor_loss"] = actor_loss

        return log_data

    def sync_target(self) -> None:
        for param, target_param in zip(self.model.parameters(), self.model_t.parameters()):
            target_param.data.copy_(self.kwargs["tau"] * param.data + (1 - self.kwargs["tau"]) * target_param.data)


class Agent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        self.alg = Algorithm(**self.kwargs)
        self.sample_step = 0
        self.learn_step = 0

        self.action_scale = torch.FloatTensor((self.kwargs["act_space"].high - self.kwargs["act_space"].low) / 2.0)
        self.action_bias = torch.FloatTensor((self.kwargs["act_space"].high + self.kwargs["act_space"].low) / 2.0)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # 评估
        obs_ts = torch.as_tensor(obs, device=next(self.alg.model.parameters()).device)
        with torch.no_grad():
            act_ts = self.alg.predict(obs_ts)
        act_np = act_ts.cpu().numpy()
        return act_np

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 训练
        if self.sample_step < self.kwargs["learning_starts"]:
            act_np = np.array([self.kwargs["act_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs_ts = torch.as_tensor(obs, device=next(self.alg.model.parameters()).device)
            with torch.no_grad():
                act_ts = self.alg.predict(obs_ts)
            # 可能有三种噪声设置
            # noise = torch.normal(self.action_bias, self.action_scale * self.kwargs["exploration_noise"]).to(
            #     next(self.alg.model.parameters()).device
            # )
            noise = torch.normal(0, self.action_scale * self.kwargs["exploration_noise"]).to(
                next(self.alg.model.parameters()).device
            )
            act_ts += noise
            act_np = act_ts.cpu().numpy().clip(self.kwargs["act_space"].low, self.kwargs["act_space"].high)
        self.sample_step += self.kwargs["num_envs"]
        return act_np

    def learn(self, data: ReplayBuffer.Samples[np.ndarray]) -> dict[str, Any]:
        # 数据预处理 & 目标网络同步
        data_ts = ReplayBuffer.Samples[torch.Tensor](
            **{
                item[0]: torch.as_tensor(item[1], device=self.kwargs["device"])
                if isinstance(item[1], np.ndarray)
                else item[1]
                for item in dataclasses.asdict(data).items()
            }
        )

        log_data = self.alg.learn(data_ts, self.sample_step % self.kwargs["policy_frequency"] == 0)
        if self.sample_step % self.kwargs["policy_frequency"] == 0:
            self.alg.sync_target()
        self.learn_step += 1
        return log_data


class Trainer:
    def __init__(
        self,
        exp_name: Optional[str] = None,
        seed: int = 1,
        device: Union[str, torch.device] = "auto",
        capture_video: bool = False,
        env_id: str = "Hopper-v4",
        num_envs: int = 1,
        total_timesteps: int = 1_000_000,
        gamma: float = 0.99,
        # Collect
        buffer_size: int = 1_000_000,
        exploration_noise: float = 0.1,
        noise_clip: float = 0.5,
        # Learn
        batch_size: int = 256,
        learning_rate: float = 3e-4,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        # Train
        learning_starts: int = 25_000,
        train_frequency: int = 1,
        policy_frequency: int = 2,
    ) -> None:
        self.kwargs = locals()
        self.kwargs.pop("self")

        if self.kwargs["exp_name"] is None:
            self.kwargs["exp_name"] = (
                f"{self.kwargs['env_id']}__{os.path.basename(__file__).rstrip('.py')}__"
                + f"{self.kwargs['seed']}__{int(time.time())}"
            )
        self.kwargs["policy_frequency"] = max(
            self.kwargs["policy_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.kwargs["num_envs"])])  # type: ignore[arg-type]
        assert isinstance(self.envs.single_action_space, gym.spaces.Box)

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
            env.observation_space.dtype = np.float32  # type: ignore[assignment]
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.kwargs["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.kwargs['exp_name']}")
            env.action_space.seed(self.kwargs["seed"])
            env.observation_space.seed(self.kwargs["seed"])
            return env

        return thunk


def wrapper_logger(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    def _wrapper(
        instance,
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
                config=instance.kwargs,
                name=instance.kwargs["exp_name"],
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{instance.kwargs['exp_name']}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in instance.kwargs.items()]),
        )

        gen = wrapped(instance, *args, **kwargs)
        for log_data in gen:
            if "logs" in log_data:
                for log_item in log_data["logs"].items():
                    writer.add_scalar(f"{log_data['log_type']}/{log_item[0]}", log_item[1], log_data["sample_step"])
            yield log_data

    return _wrapper


def wrapper_print_filter(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    def _wrapper(instance, *args, **kwargs) -> Generator[dict[str, Any], None, None]:
        gen = wrapped(instance, *args, **kwargs)
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

    Trainer.__call__ = wrapper_logger(Trainer.__call__)  # type: ignore[assignment]
    Trainer.__call__ = wrapper_print_filter(Trainer.__call__)  # type: ignore[assignment]
    fire.Fire(Trainer)
