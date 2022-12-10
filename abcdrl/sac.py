from __future__ import annotations

import copy
import dataclasses
import os
import random
import time
from typing import Any, Callable, Generator, Generic, TypeVar

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
    raise NotImplementedError(f"{env_space} observation space is not supported")


class ReplayBuffer:
    @dataclasses.dataclass(frozen=True)
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
        idxs = np.random.choice(self.size, size=batch_size, replace=True)
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
        )
        self.fc_mean = nn.Linear(256, out_n)
        self.fc_log_std = nn.Sequential(
            nn.Linear(256, out_n),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tmp = self.network(x)
        mean = self.fc_mean(tmp)
        log_std = self.fc_log_std(tmp)

        return mean, log_std


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

    def value(self, obs: torch.Tensor, act: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if act is None:
            act, _ = self.action(obs)
        return self.critic_nn_0(obs, act), self.critic_nn_1(obs, act)

    def action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        LOG_STD_MAX, LOG_STD_MIN = 2, -5
        mean, log_std = self.actor_nn(obs)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        act = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return act, log_prob


class Algorithm:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        self.model = Model(**self.kwargs).to(self.kwargs["device"])
        self.model_t = copy.deepcopy(self.model)
        self.optimizer_actor = optim.Adam(self.model.actor_nn.parameters(), lr=self.kwargs["policy_lr"])
        self.optimizer_critic = optim.Adam(
            list(self.model.critic_nn_0.parameters()) + list(self.model.critic_nn_1.parameters()),
            lr=self.kwargs["q_lr"],
        )

        self.alpha = self.kwargs["alpha"]
        if self.kwargs["autotune"]:
            self.target_entropy = -torch.prod(
                torch.Tensor(get_space_shape(self.kwargs["act_space"])).to(self.kwargs["device"])
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.kwargs["device"])
            self.alpha = self.log_alpha.exp().item()
            self.optimizer_a = optim.Adam([self.log_alpha], lr=self.kwargs["q_lr"])

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        act, _ = self.model.action(obs)
        return act

    def learn(self, data: ReplayBuffer.Samples, update_actor: bool) -> dict[str, Any]:
        with torch.no_grad():
            next_state_action, next_state_log_prob = self.model_t.action(data.next_observations)

            next_q_value_0, next_q_value_1 = self.model_t.value(data.next_observations, next_state_action)
            next_q_value = torch.min(next_q_value_0, next_q_value_1) - self.alpha * next_state_log_prob
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
            for _ in range(self.kwargs["policy_frequency"]):
                action, log_prob = self.model.action(data.observations)
                q_value_0, q_value_1 = self.model.value(data.observations, action)
                q_value = torch.min(q_value_0, q_value_1).view(-1)

                actor_loss = ((self.alpha * log_prob) - q_value).mean()
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                if self.kwargs["autotune"]:
                    _, log_prob = self.model.action(data.observations)
                    alpha_loss = (-self.log_alpha * (log_prob + self.target_entropy)).mean()

                    self.optimizer_a.zero_grad()
                    alpha_loss.backward()
                    self.optimizer_a.step()
                    self.alpha = self.log_alpha.exp().item()
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

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_ts = torch.as_tensor(obs, device=self.kwargs["device"])
        with torch.no_grad():
            act_ts = self.alg.predict(obs_ts)
        act_np = act_ts.cpu().numpy()
        return act_np

    def sample(self, obs: np.ndarray) -> np.ndarray:
        if self.sample_step < self.kwargs["learning_starts"]:
            act_np = np.array([self.kwargs["act_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs_ts = torch.as_tensor(obs, device=self.kwargs["device"])
            with torch.no_grad():
                act_ts = self.alg.predict(obs_ts)
            act_np = act_ts.cpu().numpy()

        self.sample_step += self.kwargs["num_envs"]
        if self.sample_step % self.kwargs["target_network_frequency"] == 0:
            self.alg.sync_target()
        return act_np

    def learn(self, data: ReplayBuffer.Samples[np.ndarray]) -> dict[str, Any]:
        data_ts = ReplayBuffer.Samples[torch.Tensor](
            **{
                item[0]: torch.as_tensor(item[1], device=self.kwargs["device"])
                if isinstance(item[1], np.ndarray)
                else item[1]
                for item in dataclasses.asdict(data).items()
            }
        )

        log_data = self.alg.learn(data_ts, self.sample_step % self.kwargs["policy_frequency"] == 0)
        self.learn_step += 1
        return log_data


class Trainer:
    def __init__(
        self,
        exp_name: str | None = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        capture_video: bool = False,
        env_id: str = "Hopper-v4",
        num_envs: int = 1,
        total_timesteps: int = 1_000_000,
        gamma: float = 0.99,
        # Collect
        buffer_size: int = 1_000_000,
        # Learn
        batch_size: int = 256,
        q_lr: float = 1e-3,
        policy_lr: float = 3e-4,
        tau: float = 0.005,
        alpha: float = 0.2,
        autotune: bool = True,
        # Train
        learning_starts: int = 25_000,
        train_frequency: int = 1,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
    ) -> None:
        self.kwargs = locals()
        self.kwargs.pop("self")

        if self.kwargs["exp_name"] is None:
            self.kwargs["exp_name"] = f"{self.kwargs['env_id']}__{os.path.basename(__file__).rstrip('.py')}"
        self.kwargs["policy_frequency"] = max(
            self.kwargs["policy_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        self.kwargs["target_network_frequency"] = max(
            self.kwargs["target_network_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
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

        self.obs, _ = self.envs.reset(seed=[self.kwargs["seed"] + idx for idx in range(self.kwargs["num_envs"])])
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

        real_next_obs = next_obs.copy()
        if "final_observation" in infos.keys():
            for idx, final_obs in enumerate(infos["final_observation"]):
                real_next_obs[idx] = real_next_obs[idx] if final_obs is None else final_obs

        self.buffer.add(self.obs, real_next_obs, act, reward, terminated, infos)
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
            return env

        return thunk


def wrapper_logger(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    def setup_video_monitor() -> None:
        vcr = gym.wrappers.monitoring.video_recorder.VideoRecorder
        vcr.close_ = vcr.close

        def close(self):
            vcr.close_(self)
            if self.path:
                wandb.log({"videos": wandb.Video(self.path)})
                self.path = None

        vcr.close = close

    def _wrapper(
        instance,
        *args,
        track: bool = False,
        wandb_project_name: str = "abcdrl",
        wandb_tags: list[str] = [],
        wandb_entity: str | None = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        exp_name_ = f"{instance.kwargs['exp_name']}__{instance.kwargs['seed']}__{int(time.time())}"
        if track:
            wandb.init(
                project=wandb_project_name,
                tags=wandb_tags,
                entity=wandb_entity,
                sync_tensorboard=True,
                config=instance.kwargs,
                name=exp_name_,
                save_code=True,
            )
            setup_video_monitor()

        writer = SummaryWriter(f"runs/{exp_name_}")
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
