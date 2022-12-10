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


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        total_reward = 0.0
        terminated, truncated = None, None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info


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
        optimize_memory_usage: bool = False,
    ) -> None:
        self.obs_buf = np.zeros((buffer_size,) + get_space_shape(obs_space), dtype=obs_space.dtype)
        if not optimize_memory_usage:
            self.next_obs_buf = np.zeros((buffer_size,) + get_space_shape(obs_space), dtype=obs_space.dtype)
        self.acts_buf = np.zeros((buffer_size,) + get_space_shape(act_space), dtype=act_space.dtype)
        self.rews_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.dones_buf = np.zeros((buffer_size,), dtype=np.float32)

        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.optimize_memory_usage = optimize_memory_usage

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
            if not self.optimize_memory_usage:
                self.next_obs_buf[self.ptr] = np.array(next_obs_i).copy()
            else:
                self.obs_buf[(self.ptr + 1) % self.buffer_size] = np.array(next_obs_i).copy()
            self.acts_buf[self.ptr] = np.array(act_i).copy()
            self.rews_buf[self.ptr] = np.array(rew_i).copy()
            self.dones_buf[self.ptr] = np.array(done_i).copy()
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int = 1) -> Samples[np.ndarray]:
        if not self.optimize_memory_usage:
            idxs = np.random.choice(self.size, size=batch_size, replace=True)
            next_observations = self.next_obs_buf[idxs]
        else:
            if self.size != self.buffer_size:
                idxs = np.random.choice(self.size, size=batch_size, replace=True)
            else:
                idxs = (
                    (np.random.choice(self.size - 1, size=batch_size, replace=True) + 1) + self.ptr
                ) % self.buffer_size
            next_observations = self.obs_buf[(idxs + 1) % self.buffer_size]

        return ReplayBuffer.Samples[np.ndarray](
            observations=self.obs_buf[idxs],
            next_observations=next_observations,
            actions=self.acts_buf[idxs],
            rewards=self.rews_buf[idxs],
            dones=self.dones_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class Network(nn.Module):
    def __init__(self, out_n: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, out_n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)


class Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

        self.network = Network(
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

    def learn(self, data: ReplayBuffer.Samples[torch.Tensor]) -> dict[str, Any]:
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
        obs_ts = torch.as_tensor(obs, device=self.kwargs["device"])
        with torch.no_grad():
            _, act_ts = self.alg.predict(obs_ts).max(dim=1)
        act_np = act_ts.cpu().numpy()
        return act_np

    def sample(self, obs: np.ndarray) -> np.ndarray:
        if random.random() < self._get_epsilon():
            act_np = np.array([self.kwargs["act_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs_ts = torch.as_tensor(obs, device=self.kwargs["device"])
            with torch.no_grad():
                _, act_ts = self.alg.predict(obs_ts).max(dim=1)
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

        log_data = self.alg.learn(data_ts)
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
        exp_name: str | None = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        capture_video: bool = False,
        env_id: str = "BreakoutNoFrameskip-v4",
        num_envs: int = 1,
        total_timesteps: int = 10_000_000,
        gamma: float = 0.99,
        # Collect
        buffer_size: int = 1_000_000,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.01,
        exploration_fraction: float = 0.1,
        # Learn
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        # Train
        learning_starts: int = 50_000,
        target_network_frequency: int = 1000,
        train_frequency: int = 4,
    ) -> None:
        self.kwargs = locals()
        self.kwargs.pop("self")

        if self.kwargs["exp_name"] is None:
            self.kwargs["exp_name"] = f"{self.kwargs['env_id']}__{os.path.basename(__file__).rstrip('.py')}"
        self.kwargs["target_network_frequency"] = max(
            self.kwargs["target_network_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.kwargs["num_envs"])])  # type: ignore[arg-type]
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete)

        self.kwargs["obs_space"] = self.envs.single_observation_space
        self.kwargs["act_space"] = self.envs.single_action_space

        self.buffer = ReplayBuffer(
            self.kwargs["obs_space"],
            self.kwargs["act_space"],
            buffer_size=self.kwargs["buffer_size"],
            optimize_memory_usage=True,
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
            # EpisodicLifeEnv
            if "episode" in final_info:
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
            if "NOOP" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
                env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = gym.wrappers.TransformReward(env, np.sign)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
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
        if track:
            wandb.init(
                project=wandb_project_name,
                tags=wandb_tags,
                entity=wandb_entity,
                sync_tensorboard=True,
                config=instance.kwargs,
                name=instance.kwargs["exp_name"] + f"__{instance.kwargs['seed']}__{int(time.time())}",
                save_code=True,
            )
            setup_video_monitor()

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
