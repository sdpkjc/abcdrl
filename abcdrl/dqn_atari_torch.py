from __future__ import annotations

import copy
import dataclasses
import os
import random
import time
from typing import Any, Callable, Generator, Generic, List, Optional, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wrapt

SamplesItemType = TypeVar("SamplesItemType", torch.Tensor, np.ndarray)


def get_space_shape(env_space: gym.Space) -> tuple[int, ...]:
    if isinstance(env_space, gym.spaces.Box):
        return env_space.shape
    elif isinstance(env_space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(env_space, gym.spaces.MultiDiscrete):
        return (int(len(env_space.nvec)),)
    elif isinstance(env_space, gym.spaces.MultiBinary):
        if type(env_space.n) in [tuple, list, np.ndarray]:
            return tuple(env_space.n)
        else:
            return (int(env_space.n),)
    raise NotImplementedError(f"{env_space} observation space is not supported")


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

        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

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
    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
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
        rew: np.ndarray,
        done: np.ndarray,
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
        return self.network(x)


class Model(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        self.network = Network(
            self.config["act_space"].n,
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs / 255.0)


class Algorithm:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        self.model = Model(self.config).to(self.config["device"])
        self.model_t = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        val = self.model.value(obs)
        return val

    def learn(self, data: ReplayBuffer.Samples[torch.Tensor]) -> dict[str, Any]:
        with torch.no_grad():
            target_max, _ = self.model_t.value(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.config["gamma"] * target_max * (1 - data.dones.flatten())

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
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        self.alg = Algorithm(self.config)
        self.sample_step = 0
        self.learn_step = 0

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_ts = torch.as_tensor(obs, device=self.config["device"])
        with torch.no_grad():
            _, act_ts = self.alg.predict(obs_ts).max(dim=1)
        act_np = act_ts.cpu().numpy()
        return act_np

    def sample(self, obs: np.ndarray) -> np.ndarray:
        if random.random() < self._get_epsilon():
            act_np = np.array([self.config["act_space"].sample() for _ in range(self.config["num_envs"])])
        else:
            obs_ts = torch.as_tensor(obs, device=self.config["device"])
            with torch.no_grad():
                _, act_ts = self.alg.predict(obs_ts).max(dim=1)
            act_np = act_ts.cpu().numpy()

        self.sample_step += self.config["num_envs"]
        if self.sample_step % self.config["target_network_frequency"] == 0:
            self.alg.sync_target()
        return act_np

    def learn(self, data: ReplayBuffer.Samples[np.ndarray]) -> dict[str, Any]:
        data_ts = ReplayBuffer.Samples[torch.Tensor](
            **{
                item[0]: torch.as_tensor(item[1], device=self.config["device"])
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
        slope = (self.config["end_epsilon"] - self.config["start_epsilon"]) * (
            self.sample_step / (self.config["exploration_fraction"] * self.config["total_timesteps"])
        ) + self.config["start_epsilon"]
        return max(slope, self.config["end_epsilon"])


class Trainer:
    @dataclasses.dataclass
    class Config:
        exp_name: Optional[str] = None
        seed: int = 1
        cuda: bool = True
        capture_video: bool = False
        env_id: str = "BreakoutNoFrameskip-v4"
        num_envs: int = 1
        total_timesteps: int = 10_000_000
        gamma: float = 0.99
        # Collect
        buffer_size: int = 1_000_000
        start_epsilon: float = 1.0
        end_epsilon: float = 0.01
        exploration_fraction: float = 0.1
        # Learn
        batch_size: int = 32
        learning_rate: float = 1e-4
        # Train
        learning_starts: int = 50_000
        target_network_frequency: int = 1000
        train_frequency: int = 4

    def __init__(self, config: Config = Config()) -> None:
        self.config = dataclasses.asdict(config)
        if self.config["exp_name"] is None:
            self.config["exp_name"] = f"{self.config['env_id']}__{os.path.basename(__file__).rstrip('.py')}"
        self.config["run_name"] = f"{self.config['exp_name']}__{self.config['seed']}__{int(time.time())}"
        self.config["target_network_frequency"] = max(
            self.config["target_network_frequency"] // self.config["num_envs"] * self.config["num_envs"], 1
        )
        self.config["device"] = "cuda" if self.config["cuda"] and torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.config["num_envs"])])  # type: ignore[arg-type]
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete)

        self.config["obs_space"] = self.envs.single_observation_space
        self.config["act_space"] = self.envs.single_action_space

        self.buffer = ReplayBuffer(
            self.config["obs_space"],
            self.config["act_space"],
            buffer_size=self.config["buffer_size"],
            optimize_memory_usage=True,
        )

        self.obs, _ = self.envs.reset(seed=self.config["seed"])
        self.agent = Agent(self.config)

    def __call__(self) -> Generator[dict[str, Any], None, None]:
        for _ in range(self.config["learning_starts"]):
            yield self._run_collect()
        while self.agent.sample_step < self.config["total_timesteps"]:
            for _ in range(self.config["train_frequency"]):
                if not self.agent.sample_step < self.config["total_timesteps"]:
                    break
                yield self._run_collect()
            yield self._run_train()

        self.envs.close_extras()

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
        data = self.buffer.sample(batch_size=self.config["batch_size"])
        log_data = self.agent.learn(data)

        return {"log_type": "train", "sample_step": self.agent.sample_step, "logs": log_data}

    def _make_env(self, idx: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(self.config["env_id"], render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.config["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.config['run_name']}")
            if "NOOP" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
                env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
                env = FireResetEnv(env)
            env = gym.wrappers.TransformReward(env, np.sign)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            env.action_space.seed(self.config["seed"] + idx)
            env.observation_space.seed(self.config["seed"] + idx)
            return env

        return thunk


class Logger:
    @dataclasses.dataclass
    class Config:
        track: bool = False
        wandb_project_name: str = "abcdrl"
        wandb_tags: List[str] = dataclasses.field(default_factory=lambda: [])
        wandb_entity: Optional[str] = None

    @classmethod
    def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
        import wandb
        from torch.utils.tensorboard import SummaryWriter

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            if config.track:
                wandb.init(
                    project=config.wandb_project_name,
                    tags=config.wandb_tags,
                    entity=config.wandb_entity,
                    sync_tensorboard=True,
                    config=instance.config,
                    name=instance.config["run_name"],
                    monitor_gym=True,
                    save_code=True,
                )

            writer = SummaryWriter(f"runs/{instance.config['run_name']}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in instance.config.items()]),
            )

            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if "logs" in log_data:
                    for log_item in log_data["logs"].items():
                        writer.add_scalar(f"{log_data['log_type']}/{log_item[0]}", log_item[1], log_data["sample_step"])
                yield log_data

        return wrapper


if __name__ == "__main__":
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def main(trainer: Trainer.Config, logger: Logger.Config) -> None:
        Trainer.__call__ = Logger.decorator(logger)(Trainer.__call__)  # type: ignore[assignment]
        for log_data in Trainer(trainer)():
            if "logs" in log_data and log_data["log_type"] != "train":
                print(log_data)

    tyro.cli(main)
