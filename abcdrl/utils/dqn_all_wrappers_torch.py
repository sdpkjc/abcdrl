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
        rew: np.ndarray,
        done: np.ndarray,
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
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        self.network = Network(
            int(np.prod(get_space_shape(self.config["obs_space"]))),
            self.config["act_space"].n,
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


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
        env_id: str = "CartPole-v1"
        num_envs: int = 1
        total_timesteps: int = 500_000
        gamma: float = 0.99
        # Collect
        buffer_size: int = 10_000
        start_epsilon: float = 1.0
        end_epsilon: float = 0.05
        exploration_fraction: float = 0.5
        # Learn
        batch_size: int = 128
        learning_rate: float = 2.5e-4
        # Train
        learning_starts: int = 10_000
        target_network_frequency: int = 500
        train_frequency: int = 10

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
            env.action_space.seed(self.config["seed"] + idx)
            env.observation_space.seed(self.config["seed"] + idx)
            return env

        return thunk


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


class Saver:
    @dataclasses.dataclass
    class Config:
        save_frequency: int = 10_000

    @classmethod
    def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
        import dill

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            save_frequency = max(config.save_frequency // instance.config["num_envs"] * instance.config["num_envs"], 1)

            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if not log_data["sample_step"] % save_frequency:
                    if not os.path.exists(f"models/{instance.config['exp_name']}"):
                        os.makedirs(f"models/{instance.config['exp_name']}")
                    with open(
                        f"models/{instance.config['exp_name']}/s{instance.agent.sample_step}.agent", "wb+"
                    ) as file:
                        dill.dump(instance.agent, file)
                yield log_data

        return wrapper


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


class Filter:
    @classmethod
    def decorator(cls) -> Callable[..., Generator[dict[str, Any], None, None]]:
        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if "logs" in log_data and log_data["log_type"] != "train":
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

    def main(trainer: Trainer.Config, evaluator: Evaluator.Config, saver: Saver.Config, logger: Logger.Config) -> None:
        Trainer.__call__ = Evaluator.decorator(evaluator)(Trainer.__call__)  # type: ignore[assignment]
        Trainer.__call__ = Saver.decorator(saver)(Trainer.__call__)  # type: ignore[assignment]
        Trainer.__call__ = Logger.decorator(logger)(Trainer.__call__)  # type: ignore[assignment]
        Trainer.__call__ = Filter.decorator()(Trainer.__call__)  # type: ignore[assignment]

        for log_data in Trainer(trainer)():
            if "logs" in log_data and log_data["log_type"] != "train":
                print(log_data)

    tyro.cli(main)
