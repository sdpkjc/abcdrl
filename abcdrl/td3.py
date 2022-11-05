import copy
import os
import random
import time
from typing import Callable, Dict, Generator, NamedTuple, Optional, Tuple

import dill
import fire
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter


def get_space_shape(env_space: gym.spaces.Space):
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
        observations: torch.Tensor
        actions: torch.Tensor
        next_observations: torch.Tensor
        dones: torch.Tensor
        rewards: torch.Tensor

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        buffer_size: int = 1_000_0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.obs_buf = np.zeros((buffer_size,) + get_space_shape(obs_space), dtype=obs_space.dtype)
        self.next_obs_buf = np.zeros((buffer_size,) + get_space_shape(obs_space), dtype=obs_space.dtype)
        self.acts_buf = np.zeros((buffer_size,) + get_space_shape(act_space), dtype=act_space.dtype)
        self.rews_buf = np.zeros([buffer_size], dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)

        self.max_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        infos: dict,
    ):
        for obs_i, next_obs_i, act_i, rew_i, done_i in zip(obs, next_obs, act, rew, done):
            self.obs_buf[self.ptr] = obs_i
            self.next_obs_buf[self.ptr] = next_obs_i
            self.acts_buf[self.ptr] = act_i
            self.rews_buf[self.ptr] = rew_i
            self.done_buf[self.ptr] = done_i
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=1) -> Samples:
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        return ReplayBuffer.Samples(
            observations=torch.tensor(self.obs_buf[idxs]).to(self.device),
            next_observations=torch.tensor(self.next_obs_buf[idxs]).to(self.device),
            actions=torch.tensor(self.acts_buf[idxs]).to(self.device),
            rewards=torch.tensor(self.rews_buf[idxs]).to(self.device),
            dones=torch.tensor(self.done_buf[idxs]).to(self.device),
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

    def value(self, obs: torch.Tensor, act: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def learn(self, data: ReplayBuffer.Samples, update_actor: bool) -> Dict:
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
        obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
        with torch.no_grad():
            act = self.alg.predict(obs)
        act = act.cpu().numpy()
        return act

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 训练
        if self.sample_step < self.kwargs["learning_starts"]:
            act = np.array([self.kwargs["act_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
            with torch.no_grad():
                act = self.alg.predict(obs)
            # 可能有三种噪声设置
            # noise = torch.normal(self.action_bias, self.action_scale * self.kwargs["exploration_noise"]).to(
            #     next(self.alg.model.parameters()).device
            # )
            noise = torch.normal(0, self.action_scale * self.kwargs["exploration_noise"]).to(
                next(self.alg.model.parameters()).device
            )
            act += noise
            act = act.cpu().numpy().clip(self.kwargs["act_space"].low, self.kwargs["act_space"].high)
        self.sample_step += self.kwargs["num_envs"]
        return act

    def learn(self, data: ReplayBuffer.Samples) -> Dict:
        # 数据预处理 & 目标网络同步
        log_data = self.alg.learn(data, self.sample_step % self.kwargs["policy_frequency"] == 0)
        if self.sample_step % self.kwargs["policy_frequency"] == 0:
            self.alg.sync_target()
        self.learn_step += 1
        return log_data


class Trainer:
    def __init__(
        self,
        exp_name: Optional[str] = None,
        seed: int = 1,
        device: str = "auto",
        capture_video: bool = False,
        env_id: str = "Hopper-v2",
        num_envs: int = 1,
        eval_frequency: int = 5_000,
        num_steps_eval: int = 500,
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
        self.kwargs["eval_frequency"] = max(
            self.kwargs["eval_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        self.kwargs["policy_frequency"] = max(
            self.kwargs["policy_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.kwargs["num_envs"])])
        self.envs.single_observation_space.dtype = np.float32
        self.eval_env = gym.vector.SyncVectorEnv([self._make_env(1)])
        self.eval_env.single_observation_space.dtype = np.float32
        assert isinstance(self.envs.single_action_space, gym.spaces.Box)

        self.kwargs["obs_space"] = self.envs.single_observation_space
        self.kwargs["act_space"] = self.envs.single_action_space

        self.buffer = ReplayBuffer(
            self.kwargs["obs_space"],
            self.kwargs["act_space"],
            buffer_size=self.kwargs["buffer_size"],
            device=self.kwargs["device"],
        )

        self.obs, self.eval_obs = self.envs.reset(), self.eval_env.reset()
        self.agent = Agent(**self.kwargs)

    def __call__(self) -> Generator:
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
        return {"log_type": "collect", "sample_step": self.agent.sample_step}

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

        return {"log_type": "evaluate", "sample_step": self.agent.sample_step}

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
        *args,
        track: bool = False,
        wandb_project_name: str = "abcdrl",
        wandb_tags: list = [],
        wandb_entity: Optional[str] = None,
        **kwargs,
    ) -> Generator:
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


def saver(wrapped) -> Callable:
    def _wrapper(*args, save_frequency=1_000_0, **kwargs) -> Generator:
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


def filter(wrapped) -> Callable:
    def _wrapper(*args, **kwargs) -> Generator:
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

    Trainer.__call__ = filter(saver(logger(Trainer.__call__)))
    fire.Fire(Trainer)
