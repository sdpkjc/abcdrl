import copy
import os
import random
import time
from typing import Callable, Dict, Generator, Optional, Tuple

import dill
import fire
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch.utils.tensorboard import SummaryWriter


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            int(np.prod(self.kwargs["envs_single_observation_space"].shape)),
            int(np.prod(self.kwargs["envs_single_action_space"].shape)),
        )
        self.critic_nn_0 = CriticNetwork(
            int(
                np.prod(self.kwargs["envs_single_observation_space"].shape)
                + np.prod(self.kwargs["envs_single_action_space"].shape)
            )
        )
        self.critic_nn_1 = CriticNetwork(
            int(
                np.prod(self.kwargs["envs_single_observation_space"].shape)
                + np.prod(self.kwargs["envs_single_action_space"].shape)
            )
        )

        self.register_buffer(
            "action_scale",
            torch.FloatTensor(
                (self.kwargs["envs_single_action_space"].high - self.kwargs["envs_single_action_space"].low) / 2.0
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor(
                (self.kwargs["envs_single_action_space"].high + self.kwargs["envs_single_action_space"].low) / 2.0
            ),
        )

    def value(self, obs: torch.Tensor, act: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if act is None:
            act, _ = self.action(obs)
        return self.critic_nn_0(obs, act), self.critic_nn_1(obs, act)

    def action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
                torch.Tensor(self.kwargs["envs_single_action_space"].shape).to(self.kwargs["device"])
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.kwargs["device"])
            self.alpha = self.log_alpha.exp().item()
            self.optimizer_a = optim.Adam([self.log_alpha], lr=self.kwargs["q_lr"])

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        act, _ = self.model.action(obs)
        return act

    def learn(self, data: ReplayBufferSamples, update_actor: bool) -> Dict:
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
        # 评估
        obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
        with torch.no_grad():
            act = self.alg.predict(obs)
        act = act.cpu().numpy()
        return act

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 训练
        if self.sample_step < self.kwargs["learning_starts"]:
            act = np.array([self.kwargs["envs_single_action_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
            with torch.no_grad():
                act = self.alg.predict(obs)
            act = act.cpu().numpy()
        self.sample_step += self.kwargs["num_envs"]
        return act

    def learn(self, data: ReplayBufferSamples) -> Dict:
        # 数据预处理 & 目标网络同步
        log_data = self.alg.learn(data, self.sample_step % self.kwargs["policy_frequency"] == 0)
        if self.sample_step % self.kwargs["target_network_frequency"] == 0:
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
        self.kwargs["target_network_frequency"] = max(
            self.kwargs["target_network_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.kwargs["num_envs"])])
        self.envs.single_observation_space.dtype = np.float32
        self.eval_env = gym.vector.SyncVectorEnv([self._make_env(1)])
        self.eval_env.single_observation_space.dtype = np.float32

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
