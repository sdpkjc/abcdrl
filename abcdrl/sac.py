import argparse
import copy
import os
import random
import time
from distutils.util import strtobool
from typing import Callable, Dict, Optional, Tuple

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


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--device", type=str, default='auto',
        help="device of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="rl_lab",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--env-id", type=str, default="Hopper-v2",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--eval-frequency", type=int, default=5_000,
        help="the frequency of evaluate")
    parser.add_argument("--num-steps-eval", type=int, default=500,
        help="the number of steps in a evaluation")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")

    # Collect
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    # Learn
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    # Train
    parser.add_argument("--learning-starts", type=int, default=int(25e3),
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=1,
        help="the frequency of training")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target nerworks")

    args = parser.parse_args()
    args.eval_frequency = max(args.eval_frequency // args.num_envs * args.num_envs, 1)
    args.policy_frequency = max(args.policy_frequency // args.num_envs * args.num_envs, 1)
    args.target_network_frequency = max(args.target_network_frequency // args.num_envs * args.num_envs, 1)
    # fmt: on
    return args


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
    def __init__(self, kwargs: Dict) -> None:
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
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        self.model = Model(kwargs).to(kwargs["device"])
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

        actor_loss = None
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

        log_data = {
            "td_loss": critic_loss,
            "td_loss_0": critic_loss_0,
            "td_loss_1": critic_loss_1,
            "actor_loss": actor_loss,
            "q_value": ((old_val_0 + old_val_1) / 2).mean(),
            "q_value_0": old_val_0.mean(),
            "q_value_1": old_val_1.mean(),
        }
        return log_data

    def sync_target(self) -> None:
        for param, target_param in zip(self.model.parameters(), self.model_t.parameters()):
            target_param.data.copy_(self.kwargs["tau"] * param.data + (1 - self.kwargs["tau"]) * target_param.data)


class Agent:
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        self.alg = Algorithm(kwargs)
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
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(kwargs["num_envs"])])
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

        self.agent = Agent(self.kwargs)

    def run(self) -> None:
        self.start_time = time.time()

        self.obs, self.eval_obs = self.envs.reset(), self.eval_env.reset()

        self._run_collect(n_steps=self.kwargs["learning_starts"])
        while self.agent.sample_step < self.kwargs["total_timesteps"]:
            for _ in range(self.kwargs["train_frequency"]):
                self._run_collect(n_steps=1)
                if self.agent.sample_step % self.kwargs["eval_frequency"] == 0:
                    self._run_evaluate(n_steps=self.kwargs["num_steps_eval"])
            self._run_train()

    def _run_collect(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            act = self.agent.sample(self.obs)
            next_obs, reward, done, infos = self.envs.step(act)
            real_next_obs = next_obs.copy()

            for idx, d in enumerate(done):
                if d and infos[idx].get("terminal_observation") is not None:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]

            self.buffer.add(self.obs, real_next_obs, act, reward, done, infos)
            self.obs = next_obs

            # logger
            for info in infos:
                if "episode" in info.keys():
                    writer.add_scalar(
                        "collect/episodic_length",
                        info["episode"]["l"],
                        self.agent.sample_step,
                    )
                    writer.add_scalar(
                        "collect/episodic_return",
                        info["episode"]["r"],
                        self.agent.sample_step,
                    )
                    print(
                        f"{self.agent.sample_step}: "
                        + f"episodic_length {info['episode']['l']}, "
                        + f"episodic_return {info['episode']['r']}"
                    )
                    break

    def _run_train(self) -> None:
        data = self.buffer.sample(self.kwargs["batch_size"])
        log_data = self.agent.learn(data)

        for log_item in log_data.items():
            if log_item[1] is not None:
                writer.add_scalar(f"train/{log_item[0]}", log_item[1], self.agent.sample_step)

    def _run_evaluate(self, n_steps: int = 1) -> None:
        el_list, er_list = [], []
        for _ in range(n_steps):
            act = self.agent.predict(self.eval_obs)
            self.eval_obs, _, _, infos = self.eval_env.step(act)

            # logger
            for info in infos:
                if "episode" in info.keys():
                    el_list.append(info["episode"]["l"])
                    er_list.append(info["episode"]["r"])
                    break

        if el_list:
            mena_el = sum(el_list) / len(el_list)
            mena_er = sum(er_list) / len(er_list)
            writer.add_scalar(
                "evaluate/episodic_length",
                mena_el,
                self.agent.sample_step,
            )
            writer.add_scalar(
                "evaluate/episodic_return",
                mena_er,
                self.agent.sample_step,
            )
            print(
                f"Eval {self.agent.sample_step}: "
                + f"mean_episodic_length {mena_el}, "
                + f"mean_episodic_return {mena_er}"
            )

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


if __name__ == "__main__":
    args = parse_args()

    # 固定随机数种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)

    kwargs = vars(args)
    kwargs["exp_name"] = f"{kwargs['env_id']}__{kwargs['exp_name']}__{kwargs['seed']}__{int(time.time())}"

    # 初始化 tensorboard & wandb
    if kwargs["track"]:
        wandb.init(
            project=kwargs["wandb_project_name"],
            entity=kwargs["wandb_entity"],
            sync_tensorboard=True,
            config=kwargs,
            name=kwargs["exp_name"],
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{kwargs['exp_name']}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in kwargs.items()])),
    )

    # 开始训练
    trainer = Trainer(kwargs)
    trainer.run()
