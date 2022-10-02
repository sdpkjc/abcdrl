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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="rl_lab",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)",
    )

    parser.add_argument(
        "--env-id",
        type=str,
        default="Hopper-v2",
        help="the id of the environment",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="the number of environments")
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10000,
        help="the frequency of evaluate",
    )
    parser.add_argument(
        "--num-ep-eval",
        type=int,
        default=5,
        help="number of episodic in a evaluation",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="total timesteps of the experiments",
    )

    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    # Collect
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="the replay memory buffer size")
    parser.add_argument("--exploration-noise", type=float, default=0.1, help="the scale of exploration noise")
    parser.add_argument(
        "--noise-clip",
        type=float,
        default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization",
    )
    # Learn
    parser.add_argument("--batch-size", type=int, default=256, help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--policy-noise", type=float, default=0.2, help="the scale of policy noise")
    # Train
    parser.add_argument("--learning-starts", type=int, default=int(25e3), help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=1, help="the frequency of training")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency of training policy (delayed)")

    args = parser.parse_args()
    return args


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
    def __init__(self, kwargs: Dict) -> None:
        super().__init__()
        self.kwargs = kwargs

        self.actor_nn = ActorNetwork(
            int(np.array(self.kwargs["envs_single_observation_space"].shape).prod()),
            int(np.array(self.kwargs["envs_single_action_space"].shape).prod()),
        )
        self.critic_nn_0 = CriticNetwork(
            int(
                np.array(self.kwargs["envs_single_observation_space"].shape).prod()
                + np.prod(self.kwargs["envs_single_action_space"].shape)
            )
        )
        self.critic_nn_1 = CriticNetwork(
            int(
                np.array(self.kwargs["envs_single_observation_space"].shape).prod()
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

    def value(self, x: torch.Tensor, a: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if a is None:
            a = self.action(x)
        return self.critic_nn_0(x, a), self.critic_nn_1(x, a)

    def action(self, x: torch.Tensor) -> torch.Tensor:
        a = self.actor_nn(x) * self.action_scale + self.action_bias
        return a


class Algorithm:
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        self.model = Model(kwargs).to(kwargs["device"])
        self.model_t = copy.deepcopy(self.model)
        self.optimizer_actor = optim.Adam(self.model.actor_nn.parameters(), lr=self.kwargs["learning_rate"])
        self.optimizer_critic = optim.Adam(
            list(self.model.critic_nn_0.parameters()) + list(self.model.critic_nn_1.parameters()),
            lr=self.kwargs["learning_rate"],
        )

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        a = self.model.action(obs)
        return a

    def learn(self, data: ReplayBufferSamples, update_actor: bool) -> Dict:
        with torch.no_grad():
            clipped_noise = (torch.randn_like(torch.Tensor(data.actions[0])) * self.kwargs["policy_noise"]).clamp(
                -self.kwargs["noise_clip"], self.kwargs["noise_clip"]
            )
            next_state_action = self.model_t.action(data.next_observations)
            next_state_action = (next_state_action + clipped_noise.to(next(self.model_t.parameters()).device)).clamp(
                self.kwargs["envs_single_action_space"].low[0], self.kwargs["envs_single_action_space"].high[0]
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

        actor_loss = None
        if update_actor:
            actor_loss, _ = self.model.value(data.observations)
            actor_loss = -actor_loss.mean()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        metadata = {
            "td_loss": critic_loss / 2,
            "td_loss_0": critic_loss_0,
            "td_loss_1": critic_loss_1,
            "actor_loss": actor_loss,
            "q_value": ((old_val_0 + old_val_1) / 2).mean(),
            "q_value_0": old_val_0.mean(),
            "q_value_1": old_val_1.mean(),
        }
        return metadata

    def sync_target(self) -> None:
        for param, target_param in zip(self.model.parameters(), self.model_t.parameters()):
            target_param.data.copy_(self.kwargs["tau"] * param.data + (1 - self.kwargs["tau"]) * target_param.data)


class Agent:
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        self.alg = Algorithm(kwargs)
        self.sample_step = 0
        self.learn_step = 0

        self.action_scale = torch.FloatTensor(
            (self.kwargs["envs_single_action_space"].high - self.kwargs["envs_single_action_space"].low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (self.kwargs["envs_single_action_space"].high + self.kwargs["envs_single_action_space"].low) / 2.0
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # 评估
        obs = torch.Tensor(obs).to(next(self.alg.model.parameters()))
        with torch.no_grad():
            act = self.alg.predict(obs)
        act = act.cpu().numpy()
        return act

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 训练
        if self.sample_step < self.kwargs["learning_starts"]:
            act = np.array([self.kwargs["envs_single_action_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs = torch.Tensor(obs).to(next(self.alg.model.parameters()))
            with torch.no_grad():
                act = self.alg.predict(obs)
            # 可能有三种噪声设置
            # noise = torch.normal(self.action_bias, self.action_scale * kwargs["exploration_noise"]).to(
            #     next(self.alg.model.parameters())
            # )
            noise = torch.normal(0, self.action_scale * kwargs["exploration_noise"]).to(
                next(self.alg.model.parameters())
            )
            act += noise
            act = (
                act.cpu().numpy().clip(kwargs["envs_single_action_space"].low, kwargs["envs_single_action_space"].high)
            )
        self.sample_step += 1
        return act

    def learn(self, data: ReplayBufferSamples) -> Dict:
        # 数据预处理 & 目标网络同步
        metadata = self.alg.learn(data, self.sample_step % self.kwargs["policy_frequency"] == 0)
        if self.sample_step % self.kwargs["policy_frequency"] == 0:
            self.alg.sync_target()
        self.learn_step += 1
        return metadata


class Trainer:
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        self.kwargs["device"] = torch.device("cuda" if torch.cuda.is_available() and kwargs["cuda"] else "cpu")
        self.envs = gym.vector.SyncVectorEnv(
            [
                self._make_env(
                    kwargs["env_id"],
                    kwargs["seed"],
                    i,
                    kwargs["capture_video"],
                )
                for i in range(kwargs["num_envs"])
            ]
        )
        self.envs.single_observation_space.dtype = np.float32
        self.eval_env = gym.vector.SyncVectorEnv([self._make_env(kwargs["env_id"], 0, 0, False)])

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

        self.obs = self.envs.reset()
        self._run_collect(n=self.kwargs["learning_starts"])
        while self.agent.sample_step < self.kwargs["total_timesteps"]:
            self._run_collect(n=self.kwargs["train_frequency"])
            self._run_train()
            if self.agent.learn_step % self.kwargs["eval_frequency"] == 0:
                self._run_evaluate(self.kwargs["num_ep_eval"])

    def _run_collect(self, n: int = 1) -> None:
        for _ in range(n):
            action = self.agent.sample(self.obs)
            next_obs, reward, done, infos = self.envs.step(action)
            real_next_obs = next_obs.copy()

            for idx, d in enumerate(done):
                if d and infos[idx].get("terminal_observation") is not None:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]

            self.buffer.add(self.obs, real_next_obs, action, reward, done, infos)
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
                        self.agent.sample_step,
                        ": episodic_length",
                        info["episode"]["l"],
                        ", episodic_return",
                        info["episode"]["r"],
                    )
                    break

    def _run_train(self) -> None:
        data = self.buffer.sample(self.kwargs["batch_size"])
        metadata = self.agent.learn(data)

        writer.add_scalar("train/td_loss", metadata["td_loss"], self.agent.sample_step)
        writer.add_scalar("train/td_loss_0", metadata["td_loss_0"], self.agent.sample_step)
        writer.add_scalar("train/td_loss_1", metadata["td_loss_1"], self.agent.sample_step)
        writer.add_scalar("train/q_value", metadata["q_value"], self.agent.sample_step)
        writer.add_scalar("train/q_value_0", metadata["q_value_0"], self.agent.sample_step)
        writer.add_scalar("train/q_value_1", metadata["q_value_1"], self.agent.sample_step)
        if metadata["actor_loss"] is not None:
            writer.add_scalar("train/actor_loss", metadata["actor_loss"], self.agent.sample_step)

    def _run_evaluate(self, n_episodic: int = 1) -> None:
        eval_obs = self.eval_env.reset()
        cnt_episodic = 0
        mean_episodic_length = 0.0
        mean_episodic_return = 0.0
        while cnt_episodic < n_episodic:
            action = self.agent.predict(eval_obs)
            eval_next_obs, reward, done, infos = self.eval_env.step(action)
            eval_obs = eval_next_obs
            cnt_episodic += done

            # logger
            for info in infos:
                if "episode" in info.keys():
                    mean_episodic_length = ((cnt_episodic - 1) / cnt_episodic) * mean_episodic_length + (
                        1 / cnt_episodic
                    ) * info["episode"]["l"]
                    mean_episodic_return = ((cnt_episodic - 1) / cnt_episodic) * mean_episodic_return + (
                        1 / cnt_episodic
                    ) * info["episode"]["r"]
                    print(
                        "Eval: episodic_length",
                        info["episode"]["l"],
                        ", episodic_return",
                        info["episode"]["r"],
                    )
                    break
        writer.add_scalar(
            "evaluate/episodic_length",
            mean_episodic_length,
            self.agent.sample_step,
        )
        writer.add_scalar(
            "evaluate/episodic_return",
            mean_episodic_return,
            self.agent.sample_step,
        )
        print(
            "Eval: mean_episodic_length",
            mean_episodic_length,
            ", mean_episodic_return",
            mean_episodic_return,
        )

    def _make_env(self, env_id: str, seed: int, idx: int, capture_video: bool) -> Callable:
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
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
