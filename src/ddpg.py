import argparse
from asyncio.log import logger
import os
import random
import time
import copy
from distutils.util import strtobool

import wandb
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gym.vector.sync_vector_env import SyncVectorEnv
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Any, Callable, Dict, Optional, Union, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="rltest",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v2",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=int(25e3),
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=1,
                        help="the frequency of training")

    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--policy-frequency", type=int, default=2,
                        help="the frequency of training policy (delayed)")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
                        help="the scale of exploration noise")

    parser.add_argument("--num-envs", type=int, default=1,
                    help="number of envs")
                    
    args = parser.parse_args()
    return args


class MyLogger():
    def __init__(self, run_name: str, args: argparse.Namespace) -> None:
        global global_step
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        self.writer.close()
        self.start_time: float = time.time()
   
    def __del__(self):
        self.writer.close()
   
    def logging(self, td_loss: Optional[torch.Tensor] = None,
                q_values: Optional[float] = None,
                actor_loss: Optional[torch.Tensor] = None,
                episodic_return: Optional[np.ndarray] = None,
                episodic_length: Optional[np.ndarray] = None,) -> None:
        global global_step

        # Trainer
        if td_loss is not None:
            self.writer.add_scalar("losses/td_loss", td_loss, global_step)
        if q_values is not None:
            self.writer.add_scalar("losses/q_values", q_values, global_step)
        if actor_loss is not None:
            self.writer.add_scalar("losses/actor_loss", actor_loss, global_step)
        if (td_loss is not None) and (q_values is not None):
            SPS = int(global_step / (time.time() - self.start_time))
            self.writer.add_scalar("charts/SPS", SPS, global_step)
            print("SPS:", SPS)
        # Collector
        if episodic_return is not None:
            self.writer.add_scalar(
                "charts/episodic_return", episodic_return, global_step)
            print(
                f"global_step={global_step}, episodic_return={episodic_return}")
        if episodic_length is not None:
            self.writer.add_scalar(
                "charts/episodic_length", episodic_length, global_step)


class ActorNetwork(nn.Module):
    def __init__(self, in_n: int, out_n: int,
                 envs_action_space_high: float,
                 envs_action_space_low: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_n, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_n),
            nn.Tanh(),
        )
        self.register_buffer("action_scale", torch.FloatTensor((envs_action_space_high - envs_action_space_low) / 2.0))
        self.register_buffer("action_bias", torch.FloatTensor((envs_action_space_high + envs_action_space_low) / 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) * self.action_scale + self.action_bias


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


class Policy(nn.Module):
    def __init__(self, envs_single_observation_space: gym.spaces,
                 envs_single_action_space: gym.spaces,) -> None:
        super().__init__()
        self.actor_nn = ActorNetwork(
            int(np.array(envs_single_observation_space.shape).prod()),
            int(np.array(envs_single_action_space.shape).prod()),
            envs_single_action_space.high, envs_single_action_space.low)
        self.critic_nn = CriticNetwork(
            int(np.array(envs_single_observation_space.shape).prod() +
                np.prod(envs_single_action_space.shape)))

    def value(self, obs: torch.Tensor, act: Optional[torch.Tensor] = None) -> torch.Tensor:
        if act is None:
            act = self.action(obs)
        return self.critic_nn(obs, act)

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_nn(obs)


class Buffer():
    def __init__(self, envs_single_observation_space: gym.spaces,
                 envs_single_action_space: gym.spaces,
                 device: torch.device,
                 num_envs: int = 1,
                 buffer_size: int = 10000,) -> None:
        self.replaybuffer = ReplayBuffer(
            buffer_size,
            envs_single_observation_space,
            envs_single_action_space,
            device,
            n_envs = num_envs,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

    def store(self, *args: Any) -> None:
        self.replaybuffer.add(*args)

    def sample(self, batch_size: int = 1) -> ReplayBufferSamples:
        return self.replaybuffer.sample(batch_size)

class Collector():
    def __init__(self, envs: SyncVectorEnv,
                 policy: Policy, buffer_store: Callable, logger_logging: Callable,
                 learning_starts: int = int(25e3),
                 exploration_noise: float = 0.1) -> None:
        self.envs = envs
        self.policy = policy

        self.store_func = buffer_store
        self.logging_func = logger_logging

        self.exploration_noise = exploration_noise
        self.learning_starts = learning_starts

        self.obs = self.envs.reset()

    def _get_actions(self) -> np.ndarray:
        global global_step
        if global_step < self.learning_starts:
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        else:
            with torch.no_grad():
                actions = self.policy.action(torch.Tensor(self.obs).to(next(self.policy.parameters()).device))
                actions += torch.normal(self.policy.actor_nn.action_bias,
                                        self.policy.actor_nn.action_scale * self.exploration_noise)
                actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)
        return actions

    def step(self, n: int = 1) -> float:
        global global_step
        for t in range(n):
            actions = self._get_actions()
            next_obs, rewards, dones, infos = self.envs.step(actions)

            for info in infos:
                if "episode" in info.keys():
                    self.logging_func(episodic_return=info["episode"]["r"],
                                      episodic_length=info["episode"]["l"],)
                    break

            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d and infos[idx].get("terminal_observation") is not None:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]

            self.store_func(self.obs, real_next_obs,
                            actions, rewards, dones, infos)

            self.obs = next_obs
            global_step += 1


class Trainer():
    def __init__(self, policy: Policy,
                 buffer_get_sample: Callable, logger_logging: Callable,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.1,
                 policy_frequency: int = 2,
                 batch_size: int = 256,) -> None:
        self.policy = policy

        self.get_sample_func = buffer_get_sample
        self.logging_func = logger_logging

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.policy_frequency = policy_frequency
        self.batch_size = batch_size

        self.policy_t = copy.deepcopy(self.policy)
        self.optimizer_actor = optim.Adam(
            self.policy.actor_nn.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(
            self.policy.critic_nn.parameters(), lr=self.learning_rate)

    def _loss(self, data: ReplayBufferSamples) -> torch.Tensor:
        global global_step

        with torch.no_grad():
            next_q_value = self.policy_t.value(data.next_observations)
            td_target = data.rewards.flatten() + self.gamma * next_q_value.view(-1) * \
                (1 - data.dones.flatten())
        old_val = self.policy.value(data.observations, data.actions).view(-1)   

        critic_loss = F.mse_loss(td_target, old_val)
        actor_loss = -self.policy.value(data.observations).mean()
        
        if global_step % 100 == 0:
            self.logging_func(td_loss=critic_loss.item(),
                              q_values=old_val.mean().item())

        return critic_loss, actor_loss

    def _update_sth(self, critic_loss: torch.Tensor,
                    actor_loss: Optional[torch.Tensor] = None) -> None:
        # Policy.actor
        if actor_loss is not None:
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        # Policy.critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Policy.actor
        if actor_loss is not None:
            for param, target_param in zip(self.policy.parameters(), self.policy_t.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def step(self, n: int = 1) -> None:
        global global_step
        for t in range(n):
            data = self.get_sample_func(self.batch_size)

            critic_loss, actor_loss = self._loss(data)

            if global_step % self.policy_frequency != 0:
                actor_loss = None
            self._update_sth(critic_loss=critic_loss, actor_loss=actor_loss)


class Mediator_DDPG():
    def __init__(self, args: argparse.Namespace) -> None:
        global global_step
        global_step = 0

        # set config
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.learning_starts = args.learning_starts
        self.train_frequency = args.train_frequency
        self.total_timesteps = args.total_timesteps

        # set link
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.logger = MyLogger(self.run_name, args)
        self.envs = gym.vector.SyncVectorEnv(
            [self._make_env(args.env_id, args.seed, 0, args.capture_video)])
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        self.envs.single_observation_space.dtype = np.float32
        self.policy = Policy(self.envs.single_observation_space,
                             self.envs.single_action_space,).to(self.device)
        self.buffer = Buffer(self.envs.single_observation_space,
                             self.envs.single_action_space,
                             self.device,
                             num_envs=args.num_envs,
                             buffer_size=args.buffer_size,)
        self.collector = Collector(self.envs, self.policy,
                                   self.buffer.store, self.logger.logging,
                                   exploration_noise=args.exploration_noise,
                                   learning_starts=args.learning_starts,)
        self.trainer = Trainer(self.policy,
                               self.buffer.sample, self.logger.logging,
                               learning_rate=args.learning_rate,
                               gamma=args.gamma,
                               tau=args.tau,
                               policy_frequency=args.policy_frequency,
                               batch_size=args.batch_size,)

    def __del__(self) -> None:
        self.envs.close()

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

    def run(self) -> None:
        global global_step
        self.collector.step(n=self.learning_starts)
        while global_step < self.total_timesteps:
            self.collector.step(n=self.train_frequency)
            self.trainer.step(n=1)


if __name__ == "__main__":
    dqn = Mediator_DDPG(parse_args())
    dqn.run()
