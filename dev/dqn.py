import argparse
import os
import random
import time
import copy
from distutils.util import strtobool
# test
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
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-epsilon", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-epsilon", type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
                        help="the frequency of training")

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
                episodic_return: Optional[np.ndarray] = None,
                episodic_length: Optional[np.ndarray] = None,
                epsilon: Optional[float] = None) -> None:
        global global_step

        # Trainer
        if td_loss is not None:
            self.writer.add_scalar("losses/td_loss", td_loss, global_step)
        if q_values is not None:
            self.writer.add_scalar("losses/q_values", q_values, global_step)
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
        if epsilon is not None:
            self.writer.add_scalar("charts/epsilon", epsilon, global_step)
            

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


class Policy(nn.Module):
    def __init__(self, envs_single_observation_space: gym.spaces, envs_single_action_space: gym.spaces) -> None:
        super().__init__()
        self.nn = Network(int(np.array(envs_single_observation_space.shape).prod()), envs_single_action_space.n)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        Q = self.nn(obs)
        return Q

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        Q = self.value(obs)
        actions = torch.argmax(Q, dim=1)
        return actions


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
                 start_epsilon: float = 1.0, end_epsilon: float = 0.05,
                 exploration_fraction: float = 0.5,
                 total_timesteps: int = 500000,) -> None:
        self.envs = envs
        self.policy = policy

        self.store_func = buffer_store
        self.logging_func = logger_logging

        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps

        self.obs = self.envs.reset()

    def _get_actions(self) -> np.ndarray:
        if random.random() < self._get_epsilon():
            actions = np.array([self.envs.single_action_space.sample()
                               for _ in range(self.envs.num_envs)])
        else:
            actions = self.policy.action(torch.Tensor(self.obs).to(
                next(self.policy.parameters()).device)).cpu().numpy()
        return actions

    def _get_epsilon(self) -> float:
        global global_step
        slope = (self.end_epsilon - self.start_epsilon) * \
                (global_step / (self.exploration_fraction *
                 self.total_timesteps)) + self.start_epsilon
        return max(slope, self.end_epsilon)

    def step(self, n: int = 1) -> float:
        global global_step
        for t in range(n):
            actions = self._get_actions()
            next_obs, rewards, dones, infos = self.envs.step(actions)

            for info in infos:
                if "episode" in info.keys():
                    self.logging_func(episodic_return=info["episode"]["r"],
                                      episodic_length=info["episode"]["l"],
                                      epsilon=self._get_epsilon(),)
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
                 learning_rate: float = 2.5e-4, gamma: float = 0.99,
                 target_network_frequency: int = 500,
                 batch_size: int = 128,) -> None:
        self.policy = policy

        self.get_sample_func = buffer_get_sample
        self.logging_func = logger_logging

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_network_frequency = target_network_frequency
        self.batch_size = batch_size

        self.policy_t = copy.deepcopy(policy)
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.learning_rate)

    def _loss(self, data: ReplayBufferSamples) -> torch.Tensor:
        global global_step
        with torch.no_grad():
            target_max, target_argmax = self.policy_t.value(
                data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.gamma * target_max * \
                (1 - data.dones.flatten())
        old_val = self.policy.value(data.observations).gather(1, data.actions).squeeze()

        loss = F.mse_loss(td_target, old_val)
        
        if global_step % 100 == 0:
            self.logging_func(td_loss=loss, q_values=old_val.mean().item())

        return loss

    def _update_sth(self, loss: torch.Tensor,
                    target_policy_update: bool = False) -> None:
        # Policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if target_policy_update:
            self.policy_t.load_state_dict(self.policy.state_dict())

    def step(self, n: int = 1) -> None:
        global global_step
        for t in range(n):
            data = self.get_sample_func(self.batch_size)
            loss = self._loss(data)
            target_policy_update = global_step % self.target_network_frequency == 0
            self._update_sth(
                loss=loss, target_policy_update=target_policy_update)


class Mediator_DQN():
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
            [self._make_env(args.env_id, args.seed, i, args.capture_video) for i in range(args.num_envs)])
        assert isinstance(self.envs.single_action_space,
                          gym.spaces.Discrete), "only discrete action space is supported"
        self.policy = Policy(self.envs.single_observation_space,
                             self.envs.single_action_space).to(self.device)
        self.buffer = Buffer(self.envs.single_observation_space,
                             self.envs.single_action_space,
                             self.device,
                             num_envs=args.num_envs,
                             buffer_size=args.buffer_size,)
        self.collector = Collector(self.envs, self.policy,
                                   self.buffer.store, self.logger.logging,
                                   start_epsilon=args.start_epsilon,
                                   end_epsilon=args.end_epsilon,
                                   exploration_fraction=args.exploration_fraction,
                                   total_timesteps=args.total_timesteps,)
        self.trainer = Trainer(self.policy,
                               self.buffer.sample, self.logger.logging,
                               learning_rate=args.learning_rate,
                               gamma=args.gamma,
                               batch_size=args.batch_size,
                               target_network_frequency=args.target_network_frequency,)

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
    dqn = Mediator_DQN(parse_args())
    dqn.run()
