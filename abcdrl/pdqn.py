import argparse
import copy
import operator
import os
import random
import time
from distutils.util import strtobool
from typing import Callable, Dict, List, NamedTuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
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

    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--eval-frequency", type=int, default=1_000_0,
        help="the frequency of evaluate")
    parser.add_argument("--num-ep-eval", type=int, default=5,
        help="number of episodic in a evaluation")
    parser.add_argument("--total-timesteps", type=int, default=5_000_00,
        help="total timesteps of the experiments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")

    # Collect
    parser.add_argument("--alpha", type=float, default=0.2,
        help="PER's alpha")
    parser.add_argument("--beta", type=float, default=0.6,
        help="PER's beta")
    parser.add_argument("--buffer-size", type=int, default=1_000_0,
        help="the replay memory buffer size")
    parser.add_argument("--start-epsilon", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-epsilon", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    # Learn
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    # Train
    parser.add_argument("--learning-starts", type=int, default=1_000_0,
        help="timestep to start learning")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")

    args = parser.parse_args()
    # fmt: on
    return args


# https://github.com/DLR-RM/stable-baselines3/blob/57e0054e62ac5a964f9c1e557b59028307d21bff/stable_baselines3/common/type_aliases.py
class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    infos: List
    weights: List
    indices: List


# https://github.com/Curt-Park/rainbow-is-all-you-need
class ReplayBuffer:
    def __init__(self, envs_single_observation_space, envs_single_action_space, size: int, device):
        self.obs_buf = np.zeros([size, *envs_single_observation_space.shape], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *envs_single_observation_space.shape], dtype=np.float32)
        self.acts_buf = np.zeros([size, *envs_single_action_space.shape, 1], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.infos_buf = [{} for _ in range(size)]
        self.max_size = size
        self.ptr, self.size, = (
            0,
            0,
        )
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
        for obs_i, next_obs_i, act_i, rew_i, done_i, infos_i in zip(obs, next_obs, act, rew, done, infos):
            self.obs_buf[self.ptr] = obs_i
            self.next_obs_buf[self.ptr] = next_obs_i
            self.acts_buf[self.ptr] = act_i
            self.rews_buf[self.ptr] = rew_i
            self.done_buf[self.ptr] = done_i
            self.infos_buf[self.ptr] = infos_i
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=1) -> ReplayBufferSamples:
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        return ReplayBufferSamples(
            observations=torch.tensor(self.obs_buf[idxs]).to(self.device),
            next_observations=torch.tensor(self.next_obs_buf[idxs]).to(self.device),
            actions=torch.tensor(self.acts_buf[idxs]).to(self.device).long(),
            rewards=torch.tensor(self.rews_buf[idxs]).to(self.device),
            dones=torch.tensor(self.done_buf[idxs]).to(self.device),
            infos=self.infos_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


# https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/segment_tree.py
class SegmentTree:
    def __init__(self, capacity: int, operation: Callable, init_value: float):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super(SumSegmentTree, self).__init__(capacity=capacity, operation=operator.add, init_value=0.0)

    def sum(self, start: int = 0, end: int = 0) -> float:
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1
        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min, init_value=float("inf"))

    def min(self, start: int = 0, end: int = 0) -> float:
        return super(MinSegmentTree, self).operate(start, end)


# https://nbviewer.org/github/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        envs_single_observation_space: gym.spaces,
        envs_single_action_space: gym.spaces,
        buffer_size: int = 10000,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_envs: int = 1,
        alpha: float = 0.2,
    ):
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            envs_single_observation_space, envs_single_action_space, buffer_size, device
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.num_envs = num_envs
        self.alpha = alpha
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        act: int,
        rew: float,
        done: bool,
        infos: dict,
    ):
        """Store experience and priority."""
        super().add(obs, next_obs, act, rew, done, infos)

        for i in range(self.num_envs):
            self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, batch_size=1, beta: float = 0.6) -> ReplayBufferSamples:
        """Sample a batch of experiences."""
        assert len(self) >= batch_size
        assert beta > 0

        indices = self._sample_proportional(batch_size)

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        infos = [self.infos_buf[i] for i in indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return ReplayBufferSamples(
            observations=torch.tensor(obs).to(self.device),
            next_observations=torch.tensor(next_obs).to(self.device),
            actions=torch.tensor(acts).to(self.device).long(),
            rewards=torch.tensor(rews).to(self.device),
            dones=torch.tensor(done).to(self.device),
            infos=infos,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size=1) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


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
    def __init__(self, kwargs: Dict) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.network = Network(
            int(np.prod(self.kwargs["envs_single_observation_space"].shape)),
            self.kwargs["envs_single_action_space"].n,
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class Algorithm:
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        self.model = Model(kwargs).to(kwargs["device"])
        self.model_t = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs["learning_rate"])

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        val = self.model.value(obs)
        return val

    def learn(self, data: ReplayBufferSamples) -> Dict:
        with torch.no_grad():
            target_max, _ = self.model_t.value(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.kwargs["gamma"] * target_max * (1 - data.dones.flatten())

        old_val = self.model.value(data.observations).gather(1, data.actions).squeeze()
        td_loss = F.mse_loss(td_target, old_val)

        weights = torch.FloatTensor(data.weights.reshape(-1, 1)).to(next(self.model.parameters()).device)
        elementwise_td_loss = F.mse_loss(td_target, old_val, reduction="none")
        td_loss = torch.mean(elementwise_td_loss * weights)  # PER

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        log_data = {"td_loss": td_loss, "elementwise_td_loss": elementwise_td_loss, "q_value": old_val.mean()}
        return log_data

    def sync_target(self) -> None:
        self.model_t.load_state_dict(self.model.state_dict())


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
            _, act = self.alg.predict(obs).max(dim=1)
        act = act.cpu().numpy()
        return act

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 训练
        if random.random() < self._get_epsilon():
            act = np.array([self.kwargs["envs_single_action_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
            with torch.no_grad():
                _, act = self.alg.predict(obs).max(dim=1)
            act = act.cpu().numpy()
        self.sample_step += 1
        return act

    def learn(self, data: ReplayBufferSamples) -> Dict:
        # 数据预处理 & 目标网络同步
        log_data = self.alg.learn(data)
        if self.sample_step % self.kwargs["target_network_frequency"] == 0:
            self.alg.sync_target()
        self.learn_step += 1
        log_data["epsilon"] = self._get_epsilon()
        return log_data

    def _get_epsilon(self) -> float:
        slope = (self.kwargs["end_epsilon"] - self.kwargs["start_epsilon"]) * (
            self.sample_step / (self.kwargs["exploration_fraction"] * self.kwargs["total_timesteps"])
        ) + self.kwargs["start_epsilon"]
        return max(slope, self.kwargs["end_epsilon"])


class Trainer:
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(kwargs["num_envs"])])
        self.eval_env = gym.vector.SyncVectorEnv([self._make_env(1)])

        self.kwargs["envs_single_observation_space"] = self.envs.single_observation_space
        self.kwargs["envs_single_action_space"] = self.envs.single_action_space

        self.buffer = PrioritizedReplayBuffer(
            self.kwargs["envs_single_observation_space"],
            self.kwargs["envs_single_action_space"],
            buffer_size=self.kwargs["buffer_size"],
            device=self.kwargs["device"],
            num_envs=self.kwargs["num_envs"],
            alpha=kwargs["alpha"],
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
                        f"{self.agent.sample_step}: episodic_length {info['episode']['l']}, episodic_return {info['episode']['r']}"
                    )
                    break

    def _run_train(self) -> None:
        data = self.buffer.sample(self.kwargs["batch_size"])
        log_data = self.agent.learn(data)

        loss_for_prior = log_data["elementwise_td_loss"].detach().cpu().numpy() + 1e-6
        self.buffer.update_priorities(data.indices, loss_for_prior)

        log_data["elementwise_td_loss"] = None
        for log_item in log_data.items():
            if log_item[1] is not None:
                writer.add_scalar(f"train/{log_item[0]}", log_item[1], self.agent.sample_step)

    def _run_evaluate(self, n_episodic: int = 1) -> None:
        eval_obs = self.eval_env.reset()

        sum_episodic_length, sum_episodic_return = 0.0, 0.0
        cnt_episodic = 0
        while cnt_episodic < n_episodic:
            act = self.agent.predict(eval_obs)
            eval_next_obs, _, done, infos = self.eval_env.step(act)
            eval_obs = eval_next_obs
            cnt_episodic += done

            # logger
            for info in infos:
                if "episode" in info.keys():
                    sum_episodic_length += info["episode"]["l"]
                    sum_episodic_return += info["episode"]["r"]
                    print(f"Eval: episodic_length {info['episode']['l']}, episodic_return {info['episode']['r']}")
                    break

        mean_episodic_length = sum_episodic_length / n_episodic
        mean_episodic_return = sum_episodic_return / n_episodic
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
        print(f"Eval: mean_episodic_length {mean_episodic_length}, mean_episodic_return {mean_episodic_return}")

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
