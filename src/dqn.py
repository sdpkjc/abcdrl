import argparse
import copy
import os
import random
import time
from distutils.util import strtobool
from typing import Callable, Dict

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
        default="CartPole-v1",
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
        default=500000,
        help="total timesteps of the experiments",
    )

    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    # Collect
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10000,
        help="the replay memory buffer size",
    )
    parser.add_argument(
        "--start-epsilon",
        type=float,
        default=1,
        help="the starting epsilon for exploration",
    )
    parser.add_argument(
        "--end-epsilon",
        type=float,
        default=0.05,
        help="the ending epsilon for exploration",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e",
    )
    # Learn
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="the batch size of sample from the reply memory",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    # Train
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=10000,
        help="timestep to start learning",
    )
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=500,
        help="the timesteps it takes to update the target network",
    )
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=10,
        help="the frequency of training",
    )

    args = parser.parse_args()
    return args


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
        self.nn = Network(
            int(np.array(self.kwargs["envs_single_observation_space"].shape).prod()),
            self.kwargs["envs_single_action_space"].n,
        )

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class Algorithm:
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        self.model = Model(kwargs).to(kwargs["device"])
        self.model_t = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs["learning_rate"])

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        Q = self.model.value(obs)
        return Q

    def learn(self, data: ReplayBufferSamples) -> Dict:
        with torch.no_grad():
            target_max, target_argmax = self.model_t.value(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.kwargs["gamma"] * target_max * (1 - data.dones.flatten())

        old_val = self.model.value(data.observations).gather(1, data.actions).squeeze()
        td_loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        metadata = {"td_loss": td_loss, "q_value": old_val.mean()}
        return metadata

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
        obs = torch.Tensor(obs).to(next(self.alg.model.parameters()))
        with torch.no_grad():
            _, act = self.alg.predict(obs).max(dim=1)
        act = act.cpu().numpy()
        return act

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 训练
        if random.random() < self._get_epsilon():
            act = np.array([self.kwargs["envs_single_action_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs = torch.Tensor(obs).to(next(self.alg.model.parameters()))
            with torch.no_grad():
                _, act = self.alg.predict(obs).max(dim=1)
            act = act.cpu().numpy()
        self.sample_step += 1
        return act

    def learn(self, data: ReplayBufferSamples) -> Dict:
        # 数据预处理 & 目标网络同步
        metadata = self.alg.learn(data)
        if self.sample_step % self.kwargs["target_network_frequency"] == 0:
            self.alg.sync_target()
        self.learn_step += 1
        metadata["epsilon"] = self._get_epsilon()
        return metadata

    def _get_epsilon(self) -> float:
        slope = (self.kwargs["end_epsilon"] - self.kwargs["start_epsilon"]) * (
            self.sample_step / (self.kwargs["exploration_fraction"] * self.kwargs["total_timesteps"])
        ) + self.kwargs["start_epsilon"]
        return max(slope, self.kwargs["end_epsilon"])


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
        writer.add_scalar("train/q_value", metadata["q_value"], self.agent.sample_step)
        writer.add_scalar("train/epsilon", metadata["epsilon"], self.agent.sample_step)

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
