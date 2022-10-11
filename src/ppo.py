import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Callable, Dict, Generator, List, Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
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
    parser.add_argument("--eval-frequency", type=int, default=100,
        help="the frequency of evaluate",)
    parser.add_argument("--num-ep-eval", type=int, default=5,
        help="number of episodic in a evaluation",)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")

    # Collect
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    # Learn
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    # Train
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    def __init__(self, kwargs: Dict) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.network_mean = nn.Sequential(
            layer_init(nn.Linear(np.prod(self.kwargs["envs_single_observation_space"].shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(self.kwargs["envs_single_action_space"].shape)),
                std=0.01,
            ),
        )
        self.network_logstd = nn.Parameter(
            torch.zeros(1, np.array(self.kwargs["envs_single_action_space"].shape).prod())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act_mean = self.network_mean(x)
        act_logstd = self.network_logstd.expand_as(act_mean)
        act_std = torch.exp(act_logstd)
        return act_mean, act_std


class CriticNetwork(nn.Module):
    def __init__(self, kwargs: Dict) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.prod(self.kwargs["envs_single_observation_space"].shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)


class Model(nn.Module):
    def __init__(self, kwargs: Dict) -> None:
        super().__init__()
        self.kwargs = kwargs

        self.actor_nn = ActorNetwork(kwargs)
        self.critic_nn = CriticNetwork(kwargs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_nn(obs)

    def action(
        self, obs: torch.Tensor, act: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        act_mean, act_std = self.actor_nn(obs)
        probs = Normal(act_mean, act_std)
        if act is None:
            act = probs.sample()
        return act, probs.entropy().sum(1), probs.log_prob(act).sum(1)


class Algorithm:
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        self.model = Model(kwargs).to(kwargs["device"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs["learning_rate"])

    def predict(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        act, _, log_prob = self.model.action(obs)
        val = self.model.value(obs)
        return act, log_prob, val

    def learn(self, data_generator: Generator[RolloutBufferSamples, None, None]) -> Dict:
        clipfracs = []
        for data in data_generator:
            _, entropy, newlogprob = self.model.action(data.observations, data.actions)

            # Policy loss
            logratio = newlogprob - data.old_log_prob
            ratio = logratio.exp()
            advantages = data.advantages
            if self.kwargs["norm_adv"]:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.kwargs["clip_coef"], 1 + self.kwargs["clip_coef"])
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            new_val = self.model.value(data.observations)
            new_val = new_val.view(-1)
            if self.kwargs["clip_vloss"]:
                v_loss_unclipped = (new_val - data.returns) ** 2
                v_clipped = data.old_values + torch.clamp(
                    new_val - data.old_values,
                    -self.kwargs["clip_coef"],
                    self.kwargs["clip_coef"],
                )
                v_loss_clipped = (v_clipped - data.returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_val - data.returns) ** 2).mean()

            # Entropy loss
            entropy_loss = entropy.mean()

            loss = pg_loss + self.kwargs["vf_coef"] * v_loss - self.kwargs["ent_coef"] * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs["max_grad_norm"])
            self.optimizer.step()

            # log
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > self.kwargs["clip_coef"]).float().mean().item()]

            if self.kwargs["target_kl"] is not None:
                if approx_kl > self.kwargs["target_kl"]:
                    break

        y_pred, y_true = data.old_values.cpu().numpy(), data.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        log_data = {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var,
        }
        return log_data


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
            act, _, _ = self.alg.predict(obs)
        act = act.cpu().numpy()
        return act

    def sample(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 训练
        obs = torch.Tensor(obs).to(next(self.alg.model.parameters()))
        with torch.no_grad():
            act, log_prob, v = self.alg.predict(obs)
        act = act.cpu().numpy()
        self.sample_step += 1
        return act, log_prob, v

    def learn(self, data_generator_list: List[Generator[RolloutBufferSamples, None, None]]) -> Dict:
        # 数据预处理
        self._update_lr()
        log_data_list = []
        for data_generator in data_generator_list:
            log_data_list += [self.alg.learn(data_generator)]
        self.learn_step += 1
        return log_data_list

    def _update_lr(self):
        if self.kwargs["anneal_lr"]:
            frac = 1.0 - (self.learn_step - 1.0) / (self.kwargs["total_timesteps"] // self.kwargs["batch_size"])
            lrnow = frac * self.kwargs["learning_rate"]
            self.alg.optimizer.param_groups[0]["lr"] = lrnow


class Trainer:
    def __init__(self, kwargs: Dict) -> None:
        self.kwargs = kwargs

        self.kwargs["device"] = torch.device("cuda" if torch.cuda.is_available() and kwargs["cuda"] else "cpu")
        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(kwargs["num_envs"])])
        self.envs.single_observation_space.dtype = np.float32
        self.eval_env = gym.vector.SyncVectorEnv([self._make_env(kwargs["env_id"], 0, 0, False)])

        self.kwargs["envs_single_observation_space"] = self.envs.single_observation_space
        self.kwargs["envs_single_action_space"] = self.envs.single_action_space

        self.buffer = RolloutBuffer(
            self.kwargs["num_steps"],
            self.kwargs["envs_single_observation_space"],
            self.kwargs["envs_single_action_space"],
            self.kwargs["device"],
            n_envs=self.kwargs["num_envs"],
            gae_lambda=self.kwargs["gae_lambda"],
            gamma=self.kwargs["gamma"],
        )
        self.agent = Agent(self.kwargs)

    def run(self) -> None:
        self.start_time = time.time()

        self.obs = self.envs.reset()
        while self.agent.sample_step < self.kwargs["total_timesteps"]:
            self.buffer.reset()
            self._run_collect(n=self.kwargs["num_steps"])
            self._run_train()
            if self.agent.learn_step % self.kwargs["eval_frequency"] == 0:
                self._run_evaluate(self.kwargs["num_ep_eval"])

    def _run_collect(self, n: int = 1) -> None:
        for _ in range(n):
            act, log_prob, val = self.agent.sample(self.obs)
            next_obs, reward, done, infos = self.envs.step(act)
            real_next_obs = next_obs.copy()

            for idx, d in enumerate(done):
                if d and infos[idx].get("terminal_observation") is not None:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]

            self.buffer.add(self.obs, act, reward, done, val, log_prob)
            self.obs = next_obs

            # log
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

        act, _, next_val = self.agent.sample(next_obs)
        self.agent.sample_step -= 1
        _, _, next_done, _ = self.envs.step(act)
        self.buffer.compute_returns_and_advantage(next_val, next_done)

    def _run_train(self) -> None:
        data_generator_list = [
            self.buffer.get(self.kwargs["minibatch_size"]) for _ in range(self.kwargs["update_epochs"])
        ]

        log_data_list = self.agent.learn(data_generator_list)

        for log_data in log_data_list[0].items():
            if log_data[1] is not None:
                writer.add_scalar(f"train/{log_data[0]}", log_data[1], self.agent.sample_step)

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

        writer.add_scalar(
            "evaluate/episodic_length",
            sum_episodic_length / n_episodic,
            self.agent.sample_step,
        )
        writer.add_scalar(
            "evaluate/episodic_return",
            sum_episodic_return / n_episodic,
            self.agent.sample_step,
        )
        print(f"Eval: mean_episodic_length {sum_episodic_length}, mean_episodic_return {sum_episodic_return}")

    def _make_env(self, idx: int) -> Callable:
        def thunk():
            env = gym.make(self.kwargs["env_id"])
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.kwargs["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.kwargs['exp_name']}")
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=self.kwargs["gamma"])
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
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
