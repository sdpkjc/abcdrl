from __future__ import annotations

import os
import random
import time
from typing import Callable, Generator, NamedTuple, Optional, Union

import dill
import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def get_space_shape(env_space: gym.Space) -> tuple:
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


class RolloutBuffer:
    class Samples(NamedTuple):
        observations: Union[torch.Tensor, np.ndarray]
        actions: Union[torch.Tensor, np.ndarray]
        old_values: Union[torch.Tensor, np.ndarray]
        old_log_prob: Union[torch.Tensor, np.ndarray]
        advantages: Union[torch.Tensor, np.ndarray]
        returns: Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        buffer_size: int,
        n_envs: int = 1,
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
    ) -> None:
        self.obs_space = obs_space
        self.act_space = act_space

        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reset()

    def reset(self) -> None:
        buf_shape_prefix = (self.buffer_size // self.n_envs, self.n_envs)
        self.obs_buf = np.zeros(buf_shape_prefix + get_space_shape(self.obs_space), dtype=self.obs_space.dtype)
        self.next_obs_buf = np.zeros(buf_shape_prefix + get_space_shape(self.obs_space), dtype=self.obs_space.dtype)
        self.acts_buf = np.zeros(buf_shape_prefix + get_space_shape(self.act_space), dtype=self.act_space.dtype)
        self.rews_buf = np.zeros(buf_shape_prefix, dtype=np.float32)
        self.returns = np.zeros(buf_shape_prefix, dtype=np.float32)
        self.episode_starts = np.zeros(buf_shape_prefix, dtype=np.float32)
        self.values = np.zeros(buf_shape_prefix, dtype=np.float32)
        self.log_probs = np.zeros(buf_shape_prefix, dtype=np.float32)
        self.advantages = np.zeros(buf_shape_prefix, dtype=np.float32)

        self.ptr = 0
        self.full = False
        self.generator_ready = False

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size // self.n_envs)):
            if step == self.buffer_size // self.n_envs - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rews_buf[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        episode_start: np.ndarray,
        val: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        assert not self.full

        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        self.obs_buf[self.ptr] = np.array(obs).copy()
        self.acts_buf[self.ptr] = np.array(act).copy()
        self.rews_buf[self.ptr] = np.array(rew).copy()
        self.episode_starts[self.ptr] = np.array(episode_start).copy()
        self.values[self.ptr] = val.clone().cpu().numpy().flatten()
        self.log_probs[self.ptr] = log_prob.clone().cpu().numpy()

        self.ptr += 1
        if self.ptr == self.buffer_size // self.n_envs:
            self.full = True

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[Samples, None, None]:
        assert self.full

        if not self.generator_ready:
            for t_name in ["obs_buf", "acts_buf", "values", "log_probs", "advantages", "returns"]:
                t_shape = self.__dict__[t_name].shape
                self.__dict__[t_name] = self.__dict__[t_name].reshape((t_shape[0] * t_shape[1], *t_shape[2:]))
        self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size

        indices = np.random.permutation(self.buffer_size)
        for start_idx in range(0, self.buffer_size, batch_size):
            idxs = indices[start_idx : start_idx + batch_size]
            yield RolloutBuffer.Samples(
                observations=self.obs_buf[idxs],
                actions=self.acts_buf[idxs],
                old_values=self.values[idxs],
                old_log_prob=self.log_probs[idxs],
                advantages=self.advantages[idxs],
                returns=self.returns[idxs],
            )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    def __init__(self, in_n: int, out_n: int) -> None:
        super().__init__()
        self.network_mean = nn.Sequential(
            layer_init(nn.Linear(in_n, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, out_n),
                std=0.01,
            ),
        )
        self.network_logstd = nn.Parameter(torch.zeros(1, out_n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act_mean = self.network_mean(x)
        act_logstd = self.network_logstd.expand_as(act_mean)
        act_std = torch.exp(act_logstd)
        return act_mean, act_std


class CriticNetwork(nn.Module):
    def __init__(self, in_n: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(in_n, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)


class Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

        self.actor_nn = ActorNetwork(
            int(np.prod(get_space_shape(self.kwargs["obs_space"]))),
            int(np.prod(get_space_shape(self.kwargs["act_space"]))),
        )
        self.critic_nn = CriticNetwork(int(np.prod(get_space_shape(self.kwargs["obs_space"]))))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_nn(obs)

    def action(
        self, obs: torch.Tensor, act: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        act_mean, act_std = self.actor_nn(obs)
        probs = Normal(act_mean, act_std)
        if act is None:
            act = probs.sample()
        return act, probs.entropy().sum(1), probs.log_prob(act).sum(1)


class Algorithm:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        self.model = Model(**self.kwargs).to(self.kwargs["device"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs["learning_rate"])

    def predict(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        act, _, log_prob = self.model.action(obs)
        val = self.model.value(obs)
        return act, log_prob, val

    def learn(self, data_generator: Generator[RolloutBuffer.Samples, None, None]) -> dict:
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
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        self.alg = Algorithm(**self.kwargs)
        self.sample_step = 0
        self.learn_step = 0

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # 评估
        obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
        with torch.no_grad():
            act, _, _ = self.alg.predict(obs)
        act = act.cpu().numpy()
        return act

    def sample(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 训练
        obs = torch.Tensor(obs).to(next(self.alg.model.parameters()).device)
        with torch.no_grad():
            act, log_prob, val = self.alg.predict(obs)
        act = act.cpu().numpy()
        self.sample_step += self.kwargs["num_envs"]
        return act, log_prob, val

    def learn(self, data_generator_list: list[Generator[RolloutBuffer.Samples, None, None]]) -> dict:
        # 数据预处理
        self._update_lr()
        log_data_list = []
        for data_generator in data_generator_list:
            data_generator = (
                data._replace(
                    **{
                        item[0]: torch.tensor(item[1]).to(self.kwargs["device"])
                        for item in data._asdict().items()
                        if isinstance(item[1], np.ndarray)
                    }
                )
                for data in data_generator
            )
            log_data_list += [self.alg.learn(data_generator)]

        self.learn_step += 1
        return log_data_list

    def _update_lr(self):
        if self.kwargs["anneal_lr"]:
            frac = 1.0 - (self.learn_step - 1.0) / (self.kwargs["total_timesteps"] // self.kwargs["batch_size"])
            lrnow = frac * self.kwargs["learning_rate"]
            self.alg.optimizer.param_groups[0]["lr"] = lrnow


class Trainer:
    def __init__(
        self,
        exp_name: Optional[str] = None,
        seed: int = 1,
        device: Union[str, torch.device] = "auto",
        capture_video: bool = False,
        env_id: str = "Hopper-v4",
        num_envs: int = 1,
        eval_frequency: int = 5_000,
        num_steps_eval: int = 500,
        total_timesteps: int = 1_000_000,
        gamma: float = 0.99,
        # Collect
        num_steps: int = 2048,
        # Learn
        learning_rate: float = 3e-4,
        anneal_lr: bool = True,
        update_epochs: int = 10,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        # Train
        num_minibatches: int = 32,
        gae_lambda: float = 0.95,
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
        self.kwargs["batch_size"] = self.kwargs["num_envs"] * self.kwargs["num_steps"]
        self.kwargs["minibatch_size"] = self.kwargs["batch_size"] // self.kwargs["num_minibatches"]
        if self.kwargs["device"] == "auto":
            self.kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.kwargs["num_envs"])])
        self.eval_env = gym.vector.SyncVectorEnv([self._make_env(1)])
        assert isinstance(self.envs.single_action_space, gym.spaces.Box)

        self.kwargs["obs_space"] = self.envs.single_observation_space
        self.kwargs["act_space"] = self.envs.single_action_space

        self.buffer = RolloutBuffer(
            self.kwargs["obs_space"],
            self.kwargs["act_space"],
            self.kwargs["batch_size"],
            n_envs=self.kwargs["num_envs"],
            gae_lambda=self.kwargs["gae_lambda"],
            gamma=self.kwargs["gamma"],
        )

        self.obs, _ = self.envs.reset(seed=[seed for seed in range(self.kwargs["num_envs"])])
        self.eval_obs, _ = self.eval_env.reset(seed=1)
        self.agent = Agent(**self.kwargs)

    def __call__(self) -> Generator[dict, None, None]:
        while self.agent.sample_step < self.kwargs["total_timesteps"]:
            self.buffer.reset()

            while not self.buffer.full:
                yield self._run_collect()
                if self.agent.sample_step % self.kwargs["eval_frequency"] == 0:
                    yield self._run_evaluate(n_steps=self.kwargs["num_steps_eval"])
            yield self._run_train()

    def _run_collect(self) -> dict:
        act, log_prob, val = self.agent.sample(self.obs)
        next_obs, reward, terminated, truncated, infos = self.envs.step(act)
        done = terminated | truncated

        self.real_next_obs = next_obs.copy()
        if "final_observation" in infos.keys():
            for idx, final_obs in enumerate(infos["final_observation"]):
                self.real_next_obs[idx] = self.real_next_obs[idx] if final_obs is None else final_obs

        self.buffer.add(self.obs, act, reward, done, val, log_prob)
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

    def _run_train(self) -> dict:
        act, _, next_val = self.agent.sample(self.real_next_obs)
        self.agent.sample_step -= self.kwargs["num_envs"]
        _, _, next_terminated, next_truncated, _ = self.envs.step(act)
        next_done = next_terminated | next_truncated
        self.buffer.compute_returns_and_advantage(next_val, next_done)

        data_generator_list = [
            self.buffer.get(batch_size=self.kwargs["minibatch_size"]) for _ in range(self.kwargs["update_epochs"])
        ]

        log_data = self.agent.learn(data_generator_list)[0]

        return {"log_type": "train", "sample_step": self.agent.sample_step, "logs": log_data}

    def _run_evaluate(self, n_steps: int = 1) -> dict:
        el_list, er_list = [], []
        for _ in range(n_steps):
            act = self.agent.predict(self.eval_obs)
            self.eval_obs, _, _, _, infos = self.eval_env.step(act)

            if "final_info" in infos.keys():
                final_info = next(item for item in infos["final_info"] if item is not None)
                el_list.append(final_info["episode"]["l"][0])
                er_list.append(final_info["episode"]["r"][0])

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

    def _make_env(self, idx: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(self.kwargs["env_id"], render_mode="rgb_array")
            env.observation_space.dtype = np.float32
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.kwargs["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.kwargs['exp_name']}")
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=self.kwargs["gamma"])
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            env.action_space.seed(self.kwargs["seed"])
            env.observation_space.seed(self.kwargs["seed"])
            return env

        return thunk


def logger(wrapped) -> Callable[..., Generator[dict, None, None]]:
    def _wrapper(
        *args,
        track: bool = False,
        wandb_project_name: str = "abcdrl",
        wandb_tags: list = [],
        wandb_entity: Optional[str] = None,
        **kwargs,
    ) -> Generator[dict, None, None]:
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


def saver(wrapped) -> Callable[..., Generator[dict, None, None]]:
    def _wrapper(*args, save_frequency: int = 1_000_0, **kwargs) -> Generator[dict, None, None]:
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


def filter(wrapped) -> Callable[..., Generator[dict, None, None]]:
    def _wrapper(*args, **kwargs) -> Generator[dict, None, None]:
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
