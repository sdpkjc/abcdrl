from __future__ import annotations

import dataclasses
import os
import random
import time
from typing import Any, Callable, Generator, Generic, List, Optional, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import wrapt
from torch.distributions.normal import Normal

SamplesItemType = TypeVar("SamplesItemType", torch.Tensor, np.ndarray)


def get_space_shape(env_space: gym.Space) -> tuple[int, ...]:
    if isinstance(env_space, gym.spaces.Box):
        return env_space.shape
    elif isinstance(env_space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(env_space, gym.spaces.MultiDiscrete):
        return (int(len(env_space.nvec)),)
    elif isinstance(env_space, gym.spaces.MultiBinary):
        if type(env_space.n) in [tuple, list, np.ndarray]:
            return tuple(env_space.n)
        else:
            return (int(env_space.n),)
    raise NotImplementedError(f"{env_space} observation space is not supported")


class RolloutBuffer:
    @dataclasses.dataclass(frozen=True)
    class Samples(Generic[SamplesItemType]):
        observations: SamplesItemType
        actions: SamplesItemType
        old_values: SamplesItemType
        old_log_prob: SamplesItemType
        advantages: SamplesItemType
        returns: SamplesItemType

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

    def compute_returns_and_advantage(self, last_values: np.ndarray, dones: np.ndarray) -> None:
        last_values = last_values.flatten()

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
        val: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        assert not self.full

        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        self.obs_buf[self.ptr] = np.array(obs).copy()
        self.acts_buf[self.ptr] = np.array(act).copy()
        self.rews_buf[self.ptr] = np.array(rew).copy()
        self.episode_starts[self.ptr] = np.array(episode_start).copy()
        self.values[self.ptr] = np.array(val).copy().flatten()
        self.log_probs[self.ptr] = np.array(log_prob).copy()

        self.ptr += 1
        if self.ptr == self.buffer_size // self.n_envs:
            self.full = True

    def get(
        self,
        batch_size: int | None = None,
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
            yield RolloutBuffer.Samples[np.ndarray](
                observations=self.obs_buf[idxs],
                actions=self.acts_buf[idxs],
                old_values=self.values[idxs],
                old_log_prob=self.log_probs[idxs],
                advantages=self.advantages[idxs],
                returns=self.returns[idxs],
            )


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        self.actor_nn = ActorNetwork(
            int(np.prod(get_space_shape(self.config["obs_space"]))),
            int(np.prod(get_space_shape(self.config["act_space"]))),
        )
        self.critic_nn = CriticNetwork(int(np.prod(get_space_shape(self.config["obs_space"]))))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_nn(obs)

    def action(
        self, obs: torch.Tensor, act: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        act_mean, act_std = self.actor_nn(obs)
        probs = Normal(act_mean, act_std)
        if act is None:
            act = probs.sample()
        return act, probs.entropy().sum(1), probs.log_prob(act).sum(1)


class Algorithm:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        self.model = Model(self.config).to(self.config["device"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def predict(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        act, _, log_prob = self.model.action(obs)
        val = self.model.value(obs)
        return act, log_prob, val

    def learn(self, data_generator: Generator[RolloutBuffer.Samples[torch.Tensor], None, None]) -> dict[str, Any]:
        clipfracs = []
        for data in data_generator:
            _, entropy, newlogprob = self.model.action(data.observations, data.actions)

            # Policy loss
            logratio = newlogprob - data.old_log_prob
            ratio = logratio.exp()
            advantages = data.advantages
            if self.config["norm_adv"]:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.config["clip_coef"], 1 + self.config["clip_coef"])
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            new_val = self.model.value(data.observations)
            new_val = new_val.view(-1)
            if self.config["clip_vloss"]:
                v_loss_unclipped = (new_val - data.returns) ** 2
                v_clipped = data.old_values + torch.clamp(
                    new_val - data.old_values,
                    -self.config["clip_coef"],
                    self.config["clip_coef"],
                )
                v_loss_clipped = (v_clipped - data.returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_val - data.returns) ** 2).mean()

            # Entropy loss
            entropy_loss = entropy.mean()

            loss = pg_loss + self.config["vf_coef"] * v_loss - self.config["ent_coef"] * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()

            # log
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > self.config["clip_coef"]).float().mean().item()]

            if self.config["target_kl"] is not None:
                if approx_kl > self.config["target_kl"]:
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
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        self.alg = Algorithm(self.config)
        self.sample_step = 0
        self.learn_step = 0

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_ts = torch.as_tensor(obs, device=self.config["device"])
        with torch.no_grad():
            act_ts, _, _ = self.alg.predict(obs_ts)
        act_np = act_ts.cpu().numpy()
        return act_np

    def sample(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_ts = torch.as_tensor(obs, device=self.config["device"])
        with torch.no_grad():
            act_ts, log_prob_ts, val_ts = self.alg.predict(obs_ts)
        act_np, log_prob_np, val_np = act_ts.cpu().numpy(), log_prob_ts.cpu().numpy(), val_ts.cpu().numpy()
        self.sample_step += self.config["num_envs"]
        return act_np, log_prob_np, val_np

    def learn(
        self, data_generator_list: list[Generator[RolloutBuffer.Samples[np.ndarray], None, None]]
    ) -> list[dict[str, Any]]:
        self._update_lr()
        log_data_list = []
        for data_generator_np in data_generator_list:
            data_generator_ts = (
                RolloutBuffer.Samples[torch.Tensor](
                    **{
                        item[0]: torch.as_tensor(item[1], device=self.config["device"])
                        if isinstance(item[1], np.ndarray)
                        else item[1]
                        for item in dataclasses.asdict(data).items()
                    }
                )
                for data in data_generator_np
            )
            log_data_list += [self.alg.learn(data_generator_ts)]

        self.learn_step += 1
        return log_data_list

    def _update_lr(self):
        if self.config["anneal_lr"]:
            frac = 1.0 - (self.learn_step - 1.0) / (self.config["total_timesteps"] // self.config["batch_size"])
            lrnow = frac * self.config["learning_rate"]
            self.alg.optimizer.param_groups[0]["lr"] = lrnow


class Trainer:
    @dataclasses.dataclass
    class Config:
        exp_name: Optional[str] = None
        seed: int = 1
        cuda: bool = True
        capture_video: bool = False
        env_id: str = "Hopper-v4"
        num_envs: int = 1
        total_timesteps: int = 1_000_000
        gamma: float = 0.99
        # Collect
        num_steps: int = 2048
        # Learn
        learning_rate: float = 3e-4
        anneal_lr: bool = True
        update_epochs: int = 10
        norm_adv: bool = True
        clip_coef: float = 0.2
        clip_vloss: bool = True
        ent_coef: float = 0.0
        vf_coef: float = 0.5
        max_grad_norm: float = 0.5
        target_kl: Optional[float] = None
        # Train
        num_minibatches: int = 32
        gae_lambda: float = 0.95

    def __init__(self, config: Config = Config()) -> None:
        self.config = dataclasses.asdict(config)
        if self.config["exp_name"] is None:
            self.config["exp_name"] = f"{self.config['env_id']}__{os.path.basename(__file__).rstrip('.py')}"
        self.config["run_name"] = f"{self.config['exp_name']}__{self.config['seed']}__{int(time.time())}"
        self.config["batch_size"] = self.config["num_envs"] * self.config["num_steps"]
        self.config["minibatch_size"] = self.config["batch_size"] // self.config["num_minibatches"]
        self.config["device"] = "cuda" if self.config["cuda"] and torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.config["num_envs"])])  # type: ignore[arg-type]
        assert isinstance(self.envs.single_action_space, gym.spaces.Box)

        self.config["obs_space"] = self.envs.single_observation_space
        self.config["act_space"] = self.envs.single_action_space

        self.buffer = RolloutBuffer(
            self.config["obs_space"],
            self.config["act_space"],
            self.config["batch_size"],
            n_envs=self.config["num_envs"],
            gae_lambda=self.config["gae_lambda"],
            gamma=self.config["gamma"],
        )

        self.obs, _ = self.envs.reset(seed=self.config["seed"])
        self.terminated = np.zeros((self.config["num_envs"],), dtype=np.float32)

        self.agent = Agent(self.config)

    def __call__(self) -> Generator[dict, None, None]:
        while self.agent.sample_step < self.config["total_timesteps"]:
            self.buffer.reset()
            while not self.buffer.full:
                if not self.agent.sample_step < self.config["total_timesteps"]:
                    break
                yield self._run_collect()
            else:
                yield self._run_train()

        self.envs.close_extras()

    def _run_collect(self) -> dict[str, Any]:
        act, log_prob, val = self.agent.sample(self.obs)
        next_obs, reward, next_terminated, next_truncated, infos = self.envs.step(act)

        real_next_obs = next_obs.copy()
        if "final_observation" in infos.keys():
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    real_next_obs[idx] = final_obs
                    _, _, terminal_value = self.agent.sample(np.expand_dims(real_next_obs[idx], axis=0))
                    reward[idx] += self.config["gamma"] * (1 - next_terminated[idx]) * terminal_value

        self.buffer.add(self.obs, act, reward, self.terminated, val, log_prob)
        if self.buffer.full:
            _, _, next_val = self.agent.sample(real_next_obs)
            self.buffer.compute_returns_and_advantage(next_val, next_terminated)

        self.obs = next_obs
        self.terminated = next_terminated

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

    def _run_train(self) -> dict[str, Any]:
        data_generator_list = [
            self.buffer.get(batch_size=self.config["minibatch_size"]) for _ in range(self.config["update_epochs"])
        ]

        log_data = self.agent.learn(data_generator_list)[0]

        return {"log_type": "train", "sample_step": self.agent.sample_step, "logs": log_data}

    def _make_env(self, idx: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(self.config["env_id"], render_mode="rgb_array")
            env.observation_space.dtype = np.float32  # type: ignore[assignment]
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.config["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.config['run_name']}")
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=self.config["gamma"])
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            env.action_space.seed(self.config["seed"] + idx)
            env.observation_space.seed(self.config["seed"] + idx)
            return env

        return thunk


class Logger:
    @dataclasses.dataclass
    class Config:
        track: bool = False
        wandb_project_name: str = "abcdrl"
        wandb_tags: List[str] = dataclasses.field(default_factory=lambda: [])
        wandb_entity: Optional[str] = None

    @classmethod
    def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
        import wandb
        from torch.utils.tensorboard import SummaryWriter

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            if config.track:
                wandb.init(
                    project=config.wandb_project_name,
                    tags=config.wandb_tags,
                    entity=config.wandb_entity,
                    sync_tensorboard=True,
                    config=instance.config,
                    name=instance.config["run_name"],
                    monitor_gym=True,
                    save_code=True,
                )

            writer = SummaryWriter(f"runs/{instance.config['run_name']}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in instance.config.items()]),
            )

            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if "logs" in log_data:
                    for log_item in log_data["logs"].items():
                        writer.add_scalar(f"{log_data['log_type']}/{log_item[0]}", log_item[1], log_data["sample_step"])
                yield log_data

        return wrapper


if __name__ == "__main__":
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def main(trainer: Trainer.Config, logger: Logger.Config) -> None:
        Trainer.__call__ = Logger.decorator(logger)(Trainer.__call__)  # type: ignore[assignment]
        for log_data in Trainer(trainer)():
            if "logs" in log_data and log_data["log_type"] != "train":
                print(log_data)

    tyro.cli(main)
