from __future__ import annotations

import dataclasses
import os
import random
import time
from typing import Any, Callable, Generator, Generic, TypeVar

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from combine_signatures.combine_signatures import combine_signatures
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
        self, obs: torch.Tensor, act: torch.Tensor | None = None
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

    def learn(self, data_generator: Generator[RolloutBuffer.Samples[torch.Tensor], None, None]) -> dict[str, Any]:
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
        obs_ts = torch.as_tensor(obs, device=self.kwargs["device"])
        with torch.no_grad():
            act_ts, _, _ = self.alg.predict(obs_ts)
        act_np = act_ts.cpu().numpy()
        return act_np

    def sample(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_ts = torch.as_tensor(obs, device=self.kwargs["device"])
        with torch.no_grad():
            act_ts, log_prob_ts, val_ts = self.alg.predict(obs_ts)
        act_np, log_prob_np, val_np = act_ts.cpu().numpy(), log_prob_ts.cpu().numpy(), val_ts.cpu().numpy()
        self.sample_step += self.kwargs["num_envs"]
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
                        item[0]: torch.as_tensor(item[1], device=self.kwargs["device"])
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
        if self.kwargs["anneal_lr"]:
            frac = 1.0 - (self.learn_step - 1.0) / (self.kwargs["total_timesteps"] // self.kwargs["batch_size"])
            lrnow = frac * self.kwargs["learning_rate"]
            self.alg.optimizer.param_groups[0]["lr"] = lrnow


class Trainer:
    def __init__(
        self,
        exp_name: str | None = None,
        seed: int = 1,
        cuda: bool = True,
        capture_video: bool = False,
        env_id: str = "Hopper-v4",
        num_envs: int = 1,
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
        target_kl: float | None = None,
        # Train
        num_minibatches: int = 32,
        gae_lambda: float = 0.95,
    ) -> None:
        self.kwargs = locals()
        self.kwargs.pop("self")

        if self.kwargs["exp_name"] is None:
            self.kwargs["exp_name"] = f"{self.kwargs['env_id']}__{os.path.basename(__file__).rstrip('.py')}"
        self.kwargs["batch_size"] = self.kwargs["num_envs"] * self.kwargs["num_steps"]
        self.kwargs["minibatch_size"] = self.kwargs["batch_size"] // self.kwargs["num_minibatches"]
        self.kwargs["device"] = "cuda" if self.kwargs["cuda"] and torch.cuda.is_available() else "cpu"

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.kwargs["num_envs"])])  # type: ignore[arg-type]
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

        self.obs, _ = self.envs.reset(seed=self.kwargs["seed"])
        self.terminated = np.zeros((self.kwargs["num_envs"],), dtype=np.float32)

        self.agent = Agent(**self.kwargs)

    def __call__(self) -> Generator[dict, None, None]:
        while self.agent.sample_step < self.kwargs["total_timesteps"]:
            self.buffer.reset()
            while not self.buffer.full:
                if not self.agent.sample_step < self.kwargs["total_timesteps"]:
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
                    reward[idx] += self.kwargs["gamma"] * (1 - next_terminated[idx]) * terminal_value

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
            self.buffer.get(batch_size=self.kwargs["minibatch_size"]) for _ in range(self.kwargs["update_epochs"])
        ]

        log_data = self.agent.learn(data_generator_list)[0]

        return {"log_type": "train", "sample_step": self.agent.sample_step, "logs": log_data}

    def _make_env(self, idx: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(self.kwargs["env_id"], render_mode="rgb_array")
            env.observation_space.dtype = np.float32  # type: ignore[assignment]
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.kwargs["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.kwargs['exp_name']}")
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=self.kwargs["gamma"])
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            env.action_space.seed(self.kwargs["seed"] + idx)
            env.observation_space.seed(self.kwargs["seed"] + idx)
            return env

        return thunk


def wrapper_logger_torch(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    import wandb
    from torch.utils.tensorboard import SummaryWriter

    def setup_video_monitor() -> None:
        vcr = gym.wrappers.monitoring.video_recorder.VideoRecorder
        vcr.close_ = vcr.close  # type: ignore[attr-defined]

        def close(self):
            vcr.close_(self)
            if self.path:
                wandb.log({"videos": wandb.Video(self.path)})
                self.path = None

        vcr.close = close  # type: ignore[assignment]

    @combine_signatures(wrapped)
    def _wrapper(
        *args,
        track: bool = False,
        wandb_project_name: str = "abcdrl",
        wandb_tags: list[str] = [],
        wandb_entity: str | None = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        instance = args[0]
        exp_name_ = f"{instance.kwargs['exp_name']}__{instance.kwargs['seed']}__{int(time.time())}"
        if track:
            wandb.init(
                project=wandb_project_name,
                tags=wandb_tags,
                entity=wandb_entity,
                sync_tensorboard=True,
                config=instance.kwargs,
                name=exp_name_,
                save_code=True,
            )
            setup_video_monitor()

        writer = SummaryWriter(f"runs/{exp_name_}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in instance.kwargs.items()]),
        )

        gen = wrapped(*args, **kwargs)
        for log_data in gen:
            if "logs" in log_data:
                for log_item in log_data["logs"].items():
                    writer.add_scalar(f"{log_data['log_type']}/{log_item[0]}", log_item[1], log_data["sample_step"])
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

    Trainer.__call__ = wrapper_logger_torch(Trainer.__call__)  # type: ignore[assignment]
    fire.Fire(
        Trainer,
        serialize=lambda gen: (log_data for log_data in gen if "logs" in log_data and log_data["log_type"] != "train"),
    )
