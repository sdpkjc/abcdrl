from __future__ import annotations

import copy
import dataclasses
import operator
import os
import random
import time
from typing import Any, Callable, Generator, Generic, TypeVar

import fire
import gymnasium as gym
import numpy as np
import tensorflow as tf
from combine_signatures.combine_signatures import combine_signatures
from tensorflow.keras import layers, losses, models, optimizers

SamplesItemType = TypeVar("SamplesItemType", tf.Tensor, np.ndarray)


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


class PrioritizedReplayBuffer:
    @dataclasses.dataclass(frozen=True)
    class Samples(Generic[SamplesItemType]):
        observations: SamplesItemType
        actions: SamplesItemType
        next_observations: SamplesItemType
        dones: SamplesItemType
        rewards: SamplesItemType
        weights: SamplesItemType
        indices: list[int]

    class SegmentTree:
        def __init__(self, capacity: int, operation: Callable[[float, float], float], init_value: float) -> None:
            assert capacity > 0 and capacity % 2 == 0

            self.capacity = capacity
            self.operation = operation
            self.init_value = init_value
            self.tree = [init_value for _ in range(2 * capacity)]

        def query(self, start: int, end: int) -> float:
            return self._query(start, end, 1, 0, self.capacity)

        def _query(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
            if start <= node_start and node_end <= end:
                return self.tree[node]

            mid = (node_start + node_end) // 2
            return self.operation(
                self._query(start, end, 2 * node, node_start, mid) if start <= mid else self.init_value,
                self._query(start, end, 2 * node + 1, mid + 1, node_end) if end > mid else self.init_value,
            )

        def retrieve(self, upperbound: float) -> int:
            assert self.operation == operator.add
            assert 0 <= upperbound <= self.query(0, self.capacity - 1) + 1e-5

            idx = 1
            while idx < self.capacity:
                idx *= 2
                if self.tree[idx] <= upperbound:
                    upperbound -= self.tree[idx]
                    idx += 1

            return idx - self.capacity

        def __setitem__(self, idx: int, val: float) -> None:
            idx += self.capacity
            self.tree[idx] = val

            while idx // 2:
                idx //= 2
                self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])

        def __getitem__(self, idx: int) -> float:
            return self.tree[self.capacity + idx]

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        buffer_size: int = 1_000_0,
        alpha: float = 0.2,
    ) -> None:
        assert alpha >= 0

        self.obs_buf = np.zeros((buffer_size,) + get_space_shape(obs_space), dtype=obs_space.dtype)
        self.next_obs_buf = np.zeros((buffer_size,) + get_space_shape(obs_space), dtype=obs_space.dtype)
        self.acts_buf = np.zeros((buffer_size,) + get_space_shape(act_space), dtype=act_space.dtype)
        self.rews_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.dones_buf = np.zeros((buffer_size,), dtype=np.float32)

        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = self.SegmentTree(tree_capacity, operator.add, 0.0)
        self.min_tree = self.SegmentTree(tree_capacity, min, float("inf"))

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        done: np.ndarray,
        infos: dict[str, Any],
    ) -> None:
        for obs_i, next_obs_i, act_i, rew_i, done_i in zip(obs, next_obs, act, rew, done):  # type: ignore[call-overload]
            self.obs_buf[self.ptr] = np.array(obs_i).copy()
            self.next_obs_buf[self.ptr] = np.array(next_obs_i).copy()
            self.acts_buf[self.ptr] = np.array(act_i).copy()
            self.rews_buf[self.ptr] = np.array(rew_i).copy()
            self.dones_buf[self.ptr] = np.array(done_i).copy()
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

            self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

    def sample(self, batch_size: int = 1, beta: float = 0.6) -> Samples:
        assert len(self) >= batch_size
        assert beta > 0

        indices = self._sample_proportional(batch_size)

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.dones_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices], dtype=np.float32).reshape(-1, 1)

        return PrioritizedReplayBuffer.Samples(
            observations=obs,
            next_observations=next_obs,
            actions=acts,
            rewards=rews,
            dones=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: list[int], priorities: np.ndarray) -> None:
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.size

    def _sample_proportional(self, batch_size: int = 1) -> list[int]:
        indices = []
        p_total = self.sum_tree.query(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            upperbound = random.uniform(segment * i, segment * (i + 1))
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float) -> float:
        p_min = self.min_tree.query(0, len(self) - 1) / self.sum_tree.query(0, len(self) - 1)
        max_weight = (p_min * len(self)) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.query(0, len(self) - 1)
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        return float(weight)


class Network(models.Model):
    def __init__(self, out_n: int, name: str = "q_network", **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_dense_0 = layers.Dense(120, activation="relu")
        self.layer_dense_1 = layers.Dense(84, activation="relu")
        self.layer_output = layers.Dense(out_n)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.layer_dense_0(x)
        x = self.layer_dense_1(x)
        x = self.layer_output(x)
        return x


class Model:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        # input shape: int(np.prod(get_space_shape(self.kwargs["obs_space"])))
        self.network = Network(
            self.kwargs["act_space"].n,
        )

    def value(self, obs: tf.Tensor) -> tf.Tensor:
        return self.network(obs)


class Algorithm:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        self.model = Model(**self.kwargs)
        self.model_t = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam(self.kwargs["learning_rate"])
        self.loss_func = losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        model_init_obs = tf.convert_to_tensor(np.array([self.kwargs["obs_space"].sample()]))
        self.model.value(model_init_obs)
        self.model_t.value(model_init_obs)

    def predict(self, obs: tf.Tensor) -> tf.Tensor:
        val = self.model.value(obs)
        return val

    def learn(self, data: PrioritizedReplayBuffer.Samples[tf.Tensor]) -> dict[str, Any]:
        target_max = tf.math.reduce_max(self.model_t.value(data.next_observations), axis=1)
        td_target = data.rewards + self.kwargs["gamma"] * target_max * (1 - data.dones)

        with tf.GradientTape() as tape:
            old_val = tf.squeeze(
                tf.gather(
                    self.model.value(data.observations),
                    indices=tf.squeeze(data.actions),
                    batch_dims=tf.rank(tf.squeeze(data.actions)),
                    axis=1,
                )
            )
            elementwise_td_loss = self.loss_func(tf.expand_dims(td_target, -1), tf.expand_dims(old_val, -1))
            td_loss = tf.math.reduce_mean(elementwise_td_loss * data.weights)  # PER

        grads = tape.gradient(td_loss, self.model.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.network.trainable_variables))

        log_data = {"td_loss": td_loss, "elementwise_td_loss": elementwise_td_loss, "q_value": tf.reduce_mean(old_val)}
        return log_data

    def sync_target(self) -> None:
        self.model_t.network.set_weights(self.model.network.get_weights())


class Agent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        self.alg = Algorithm(**self.kwargs)
        self.sample_step = 0
        self.learn_step = 0

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_ts = tf.convert_to_tensor(obs)
        act_ts = tf.math.argmax(self.alg.predict(obs_ts), axis=1)
        act_np = act_ts.numpy()
        return act_np

    def sample(self, obs: np.ndarray) -> np.ndarray:
        if random.random() < self._get_epsilon():
            act_np = np.array([self.kwargs["act_space"].sample() for _ in range(self.kwargs["num_envs"])])
        else:
            obs_ts = tf.convert_to_tensor(obs)
            act_ts = tf.math.argmax(self.alg.predict(obs_ts), axis=1)
            act_np = act_ts.numpy()

        self.sample_step += self.kwargs["num_envs"]
        if self.sample_step % self.kwargs["target_network_frequency"] == 0:
            self.alg.sync_target()
        return act_np

    def learn(self, data: PrioritizedReplayBuffer.Samples[np.ndarray]) -> dict[str, Any]:
        data_ts = PrioritizedReplayBuffer.Samples[tf.Tensor](
            **{
                item[0]: tf.convert_to_tensor(item[1]) if isinstance(item[1], np.ndarray) else item[1]
                for item in dataclasses.asdict(data).items()
            }
        )

        log_data = self.alg.learn(data_ts)
        self.learn_step += 1
        log_data["epsilon"] = self._get_epsilon()
        return log_data

    def _get_epsilon(self) -> float:
        slope = (self.kwargs["end_epsilon"] - self.kwargs["start_epsilon"]) * (
            self.sample_step / (self.kwargs["exploration_fraction"] * self.kwargs["total_timesteps"])
        ) + self.kwargs["start_epsilon"]
        return max(slope, self.kwargs["end_epsilon"])


class Trainer:
    def __init__(
        self,
        exp_name: str | None = None,
        seed: int = 1,
        cuda: bool = True,
        capture_video: bool = False,
        env_id: str = "CartPole-v1",
        num_envs: int = 1,
        total_timesteps: int = 500_000,
        gamma: float = 0.99,
        # Collect
        alpha: float = 0.2,
        beta: float = 0.6,
        buffer_size: int = 10_000,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.05,
        exploration_fraction: float = 0.5,
        # Learn
        batch_size: int = 128,
        learning_rate: float = 2.5e-4,
        # Train
        learning_starts: int = 10_000,
        target_network_frequency: int = 500,
        train_frequency: int = 10,
    ) -> None:
        self.kwargs = locals()
        self.kwargs.pop("self")

        if self.kwargs["exp_name"] is None:
            self.kwargs["exp_name"] = f"{self.kwargs['env_id']}__{os.path.basename(__file__).rstrip('.py')}"
        self.kwargs["run_name"] = f"{self.kwargs['exp_name']}__{self.kwargs['seed']}__{int(time.time())}"
        self.kwargs["target_network_frequency"] = max(
            self.kwargs["target_network_frequency"] // self.kwargs["num_envs"] * self.kwargs["num_envs"], 1
        )
        if not self.kwargs["cuda"]:
            tf.config.experimental.set_visible_devices([], "GPU")

        self.envs = gym.vector.SyncVectorEnv([self._make_env(i) for i in range(self.kwargs["num_envs"])])  # type: ignore[arg-type]
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete)

        self.kwargs["obs_space"] = self.envs.single_observation_space
        self.kwargs["act_space"] = self.envs.single_action_space

        self.buffer = PrioritizedReplayBuffer(
            self.kwargs["obs_space"],
            self.kwargs["act_space"],
            buffer_size=self.kwargs["buffer_size"],
            alpha=self.kwargs["alpha"],
        )

        self.obs, _ = self.envs.reset(seed=self.kwargs["seed"])
        self.agent = Agent(**self.kwargs)

    def __call__(self) -> Generator[dict[str, Any], None, None]:
        for _ in range(self.kwargs["learning_starts"]):
            yield self._run_collect()
        while self.agent.sample_step < self.kwargs["total_timesteps"]:
            for _ in range(self.kwargs["train_frequency"]):
                if not self.agent.sample_step < self.kwargs["total_timesteps"]:
                    break
                yield self._run_collect()
            yield self._run_train()

        self.envs.close_extras()

    def _run_collect(self) -> dict[str, Any]:
        act = self.agent.sample(self.obs)
        next_obs, reward, terminated, truncated, infos = self.envs.step(act)

        real_next_obs = next_obs.copy()
        if "final_observation" in infos.keys():
            for idx, final_obs in enumerate(infos["final_observation"]):
                real_next_obs[idx] = real_next_obs[idx] if final_obs is None else final_obs

        self.buffer.add(self.obs, real_next_obs, act, reward, terminated, infos)
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

    def _run_train(self) -> dict[str, Any]:
        data = self.buffer.sample(batch_size=self.kwargs["batch_size"])
        log_data = self.agent.learn(data)

        loss_for_prior = log_data["elementwise_td_loss"].numpy() + 1e-6
        self.buffer.update_priorities(data.indices, loss_for_prior)
        log_data.pop("elementwise_td_loss")

        return {"log_type": "train", "sample_step": self.agent.sample_step, "logs": log_data}

    def _make_env(self, idx: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(self.kwargs["env_id"], render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.kwargs["capture_video"]:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.kwargs['run_name']}")
            env.action_space.seed(self.kwargs["seed"] + idx)
            env.observation_space.seed(self.kwargs["seed"] + idx)
            return env

        return thunk


def wrapper_logger_tf(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    import wandb

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
        if track:
            wandb.init(
                project=wandb_project_name,
                tags=wandb_tags,
                entity=wandb_entity,
                sync_tensorboard=True,
                config=instance.kwargs,
                name=instance.kwargs["run_name"],
                save_code=True,
            )
            setup_video_monitor()

        writer = tf.summary.create_file_writer(f"runs/{instance.kwargs['run_name']}")
        with writer.as_default():
            tf.summary.text(
                "hyperparameters",
                "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in instance.kwargs.items()]),
                0,
            )

            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if "logs" in log_data:
                    for log_item in log_data["logs"].items():
                        tf.summary.scalar(f"{log_data['log_type']}/{log_item[0]}", log_item[1], log_data["sample_step"])
                yield log_data

    return _wrapper


if __name__ == "__main__":
    SEED = 1234
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    Trainer.__call__ = wrapper_logger_tf(Trainer.__call__)  # type: ignore[assignment]
    fire.Fire(
        Trainer,
        serialize=lambda gen: (log_data for log_data in gen if "logs" in log_data and log_data["log_type"] != "train"),
    )
