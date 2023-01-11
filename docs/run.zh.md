# 运行 🏃

安装好所有依赖后，即可直接运行算法文件。

```shell
python abcdrl/dqn_torch.py \
    --env-id Cartpole-v1 \
    --total_timesteps 500000 \ #(1)!
    --gamma 0.99 \
    --learning-rate 2.5e-4 \ #(2)!
    --capture-video True \
    --track \ #(3)!
    --wandb-project-name 'abcdrl' \
    --wandb-tags "['tag1', 'tag2']"
```

1.  连接符可以使用 `_` 或 `-`
2.  或 `0.00025`
3.  或 `--track True`

算法文件中的参数，由两部分组成。第一部分是算法主体 `Trainer🔁` 的参数，第二部分是功能（`logger`, ...）的参数。

=== "算法参数"

    ```python title="abcdrl/dqn_torch.py" linenums="205" hl_lines="4-11 13-16 18-19 21-23"
    class Trainer:
        def __init__(
            self,
            exp_name: str | None = None,
            seed: int = 1,
            cuda: bool = True,
            capture_video: bool = False,
            env_id: str = "CartPole-v1",
            num_envs: int = 1,
            total_timesteps: int = 5_000_00,
            gamma: float = 0.99,
            # Collect
            buffer_size: int = 1_000_0,
            start_epsilon: float = 1.0,
            end_epsilon: float = 0.05,
            exploration_fraction: float = 0.5,
            # Learn
            batch_size: int = 128,
            learning_rate: float = 2.5e-4,
            # Train
            learning_starts: int = 1_000_0,
            target_network_frequency: int = 500,
            train_frequency: int = 10,
        ) -> None:
    ```

=== "功能参数"

    ```python title="abcdrl/dqn_torch.py" linenums="310" hl_lines="22-25"
    def wrapper_logger(
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
    ```

!!! note
    可使用 `python abcdrl/dqn_torch.py --help` 命令查看算法参数，使用 `python abcdrl/dqn_torch.py __call__ --help` 命令查看功能参数。
