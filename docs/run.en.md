# Run 🏃

After dependencies are installed, you can run the algorithm file directly.

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

1.  The connector can use `_` or `-`
2.  or `0.00025`
3.  or `--track True`

Parameters in the algorithm file, consisting of two parts. The first part is the initialization parameters of `Trainer🔁`, and the second part is the parameters of the feature (`logger`, ...).

=== "Algorithm Parameters"

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
            total_timesteps: int = 500_000,
            gamma: float = 0.99,
            # Collect
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
    ```

=== "Features Parameters"

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
    You can use the `python abcdrl/dqn_torch.py --help` command to view algorithm parameters and the `python abcdrl/dqn_torch.py __call__ --help` command to view features parameters.
