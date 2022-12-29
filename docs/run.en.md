# Run 🏃

After the dependency is installed, you can run the algorithm file directly.

```bash
python abcdrl/dqn.py \
    --env-id Cartpole-v1 \
    --device "cuda:1" \ #(1)!
    --total_timesteps 500000 \ #(2)!
    --gamma 0.99 \
    --learning-rate 2.5e-4 \ #(3)!
    --capture-video True \
    --track \ #(4)!
    --wandb-project-name 'abcdrl' \
    --wandb-tags "['tag1', 'tag2']"
```

1.  or `--device cuda:1`
2.  The connector can use `_` or `-`
3.  or `0.00025`
4.  or `--track True`

Parameters in the algorithm file, consisting of two parts. The first part is the initialization parameters of `Trainer`, and the second part is the parameters of the external feature (`logger`).

=== "Algorithm Parameters"

    ```python title="dqn.py" linenums="206" hl_lines="4-11 13-16 18-19 21-23"
    class Trainer:
        def __init__(
            self,
            exp_name: str | None = None,
            seed: int = 1,
            device: str | torch.device = "auto",
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

    ```python title="dqn.py" linenums="312" hl_lines="21-24"
    def wrapper_logger(
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
    ```

!!! note
    You can use the `python abcdrl/dqn.py --help` command to view algorithm parameters and the `python abcdrl/dqn.py __call__ --help` command to view external features parameters.
