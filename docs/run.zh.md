# è¿è¡ ð

å®è£å¥½ææä¾èµåï¼å³å¯ç´æ¥è¿è¡ç®æ³æä»¶ã

```shell
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

1.  æ `--device cuda:1`
2.  è¿æ¥ç¬¦å¯ä»¥ä½¿ç¨ `_` æ `-`
3.  æ `0.00025`
4.  æ `--track True`

ç®æ³æä»¶ä¸­çåæ°ï¼ç±ä¸¤é¨åç»æãç¬¬ä¸é¨åæ¯ç®æ³ä¸»ä½ `Trainerð` çåæ°ï¼ç¬¬äºé¨åæ¯åè½ï¼`logger`, ...ï¼çåæ°ã

=== "ç®æ³åæ°"

    ```python title="abcdrl/dqn.py" linenums="206" hl_lines="4-11 13-16 18-19 21-23"
    class Trainer:
        def __init__(
            self,
            exp_name: str | None = None,
            seed: int = 1,
            device: str | torch.device = "auto",
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

=== "åè½åæ°"

    ```python title="abcdrl/dqn.py" linenums="312" hl_lines="21-24"
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
    å¯ä½¿ç¨ `python abcdrl/dqn.py --help` å½ä»¤æ¥çç®æ³åæ°ï¼ä½¿ç¨ `python abcdrl/dqn.py __call__ --help` å½ä»¤æ¥çåè½åæ°ã
