# 运行 🏃

安装好所有依赖后，即可直接运行算法文件。

```shell
python abcdrl/dqn_torch.py \
    --trainer.env-id Cartpole-v1 \
    --trainer.total_timesteps 500000 \ #(1)!
    --trainer.gamma 0.99 \
    --trainer.learning-rate 2.5e-4 \ #(2)!
    --trainer.capture-video \
    --logger.track \
    --logger.wandb-project-name abcdrl \
    --logger.wandb-tags tag1 tag2
```

1.  连接符可以使用 `_` 或 `-`
2.  或 `0.00025`

!!! example "指定 GPU 设备"
    - 使用 `gpu:0` 和 `gpu:1` 👇
        - `CUDA_VISIBLE_DEVICES="0,1" python abcdrl/dqn_torch.py`
    - 使用 `gpu:1` 👇
        - `CUDA_VISIBLE_DEVICES="1" python abcdrl/dqn_torch.py`
    - 仅使用 `cpu` 👇
        - `python abcdrl/dqn_torch.py --no-cuda`
        - `CUDA_VISIBLE_DEVICES="" python abcdrl/dqn_torch.py`
        - `CUDA_VISIBLE_DEVICES="-1" python abcdrl/dqn_torch.py`

算法文件中的参数，由两部分组成。第一部分是算法主体 `Trainer🔁` 的参数，第二部分是功能（`Logger📊`, ...）的参数。

=== "算法参数"

    ```python title="abcdrl/dqn_torch.py" linenums="205" hl_lines="4-11 13-16 18-19 21-23"
    class Trainer:
        @dataclasses.dataclass
        class Config:
            exp_name: Optional[str] = None
            seed: int = 1
            cuda: bool = True
            capture_video: bool = False
            env_id: str = "CartPole-v1"
            num_envs: int = 1
            total_timesteps: int = 500_000
            gamma: float = 0.99
            # Collect
            buffer_size: int = 10_000
            start_epsilon: float = 1.0
            end_epsilon: float = 0.05
            exploration_fraction: float = 0.5
            # Learn
            batch_size: int = 128
            learning_rate: float = 2.5e-4
            # Train
            learning_starts: int = 10_000
            target_network_frequency: int = 500
            train_frequency: int = 10

        def __init__(self, config: Config = Config()) -> None:
            # ...
    ```

=== "功能参数"

    ```python title="abcdrl/dqn_torch.py" linenums="310" hl_lines="4-7"
    class Logger:
        @dataclasses.dataclass
        class Config:
            track: bool = False
            wandb_project_name: str = "abcdrl"
            wandb_tags: List[str] = dataclasses.field(default_factory=lambda: [])
            wandb_entity: Optional[str] = None

        @classmethod
        def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
            # ...
    ```

!!! note
    可使用 `python abcdrl/dqn_torch.py -h` 命令查看算法参数和功能参数。

    ```shell title="python abcdrl/dqn_torch.py -h"
    usage: dqn_torch.py [-h] [--trainer.exp-name {None}|STR] [--trainer.seed INT]
                        [--trainer.no-cuda] [--trainer.capture-video]
                        [--trainer.env-id STR] [--trainer.num-envs INT]
                        [--trainer.total-timesteps INT] [--trainer.gamma FLOAT]
                        [--trainer.buffer-size INT]
                        [--trainer.start-epsilon FLOAT]
                        [--trainer.end-epsilon FLOAT]
                        [--trainer.exploration-fraction FLOAT]
                        [--trainer.batch-size INT] [--trainer.learning-rate FLOAT]
                        [--trainer.learning-starts INT]
                        [--trainer.target-network-frequency INT]
                        [--trainer.train-frequency INT] [--logger.track]
                        [--logger.wandb-project-name STR]
                        [--logger.wandb-tags STR [STR ...]]
                        [--logger.wandb-entity {None}|STR]

    ╭─ arguments ─────────────────────────────────────────────╮
    │ -h, --help              show this help message and exit │
    ╰─────────────────────────────────────────────────────────╯
    ╭─ trainer arguments ─────────────────────────────────────╮
    │ --trainer.exp-name {None}|STR                           │
    │                         (default: None)                 │
    │ --trainer.seed INT      (default: 1)                    │
    │ --trainer.no-cuda       (sets: cuda=False)              │
    │ --trainer.capture-video                                 │
    │                         (sets: capture_video=True)      │
    │ --trainer.env-id STR    (default: CartPole-v1)          │
    │ --trainer.num-envs INT  (default: 1)                    │
    │ --trainer.total-timesteps INT                           │
    │                         (default: 500000)               │
    │ --trainer.gamma FLOAT   (default: 0.99)                 │
    │ --trainer.buffer-size INT                               │
    │                         Collect (default: 10000)        │
    │ --trainer.start-epsilon FLOAT                           │
    │                         Collect (default: 1.0)          │
    │ --trainer.end-epsilon FLOAT                             │
    │                         Collect (default: 0.05)         │
    │ --trainer.exploration-fraction FLOAT                    │
    │                         Collect (default: 0.5)          │
    │ --trainer.batch-size INT                                │
    │                         Learn (default: 128)            │
    │ --trainer.learning-rate FLOAT                           │
    │                         Learn (default: 0.00025)        │
    │ --trainer.learning-starts INT                           │
    │                         Train (default: 10000)          │
    │ --trainer.target-network-frequency INT                  │
    │                         Train (default: 500)            │
    │ --trainer.train-frequency INT                           │
    │                         Train (default: 10)             │
    ╰─────────────────────────────────────────────────────────╯
    ╭─ logger arguments ──────────────────────────────────────╮
    │ --logger.track          (sets: track=True)              │
    │ --logger.wandb-project-name STR                         │
    │                         (default: abcdrl)               │
    │ --logger.wandb-tags STR [STR ...]                       │
    │                         (default: )                     │
    │ --logger.wandb-entity {None}|STR                        │
    │                         (default: None)                 │
    ╰─────────────────────────────────────────────────────────╯
    ```
