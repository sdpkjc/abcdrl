# Modify 🖌

!!! info
    It is advised to read [the abstractions page](abstractions.en.md) before making changes.

## Parameters

Our code uses [brentyi/tyro](https://github.com/brentyi/tyro) to manage parameters. To help you understand what `tyro` does, here's an equivalent implementation using [argparse](https://docs.python.org/3/library/argparse.html).

=== "python-fire"

    ```python
    class Trainer:
        @dataclasses.dataclass
        class Config:
            exp_name: Optional[str] = None
            seed: int = 1
            # ...
        # ...


    if __name__ == "__main__":
        # ...
        def main(trainer: Trainer.Config) -> None:
            for log_data in Trainer(trainer)():
                if "logs" in log_data and log_data["log_type"] != "train":
                    print(log_data)

        tyro.cli(main)
    ```

=== "argparse"

    ```python
    def parse_args() -> argparse.Namespace:
        # fmt: off
        parser = argparse.ArgumentParser()
        parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
        parser.add_argument("--seed", type=int, default=1)
        # ...
        args = parser.parse_args()
        # fmt: on
        return args


    if __name__ == "__main__":
        # ...
        kwargs = vars(parse_args())
        trainer = Trainer(**kwargs)
        serialize = lambda gen: (log_data for log_data in gen if "logs" in log_data and log_data["log_type"] != "train")
        for log_data in serialize(trainer(**kwargs)):
            print(log_data)
    ```

## Modify Algorithm

Our Algorithm is completely implemented in a single file, and we can directly modify four classes: `Model📦`, `Algorithm👣`, `Agent🤖`, `Trainer🔁`.

Our modular design does not prescribe a strict interface, and you are free to modify these four classes as long as it works. To use the features we provided (e.g. logger, model saving, model evaluation), you need to keep the `Trainer🔁` interface.

## Modify Feature

### Writing Decorator

The generic feature is implemented as a decorator, you can refer to the code below and `abcdrl/utils/*.py` file to implement the new feature you want and apply it to all algorithms.

```python hl_lines="8-9 13 15"
class Example:
    @dataclasses.dataclass
    class Config:
        # Add additional parameters
        new_arg: int = 1

    @classmethod
    def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            # After initializing the Trainer, before running the algorithm
            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if "logs" in log_data and log_data["log_type"] != "train":
                    # Here, control flow is modified and log data is handled
                    yield log_data # Each step of the algorithm
            # After running the algorithm
        return _wrapper
```

### Using Decorator

```python hl_lines="1-16 29-32"
# Step 1: Copy the decorators you need
class Example:
    @dataclasses.dataclass
    class Config:
        new_arg: int = 1

    @classmethod
    def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if "logs" in log_data and log_data["log_type"] != "train":
                    print(config.new_arg)
                    yield log_data
        return _wrapper


if __name__ == "__main__":
    SEED=1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Step 2: Add decorator arguments to the main function
    def main(trainer: Trainer.Config, example: Example.Config) -> None:
        # Step 3: Decorate the Trainer.__call__ function
        Trainer.__call__ = Example.decorator(example)(Trainer.__call__)  # type: ignore[assignment]
        for log_data in Trainer(trainer)():
            if "logs" in log_data and log_data["log_type"] != "train":
                print(log_data)

    tyro.cli(main)
```
