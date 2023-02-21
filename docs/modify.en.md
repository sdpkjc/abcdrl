# Modify 🖌

!!! info
    It is advised to read [the abstractions page](abstractions.en.md) before making changes.

## Parameters & Loop

Our code uses [google/python-fire](https://github.com/google/python-fire) to manage parameters and repeatedly call the algorithm interface. To help you understand what fire does, here's an equivalent implementation using [argparse](https://docs.python.org/3/library/argparse.html).

=== "python-fire"

    ```python
    if __name__ == "__main__":
        # ...
        fire.Fire(
            Trainer,
            serialize=lambda gen: (log_data for log_data in gen if "logs" in log_data and log_data["log_type"] != "train"),
        )
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

The generic feature is implemented as a decorator, you can refer to the code below and `abcdrl/utils/wrapper_*.py` file to implement the new feature you want and apply it to all algorithms.

```python hl_lines="8-9 13 15"
from combine_signatures.combine_signatures import combine_signatures


def wrapper_example(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    @combine_signatures(wrapped)
    def _wrapper(*args, new_arg: int = 1, **kwargs) -> Generator[dict[str, Any], None, None]: # Add additional parameters
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

```python hl_lines="1-11 25-26"
# Step 1：Copy the decorators you need
def wrapper_example(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    @combine_signatures(wrapped)
    def _wrapper(*args, new_arg: int = 1, **kwargs) -> Generator[dict[str, Any], None, None]:
        gen = wrapped(*args, **kwargs)
        for log_data in gen:
            if "logs" in log_data and log_data["log_type"] != "train":
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

    Trainer.__call__ = wrapper_logger(Trainer.__call__)  # type: ignore[assignment]
    # Step 2：Decorate the Trainer.__call__ function
    Trainer.__call__ = wrapper_example(Trainer.__call__)  # type: ignore[assignment]
    fire.Fire(
        Trainer,
        serialize=lambda gen: (log_data for log_data in gen if "logs" in log_data and log_data["log_type"] != "train"),
    )
```
