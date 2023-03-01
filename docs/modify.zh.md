# 修改 🖌

!!! info
    建议在修改前阅读[模块设计页面](abstractions.zh.md)。

## 参数

我们的代码使用 [brentyi/tyro](https://github.com/brentyi/tyro) 管理参数，为便于大家理解 `tyro` 做了什么，我们下面给出使用 [argparse](https://docs.python.org/3/library/argparse.html) 的等价代码。

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

## 修改算法

我们的算法完整实现在单个文件中，直接修改 `Model📦`, `Algorithm👣`, `Agent🤖`, `Trainer🔁` 四个类即可。

我们的模块化设计没有规定严格的接口，你可以随意修改这四个类，只要它可以工作。若要使用我们提供的功能（例如：logger，模型保存，模型评估）需要维持 `Trainer🔁` 的接口不变。

## 修改功能

### 编写装饰器

我们的通用功能主要通过装饰器实现，可以参考以下代码和 `abcdrl/utils/*.py` 文件，实现你想要的新功能并应用到所有算法上。

```python hl_lines="4-5 11 15 18"
class Example:
    @dataclasses.dataclass
    class Config:
        # 添加额外的参数
        new_arg: int = 1

    @classmethod
    def decorator(cls, config: Config = Config()) -> Callable[..., Generator[dict[str, Any], None, None]]:
        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            # 初始化 Trainer 后，运行算法前
            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if "logs" in log_data and log_data["log_type"] != "train":
                    # 在这里处理 log_data 和调整控制流
                    print(config.new_arg) # 使用 config
                    yield log_data # 算法运行的每步
            # 运行结束后
        return _wrapper
```

### 使用装饰器

```python hl_lines="1-16 29-32"
# 第一步：复制需要的装饰器
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

    # 第二步：在主函数中添加装饰器参数
    def main(trainer: Trainer.Config, example: Example.Config) -> None:
        # 第三步：对 Trainer.__call__ 函数装饰
        Trainer.__call__ = Example.decorator(example)(Trainer.__call__)  # type: ignore[assignment]
        for log_data in Trainer(trainer)():
            if "logs" in log_data and log_data["log_type"] != "train":
                print(log_data)

    tyro.cli(main)
```
