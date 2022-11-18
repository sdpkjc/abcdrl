# 模块设计 🏗

每个算法以 `Model📦`, `Algorithm👣`, `Agent🤖`, `Trainer🔁` 四个类为主组成，并以组合的方式进行交互。

- `Model📦`：定义单个或多个前向网络；输入是环境状态，输出是网络的原始输出。
- `Algorithm👣`：定义 `Model📦` 的更新算法和 `Model📦` 输出的后处理（`argmax`, ...）。
- `Agent🤖`：定义 `Algorithm👣` 与环境交互的接口和训练数据的预处理。
- `Trainer🔁`：定义 `Agent🤖` 的整体训练流程和辅助训练的工具（`Buffer`, ...）。

调用 `Trainer.__call__` 函数将得到一个生成器📽，该生成器保存了训练流程和所有相关数据。生成器每步返回一个 `log_data` 训练日志📒，持续调用该生成器即可完成训练并得到所有 `log_data`。

`logger📊` 部分使用 [Tensorboard](https://www.tensorflow.org/tensorboard) 和 [Weights & Biases](https://wandb.ai/) 记录训练日志。对 `Trainer.__call__` 函数进行装饰，具体实现见核心代码。

---

<figure markdown>
  ![Adam](imgs/adam.svg){ width="500" }
  <figcaption>Adam</figcaption>
</figure>

!!! note
    🧵 实线表示函数调用关系；虚线表示数据流向。

<figure markdown>
  ![abstractions_img](/imgs/abstractions.png)
  <figcaption>模型结构图</figcaption>
</figure>

---

```python title="abstractions.py"
class Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        pass

    def value(self, x: torch.Tensor, a: Optional[torch.Tensor] = None) -> tuple[Any]:
        # 返回 单个 或 多个 critic 的输出值
        pass

    def action(self, x: torch.Tensor) -> tuple[Any]:
        # 返回 动作 | 动作概率分布
        pass


class Algorithm:
    def __init__(self, **kwargs) -> None:
        self.model = Model(**kwargs)
        # 1. 初始化 model, target_model
        # 2. 初始化 optimizer
        pass

    def predict(self, obs: torch.Tensor) -> tuple[Any]:
        # 返回 动作 | 动作概率分布 | Q函数的预估值
        pass

    def learn(self, data: BufferSamples) -> dict[str, Any]:
        # 根据训练数据（观测量和输入的reward），定义损失函数，用于更新 Model 中的参数。

        # 1. 计算目标
        # 2. 计算损失
        # 3. 优化模型
        # 4. 返回训练信息
        pass

    def sync_target(self) -> None:
        # 同步 model 和 target_model
        pass


class Agent:
    def __init__(self, **kwargs) -> None:
        self.alg = Algorithm(**kwargs)
        # 1. 初始化 Algorithm
        # 2. 初始化 运行步数变量
        pass

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # 1. obs 预处理 to_tensor & to_device
        # 2. Algorithm.predict 得到 act
        # 3. act 后处理 to_numpy & to_cpu
        # 4. 返回评估使用的 act
        pass

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 1. obs 预处理 to_tensor & to_device
        # 2. Algorithm.predict 得到 act
        # 3. act 后处理 to_numpy & to_cpu
        # 4. 返回训练使用的 act
        pass

    def learn(self, data: BufferSamples) -> dict[str, Any]:
        # 数据预处理
        # 调用 Algorithm.learn
        # 返回 Algorithm.learn 的返回值
        pass


class Trainer:
    def __init__(self, **kwargs) -> None:
        self.agent = Agent(**kwargs)
        # 1. 初始化参数
        # 2. 初始化训练和评估环境
        # 3. 初始化 Buffer
        # 4. 初始化 Agent
        pass

    def __call__(self) -> Generator[dict[str, Any], None, None]:
        # 1. 规定训练流程
        # 2. 返回一个生成器，生成器每步返回一个 log_data 字典
        pass

    def _run_collect(self) -> dict[str, Any]:
        # 1. 采样一步，并加入到 Buffer 中
        # 2. 返回 log_data 字典
        pass

    def _run_train(self) -> dict[str, Any]:
        # 1. 从 Buffer 取出一组训练数据
        # 2. 训练单步
        # 3. 返回 log_data 字典
        pass


if __name__ == "__main__":
    trainer = Trainer()
    for log_data in trainer():
        print(log_data)
```
