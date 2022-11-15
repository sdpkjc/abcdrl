# Abstractions

Each algorithm is mainly composed of four classes: `Model📦`, `Algorithm👣`, `Agent🤖`, `Trainer🔁` with HAS-A relationship.

- `Model📦`: Define a single or multiple forward networks. The input is the observations and the output is the original output of networks.
- `Algorithm👣`: Define the mechanism to update parameters in the `Model📦` and the post-processing of the output of `Model📦` (`argmax`, ...).
- `Agent🤖`: A data bridge between the `Environment🗺` and the `Algorithm👣`.
- `Trainer🔁`: Define the overall training process of `Agent🤖` and the tools to assist the training (`Buffe`,...).

The `Trainer.__call__` function returns a generator that holds the training control-flow and all related data. The generator returns a `log_data` training log at each step. Keep calling the generator to complete the training and get all the `log_data`.

The `logger📊` section uses [Tensorboard](https://www.tensorflow.org/tensorboard) and [Weights & Biases](https://wandb.ai/) to record training logs and decorates the `Trainer.__call__` function, see the core code for the specific implementation.

---

!!! note
    🧵 The solid line indicates the control-flow; The dotted line indicates the data-flow.

![abstractions_img](/imgs/abstractions.jpg)

---

```python title="abstractions.py"
class Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        pass

    def value(self, x: torch.Tensor, a: Optional[torch.Tensor] = None) -> Tuple[Any]:
        # Returns the output value of a single or multiple critics
        pass

    def action(self, x: torch.Tensor) -> Tuple[Any]:
        # Returns the action or action probability distribution
        pass


class Algorithm:
    def __init__(self, **kwargs) -> None:
        self.model = Model(**kwargs)
        # 1. Initialize model, target model
        # 2. Initialize optimizer
        pass

    def predict(self, obs: torch.Tensor) -> Tuple[Any]:
        # Returns an estimated value of an action or an action probability distribution or Q-function
        pass

    def learn(self, data: BufferSamples) -> dict[str, Any]:
        # Given the training data, it defines a loss function to update the parameters in the Model.

        # 1. Computing target
        # 2. Computing loss
        # 3. Update model
        # 4. Returns log_data of train
        pass

    def sync_target(self) -> None:
        # Synchronize model and target model
        pass


class Agent:
    def __init__(self, **kwargs) -> None:
        self.alg = Algorithm(**kwargs)
        # 1. Initialize Algorithm
        # 2. Initialize 运行步数变量
        pass

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # 1. obs pre-processing (to_tensor & to_device)
        # 2. act = Algorithm.predict
        # 3. act post-processing (to_numpy & to_cpu)
        # 4. Returns the act used for the evaluation
        pass

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 1. obs pre-processing (to_tensor & to_device)
        # 2. act = Algorithm.predict
        # 3. act post-processing (to_numpy & to_cpu & add noise)
        # 4. Returns the act used for training
        pass

    def learn(self, data: BufferSamples) -> dict[str, Any]:
        # Data pre-processing
        # Calling Algorithm.learn
        # Returns return of Algorithm.learn
        pass


class Trainer:
    def __init__(self, **kwargs) -> None:
        self.agent = Agent(**kwargs)
        # 1. Initialize args
        # 2. Initialize the training and evaluation environment
        # 3. Initialize Buffer
        # 4. Initialize Agent
        pass

    def __call__(self) -> Generator[dict[str, Any], None, None]:
        # 1. Define the training control-flow
        # 2. Returns a generator
        pass

    def _run_collect(self) -> dict[str, Any]:
        # 1. Sample a step and add data to the Buffer
        # 2. Returns log_data
        pass

    def _run_train(self) -> dict[str, Any]:
        # 1. Samples data from the Buffer
        # 2. Training single step
        # 3. Returns log_data
        pass

    def _run_evaluate(self, n_steps: int = 1) -> dict[str, Any]:
        # 1. Evaluating multiple steps
        # 2. Returns log_data
        pass


if __name__ == "__main__":
    trainer = Trainer()
    for log_data in trainer():
        print(log_data)
```
