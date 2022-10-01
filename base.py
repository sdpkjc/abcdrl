
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class Model(nn.Module):
    def __init__(self, kwargs: Dict) -> None:
        pass

    def value(self, x: torch.Tensor, a: Optional[torch.Tensor] = None) -> Tuple:
        # 多个 或 单个 critic 的输出值
        pass
    
    def action(self, x: torch.Tensor) -> Tuple:
        # 给出 动作 | 动作概率分布
        pass

class Algorithm():
    def __init__(self, kwargs: Dict) -> None:
        # 1. 初始化 model, target_model
        # 2. 初始化 optimizer
        pass

    def predict(self, obs: torch.Tensor) -> Tuple:
        # 给出 动作 | 动作概率分布 | Q函数的预估值
        pass

    def learn(self, data: ReplayBufferSamples) -> Dict:
        # 根据训练数据（观测量和输入的reward），定义损失函数，用于更新 Model 中的参数。
        
        # 1. 计算目标
        # 2. 计算损失
        # 3. 优化模型
        # 4. 返回训练信息
        pass

    def sync_target(self) -> None:
        # 同步 model 和 target_model
        pass


class Agent():
    def __init__(self, kwargs: Dict) -> None:
        pass

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # 评估
        # obs预处理 to_tensor & to_device
        # act后处理 to_numpy & to_cpu
        pass

    def sample(self, obs: np.ndarray) -> np.ndarray:
        # 训练
        # obs预处理 to_tensor & to_device
        # act后处理 to_numpy & to_cpu
        pass

    def learn(self, data: ReplayBufferSamples) -> Dict:
        # 数据预处理
        # 同步操作或定时操作控制
        pass


class Trainer():
    # Buffer, Env, Agent
    # logger
    def __init__(self, kwargs: Dict) -> None:
        pass
    def _run_collect(self, n: int = 1) -> None:
        pass
    def _run_train(self):
        pass
    def _run_evaluate(self, n_episodic: int = 1) -> None:
        pass
    def run(self):
        pass
    
