from __future__ import annotations

import inspect

from abcdrl import (
    ddpg_torch,
    ddqn_tf,
    ddqn_torch,
    dqn_atari_tf,
    dqn_atari_torch,
    dqn_tf,
    dqn_torch,
    pdqn_tf,
    pdqn_torch,
    ppo_torch,
    sac_torch,
    td3_torch,
)
from abcdrl.utils import (
    dqn_all_wrappers_torch,
    wrapper_eval_step,
    wrapper_logger_tf,
    wrapper_logger_torch,
    wrapper_print_filter,
    wrapper_save_model,
)


def test_codes_buffer() -> None:
    # ReplayBuffer
    assert inspect.getsource(dqn_torch.ReplayBuffer) == inspect.getsource(ddqn_torch.ReplayBuffer)
    assert inspect.getsource(dqn_torch.ReplayBuffer) == inspect.getsource(ddpg_torch.ReplayBuffer)
    assert inspect.getsource(dqn_torch.ReplayBuffer) == inspect.getsource(td3_torch.ReplayBuffer)
    assert inspect.getsource(dqn_torch.ReplayBuffer) == inspect.getsource(sac_torch.ReplayBuffer)
    assert inspect.getsource(dqn_torch.ReplayBuffer) == inspect.getsource(dqn_tf.ReplayBuffer)
    assert inspect.getsource(dqn_torch.ReplayBuffer) == inspect.getsource(ddqn_tf.ReplayBuffer)

    # ReplayBuffer optimize_memory_usage
    assert inspect.getsource(dqn_atari_torch.ReplayBuffer) == inspect.getsource(dqn_atari_tf.ReplayBuffer)

    # PrioritizedReplayBuffer
    assert inspect.getsource(pdqn_torch.PrioritizedReplayBuffer) == inspect.getsource(pdqn_tf.PrioritizedReplayBuffer)


def test_codes_network() -> None:
    # torch
    assert inspect.getsource(dqn_torch.Network) == inspect.getsource(ddqn_torch.Network)
    assert inspect.getsource(dqn_torch.Network) == inspect.getsource(pdqn_torch.Network)

    assert inspect.getsource(td3_torch.CriticNetwork) == inspect.getsource(sac_torch.CriticNetwork)

    # tensorflow
    assert inspect.getsource(dqn_tf.Network) == inspect.getsource(ddqn_tf.Network)
    assert inspect.getsource(dqn_tf.Network) == inspect.getsource(pdqn_tf.Network)


def test_codes_model() -> None:
    # torch
    assert inspect.getsource(dqn_torch.Model) == inspect.getsource(ddqn_torch.Model)
    assert inspect.getsource(dqn_torch.Model) == inspect.getsource(pdqn_torch.Model)

    # tensorflow
    assert inspect.getsource(dqn_tf.Model) == inspect.getsource(ddqn_tf.Model)
    assert inspect.getsource(dqn_tf.Model) == inspect.getsource(pdqn_tf.Model)


def test_codes_agent() -> None:
    # torch
    assert inspect.getsource(dqn_torch.Agent) == inspect.getsource(ddqn_torch.Agent)
    assert inspect.getsource(ddpg_torch.Agent) == inspect.getsource(td3_torch.Agent)
    assert inspect.getsource(dqn_torch.Agent) == inspect.getsource(dqn_atari_torch.Agent)

    # tensorflow
    assert inspect.getsource(dqn_tf.Agent) == inspect.getsource(ddqn_tf.Agent)


def test_codes_trainer() -> None:
    assert inspect.getsource(dqn_torch.Trainer.__call__) == inspect.getsource(ddqn_torch.Trainer.__call__)
    assert inspect.getsource(dqn_torch.Trainer.__call__) == inspect.getsource(pdqn_torch.Trainer.__call__)
    assert inspect.getsource(dqn_torch.Trainer.__call__) == inspect.getsource(ddpg_torch.Trainer.__call__)
    assert inspect.getsource(dqn_torch.Trainer.__call__) == inspect.getsource(td3_torch.Trainer.__call__)
    assert inspect.getsource(dqn_torch.Trainer.__call__) == inspect.getsource(sac_torch.Trainer.__call__)
    assert inspect.getsource(dqn_torch.Trainer.__call__) == inspect.getsource(dqn_atari_torch.Trainer.__call__)
    assert inspect.getsource(dqn_torch.Trainer.__call__) == inspect.getsource(ddqn_tf.Trainer.__call__)
    assert inspect.getsource(dqn_torch.Trainer.__call__) == inspect.getsource(pdqn_tf.Trainer.__call__)
    assert inspect.getsource(dqn_torch.Trainer.__call__) == inspect.getsource(dqn_atari_tf.Trainer.__call__)

    assert inspect.getsource(dqn_torch.Trainer._run_collect) == inspect.getsource(ddqn_torch.Trainer._run_collect)
    assert inspect.getsource(dqn_torch.Trainer._run_collect) == inspect.getsource(pdqn_torch.Trainer._run_collect)
    assert inspect.getsource(dqn_torch.Trainer._run_collect) == inspect.getsource(ddpg_torch.Trainer._run_collect)
    assert inspect.getsource(dqn_torch.Trainer._run_collect) == inspect.getsource(td3_torch.Trainer._run_collect)
    assert inspect.getsource(dqn_torch.Trainer._run_collect) == inspect.getsource(sac_torch.Trainer._run_collect)
    assert inspect.getsource(dqn_torch.Trainer._run_collect) == inspect.getsource(ddqn_tf.Trainer._run_collect)
    assert inspect.getsource(dqn_torch.Trainer._run_collect) == inspect.getsource(pdqn_tf.Trainer._run_collect)
    assert inspect.getsource(dqn_torch.Trainer._run_collect) == inspect.getsource(dqn_atari_tf.Trainer._run_collect)

    assert inspect.getsource(dqn_torch.Trainer._run_train) == inspect.getsource(ddqn_torch.Trainer._run_train)
    assert inspect.getsource(dqn_torch.Trainer._run_train) == inspect.getsource(ddpg_torch.Trainer._run_train)
    assert inspect.getsource(dqn_torch.Trainer._run_train) == inspect.getsource(td3_torch.Trainer._run_train)
    assert inspect.getsource(dqn_torch.Trainer._run_train) == inspect.getsource(sac_torch.Trainer._run_train)
    assert inspect.getsource(dqn_torch.Trainer._run_train) == inspect.getsource(dqn_atari_torch.Trainer._run_train)
    assert inspect.getsource(dqn_torch.Trainer._run_train) == inspect.getsource(dqn_tf.Trainer._run_train)
    assert inspect.getsource(dqn_torch.Trainer._run_train) == inspect.getsource(ddqn_tf.Trainer._run_train)
    assert inspect.getsource(dqn_atari_torch.Trainer._run_train) == inspect.getsource(dqn_atari_tf.Trainer._run_train)

    assert inspect.getsource(dqn_torch.Trainer._make_env) == inspect.getsource(ddqn_torch.Trainer._make_env)
    assert inspect.getsource(dqn_torch.Trainer._make_env) == inspect.getsource(pdqn_torch.Trainer._make_env)
    assert inspect.getsource(ddpg_torch.Trainer._make_env) == inspect.getsource(td3_torch.Trainer._make_env)
    assert inspect.getsource(ddpg_torch.Trainer._make_env) == inspect.getsource(sac_torch.Trainer._make_env)
    assert inspect.getsource(dqn_torch.Trainer._run_train) == inspect.getsource(dqn_tf.Trainer._run_train)
    assert inspect.getsource(dqn_torch.Trainer._run_train) == inspect.getsource(ddqn_tf.Trainer._run_train)
    assert inspect.getsource(dqn_atari_torch.Trainer._run_train) == inspect.getsource(dqn_atari_tf.Trainer._run_train)

    assert inspect.getsource(dqn_torch.Trainer) == inspect.getsource(ddqn_torch.Trainer)

    assert inspect.getsource(dqn_tf.Trainer) == inspect.getsource(ddqn_tf.Trainer)


def test_codes_wrapper() -> None:
    # wrapper_logger_torch
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        dqn_torch.wrapper_logger_torch
    )
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        ddqn_torch.wrapper_logger_torch
    )
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        pdqn_torch.wrapper_logger_torch
    )
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        ddpg_torch.wrapper_logger_torch
    )
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        td3_torch.wrapper_logger_torch
    )
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        sac_torch.wrapper_logger_torch
    )
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        ppo_torch.wrapper_logger_torch
    )
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        dqn_atari_torch.wrapper_logger_torch
    )
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        dqn_all_wrappers_torch.wrapper_logger_torch
    )

    # wrapper_logger_tf
    assert inspect.getsource(wrapper_logger_tf.wrapper_logger_tf) == inspect.getsource(dqn_tf.wrapper_logger_tf)
    assert inspect.getsource(wrapper_logger_tf.wrapper_logger_tf) == inspect.getsource(ddqn_tf.wrapper_logger_tf)
    assert inspect.getsource(wrapper_logger_tf.wrapper_logger_tf) == inspect.getsource(pdqn_tf.wrapper_logger_tf)
    assert inspect.getsource(wrapper_logger_tf.wrapper_logger_tf) == inspect.getsource(dqn_atari_tf.wrapper_logger_tf)

    # wrapper_filter
    assert inspect.getsource(wrapper_print_filter.wrapper_print_filter) == inspect.getsource(
        dqn_all_wrappers_torch.wrapper_print_filter
    )

    # wrapper_eval_step
    assert inspect.getsource(wrapper_eval_step.wrapper_eval_step) == inspect.getsource(
        dqn_all_wrappers_torch.wrapper_eval_step
    )

    # wrapper_save_model
    assert inspect.getsource(wrapper_save_model.wrapper_save_model) == inspect.getsource(
        dqn_all_wrappers_torch.wrapper_save_model
    )


def test_codes_example() -> None:
    assert inspect.getsource(dqn_torch.Trainer) == inspect.getsource(dqn_all_wrappers_torch.Trainer)
    assert inspect.getsource(dqn_torch.Agent) == inspect.getsource(dqn_all_wrappers_torch.Agent)
    assert inspect.getsource(dqn_torch.Algorithm) == inspect.getsource(dqn_all_wrappers_torch.Algorithm)
    assert inspect.getsource(dqn_torch.Model) == inspect.getsource(dqn_all_wrappers_torch.Model)
    assert inspect.getsource(dqn_torch.ReplayBuffer) == inspect.getsource(dqn_all_wrappers_torch.ReplayBuffer)
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(dqn_all_wrappers_torch.get_space_shape)


def test_codes_other() -> None:
    # get_space_shape()
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(ddqn_torch.get_space_shape)
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(pdqn_torch.get_space_shape)
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(ddpg_torch.get_space_shape)
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(td3_torch.get_space_shape)
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(sac_torch.get_space_shape)
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(ppo_torch.get_space_shape)
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(dqn_atari_torch.get_space_shape)
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(dqn_tf.get_space_shape)
    assert inspect.getsource(dqn_torch.get_space_shape) == inspect.getsource(dqn_atari_tf.get_space_shape)

    # torch if __name__ == "__main__":
    dqn_codes = inspect.getsource(dqn_torch)
    dqn_codes = dqn_codes[dqn_codes.find('if __name__ == "__main__":') :]
    ddqn_codes = inspect.getsource(ddqn_torch)
    ddqn_codes = ddqn_codes[ddqn_codes.find('if __name__ == "__main__":') :]
    pdqn_codes = inspect.getsource(pdqn_torch)
    pdqn_codes = pdqn_codes[pdqn_codes.find('if __name__ == "__main__":') :]
    ddpg_codes = inspect.getsource(ddpg_torch)
    ddpg_codes = ddpg_codes[ddpg_codes.find('if __name__ == "__main__":') :]
    td3_codes = inspect.getsource(td3_torch)
    td3_codes = td3_codes[td3_codes.find('if __name__ == "__main__":') :]
    sac_codes = inspect.getsource(sac_torch)
    sac_codes = sac_codes[sac_codes.find('if __name__ == "__main__":') :]
    ppo_codes = inspect.getsource(ppo_torch)
    ppo_codes = ppo_codes[ppo_codes.find('if __name__ == "__main__":') :]
    dqn_atari_codes = inspect.getsource(dqn_atari_torch)
    dqn_atari_codes = dqn_atari_codes[dqn_atari_codes.find('if __name__ == "__main__":') :]
    assert dqn_codes == ddqn_codes
    assert dqn_codes == pdqn_codes
    assert dqn_codes == ddpg_codes
    assert dqn_codes == td3_codes
    assert dqn_codes == sac_codes
    assert dqn_codes == ppo_codes
    assert dqn_codes == dqn_atari_codes

    # tensorflow if __name__ == "__main__":
    dqn_codes = inspect.getsource(dqn_tf)
    dqn_codes = dqn_codes[dqn_codes.find('if __name__ == "__main__":') :]
    ddqn_codes = inspect.getsource(ddqn_tf)
    ddqn_codes = ddqn_codes[ddqn_codes.find('if __name__ == "__main__":') :]
    pdqn_codes = inspect.getsource(pdqn_tf)
    pdqn_codes = pdqn_codes[pdqn_codes.find('if __name__ == "__main__":') :]
    dqn_atari_codes = inspect.getsource(dqn_atari_tf)
    dqn_atari_codes = dqn_atari_codes[dqn_atari_codes.find('if __name__ == "__main__":') :]
    assert dqn_codes == ddqn_codes
    assert dqn_codes == pdqn_codes
    assert dqn_codes == dqn_atari_codes
