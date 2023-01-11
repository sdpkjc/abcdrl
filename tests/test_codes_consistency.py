from __future__ import annotations

import inspect

from abcdrl import ddpg, ddqn, dqn, dqn_atari, dqn_tf, pdqn, ppo, sac, td3
from abcdrl.utils import (
    dqn_all_wrappers,
    wrapper_eval_step,
    wrapper_logger_tf,
    wrapper_logger_torch,
    wrapper_print_filter,
    wrapper_save_model,
)


def test_codes_buffer() -> None:
    # ReplayBuffer
    assert inspect.getsource(dqn.ReplayBuffer) == inspect.getsource(ddqn.ReplayBuffer)
    assert inspect.getsource(dqn.ReplayBuffer) == inspect.getsource(ddpg.ReplayBuffer)
    assert inspect.getsource(dqn.ReplayBuffer) == inspect.getsource(td3.ReplayBuffer)
    assert inspect.getsource(dqn.ReplayBuffer) == inspect.getsource(sac.ReplayBuffer)
    assert inspect.getsource(dqn.ReplayBuffer) == inspect.getsource(dqn_tf.ReplayBuffer)


def test_codes_network() -> None:
    assert inspect.getsource(dqn.Network) == inspect.getsource(ddqn.Network)
    assert inspect.getsource(dqn.Network) == inspect.getsource(pdqn.Network)

    assert inspect.getsource(td3.CriticNetwork) == inspect.getsource(sac.CriticNetwork)


def test_codes_model() -> None:
    assert inspect.getsource(dqn.Model) == inspect.getsource(ddqn.Model)
    assert inspect.getsource(dqn.Model) == inspect.getsource(pdqn.Model)


def test_codes_agent() -> None:
    assert inspect.getsource(dqn.Agent) == inspect.getsource(ddqn.Agent)
    assert inspect.getsource(ddpg.Agent) == inspect.getsource(td3.Agent)
    assert inspect.getsource(dqn.Agent) == inspect.getsource(dqn_atari.Agent)


def test_codes_trainer() -> None:
    assert inspect.getsource(dqn.Trainer.__call__) == inspect.getsource(ddqn.Trainer.__call__)
    assert inspect.getsource(dqn.Trainer.__call__) == inspect.getsource(pdqn.Trainer.__call__)
    assert inspect.getsource(dqn.Trainer.__call__) == inspect.getsource(ddpg.Trainer.__call__)
    assert inspect.getsource(dqn.Trainer.__call__) == inspect.getsource(td3.Trainer.__call__)
    assert inspect.getsource(dqn.Trainer.__call__) == inspect.getsource(sac.Trainer.__call__)
    assert inspect.getsource(dqn.Trainer.__call__) == inspect.getsource(dqn_atari.Trainer.__call__)

    assert inspect.getsource(dqn.Trainer._run_collect) == inspect.getsource(ddqn.Trainer._run_collect)
    assert inspect.getsource(dqn.Trainer._run_collect) == inspect.getsource(pdqn.Trainer._run_collect)
    assert inspect.getsource(dqn.Trainer._run_collect) == inspect.getsource(ddpg.Trainer._run_collect)
    assert inspect.getsource(dqn.Trainer._run_collect) == inspect.getsource(td3.Trainer._run_collect)
    assert inspect.getsource(dqn.Trainer._run_collect) == inspect.getsource(sac.Trainer._run_collect)

    assert inspect.getsource(dqn.Trainer._run_train) == inspect.getsource(ddqn.Trainer._run_train)
    assert inspect.getsource(dqn.Trainer._run_train) == inspect.getsource(ddpg.Trainer._run_train)
    assert inspect.getsource(dqn.Trainer._run_train) == inspect.getsource(td3.Trainer._run_train)
    assert inspect.getsource(dqn.Trainer._run_train) == inspect.getsource(sac.Trainer._run_train)
    assert inspect.getsource(dqn.Trainer._run_train) == inspect.getsource(dqn_atari.Trainer._run_train)

    assert inspect.getsource(dqn.Trainer._make_env) == inspect.getsource(ddqn.Trainer._make_env)
    assert inspect.getsource(dqn.Trainer._make_env) == inspect.getsource(pdqn.Trainer._make_env)
    assert inspect.getsource(ddpg.Trainer._make_env) == inspect.getsource(td3.Trainer._make_env)
    assert inspect.getsource(ddpg.Trainer._make_env) == inspect.getsource(sac.Trainer._make_env)

    assert inspect.getsource(dqn.Trainer) == inspect.getsource(ddqn.Trainer)


def test_codes_wrapper() -> None:
    # wrapper_logger_torch
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(dqn.wrapper_logger_torch)
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(ddqn.wrapper_logger_torch)
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(pdqn.wrapper_logger_torch)
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(ddpg.wrapper_logger_torch)
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(td3.wrapper_logger_torch)
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(sac.wrapper_logger_torch)
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(ppo.wrapper_logger_torch)
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        dqn_atari.wrapper_logger_torch
    )
    assert inspect.getsource(wrapper_logger_torch.wrapper_logger_torch) == inspect.getsource(
        dqn_all_wrappers.wrapper_logger_torch
    )

    # wrapper_logger_tf
    assert inspect.getsource(wrapper_logger_tf.wrapper_logger_tf) == inspect.getsource(dqn_tf.wrapper_logger_tf)

    # wrapper_filter
    assert inspect.getsource(wrapper_print_filter.wrapper_print_filter) == inspect.getsource(
        dqn_all_wrappers.wrapper_print_filter
    )

    # wrapper_eval_step
    assert inspect.getsource(wrapper_eval_step.wrapper_eval_step) == inspect.getsource(
        dqn_all_wrappers.wrapper_eval_step
    )

    # wrapper_save_model
    assert inspect.getsource(wrapper_save_model.wrapper_save_model) == inspect.getsource(
        dqn_all_wrappers.wrapper_save_model
    )


def test_codes_example() -> None:
    assert inspect.getsource(dqn.Trainer) == inspect.getsource(dqn_all_wrappers.Trainer)
    assert inspect.getsource(dqn.Agent) == inspect.getsource(dqn_all_wrappers.Agent)
    assert inspect.getsource(dqn.Algorithm) == inspect.getsource(dqn_all_wrappers.Algorithm)
    assert inspect.getsource(dqn.Model) == inspect.getsource(dqn_all_wrappers.Model)
    assert inspect.getsource(dqn.ReplayBuffer) == inspect.getsource(dqn_all_wrappers.ReplayBuffer)
    assert inspect.getsource(dqn.get_space_shape) == inspect.getsource(dqn_all_wrappers.get_space_shape)


def test_codes_other() -> None:
    # get_space_shape()
    assert inspect.getsource(dqn.get_space_shape) == inspect.getsource(ddqn.get_space_shape)
    assert inspect.getsource(dqn.get_space_shape) == inspect.getsource(pdqn.get_space_shape)
    assert inspect.getsource(dqn.get_space_shape) == inspect.getsource(ddpg.get_space_shape)
    assert inspect.getsource(dqn.get_space_shape) == inspect.getsource(td3.get_space_shape)
    assert inspect.getsource(dqn.get_space_shape) == inspect.getsource(sac.get_space_shape)
    assert inspect.getsource(dqn.get_space_shape) == inspect.getsource(ppo.get_space_shape)
    assert inspect.getsource(dqn.get_space_shape) == inspect.getsource(dqn_atari.get_space_shape)
    assert inspect.getsource(dqn.get_space_shape) == inspect.getsource(dqn_tf.get_space_shape)

    # if __name__ == "__main__":
    dqn_codes = inspect.getsource(dqn)
    dqn_codes = dqn_codes[dqn_codes.find('if __name__ == "__main__":') :]
    ddqn_codes = inspect.getsource(ddqn)
    ddqn_codes = ddqn_codes[ddqn_codes.find('if __name__ == "__main__":') :]
    pdqn_codes = inspect.getsource(pdqn)
    pdqn_codes = pdqn_codes[pdqn_codes.find('if __name__ == "__main__":') :]
    ddpg_codes = inspect.getsource(ddpg)
    ddpg_codes = ddpg_codes[ddpg_codes.find('if __name__ == "__main__":') :]
    td3_codes = inspect.getsource(td3)
    td3_codes = td3_codes[td3_codes.find('if __name__ == "__main__":') :]
    sac_codes = inspect.getsource(sac)
    sac_codes = sac_codes[sac_codes.find('if __name__ == "__main__":') :]
    ppo_codes = inspect.getsource(ppo)
    ppo_codes = ppo_codes[ppo_codes.find('if __name__ == "__main__":') :]
    dqn_atari_codes = inspect.getsource(dqn_atari)
    dqn_atari_codes = dqn_atari_codes[dqn_atari_codes.find('if __name__ == "__main__":') :]
    assert dqn_codes == ddqn_codes
    assert dqn_codes == pdqn_codes
    assert dqn_codes == ddpg_codes
    assert dqn_codes == td3_codes
    assert dqn_codes == sac_codes
    assert dqn_codes == ppo_codes
    assert dqn_codes == dqn_atari_codes
