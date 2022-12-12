from __future__ import annotations

import inspect

import abcdrl
import abcdrl_copy_from


def test_codes_buffer() -> None:
    # ReplayBuffer
    assert inspect.getsource(abcdrl.dqn.ReplayBuffer) == inspect.getsource(abcdrl.ddqn.ReplayBuffer)
    assert inspect.getsource(abcdrl.dqn.ReplayBuffer) == inspect.getsource(abcdrl.ddpg.ReplayBuffer)
    assert inspect.getsource(abcdrl.dqn.ReplayBuffer) == inspect.getsource(abcdrl.td3.ReplayBuffer)
    assert inspect.getsource(abcdrl.dqn.ReplayBuffer) == inspect.getsource(abcdrl.sac.ReplayBuffer)


def test_codes_network() -> None:
    assert inspect.getsource(abcdrl.dqn.Network) == inspect.getsource(abcdrl.ddqn.Network)
    assert inspect.getsource(abcdrl.dqn.Network) == inspect.getsource(abcdrl.pdqn.Network)

    assert inspect.getsource(abcdrl.td3.CriticNetwork) == inspect.getsource(abcdrl.sac.CriticNetwork)


def test_codes_model() -> None:
    assert inspect.getsource(abcdrl.dqn.Model) == inspect.getsource(abcdrl.ddqn.Model)
    assert inspect.getsource(abcdrl.dqn.Model) == inspect.getsource(abcdrl.pdqn.Model)


def test_codes_agent() -> None:
    assert inspect.getsource(abcdrl.dqn.Agent) == inspect.getsource(abcdrl.ddqn.Agent)
    assert inspect.getsource(abcdrl.ddpg.Agent) == inspect.getsource(abcdrl.td3.Agent)
    assert inspect.getsource(abcdrl.dqn.Agent) == inspect.getsource(abcdrl.dqn_atari.Agent)


def test_codes_trainer() -> None:
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.ddqn.Trainer.__call__)
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.pdqn.Trainer.__call__)
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.ddpg.Trainer.__call__)
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.td3.Trainer.__call__)
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.sac.Trainer.__call__)
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.dqn_atari.Trainer.__call__)

    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.ddqn.Trainer._run_collect)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.pdqn.Trainer._run_collect)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.ddpg.Trainer._run_collect)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.td3.Trainer._run_collect)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.sac.Trainer._run_collect)

    assert inspect.getsource(abcdrl.dqn.Trainer._run_train) == inspect.getsource(abcdrl.ddqn.Trainer._run_train)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_train) == inspect.getsource(abcdrl.ddpg.Trainer._run_train)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_train) == inspect.getsource(abcdrl.td3.Trainer._run_train)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_train) == inspect.getsource(abcdrl.sac.Trainer._run_train)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_train) == inspect.getsource(abcdrl.dqn_atari.Trainer._run_train)

    assert inspect.getsource(abcdrl.dqn.Trainer._make_env) == inspect.getsource(abcdrl.ddqn.Trainer._make_env)
    assert inspect.getsource(abcdrl.dqn.Trainer._make_env) == inspect.getsource(abcdrl.pdqn.Trainer._make_env)
    assert inspect.getsource(abcdrl.ddpg.Trainer._make_env) == inspect.getsource(abcdrl.td3.Trainer._make_env)
    assert inspect.getsource(abcdrl.ddpg.Trainer._make_env) == inspect.getsource(abcdrl.sac.Trainer._make_env)

    assert inspect.getsource(abcdrl.dqn.Trainer) == inspect.getsource(abcdrl.ddqn.Trainer)


def test_codes_wrapper() -> None:
    # wrapper_logger
    assert inspect.getsource(abcdrl_copy_from.wrapper_logger) == inspect.getsource(abcdrl.dqn.wrapper_logger)
    assert inspect.getsource(abcdrl_copy_from.wrapper_logger) == inspect.getsource(abcdrl.ddqn.wrapper_logger)
    assert inspect.getsource(abcdrl_copy_from.wrapper_logger) == inspect.getsource(abcdrl.pdqn.wrapper_logger)
    assert inspect.getsource(abcdrl_copy_from.wrapper_logger) == inspect.getsource(abcdrl.ddpg.wrapper_logger)
    assert inspect.getsource(abcdrl_copy_from.wrapper_logger) == inspect.getsource(abcdrl.td3.wrapper_logger)
    assert inspect.getsource(abcdrl_copy_from.wrapper_logger) == inspect.getsource(abcdrl.sac.wrapper_logger)
    assert inspect.getsource(abcdrl_copy_from.wrapper_logger) == inspect.getsource(abcdrl.ppo.wrapper_logger)
    assert inspect.getsource(abcdrl_copy_from.wrapper_logger) == inspect.getsource(abcdrl.dqn_atari.wrapper_logger)
    assert inspect.getsource(abcdrl_copy_from.wrapper_logger) == inspect.getsource(
        abcdrl_copy_from.dqn_all_wrappers.wrapper_logger
    )

    # wrapper_filter
    assert inspect.getsource(abcdrl_copy_from.wrapper_print_filter) == inspect.getsource(
        abcdrl.dqn.wrapper_print_filter
    )
    assert inspect.getsource(abcdrl_copy_from.wrapper_print_filter) == inspect.getsource(
        abcdrl.ddqn.wrapper_print_filter
    )
    assert inspect.getsource(abcdrl_copy_from.wrapper_print_filter) == inspect.getsource(
        abcdrl.pdqn.wrapper_print_filter
    )
    assert inspect.getsource(abcdrl_copy_from.wrapper_print_filter) == inspect.getsource(
        abcdrl.ddpg.wrapper_print_filter
    )
    assert inspect.getsource(abcdrl_copy_from.wrapper_print_filter) == inspect.getsource(
        abcdrl.td3.wrapper_print_filter
    )
    assert inspect.getsource(abcdrl_copy_from.wrapper_print_filter) == inspect.getsource(
        abcdrl.sac.wrapper_print_filter
    )
    assert inspect.getsource(abcdrl_copy_from.wrapper_print_filter) == inspect.getsource(
        abcdrl.ppo.wrapper_print_filter
    )
    assert inspect.getsource(abcdrl_copy_from.wrapper_print_filter) == inspect.getsource(
        abcdrl.dqn_atari.wrapper_print_filter
    )
    assert inspect.getsource(abcdrl_copy_from.wrapper_print_filter) == inspect.getsource(
        abcdrl_copy_from.dqn_all_wrappers.wrapper_print_filter
    )

    # wrapper_eval_step
    assert inspect.getsource(abcdrl_copy_from.wrapper_eval_step) == inspect.getsource(
        abcdrl_copy_from.dqn_all_wrappers.wrapper_eval_step
    )

    # wrapper_save_model
    assert inspect.getsource(abcdrl_copy_from.wrapper_save_model) == inspect.getsource(
        abcdrl_copy_from.dqn_all_wrappers.wrapper_save_model
    )


def test_codes_example() -> None:
    assert inspect.getsource(abcdrl.dqn.Trainer) == inspect.getsource(abcdrl_copy_from.dqn_all_wrappers.Trainer)
    assert inspect.getsource(abcdrl.dqn.Agent) == inspect.getsource(abcdrl_copy_from.dqn_all_wrappers.Agent)
    assert inspect.getsource(abcdrl.dqn.Algorithm) == inspect.getsource(abcdrl_copy_from.dqn_all_wrappers.Algorithm)
    assert inspect.getsource(abcdrl.dqn.Model) == inspect.getsource(abcdrl_copy_from.dqn_all_wrappers.Model)
    assert inspect.getsource(abcdrl.dqn.ReplayBuffer) == inspect.getsource(
        abcdrl_copy_from.dqn_all_wrappers.ReplayBuffer
    )
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(
        abcdrl_copy_from.dqn_all_wrappers.get_space_shape
    )


def test_codes_other() -> None:
    # get_space_shape()
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.ddqn.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.pdqn.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.ddpg.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.td3.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.sac.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.ppo.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.dqn_atari.get_space_shape)

    # if __name__ == "__main__":
    dqn_codes = inspect.getsource(abcdrl.dqn)
    dqn_codes = dqn_codes[dqn_codes.find('if __name__ == "__main__":') :]
    ddqn_codes = inspect.getsource(abcdrl.ddqn)
    ddqn_codes = ddqn_codes[ddqn_codes.find('if __name__ == "__main__":') :]
    pdqn_codes = inspect.getsource(abcdrl.pdqn)
    pdqn_codes = pdqn_codes[pdqn_codes.find('if __name__ == "__main__":') :]
    ddpg_codes = inspect.getsource(abcdrl.ddpg)
    ddpg_codes = ddpg_codes[ddpg_codes.find('if __name__ == "__main__":') :]
    td3_codes = inspect.getsource(abcdrl.td3)
    td3_codes = td3_codes[td3_codes.find('if __name__ == "__main__":') :]
    sac_codes = inspect.getsource(abcdrl.sac)
    sac_codes = sac_codes[sac_codes.find('if __name__ == "__main__":') :]
    ppo_codes = inspect.getsource(abcdrl.ppo)
    ppo_codes = ppo_codes[ppo_codes.find('if __name__ == "__main__":') :]
    dqn_atari_codes = inspect.getsource(abcdrl.dqn_atari)
    dqn_atari_codes = dqn_atari_codes[dqn_atari_codes.find('if __name__ == "__main__":') :]
    assert dqn_codes == ddqn_codes
    assert dqn_codes == pdqn_codes
    assert dqn_codes == ddpg_codes
    assert dqn_codes == td3_codes
    assert dqn_codes == sac_codes
    assert dqn_codes == ppo_codes
    assert dqn_codes == dqn_atari_codes
