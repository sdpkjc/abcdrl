import inspect

import abcdrl


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
    assert inspect.getsource(abcdrl.dqn.Network) == inspect.getsource(abcdrl.ddqn.Network)
    assert inspect.getsource(abcdrl.dqn.Network) == inspect.getsource(abcdrl.pdqn.Network)


def test_codes_agent() -> None:
    assert inspect.getsource(abcdrl.dqn.Agent) == inspect.getsource(abcdrl.ddqn.Agent)
    assert inspect.getsource(abcdrl.ddpg.Agent) == inspect.getsource(abcdrl.td3.Agent)


def test_codes_trainer() -> None:
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.ddqn.Trainer.__call__)
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.pdqn.Trainer.__call__)
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.ddpg.Trainer.__call__)
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.td3.Trainer.__call__)
    assert inspect.getsource(abcdrl.dqn.Trainer.__call__) == inspect.getsource(abcdrl.sac.Trainer.__call__)

    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.ddqn.Trainer._run_collect)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.pdqn.Trainer._run_collect)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.ddpg.Trainer._run_collect)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.td3.Trainer._run_collect)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_collect) == inspect.getsource(abcdrl.sac.Trainer._run_collect)

    assert inspect.getsource(abcdrl.dqn.Trainer._run_train) == inspect.getsource(abcdrl.ddqn.Trainer._run_train)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_train) == inspect.getsource(abcdrl.ddpg.Trainer._run_train)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_train) == inspect.getsource(abcdrl.td3.Trainer._run_train)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_train) == inspect.getsource(abcdrl.sac.Trainer._run_train)

    assert inspect.getsource(abcdrl.dqn.Trainer._run_evaluate) == inspect.getsource(abcdrl.ddqn.Trainer._run_evaluate)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_evaluate) == inspect.getsource(abcdrl.pdqn.Trainer._run_evaluate)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_evaluate) == inspect.getsource(abcdrl.ddpg.Trainer._run_evaluate)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_evaluate) == inspect.getsource(abcdrl.td3.Trainer._run_evaluate)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_evaluate) == inspect.getsource(abcdrl.sac.Trainer._run_evaluate)
    assert inspect.getsource(abcdrl.dqn.Trainer._run_evaluate) == inspect.getsource(abcdrl.ppo.Trainer._run_evaluate)

    assert inspect.getsource(abcdrl.dqn.Trainer._make_env) == inspect.getsource(abcdrl.ddqn.Trainer._make_env)
    assert inspect.getsource(abcdrl.dqn.Trainer._make_env) == inspect.getsource(abcdrl.pdqn.Trainer._make_env)
    assert inspect.getsource(abcdrl.ddpg.Trainer._make_env) == inspect.getsource(abcdrl.td3.Trainer._make_env)
    assert inspect.getsource(abcdrl.ddpg.Trainer._make_env) == inspect.getsource(abcdrl.sac.Trainer._make_env)

    assert inspect.getsource(abcdrl.dqn.Trainer) == inspect.getsource(abcdrl.ddqn.Trainer)


def test_codes_wrapper() -> None:
    # logger
    assert inspect.getsource(abcdrl.dqn.logger) == inspect.getsource(abcdrl.ddqn.logger)
    assert inspect.getsource(abcdrl.dqn.logger) == inspect.getsource(abcdrl.pdqn.logger)
    assert inspect.getsource(abcdrl.dqn.logger) == inspect.getsource(abcdrl.ddpg.logger)
    assert inspect.getsource(abcdrl.dqn.logger) == inspect.getsource(abcdrl.td3.logger)
    assert inspect.getsource(abcdrl.dqn.logger) == inspect.getsource(abcdrl.sac.logger)
    assert inspect.getsource(abcdrl.dqn.logger) == inspect.getsource(abcdrl.ppo.logger)

    # saver
    assert inspect.getsource(abcdrl.dqn.saver) == inspect.getsource(abcdrl.ddqn.saver)
    assert inspect.getsource(abcdrl.dqn.saver) == inspect.getsource(abcdrl.pdqn.saver)
    assert inspect.getsource(abcdrl.dqn.saver) == inspect.getsource(abcdrl.ddpg.saver)
    assert inspect.getsource(abcdrl.dqn.saver) == inspect.getsource(abcdrl.td3.saver)
    assert inspect.getsource(abcdrl.dqn.saver) == inspect.getsource(abcdrl.sac.saver)
    assert inspect.getsource(abcdrl.dqn.saver) == inspect.getsource(abcdrl.ppo.saver)

    # filter
    assert inspect.getsource(abcdrl.dqn.filter) == inspect.getsource(abcdrl.ddqn.filter)
    assert inspect.getsource(abcdrl.dqn.filter) == inspect.getsource(abcdrl.pdqn.filter)
    assert inspect.getsource(abcdrl.dqn.filter) == inspect.getsource(abcdrl.ddpg.filter)
    assert inspect.getsource(abcdrl.dqn.filter) == inspect.getsource(abcdrl.td3.filter)
    assert inspect.getsource(abcdrl.dqn.filter) == inspect.getsource(abcdrl.sac.filter)
    assert inspect.getsource(abcdrl.dqn.filter) == inspect.getsource(abcdrl.ppo.filter)


def test_codes_other() -> None:
    # get_space_shape()
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.ddqn.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.pdqn.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.ddpg.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.td3.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.sac.get_space_shape)
    assert inspect.getsource(abcdrl.dqn.get_space_shape) == inspect.getsource(abcdrl.ppo.get_space_shape)

    # if __name__ == "__main__":
    dqn_codes = inspect.getsource(abcdrl.dqn)
    dqn_codes = dqn_codes[dqn_codes.find('if __name__ == "__main__":') :]
    ddqn_codes = inspect.getsource(abcdrl.dqn)
    ddqn_codes = ddqn_codes[ddqn_codes.find('if __name__ == "__main__":') :]
    pdqn_codes = inspect.getsource(abcdrl.dqn)
    pdqn_codes = pdqn_codes[pdqn_codes.find('if __name__ == "__main__":') :]
    ddpg_codes = inspect.getsource(abcdrl.dqn)
    ddpg_codes = ddpg_codes[ddpg_codes.find('if __name__ == "__main__":') :]
    td3_codes = inspect.getsource(abcdrl.dqn)
    td3_codes = td3_codes[td3_codes.find('if __name__ == "__main__":') :]
    sac_codes = inspect.getsource(abcdrl.dqn)
    sac_codes = sac_codes[sac_codes.find('if __name__ == "__main__":') :]
    ppo_codes = inspect.getsource(abcdrl.dqn)
    ppo_codes = ppo_codes[ppo_codes.find('if __name__ == "__main__":') :]
    assert dqn_codes == ddqn_codes
    assert dqn_codes == pdqn_codes
    assert dqn_codes == ddpg_codes
    assert dqn_codes == td3_codes
    assert dqn_codes == sac_codes
    assert dqn_codes == ppo_codes