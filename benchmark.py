import argparse
import copy
import subprocess
import threading
from distutils.util import strtobool

import tomli


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="dqn",
        help="the algorithm of this experiment")
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--device", type=str, default='auto',
        help="device of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="abcdrl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--seeds", type=int, action='append', nargs='*', default=[1],
        help="seed of the experiment")

    args = parser.parse_args()
    # fmt: on
    if len(args.seeds) > 1:
        args.seeds = args.seeds[1]
    return args


def train_process(cmd, kwargs):
    for param in kwargs.items():
        cmd += f" --{param[0].replace('_','-')} {param[1]}"
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    args = parse_args()

    kwargs = vars(args)
    with open("benchmark.toml", "rb") as file:
        cmd = tomli.load(file)[kwargs["alg"]][kwargs["env_id"]]

    kwargs.pop("alg")
    kwargs.pop("env_id")
    seeds = kwargs.pop("seeds")

    for seed in seeds:
        kwargs_tmp = copy.deepcopy(kwargs)
        kwargs_tmp["seed"] = seed
        thread = threading.Thread(target=train_process, args=(cmd, kwargs_tmp))
        thread.start()
