from __future__ import annotations

import copy
import subprocess
import threading
from typing import Any, Optional

import fire
import tomli


def train_process(cmd: str, kwargs: dict[str, Any]) -> None:
    for param in kwargs.items():
        cmd += f" --{param[0].replace('_','-')} {param[1]}"
    subprocess.run(cmd, shell=True, check=True)


def main(
    alg: str = "dqn",
    env_id: str = "CartPole-v1",
    device: str = "auto",
    track: bool = False,
    wandb_project_name: str = "abcdrl",
    wandb_entity: Optional[str] = None,
    wandb_tags: list[str] = [],
    capture_video: bool = True,
    seeds: list[int] = [1],
):
    kwargs = locals()
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


if __name__ == "__main__":
    fire.Fire(main)
