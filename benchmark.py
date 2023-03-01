from __future__ import annotations

import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import tomli
import tyro


def train_process(alg: str, framework: str, env_id: str, config: dict[str, Any]) -> None:
    with open("benchmark.toml", "rb") as file:
        cmd = tomli.load(file)[alg][framework][env_id]
    for param in config.items():
        if isinstance(param[1], list):
            cmd += f" --{param[0]} {' '.join(param[1])}"
        elif isinstance(param[1], bool):
            cmd += f" --{param[0]}"
        else:
            cmd += f" --{param[0]} {param[1]}"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def main(
    algs: List[str] = ["dqn"],
    frameworks: List[str] = ["torch"],
    env_ids: List[str] = ["CartPole-v1"],
    seeds: List[int] = [1],
    cuda: bool = True,
    workers: int = 3,
    track: bool = False,
    wandb_project_name: str = "abcdrl",
    wandb_entity: Optional[str] = None,
    wandb_tags: List[str] = [],
    capture_video: bool = False,
):
    git_commit_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    wandb_tags.append(git_commit_sha)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for alg, framework, env_id, seed in itertools.product(algs, frameworks, env_ids, seeds):
            config = {
                "logger.wandb-project-name": wandb_project_name,
                "logger.wandb-entity": wandb_entity,
                "logger.wandb-tags": wandb_tags,
                "trainer.capture-video": capture_video,
                "trainer.seed": seed,
            }
            if not cuda:
                config = {**config, "trainer.no-cuda": False}
            if track:
                config = {**config, "logger.track": True}

            executor.submit(train_process, alg, framework, env_id, config)


if __name__ == "__main__":
    tyro.cli(main)
