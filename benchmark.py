from __future__ import annotations

import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import fire
import tomli


def train_process(alg: str, framework: str, env_id: str, kwargs: dict[str, Any]) -> None:
    with open("benchmark.toml", "rb") as file:
        cmd = tomli.load(file)[alg][framework][env_id]
    for param in kwargs.items():
        if isinstance(param[1], list):
            cmd += f' --{param[0]} "{param[1]}"'
        else:
            cmd += f" --{param[0]} {param[1]}"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def main(
    algs: str | list[str] = ["dqn"],
    frameworks: str | list[str] = ["torch"],
    env_ids: str | list[str] = ["CartPole-v1"],
    seeds: int | list[int] = [1],
    cuda: bool = True,
    workers: int = 3,
    track: bool = False,
    wandb_project_name: str = "abcdrl",
    wandb_entity: str | None = None,
    wandb_tags: str | list[str] = [],
    capture_video: bool = False,
):
    if not isinstance(algs, list):
        algs = [algs]
    if not isinstance(frameworks, list):
        frameworks = [frameworks]
    if not isinstance(env_ids, list):
        env_ids = [env_ids]
    if not isinstance(seeds, list):
        seeds = [seeds]
    if not isinstance(wandb_tags, list):
        wandb_tags = [wandb_tags]

    git_commit_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    wandb_tags.append(git_commit_sha)
    wandb_tags.append("latest")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for alg, framework, env_id, seed in itertools.product(algs, frameworks, env_ids, seeds):
            kwargs = {
                "cuda": cuda,
                "track": track,
                "wandb-project-name": wandb_project_name,
                "wandb-entity": wandb_entity,
                "wandb-tags": wandb_tags,
                "capture-video": capture_video,
                "seed": seed,
            }
            executor.submit(train_process, alg, framework, env_id, kwargs)


if __name__ == "__main__":
    fire.Fire(main)
