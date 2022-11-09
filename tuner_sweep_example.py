from __future__ import annotations

import argparse
import time

import wandb

import abcdrl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--run-count", type=str, default=1)
    parser.add_argument("--wandb-project-name", type=str, default="abcdrl")

    args = parser.parse_args()
    return args


sweep_configuration = {
    "method": "random",
    "name": "dqn_sweep",
    "metric": {"goal": "maximize", "name": "episodic_return"},
    "parameters": {
        "env_id": {"values": ["CartPole-v1"]},
        "learning_rate": {"values": [0.00020, 0.00025, 0.00035]},
        "batch_size": {"values": [64, 128]},
        "train_frequency": {"values": [10, 20]},
        "target_network_frequency": {"values": [300, 500, 1000]},
        "exploration_fraction": {"values": [0.3, 0.5, 0.7]},
    },
}


def tune_agent() -> None:
    writer = wandb.init(project=args.wandb_project_name, name=f"{sweep_configuration['name']}__{int(time.time())}")
    wandb.save("abcdrl/dqn.py")

    trainer = abcdrl.dqn.Trainer(
        env_id=wandb.config.env_id,
        learning_rate=wandb.config.learning_rate,
        batch_size=wandb.config.batch_size,
        train_frequency=wandb.config.train_frequency,
        target_network_frequency=wandb.config.target_network_frequency,
        exploration_fraction=wandb.config.exploration_fraction,
    )

    for log_data in trainer():
        if "logs" in log_data:
            if log_data["log_type"] != "train":
                print(log_data)
            writer.log(
                {f"{log_data['log_type']}/{x[0]}": x[1] for x in log_data["logs"].items()}, step=log_data["sample_step"]
            )
            writer.log({"global_step": log_data["sample_step"]}, step=log_data["sample_step"])


if __name__ == "__main__":
    args = parse_args()

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.wandb_project_name)
    wandb.agent(sweep_id, function=tune_agent, count=args.run_count)
