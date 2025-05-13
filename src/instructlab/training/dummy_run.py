from instructlab.training.logger import setup_metric_logger
import logging
import random

import subprocess


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def main():
    setup_metric_logger(
        loggers=["wandb"],
        run_name=f"run_training_wandb.yml@{get_git_revision_hash()}",
        output_dir="./logs",
    )
    logger = logging.getLogger("instructlab.training.metrics")

    logger.info({"lr": 1e-4, "epochs": 10}, extra={"hparams": True})

    global_step = 0
    for epoch in range(10):
        for _ in range(100):
            logger.info(
                {
                    "train/loss": 10 - 0.001 * global_step + random.random(),
                    "epoch": epoch,
                    "step": global_step,
                }
            )
            global_step += 1
        logger.info(
            {"val/loss": 1 - 0.01 * epoch + random.random() / 3},
            extra={"step": global_step},
        )


if __name__ == "__main__":
    main()
