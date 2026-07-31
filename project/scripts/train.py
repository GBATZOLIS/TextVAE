"""Training entry point.

    python -m project.scripts.train --config project/configs/default.yaml
    python -m project.scripts.train --config project/configs/coco.yaml --set optim.lr=5e-5
    accelerate launch -m project.scripts.train --config project/configs/coco.yaml

Any config field can be overridden from the command line with dotted ``key=value`` pairs
after ``--set``. Multi-GPU is handled by ``accelerate launch``; a single-process run works
without any accelerate configuration.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from ..datasets.builder import build_dataloader
from ..models.vae import TextVAE
from ..training.trainer import Trainer
from ..utils.config import Config, load_config, save_config
from ..utils.logging_utils import setup_logging
from ..utils.seed import set_seed

logger = logging.getLogger("textvae.train")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the text-latent VAE (paper §6)."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="path to a YAML config"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. model.encoder.latent_length=48",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="checkpoint path, or 'auto'"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def build_everything(config: Config) -> Trainer:
    """Instantiate the model, dataloaders and trainer from a config."""
    model = TextVAE.from_config(config)
    # The prior's tokenizer is only needed for the optional caption warm-start.
    tokenizer = model.prior.tokenizer if config.loss.aux_caption_ce_weight > 0 else None
    train_loader = build_dataloader(config.data, train=True, tokenizer=tokenizer)
    val_loader = build_dataloader(
        config.validation_data(), train=False, tokenizer=tokenizer
    )
    return Trainer(config, model, train_loader, val_loader)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config, overrides=args.overrides)
    if args.resume is not None:
        config.train.resume = args.resume
    config.validate()

    run_dir = Path(config.train.output_dir) / config.train.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, log_file=run_dir / "train.log")
    save_config(config, run_dir / "config.yaml")  # the exact config of this run
    set_seed(config.train.seed)

    logger.info("Run directory: %s", run_dir)
    trainer = build_everything(config)
    logger.info("Model: %s", trainer.state_summary())
    trainer.fit()


if __name__ == "__main__":
    main()
