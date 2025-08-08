# main.py
import argparse
import logging
import deepspeed

from config import ProjectConfig
from dataset import ImageDataset
from models.vae import TextVAE
from trainer import Trainer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume from.",
    )
    # DeepSpeed will add its own arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    config = ProjectConfig()

    # --- 1. Create Unified Model and Dataset ---
    model = TextVAE(config)
    # The training dataset is passed to DeepSpeed, which creates the dataloader
    train_dataset = ImageDataset(
        root=config.data.data_dir, image_size=config.data.image_size
    )

    # --- 2. Initialize DeepSpeed Engine ---
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
    )

    # --- 3. Initialize Trainer ---
    # The trainer now receives the DeepSpeed engine and dataloader
    trainer = Trainer(
        config=config, model_engine=model_engine, train_dataloader=train_dataloader
    )
    # model_engine.module.diffusion_decoder.unet.enable_xformers_memory_efficient_attention()
    # logger.info("Enabled xformers memory-efficient attention on the U-Net.")

    # --- 4. Handle Resuming ---
    if args.resume_from:
        # DeepSpeed handles loading its complex sharded state
        trainer.load_checkpoint(args.resume_from)

    # --- 5. Start Training ---
    trainer.train()


if __name__ == "__main__":
    main()
