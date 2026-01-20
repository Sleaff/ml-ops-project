import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler, PyTorchProfiler, AdvancedProfiler
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from project.data import MyDataset
from project.dataset import NewsDataset
from project.model import Model

log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    # Get original working directory (Hydra changes cwd to outputs folder)
    orig_cwd = hydra.utils.get_original_cwd()

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    log.info(f"Set random seed to {cfg.seed}")

    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Use absolute paths (relative to original cwd)
    processed_path = os.path.join(orig_cwd, cfg.processed_data_path)
    data_path = os.path.join(orig_cwd, cfg.data_path)
    save_dir = Path(os.path.join(orig_cwd, cfg.save_dir))

    if not os.path.exists(processed_path):
        log.info("Preprocessing data...")
        dataset = MyDataset(Path(data_path))
        dataset.preprocess(Path(data_path) / "processed")

    dataset = NewsDataset(processed_path, tokenizer, max_length=cfg.max_length)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_size = int(cfg.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    log.info(f"Dataset split: {train_size} train, {val_size} val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=9, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=9, pin_memory=True)

    model = Model(model_name=cfg.model_name, lr=cfg.lr).to(device)
    log.info(f"Model: {cfg.model_name}, lr: {cfg.lr}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="best_model_{epoch:02d}_{val_loss:.4f}",
        monitor="loss/val_loss",
        mode="min",
        save_top_k=cfg.save_top_k,
        save_last=True,
    )

    if cfg.wandb_project:
        wandb_logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            log_model="all",
        )
    else:
        wandb_logger = None  # Or use pl.loggers.CSVLogger(save_dir)

    # Profiler
    profiler = None
    if cfg.get("profiler_path") is not None:
        
        abs_prof_path = os.path.join(orig_cwd, cfg.profiler_path)

        os.makedirs(abs_prof_path, exist_ok=True)

        if cfg.profiler_type=='simple': profiler = SimpleProfiler(dirpath=abs_prof_path, filename=f"{cfg.profiler_type}_profile_results")
        if cfg.profiler_type=='pytorch': profiler = PyTorchProfiler(dirpath=abs_prof_path, filename=f"{cfg.profiler_type}_profile_results")
        if cfg.profiler_type=='advanced': profiler = AdvancedProfiler(dirpath=abs_prof_path, filename=f"{cfg.profiler_type}_profile_results")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=cfg.epochs,
        limit_train_batches=cfg.limit_train_batches,
        callbacks=[checkpoint_callback],
        default_root_dir=save_dir,
        enable_progress_bar=True,
        log_every_n_steps=cfg.log_every_n_steps,
        logger=wandb_logger,
        deterministic=True,
        profiler=profiler
    )

    checkpoint_path = None
    if cfg.checkpoint_path is not None:
        checkpoint_path = os.path.join(orig_cwd, cfg.checkpoint_path)

    log.info(f"Starting training for {cfg.epochs} epochs")
    if checkpoint_path is not None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader, val_loader)

    log.info("Training complete!")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration:\n{cfg}")
    train(cfg)


if __name__ == "__main__":
    main()
