from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from cardiac_latent_ode.datamodule.datamodule import DataModule
from cardiac_latent_ode.model.model import CardiacLatentODE
from cardiac_latent_ode.trainer.trainer import Trainer
from cardiac_latent_ode.utils.pylogger import RichLogger
from cardiac_latent_ode.utils.utils import seed_everything

log = RichLogger(__name__)

# Get absolute path to configs directory
CONFIG_PATH = str(Path(__file__).parent.parent.parent / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="predict.yaml")
def predict(cfg: DictConfig) -> None:
    """Predict with the model.

    Args:
        cfg: Configuration composed by Hydra.
    """
    torch.set_float32_matmul_precision("medium")

    if cfg.get("seed"):
        seed_everything(cfg.seed)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: DataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: CardiacLatentODE = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    trainer_logger = getattr(trainer, "logger", None)
    if trainer_logger is not None:
        trainer_logger.log_hyperparams(cfg)

    log.info("Starting predicting!")
    trainer.predict(
        model=model,
        datamodule=datamodule,
        output_dir=cfg.output_dir,
        ckpt_path=cfg.ckpt_path,
        mode=cfg.mode,
        inference_split=cfg.inference_split,
        max_samples=cfg.max_samples,
    )


if __name__ == "__main__":
    predict()
