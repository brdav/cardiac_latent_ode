import hydra
import torch
from pathlib import Path
from omegaconf import DictConfig

from cardiac_latent_ode.datamodule.datamodule import DataModule
from cardiac_latent_ode.model.model import CardiacLatentODE
from cardiac_latent_ode.trainer.trainer import Trainer
from cardiac_latent_ode.utils.utils import seed_everything
from cardiac_latent_ode.utils.pylogger import RichLogger

log = RichLogger(__name__)

# Get absolute path to configs directory
CONFIG_PATH = str(Path(__file__).parent.parent.parent / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    """Train the model and optionally evaluate on a testset using best weights.

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

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = getattr(trainer, "best_model_path", None)
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")


if __name__ == "__main__":
    train()
