from __future__ import annotations

import math
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tqdm import tqdm
import trimesh

from cardiac_latent_ode.model.modules import validate_and_reshape
from cardiac_latent_ode.trainer.metrics import (
    denormalize_trajectory,
    mmd_to_closest_real,
    reconstruction_metric_sums,
    sample_prior_trajectories,
    second_difference,
    wasserstein_distance_clinical_markers,
)
from cardiac_latent_ode.utils.constants import EPS
from cardiac_latent_ode.utils.mesh_processing import extract_clinical_markers
from cardiac_latent_ode.utils.pylogger import RichLogger
from cardiac_latent_ode.utils.utils import save_trimesh_as_vtk

log = RichLogger(__name__)


class Trainer:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        logger: Any | None = None,
        max_epochs: int = 150,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int | None = 10,
        progress_bar_update_interval: int = 10,
        max_generation_samples: int = 1000,
        subsample_generation_frames: int = 5,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            checkpoint_dir: Directory to write ``best.pt`` and ``last.pt``.
            logger: Optional logger; receives ``train_loss`` and ``val_loss`` per epoch.
            max_epochs: Maximum number of training epochs.
            grad_clip_norm: Max gradient norm; disabled when 0.
            early_stopping_patience: Stop after this many epochs with no val improvement.
                Disabled when ``None``.
            progress_bar_update_interval: Minimum interval (in seconds) between progress bar updates.
            max_generation_samples: Maximum total number of generated samples to produce during testing.
            subsample_generation_frames: Number of evenly-spaced frames to select per trajectory when
                computing MMD. Keeps the pairwise distance matrix tractable.
            verbose: Print per-epoch progress to logger.
        """
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if grad_clip_norm < 0:
            raise ValueError(
                f"grad_clip_norm must be non-negative, got {grad_clip_norm}"
            )
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, got {early_stopping_patience}"
            )
        if subsample_generation_frames <= 0:
            raise ValueError(
                f"subsample_generation_frames must be positive, got {subsample_generation_frames}"
            )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger = logger
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.progress_bar_update_interval = progress_bar_update_interval
        self.max_generation_samples = max_generation_samples
        self.subsample_generation_frames = subsample_generation_frames
        self.current_epoch = 0
        self.best_val_loss = math.inf
        self.best_model_path = ""
        self.test_metrics: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        model: Any,
        epoch: int,
        train_loss: float,
        val_loss: float,
        name: str,
    ) -> None:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": model.optimizer.state_dict(),
            "epoch_num": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.checkpoint_dir / name)

    def _load_checkpoint(self, model: Any, ckpt_path: str | Path) -> int:
        """Restore model and optimizer state from a checkpoint.

        Returns:
            The next epoch to start from (saved epoch + 1).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_val_loss = float(ckpt.get("val_loss", math.inf))
        return int(ckpt.get("epoch_num", 0)) + 1

    @staticmethod
    def _get_denormalization_stats(
        dataset: Any,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vertex_mean = getattr(dataset, "vertex_mean", None)
        vertex_std = getattr(dataset, "vertex_std", None)
        if vertex_mean is None or vertex_std is None:
            raise RuntimeError("Test dataset must expose vertex_mean and vertex_std")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if vertex_mean.shape[0] % num_frames != 0:
            raise RuntimeError(
                "vertex_mean length must be divisible by num_frames for denormalization"
            )

        num_vertices = vertex_mean.shape[0] // num_frames
        mean_vtc = vertex_mean.view(num_vertices, num_frames, -1)
        std_vtc = vertex_std.view(num_vertices, num_frames, -1)
        return mean_vtc, std_vtc

    def _prepare_inference(
        self,
        model: Any,
        datamodule: Any,
        ckpt_path: str | Path | None,
        split: Literal["test", "val", "train", "all"] = "test",
    ) -> tuple[torch.device, Any, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if ckpt_path is not None:
            self._load_checkpoint(model, ckpt_path)
            if self.verbose:
                log.info(f"Loaded checkpoint for testing: {ckpt_path}")

        model.to(device)
        model.eval()
        if self.verbose:
            log.info(f"Using device: {device}")

        if split == "test":
            datamodule.setup(stage="test")
            loader = datamodule.test_dataloader()
            dataset = getattr(datamodule, "dataset_test", None)
            if dataset is None:
                raise RuntimeError(
                    "datamodule.setup('test') did not initialize dataset_test"
                )
            return device, loader, dataset

        if split == "train":
            datamodule.setup(stage="fit")
            loader = datamodule.train_dataloader()
            dataset = getattr(datamodule, "dataset_train", None)
            if dataset is None:
                raise RuntimeError(
                    "datamodule.setup('fit') did not initialize dataset_train"
                )
            return device, loader, dataset

        if split == "val":
            datamodule.setup(stage="fit")
            loader = datamodule.val_dataloader()
            dataset = getattr(datamodule, "dataset_val", None)
            if dataset is None:
                raise RuntimeError(
                    "datamodule.setup('fit') did not initialize dataset_val"
                )
            return device, loader, dataset

        if split == "all":
            datamodule.setup(stage="fit")
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            dataset = getattr(datamodule, "dataset_train", None)
            if dataset is None:
                raise RuntimeError(
                    "datamodule.setup('fit') did not initialize dataset_train"
                )

            datamodule.setup(stage="test")
            test_loader = datamodule.test_dataloader()
            if getattr(datamodule, "dataset_test", None) is None:
                raise RuntimeError(
                    "datamodule.setup('test') did not initialize dataset_test"
                )
            loader = chain(train_loader, val_loader, test_loader)
            return device, loader, dataset

        raise ValueError(f"Unsupported inference split: {split}")

    def _predict_latent_embeddings(
        self,
        model: Any,
        test_loader: Any,
        device: torch.device,
        output_dir: Path,
        max_samples: int | None = None,
    ) -> Path:
        case_ids_all: list[str] = []
        latents_all: list[np.ndarray] = []
        num_processed = 0

        with torch.inference_mode():
            with tqdm(
                test_loader,
                desc="Predicting latents",
                miniters=self.progress_bar_update_interval,
                maxinterval=120,
                disable=not self.verbose,
            ) as pbar:
                for data in pbar:
                    if max_samples is not None and num_processed >= max_samples:
                        break

                    case_ids_all.extend(data.case_id)
                    data = data.to(device)
                    prior_mean, prior_logvar = model.prior(data)
                    posterior_mean, _ = model.posterior(data)
                    # Compute residual embeddings
                    embedding = (posterior_mean - prior_mean) / torch.exp(
                        0.5 * prior_logvar
                    ).clamp(min=EPS)
                    if max_samples is not None:
                        remaining = max_samples - num_processed
                        if remaining <= 0:
                            break
                        if embedding.shape[0] > remaining:
                            embedding = embedding[:remaining]
                            case_ids_all = case_ids_all[: num_processed + remaining]

                    latents_all.append(
                        embedding.detach().cpu().numpy().astype(np.float32)
                    )
                    num_processed += int(posterior_mean.shape[0])

        if not latents_all:
            raise RuntimeError("No test samples were processed")

        output_path = output_dir / "latents.npz"
        np.savez_compressed(
            output_path,
            case_id=np.asarray(case_ids_all, dtype=str),
            z=np.concatenate(latents_all, axis=0),
        )
        return output_path

    def _predict_mesh_reconstructions(
        self,
        model: Any,
        test_loader: Any,
        dataset: Any,
        device: torch.device,
        output_dir: Path,
        max_samples: int | None = None,
    ) -> Path:
        decoder = getattr(model, "decoder", None)
        faces = getattr(decoder, "template_faces", None)
        if faces is None:
            raise RuntimeError(
                "Model decoder does not expose template_faces; cannot export meshes"
            )

        faces_np = np.asarray(faces.detach().cpu().numpy(), dtype=np.int64)
        if faces_np.ndim != 2 or faces_np.shape[1] != 3:
            raise RuntimeError(
                f"template_faces must have shape [F, 3], got {tuple(faces_np.shape)}"
            )

        sample_index = 0

        with torch.inference_mode():
            with tqdm(
                test_loader,
                desc="Predicting meshes",
                miniters=self.progress_bar_update_interval,
                maxinterval=120,
                disable=not self.verbose,
            ) as pbar:
                for data in pbar:
                    if max_samples is not None and sample_index >= max_samples:
                        break

                    case_ids = data.case_id
                    data = data.to(device)
                    x_recon_norm, _, _ = model(data, sample_latents=False)

                    num_frames = x_recon_norm.shape[2]
                    mean_vtc, std_vtc = self._get_denormalization_stats(
                        dataset, num_frames
                    )
                    x_recon = denormalize_trajectory(x_recon_norm, mean_vtc, std_vtc)

                    x_recon_np = (
                        x_recon.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    if max_samples is not None:
                        remaining = max_samples - sample_index
                        if remaining <= 0:
                            break
                        x_recon_np = x_recon_np[:remaining]
                        case_ids = case_ids[:remaining]

                    for batch_idx, case_id in enumerate(case_ids):
                        case_dir = output_dir / case_id
                        case_dir.mkdir(parents=True, exist_ok=True)
                        x_sample = x_recon_np[batch_idx]
                        num_frames = x_sample.shape[1]
                        for frame_idx in range(num_frames):
                            mesh_sample = trimesh.Trimesh(
                                vertices=x_sample[:, frame_idx, :],
                                faces=faces_np,
                            )
                            save_trimesh_as_vtk(mesh_sample, case_dir / f"frame_{frame_idx:03d}.vtk")
                        sample_index += 1

        if sample_index == 0:
            raise RuntimeError("No test samples were processed")

        return output_dir

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        device: torch.device,
    ) -> tuple[float, float]:
        """Run one train + val epoch. Returns mean (train_loss, val_loss)."""
        train_losses = []
        with tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch} [train]",
            miniters=self.progress_bar_update_interval,
            maxinterval=120,
            leave=False,
            disable=not self.verbose,
        ) as pbar:
            for batch in pbar:
                loss = model.train_step(
                    batch.to(device), grad_clip_norm=self.grad_clip_norm
                )
                train_losses.append(loss)
                pbar.set_postfix(
                    loss=f"{sum(train_losses) / len(train_losses):.6f}", refresh=False
                )

        val_losses = []
        with tqdm(
            val_loader,
            desc=f"Epoch {self.current_epoch} [val]  ",
            miniters=self.progress_bar_update_interval,
            maxinterval=120,
            leave=False,
            disable=not self.verbose,
        ) as pbar:
            for batch in pbar:
                loss = model.val_step(batch.to(device))
                val_losses.append(loss)
                pbar.set_postfix(
                    loss=f"{sum(val_losses) / len(val_losses):.6f}", refresh=False
                )

        train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        return train_loss, val_loss

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self, model: Any, datamodule: Any, ckpt_path: str | Path | None = None
    ) -> None:
        """Train the model, saving best and last checkpoints.

        Args:
            model: Must implement ``train_step(batch, grad_clip_norm)`` and ``val_step(batch)``.
            datamodule: Provides ``train_dataloader()`` and ``val_dataloader()``.
            ckpt_path: Optional checkpoint path to resume training from.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if self.verbose:
            log.info(f"Using device: {device}")

        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        start_epoch = 0
        if ckpt_path is not None:
            start_epoch = self._load_checkpoint(model, ckpt_path)
            if self.verbose:
                log.info(f"Resumed from {ckpt_path}, starting at epoch {start_epoch}.")

        epochs_without_improvement = 0

        for epoch in range(start_epoch, self.max_epochs):
            self.current_epoch = epoch
            train_loss, val_loss = self._run_epoch(
                model, train_loader, val_loader, device
            )

            if self.logger:
                self.logger.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss},
                    step=epoch,
                )
            if self.verbose:
                log.info(
                    f"Epoch {epoch}/{self.max_epochs - 1}  "
                    f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
                )

            self._save_checkpoint(model, epoch, train_loss, val_loss, "last.pt")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(model, epoch, train_loss, val_loss, "best.pt")
                self.best_model_path = str(self.checkpoint_dir / "best.pt")
                epochs_without_improvement = 0
                if self.verbose:
                    log.info(f"  New best val_loss: {val_loss:.6f}")
            else:
                epochs_without_improvement += 1

            if (
                self.early_stopping_patience is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                if self.verbose:
                    log.info(
                        f"Early stopping triggered after "
                        f"{epochs_without_improvement} epochs without improvement."
                    )
                break

    def test(
        self, model: Any, datamodule: Any, ckpt_path: str | Path | None = None
    ) -> None:
        """Test the model using the provided datamodule.

        Args:
            model: Must implement ``test_step``.
            datamodule: Provides ``test_dataloader()``.
            ckpt_path: Optional checkpoint to load weights from before testing.
        """
        device, test_loader, dataset = self._prepare_inference(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            split="test",
        )

        decoder = getattr(model, "decoder", None)
        faces = getattr(decoder, "template_faces", None)
        if faces is None:
            raise RuntimeError(
                "Model decoder does not expose template_faces; cannot compute normal-angle errors"
            )

        metric_totals = {
            "euclid_sum": 0.0,
            "euclid_count": 0.0,
            "normal_sum": 0.0,
            "normal_count": 0.0,
            "accel_sum": 0.0,
            "accel_count": 0.0,
        }
        real_meshes_sub: list[np.ndarray] = []
        generated_meshes_sub: list[np.ndarray] = []
        smoothness_sum = 0.0
        smoothness_count = 0
        generation_samples_count = 0
        real_clinical_markers_all: list[dict] = []
        generated_clinical_markers_all: list[dict] = []

        with torch.inference_mode():
            with tqdm(
                test_loader,
                desc="Testing",
                miniters=self.progress_bar_update_interval,
                maxinterval=120,
                disable=not self.verbose,
            ) as pbar:
                for data in pbar:
                    data = data.to(device)
                    x_target_norm = validate_and_reshape(data)
                    x_recon_norm, _, _ = model(data, sample_latents=False)

                    num_frames = x_target_norm.shape[2]
                    mean_vtc, std_vtc = self._get_denormalization_stats(
                        dataset, num_frames
                    )

                    x_target = denormalize_trajectory(x_target_norm, mean_vtc, std_vtc)
                    x_recon = denormalize_trajectory(x_recon_norm, mean_vtc, std_vtc)

                    batch_metrics = reconstruction_metric_sums(x_recon, x_target, faces)
                    for key, value in batch_metrics.items():
                        metric_totals[key] += float(value)

                    if generation_samples_count < self.max_generation_samples:

                        generated = sample_prior_trajectories(model, data)
                        generated = denormalize_trajectory(generated, mean_vtc, std_vtc)

                        T = x_target.shape[2]
                        K = min(self.subsample_generation_frames, T)
                        # Select K evenly spaced frames, excluding the final frame
                        # since data is periodic.
                        frame_idx = torch.linspace(
                            0, T, K + 1, dtype=torch.long, device=device
                        )[:-1]

                        # Flatten subsampled frames into vertex dimension: [B, V*K, 3]
                        real_sub = x_target[:, :, frame_idx, :].reshape(
                            x_target.shape[0], -1, 3
                        )
                        gen_sub = generated[:, :, frame_idx, :].reshape(
                            generated.shape[0], -1, 3
                        )

                        real_meshes_sub.append(
                            real_sub.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        generated_meshes_sub.append(
                            gen_sub.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )

                        generated_acc_mag = torch.linalg.norm(
                            second_difference(generated), dim=-1
                        )
                        smoothness_sum += generated_acc_mag.sum().item()
                        smoothness_count += generated_acc_mag.numel()

                        # Unnormalize BSA for clinical marker computation
                        bsa = (
                            (data.bsa * dataset.bsa_std + dataset.bsa_mean)
                            .cpu()
                            .numpy()
                        )
                        real_clinical_markers = extract_clinical_markers(
                            x_target, bsa=bsa
                        )
                        generated_clinical_markers = extract_clinical_markers(
                            generated, bsa=bsa
                        )
                        real_clinical_markers_all.append(real_clinical_markers)
                        generated_clinical_markers_all.append(
                            generated_clinical_markers
                        )

                        generation_samples_count += generated.shape[0]

                    mean_euclid = (
                        metric_totals["euclid_sum"] / metric_totals["euclid_count"]
                        if metric_totals["euclid_count"] > 0
                        else 0.0
                    )
                    pbar.set_postfix(
                        recon_vertex_l2=f"{mean_euclid:.6f}", refresh=False
                    )

        if not real_meshes_sub or not generated_meshes_sub:
            raise RuntimeError("No test samples were processed")

        real_sub = np.concatenate(real_meshes_sub, axis=0)
        generated_sub = np.concatenate(generated_meshes_sub, axis=0)
        mmd = mmd_to_closest_real(real_sub, generated_sub)

        # Concatenate per-batch clinical markers and compute Wasserstein distance
        real_clinical_concat = {
            k: np.concatenate([m[k] for m in real_clinical_markers_all], axis=0)
            for k in real_clinical_markers_all[0]
        }
        generated_clinical_concat = {
            k: np.concatenate([m[k] for m in generated_clinical_markers_all], axis=0)
            for k in generated_clinical_markers_all[0]
        }
        clinical_wasserstein = wasserstein_distance_clinical_markers(
            real_clinical_concat, generated_clinical_concat
        )

        reconstruction_vertex_error = (
            metric_totals["euclid_sum"] / metric_totals["euclid_count"]
            if metric_totals["euclid_count"] > 0
            else 0.0
        )
        reconstruction_normal_error_deg = (
            metric_totals["normal_sum"] / metric_totals["normal_count"]
            if metric_totals["normal_count"] > 0
            else 0.0
        )
        reconstruction_acceleration_error = (
            metric_totals["accel_sum"] / metric_totals["accel_count"]
            if metric_totals["accel_count"] > 0
            else 0.0
        )
        generative_temporal_smoothness = (
            smoothness_sum / float(smoothness_count) if smoothness_count > 0 else 0.0
        )

        metrics = {
            "test/recon_vertex_euclidean_error": float(reconstruction_vertex_error),
            "test/recon_surface_normal_angular_error_deg": float(
                reconstruction_normal_error_deg
            ),
            "test/recon_vertex_acceleration_error": float(
                reconstruction_acceleration_error
            ),
            "test/gen_mmd": float(mmd),
            "test/gen_temporal_smoothness_acceleration": float(
                generative_temporal_smoothness
            ),
            "test/gen_clinical_wasserstein": float(clinical_wasserstein),
        }
        self.test_metrics = metrics

        if self.logger:
            self.logger.log_metrics(metrics, step=self.current_epoch)

        if self.verbose:
            log.info("Test metrics:")
            for name, value in metrics.items():
                if math.isnan(value):
                    log.info(f"  {name}=nan")
                else:
                    log.info(f"  {name}={value:.6f}")

    def predict(
        self,
        model: Any,
        datamodule: Any,
        output_dir: str | Path,
        mode: Literal["latent", "mesh"],
        ckpt_path: str | Path | None = None,
        inference_split: Literal["all", "test", "val", "train"] = "all",
        max_samples: int | None = None,
    ) -> Path:
        """Run deterministic prediction on the test split and save outputs to disk.

        Args:
            model: Trained model used for prediction.
            datamodule: Provides ``test_dataloader()`` and exposes ``dataset_test`` after setup.
            output_dir: Directory where predictions are written.
            mode: ``"latent"`` to save posterior-mean embeddings, ``"mesh"`` to save
                reconstructed meshes.
            ckpt_path: Optional checkpoint to load before prediction.
            inference_split: Which datamodule split to run inference on.
            max_samples: Optional maximum number of samples to process.

        Returns:
            The output path containing the saved predictions.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        device, test_loader, dataset = self._prepare_inference(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            split=inference_split,
        )

        if mode == "latent":
            return self._predict_latent_embeddings(
                model=model,
                test_loader=test_loader,
                device=device,
                output_dir=output_path,
                max_samples=max_samples,
            )
        if mode == "mesh":
            return self._predict_mesh_reconstructions(
                model=model,
                test_loader=test_loader,
                dataset=dataset,
                device=device,
                output_dir=output_path,
                max_samples=max_samples,
            )
        raise ValueError(f"Unsupported prediction mode: {mode}")
