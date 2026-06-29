from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.stats import wasserstein_distance as _scipy_wasserstein

from cardiac_latent_ode.model.modules import reparameterize


def denormalize_trajectory(
    x_bvtc: torch.Tensor,
    mean_vtc: torch.Tensor,
    std_vtc: torch.Tensor,
) -> torch.Tensor:
    """Denormalize trajectories from standardized space to physical units."""
    if x_bvtc.ndim != 4:
        raise ValueError(
            f"Expected x_bvtc with shape [B, V, T, C], got {tuple(x_bvtc.shape)}"
        )
    if mean_vtc.ndim != 3 or std_vtc.ndim != 3:
        raise ValueError("mean_vtc and std_vtc must have shape [V, T, C]")
    if x_bvtc.shape[1:] != mean_vtc.shape or x_bvtc.shape[1:] != std_vtc.shape:
        raise ValueError(
            "Shape mismatch between trajectory and normalization stats: "
            f"x={tuple(x_bvtc.shape)}, mean={tuple(mean_vtc.shape)}, std={tuple(std_vtc.shape)}"
        )

    mean = mean_vtc.to(device=x_bvtc.device, dtype=x_bvtc.dtype).unsqueeze(0)
    std = std_vtc.to(device=x_bvtc.device, dtype=x_bvtc.dtype).unsqueeze(0)
    return x_bvtc * std + mean


def second_difference(vertices_bvtc: torch.Tensor) -> torch.Tensor:
    """Finite-difference acceleration along time, shape [B,V,T-2,3]."""
    if vertices_bvtc.ndim != 4:
        raise ValueError(
            f"vertices_bvtc must have shape [B, V, T, C], got {tuple(vertices_bvtc.shape)}"
        )
    if vertices_bvtc.shape[2] < 3:
        return vertices_bvtc.new_zeros(
            vertices_bvtc.shape[0], vertices_bvtc.shape[1], 0, vertices_bvtc.shape[3]
        )
    return (
        vertices_bvtc[:, :, 2:, :]
        - 2.0 * vertices_bvtc[:, :, 1:-1, :]
        + vertices_bvtc[:, :, :-2, :]
    )


def reconstruction_metric_sums(
    pred_bvtc: torch.Tensor,
    target_bvtc: torch.Tensor,
    faces_f3: torch.Tensor,
) -> dict[str, float]:
    """Return additive sums/counts for reconstruction metrics."""
    if pred_bvtc.shape != target_bvtc.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {tuple(pred_bvtc.shape)} vs {tuple(target_bvtc.shape)}"
        )

    diff = pred_bvtc - target_bvtc
    per_vertex_l2 = torch.linalg.norm(diff, dim=-1)
    euclid_sum = float(per_vertex_l2.sum().item())
    euclid_count = int(per_vertex_l2.numel())

    faces = faces_f3.to(device=pred_bvtc.device, dtype=torch.long)
    pred_fn = torch.nn.functional.normalize(
        torch.cross(
            pred_bvtc[:, faces[:, 1], :, :] - pred_bvtc[:, faces[:, 0], :, :],
            pred_bvtc[:, faces[:, 2], :, :] - pred_bvtc[:, faces[:, 0], :, :],
            dim=-1,
        ),
        dim=-1,
        eps=1e-8,
    )
    target_fn = torch.nn.functional.normalize(
        torch.cross(
            target_bvtc[:, faces[:, 1], :, :] - target_bvtc[:, faces[:, 0], :, :],
            target_bvtc[:, faces[:, 2], :, :] - target_bvtc[:, faces[:, 0], :, :],
            dim=-1,
        ),
        dim=-1,
        eps=1e-8,
    )
    cos_sim = (pred_fn * target_fn).sum(dim=-1).clamp(-1.0, 1.0)
    ang_deg = torch.arccos(cos_sim) * (180.0 / math.pi)
    normal_sum = float(ang_deg.sum().item())
    normal_count = int(ang_deg.numel())

    pred_acc = second_difference(pred_bvtc)
    target_acc = second_difference(target_bvtc)
    acc_error = torch.linalg.norm(pred_acc - target_acc, dim=-1)
    accel_sum = float(acc_error.sum().item())
    accel_count = int(acc_error.numel())

    return {
        "euclid_sum": euclid_sum,
        "euclid_count": float(euclid_count),
        "normal_sum": normal_sum,
        "normal_count": float(normal_count),
        "accel_sum": accel_sum,
        "accel_count": float(accel_count),
    }


def sample_prior_trajectories(
    model: Any,
    data: Any,
) -> torch.Tensor:
    """Sample trajectories from p(z|covariates) using test covariates as seeds."""
    if not hasattr(data, "num_frames"):
        raise ValueError("Data batch must contain num_frames")

    num_frames = data.num_frames[0].item()
    t_grid = model._build_time_grid(data, num_frames)

    prior_mean, prior_logvar = model.prior(data)
    z = reparameterize(prior_mean, prior_logvar)

    heart_rate = data.heart_rate.unsqueeze(1)
    theta_grid = model.phase_warping(heart_rate, t_grid=t_grid)
    generated = model.decoder(z, t_grid, theta_grid)
    return generated


def mmd_to_closest_real(
    real_meshes: np.ndarray,
    generated_meshes: np.ndarray,
) -> float:
    """Compute MMD as mean min distance from each generated mesh to real meshes."""
    if real_meshes.ndim != 3 or generated_meshes.ndim != 3:
        raise ValueError("Inputs must have shape [N, V, 3]")
    if real_meshes.shape[1:] != generated_meshes.shape[1:]:
        raise ValueError(
            f"Shape mismatch: real={tuple(real_meshes.shape)}, generated={tuple(generated_meshes.shape)}"
        )

    real = np.asarray(real_meshes, dtype=np.float32)
    generated = np.asarray(generated_meshes, dtype=np.float32)

    min_distances = np.empty((generated.shape[0],), dtype=np.float64)
    for i in range(generated.shape[0]):
        diff = real - generated[i][None, :, :]
        mean_vertex_dist = np.linalg.norm(diff, axis=2).mean(axis=1)
        min_distances[i] = float(mean_vertex_dist.min(initial=np.inf))

    return float(min_distances.mean(dtype=np.float64))


def temporal_smoothness_from_acceleration(generated_bvtc: torch.Tensor) -> float:
    """Return mean acceleration magnitude for generated trajectories."""
    acc = second_difference(generated_bvtc)
    if acc.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(acc, dim=-1).mean().item())


def wasserstein_distance_clinical_markers(
    real_markers: dict[str, np.ndarray],
    generated_markers: dict[str, np.ndarray],
) -> float:
    """Mean Wasserstein distance between generated and real clinical marker distributions.

    Each marker is normalized by the std of the real distribution before computing
    the distance, so all seven markers contribute on a comparable scale.  NaN/inf
    entries are removed per marker before the computation.  Returns NaN if no valid
    marker could be computed.
    """
    distances = []
    for key in real_markers:
        real = np.asarray(real_markers[key], dtype=np.float64)
        gen = np.asarray(generated_markers[key], dtype=np.float64)

        real = real[np.isfinite(real)]
        gen = gen[np.isfinite(gen)]

        if len(real) == 0 or len(gen) == 0:
            continue

        std = np.std(real)
        if std == 0.0 or not np.isfinite(std):
            continue

        distances.append(_scipy_wasserstein(real / std, gen / std))

    if not distances:
        return float("nan")
    return np.mean(distances).item()
