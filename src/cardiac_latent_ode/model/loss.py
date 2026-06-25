from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.data import Data

from cardiac_latent_ode.model.modules import validate_and_reshape


class BetaVAELoss(nn.Module):
    """Beta-VAE loss with reconstruction and KL terms."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        self.beta = beta

    @staticmethod
    def reconstruction_loss(
        likelihood_mean: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        if likelihood_mean.shape != ground_truth.shape:
            raise ValueError(
                "likelihood_mean and ground_truth must match in shape "
                f"(got {tuple(likelihood_mean.shape)} and {tuple(ground_truth.shape)})"
            )
        return torch.sum(torch.mean((likelihood_mean - ground_truth) ** 2, dim=(0, 2)))

    @staticmethod
    def kl_diag_gaussian_diag_prior(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q || p) for diagonal Gaussians with diagonal prior variances.

        q = N(mu, diag(exp(logvar)))
        p = N(0, diag(prior_var))

        mu/logvar can be [B, D] or [B, D, M]. prior_var must broadcast.
        Returns mean KL over batch.
        """
        if mu.shape != logvar.shape:
            raise ValueError(
                f"mu/logvar shape mismatch: {tuple(mu.shape)} vs {tuple(logvar.shape)}"
            )
        if prior_mean.shape != prior_logvar.shape:
            raise ValueError(
                "prior_mean/prior_logvar shape mismatch: "
                f"{tuple(prior_mean.shape)} vs {tuple(prior_logvar.shape)}"
            )
        if mu.dim() not in (2, 3):
            raise ValueError(f"Expected mu/logvar with dim 2 or 3, got {mu.dim()}")

        # We keep prior_logvar in log-variance (not variance) units.
        prior_var = torch.exp(prior_logvar)
        var = torch.exp(logvar)

        mu_centered = mu - prior_mean
        kl_terms = (var + mu_centered**2) / prior_var - 1.0 + prior_logvar - logvar
        kl = 0.5 * torch.sum(kl_terms.reshape(mu.shape[0], -1), dim=1)
        return torch.mean(kl)

    def forward(
        self,
        data: Data,
        likelihood_mean: torch.Tensor,
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        ground_truth = validate_and_reshape(data)
        recon_loss = self.reconstruction_loss(likelihood_mean, ground_truth)
        kl_loss = self.kl_diag_gaussian_diag_prior(
            posterior_mean, posterior_logvar, prior_mean, prior_logvar
        )
        return recon_loss + self.beta * kl_loss
