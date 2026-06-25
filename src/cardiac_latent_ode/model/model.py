from typing import Callable

import torch
import torch.nn as nn
from torch_geometric.data import Data

from cardiac_latent_ode.model.modules import (
    Decoder,
    PhaseWarpModule,
    PosteriorEncoder,
    PriorEncoder,
    reparameterize,
    validate_and_reshape,
)


class CardiacLatentODE(nn.Module):
    """Latent ODE model for cardiac mesh trajectories."""

    def __init__(
        self,
        phase_warping: PhaseWarpModule,
        prior_encoder: PriorEncoder,
        posterior_encoder: PosteriorEncoder,
        decoder: Decoder,
        optimizer_cls: Callable[..., torch.optim.Optimizer],
        loss_fn: nn.Module,
    ) -> None:
        super().__init__()
        self.phase_warping = phase_warping
        self.prior_encoder = prior_encoder
        self.posterior_encoder = posterior_encoder
        self.decoder = decoder
        self.loss_fn = loss_fn

        self.optimizer = optimizer_cls(self.parameters())

    @staticmethod
    def _get_covariates(
        data: Data,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sex = data.sex.unsqueeze(1)
        age = data.age.unsqueeze(1)
        bsa = data.bsa.unsqueeze(1)
        heart_rate = data.heart_rate.unsqueeze(1)
        return sex, age, bsa, heart_rate

    @staticmethod
    def _build_time_grid(data: Data, num_frames: int) -> torch.Tensor:
        return torch.linspace(
            0.0,
            1.0,
            num_frames + 1,
            device=data.x.device,
            dtype=data.x.dtype,
        )[:-1]

    def _compute_loss(
        self,
        data: Data,
        sample_latents: bool,
    ) -> torch.Tensor:
        likelihood_mean, posterior_mean, posterior_logvar = self(
            data, sample_latents=sample_latents
        )
        prior_mean, prior_logvar = self.prior(data)
        return self.loss_fn(
            data,
            likelihood_mean,
            posterior_mean,
            posterior_logvar,
            prior_mean,
            prior_logvar,
        )

    def forward(
        self,
        data: Data,
        sample_latents: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample_latents is None:
            sample_latents = self.training

        x_t = validate_and_reshape(data)
        num_frames = x_t.shape[2]
        sex, age, bsa, heart_rate = self._get_covariates(data)

        # Compute the warped phase of each time step
        # Trajectory uses [0,1) to avoid duplicate start/end phases.
        t_grid = self._build_time_grid(data, num_frames)
        theta_grid = self.phase_warping(heart_rate, t_grid=t_grid)
        posterior_mean, posterior_logvar = self.posterior_encoder(
            x_t, sex, age, bsa, theta_grid
        )
        z = (
            reparameterize(posterior_mean, posterior_logvar)
            if sample_latents
            else posterior_mean
        )
        likelihood_mean = self.decoder(z, t_grid, theta_grid)
        return likelihood_mean, posterior_mean, posterior_logvar

    def prior(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        sex, age, bsa, _ = self._get_covariates(data)
        prior_mean, prior_logvar = self.prior_encoder(sex, age, bsa)
        return prior_mean, prior_logvar
    
    def posterior(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = validate_and_reshape(data)
        num_frames = x_t.shape[2]
        sex, age, bsa, heart_rate = self._get_covariates(data)

        # Compute the warped phase of each time step
        # Trajectory uses [0,1) to avoid duplicate start/end phases.
        t_grid = self._build_time_grid(data, num_frames)
        theta_grid = self.phase_warping(heart_rate, t_grid=t_grid)
        posterior_mean, posterior_logvar = self.posterior_encoder(
            x_t, sex, age, bsa, theta_grid
        )
        return posterior_mean, posterior_logvar

    def train_step(self, data: Data, grad_clip_norm: float = 0.0) -> torch.Tensor:
        self.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss = self._compute_loss(data, sample_latents=True)
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
        self.optimizer.step()

        return loss.detach()

    @torch.inference_mode()
    def val_step(self, data: Data) -> torch.Tensor:
        self.eval()
        return self._compute_loss(data, sample_latents=False).detach()

    def test_step(self, data: Data) -> torch.Tensor:
        raise NotImplementedError("test_step() must be implemented by subclasses")
