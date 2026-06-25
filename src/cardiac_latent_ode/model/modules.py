from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch_geometric.data import Data

from cardiac_latent_ode.model.spiral_conv import (
    SpiralEnblock,
    SpiralDeblock,
    SpiralConv,
)
from cardiac_latent_ode.model.mesh_operations import (
    scipy_to_torch_sparse,
    generate_transform_matrices,
    preprocess_spiral,
)
from cardiac_latent_ode.utils.utils import load_vtk_as_trimesh
from cardiac_latent_ode.utils.constants import LOGVAR_MIN, LOGVAR_MAX


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        # Keep LayerNorm identity-like at start
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def validate_and_reshape(data: Data) -> torch.Tensor:
    batch_size = int(data.num_graphs)
    if batch_size <= 0:
        raise ValueError(f"data.num_graphs must be positive, got {batch_size}")
    x = data.x
    if x is None:
        raise ValueError("data.x must not be None")
    if x.ndim != 2:
        raise ValueError(f"data.x must have shape [B*V*T, C], got {tuple(x.shape)}")
    if data.num_frames.numel() == 0:
        raise ValueError("data.num_frames must contain at least one entry")
    if not torch.equal(data.num_frames, data.num_frames[0].expand_as(data.num_frames)):
        raise ValueError("All graphs in a batch must have the same num_frames")

    num_channels = x.shape[1]
    num_frames = int(data.num_frames[0].item())
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if x.shape[0] % batch_size != 0:
        raise ValueError(
            "data.x first dimension must be divisible by batch size "
            f"(got {x.shape[0]} and {batch_size})"
        )

    total_vertices_over_time = x.shape[0] // batch_size
    if total_vertices_over_time % num_frames != 0:
        raise ValueError(
            "Per-graph trajectory size must be divisible by num_frames "
            f"(got {total_vertices_over_time} and {num_frames})"
        )

    num_vertices = total_vertices_over_time // num_frames
    return x.view(batch_size, num_vertices, num_frames, num_channels)


class CovariateTokenizer(nn.Module):
    """Embed covariates into per-covariate tokens.

    Embed binary sex (0/1) with an nn.Embedding, and treat the remaining
    covariates as continuous scalars embedded via a linear layer.

    Output shape: [B, 3, d_model]
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.sex_embedding = nn.Embedding(2, d_model)
        self.age_embedder = nn.Linear(1, d_model)
        self.bsa_embedder = nn.Linear(1, d_model)
        self.apply(init_weights)

    def forward(
        self, sex: torch.Tensor, age: torch.Tensor, bsa: torch.Tensor
    ) -> torch.Tensor:
        sex_token = self.sex_embedding(sex.squeeze(-1)).unsqueeze(1)  # [B,1,d]
        age_token = self.age_embedder(age).unsqueeze(1)  # [B,1,d]
        bsa_token = self.bsa_embedder(bsa).unsqueeze(1)  # [B,1,d]
        return torch.cat([sex_token, age_token, bsa_token], dim=1)


class AttentivePool1D(nn.Module):
    """Attention pooling over a [B,T,d] sequence.

    Uses a single learnable query vector and dot-product attention.
    """

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if self.d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.query = nn.Parameter(torch.empty(1, 1, self.d_model))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.mha(self.query.expand(x.size(0), -1, -1), x, x)
        # z shape: [B, 1, D] -> squeeze to [B, D]
        return z.squeeze(1)


class PhaseWarpModule(nn.Module):
    def __init__(
        self,
        warp_mlp_hidden: int = 16,
    ) -> None:
        super().__init__()
        if warp_mlp_hidden <= 0:
            raise ValueError(f"warp_mlp_hidden must be positive, got {warp_mlp_hidden}")
        self.warp_mlp = nn.Sequential(
            nn.Linear(3, warp_mlp_hidden),
            nn.Tanh(),
            nn.Linear(warp_mlp_hidden, 1),
            nn.Softplus(),
        )

    def forward(self, heart_rate: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        if heart_rate.ndim != 2:
            raise ValueError(
                f"heart_rate must have shape [B, 1], got {tuple(heart_rate.shape)}"
            )
        if t_grid.ndim != 1:
            raise ValueError(f"t_grid must be 1D, got shape {tuple(t_grid.shape)}")
        if t_grid.numel() < 2:
            raise ValueError("t_grid must contain at least 2 time points")

        phase_start = (2.0 * math.pi) * t_grid.view(1, -1, 1)
        phase_sin = torch.sin(phase_start).expand(heart_rate.shape[0], -1, -1)
        phase_cos = torch.cos(phase_start).expand(heart_rate.shape[0], -1, -1)
        hr_expand = heart_rate.unsqueeze(1).expand(-1, phase_start.shape[1], -1)
        hr_phase = torch.cat([hr_expand, phase_sin, phase_cos], dim=-1)
        speed = self.warp_mlp(hr_phase).squeeze(-1)
        speed_sum = speed.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(speed.dtype).eps
        )
        delta_theta = (2.0 * math.pi) * speed / speed_sum
        theta_grid = torch.cat(
            [
                torch.zeros(
                    heart_rate.shape[0],
                    1,
                    device=heart_rate.device,
                    dtype=heart_rate.dtype,
                ),
                torch.cumsum(delta_theta, dim=1)[:, :-1],
            ],
            dim=1,
        )
        return theta_grid


class ODEFunc(nn.Module):
    def __init__(
        self,
        latent_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if latent_size <= 0:
            raise ValueError(f"latent_size must be positive, got {latent_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self._phase_theta_grid: torch.Tensor | None = None
        self._phase_time_grid: torch.Tensor | None = None

        layers = []
        d_in = latent_size + 2
        for _ in range(max(1, int(num_layers)) - 1):
            layers += [nn.Linear(d_in, int(hidden_size)), nn.Tanh()]
            d_in = int(hidden_size)
        layers.append(nn.Linear(d_in, latent_size))
        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

        # Initialize last layer to small weights
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.trunc_normal_(last_layer.weight, mean=0.0, std=1e-5)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)

    def set_phase_trajectory_context(
        self,
        theta_grid: torch.Tensor | None,
        t_grid: torch.Tensor | None,
    ) -> None:
        if theta_grid is None or t_grid is None:
            self._phase_theta_grid = None
            self._phase_time_grid = None
            return
        self._phase_theta_grid = theta_grid
        self._phase_time_grid = t_grid

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        phase_theta_grid = self._phase_theta_grid
        phase_time_grid = self._phase_time_grid
        if phase_theta_grid is None or phase_time_grid is None:
            raise RuntimeError("Precomputed phase context is incomplete")

        # interpolate phase trajectory
        t_scalar = t.reshape(()).to(
            device=phase_time_grid.device, dtype=phase_time_grid.dtype
        )
        t_scalar = torch.clamp(
            t_scalar, min=phase_time_grid[0], max=phase_time_grid[-1]
        )
        right = int(
            torch.searchsorted(
                phase_time_grid, t_scalar.unsqueeze(0), right=True
            ).item()
        )
        right = min(max(right, 1), int(phase_time_grid.numel()) - 1)
        left = right - 1
        t_left = phase_time_grid[left]
        t_right = phase_time_grid[right]
        denom = (t_right - t_left).clamp_min(torch.finfo(phase_time_grid.dtype).eps)
        alpha = ((t_scalar - t_left) / denom).to(phase_theta_grid.dtype).view(1, 1)
        theta_left = phase_theta_grid[:, left : left + 1]
        theta_right = phase_theta_grid[:, right : right + 1]
        theta = theta_left + alpha * (theta_right - theta_left)

        phase_feat = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        dz = self.net(torch.cat([z, phase_feat], dim=-1))
        return dz


class PosteriorEncoder(nn.Module):

    def __init__(
        self,
        latent_size: int,
        template_mesh_path: str,
        gnn_channels: int = 64,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 4,
        transformer_ff_dim: int = 512,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_size <= 0:
            raise ValueError(f"latent_size must be positive, got {latent_size}")
        if latent_size % 2 != 0:
            raise ValueError(
                f"latent_size must be even for sinusoidal features, got {latent_size}"
            )
        if transformer_num_heads <= 0:
            raise ValueError(
                f"transformer_num_heads must be positive, got {transformer_num_heads}"
            )
        if latent_size % transformer_num_heads != 0:
            raise ValueError(
                "latent_size must be divisible by transformer_num_heads "
                f"(got {latent_size} and {transformer_num_heads})"
            )

        # Derive downsampling transform from template mesh
        template_mesh = load_vtk_as_trimesh(template_mesh_path)
        _, _, D, _, F, V = generate_transform_matrices(template_mesh, [4, 4, 4])
        spiral_indices = [
            preprocess_spiral(F[i], 9, V[i], 1) for i in range(len(F) - 1)
        ]
        down_transform_0 = scipy_to_torch_sparse(D[0])
        down_transform_1 = scipy_to_torch_sparse(D[1])
        down_transform_2 = scipy_to_torch_sparse(D[2])
        self.register_buffer("down_transform_0", down_transform_0)
        self.register_buffer("down_transform_1", down_transform_1)
        self.register_buffer("down_transform_2", down_transform_2)

        num_vert = down_transform_2.shape[0]
        self.gnn = nn.ModuleList(
            [
                SpiralEnblock(3, gnn_channels, spiral_indices[0]),
                SpiralEnblock(gnn_channels, gnn_channels, spiral_indices[1]),
                SpiralEnblock(gnn_channels, gnn_channels, spiral_indices[2]),
                nn.Linear(num_vert * gnn_channels, latent_size),
            ]
        )

        time_frequencies = torch.arange(1, (latent_size // 2) + 1).float()
        self.register_buffer(
            "time_frequencies",
            time_frequencies,
        )

        # Initialize transformer encoder for aggregating temporal features
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_size,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=transformer_num_layers,
            enable_nested_tensor=False,
        )

        self.attn_pool = AttentivePool1D(latent_size, n_heads=transformer_num_heads)
        self.covariate_tokenizer = CovariateTokenizer(latent_size)

        self.mu_head = nn.Linear(latent_size, latent_size)
        self.logvar_head = nn.Linear(latent_size, latent_size)

        self.apply(init_weights)

    def forward(
        self,
        x_t: torch.Tensor,
        sex: torch.Tensor,
        age: torch.Tensor,
        bsa: torch.Tensor,
        theta_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, T, C = x_t.shape

        # Reshape to (B*T, V, C) for GNN processing
        x_t = x_t.permute(0, 2, 1, 3).contiguous()  # [B, T, V, C]
        x_t = x_t.view(B * T, V, C)  # [B*T, V, C]

        down_transform_0 = cast(torch.Tensor, self.down_transform_0)
        down_transform_1 = cast(torch.Tensor, self.down_transform_1)
        down_transform_2 = cast(torch.Tensor, self.down_transform_2)

        x_t = self.gnn[0](x_t, down_transform_0)
        x_t = self.gnn[1](x_t, down_transform_1)
        x_t = self.gnn[2](x_t, down_transform_2)
        h_t = self.gnn[3](x_t.view(B * T, -1)).view(B, T, -1)

        # Compute positional embeddings
        time_frequencies = cast(torch.Tensor, self.time_frequencies)
        phases = theta_grid.unsqueeze(-1) * time_frequencies.reshape(1, 1, -1)
        pos_embedding = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)

        mesh_tokens = h_t + pos_embedding

        covariate_tokens = self.covariate_tokenizer(sex, age, bsa)
        seq = torch.cat([covariate_tokens, mesh_tokens], dim=1)

        out = self.transformer(seq)

        # We pool only over the time steps
        pooled = self.attn_pool(out[:, 3:, :])

        posterior_mean = self.mu_head(pooled)
        posterior_logvar = torch.clamp(
            self.logvar_head(pooled),
            min=LOGVAR_MIN,
            max=LOGVAR_MAX,
        )
        return posterior_mean, posterior_logvar


class PriorEncoder(nn.Module):

    def __init__(
        self,
        latent_size: int,
        mlp_hidden: int = 64,
    ) -> None:
        super().__init__()
        if latent_size <= 0:
            raise ValueError(f"latent_size must be positive, got {latent_size}")
        if mlp_hidden <= 0:
            raise ValueError(f"mlp_hidden must be positive, got {mlp_hidden}")
        self.latent_size = latent_size
        self.prior_mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 2 * latent_size),
        )

    def forward(
        self, sex: torch.Tensor, age: torch.Tensor, bsa: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        covariates = torch.cat([sex, age, bsa], dim=1)  # [B, 3]
        prior_mean, prior_logvar = torch.split(
            self.prior_mlp(covariates), self.latent_size, dim=1
        )
        prior_logvar = torch.clamp(prior_logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)
        return prior_mean, prior_logvar


class Decoder(nn.Module):

    def __init__(
        self,
        latent_size: int,
        template_mesh_path: str,
        gnn_channels: int = 64,
        node_hidden_size: int = 64,
        node_num_layers: int = 3,
        ode_rtol: float = 0.001,
        ode_atol: float = 0.0001,
    ) -> None:
        super().__init__()
        if latent_size <= 0:
            raise ValueError(f"latent_size must be positive, got {latent_size}")
        if gnn_channels <= 0:
            raise ValueError(f"gnn_channels must be positive, got {gnn_channels}")
        if node_hidden_size <= 0:
            raise ValueError(
                f"node_hidden_size must be positive, got {node_hidden_size}"
            )
        if node_num_layers <= 0:
            raise ValueError(f"node_num_layers must be positive, got {node_num_layers}")
        if ode_rtol <= 0:
            raise ValueError(f"ode_rtol must be positive, got {ode_rtol}")
        if ode_atol <= 0:
            raise ValueError(f"ode_atol must be positive, got {ode_atol}")

        self.template_mesh_path = template_mesh_path

        # Store ODE solver settings
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        # Derive upsampling transform from template mesh
        template_mesh = load_vtk_as_trimesh(template_mesh_path)
        self.register_buffer(
            "template_faces",
            torch.from_numpy(template_mesh.faces.astype("int64", copy=False)),
            persistent=False,
        )
        _, _, _, U, F, V = generate_transform_matrices(template_mesh, [4, 4, 4])
        spiral_indices = [
            preprocess_spiral(F[i], 9, V[i], 1) for i in range(len(F) - 1)
        ]
        up_transform_0 = scipy_to_torch_sparse(U[0])
        up_transform_1 = scipy_to_torch_sparse(U[1])
        up_transform_2 = scipy_to_torch_sparse(U[2])
        self.register_buffer("up_transform_0", up_transform_0)
        self.register_buffer("up_transform_1", up_transform_1)
        self.register_buffer("up_transform_2", up_transform_2)

        self.num_vert = up_transform_2.shape[0]
        self.gnn = nn.ModuleList(
            [
                nn.Linear(latent_size, self.num_vert * gnn_channels),
                SpiralDeblock(gnn_channels, gnn_channels, spiral_indices[2]),
                SpiralDeblock(gnn_channels, gnn_channels, spiral_indices[1]),
                SpiralDeblock(gnn_channels, gnn_channels, spiral_indices[0]),
                SpiralConv(gnn_channels, 3, spiral_indices[0]),
            ]
        )
        self.gnn.apply(init_weights)

        self.ode_func = ODEFunc(
            latent_size=latent_size,
            hidden_size=node_hidden_size,
            num_layers=node_num_layers,
        )

    def _integrate_ode(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        method: str = "dopri5",
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            odeint(
                self.ode_func,
                y0,
                t,
                method=method,
                rtol=self.ode_rtol,
                atol=self.ode_atol,
            ),
        )

    def forward(
        self,
        z: torch.Tensor,
        t_grid: torch.Tensor,
        theta_grid: torch.Tensor,
    ) -> torch.Tensor:
        B, D = z.shape
        T = len(t_grid)

        if theta_grid.ndim != 2:
            raise ValueError(
                f"theta_grid must have shape [B, T], got {tuple(theta_grid.shape)}"
            )
        if theta_grid.shape[0] != B or theta_grid.shape[1] != T:
            raise ValueError(
                "theta_grid shape must match batch/time dimensions from z and t_grid "
                f"(expected [{B}, {T}], got {list(theta_grid.shape)})"
            )

        # ODE expander
        self.ode_func.set_phase_trajectory_context(theta_grid, t_grid)
        try:
            z_t = self._integrate_ode(y0=z, t=t_grid).permute(1, 0, 2)  # [B, T, D]
        finally:
            self.ode_func.set_phase_trajectory_context(None, None)

        # Decode meshes from latent trajectory
        z_t = z_t.reshape(B * T, D)
        h_t = self.gnn[0](z_t).view(B * T, self.num_vert, -1)
        up_transform_2 = cast(torch.Tensor, self.up_transform_2)
        up_transform_1 = cast(torch.Tensor, self.up_transform_1)
        up_transform_0 = cast(torch.Tensor, self.up_transform_0)

        h_t = self.gnn[1](h_t, up_transform_2)
        h_t = self.gnn[2](h_t, up_transform_1)
        h_t = self.gnn[3](h_t, up_transform_0)
        x_t = self.gnn[4](h_t)
        x_t = x_t.view(B, T, -1, 3).permute(0, 2, 1, 3).contiguous()  # [B, V, T, C]
        return x_t
