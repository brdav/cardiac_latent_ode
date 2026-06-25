import math

import pytest
import torch
from torch_geometric.data import Batch, Data

from cardiac_latent_ode.model.loss import BetaVAELoss
from cardiac_latent_ode.model.modules import PhaseWarpModule, validate_and_reshape

# ---------------------------------------------------------------------------
# 1. KL divergence is zero when posterior equals prior
# ---------------------------------------------------------------------------


def test_kl_zero_when_posterior_equals_prior():
    """KL(q || p) must be exactly 0 when q == p."""
    B, D = 4, 16
    mu = torch.randn(B, D)
    logvar = torch.randn(B, D)
    kl = BetaVAELoss.kl_diag_gaussian_diag_prior(mu, logvar, mu, logvar)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_nonnegative():
    """KL divergence must always be >= 0 (Gibbs inequality)."""
    B, D = 8, 32
    mu = torch.randn(B, D)
    logvar = torch.randn(B, D)
    prior_mean = torch.randn(B, D)
    prior_logvar = torch.randn(B, D)
    kl = BetaVAELoss.kl_diag_gaussian_diag_prior(mu, logvar, prior_mean, prior_logvar)
    assert kl.item() >= -1e-6


# ---------------------------------------------------------------------------
# 2. validate_and_reshape produces the correct [B, V, T, C] layout
# ---------------------------------------------------------------------------


def _make_batch(B: int, V: int, T: int) -> Data:
    """Build a minimal batched Data object with the HDF5-flattened layout."""
    # Each graph stores x as [V*T, 3] -> Batch stacks to [B*V*T, 3]
    x = torch.randn(B * V * T, 3)
    num_frames = torch.full((B,), T, dtype=torch.long)
    # ptr and batch are required by torch_geometric Batch
    batch_idx = torch.repeat_interleave(torch.arange(B), V * T)
    data = Batch(x=x, num_frames=num_frames, batch=batch_idx)
    data._num_graphs = B
    return data


def test_validate_and_reshape_output_shape():
    B, V, T = 3, 100, 20
    data = _make_batch(B, V, T)
    out = validate_and_reshape(data)
    assert out.shape == (B, V, T, 3)


def test_validate_and_reshape_values_preserved():
    """Reshaping must not permute or drop values."""
    B, V, T = 2, 50, 10
    data = _make_batch(B, V, T)
    out = validate_and_reshape(data)
    assert torch.allclose(out.reshape(B * V * T, 3), data.x)


def test_validate_and_reshape_rejects_mismatched_num_frames():
    B, V, T = 2, 50, 10
    data = _make_batch(B, V, T)
    data.num_frames = torch.tensor([10, 20], dtype=torch.long)
    with pytest.raises(ValueError, match="same num_frames"):
        validate_and_reshape(data)


# ---------------------------------------------------------------------------
# 3. PhaseWarpModule produces a valid (monotone) time grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,T", [(1, 20), (4, 50), (8, 10)])
def test_phase_warp_monotone(B: int, T: int):
    """theta_grid must start at 0 and be strictly increasing across time."""
    module = PhaseWarpModule(warp_mlp_hidden=16)
    heart_rate = torch.rand(B, 1) * 40 + 60  # 60–100 bpm
    t_grid = torch.linspace(0.0, 1.0, T)

    with torch.no_grad():
        theta = module(heart_rate, t_grid)  # [B, T]

    assert theta.shape == (B, T)
    # Must start at zero
    assert torch.allclose(theta[:, 0], torch.zeros(B), atol=1e-6)
    # Must be strictly increasing
    diffs = theta[:, 1:] - theta[:, :-1]
    assert (diffs > 0).all(), "theta_grid is not strictly increasing"


def test_phase_warp_total_span():
    """The sum of all delta-thetas must equal exactly 2π (the grid omits the
    final endpoint, so the last element of theta_grid is 2π - last_step)."""
    module = PhaseWarpModule(warp_mlp_hidden=16)
    heart_rate = torch.tensor([[75.0]])
    t_grid = torch.linspace(0.0, 1.0, 30)

    with torch.no_grad():
        theta = module(heart_rate, t_grid)  # [1, T]

    # Recover delta_thetas from the grid: θ[k] = sum(δ[0..k-1]), θ[0]=0
    # → delta = diff(θ) for the interior steps; last delta = 2π - θ[-1]
    diffs = theta[0, 1:] - theta[0, :-1]
    last_delta = 2 * math.pi * torch.ones(1) - theta[0, -1]
    total = diffs.sum() + last_delta
    assert total.item() == pytest.approx(2 * math.pi, rel=1e-4)
