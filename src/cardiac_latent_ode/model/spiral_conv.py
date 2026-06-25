import torch
import torch.nn as nn
import torch.nn.functional as F


class SpiralConv(nn.Module):
    """Spiral convolution over ordered vertex neighborhoods."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        indices: torch.Tensor,
        dim: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if indices.ndim != 2:
            raise ValueError(
                f"indices must have shape [N, S], got {tuple(indices.shape)}"
            )
        if indices.numel() == 0:
            raise ValueError("indices must not be empty")
        if dim < 0:
            raise ValueError(f"dim must be non-negative, got {dim}")

        self.dim = dim
        self.register_buffer("indices", indices.long())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            gathered = torch.index_select(x, 0, self.indices.reshape(-1))
            x = gathered.reshape(n_nodes, -1)
        elif x.dim() == 3:
            if self.dim >= x.dim():
                raise ValueError(
                    f"dim must be less than x.dim() for 3D input, got {self.dim} and {x.dim()}"
                )
            batch_size = x.size(0)
            gathered = torch.index_select(x, self.dim, self.indices.reshape(-1))
            x = gathered.reshape(batch_size, n_nodes, -1)
        else:
            raise ValueError(f"x.dim() must be 2 or 3, got {x.dim()}")
        return self.layer(x)


def Pool(x: torch.Tensor, trans: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if not trans.is_sparse:
        raise ValueError("trans must be a sparse tensor")
    if dim < 0 or dim >= x.dim():
        raise ValueError(f"dim must be in [0, {x.dim() - 1}], got {dim}")

    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    shape = list(out.shape)
    shape[dim] = trans.size(0)
    result = torch.zeros(shape, dtype=out.dtype, device=out.device)
    index = row.view([1 if i != dim else -1 for i in range(out.dim())]).expand_as(out)
    return result.scatter_add_(dim, index, out)


class SpiralEnblock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        indices: torch.Tensor,
    ) -> None:
        super().__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)

    def forward(self, x: torch.Tensor, down_transform: torch.Tensor) -> torch.Tensor:
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        indices: torch.Tensor,
    ) -> None:
        super().__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)

    def forward(self, x: torch.Tensor, up_transform: torch.Tensor) -> torch.Tensor:
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out
