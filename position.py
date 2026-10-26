# #jm 2026-10-19: ported verbatim from branch eeg-reference (reviewed Codex/Claude implementation); only RotaryPositions is used on master.
"""Positional encodings for EEG grids: axial RoPE over (t, x, y, z) and an absolute xyz embedding."""
import math

import torch
from torch import nn
from torch.nn import functional as F

# These were chosen once from physics (head size, electrode spacing) and are not tuned per run.
ROPE_BASE = 10000.0             # time axis: standard RoPE base
SPATIAL_ROPE_BASE = 100.0       # x, y, z axes
SPATIAL_SCALE = 100.0           # xyz are head coordinates in metres; x100 -> centimetres before rotating
FOURIER_OCTAVES = 6             # sin/cos bands of xyz with wavelengths 400, 200, 100, 50, 25, 12.5 mm
FOURIER_MAX_WAVELENGTH = 0.4    # metres, the longest band
FOURIER_EMBEDDING_SCALE = 0.02  # RMS of the xyz embedding, same as the token embedding init std


class FourierXYZ(nn.Module):
    """Absolute electrode-position embedding: sin/cos features of xyz through a small MLP (REVE-style).

    Added to the token embedding so a token knows WHERE on the scalp it is. The output is rescaled
    to a fixed RMS so it starts at the same magnitude as the token embedding.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.register_buffer("frequencies", 2.0 ** torch.arange(FOURIER_OCTAVES), persistent=False)
        self.proj = nn.Sequential(nn.Linear(3 + 6 * FOURIER_OCTAVES, n_embd),
                                  nn.GELU(), nn.Linear(n_embd, n_embd))

    def forward(self, xyz):
        xyz = xyz.float() / FOURIER_MAX_WAVELENGTH  # coordinates in units of the longest wavelength
        phase = xyz.unsqueeze(-1) * self.frequencies * (2 * math.pi)
        features = torch.cat((xyz, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)
        projected = self.proj(features)
        normalized = F.normalize(projected.float(), p=2, dim=-1, eps=1e-8)  # fix amplitude, keep direction
        return (normalized * (math.sqrt(self.n_embd) * FOURIER_EMBEDDING_SCALE)).to(projected.dtype)


class RotaryPositions(nn.Module):
    """Axial RoPE: the head dimension is split into groups, each rotated by one coordinate axis.

    dims = 't,x,y,z' head dimensions per axis (e.g. '32,8,8,8'); '' gives everything to time, or
    an equal split when spatial. Unallocated head dimensions pass through unrotated. Because q and k
    are rotated identically, attention scores depend only on coordinate DIFFERENCES (relative position).
    """

    def __init__(self, head_dim, spatial=False, dims=""):
        super().__init__()
        if dims:
            axis_dims = tuple(int(d.strip()) for d in dims.split(","))
        elif spatial:
            axis_dims = (2 * (head_dim // 8),) * 4
        else:
            axis_dims = (head_dim - head_dim % 2, 0, 0, 0)
        if (len(axis_dims) != 4 or any(d < 0 or d % 2 for d in axis_dims)
                or sum(axis_dims) > head_dim or axis_dims[0] == 0):
            raise ValueError("rope_dims must be four even dimensions t,x,y,z fitting the head")
        if spatial and any(d == 0 for d in axis_dims[1:]):
            raise ValueError("4D RoPE needs nonzero dimensions for all four axes")
        if not spatial and any(axis_dims[1:]):
            raise ValueError("Spatial rope_dims require channel_encoding='rope' or 'rope_fourier'")
        self.axis_dims = axis_dims
        self.rotary_dim = sum(axis_dims)
        for axis, dim in enumerate(axis_dims):
            base = ROPE_BASE if axis == 0 else SPATIAL_ROPE_BASE
            inv_freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / max(dim, 1))
            self.register_buffer(f"freq_{axis}", inv_freq, persistent=False)

    def prepare(self, positions, dtype=None):
        """cos/sin tables [B,1,L,rotary_dim/2] from positions [B,L,4]; computed once per forward, shared by all layers."""
        angles = []
        with torch.autocast(device_type=positions.device.type, enabled=False):
            for axis, dim in enumerate(self.axis_dims):
                if not dim:
                    continue
                scale = 1.0 if axis == 0 else SPATIAL_SCALE
                angles.append(positions[..., axis].float().unsqueeze(-1) * scale * getattr(self, f"freq_{axis}"))
            phase = torch.cat(angles, dim=-1)
            tables = (phase.cos().unsqueeze(1), phase.sin().unsqueeze(1))
        if dtype is not None:
            tables = tuple(table.to(dtype=dtype) for table in tables)
        return tables

    def apply_rotary(self, q, k, tables):
        """Rotate consecutive pairs of q and k by the prepared angles."""
        cos, sin = tables
        outputs = []
        with torch.autocast(device_type=q.device.type, enabled=False):
            for tensor in (q, k):
                pair = tensor[..., :self.rotary_dim].float().reshape(*tensor.shape[:-1], self.rotary_dim // 2, 2)
                a, b = pair.unbind(-1)
                rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
                outputs.append(torch.cat((rotated.flatten(-2).to(tensor.dtype), tensor[..., self.rotary_dim:]), dim=-1))
        return tuple(outputs)

    def forward(self, q, k, positions):
        return self.apply_rotary(q, k, self.prepare(positions))
