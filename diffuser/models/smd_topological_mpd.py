"""Checkpoint-preserving SMD unary with shared spatial map context.

The 64x64 map is encoded once per layout. At each reverse step, current
trajectory supports bilinearly sample the encoded grid and enter zero-started
additive projections at every temporal U-Net residual block.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diffuser.models.smd_map_context import StaticMapContext


class SpatialMapField:
    def __init__(self, obstacles, *, device, clearance_scale, size=64):
        self.exact = StaticMapContext(obstacles, device=device, scale=clearance_scale)
        axis = torch.linspace(-1., 1., size, device=device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((x, y), -1)
        clearance, _ = self.exact.clearance_gradient(positions)
        occupancy = (clearance < 0).to(clearance.dtype)
        normalized_sdf = (clearance / clearance_scale).clamp(-4, 4)
        self.raster = torch.stack((occupancy, normalized_sdf), 0)[None]
        self.obstacles = obstacles

    def raw_context(self, positions):
        return self.exact(positions)

    def exact_clearance(self, positions):
        return self.exact.clearance_gradient(positions)[0]


class DilatedMapBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.conv(x))


class TinySpatialMapEncoder(nn.Module):
    """16-channel full-resolution grid; dilations cover nearly the whole map."""
    def __init__(self, width=16):
        super().__init__()
        self.input = nn.Sequential(nn.Conv2d(2, width, 3, padding=1), nn.SiLU())
        self.blocks = nn.Sequential(*(DilatedMapBlock(width, d) for d in (1, 2, 4, 8, 16)))
        self.output = nn.Conv2d(width, width, 1)

    def forward(self, raster):
        return self.output(self.blocks(self.input(raster)))


class TopologicalMapMPD(nn.Module):
    def __init__(self, old_model, codec, map_width=16):
        super().__init__()
        self.model = old_model
        self.codec = codec
        self.map_encoder = TinySpatialMapEncoder(map_width)
        self.map_dim = map_width + 4  # learned F_map + exact local D, gradient, midpoint D
        blocks = []
        for stage in self.model.downs:
            blocks.extend(stage[:2])
        blocks.extend((self.model.mid_block1, self.model.mid_block2))
        for stage in self.model.ups:
            blocks.extend(stage[:2])
        self.map_projections = nn.ModuleList()
        for block in blocks:
            out_channels = block.blocks[1].block[0].out_channels
            projection = nn.Conv1d(self.map_dim, out_channels, 1)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.map_projections.append(projection)
        self.new_parameters = tuple(self.map_encoder.parameters()) + tuple(self.map_projections.parameters())

    def encode_map(self, field):
        return self.map_encoder(field.raster)

    def context_from_state(self, x, field, encoded):
        physical = x[..., :2] * self.codec.scale[:2] + self.codec.mid[:2]
        grid = physical.clamp(-1, 1)[:, :, None, :]
        sampled = F.grid_sample(encoded.expand(len(x), -1, -1, -1), grid,
                                mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return torch.cat((field.raw_context(physical), sampled), -1)

    def forward(self, x, time, context):
        if x.ndim != 3 or x.shape[-1] != 4 or context.shape != (*x.shape[:2], self.map_dim):
            raise ValueError(f"Expected x [B,64,4], context [B,64,{self.map_dim}]")
        m = self.model
        c_emb = m.time_mlp(time)
        map_channels = context.transpose(1, 2)
        projection_index = 0

        def inject(value):
            nonlocal projection_index
            local = F.interpolate(map_channels, size=value.shape[-1],
                                  mode="linear", align_corners=False)
            result = value + self.map_projections[projection_index](local)
            projection_index += 1
            return result

        hidden = x.transpose(1, 2)
        skips = []
        for resnet, resnet2, attn_self, _, downsample in m.downs:
            hidden = inject(resnet(hidden, c_emb))
            hidden = inject(resnet2(hidden, c_emb))
            hidden = attn_self(hidden)
            skips.append(hidden)
            hidden = downsample(hidden)
        hidden = inject(m.mid_block1(hidden, c_emb))
        hidden = m.mid_attn(hidden)
        hidden = inject(m.mid_block2(hidden, c_emb))
        for resnet, resnet2, attn_self, _, upsample in m.ups:
            hidden = torch.cat((hidden, skips.pop()), 1)
            hidden = inject(resnet(hidden, c_emb))
            hidden = inject(resnet2(hidden, c_emb))
            hidden = attn_self(hidden)
            hidden = upsample(hidden)
        if projection_index != len(self.map_projections):
            raise RuntimeError("Map projection/block mismatch")
        return m.final_conv(hidden).transpose(1, 2)
