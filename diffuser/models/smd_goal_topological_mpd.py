"""Goal-conditioned multiscale map context for the frozen SMD unary backbone.

All incoming map projections are zero at construction, so the old temporal
network is exactly reproduced before any adaptation. The deepest 4x4 feature
has receptive field 155x155 input pixels (larger than the 64x64 raster).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiscaleTaskEncoder(nn.Module):
    WIDTHS = (16, 24, 32, 48, 64)

    def __init__(self, local_dim=32, global_dim=32):
        super().__init__()
        self.levels = nn.ModuleList()
        self.downs = nn.ModuleList()
        previous = 4
        for level, width in enumerate(self.WIDTHS):
            self.levels.append(nn.Sequential(nn.Conv2d(previous, width, 3, padding=1), nn.SiLU(),
                                             nn.Conv2d(width, width, 3, padding=1), nn.SiLU()))
            if level != len(self.WIDTHS)-1:
                self.downs.append(nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1),
                                                nn.SiLU()))
            previous = width
        self.local_project = nn.Sequential(nn.Conv1d(sum(self.WIDTHS), local_dim, 1), nn.SiLU())
        self.global_project = nn.Sequential(nn.Linear(2*self.WIDTHS[-1], global_dim), nn.SiLU())
        self.local_dim = local_dim
        self.global_dim = global_dim

    def forward(self, raster):
        feature = raster
        pyramid = []
        for index, layer in enumerate(self.levels):
            feature = layer(feature)
            pyramid.append(feature)
            if index < len(self.downs):
                feature = self.downs[index](feature)
        deep = pyramid[-1]
        token = torch.cat((deep.mean((2, 3)), deep.amax((2, 3))), 1)
        return pyramid, self.global_project(token)

    def sample_local(self, pyramid, positions):
        grid = positions.clamp(-1, 1)[:, :, None, :]
        sampled = [F.grid_sample(level, grid, mode="bilinear", padding_mode="border",
                                 align_corners=True).squeeze(-1) for level in pyramid]
        return self.local_project(torch.cat(sampled, 1)).transpose(1, 2)


class GoalTopologicalMPD(nn.Module):
    HEATMAP_SIGMA_M = .07  # 64-grid spacing is 2/63 m; sigma spans about 2.2 cells.
    DEEPEST_RECEPTIVE_FIELD = 155

    def __init__(self, old_model, codec):
        super().__init__()
        self.model = old_model
        self.codec = codec
        self.map_encoder = MultiscaleTaskEncoder()
        blocks = []
        for stage in self.model.downs:
            blocks.extend(stage[:2])
        blocks.extend((self.model.mid_block1, self.model.mid_block2))
        for stage in self.model.ups:
            blocks.extend(stage[:2])
        self.local_injections = nn.ModuleList()
        self.global_injections = nn.ModuleList()
        for block in blocks:
            channels = block.blocks[1].block[0].out_channels
            local = nn.Conv1d(self.map_encoder.local_dim+4, channels, 1)
            global_projection = nn.Linear(self.map_encoder.global_dim, channels)
            nn.init.zeros_(local.weight)
            nn.init.zeros_(local.bias)
            nn.init.zeros_(global_projection.weight)
            nn.init.zeros_(global_projection.bias)
            self.local_injections.append(local)
            self.global_injections.append(global_projection)
        self.new_parameters = (tuple(self.map_encoder.parameters()) +
                               tuple(self.local_injections.parameters()) +
                               tuple(self.global_injections.parameters()))

    def task_raster(self, field, starts, goals):
        if starts.ndim != 2 or starts.shape != goals.shape or starts.shape[-1] != 2:
            raise ValueError("Expected start/goal [B,2]")
        batch = len(starts)
        size = field.raster.shape[-1]
        axis = torch.linspace(-1., 1., size, device=starts.device, dtype=starts.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack((xx, yy), -1)
        def heat(points):
            distance_sq = (xy[None]-points[:, None, None, :]).square().sum(-1)
            return torch.exp(-.5*distance_sq/self.HEATMAP_SIGMA_M**2)[:, None]
        return torch.cat((field.raster.expand(batch, -1, -1, -1), heat(starts), heat(goals)), 1)

    def encode_map(self, field, starts, goals):
        return self.map_encoder(self.task_raster(field, starts, goals))

    def context_from_state(self, x, field, encoded):
        physical = x[..., :2]*self.codec.scale[:2] + self.codec.mid[:2]
        pyramid, global_token = encoded
        local = self.map_encoder.sample_local(pyramid, physical)
        return torch.cat((field.raw_context(physical), local), -1), global_token

    def forward(self, x, time, context):
        local_context, global_token = context
        if x.ndim != 3 or x.shape[-1] != 4 or local_context.shape != (len(x), x.shape[1], 36):
            raise ValueError("Expected state [B,64,4] and local map context [B,64,36]")
        m = self.model
        time_embedding = m.time_mlp(time)
        local_channels = local_context.transpose(1, 2)
        projection_index = 0

        def inject(value):
            nonlocal projection_index
            sampled = F.interpolate(local_channels, size=value.shape[-1], mode="linear",
                                    align_corners=False)
            value = (value + self.local_injections[projection_index](sampled) +
                     self.global_injections[projection_index](global_token)[:, :, None])
            projection_index += 1
            return value

        hidden = x.transpose(1, 2)
        skips = []
        for first, second, attention, _, downsample in m.downs:
            hidden = inject(first(hidden, time_embedding))
            hidden = inject(second(hidden, time_embedding))
            hidden = attention(hidden)
            skips.append(hidden)
            hidden = downsample(hidden)
        hidden = inject(m.mid_block1(hidden, time_embedding))
        hidden = m.mid_attn(hidden)
        hidden = inject(m.mid_block2(hidden, time_embedding))
        for first, second, attention, _, upsample in m.ups:
            hidden = torch.cat((hidden, skips.pop()), 1)
            hidden = inject(first(hidden, time_embedding))
            hidden = inject(second(hidden, time_embedding))
            hidden = attention(hidden)
            hidden = upsample(hidden)
        if projection_index != len(self.local_injections):
            raise RuntimeError("Temporal map injection count changed")
        return m.final_conv(hidden).transpose(1, 2)
