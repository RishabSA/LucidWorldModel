import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn

n_actions = 8
action_dim = 32
cond_dim = 256
fourier_dim = 64
n_groups = 8
channel_mults = (1, 2, 4, 4)
blocks_per_level = 2
middle_blocks = 2


def group_norm(x: torch.Tensor, groups: int, eps: float = 1e-5) -> torch.Tensor:
    # Primitive ops on purpose: onnxruntime-web has no WebGPU kernel for ONNX GroupNormalization
    grouped = x.reshape(
        x.shape[0], groups, -1
    )  # shape: (batch, groups, channels / groups * height * width)
    mean = grouped.mean(dim=-1, keepdim=True)
    centered = grouped - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)

    return (centered / torch.sqrt(var + eps)).reshape(x.shape)


class GroupNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            group_norm(x, n_groups) * self.weight[:, None, None]
            + self.bias[:, None, None]
        )


class FourierFeatures(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randn(1, dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * x[:, None] * self.weight  # shape: (batch, dim / 2)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = GroupNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # AdaGN starts as plain group norm
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm(x)))

        scale, shift = self.cond_proj(cond)[:, :, None, None].chunk(
            2, dim=1
        )  # shape: (batch, out_channels, 1, 1)
        h = group_norm(h, n_groups) * (1 + scale) + shift
        h = self.conv2(F.silu(h))

        return self.skip(x) + h


class UNet(nn.Module):
    def __init__(self, width: int, context: int) -> None:
        super().__init__()
        channels = [width * mult for mult in channel_mults]

        self.noise_fourier = FourierFeatures(fourier_dim)
        self.aug_fourier = FourierFeatures(fourier_dim)
        self.action_embedding = nn.Embedding(n_actions, action_dim)
        self.cond_in = nn.Linear(2 * fourier_dim + context * action_dim, cond_dim)
        self.cond_out = nn.Linear(cond_dim, cond_dim)

        self.stem = nn.Conv2d(3 * (context + 1), width, 3, padding=1)

        self.encoder = nn.ModuleList()
        skip_channels = []
        in_channels = width

        for out_channels in channels:
            blocks = nn.ModuleList()

            for _ in range(blocks_per_level):
                blocks.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
                skip_channels.append(out_channels)
            self.encoder.append(blocks)

        self.downsamples = nn.ModuleList(
            [nn.Conv2d(c, c, 3, stride=2, padding=1) for c in channels[:-1]]
        )

        self.middle = nn.ModuleList(
            [ResBlock(in_channels, in_channels) for _ in range(middle_blocks)]
        )

        self.decoder = nn.ModuleList()

        for out_channels in reversed(channels):
            blocks = nn.ModuleList()

            for _ in range(blocks_per_level):
                blocks.append(ResBlock(in_channels + skip_channels.pop(), out_channels))
                in_channels = out_channels
            self.decoder.append(blocks)

        self.upsamples = nn.ModuleList(
            [nn.Conv2d(c, c, 3, padding=1) for c in reversed(channels[1:])]
        )

        self.head_norm = GroupNorm(width)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        c_noise: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
        sigma_aug: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenating action embeddings keeps track of which action is the current one
        actions_flat = self.action_embedding(actions).flatten(
            start_dim=1
        )  # shape: (batch, context * action_dim)
        cond = torch.cat(
            [self.noise_fourier(c_noise), self.aug_fourier(sigma_aug), actions_flat],
            dim=1,
        )
        cond = self.cond_out(F.silu(self.cond_in(cond)))  # shape: (batch, cond_dim)

        h = self.stem(torch.cat([x, context], dim=1))  # shape: (batch, width, 64, 64)

        skips = []
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                h = block(h, cond)
                skips.append(h)

            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        for block in self.middle:
            h = block(h, cond)

        for level, blocks in enumerate(self.decoder):
            for block in blocks:
                h = block(torch.cat([h, skips.pop()], dim=1), cond)

            if level < len(self.upsamples):
                h = self.upsamples[level](
                    F.interpolate(h, scale_factor=2.0, mode="nearest")
                )

        return self.head(F.silu(self.head_norm(h)))  # shape: (batch, 3, 64, 64)


class Denoiser(nn.Module):
    def __init__(
        self, width: int = 32, context: int = 4, sigma_data: float = 0.5
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.net = UNet(width, context)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
        sigma_aug: torch.Tensor,
    ) -> torch.Tensor:
        # EDM preconditioning keeps the network input and target at unit variance for every sigma
        s = sigma[:, None, None, None]
        total = s * s + self.sigma_data**2
        c_skip = self.sigma_data**2 / total
        c_out = s * self.sigma_data / torch.sqrt(total)
        c_in = 1 / torch.sqrt(total)
        c_noise = torch.log(sigma) / 4

        return c_skip * x + c_out * self.net(
            c_in * x, c_noise, context, actions, sigma_aug
        )


def sample(
    denoiser: Denoiser,
    context: torch.Tensor,
    actions: torch.Tensor,
    noise: torch.Tensor,
    steps: int = 3,
    sigma_aug: float = 0.0,
    sigma_max: float = 5.0,
    sigma_min: float = 0.002,
    rho: float = 7.0,
) -> torch.Tensor:
    # The ramp divides by K - 1, so K = 1 is a single call at sigma_max
    ramp = [i / (steps - 1) for i in range(steps)] if steps > 1 else [0.0]
    max_root, min_root = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    sigmas = [(max_root + t * (min_root - max_root)) ** rho for t in ramp] + [0.0]

    # First 3 noise channels seed the sampler, the rest augment the context (graph contract)
    batch = noise.shape[0]
    x = sigma_max * noise[:, :3]  # shape: (batch, 3, 64, 64)
    context = context + sigma_aug * noise[:, 3:]
    sigma_aug_batch = noise.new_full((batch,), sigma_aug)

    for sigma, sigma_next in itertools.pairwise(sigmas):
        denoised = denoiser(
            x, noise.new_full((batch,), sigma), context, actions, sigma_aug_batch
        )

        # The final Euler step to sigma = 0 lands exactly on the denoiser output
        x = (
            denoised
            if sigma_next == 0
            else x + (sigma_next - sigma) * (x - denoised) / sigma
        )

    return x


def snap(x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    # x: (batch, 3, height, width); palette: (colors, 3)
    distances = ((x[:, None] - palette[None, :, :, None, None]) ** 2).sum(
        dim=2
    )  # shape: (batch, colors, height, width)
    index = distances.argmin(dim=1)  # shape: (batch, height, width)

    # An embedding lookup exports as Gather; palette[index] would export as GatherND
    return F.embedding(index, palette).permute(
        0, 3, 1, 2
    )  # shape: (batch, 3, height, width)
