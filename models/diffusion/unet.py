"""U‑Net with cross‑attention to text embeddings for DDPM."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from config import VAEConfig

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.text_proj = nn.Linear(cond_dim, out_ch)
        self.attn = nn.MultiheadAttention(out_ch, num_heads=8, batch_first=True)
        self.nonlinear = nn.SiLU()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        h = self.nonlinear(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.nonlinear(self.norm2(h))
        h = self.conv2(h)
        B, C, H, W = h.shape
        h_flat = h.view(B, C, H * W).transpose(1, 2)
        txt_proj = self.text_proj(txt)
        h_attn, _ = self.attn(h_flat, txt_proj, txt_proj)
        h = h + h_attn.transpose(1, 2).view(B, C, H, W)
        return h + self.shortcut(x)

class UNet(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        ch = 64
        time_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )
        self.init_conv = nn.Conv2d(cfg.image_channels, ch, 3, padding=1)
        self.down1 = ResidualBlock(ch, ch * 2, time_dim * 4, cfg.text_decoder_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResidualBlock(ch * 2, ch * 4, time_dim * 4, cfg.text_decoder_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = ResidualBlock(ch * 4, ch * 4, time_dim * 4, cfg.text_decoder_dim)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 4, 2, stride=2)
        self.dec2 = ResidualBlock(ch * 4 + ch * 4, ch * 2, time_dim * 4, cfg.text_decoder_dim)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch * 2, 2, stride=2)
        self.dec1 = ResidualBlock(ch * 2 + ch * 2, ch, time_dim * 4, cfg.text_decoder_dim)
        self.final = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, cfg.image_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h0 = self.init_conv(x)
        h1 = self.down1(h0, t_emb, txt)
        h2 = self.down2(self.pool1(h1), t_emb, txt)
        h_mid = self.mid(self.pool2(h2), t_emb, txt)
        h = self.dec2(torch.cat([self.up2(h_mid), h2], dim=1), t_emb, txt)
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1), t_emb, txt)
        return self.final(h)
