import torch
import torch.nn as nn
import math
from tqdm import tqdm


class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes timestep information into a vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    """A residual block with two convolutional layers and time/condition embedding injection."""

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.mlp(time_emb)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """Self-attention block."""

    def __init__(self, in_channels, groups=8):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, in_channels)
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.group_norm(x)
        q = self.query(h_)
        k = self.key(h_)
        v = self.value(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)

        sim = torch.bmm(q, k) * (c**-0.5)
        attn = sim.softmax(dim=-1)

        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj_out(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn):
        super().__init__()
        self.res = ResnetBlock(in_channels, out_channels, time_emb_dim)
        self.attn = Attention(out_channels) if has_attn else nn.Identity()

    def forward(self, x, time_emb):
        x = self.res(x, time_emb)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn):
        super().__init__()
        # The input channel dimension for the ResnetBlock is doubled to account for the skip connection
        self.res = ResnetBlock(in_channels * 2, out_channels, time_emb_dim)
        self.attn = Attention(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip, time_emb):
        x = torch.cat((x, skip), dim=1)
        x = self.res(x, time_emb)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=256,
        cond_embed_dim=512,
        model_channels=64,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_embed_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder path
        self.down1 = DownBlock(model_channels, model_channels, time_emb_dim, False)
        self.down2 = DownBlock(model_channels, model_channels * 2, time_emb_dim, False)
        self.down3 = DownBlock(
            model_channels * 2, model_channels * 4, time_emb_dim, True
        )
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = ResnetBlock(model_channels * 4, model_channels * 8, time_emb_dim)
        self.bot2 = Attention(model_channels * 8)
        self.bot3 = ResnetBlock(model_channels * 8, model_channels * 4, time_emb_dim)

        # Decoder path
        self.up1 = UpBlock(model_channels * 4, model_channels * 2, time_emb_dim, True)
        self.up2 = UpBlock(model_channels * 2, model_channels, time_emb_dim, False)
        self.up3 = UpBlock(model_channels, model_channels, time_emb_dim, False)
        self.unpool = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.output = nn.Conv2d(model_channels, out_channels, 3, padding=1)

    def forward(self, x, time, cond):
        t = self.time_mlp(time)
        c = self.cond_mlp(cond)
        emb = t + c

        x_init = self.init_conv(x)

        # --- Encoder ---
        d1 = self.down1(x_init, emb)
        d2 = self.down2(self.pool(d1), emb)
        d3 = self.down3(self.pool(d2), emb)

        # --- Bottleneck ---
        b = self.bot1(self.pool(d3), emb)
        b = self.bot2(b)
        b = self.bot3(b, emb)

        # --- Decoder ---
        u1 = self.up1(self.unpool(b), d3, emb)
        u2 = self.up2(self.unpool(u1), d2, emb)
        u3 = self.up3(self.unpool(u2), d1, emb)

        return self.output(u3)


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        condition_dim,
        embed_dim,
        image_size,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.image_size = image_size
        self.attribute_embedder = nn.Linear(condition_dim, embed_dim)
        self.unet = UNet(
            cond_embed_dim=embed_dim, model_channels=128
        )  # Increased model channels for larger images

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def forward_process(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t, None, None, None]
        return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise, noise

    @torch.no_grad()
    def sample(self, num_samples, conditions, device):
        self.unet.eval()
        x = torch.randn(
            (num_samples, 3, self.image_size, self.image_size), device=device
        )
        cond_emb = self.attribute_embedder(conditions)

        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            predicted_noise = self.unet(x, t, cond_emb)

            alpha_t = 1.0 - self.betas[i]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[i]

            x = (
                x - (self.betas[i] / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
            ) / math.sqrt(alpha_t)
            if i > 0:
                z = torch.randn_like(x)
                x += torch.sqrt(self.betas[i]) * z

        self.unet.train()
        return (x.clamp(-1, 1) + 1) / 2

    def forward(self, images, conditions):
        cond_emb = self.attribute_embedder(conditions)
        t = torch.randint(
            0, self.timesteps, (images.shape[0],), device=images.device
        ).long()

        noisy_images, noise = self.forward_process(images, t)
        predicted_noise = self.unet(noisy_images, t, cond_emb)

        return nn.MSELoss()(noise, predicted_noise)
