"""
Corrected model implementations with proper diffusion components.
Includes cosine noise schedule, DDIM sampling, and ensemble support.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Cosine noise schedule (Nichol & Dhariwal 2021)
# ============================================================

def cosine_beta_schedule(timesteps=1000, s=0.008):
    """Cosine schedule as proposed in 'Improved DDPM'."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((t / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, max=0.999)
    return betas


def get_diffusion_parameters(timesteps=1000, s=0.008):
    """Compute all diffusion parameters from cosine schedule."""
    betas = cosine_beta_schedule(timesteps, s)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
        'sqrt_recip_alphas_cumprod': torch.sqrt(1.0 / alphas_cumprod),
        'sqrt_recipm1_alphas_cumprod': torch.sqrt(1.0 / alphas_cumprod - 1),
    }


# ============================================================
# Sinusoidal Position Embedding
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================
# Residual Block with conditioning
# ============================================================

class ResidualBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and optional time/conditioning injection."""
    
    def __init__(self, in_ch, out_ch, time_emb_dim=None, cond_dim=None):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # Time embedding projection
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch * 2)
            )
        else:
            self.time_mlp = None
        
        # Conditioning projection
        if cond_dim is not None:
            self.cond_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, out_ch * 2)
            )
        else:
            self.cond_proj = None
        
        # Skip connection
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, time_emb=None, cond=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Inject time embedding
        if self.time_mlp is not None and time_emb is not None:
            time_out = self.time_mlp(time_emb)
            scale, shift = time_out.chunk(2, dim=1)
            h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        
        # Inject conditioning
        if self.cond_proj is not None and cond is not None:
            cond_out = self.cond_proj(cond)
            scale, shift = cond_out.chunk(2, dim=1)
            h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


# ============================================================
# Mask Autoencoder
# ============================================================

class MaskAutoencoder(nn.Module):
    """
    Lightweight autoencoder for vessel masks.
    Encodes 1×512×512 → 64×128×128 (4× spatial compression).
    """
    
    def __init__(self, in_ch=1, base=32, latent_dim=64):
        super().__init__()
        
        # Encoder: 512 → 256 → 128
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(min(8, base), base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, base * 2), base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, latent_dim, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, latent_dim), latent_dim),
            nn.SiLU(),
        )
        
        # Decoder: 128 → 256 → 512
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, base * 2), base * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, base), base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


# ============================================================
# Image Encoder
# ============================================================

class ImageEncoder(nn.Module):
    """
    Encodes RGB fundus image to a compact feature vector for conditioning.
    3×512×512 → 128-dimensional embedding.
    """
    
    def __init__(self, in_ch=3, feat_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feat_dim),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# Diffusion U-Net
# ============================================================

class DiffusionUNet(nn.Module):
    """
    Lightweight U-Net for latent diffusion.
    Operates on 64×128×128 latents with 128-dim conditioning.
    """
    
    def __init__(self, dim=64, cond_dim=128, time_emb_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(dim, dim, 3, padding=1)
        
        # Encoder path
        self.down1 = nn.ModuleList([
            ResidualBlock(dim, dim, time_emb_dim, cond_dim),
            ResidualBlock(dim, dim, time_emb_dim, cond_dim),
        ])
        
        self.downsample1 = nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1)
        self.down2 = nn.ModuleList([
            ResidualBlock(dim * 2, dim * 2, time_emb_dim, cond_dim),
            ResidualBlock(dim * 2, dim * 2, time_emb_dim, cond_dim),
        ])
        
        self.downsample2 = nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(dim * 4, dim * 4, time_emb_dim, cond_dim),
            ResidualBlock(dim * 4, dim * 4, time_emb_dim, cond_dim),
        ])
        
        # Decoder path
        self.upsample1 = nn.ConvTranspose2d(dim * 4, dim * 2, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResidualBlock(dim * 2, dim * 2, time_emb_dim, cond_dim),
            ResidualBlock(dim * 2, dim * 2, time_emb_dim, cond_dim),
        ])
        
        self.upsample2 = nn.ConvTranspose2d(dim * 2, dim, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResidualBlock(dim, dim, time_emb_dim, cond_dim),
            ResidualBlock(dim, dim, time_emb_dim, cond_dim),
        ])
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(8, dim), dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
    
    def forward(self, x, t, cond):
        """Args: x [B,64,H,W], t [B], cond [B,128]. Returns predicted noise [B,64,H,W]."""
        time_emb = self.time_mlp(t)
        
        # Encoder
        h = self.init_conv(x)
        h1 = self.down1[0](h, time_emb, cond)
        h1 = self.down1[1](h1, time_emb, cond)
        
        h2 = self.downsample1(h1)
        h2 = self.down2[0](h2, time_emb, cond)
        h2 = self.down2[1](h2, time_emb, cond)
        
        h3 = self.downsample2(h2)
        
        # Bottleneck
        h3 = self.bottleneck[0](h3, time_emb, cond)
        h3 = self.bottleneck[1](h3, time_emb, cond)
        
        # Decoder with skip connections
        h = self.upsample1(h3)
        h = h + h2
        h = self.up1[0](h, time_emb, cond)
        h = self.up1[1](h, time_emb, cond)
        
        h = self.upsample2(h)
        h = h + h1
        h = self.up2[0](h, time_emb, cond)
        h = self.up2[1](h, time_emb, cond)
        
        h = self.final_conv(h)
        
        return h


# ============================================================
# Latent Diffusion Model (CORRECTED)
# ============================================================

class LatentDiffusionModel(nn.Module):
    """
    Full conditional latent diffusion model with PROPER DDIM sampling.
    """
    
    def __init__(self, ae, img_enc, unet, cond_dim=128, timesteps=1000):
        super().__init__()
        
        self.ae = ae
        self.img_enc = img_enc
        self.unet = unet
        self.timesteps = timesteps
        self.cond_dim = cond_dim
        
        # Register diffusion parameters as buffers
        params = get_diffusion_parameters(timesteps)
        for name, tensor in params.items():
            self.register_buffer(name, tensor)
    
    def forward(self, mask, image, t):
        """
        Training forward pass.
        Returns: eps_pred, eps_true, z0
        """
        with torch.no_grad():
            z0 = self.ae.encode(mask)
        
        img_cond = self.img_enc(image)
        eps_true = torch.randn_like(z0)
        
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        z_t = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * eps_true
        
        eps_pred = self.unet(z_t, t, img_cond)
        
        return eps_pred, eps_true, z0
    
    @torch.no_grad()
    def ddim_sample(self, mask, image, num_steps=50, noise_strength=0.1,
                    guidance_scale=1.5, max_timestep=250):
        """
        CORRECTED DDIM sampling that actually calls the diffusion U-Net.
        
        Only uses low timesteps (t < max_timestep) where the model is reliable
        for refinement tasks.
        """
        self.eval()
        
        batch_size = mask.shape[0]
        device = mask.device
        
        # Encode coarse mask
        z0 = self.ae.encode(mask)
        
        # Get image conditioning
        img_cond = self.img_enc(image)
        
        # Map noise_strength to starting timestep (only low range)
        start_t = int(noise_strength * max_timestep)
        start_t = min(start_t, max_timestep - 1)
        start_t = max(start_t, 1)
        
        # Add noise at starting timestep
        alpha_start = self.alphas_cumprod[start_t].view(1, 1, 1, 1)
        noise = torch.randn_like(z0)
        z_t = torch.sqrt(alpha_start) * z0 + torch.sqrt(1 - alpha_start) * noise
        
        # Create DDIM timesteps (descending from start_t to 0)
        step_indices = torch.linspace(start_t, 0, num_steps + 1, device=device).long()
        
        for i in range(num_steps):
            t = step_indices[i].repeat(batch_size)
            t_next = step_indices[i + 1].repeat(batch_size)
            
            # CALL THE DIFFUSION U-NET (this was the missing part!)
            eps_pred = self.unet(z_t, t, img_cond)
            
            # Classifier-free guidance
            if guidance_scale != 1.0:
                null_cond = torch.zeros_like(img_cond)
                eps_uncond = self.unet(z_t, t, null_cond)
                eps_pred = eps_uncond + guidance_scale * (eps_pred - eps_uncond)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_next = self.alphas_cumprod[t_next].view(-1, 1, 1, 1)
            
            # Predict x0
            x0_pred = (z_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
            
            # DDIM step
            dir_xt = torch.sqrt(1 - alpha_next) * eps_pred
            z_t = torch.sqrt(alpha_next) * x0_pred + dir_xt
        
        # Decode refined latent
        refined_out = self.ae.decode(z_t)
        
        if refined_out.shape[-2:] != mask.shape[-2:]:
            refined_out = F.interpolate(
                refined_out, size=mask.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        return torch.sigmoid(refined_out)
    
    @torch.no_grad()
    def ensemble_sample(self, mask, image, K=5, num_steps=50,
                       noise_strength=0.1, guidance_scale=1.5):
        """
        Ensemble sampling for uncertainty estimation.
        Returns: mean_mask [B,1,H,W], uncertainty [B,1,H,W]
        """
        samples = []
        for _ in range(K):
            refined = self.ddim_sample(
                mask, image, num_steps, noise_strength, guidance_scale
            )
            samples.append(refined)
        
        samples = torch.stack(samples, dim=0)  # [K, B, 1, H, W]
        mean_mask = samples.mean(dim=0)
        uncertainty = samples.var(dim=0)
        
        return mean_mask, uncertainty