"""
Corrected models. Total: ~427K params.
AE: ~155K | IE: ~19K | UN: ~253K
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps=1000, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((t/timesteps + s)/(1+s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, max=0.999)


def get_diffusion_parameters(timesteps=1000, s=0.008):
    betas = cosine_beta_schedule(timesteps, s)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    return {'betas': betas, 'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'alphas_cumprod_prev': alphas_cumprod_prev,
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod)}


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, cond_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch) if in_ch >= 8 else nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch) if out_ch >= 8 else nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2)) if time_emb_dim else None
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_ch * 2)) if cond_dim else None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, time_emb=None, cond=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        if self.time_mlp is not None and time_emb is not None:
            s, sh = self.time_mlp(time_emb).chunk(2, dim=1)
            h = h * (1 + s.unsqueeze(-1).unsqueeze(-1)) + sh.unsqueeze(-1).unsqueeze(-1)
        if self.cond_proj is not None and cond is not None:
            s, sh = self.cond_proj(cond).chunk(2, dim=1)
            h = h * (1 + s.unsqueeze(-1).unsqueeze(-1)) + sh.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class MaskAutoencoder(nn.Module):
    def __init__(self, in_ch=1, base=32, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.GroupNorm(8, base*2), nn.SiLU(),
            nn.Conv2d(base*2, latent_dim, 3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base*2, 4, stride=2, padding=1), nn.GroupNorm(8, base*2), nn.SiLU(),
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): return self.decode(self.encode(x))


class ImageEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, feat_dim),
        )
    def forward(self, x): return self.net(x)


class DiffusionUNet(nn.Module):
    def __init__(self, in_dim=64, dim=16, cond_dim=128, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
        )
        self.proj_in = nn.Conv2d(in_dim, dim, 3, padding=1)        # 64→16
        self.enc1 = ResidualBlock(dim, dim, time_emb_dim, cond_dim)
        self.ds1  = nn.Conv2d(dim, dim*2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(dim*2, dim*2, time_emb_dim, cond_dim)
        self.ds2  = nn.Conv2d(dim*2, dim*4, 3, stride=2, padding=1)
        self.bottleneck = ResidualBlock(dim*4, dim*4, time_emb_dim, cond_dim)
        self.us1  = nn.ConvTranspose2d(dim*4, dim*2, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(dim*2, dim*2, time_emb_dim, cond_dim)
        self.us2  = nn.ConvTranspose2d(dim*2, dim, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(dim, dim, time_emb_dim, cond_dim)
        self.proj_out = nn.Conv2d(dim, in_dim, 3, padding=1)       # 16→64
    
    def forward(self, x, t, cond):
        te = self.time_mlp(t)
        x = self.proj_in(x)
        e1 = self.enc1(x, te, cond)
        e2 = self.enc2(self.ds1(e1), te, cond)
        h  = self.bottleneck(self.ds2(e2), te, cond)
        h  = self.dec1(self.us1(h) + e2, te, cond)
        h  = self.dec2(self.us2(h) + e1, te, cond)
        return self.proj_out(h)


class LatentDiffusionModel(nn.Module):
    def __init__(self, ae, img_enc, unet, cond_dim=128, timesteps=1000):
        super().__init__()
        self.ae, self.img_enc, self.unet = ae, img_enc, unet
        self.timesteps, self.cond_dim = timesteps, cond_dim
        for k, v in get_diffusion_parameters(timesteps).items():
            self.register_buffer(k, v)
        self._diag = False
    
    def forward(self, mask, image, t):
        with torch.no_grad(): z0 = self.ae.encode(mask)
        c = self.img_enc(image)
        e = torch.randn_like(z0)
        ab = self.alphas_cumprod[t].view(-1,1,1,1)
        zt = torch.sqrt(ab)*z0 + torch.sqrt(1-ab)*e
        return self.unet(zt, t, c), e, z0
    
    @torch.no_grad()
    def ddim_sample(self, mask, image, num_steps=50, noise_strength=0.5, guidance_scale=1.0):
        self.eval()
        bs, dev = mask.shape[0], mask.device
        z0 = self.ae.encode(mask)
        c = self.img_enc(image)
        st = max(min(int(noise_strength*self.timesteps), self.timesteps-1), 100)
        if not self._diag:
            self._diag = True
            print(f"[DDIM] start_t={st}, noise={1-self.alphas_cumprod[st]:.3f}")
        a_st = self.alphas_cumprod[st].view(1,1,1,1)
        zt = torch.sqrt(a_st)*z0 + torch.sqrt(1-a_st)*torch.randn_like(z0)
        si = torch.linspace(st, 0, num_steps+1, device=dev).long()
        for i in range(num_steps):
            tc, tn = si[i].repeat(bs), si[i+1].repeat(bs)
            ep = self.unet(zt, tc, c)
            if guidance_scale != 1.0:
                ep = self.unet(zt, tc, torch.zeros_like(c)) + guidance_scale*(ep - self.unet(zt, tc, torch.zeros_like(c)))
            at = self.alphas_cumprod[tc].view(-1,1,1,1)
            an = self.alphas_cumprod[tn].view(-1,1,1,1)
            x0 = torch.clamp((zt - torch.sqrt(1-at)*ep)/torch.sqrt(at), -3, 3)
            zt = torch.sqrt(an)*x0 + torch.sqrt(1-an)*ep
        out = self.ae.decode(zt)
        if out.shape[-2:] != mask.shape[-2:]:
            out = F.interpolate(out, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(out)
    
    @torch.no_grad()
    def ensemble_sample(self, mask, image, K=5, **kw):
        s = torch.stack([self.ddim_sample(mask, image, **kw) for _ in range(K)], dim=0)
        return s.mean(dim=0), s.var(dim=0)


def build_model(device='cuda'):
    ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64).to(device)
    ie = ImageEncoder(in_ch=3, feat_dim=128).to(device)
    un = DiffusionUNet(in_dim=64, dim=16, cond_dim=128, time_emb_dim=64).to(device)
    model = LatentDiffusionModel(ae=ae, img_enc=ie, unet=un, cond_dim=128, timesteps=1000).to(device)
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in [('ae',ae),('ie',ie),('un',un)]}
    counts['total'] = sum(counts.values())
    return model, counts


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, c = build_model(device)
    print(f"AE: {c['ae']:,} | IE: {c['ie']:,} | UN: {c['un']:,} | Total: {c['total']:,}")
    x = torch.randn(1,1,512,512).to(device)
    img = torch.randn(1,3,512,512).to(device)
    t = torch.randint(0,1000,(1,)).to(device)
    ep, et, z0 = model(x, img, t)
    print(f"Forward: OK ({ep.shape})")
    out = model.ddim_sample(x, img, num_steps=5)
    print(f"DDIM: OK ({out.shape})")
    m, v = model.ensemble_sample(x, img, K=3, num_steps=5)
    print(f"Ensemble: OK (var={v.mean():.6f})")