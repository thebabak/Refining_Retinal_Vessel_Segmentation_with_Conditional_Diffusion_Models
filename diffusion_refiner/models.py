import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DiffusionScheduler, timestep_embedding

# ResBlock with optional FiLM conditioning
class ResBlock(nn.Module):
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        if cond_dim:
            self.film = nn.Linear(cond_dim, dim*2)
        else:
            self.film = None
    def forward(self, x, cond=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        if self.film is not None and cond is not None:
            g, b = self.film(cond).chunk(2, dim=1)
            g = g[...,None,None]; b = b[...,None,None]
            h = h * (1+g) + b
        h = self.act(self.conv2(h))
        return x + h

# Mask autoencoder (small)
class MaskAutoencoder(nn.Module):
    def __init__(self, in_ch=1, base=32, latent_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(base, base*2, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(base*2, latent_dim, 1)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base*2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, 1, 1)
        )
    def encode(self, x): return self.enc(x)
    def decode(self, z): return torch.sigmoid(self.dec(z))
    def forward(self, x): return self.decode(self.encode(x))

# Lightweight image encoder
class ImageEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feat_dim)
        )
    def forward(self, x): return self.net(x)

# Simple timestep embedding
def timestep_embedding_local(t, dim):
    half = dim // 2
    ex = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32)/half).to(t.device)
    emb = t[:,None] * ex[None,:]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

# Tiny diffusion U-Net in latent space
class DiffusionUNet(nn.Module):
    def __init__(self, dim=64, cond_dim=128):
        super().__init__()
        self.in_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.res1 = ResBlock(dim, cond_dim)
        self.down = nn.Conv2d(dim, dim, 4, stride=2, padding=1)
        self.res2 = ResBlock(dim, cond_dim)
        self.up = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(dim, dim, 1)
    def forward(self, z, t_emb, cond):
        h = self.in_conv(z)
        h = self.res1(h, cond)
        h = self.down(h)
        h = self.res2(h, cond)
        h = self.up(h)
        h = self.out_conv(h)
        return h

# Latent diffusion wrapper
class LatentDiffusionModel(nn.Module):
    def __init__(self, ae: MaskAutoencoder, img_enc: ImageEncoder, unet: DiffusionUNet, cond_dim=128, timesteps=1000):
        super().__init__()
        self.ae = ae
        self.img_enc = img_enc
        self.unet = unet
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.scheduler = DiffusionScheduler(timesteps=timesteps, schedule='cosine')

    def forward(self, mask, image, t):
        # encode mask to latent
        z0 = self.ae.encode(mask)
        noise = torch.randn_like(z0)
        # use scheduler to add noise
        z_t = self.scheduler.q_sample(z0, t, noise)
        t_emb = timestep_embedding_local(t.float(), 64)
        cond = self.img_enc(image)
        cond = self.cond_mlp(cond)
        eps_pred = self.unet(z_t, t_emb, cond)
        return eps_pred, noise, z0

# Loss helpers
def ddpm_loss(eps_pred, eps_true):
    return F.mse_loss(eps_pred, eps_true)

def dice_loss(pred_mask, true_mask, eps=1e-6):
    p = pred_mask.flatten(1)
    g = true_mask.flatten(1)
    return 1 - (2*(p*g).sum(1) + eps) / (p.sum(1)+g.sum(1)+eps)

def edge_loss(pred, target):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=pred.device).reshape(1,1,3,3)
    ky = kx.transpose(-1,-2)
    def grad(x,k):
        return F.conv2d(x, k, padding=1)
    return F.l1_loss(grad(pred,kx), grad(target,kx)) + F.l1_loss(grad(pred,ky), grad(target,ky))
