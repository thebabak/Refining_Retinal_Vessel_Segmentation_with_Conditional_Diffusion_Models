import torch
import torch.nn.functional as F
import math

class DiffusionScheduler:
    """DDPM diffusion scheduler with cosine noise schedule."""
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='cosine'):
        self.timesteps = timesteps
        self.schedule = schedule
        
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == 'cosine':
            # Cosine schedule (from improved diffusion paper)
            s = 0.008
            steps = torch.arange(timesteps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Posterior variance
        self.posterior_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_var_clipped = torch.log(torch.clamp(self.posterior_var, min=1e-20))
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, z0, t, noise):
        """q(z_t | z0) forward process."""
        a_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_t) * z0 + torch.sqrt(1 - a_t) * noise

    def q_posterior_mean_var(self, z0, z_t, t):
        """Posterior mean/variance q(z_{t-1} | z_t, z0)."""
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        mean = coef1 * z0 + coef2 * z_t
        var = self.posterior_var[t].view(-1, 1, 1, 1)
        return mean, var

    def p_mean_variance(self, eps_pred, z_t, t):
        """Predict mean/variance from noise prediction (one reverse step)."""
        a_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        z0_pred = (z_t - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)
        mean, var = self.q_posterior_mean_var(z0_pred, z_t, t)
        return mean, var, z0_pred


def ddim_sample(
    model,
    scheduler,
    z_shape,
    cond,
    num_steps=50,
    guidance_scale=1.0,
    generator=None
):
    """DDIM sampling (fast, deterministic).
    
    Args:
        model: LatentDiffusionModel
        scheduler: DiffusionScheduler
        z_shape: shape of latent tensor (B, C, H, W)
        cond: conditioning features (B, cond_dim)
        num_steps: number of DDIM steps (fewer = faster)
        guidance_scale: guidance scale (1.0 = no guidance)
        generator: random generator for reproducibility
    
    Returns:
        z: final latent
    """
    device = cond.device
    z = torch.randn(z_shape, device=device, generator=generator)
    
    # Time steps to use (evenly spaced)
    timesteps = torch.linspace(
        scheduler.timesteps - 1, 0, steps=num_steps, dtype=torch.long, device=device
    )
    
    for i, t_cur in enumerate(timesteps):
        t = t_cur.view(1).expand(z_shape[0]).long()
        
        # Predict noise
        t_emb = timestep_embedding(t.float(), 64).to(device)
        eps_pred = model.unet(z, t_emb, cond)
        
        # Compute alpha schedule values
        a_t = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
        a_prev = scheduler.alphas_cumprod_prev[t].view(-1, 1, 1, 1) if t[0] > 0 else torch.ones_like(a_t)
        
        # DDIM step
        sigma = 0.0  # deterministic (set > 0 for stochastic)
        z = (
            torch.sqrt(a_prev / a_t) * z
            + (torch.sqrt(1 - a_prev) - torch.sqrt((1 - a_prev) * (1 - a_t) / (1 - a_t))) * eps_pred
        )
    
    return z


def timestep_embedding(t, dim):
    """Sinusoidal timestep embedding."""
    half = dim // 2
    ex = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    emb = t[:, None] * ex[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

