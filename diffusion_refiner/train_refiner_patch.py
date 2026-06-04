import torch


def ensure_mask_shape(mask):
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask.float()


def train_step(model, optimizer, batch, device, lambda_dice=0.5, lambda_edge=0.0, edge_loss_fn=None):
    """
    Safer training step for a coarse-mask refiner.

    Requires batch['coarse'] to be a real LU-Net+RA prediction.
    batch['mask'] remains the ground-truth target.

    Note: for the strongest formulation, the model should explicitly condition on
    coarse latent features. This patch at least prevents GT/second-observer leakage
    and computes the segmentation loss against GT.
    """
    model.train()

    image = batch['image'].to(device).float()
    gt_mask = ensure_mask_shape(batch['mask'].to(device))

    if 'coarse' not in batch:
        raise KeyError("batch['coarse'] is required. Do not fall back to ground truth.")

    coarse_mask = ensure_mask_shape(batch['coarse'].to(device))
    batch_size = image.shape[0]

    t = torch.randint(0, model.scheduler.timesteps, (batch_size,), device=device)

    # Minimum-change option: train the denoiser around the coarse latent, but supervise
    # the decoded prediction against the GT mask.
    eps_pred, eps_true, z_coarse = model(coarse_mask, image, t)
    loss_ddpm = torch.nn.functional.mse_loss(eps_pred, eps_true)

    a_t = model.scheduler.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
    z_t = model.scheduler.q_sample(z_coarse, t, eps_true)
    z0_pred = (z_t - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)
    pred_mask = model.ae.decode(z0_pred) if hasattr(model.ae, 'decode') else model.ae.decoder(z0_pred)

    # dice_loss in your code returns [B], so take mean.
    from .models import dice_loss
    loss_dice = dice_loss(pred_mask, gt_mask).mean()

    loss = loss_ddpm + lambda_dice * loss_dice

    if lambda_edge > 0:
        if edge_loss_fn is None:
            from .models import edge_loss as edge_loss_fn
        loss = loss + lambda_edge * edge_loss_fn(pred_mask, gt_mask)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        'loss': float(loss.detach().cpu()),
        'loss_ddpm': float(loss_ddpm.detach().cpu()),
        'loss_dice': float(loss_dice.detach().cpu()),
    }
