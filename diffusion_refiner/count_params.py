from diffusion_refiner.models import (
    MaskAutoencoder,
    ImageEncoder,
    DiffusionUNet,
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64)
img_enc = ImageEncoder(in_ch=3, feat_dim=128)
unet = DiffusionUNet(dim=64, cond_dim=128)

ae_params = count_params(ae)
img_enc_params = count_params(img_enc)
unet_params = count_params(unet)
full_params = ae_params + img_enc_params + unet_params

print(f"Mask autoencoder: {ae_params / 1e6:.3f} M")
print(f"Image encoder: {img_enc_params / 1e6:.3f} M")
print(f"Diffusion U-Net: {unet_params / 1e6:.3f} M")
print(f"Full refinement module: {full_params / 1e6:.3f} M")



from diffusion_refiner.models import MaskAutoencoder, ImageEncoder, DiffusionUNet

def count_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64)
img_enc = ImageEncoder(in_ch=3, feat_dim=128)
unet = DiffusionUNet(dim=64, cond_dim=128)

ae_params = count_params(ae)
img_params = count_params(img_enc)
unet_params = count_params(unet)
full_params = ae_params + img_params + unet_params

print("Mask autoencoder:", ae_params, f"= {ae_params/1e6:.3f} M")
print("Image encoder:", img_params, f"= {img_params/1e6:.3f} M")
print("Diffusion U-Net:", unet_params, f"= {unet_params/1e6:.3f} M")
print("Full refinement module:", full_params, f"= {full_params/1e6:.3f} M")