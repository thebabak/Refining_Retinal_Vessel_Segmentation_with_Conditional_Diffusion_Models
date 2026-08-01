"""Quantitative uncertainty evaluation."""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from diffusion_refiner.models_corrected import build_model
from diffusion_refiner.uncertainty_metrics import compute_all_uncertainty_metrics, print_uncertainty_report


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChaseDB1Eval(Dataset):
    def __init__(self, data_dir, img_size=(512, 512)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.samples = []
        for f in sorted(self.data_dir.glob("*.jpg")):
            n = f.stem
            if "_1stHO" in n or "_2ndHO" in n: continue
            m = self.data_dir / f"{n}_1stHO.png"
            if m.exists(): self.samples.append((f, m))
        print(f"Found {len(self.samples)} images")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        img_p, msk_p = self.samples[idx]
        img = Image.open(img_p).convert('RGB').resize(self.img_size, Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        msk = Image.open(msk_p).convert('L').resize(self.img_size, Image.NEAREST)
        msk = torch.from_numpy(np.array(msk)).float() / 255.0
        msk = (msk > 0.5).float().unsqueeze(0)
        return {"image": img, "mask": msk}


def make_coarse(gt, noise=0.25, erode=0.15):
    c = gt.clone()
    c = c + torch.randn_like(c) * noise
    rm = (gt > 0.5) & (torch.rand_like(gt) < erode)
    c[rm] = 0.0
    return torch.clamp(c, 0, 1)


@torch.no_grad()
def evaluate(model, loader, device, K=5, steps=50, ns=0.5, gs=1.0):
    model.eval()
    ap, av, at = [], [], []
    for i, b in enumerate(loader):
        if i % 5 == 0: print(f"  [{i+1}/{len(loader)}]")
        img = b["image"].to(device)
        gt = b["mask"].to(device)
        if gt.ndim == 3: gt = gt.unsqueeze(1)
        coarse = make_coarse(gt)
        mean, var = model.ensemble_sample(coarse, img, K=K, num_steps=steps, noise_strength=ns, guidance_scale=gs)
        ap.append(mean.cpu().numpy())
        av.append(var.cpu().numpy())
        at.append(gt.cpu().numpy())
    return ap, av, at


def main():
    device = get_device()
    print(f"Device: {device}\n")
    
    # Data
    dd = Path("data")
    if not dd.exists(): print(f"ERROR: {dd} not found"); return
    ds = ChaseDB1Eval(dd)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    # Model
    model, counts = build_model(device)
    print(f"Model: {counts['total']:,} params\n")
    
    # Load checkpoint
    cp_paths = [
        Path("seed_outputs_properly_trained/CHASE-DB1/diffusion_refiner.pth"),
        Path("seed_outputs_corrected/CHASE-DB1/diffusion_refiner.pth"),
    ]
    cp = None
    for p in cp_paths:
        if p.exists(): cp = p; break
    
    if cp:
        print(f"Loading: {cp}")
        ck = torch.load(cp, map_location=device)
        if isinstance(ck, dict):
            for k in ["ae", "img_enc", "unet"]:
                if k in ck and ck[k] is not None:
                    getattr(model, k).load_state_dict(ck[k])
        model.eval()
    else:
        print("WARNING: No checkpoint - using untrained model!")
    
    results = {}
    
    # 20 steps
    print("\n" + "="*50)
    print("Diff Ensemble (20 steps)")
    print("="*50)
    p, v, t = evaluate(model, dl, device, K=5, steps=20, ns=0.5, gs=1.0)
    r = compute_all_uncertainty_metrics(p, v, t)
    results["Diff (20 steps)"] = r
    print_uncertainty_report(r, "Diff (20 steps)")
    
    # 50 steps
    print("="*50)
    print("Diff Ensemble (50 steps)")
    print("="*50)
    p, v, t = evaluate(model, dl, device, K=5, steps=50, ns=0.5, gs=1.0)
    r = compute_all_uncertainty_metrics(p, v, t)
    results["Diff (50 steps)"] = r
    print_uncertainty_report(r, "Diff (50 steps)")
    
    # Deterministic
    print("="*50)
    print("Deterministic")
    print("="*50)
    zv = [np.zeros_like(x) for x in p]
    r = compute_all_uncertainty_metrics(p, zv, t)
    r['correlation'] = float('nan'); r['aurc'] = float('nan')
    results["Deterministic"] = r
    print_uncertainty_report(r, "Deterministic")
    
    # Comparison
    print("\n" + "="*65)
    print("COMPARISON")
    print("="*65)
    print(f"{'Method':<25} {'Brier':>8} {'ECE':>8} {'Corr':>8} {'AURC':>8}")
    print("-"*65)
    for name, r in results.items():
        co = f"{r['correlation']:.4f}" if not np.isnan(r['correlation']) else "N/A"
        au = f"{r['aurc']:.4f}" if not np.isnan(r['aurc']) else "N/A"
        print(f"{name:<25} {r['brier']:>8.4f} {r['ece']:>8.4f} {co:>8} {au:>8}")
    
    # LaTeX
    print("\n\n% LATEX TABLE")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Quantitative uncertainty evaluation on CHASE-DB1.}")
    print(r"\label{tab:uncertainty_metrics}")
    print(r"\begin{tabular}{l c c c c}")
    print(r"\toprule")
    print(r"Method & Brier $\downarrow$ & ECE $\downarrow$ & Corr. $\uparrow$ & AURC $\downarrow$ \\")
    print(r"\midrule")
    for name in ["Deterministic", "Diff (20 steps)", "Diff (50 steps)"]:
        r = results[name]
        co = f"{r['correlation']:.4f}" if not np.isnan(r['correlation']) else "--"
        au = f"{r['aurc']:.4f}" if not np.isnan(r['aurc']) else "--"
        print(f"{name} & {r['brier']:.4f} & {r['ece']:.4f} & {co} & {au} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()