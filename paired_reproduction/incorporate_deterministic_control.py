import csv
import json
import shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parent.parent
out=root/'paired_reproduction/deterministic_control_2026-10-09_10-11-40'
tag='2026-10-09_10-11-40'
assert (out/f'verified_completion_{tag}.json').exists()
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
working=root/'template261006.tex'
raw=working.read_bytes()
text=raw.decode('utf-8-sig')
assert 'sec:deterministic_local_control' not in text
means=list(csv.DictReader((out/f'mean_and_SD_{tag}.txt').open()))
tests=json.loads((out/f'paired_tests_{tag}.json').read_text())
lookup={(r['dataset'],r['method']):r for r in means}
section=r'''
\subsection{Deterministic Learned-Refiner Control in a Separate Local Reproduction}
\label{sec:deterministic_local_control}

We performed a separate controlled experiment using the retained checkpoints of the executable local reproduction. These results are distinct from the author-supplied paired tables in Section~\ref{sec:per_image_uncertainty} and from the original single-run results; they are not substituted for either. Five deterministic refiners were independently trained per dataset with seeds 42, 123, 2026, 3000, and 3001. Each refiner used the exact frozen LU-Net+RA and mask-autoencoder checkpoints of its corresponding local diffusion run, with the same test images, ground truth, threshold, and evaluation region. There were eight CHASE-DB1, twenty DRIVE, and twelve HRF test images, yielding 200 paired image--training-seed records. Saved FP32 predictions were checked against TP, FP, FN, TN, and the reported metrics.

The deterministic control retained the local diffusion model's RGB encoder, latent U-Net, and conditioning MLP, with a matched learned parameter budget of 405,824. It predicted a residual correction to the encoded coarse map in one forward pass with fixed time conditioning at $t=0$, followed by the same frozen decoder. It used neither Gaussian corruption nor DDIM sampling or classifier-free guidance at inference. Training used $0.5(1-\mathrm{Dice})$ with the same decoded-mask Dice reduction, AdamW with learning rate $10^{-4}$ and weight decay $10^{-3}$, cosine scheduling over 100 epochs, batch size four, gradient clipping at one, and 10\% conditioning dropout. No refiner augmentation was used. This compares direct supervised refinement with the diffusion training and inference formulation; it is not a single-factor loss ablation, and fixed-time conditioning has lower effective time-dependent capacity despite the matched total parameter count. The executable parameter count differs from the archived manuscript architecture budget, so this experiment does not establish a controlled reproduction of the original architecture.

\begin{table}[htbp]
\centering
\caption{Separate local control: mean Dice $\pm$ sample SD of five training-seed means. Diffusion uses 50 DDIM transitions and five sampling realisations per trained model; these realisations are averaged before calculating between-training-seed SD. AE denotes frozen autoencoder reconstruction without a learned refiner. Values are rounded to six decimal places.}
\label{tab:deterministic_local_means}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Dataset & Coarse LU-Net+RA & AE reconstruction & Deterministic refiner & Diffusion, 50 steps \\
\midrule
'''
for dataset,label in [('CHASE','CHASE-DB1'),('DRIVE','DRIVE'),('HRF','HRF')]:
    vals=[]
    for method in ['baseline_dice','autoencoder_dice','deterministic_dice','diffusion_mean_dice']:
        r=lookup[(dataset,method)]
        vals.append(f"${float(r['mean_dice']):.6f} \\pm {float(r['sd_between_training_seeds']):.6f}$")
    section+=label+' & '+' & '.join(vals)+r' \\'+'\n'
section+=r'''\bottomrule
\end{tabular}}
\end{table}

\begin{table}[htbp]
\centering
\caption{Paired local comparisons: mean Dice change, exploratory crossed-bootstrap 95\% interval (50,000 resamples), and Holm-adjusted two-sided paired $t$ and exact signed-rank $p$ values. Each test uses five matched training-seed means. Holm adjustment is applied separately to each test family across all nine local comparisons, including the three AE-to-deterministic comparisons archived in the experiment report. ``Det.--coarse'' is deterministic minus baseline; ``Diff.--det.'' is diffusion minus deterministic.}
\label{tab:deterministic_local_tests}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
Dataset & Comparison & Mean change & 95\% interval & $p_t$ (Holm) & $p_W$ (Holm) \\
\midrule
'''
for dataset,label in [('CHASE','CHASE-DB1'),('DRIVE','DRIVE'),('HRF','HRF')]:
    for comparison,display in [('baseline_to_deterministic','Det.--coarse'),('deterministic_to_diffusion','Diff.--det.')]:
        r=next(r for r in tests if r['dataset']==dataset and r['comparison']==comparison)
        lo,hi=r['crossed_bootstrap_95ci']
        section+=f"{label} & {display} & ${r['mean_delta']:+.6f}$ & $[{lo:+.6f}, {hi:+.6f}]$ & {r['holm_paired_t_p']:.6f} & {r['holm_exact_wilcoxon_p']:.6f}"+r' \\'+'\n'
section+=r'''\bottomrule
\end{tabular}}
\end{table}

The crossed bootstrap resampled training seeds and test units, retaining both CHASE-DB1 eyes together within each of four subjects. DRIVE used image-level resampling; HRF used pathology-stratified image resampling. Patient independence for DRIVE and HRF was not established. These exploratory intervals and training-seed tests address different uncertainty assumptions and need not give identical conclusions. Five-pair exact signed-rank tests have limited resolution; the small fixed cohorts constrain inference.

In this local reproduction, deterministic refinement had higher mean Dice than diffusion on all three datasets. Compared with the coarse baseline, deterministic refinement increased mean Dice on DRIVE but decreased it on CHASE-DB1 and HRF. Thus the control does not support a general advantage of the diffusion formulation. The AE-only results also show that reconstruction loss can substantially affect the coarse prediction in the CHASE-DB1 and HRF local configurations. These findings concern the retained local checkpoints and cannot be used as a paired control for the distinct author-supplied positive-score tables.

A same-protocol numerical benchmark of HiDiffSeg, RefineCatDiff, and Li et al. remains incomplete. Verified author implementations and checkpoints were not identified in the public sources checked for this experiment; full-text publisher access was restricted for HiDiffSeg and Li et al., while the accessible RefineCatDiff article provided no repository link. RefineCatDiff uses categorical diffusion and spatial conditioning evaluated on different datasets, and Li et al. denoises intermediate backbone features. Faithful adaptations require further implementation validation. These are current reproduction limitations, not evidence that implementations do not exist or that reproduction is impossible.

'''
anchor='\\section{Discussion}'
assert text.count(anchor)==1
text=text.replace(anchor,section+anchor)
old='A deterministic learned-refiner control and target-diffusion ablation would be required to isolate the benefit of the diffusion formulation.'
new='A separate local deterministic learned-refiner control is reported in Section~\\ref{sec:deterministic_local_control}; a matched control for the original score-producing configuration and a target-diffusion ablation remain necessary to isolate the benefit of the diffusion formulation.'
assert old in text
text=text.replace(old,new)
old='Other component ablations and a deterministic learned-refiner control remain absent.'
assert old in text
text=text.replace(old,'Other component ablations remain absent. The separate local deterministic control in Section~\\ref{sec:deterministic_local_control} does not support a general diffusion advantage and does not validate the original score-producing configuration.')
for env in ['table','tabular','equation']:
    assert text.count('\\begin{'+env+'}')==text.count('\\end{'+env+'}')
version=root/f'template261006_{stamp}.tex'
backup=root/f'template261006_before_deterministic_control_{stamp}.tex'
assert not version.exists() and not backup.exists()
backup.write_bytes(raw)
newline='\r\n' if b'\r\n' in raw else '\n'
newraw=text.replace('\r\n','\n').replace('\n',newline).encode('utf-8')
version.write_bytes(newraw)
working.write_bytes(newraw)
tracking=root/'reviewer_comments/reviewer_2_comments.json'
obj=json.loads(tracking.read_text(encoding='utf-8-sig'))
c=next(c for c in obj['comments'] if c['number']==3)
c.update(manuscript_incorporated=True,manuscript_version=str(version),updated_at=stamp)
tracking.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
(tracking.parent/f'reviewer_2_comments_{stamp}.json').write_bytes(tracking.read_bytes())
txt=tracking.with_suffix('.txt')
with txt.open('a',encoding='utf-8') as f:f.write(f'\n{stamp}: Comment3 local deterministic control added as a separate manuscript section. Original author-supplied tables preserved. Related-method benchmark remains incomplete. Version: {version}\n')
(tracking.parent/f'reviewer_2_comments_{stamp}.txt').write_bytes(txt.read_bytes())
with (root/'AGENTS.md').open('a',encoding='utf-8') as f:f.write(f'\n\n# Local deterministic control added separately\nOn {stamp}, user explicitly authorized a separate manuscript section with local control results. Version: {version}. Latest alias template261006.tex updated; prior copy preserved. Twelve means/SD entries and six paired comparison entries added from verified deterministic_control_2026-10-09_10-11-40. Author-supplied tables unchanged. Parameter-budget discrepancy and absence of related-method numerical benchmarks stated; no general diffusion advantage asserted.\n')
(out/f'manuscript_incorporation_{stamp}.json').write_text(json.dumps(dict(version=str(version),backup=str(backup),section='sec:deterministic_local_control',source=str(out),pdf_compiled=False),indent=2))
print(version)
