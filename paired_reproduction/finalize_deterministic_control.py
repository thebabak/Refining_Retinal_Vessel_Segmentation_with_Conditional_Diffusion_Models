import json
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parent.parent
out = Path((root/'paired_reproduction/current_deterministic_control_path.txt').read_text().strip())
tag = out.name[-19:]
now = datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
assert (out/f'verified_completion_{tag}.json').exists()
summaries = list(__import__('csv').DictReader((out/f'mean_and_SD_{tag}.txt').open()))
audit = '''Related-method reproduction audit
HiDiffSeg: https://www.sciencedirect.com/science/article/pii/S0957417424011151
The publisher full-text endpoint returned HTTP403 in this session. The accessible abstract describes skeleton-guided generation and fundus-conditioned refinement. A verified author implementation and pretrained checkpoints were not identified in the checked public sources.
RefineCatDiff: https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202401125
The accessible publisher article describes categorical diffusion and spatial conditioning on BTCV, BraTS2020 and ISIC2018. No repository link was identified in its full text; its data-availability statement refers to requests to the corresponding author. Adaptation to binary retinal masks would require implementation and training choices that have not been validated against an author implementation.
Li et al.: https://pubmed.ncbi.nlm.nih.gov/38194885/
The method denoises intermediate segmentation-network features and is trained with the backbone, unlike a frozen coarse-mask postprocessor. The publisher full-text endpoint https://www.sciencedirect.com/science/article/pii/S0893608024000029 returned HTTP403. A verified author implementation/checkpoints were not identified in the checked public sources.
Public GitHub repository-name searches are archived separately. Negative searches do not prove that implementations do not exist. No authors were contacted. No numerical same-protocol benchmark of these three methods was performed. These are current reproduction limitations, not a claim that faithful reproduction is impossible.
The completed deterministic-control experiment supplies a direct supervised refiner under the available local protocol; it does not replace the missing three-method benchmark or validate the author-supplied positive-score tables.
'''
auditpath=out/f'related_method_audit_{now}.txt'
auditpath.write_text(audit,encoding='utf-8')
lines=['Thank you for this comment. We conducted a matched deterministic learned-refiner control using the available local reproduction. For each dataset, five refiners were independently trained with seeds 42, 123, 2026, 3000 and 3001. Each control used exactly the same frozen LU-Net+RA and mask-autoencoder checkpoints, RGB conditioning architecture, learned parameter budget (405,824), test images and evaluation protocol as its paired diffusion model. The deterministic control predicts a residual feature correction in one forward pass, without Gaussian corruption or DDIM, and is trained with decoded-mask Dice supervision. This compares the direct supervised refinement formulation with diffusion training and inference; it is not a single-loss-factor ablation.', '', 'Mean Dice +/- SD across five training seeds:']
for r in summaries:
    lines.append(f"{r['dataset']}, {r['method']}: {float(r['mean_dice']):.6f} +/- {float(r['sd_between_training_seeds']):.6f}")
lines += ['', 'We provide all 200 paired per-image control records, model checkpoints, FP32 predictions, and paired training-seed tests with crossed bootstrap intervals. Diffusion values use 50 DDIM transitions and average five sampling realisations for each trained model. These are new local reproduction results, kept separate from the author-supplied original result tables. They should not be combined with the original scores as if obtained from the same checkpoints.', '', 'A same-protocol benchmark of HiDiffSeg, RefineCatDiff and Li et al. has not been completed. We could not identify verified author implementations/checkpoints in the public sources checked in this session, and full-text access was limited for HiDiffSeg and Li et al. RefineCatDiff uses a categorical and spatially conditioned formulation evaluated on different datasets; Li et al. modifies intermediate backbone features. Unvalidated adaptations would introduce additional uncontrolled choices. We document these specific limitations and do not claim that implementations are unavailable everywhere or that reproduction is impossible.', '', 'This response is a draft supported by the new experiment; the manuscript has not yet been amended with these control results.']
lines.insert(-2, 'In this local reproduction, the deterministic refiner had higher mean Dice than diffusion on all three datasets. Relative to the coarse baseline, its mean Dice increased on DRIVE but decreased on CHASE-DB1 and HRF. Thus this control does not support a general advantage of the diffusion formulation. Statistical tests and intervals are reported separately; higher point estimates alone are not a claim of uniformly significant improvement. The executable refinement parameter count also differs from the manuscript architecture budget, so these results cannot be presented as a controlled reproduction of the original manuscript architecture without further reconciliation.')
response=out/f'reviewer_2_comment_3_response_{now}.txt'
response.write_text('\n'.join(lines)+'\n',encoding='utf-8')
tracking=root/'reviewer_comments/reviewer_2_comments.json'
obj=json.loads(tracking.read_text(encoding='utf-8-sig'))
c=next(c for c in obj['comments'] if c['number']==3)
c.update(status='Matched deterministic control completed:15 refiner training jobs,200 verified paired records. Related-method audit documented; numerical benchmark of HiDiffSeg, RefineCatDiff and Li et al. remains incomplete. Manuscript unchanged.', analysis_directory=str(out), response_file=str(response), related_method_audit=str(auditpath), manuscript_incorporated=False, updated_at=now)
tracking.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
(tracking.parent/f'reviewer_2_comments_{now}.json').write_bytes(tracking.read_bytes())
txtpath=tracking.with_suffix('.txt')
with txtpath.open('a',encoding='utf-8') as f:f.write(f'\nUpdate {now}: Comment3 deterministic control completed;15 independent refiner trainings,200 verified paired records. Related diffusion-method numerical benchmark remains incomplete. Manuscript unchanged.\nExperiment: {out}\nResponse: {response}\n')
(tracking.parent/f'reviewer_2_comments_{now}.txt').write_bytes(txtpath.read_bytes())
with (root/'AGENTS.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\n# Completed deterministic learned-refiner control\nOn {now},15 direct-refiner training jobs (five per dataset) completed using exactly frozen baseline/AE checkpoints from the actual local all-dataset reproduction. All200 paired records verified against saved FP32 predictions/GT/FOV. Output: {out}. Learned parameter budget405824 matched to this executable diffusion model, distinct from manuscript architecture counts. Related-method reproduction audit and reviewer2 comment3 draft response preserved separately. This experiment does not replace author-supplied positive tables. Manuscript unchanged.\n')
with zipfile.ZipFile(out/f'deterministic_control_data_{tag}.zip','a',compression=zipfile.ZIP_DEFLATED) as z:
    for p in [auditpath,response]:z.write(p,arcname=p.name)
with zipfile.ZipFile(out/f'deterministic_control_data_{tag}.zip') as z:assert z.testzip() is None
print(json.dumps(dict(response=str(response),audit=str(auditpath),zip_verified=True),indent=2))
