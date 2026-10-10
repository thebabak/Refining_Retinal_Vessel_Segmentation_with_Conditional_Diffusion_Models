"""Quantify sample variance/error association on all saved CHASE test maps."""
import json,csv,hashlib,itertools
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import pearsonr,spearmanr
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'paired_reproduction/paired_rerun_CHASE_2026-10-06_17-08-15'
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
OUT=ROOT/'paired_reproduction'/f'variance_error_analysis_{stamp}';OUT.mkdir()
manuscript_hash=hashlib.sha256((ROOT/'template261006.tex').read_bytes()).hexdigest()
protocol=json.loads((SOURCE/'protocol.json').read_text());hashes=json.loads((SOURCE/'input_hashes.json').read_text())
names=protocol['split']['test'];seeds=protocol['sampling_seeds'];assert len(names)==8 and len(seeds)==20
records=list(csv.DictReader((SOURCE/'paired_metrics.csv').open()))
lookup={(r['image'],int(r['seed']),int(r['steps'])):r for r in records}
rows=[];all_var=[];all_abs=[];all_binary=[];prediction_hashes={}
def correlations(x,y):
 x=x.ravel().astype(np.float64,copy=False);y=y.ravel().astype(np.float64,copy=False)
 if np.ptp(x)==0 or np.ptp(y)==0:return dict(pearson=None,spearman=None)
 return dict(pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic))
for name in names:
 gtpath=ROOT/'data'/f'{name}_1stHO.png'
 assert hashlib.sha256(gtpath.read_bytes()).hexdigest()==hashes[str(gtpath)]
 gt=np.asarray(Image.open(gtpath).convert('L').resize((512,512),Image.Resampling.NEAREST))>127
 arrays=[]
 for seed in seeds:
  path=SOURCE/f'{name}_steps50_seed{seed}.npz';p=np.load(path)['probability'].squeeze().astype(np.float64)
  assert p.shape==(512,512) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
  mask=p>=.5;tp=int((mask&gt).sum());fp=int((mask&~gt).sum());fn=int((~mask&gt).sum());tn=int((~mask&~gt).sum())
  r=lookup[(name,seed,50)]
  assert all(int(r[f'refined_{k}'])==v for k,v in [('tp',tp),('fp',fp),('fn',fn),('tn',tn)])
  arrays.append(p);prediction_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 stack=np.stack(arrays);mean=stack.mean(0);variance=stack.var(0,ddof=1)
 abs_error=np.abs(mean-gt.astype(float));binary_error=(mean>=.5)!=gt
 corr=correlations(variance,abs_error);binary_corr=correlations(variance,binary_error)
 tp=((mean>=.5)&gt).sum();fp=((mean>=.5)&~gt).sum();fn=((mean<.5)&gt).sum()
 row=dict(image=name,subject=int(name.split('_')[1][:2]),samples=20,mean_pixel_variance=float(variance.mean()),mean_absolute_probability_error=float(abs_error.mean()),ensemble_dice=float(2*tp/max(2*tp+fp+fn,1)),binary_error_fraction=float(binary_error.mean()),variance_absolute_error=corr,variance_binary_error=binary_corr,variance_absolute_error_GT_vessel=correlations(variance[gt],abs_error[gt]),variance_absolute_error_GT_background=correlations(variance[~gt],abs_error[~gt]))
 rows.append(row);all_var.append(variance.ravel());all_abs.append(abs_error.ravel());all_binary.append(binary_error.ravel())
 np.savez_compressed(OUT/f'{name}_variance_error_maps.npz',mean_probability=mean.astype(np.float32),sample_variance=variance.astype(np.float32),absolute_probability_error=abs_error.astype(np.float32),binary_error=binary_error,reference=gt)
 print(name,json.dumps(row['variance_absolute_error']),flush=True)
subjects=sorted({r['subject'] for r in rows});assert len(subjects)==4
within={}
for metric in ('pearson','spearman'):
 vals=np.array([np.mean([r['variance_absolute_error'][metric] for r in rows if r['subject']==s]) for s in subjects])
 # Exhaust all 4^4 equally likely subject-bootstrap draws rather than Monte Carlo.
 boots=vals[np.array(list(itertools.product(range(4),repeat=4)))].mean(1)
 within[metric]=dict(mean_within_image_correlation=float(vals.mean()),subject_bootstrap95ci=np.quantile(boots,[.025,.975]).tolist(),bootstrap_support_size=256)
summary=dict(primary='Within-image spatial association of sample variance and absolute error of the sample-mean probability map; mean correlation over images with exploratory subject-cluster bootstrap',within_image=within,pooled_pixel_absolute_error=correlations(np.concatenate(all_var),np.concatenate(all_abs)),pooled_pixel_binary_error=correlations(np.concatenate(all_var),np.concatenate(all_binary)),between_image_variance_vs_one_minus_ensemble_dice=correlations(np.array([r['mean_pixel_variance'] for r in rows]),np.array([1-r['ensemble_dice'] for r in rows])),per_image=rows,n_images=8,n_subjects=4,sampling_seeds=seeds,transitions=50,region='Whole resized 512x512 image, no FOV mask',variance='Unbiased sample variance across 20 probability maps, denominator19',error='Absolute sample-mean probability error |mean_p - GT|; binary error at mean_p threshold0.5 is secondary',limitations=['One fixed new reproduction training run; not original paper outputs or supplied corrected 0.835 tabulation','Only four subject clusters; bootstrap intervals exploratory','Spatial pixels are dependent; no pixelwise p-values or IID-pixel confidence intervals','Association does not establish calibration or clinical utility','Mean-map Dice is distinct from the average Dice of individual samples','Primary whole-image correlations can be influenced by class imbalance; within-GT-class values also reported'],checkpoint_sha256=json.loads((SOURCE/'summary.json').read_text())['checkpoint_sha256'],prediction_sha256=prediction_hashes,manuscript_unchanged=manuscript_hash==hashlib.sha256((ROOT/'template261006.tex').read_bytes()).hexdigest())
assert summary['manuscript_unchanged']
(OUT/'results_and_protocol.json').write_text(json.dumps(summary,indent=2))
lines=['VARIANCE-ERROR ANALYSIS: NEW CHASE REPRODUCTION',f'Analysis date {stamp}, Asia/Tehran','20 saved probability maps per image, all8test images/four subjects,50DDIM transitions; no retraining or new inference.','Sample variance = sum((p_k-mean_p)^2)/19; error=abs(mean_p-GT).','Whole resized512x512 image; threshold0.5 for secondary binary-error analysis.','These maps correspond to the executable reproduction with mean individual-sample Dice0.745476, not the corrected supplied tabulation near0.835.','', 'image | mean variance | MAE | ensemble Dice | Pearson(var, abs error) | Spearman(var, abs error)']
for r in rows:lines.append(f"{r['image']} | {r['mean_pixel_variance']:.9g} | {r['mean_absolute_probability_error']:.9f} | {r['ensemble_dice']:.9f} | {r['variance_absolute_error']['pearson']:.6f} | {r['variance_absolute_error']['spearman']:.6f}")
lines+=['',json.dumps({k:v for k,v in summary.items() if k not in ('per_image','prediction_sha256')},indent=2)]
(OUT/'variance_error_report.txt').write_text('\n'.join(lines))
rho=within['spearman'];ci=rho['subject_bootstrap95ci']
reply=f'''Thank you for this suggestion. We quantified the relationship between repeated-sampling variance and segmentation error in a separate analysis of the saved new CHASE-DB1 reproduction outputs. We used 20 initial-noise realisations per image at 50 DDIM transitions for all eight test images, keeping the model fixed. Pixel-wise variance was calculated across probability maps with denominator19, and error was defined as the absolute difference between the sample-mean probability map and the reference annotation.

The mean within-image Spearman correlation between variance and absolute error was {rho['mean_within_image_correlation']:.6f}, with an exploratory subject-cluster bootstrap95% interval [{ci[0]:.6f}, {ci[1]:.6f}]. Both eyes were retained together during resampling. Per-image Pearson and Spearman correlations, secondary correlations with binary segmentation error, and within-reference-class correlations are available in the separate report. These measurements characterise association, not calibrated uncertainty. Only four subjects and one training run were available, so the intervals are exploratory. This analysis uses executable reproduction prediction maps and does not validate the original single-image variance figure or the supplied corrected Dice tabulation. The manuscript has not yet been changed for this item.
'''
(OUT/'reviewer_variance_error_response.txt').write_text(reply)
print(json.dumps(dict(report=str(OUT/'variance_error_report.txt'),response=str(OUT/'reviewer_variance_error_response.txt'),within_image=within,pooled=summary['pooled_pixel_absolute_error'],manuscript_unchanged=True),indent=2))
