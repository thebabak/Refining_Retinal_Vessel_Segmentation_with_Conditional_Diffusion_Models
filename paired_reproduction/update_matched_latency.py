import csv,json,re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1]
out=sorted((root/'paired_reproduction').glob('pipeline_latency_*'))[-1]
s=json.loads((out/'summary.json').read_text())
rows=list(csv.DictReader((out/'measurements.csv').open()))
assert len(rows)==320
for mode,stats in s.items():
 values=[float(r['ms']) for r in rows if r['mode']==mode]
 assert len(values)==80 and abs(sum(values)/80-stats['mean_ms'])<1e-8
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
p=root/'template261006.tex';old=p.read_text(encoding='utf8');new=old
names={'baseline_only':'LU-Net+RA only','refinement_only':'Refinement only','model_pipeline':'Complete model pipeline','decoded_image_to_cpu_mask':'Decoded RGB to CPU binary mask'}
idx=new.index('\\label{tab:pipeline_latency}');end=new.index('\\end{table}',idx)
part=new[idx:end]
for mode,stats in s.items():
 name=names[mode]
 replacement=f"{name} & {stats['mean_ms']:.2f} & {stats['sd_ms']:.2f} & {stats['median_ms']:.2f} & {stats['p95_ms']:.2f} & {stats['serial_images_per_s']:.2f} \\\\"
 part,count=re.subn(re.escape(name)+r' & [^\n]+',lambda m:replacement,part)
 assert count==1
new=new[:idx]+part+new[end:]
marker='\\paragraph{CUDA-event timing.}'
assert new.count(marker)==1
model=s['model_pipeline'];full=s['decoded_image_to_cpu_mask']
note=f"All four wall-clock timing modes in Table~\\ref{{tab:pipeline_latency}} were remeasured in one session ({out.name.removeprefix('pipeline_latency_').replace('_',' ')}, Asia/Tehran). Complete-model latency was {model['mean_ms']:.2f} ms, and decoded-image-to-CPU-mask latency was {full['mean_ms']:.2f} ms. These values replace the earlier wall-clock session in this table; they must not be combined with CUDA-event results from a different session to estimate preprocessing overhead. The two full-pipeline modes were measured directly, rather than by summing separately timed stages.\n\n"
new=new.replace(marker,note+marker)
(root/f'template261006_before_matched_latency_{stamp}.tex').write_text(old,encoding='utf8')
p.write_text(new,encoding='utf8');version=root/f'template261006_{stamp}.tex';version.write_text(new,encoding='utf8')
reply=f'''Thank you for highlighting the timing scope. The original approximately 10 images/s refers only to refinement. We have clarified this and report a new, matched timing benchmark of the complete available reproduction, including LU-Net+RA and refinement.

All stages were remeasured in one session on an NVIDIA GeForce RTX 4070 Ti SUPER, with FP32, batch size one, 512 x 512 model inputs, and 50 DDIM transitions. The complete model, from preprocessed GPU inputs to the GPU probability map, required {model['mean_ms']:.2f} ms on average (SD {model['sd_ms']:.2f} ms; median {model['median_ms']:.2f} ms; P95 {model['p95_ms']:.2f} ms). The broader pipeline, from native-resolution RGB already decoded in CPU memory to the returned CPU binary mask, required {full['mean_ms']:.2f} ms (SD {full['sd_ms']:.2f} ms; median {full['median_ms']:.2f} ms; P95 {full['p95_ms']:.2f} ms), equivalent to {full['serial_images_per_s']:.2f} images/s for serial execution. This broader timing includes CPU preprocessing, transfers to GPU, LU-Net+RA, mask encoding, RGB conditioning, Gaussian generation, DDIM sampling, decoding, thresholding, and return to CPU. It excludes disk access/JPEG decoding, checkpoint loading, metric calculation, and file saving.

For each timing mode, eight images received five warm-ups and ten CUDA-synchronised timed repetitions, with rotated measurement order (80 measurements per mode). Table "Measured latency of the new reproduction at 50 DDIM transitions" now reports the matched session. Separate CUDA-event measurements are explicitly identified as a different session and are not combined with these measurements. This benchmark uses the available new reproduction checkpoints; it does not retrospectively verify the unavailable original runs or change the original Dice results.
'''
response=root/f'reviewer_complete_pipeline_latency_response_{stamp}.txt';response.write_text(reply,encoding='utf8')
print(json.dumps(dict(version=str(version),response=str(response),report=str(out/'experiment_report.txt'),summary=s),indent=2))
