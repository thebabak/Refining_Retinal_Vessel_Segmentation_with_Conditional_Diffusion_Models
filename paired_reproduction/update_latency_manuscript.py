import json,re,csv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1]
out=sorted((root/'paired_reproduction').glob('pipeline_latency_*'))[-1]
s=json.loads((out/'summary.json').read_text())
protocol=json.loads((out/'benchmark_protocol.json').read_text())
rows=list(csv.DictReader((out/'measurements.csv').open()))
assert len(rows)==320
for mode,v in s.items():
    vals=[float(r['ms']) for r in rows if r['mode']==mode]
    assert len(vals)==80 and abs(sum(vals)/80-v['mean_ms'])<1e-9
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
p=root/'template261006.tex';old=p.read_text(encoding='utf-8')
(root/f'template261006_before_latency_{stamp}.tex').write_text(old,encoding='utf-8')
names={'baseline_only':'LU-Net+RA only','refinement_only':'Refinement only','model_pipeline':'Complete model pipeline','decoded_image_to_cpu_mask':'Decoded RGB to CPU binary mask'}
table='\n'.join(f"{names[k]} & {v['mean_ms']:.2f} & {v['sd_ms']:.2f} & {v['median_ms']:.2f} & {v['p95_ms']:.2f} & {v['serial_images_per_s']:.2f} \\\\" for k,v in s.items())
new_section=r'''
\subsection{Complete-Pipeline Latency on the Available Reproduction}
\label{sec:pipeline_latency}

The approximately 10 images/s value in Table~\ref{tab:ddim_steps} is refinement-stage throughput only. To measure the complete segmentation pipeline, we performed a separate benchmark using the available newly trained CHASE-DB1 reproduction checkpoints, including the official LU-Net+RA backbone and the 527,233-parameter refiner configuration. These checkpoints are distinct from the unavailable original runs; the new measurements are not a retrospective validation of the original timing or Dice values.

We used an NVIDIA GeForce RTX 4070 Ti SUPER, PyTorch 2.9.1 with CUDA 12.8, FP32, evaluation mode, disabled gradient computation, batch size one, and $512\times512$ model inputs. The sampler used 50 DDIM transitions, starting timestep 99, and classifier-free guidance 1.5 (two denoiser evaluations per transition). Each of eight test images received five warm-up executions per timing mode followed by ten measured repetitions, giving 80 measurements per mode. Measurement order was rotated across repetitions. Wall-clock timing used CUDA synchronisation immediately before and after each measured call. We report mean, standard deviation (SD), median, and the empirical 95th percentile (P95); P95 is a latency percentile, not a confidence interval.

\begin{table}[htbp]
\centering
\caption{Measured latency of the new reproduction at 50 DDIM transitions. All latency columns are in milliseconds. Serial rate is $1000/\overline{T}$ images/s, where $\overline{T}$ is mean latency; it does not represent batched or concurrent throughput.}
\label{tab:pipeline_latency}
\small
\begin{tabular}{lrrrrr}
\toprule
Timing scope & Mean & SD & Median & P95 & Serial rate \\
\midrule
TABLE_ROWS
\bottomrule
\end{tabular}
\end{table}

The complete model measurement runs LU-Net+RA and the refiner sequentially from GPU-resident preprocessed inputs to the GPU probability map. It includes mask encoding, RGB conditioning, Gaussian-noise generation, DDIM sampling, and mask decoding; no cached coarse prediction is used in this mode. The broader decoded-image-to-mask measurement additionally includes CPU green-channel extraction, CLAHE, inverse gamma correction, resizing, normalisation, input transfers to the GPU, thresholding at 0.5, and transfer of the binary mask to CPU memory. It starts from a native-resolution RGB image already decoded in CPU memory and excludes disk access/JPEG decoding, checkpoint loading, metric calculation, and output-file saving. Thus its scope is explicitly decoded image to returned segmentation mask, rather than file-to-file processing. These are measurements from one machine and session; repeated timing trials do not provide independent patient samples or establish deployment performance on other hardware.

'''.replace('TABLE_ROWS',table).replace('PyTorch 2.9.1 with CUDA 12.8',f"PyTorch {protocol['torch']} with CUDA {protocol['cuda']}")
marker='\\subsection{Qualitative Results}'
assert old.count(marker)==1
new=old.replace(marker,new_section+marker)
new=new.replace('with throughput decreasing from approximately 50 to 10 images/s','with refinement-stage throughput decreasing from approximately 50 to 10 images/s')
new=new.replace('while throughput decreased from 10 to five images/s','while refinement-stage throughput decreased from 10 to five images/s')
new=new.replace('The coarse LU-Net+RA forward pass is excluded, so the values do not describe complete-pipeline latency or establish real-time performance.','The coarse LU-Net+RA forward pass is excluded from those original estimates, so they do not describe complete-pipeline latency or establish real-time performance. A separate benchmark of the available new reproduction includes backbone inference and explicitly defined preprocessing and transfer costs (Section~\\ref{sec:pipeline_latency}).')
assert sorted(re.findall(r'\\begin\{([^}]+)\}',new_section))==sorted(re.findall(r'\\end\{([^}]+)\}',new_section))
p.write_text(new,encoding='utf-8');version=root/f'template261006_{stamp}.tex';version.write_text(new,encoding='utf-8')
model=s['model_pipeline'];full=s['decoded_image_to_cpu_mask'];ref=s['refinement_only'];base=s['baseline_only']
reply=f'''Thank you for pointing out this omission. We agree that the approximately 10 images/s originally reported at 50 DDIM transitions refers only to refinement and is not complete-pipeline throughput. We have made that distinction explicit and added a separate complete-pipeline latency benchmark in Section "Complete-Pipeline Latency on the Available Reproduction" and Table "Measured latency of the new reproduction at 50 DDIM transitions."

On an NVIDIA GeForce RTX 4070 Ti SUPER with FP32, batch size one, and 512 x 512 model inputs, the available reproduction required {base['mean_ms']:.2f} ms for LU-Net+RA alone and {ref['mean_ms']:.2f} ms for refinement alone. Directly timing LU-Net+RA followed by refinement gave {model['mean_ms']:.2f} ms mean latency (SD {model['sd_ms']:.2f} ms; median {model['median_ms']:.2f} ms; P95 {model['p95_ms']:.2f} ms), equivalent to {model['serial_images_per_s']:.2f} images/s for serial execution. Including preprocessing, input transfers, thresholding, and return of the binary mask to CPU gave {full['mean_ms']:.2f} ms (SD {full['sd_ms']:.2f} ms; median {full['median_ms']:.2f} ms; P95 {full['p95_ms']:.2f} ms), or {full['serial_images_per_s']:.2f} images/s. The latter starts from a decoded RGB image in CPU memory and excludes disk/JPEG decoding, checkpoint loading, metric calculation, and saving.

Each timing mode used five warm-ups and ten CUDA-synchronised measurements for each of eight images (80 measurements per mode), with rotated measurement order. Both the timing scope and dispersion are now reported. Because the original checkpoints are unavailable, this benchmark uses the separately documented new reproduction and is explicitly identified as such; it does not retrospectively verify the original timing or original Dice results.
'''
response=root/f'reviewer_complete_pipeline_latency_response_{stamp}.txt';response.write_text(reply,encoding='utf-8')
print(json.dumps(dict(version=str(version),response=str(response),report=str(out/'experiment_report.txt'),summary=s),indent=2))
