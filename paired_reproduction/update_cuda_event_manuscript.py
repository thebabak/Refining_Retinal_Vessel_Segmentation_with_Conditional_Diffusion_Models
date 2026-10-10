import json,csv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1]
out=sorted((root/'paired_reproduction').glob('cuda_event_latency_*'))[-1]
s=json.loads((out/'summary.json').read_text());protocol=json.loads((out/'benchmark_protocol.json').read_text())
rows=list(csv.DictReader((out/'measurements.csv').open()))
assert len(rows)==240
for mode,v in s.items():
 for metric,stats in v.items():
  a=[float(r[metric]) for r in rows if r['mode']==mode]
  assert len(a)==80 and abs(sum(a)/80-stats['mean_ms'])<1e-7
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
p=root/'template261006.tex';old=p.read_text(encoding='utf8')
(root/f'template261006_before_cuda_events_{stamp}.tex').write_text(old,encoding='utf8')
names={'baseline_only':'LU-Net+RA only','refinement_only':'Refinement only','model_pipeline':'Complete model pipeline'}
table='\n'.join(f"{names[k]} & {v['cuda_event_ms']['mean_ms']:.2f} & {v['cuda_event_ms']['sd_ms']:.2f} & {v['cuda_event_ms']['p95_ms']:.2f} & {v['synchronised_wall_ms']['mean_ms']:.2f} \\\\" for k,v in s.items())
text=r'''
\paragraph{CUDA-event timing.}
In a separate timing session, we additionally measured CUDA-stream elapsed time with CUDA events on the same GPU, using the same reproduction checkpoints, FP32, batch size one, and 50 DDIM transitions. Inputs were already preprocessed and resident on the GPU, and outputs were GPU probability maps. Each of eight test images had five warm-ups and ten repetitions per timing mode, with rotated measurement order (80 measurements per mode). A synchronised wall-clock measurement was recorded for the same call alongside every CUDA-event measurement. Both backbone inference and refinement are executed within the complete-model event interval. CPU preprocessing, host-device transfers, and file operations are excluded from this measurement.

\begin{table}[htbp]
\centering
\caption{CUDA-event timing of the available new reproduction, in milliseconds. SD and P95 describe the event measurements; the final column gives same-call mean synchronised wall latency. This is a separate session from Table~\ref{tab:pipeline_latency}.}
\label{tab:cuda_event_latency}
\small
\begin{tabular}{lrrrr}
\toprule
Timing scope & Event mean & Event SD & Event P95 & Wall mean \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}

CUDA events measure elapsed time on the stream, which can include idle gaps between CPU kernel launches; they are not a sum of isolated kernel durations. Their reciprocal should not be presented as measured end-to-end application throughput. The separately reported decoded-image-to-CPU-mask wall latency retains the preprocessing and transfer costs relevant to the broader segmentation pipeline. Differences between timing sessions cannot be attributed solely to the timing method. These new measurements do not validate the unavailable original runs.

'''.replace('ROWS',table)
marker='\\subsection{Qualitative Results}';assert old.count(marker)==1
new=old.replace(marker,text+marker)
p.write_text(new,encoding='utf8');version=root/f'template261006_{stamp}.tex';version.write_text(new,encoding='utf8')
base_response=sorted(root.glob('reviewer_complete_pipeline_latency_response_*.txt'))[-1].read_text(encoding='utf8')
full=s['model_pipeline']
addition=f"\nWe also added separate CUDA-event measurements from GPU-resident preprocessed inputs to the GPU probability map, including both LU-Net+RA and refinement. In that session, complete-model event latency averaged {full['cuda_event_ms']['mean_ms']:.2f} ms (SD {full['cuda_event_ms']['sd_ms']:.2f} ms; P95 {full['cuda_event_ms']['p95_ms']:.2f} ms), while same-call synchronised wall latency averaged {full['synchronised_wall_ms']['mean_ms']:.2f} ms. These measurements exclude preprocessing and transfers and are explicitly distinguished from the broader end-to-end wall timing. CUDA events measure stream elapsed time, including possible launch gaps, and do not establish complete application throughput.\n"
response=root/f'reviewer_complete_pipeline_latency_response_{stamp}.txt';response.write_text(base_response+addition,encoding='utf8')
print(json.dumps(dict(version=str(version),response=str(response),report=str(out/'experiment_report.txt'),results=s),indent=2))
