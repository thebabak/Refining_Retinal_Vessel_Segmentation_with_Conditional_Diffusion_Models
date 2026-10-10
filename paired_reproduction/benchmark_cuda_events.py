"""CUDA-event timing, alongside separate wall-clock latency; no model training."""
from pathlib import Path
source=Path(__file__).resolve().parent/'benchmark_pipeline.py'
prefix=source.read_text().split('raw={')[0].replace("'pipeline_latency_'","'cuda_event_latency_'")
exec(compile(prefix,str(source),'exec'),globals())
rows=[]
with torch.inference_mode():
    for name in manifest['test']:
        x=data[name]['x'][None].to(DEVICE);rgb=data[name]['rgb'][None].to(DEVICE);coarse=baseline(x)
        calls={'baseline_only':lambda:baseline(x),'refinement_only':lambda:refine(coarse,rgb),'model_pipeline':lambda:refine(baseline(x),rgb)}
        for fn in calls.values():
            for _ in range(5):fn()
        torch.cuda.synchronize()
        for rep in range(10):
            modes=list(calls);modes=modes[rep%3:]+modes[:rep%3]
            for mode in modes:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                wall_start=time.perf_counter();start.record()
                result=calls[mode]()
                end.record();end.synchronize();wall_ms=1000*(time.perf_counter()-wall_start)
                ms=start.elapsed_time(end)
                assert torch.isfinite(result).all()
                rows.append(dict(image=name,repetition=rep,mode=mode,cuda_event_ms=ms,synchronised_wall_ms=wall_ms))
        print('timed',name,flush=True)
with (OUT/'measurements.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for mode in calls:
    summary[mode]={}
    for metric in ('cuda_event_ms','synchronised_wall_ms'):
        v=np.array([r[metric] for r in rows if r['mode']==mode])
        summary[mode][metric]=dict(n=len(v),mean_ms=float(v.mean()),sd_ms=float(v.std(ddof=1)),median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),reciprocal_mean_images_per_s=float(1000/v.mean()))
protocol=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,precision='FP32',batch_size=1,input_size=[512,512],transitions=50,start_timestep=99,guidance=1.5,warmups_per_image_mode=5,repetitions_per_image_mode=10,images=manifest['test'],timing='CUDA events on current stream; same-call synchronised wall clock also recorded; rotated mode order',scope='GPU-resident preprocessed inputs to GPU probability map; excludes preprocessing, transfers, file operations and model loading; complete model includes LU-Net+RA, mask encoder, RGB conditioning, noise generation, 50 DDIM transitions with 2 denoiser calls each, decoder',limitations=['New available reproduction, not original manuscript checkpoints','CUDA events measure stream elapsed time, including possible gaps between kernel launches; not sum of kernel execution durations','Reciprocal event latency is not measured deployment throughput','One machine/session'],checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'results').glob('*.pt')})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'benchmark_protocol.json').write_text(json.dumps(protocol,indent=2))
(OUT/'experiment_report.txt').write_text(json.dumps(dict(protocol=protocol,results=summary),indent=2))
print('OUTPUT',OUT,flush=True);print(json.dumps(summary,indent=2),flush=True)
