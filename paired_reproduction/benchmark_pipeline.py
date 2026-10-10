"""Auditable latency benchmark of the available NEW reproduction, not original runs."""
import json, csv, time, hashlib, platform
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/('pipeline_latency_'+datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S'))
OUT.mkdir()
experiment_output=OUT
source=HERE/'run.py'
prefix=source.read_text().split('model=LNet_RA(in_channels=1).to(DEVICE)')[0]
prefix=prefix.replace("OUT = Path(__file__).resolve().parent / 'results'",'OUT = experiment_output')
exec(compile(prefix,str(source),'exec'),globals())
baseline=LNet_RA(in_channels=1).to(DEVICE)
baseline.load_state_dict(torch.load(HERE/'results/baseline.pt',weights_only=False)['state']);baseline.eval()
ae=MaskAutoencoder().to(DEVICE)
ae.load_state_dict(torch.load(HERE/'results/autoencoder.pt',weights_only=True));ae.eval()
ie=ImageEncoder().to(DEVICE);un=DiffusionUNet().to(DEVICE)
tm=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(DEVICE)
saved=torch.load(HERE/'results/refiner.pt',weights_only=True)
for key,m in [('ie',ie),('un',un),('time',tm)]:m.load_state_dict(saved[key]);m.eval()
t=torch.arange(1001,device=DEVICE)/1000
a=torch.cos((t+.008)/1.008*math.pi/2)**2;a=a/a[0]
alpha=torch.cumprod(1-(1-a[1:]/a[:-1]).clamp(max=.999),0)
ts=torch.linspace(99,0,50,device=DEVICE).long()
def emb(t):
    v=t[:,None]*torch.exp(-math.log(10000)*torch.arange(32,device=DEVICE)/32)
    return torch.cat([v.sin(),v.cos()],1)
def refine(coarse,rgb):
    z0=ae.encode(coarse);c=ie(rgb)
    z=alpha[99].sqrt()*z0+(1-alpha[99]).sqrt()*torch.randn_like(z0)
    for j,ti in enumerate(ts):
        e=emb(ti[None]);temporal=tm(e)
        e0=un(z,e,temporal);e1=un(z,e,c+temporal)
        eps=e0+1.5*(e1-e0)
        estimate=(z-(1-alpha[ti]).sqrt()*eps)/alpha[ti].sqrt()
        prev=alpha[ts[j+1]] if j+1<len(ts) else torch.tensor(1.,device=DEVICE)
        z=prev.sqrt()*estimate+(1-prev).sqrt()*eps
    return ae.decode(z)
raw={n:np.asarray(Image.open(ROOT/'data'/f'{n}.jpg').convert('RGB')) for n in manifest['test']}
def prepare(rgb):
    green=cv2.createCLAHE(clipLimit=5.,tileGridSize=(32,32)).apply(rgb[:,:,1])
    green=np.uint8(np.clip((green/255.)**(1/1.2)*255,0,255))
    green=cv2.resize(green,(512,512),interpolation=cv2.INTER_LINEAR)
    color=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR)
    x=torch.tensor(green/255.,dtype=torch.float32)[None,None].to(DEVICE)
    c=torch.tensor(color/255.,dtype=torch.float32).permute(2,0,1)[None].to(DEVICE)
    return x,c
def full(rgb):
    x,c=prepare(rgb)
    return (refine(baseline(x),c)>=.5).cpu().numpy()
rows=[]
with torch.inference_mode():
    for name in manifest['test']:
        x=data[name]['x'][None].to(DEVICE);rgb=data[name]['rgb'][None].to(DEVICE);coarse=baseline(x)
        calls={'baseline_only':lambda:baseline(x),'refinement_only':lambda:refine(coarse,rgb),
               'model_pipeline':lambda:refine(baseline(x),rgb),'decoded_image_to_cpu_mask':lambda:full(raw[name])}
        for fn in calls.values():
            for _ in range(5):fn()
        for rep in range(10):
            modes=list(calls);modes=modes[rep%4:]+modes[:rep%4]
            for mode in modes:
                torch.cuda.synchronize();start=time.perf_counter();result=calls[mode]();torch.cuda.synchronize()
                ms=1000*(time.perf_counter()-start)
                rows.append(dict(image=name,repetition=rep,mode=mode,ms=ms))
                if isinstance(result,torch.Tensor):assert torch.isfinite(result).all()
        print('timed',name,flush=True)
with (OUT/'measurements.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for mode in calls:
    values=np.array([r['ms'] for r in rows if r['mode']==mode])
    summary[mode]=dict(n=len(values),mean_ms=float(values.mean()),sd_ms=float(values.std(ddof=1)),median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),serial_images_per_s=float(1000/values.mean()))
protocol=dict(claim='New available CHASE reproduction; does not validate original timing or original Dice',gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),cpu=platform.processor(),size=[512,512],batch_size=1,precision='FP32',warmup_per_image_per_mode=5,repetitions_per_image=10,n_images=8,transitions=50,guidance=1.5,start_timestep=99,gradients=False,timing='CUDA-synchronised perf_counter; rotated mode order',threads=torch.get_num_threads(),disk_decode_excluded=True,definitions={'model_pipeline':'GPU-resident preprocessed inputs -> LU-Net+RA -> mask encoder -> RGB conditioning -> Gaussian generation -> 50 DDIM transitions (two denoiser calls each) -> decoder; output GPU probability map','decoded_image_to_cpu_mask':'Predecoded native RGB in CPU memory -> CLAHE/gamma/resize/normalisation -> CPU-to-GPU transfers -> complete model pipeline -> threshold0.5 -> GPU-to-CPU binary mask; excludes file reading/JPEG decode, loading checkpoints, metric calculation, saving'},checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'results').glob('*.pt')},limitations=['One machine and benchmark session','Eight images; repeated timings are not independent patient samples','Serial reciprocal mean latency, not concurrent or batched throughput','New implementation/checkpoints; original files deleted'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'benchmark_protocol.json').write_text(json.dumps(protocol,indent=2))
(OUT/'experiment_report.txt').write_text(json.dumps(dict(protocol=protocol,results=summary),indent=2))
print('OUTPUT',OUT,flush=True);print(json.dumps(summary,indent=2),flush=True)
