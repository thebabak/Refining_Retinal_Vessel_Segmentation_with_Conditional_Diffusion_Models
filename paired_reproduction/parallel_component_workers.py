import sys,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from run_component_ablations import SEEDS,ROOT,HERE
out=Path(sys.argv[1]);tag=out.name[-19:]
jobs=[(d,s) for s in reversed(SEEDS) for d in ['HRF','DRIVE','CHASE'] if (d,s)!=('CHASE',42)]
def run(pair):
    d,s=pair
    with (out/f'parallel_{d}_seed{s}_{tag}.log').open('a') as f:
        p=subprocess.run([sys.executable,str(HERE/'run_component_ablations.py'),'--output',str(out),'--dataset',d,'--seed',str(s)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'Parallel worker failed {d} {s}')
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
