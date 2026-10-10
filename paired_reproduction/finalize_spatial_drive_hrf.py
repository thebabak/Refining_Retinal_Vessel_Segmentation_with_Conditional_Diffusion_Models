import csv,json,zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parent.parent
out=Path((root/'paired_reproduction/current_spatial_drive_hrf_path.txt').read_text().strip())
tag=out.name[-19:];now=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
assert json.loads((out/'status.json').read_text())['stage']=='completed'
results=json.loads((out/f'verified_results_{tag}.json').read_text())
chase=root/'revise/new_spatial_refiner_2026-10-09_10-34-17'
ch=json.loads((chase/'verified_results_2026-10-09_10-34-17.json').read_text());ch['dataset']='CHASE-DB1'
summaries=[ch]+results
table=out/f'all_datasets_spatial_refiner_summary_{now}.txt'
fields=['dataset','baseline_mean','baseline_SD','refined_mean','refined_SD','mean_delta','paired_t_p','exact_signed_rank_p','crossed_bootstrap_95CI']
with table.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in summaries)
allrows=list(csv.DictReader((chase/'all_per_image_2026-10-09_10-34-17.txt').open()))
for r in allrows:r['dataset']='CHASE-DB1';r['pathology']=''
allrows+=list(csv.DictReader((out/f'all_per_image_{tag}.txt').open()))
assert len(allrows)==200
combined=out/f'all_datasets_200_per_image_records_{now}.txt'
with combined.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
lines=['New spatial RGB+coarse residual U-Net control;488145 learned parameters;BCE/Dice;five independent refiner training seeds per dataset.',
       'Baseline frozen from available executable checkpoints. Each baseline/refiner pair evaluated on identical inputs/GT/region;validated checkpoint selection including identity.',
       'Original supplied tables unchanged;new results kept separately. This is not a paired comparison to original diffusion scores or a parameter-matched diffusion ablation.',
       '40 CHASE,100 DRIVE,60 HRF image-seed records;all200 checked against saved FP32 predictions and confusion counts.',
       'CHASE intervals resample4subjects with both eyes retained;DRIVE image-level;HRF pathology-stratified image-level;training seeds also resampled50,000times.',
       'Training-seed paired t and exact signed-rank tests. DRIVE/HRF have separate Holm correction across the two new comparisons. CHASE was earlier standalone;do not treat combined nominal values as a new multiplicity-controlled family.',
       'Small fixed test cohorts and unknown patient independence for DRIVE/HRF limit interpretation. Signed-rank resolution with only5seeds is coarse.','']
for r in summaries:
    epochs=ch['selection_history'] if r is ch else r['training_history']
    lines += [f"{r['dataset']}: baseline {r['baseline_mean']:.9f} +/- {r['baseline_SD']:.9f}; refiner {r['refined_mean']:.9f} +/- {r['refined_SD']:.9f}; delta {r['mean_delta']:+.9f};CI {r['crossed_bootstrap_95CI']};t p {r['paired_t_p']};signed-rank p {r['exact_signed_rank_p']}",
              'Selected epochs: '+str([(c['seed'],c['selected_epoch']) for c in epochs])]
report=out/f'all_datasets_experiment_report_{now}.txt';report.write_text('\n'.join(lines)+'\n')
archive=out/f'spatial_refiner_complete_data_{tag}.zip'
with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED) as z:
    for p in [table,combined,report]:z.write(p,arcname=p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
tracking=root/'reviewer_comments/reviewer_2_comments.json';obj=json.loads(tracking.read_text(encoding='utf-8-sig'));c=next(c for c in obj['comments'] if c['number']==3)
c.update(updated_at=now,spatial_refiner_all_dataset_report=str(report),spatial_refiner_all_dataset_records=str(combined),spatial_refiner_drive_hrf_directory=str(out),spatial_refiner_status='Five spatial-refiner trainings per dataset completed;200 verified image-seed pairs across3datasets. New paired baseline-control comparison only;original diffusion-score pairing and named related-method benchmarks remain unavailable. Manuscript unchanged.')
tracking.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');(tracking.parent/f'reviewer_2_comments_{now}.json').write_bytes(tracking.read_bytes())
txt=tracking.with_suffix('.txt')
with txt.open('a',encoding='utf-8') as f:f.write(f'\n{now}: Comment3 spatial refiner extended to DRIVE/HRF with5trainings each. All3datasets now200verified paired records. No paired comparison to original diffusion. Report:{report}\n')
(tracking.parent/f'reviewer_2_comments_{now}.txt').write_bytes(txt.read_bytes())
with (root/'AGENTS.md').open('a',encoding='utf-8') as f:f.write(f'\n\n# Spatial refiner extended to DRIVE and HRF\nOn {now}, user requested completing the other datasets. Five new spatial residual U-Net trainings per DRIVE/HRF completed,160 image-seed records verified with saved FP32 predictions/GT/FOV. Combined with preceding CHASE experiment:15refiner trainings,200 verified records. Original supplied tables remain unchanged. New directory:{out}. Combined report:{report}. This is paired coarse-to-refiner evaluation only;not paired to original diffusion. Manuscript unchanged.\n')
print(report)
print('\n'.join(lines[-6:]))
