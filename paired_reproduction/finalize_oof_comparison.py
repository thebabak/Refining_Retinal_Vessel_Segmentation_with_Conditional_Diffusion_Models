"""Complete-metric verification and timestamped deliverables for OOF study."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv, json, hashlib, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
out=Path((ROOT/'paired_reproduction/current_oof_comparison_path.txt').read_text().strip())
tag=out.name[-19:]
assert (out/f'completed_{tag}.json').exists()
now=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
rows=list(csv.DictReader((out/f'all_verified_per_image_{tag}.txt').open()))
assert len(rows)==2000
for r in rows:
    for prefix in ['baseline','refined']:
        tp,fp,fn,tn=[int(r[prefix+'_'+k]) for k in ['tp','fp','fn','tn']]
        expected=dict(dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),sensitivity=tp/max(tp+fn,1),specificity=tn/max(tn+fp,1),accuracy=(tp+tn)/(tp+fp+fn+tn))
        for k,v in expected.items():assert abs(float(r[prefix+'_'+k])-v)<1e-12
    assert abs(float(r['delta'])-float(r['refined_dice'])+float(r['baseline_dice']))<1e-12
paired=[]
for dataset in ['CHASE','DRIVE','HRF']:
    for seed in [42,123,2026,3000,3001]:
        images=sorted({r['image'] for r in rows if r['dataset']==dataset and int(r['training_seed'])==seed})
        job=out/dataset/f'seed_{seed}'
        protocol=json.loads((job/f'oof_protocol_{tag}.json').read_text())
        for fold in job.glob('fold_*'):
            inventory=json.loads((fold/f'fold_inventory_{tag}.json').read_text())
            held,fit,val=map(set,[inventory['held_out'],inventory['fit'],inventory['validation']])
            assert held.isdisjoint(fit|val|set(protocol['split']['test']))
            assert fit|held==set(protocol['split']['train'])
            assert inventory['checkpoint_sha256']==hashlib.sha256((fold/'baseline.pt').read_bytes()).hexdigest()
        for image in images:
            arms={a:[r for r in rows if r['dataset']==dataset and int(r['training_seed'])==seed and r['image']==image and r['arm']==a] for a in ['in_sample','out_of_fold']}
            assert all(len(a)==5 for a in arms.values())
            means={a:float(np.mean([float(r['refined_dice']) for r in rr])) for a,rr in arms.items()}
            first=arms['in_sample'][0]
            paired.append(dict(dataset=dataset,image=image,subject=first['subject'],pathology=first['pathology'],training_seed=seed,sampling_realisations_per_arm=5,baseline_dice=first['baseline_dice'],in_sample_refined_dice=means['in_sample'],oof_refined_dice=means['out_of_fold'],oof_minus_in_sample=means['out_of_fold']-means['in_sample']))
assert len(paired)==200
pairedfile=out/f'paired_per_image_sampling_averaged_{now}.txt'
with pairedfile.open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
summary=list(csv.DictReader((out/f'mean_and_SD_{tag}.txt').open()))
tests=json.loads((out/f'paired_tests_{tag}.json').read_text())
response=['Response to Reviewer 2, Comment 6','',
'Thank you for this comment. We confirmed that the available implementation uses in-sample coarse predictions for refiner training. We have therefore executed a separate matched comparison using out-of-fold coarse predictions on CHASE-DB1, DRIVE, and HRF, with five refiner training seeds per dataset.',
'CHASE-DB1 uses four subject-grouped folds, retaining both eyes together; DRIVE uses four image folds; HRF uses three pathology-stratified folds. Each fold baseline excludes the held-out training images from both fitting and checkpoint selection. Checkpoint selection uses the separate fixed validation set (maximum validation mean Dice; maximum 1,000 epochs, patience 20). No test images enter fitting or checkpoint selection.',
'The two refiner arms share the retained full-training baseline and its test predictions, the frozen initial autoencoder, architecture, initial refiner weights, optimizer, loss, training RNG, 100-epoch final-checkpoint rule, and matched sampling seeds at 50 DDIM transitions. Only the coarse maps supplied during refiner training differ. The in-sample comparator is the saved, previously executed component-study reference; its checkpoint and predictions are preserved rather than retrained.',
'','Dataset | In-sample Dice mean +/- SD | OOF Dice mean +/- SD | OOF minus in-sample | exploratory95%CI | Holm paired-t p | Holm exact signed-rank p']
for t in tests:
    d=t['dataset'];a=next(r for r in summary if r['dataset']==d and r['arm']=='in_sample');b=next(r for r in summary if r['dataset']==d and r['arm']=='out_of_fold')
    response.append(f"{d} | {float(a['mean_dice']):.9f} +/- {float(a['sd_between_training_seeds']):.9f} | {float(b['mean_dice']):.9f} +/- {float(b['sd_between_training_seeds']):.9f} | {t['mean_oof_minus_in_sample']:+.9f} | {t['crossed_bootstrap_95ci']} | {t['holm_paired_t_p']:.9g} | {t['holm_exact_signed_rank_p']:.9g}")
response.extend(['','SD is across five trained refiners after averaging sampling realisations. Paired tests use matched training-seed means with Holm correction across three datasets separately for each test family. Exploratory crossed-bootstrap intervals use50,000 resamples of training seeds and test units, grouping both eyes of each CHASE-DB1 subject and stratifying HRF image resampling by pathology. Five seed pairs and small fixed test cohorts limit inference; non-significance does not establish equivalence or absence of shift.',
'The OOF fold baselines have fewer fitting examples, so this comparison combines the effect of held-out predictions with reduced baseline training size. The shared DRIVE autoencoder remains pretrained on in-sample coarse maps; the experiment isolates the refiner-training coarse-map change and does not eliminate every possible input shift. DRIVE/HRF patient independence and upstream HRF annotation conversion provenance remain unresolved.',
'These are newly executed available-configuration results, not recovery or verification of the original author-supplied positive diffusion scores. No original table was replaced. The manuscript has not been modified by this experiment.'])
responsefile=out/f'reviewer_2_comment_6_response_{now}.txt';responsefile.write_text('\n'.join(response)+'\n',encoding='utf-8')
manifest=out/f'final_verification_{now}.json';manifest.write_text(json.dumps(dict(verified_records=2000,paired_sampling_averaged_records=200,fold_models=55,new_refiners=15,completed_at=now,all_confusion_counts_and_metrics='PASS',fold_membership_and_checkpoint_hashes='PASS'),indent=2),encoding='utf-8')
archive=out/f'oof_comparison_results_{now}.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.suffix in ['.txt','.json','.log','.py']:
            z.write(p,p.relative_to(out).as_posix())
    z.write(Path(__file__),f'source_finalize_oof_comparison_{now}.py')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps(dict(archive=str(archive),response=str(responsefile),paired_table=str(pairedfile),summary=summary,tests=tests),indent=2))
