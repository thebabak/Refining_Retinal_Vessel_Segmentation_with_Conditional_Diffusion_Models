import csv,json,zipfile,hashlib,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parent.parent
out=Path((root/'paired_reproduction/current_component_ablations_path.txt').read_text().strip())
run=out.name[-19:];tag=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
completion=json.loads((out/f'verified_completion_{run}.json').read_text())
assert completion['trained_models']==105 and completion['verified_records']==7200
summary=list(csv.DictReader((out/f'mean_and_SD_{run}.txt').open()))
tests=json.loads((out/f'paired_tests_{run}.json').read_text())
assert len(summary)==27 and len(tests)==24
lookup={(r['dataset'],r['variant']):r for r in summary}
labels={'coarse_baseline':'Coarse baseline','global_frozen':'Reference: coarse diffusion, global FiLM, frozen AE',
        'deterministic':'Matched latent deterministic refiner','target_warm':'Target diffusion, coarse warm start',
        'target_full':'Target diffusion, full Gaussian start','no_image':'No RGB conditioning',
        'spatial_film':'Spatial RGB conditioning, FiLM','global_additive':'Global additive injection',
        'ae_finetuned':'Jointly fine-tuned mask autoencoder'}
lines=['Thank you for this comment. We conducted a new controlled component-ablation study on CHASE-DB1, DRIVE, and HRF. For each dataset we used five matched training-seed settings, retaining exactly the same per-seed coarse-predictor checkpoint, initial autoencoder, data split, coarse probability maps, and evaluation protocol. Seven configurations were independently trained for100epochs each, producing105 trained refiners. Five matched sampling realisations were evaluated at50 DDIM transitions for stochastic methods; the deterministic model was evaluated once per image. All7,200 image/training-seed/sampling records were verified against saved FP32 predictions, ground truth, FOV masks, and confusion counts.',
       '', 'The study includes the four requested controls—deterministic refinement, target diffusion, no image conditioning, and spatial rather than global conditioning—and additionally compares additive versus FiLM injection and frozen versus jointly fine-tuned autoencoder weights. Target diffusion is evaluated with both coarse warm-start and full-Gaussian initialisation, using the same target-trained checkpoint. The warm-start comparison changes the diffused training target; full-Gaussian inference also changes initialisation and is therefore not a single-factor comparison.',
       '', 'Mean Dice +/- between-training-seed SD:', 'Configuration | CHASE-DB1 | DRIVE | HRF']
for mode,label in labels.items():
    values=[]
    for dataset in ['CHASE','DRIVE','HRF']:
        r=lookup[(dataset,mode)];values.append(f"{float(r['mean_dice']):.9f} +/- {float(r['sd_between_training_seeds']):.9f}")
    lines.append(label+' | '+' | '.join(values))
lines+=['', 'We report paired training-seed t tests, exact signed-rank tests, separate Holm adjustment across24 comparisons for each test family, and exploratory crossed-bootstrap95% intervals with50,000 resamples. CHASE resampling retains both eyes within each of four subjects; DRIVE uses image-level resampling, and HRF uses pathology-stratified image resampling. Sampling repetitions are averaged before calculating between-training-seed SD; they are not counted as independent training runs.',
        '', 'Global FiLM, spatial FiLM, target diffusion, additive injection, and the matched latent deterministic control each update405,824 parameters in this executable configuration. No-image conditioning updates378,112; jointly fine-tuning the autoencoder updates527,233. The additive implementation combines the existing two affine projections, matching stored parameter count but using a redundant additive parameterisation. Equal parameter budgets do not imply equal effective capacity, activation memory, or computational cost.',
        '', 'These experiments quantify component effects within the retained executable configuration. They are separate from the original author-supplied positive-score table and do not reproduce or validate its score-producing checkpoints or architectural parameter budget. The design is not claimed to be universally optimal, and the small fixed test sets and five training pairs limit inference. Full statistical results and per-image records accompany this response.',
        '', 'Manuscript status: the new ablation experiment has not yet been inserted into the manuscript. This file is a result-supported response draft, not a claim that manuscript revisions have already been made.']
text='\n'.join(lines).replace('for100epochs','for 100 epochs').replace('producing105','producing 105').replace('at50','at 50').replace('All7,200','All 7,200').replace('across24','across 24').replace('crossed-bootstrap95%','crossed-bootstrap 95%').replace('with50,000','with 50,000').replace('update405,824','update 405,824').replace('updates378,112','updates 378,112').replace('updates527,233','updates 527,233')+'\n'
response=out/f'reviewer_2_comment_4_response_{tag}.txt';response.write_text(text,encoding='utf-8')
method=out/f'method_and_source_notes_{tag}.json'
method.write_text(json.dumps(dict(training_initialisation='Model initialisation job seed; identical initial states copied within each paired dataset/seed batch',training_rng='torch.manual_seed(training_seed+98765),same timestep/noise/dropout draws across variants;deterministic variant draws and discards timestep/noise to keep conditioning-dropout sequence aligned',inference_rng='Independent explicitly seeded CUDA generators for each sampling label;matched Gaussian tensor across modes',GT_usage='Training supervision and final metric calculation only;not provided to test-time denoiser',parameter_scope='Updated parameters in new executable implementation,not original manuscript architecture;frozen baseline haszero updates duringrefinement',statistics='Exploratory CIs are not simultaneous confidence intervals;Holm correction applies separately to24t tests and24exact signed-rank tests',original_data_policy='Original supplied tables remain unchanged and are not merged into controlled component tests',source_folder=str(out)),indent=2),encoding='utf-8')
copied=[]
for dataset in ['CHASE','DRIVE','HRF']:
    for seed in [42,123,2026,3000,3001]:
        origin=root/f'paired_reproduction/all_datasets_five_full_runs_2026-10-08_15-38-24/{dataset}/train_seed_{seed}'
        dest=out/dataset/f'seed_{seed}'
        for source,name in [(origin/'baseline.pt',f'frozen_baseline_{tag}.pt'),(origin/'autoencoder.pt',f'initial_autoencoder_{tag}.pt')]:
            target=dest/name;assert not target.exists();shutil.copy2(source,target)
            h=hashlib.sha256(target.read_bytes()).hexdigest();assert h==hashlib.sha256(source.read_bytes()).hexdigest()
            copied.append(dict(source=str(source),saved=str(target),sha256=h))
manifest=out/f'frozen_source_checkpoint_manifest_{tag}.json';manifest.write_text(json.dumps(copied,indent=2))
archive=out/f'component_ablation_tables_{run}.zip'
with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED) as z:
    for p in [response,method,manifest]:z.write(p,arcname=p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
tracking=root/'reviewer_comments/reviewer_2_comments.json';obj=json.loads(tracking.read_text(encoding='utf-8-sig'));c=next(c for c in obj['comments'] if c['number']==4)
c.update(status='All requested component ablations completed in a new controlled experiment:105trained models,7200verified records,three datasets,five matched training seeds. Original author-supplied experiment remains distinct. New ablation tables not yet inserted into manuscript.',updated_at=tag,experiment_directory=str(out),response_file=str(response),verified_completion=str(out/f'verified_completion_{run}.json'),manuscript_incorporated=False)
tracking.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');(tracking.parent/f'reviewer_2_comments_{tag}.json').write_bytes(tracking.read_bytes())
txt=tracking.with_suffix('.txt')
with txt.open('a',encoding='utf-8') as f:f.write(f'\n{tag}: Comment4 component study completed:105trained models,7200verified evaluation records,allthree datasets. User-supplied original tables distinct. Manuscript unchanged. Experiment:{out}\nResponse:{response}\n')
(tracking.parent/f'reviewer_2_comments_{tag}.txt').write_bytes(txt.read_bytes())
with (root/'AGENTS.md').open('a',encoding='utf-8') as f:f.write(f'\n\n# Reviewer2 Comment4 complete new controlled component study\nOn {tag}, all105newrefiners completed (7variants x5seeds x3datasets) and7200records verified against FP32 predictions/GT/FOV/confusion counts. Eight evaluated modes include targetwarm/full-noise. Mean/SD,24paired comparisons with separateHolm t/W andcrossedbootstrap50000 generated. Output:{out}. DataZIP:{archive}. Response:{response}. Frozen baseline/initialAEsource checkpoints copied byte-for-byte withmanifest. Original positiveuser tables andmanuscript unchanged. Newimplementation405824updatedreferenceparameters differs fromoriginalarchitecturebudget;do not claim recovery or validation oforiginal score-producing configuration.\n')
print(response);print(archive)
