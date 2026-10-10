import json,csv,re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parent.parent
tag=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
out=root/'revise/reviewer_2_question_3';out.mkdir(exist_ok=True)
ch=json.loads((root/'revise/new_spatial_refiner_2026-10-09_10-34-17/verified_results_2026-10-09_10-34-17.json').read_text());ch['dataset']='CHASE-DB1'
others=json.loads((root/'revise/new_spatial_refiner_DRIVE_HRF_2026-10-09_12-20-45/verified_results_2026-10-09_12-20-45.json').read_text())
results=[ch]+others
supplied={'CHASE-DB1':'0.8349996208320358','DRIVE':'0.8350002704911688','HRF':'0.7929995417119752'}
source=next((root/'revise/user_supplied_files').glob('*815c0784*.txt'))
rows=list(csv.DictReader(source.open(encoding='utf-8-sig',newline='')))
for r in results:
    dataset='CHASE' if r['dataset']=='CHASE-DB1' else r['dataset']
    vals=[float(x['refined_dice']) for x in rows if x['dataset']==dataset and x['steps']=='50']
    assert abs(sum(vals)/len(vals)-float(supplied[r['dataset']]))<1e-14
working=root/'template261006.tex';raw=working.read_bytes();text=raw.decode('utf-8-sig')
assert 'sec:spatial_refiner_context' not in text
section=r'''
\subsection{Additional Spatial Learned Refiner and Contextual Comparison}
\label{sec:spatial_refiner_context}

We conducted an additional deterministic refinement experiment using a spatial residual U-Net that receives the RGB image and coarse probability map as four input channels. The model uses encoder widths 16, 32, 64, and 128, three skip-connected decoder blocks, bilinear upsampling, GroupNorm, and SiLU activations, with 488,145 learned parameters. A zero-initialised output layer predicts an additive correction to the coarse logits, avoiding the mask-autoencoder reconstruction bottleneck. The objective combines binary cross-entropy and soft Dice loss with equal weights. Training used AdamW at $3\times10^{-4}$, weight decay $10^{-4}$, cosine scheduling over a maximum of 120 epochs, batch size four, gradient clipping at one, and aligned flips and right-angle rotations of RGB, coarse maps, and annotations. Checkpoint selection used validation mean image Dice with patience 25 and included the initial zero-residual model; test results were not used for selection.

Five refiners were independently trained per dataset using seeds 42, 123, 2026, 3000, and 3001, with frozen per-seed coarse predictors from the available executable reproduction. Evaluation used eight CHASE-DB1, twenty DRIVE, and twelve HRF test images, yielding 200 paired image--seed records. The coarse and deterministic predictions were evaluated on identical images, annotations, and regions with threshold 0.5; CHASE-DB1 used the whole image and DRIVE/HRF used FOV masks. Metrics and confusion counts were checked against saved FP32 predictions. The initial zero-residual checkpoint was selected in two CHASE-DB1 runs and in none of the DRIVE or HRF runs.

\begin{table}[htbp]
\centering
\caption{Contextual Dice summary of the additional spatial-refiner experiment and the supplied diffusion evaluation table. The coarse and spatial-refiner columns are paired within the new experiment; the diffusion column is from the separate author-supplied table at 50 transitions. This is not a controlled paired spatial-refiner--diffusion benchmark. Coarse and spatial-refiner means are rounded to six decimals; the diffusion means retain the supplied computed decimal values. The number of displayed digits does not imply corresponding measurement precision.}
\label{tab:spatial_refiner_context}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Dataset & Coarse LU-Net+RA & Spatial deterministic refiner & Supplied diffusion, 50 steps \\
\midrule
'''
for r in results:
    section+=f"{r['dataset']} & {r['baseline_mean']:.6f} & {r['refined_mean']:.6f} & {supplied[r['dataset']]}"+r' \\'+'\n'
section+=r'''\bottomrule
\end{tabular}}
\end{table}

'''
for r in results:
    lo,hi=r['crossed_bootstrap_95CI']
    section+=f"On {r['dataset']}, spatial-refiner mean Dice was {r['refined_mean']:.9f} (between-training-seed SD {r['refined_SD']:.9f}), compared with coarse Dice {r['baseline_mean']:.9f} (SD {r['baseline_SD']:.9f}). The paired mean change was ${r['mean_delta']:+.9f}$, with an exploratory crossed-bootstrap 95\\% interval $[{lo:+.9f}, {hi:+.9f}]$. "
section+=r'''

Intervals used 50,000 resamples of training seeds and test units, retaining both CHASE-DB1 eyes within each of four subjects, using image-level resampling for DRIVE and pathology-stratified image resampling for HRF. Patient independence for DRIVE and HRF was not established. Two-sided paired training-seed $t$ tests gave nominal $p$ values 0.219939, 0.030057, and 0.013329 for CHASE-DB1, DRIVE, and HRF, respectively; exact signed-rank $p$ values were 0.25, 0.0625, and 0.0625. For the two newly added DRIVE/HRF comparisons, Holm-adjusted $t$ values were 0.030057 and 0.026657, while both adjusted signed-rank values were 0.125. Small test cohorts, five training pairs, and different inference assumptions constrain interpretation. These statistics concern coarse-to-spatial-refiner changes only.

The supplied diffusion means are numerically higher than the spatial-refiner means in Table~\ref{tab:spatial_refiner_context}. However, matching original diffusion checkpoint and prediction records were unavailable, so common score-producing checkpoints and evaluation provenance across these two sources were not established. Consequently, their differences are descriptive and do not isolate a diffusion-specific benefit or establish superiority over deterministic refinement. A matched diffusion--spatial-refiner experiment remains necessary. Same-protocol numerical comparisons with HiDiffSeg, RefineCatDiff, and Li et al. also remain outstanding; architectural differences alone do not show that reproduction is impossible.

'''
anchor=r'\section{Discussion}';assert text.count(anchor)==1;text=text.replace(anchor,section+anchor)
old='A deterministic learned-refiner control and target-diffusion ablation would be required to isolate the benefit of the diffusion formulation.'
assert old in text
text=text.replace(old,'An additional spatial learned-refiner experiment is reported in Section~\\ref{sec:spatial_refiner_context}; a matched comparison to the original diffusion configuration and a target-diffusion ablation remain required to isolate the benefit of the diffusion formulation.')
old='Other component ablations and a deterministic learned-refiner control remain absent.';assert old in text
text=text.replace(old,'Other component ablations remain absent. The additional spatial deterministic refiner in Section~\\ref{sec:spatial_refiner_context} is evaluated in a separate experiment, and its contextual comparison with the supplied diffusion scores does not establish a diffusion-specific benefit.')
labels=re.findall(r'\\label\{([^}]+)\}',text);assert len(labels)==len(set(labels))
for env in ['table','tabular','equation']:assert text.count('\\begin{'+env+'}')==text.count('\\end{'+env+'}')
backup=root/f'template261006_before_spatial_context_{tag}.tex';version=root/f'template261006_{tag}.tex'
assert not backup.exists() and not version.exists();backup.write_bytes(raw)
newline='\r\n' if b'\r\n' in raw else '\n';updated=text.replace('\r\n','\n').replace('\n',newline).encode('utf-8')
version.write_bytes(updated);working.write_bytes(updated)
response='''Thank you for this comment. We added an additional deterministic learned-refiner experiment on all three datasets. The spatial residual U-Net receives RGB images and coarse probability maps, uses488,145 learned parameters, and is trained with equally weighted BCE and Dice losses. Five refiners were independently trained per dataset with validation-based checkpoint selection. We provide200 paired per-image baseline/refiner records, between-training-seed SDs, paired tests, and exploratory confidence intervals.

The following contextual summary has been added to the manuscript:
Dataset | Paired coarse baseline | Spatial deterministic refiner | Supplied diffusion at50steps
'''
for r in results:response+=f"{r['dataset']} | {r['baseline_mean']:.6f} | {r['refined_mean']:.6f} | {supplied[r['dataset']]}\n"
response+='''
The supplied diffusion means are numerically higher on all three datasets. However, the deterministic-refiner experiment and supplied diffusion table have distinct experimental provenance. The coarse/refiner pairs are matched within the new experiment, but a matching original diffusion checkpoint and prediction record was unavailable. We therefore present the table as contextual, not as a controlled paired comparison or evidence that the gain is specific to diffusion.

Same-protocol evaluations of HiDiffSeg, RefineCatDiff and Li et al. remain outstanding. We do not treat architectural differences as proof that reproduction is impossible. The new deterministic experiment adds an informative comparator, while a fully matched diffusion/comparator benchmark is still needed to isolate the diffusion formulation's benefit. These limitations are stated explicitly in the revised manuscript.
'''
responsepath=out/f'reviewer_2_comment_3_response_{tag}.txt';responsepath.write_text(response,encoding='utf-8')
tracking=root/'reviewer_comments/reviewer_2_comments.json';obj=json.loads(tracking.read_text(encoding='utf-8-sig'));c=next(c for c in obj['comments'] if c['number']==3)
c.update(status='Additional spatial-refiner experiment and contextual mixed-source table incorporated.200verified coarse/refiner pairs;original diffusion pairing and named related-method benchmark remain outstanding.',current_status='Partially addressed with separately identified spatial-refiner experiment;not fully resolved.',manuscript_incorporated=True,manuscript_version=str(version),response_file=str(responsepath),updated_at=tag)
tracking.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');(tracking.parent/f'reviewer_2_comments_{tag}.json').write_bytes(tracking.read_bytes())
txt=tracking.with_suffix('.txt')
with txt.open('a',encoding='utf-8') as f:f.write(f'\n{tag}: Author requested spatial-refiner/supplied-diffusion contextual table incorporated. Source distinction explicit;comment3 partially addressed,not fully resolved. Version:{version}\nResponse:{responsepath}\n')
(tracking.parent/f'reviewer_2_comments_{tag}.txt').write_bytes(txt.read_bytes())
with (root/'AGENTS.md').open('a',encoding='utf-8') as f:f.write(f'\n\n# Contextual spatial-refiner/supplied-diffusion table authorized\nOn {tag}, user explicitly requested adding the combined table and drafting reviewer2 comment3 response. Manuscript:{version};latestalias updated. Original author diffusion means retained full decimals;new baseline/refiner6decimal contextual columns;source distinction explicit. Additional method,training,SD,paired baseline-refiner statistics added;no paired refiner-originaldiffusion superiority claim. Named-method benchmarks remain absent. Earlieruser tables unchanged. Response:{responsepath}. PDF not compiled.\n')
(out/f'incorporation_{tag}.json').write_text(json.dumps(dict(version=str(version),backup=str(backup),response=str(responsepath),source_user_table=str(source),pdf_compiled=False),indent=2))
print(version);print(responsepath)
