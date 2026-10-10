"""Insert the accepted author-supplied Dice-weight table; calculate seed-level CIs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,re
import numpy as np
from scipy.stats import t
root=Path(__file__).resolve().parents[1]
values={0:[.834067346,.834175947,.833992321],.25:[.834366303,.834423565,.834239561],.5:[.835,.835100727,.834863953],1:[.835967600,.835835543,.836024543]}
stats={}
for weight,arr in values.items():
 arr=np.array(arr);delta=arr-np.array(values[0]);mean=delta.mean();radius=t.ppf(.975,2)*delta.std(ddof=1)/np.sqrt(3)
 stats[str(weight)]=dict(mean_dice=float(arr.mean()),seed_means=arr.tolist(),difference_vs_zero=float(mean),paired_training_seed_t_95ci=[float(mean-radius),float(mean+radius)])
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
path=root/'template261006.tex';old=path.read_text(encoding='utf8')
assert '\\label{sec:dice_weight_ablation}' not in old
seed_rows='\n'.join(f"{weight:g} & "+' & '.join(f'{v:.9f}' for v in values[weight])+f" & {stats[str(weight)]['mean_dice']:.9f} \\\\" for weight in values)
ci_rows='\n'.join(f"{weight:g} & {stats[str(weight)]['difference_vs_zero']:.9f} & $[{stats[str(weight)]['paired_training_seed_t_95ci'][0]:.9f},\\,{stats[str(weight)]['paired_training_seed_t_95ci'][1]:.9f}]$ \\\\" for weight in (.25,.5,1))
section=r'''
\subsection{Contribution of the Dice Supervision Term}
\label{sec:dice_weight_ablation}

We evaluated $\lambda_{\mathrm{dice}}\in\{0,0.25,0.5,1\}$ on CHASE-DB1 at 50 DDIM transitions, holding the remaining settings fixed within the ablation. Each weight was evaluated using three training seeds (42, 123, and 2026) and five sampling seeds per trained model on eight test images. Each training-seed mean averages Dice over the eight images and five sampling realisations, and the overall mean averages the three training-seed means. These 120 predictions per weight do not represent 120 independent test images. Table~\ref{tab:dice_weight_seed_means} reports this supplementary multi-seed experiment separately from the original single-run dataset results.

\begin{table}[htbp]
\centering
\caption{Dice-weight ablation on CHASE-DB1 at 50 DDIM transitions. Each seed column gives a mean Dice over the same test images and sampling realisations.}
\label{tab:dice_weight_seed_means}
\scriptsize
\begin{tabular}{rrrrr}
\toprule
$\lambda_{\mathrm{dice}}$ & Seed 42 & Seed 123 & Seed 2026 & Overall mean \\
\midrule
SEED_ROWS
\bottomrule
\end{tabular}
\end{table}

For each nonzero weight, we formed paired differences $d_s=D_{\lambda,s}-D_{0,s}$ between corresponding training-seed means. An exploratory 95\% interval was calculated as $\overline{d}\pm t_{0.975,2}s_d/\sqrt{3}$, where $s_d$ is the sample standard deviation of the three differences. This paired $t$ interval assumes approximately normal independent differences across training runs. With only three training seeds, that assumption cannot be assessed reliably. The intervals condition on the fixed test images and the five sampling realisations; they do not quantify uncertainty from resampling test images or subjects, and are not subject-level confidence intervals. Secondary comparisons are exploratory and have not been adjusted for multiple comparisons.

\begin{table}[htbp]
\centering
\caption{Paired Dice differences relative to $\lambda_{\mathrm{dice}}=0$, with exploratory 95\% $t$ intervals across three training-seed means on a fixed test set.}
\label{tab:dice_weight_seed_intervals}
\small
\begin{tabular}{rrl}
\toprule
$\lambda_{\mathrm{dice}}$ & Mean difference & 95\% interval \\
\midrule
CI_ROWS
\bottomrule
\end{tabular}
\end{table}

At $\lambda_{\mathrm{dice}}=0.5$, mean Dice increased by 0.000909689 (approximately 0.0910 percentage points) relative to the denoising-only refiner objective, with an exploratory seed-level interval of $[0.000827234,\,0.000992143]$. This is a small observed contribution from the additional Dice term on the fixed test set. Setting $\lambda_{\mathrm{dice}}=0$ removes direct decoded-mask Dice supervision from refiner training; it does not remove the segmentation supervision previously used for the shared backbone and mask autoencoder. The highest mean in this ablation was obtained with $\lambda_{\mathrm{dice}}=1$, so these results do not establish that the original weight 0.5 is optimal. This ablation isolates the loss-term comparison; it does not independently establish an improvement over the coarse baseline or quantify uncertainty across patients.

'''.replace('SEED_ROWS',seed_rows).replace('CI_ROWS',ci_rows)
assert sorted(re.findall(r'\\begin\{([^}]+)\}',section))==sorted(re.findall(r'\\end\{([^}]+)\}',section))
marker='\\subsection{DDIM Transition-Count Analysis}'
if marker not in old:
 marker='\\subsection{Exploratory Repeated-Sampling Comparison on a New Configuration}'
assert old.count(marker)==1
new=old.replace(marker,section+marker)
phrase='In the implementation evaluated here, $\\lambda_{\\text{dice}}=0.5$; no separate Sobel edge-loss term is applied.'
assert old.count(phrase)==1
new=new.replace(phrase,phrase+' A supplementary comparison with weights 0, 0.25, and 1 is reported in Section~\\ref{sec:dice_weight_ablation}.')
phrase='The study reports neither component ablations nor a deterministic learned-refiner control.'
assert new.count(phrase)==1
new=new.replace(phrase,'The Dice-weight ablation in Section~\\ref{sec:dice_weight_ablation} addresses the additional segmentation-loss term, but uses only three training seeds on a fixed test set; its exploratory intervals do not account for patient-level uncertainty. Other component ablations and a deterministic learned-refiner control remain absent.')
# Keep all existing tables byte-identical within their environments.
for label in re.findall(r'\\label\{(tab:[^}]+)\}',old):
 def extract(text):
  idx=text.index('\\label{'+label+'}');a=text.rfind('\\begin{table}',0,idx);b=text.index('\\end{table}',idx)+len('\\end{table}');return text[a:b]
 assert extract(old)==extract(new),label
backup=root/f'template261006_before_dice_ablation_{stamp}.tex';backup.write_text(old,encoding='utf8')
path.write_text(new,encoding='utf8');version=root/f'template261006_{stamp}.tex';version.write_text(new,encoding='utf8')
audit=root/f'reported_Dice_weight_statistics_{stamp}.json'
audit.write_text(json.dumps(dict(source='Author-supplied training-seed mean table accepted in conversation; not substituted into executable experiment outputs',training_seeds=[42,123,2026],method='Paired t interval, n=3, df=2, fixed test set; no subject resampling',weights=stats),indent=2),encoding='utf8')
reply='''Thank you for this comment. We have added a Dice-weight ablation in the subsection "Contribution of the Dice Supervision Term", including separate tables of training-seed means and paired seed-level intervals. On CHASE-DB1, we compared lambda_Dice = 0, 0.25, 0.5, and 1 under otherwise identical settings, with three training seeds and five sampling seeds per trained model at 50 DDIM transitions. The mean Dice scores were 0.834079, 0.834343, 0.834988, and 0.835943, respectively.

Relative to the denoising-only refiner objective (lambda_Dice = 0), lambda_Dice = 0.5 increased mean Dice by 0.000910. An exploratory paired t-based 95% interval across the three training-seed means was [0.000827, 0.000992]. This interval describes variability between training seeds on the fixed test set and conditions on the sampling realisations; it does not account for uncertainty across test images or subjects. With only three training seeds, the t-based assumptions cannot be assessed reliably. Secondary comparisons are exploratory and not adjusted for multiple comparisons.

The results indicate a small observed contribution from additional Dice supervision. However, lambda_Dice = 1 achieved the highest mean Dice, so the ablation does not establish that 0.5 is optimal. Setting lambda_Dice = 0 removes the direct Dice loss from refiner training, but does not remove the segmentation supervision previously used to train the shared backbone and autoencoder. We have clarified these points in the manuscript and updated the limitations accordingly. The original dataset result tables remain unchanged.
'''
response=root/f'reviewer_Dice_term_response_{stamp}.txt';response.write_text(reply,encoding='utf8')
print(json.dumps(dict(version=str(version),response=str(response),statistics=str(audit),existing_tables='all unchanged'),indent=2))
