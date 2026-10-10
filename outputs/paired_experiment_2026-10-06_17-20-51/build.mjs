import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const dir=path.dirname(fileURLToPath(import.meta.url));
const data=JSON.parse(await fs.readFile(path.join(dir,'data.json'),'utf8'));
const wb=Workbook.create();
const summary=wb.worksheets.add('Summary');
const per=wb.worksheets.add('Per image');
const raw=wb.worksheets.add('All seeds');
const col=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const keys=Object.keys(data.raw[0]);
const rawEnd= data.raw.length+1;
function baseStyle(sheet,lastRow,lastCol){
 sheet.showGridLines=false;
 const range=sheet.getRange(`A1:${lastCol}${lastRow}`);
 range.format.font={name:'Arial',size:10,color:'#172B4D'};
 range.format.rowHeight=21;
 range.format.columnWidth=15;
 range.format.verticalAlignment='center';
}
function header(sheet,range){
 const r=sheet.getRange(range);
 r.format.fill='#203864';r.format.font={name:'Arial',size:10,color:'#FFFFFF',bold:true};
 r.format.wrapText=true;r.format.rowHeight=44;r.format.horizontalAlignment='center';
}
baseStyle(raw,rawEnd,'Y');
const rawHeaders=keys.map(k=>k.replace('baseline_','Base ').replace('refined_','Ref ').replaceAll('_',' '));
raw.getRange(`A1:Y${rawEnd}`).values=[rawHeaders,...data.raw.map(r=>keys.map(k=>r[k]))];
header(raw,'A1:Y1');
for(let i=0;i<keys.length;i++){
 const k=keys[i];
 if(!['dataset','image','unit','pathology'].includes(k)){
  const r=raw.getRange(`${col(i)}2:${col(i)}${rawEnd}`);
  r.setNumberFormat(k==='seed'||k==='steps'||/_(tp|fp|fn|tn)$/.test(k)?'#,##0':'0.000000');
  r.format.horizontalAlignment='right';
 }
}
raw.getRange(`A2:B${rawEnd}`).format.columnWidth=17;
raw.getRange(`C2:D${rawEnd}`).format.columnWidth=12;
raw.freezePanes.freezeRows(1);raw.freezePanes.freezeColumns(2);
raw.tables.add(`A1:Y${rawEnd}`,true,'AllSeedResults');
const metricKeys=keys.filter(k=>k.startsWith('baseline_')||k.startsWith('refined_'));
const perKeys=['dataset','image','unit','pathology','steps','sampling seeds',...metricKeys,'delta'];
const unique=[...new Map(data.raw.map(r=>[`${r.dataset}/${r.image}/${r.steps}`,r])).values()];
const perEnd=unique.length+1;
baseStyle(per,perEnd,'Y');
per.getRange(`A1:Y${perEnd}`).values=[perKeys.map(k=>k.replace('baseline_','Base ').replace('refined_','Ref mean ').replaceAll('_',' ')),...unique.map(r=>perKeys.map((k,i)=>i<5?r[k]:i===5?20:null))];
const ri=k=>col(keys.indexOf(k));
for(let i=0;i<unique.length;i++){
 const row=i+2;
 for(let j=0;j<metricKeys.length;j++){
  const k=metricKeys[j];
  per.getRange(`${col(j+6)}${row}`).formulas=[[`=AVERAGEIFS('All seeds'!$${ri(k)}$2:$${ri(k)}$${rawEnd},'All seeds'!$A$2:$A$${rawEnd},$A${row},'All seeds'!$B$2:$B$${rawEnd},$B${row},'All seeds'!$F$2:$F$${rawEnd},$E${row})`]];
 }
 per.getRange(`Y${row}`).formulas=[[`=P${row}-G${row}`]];
}
header(per,'A1:Y1');
per.getRange(`G2:Y${perEnd}`).setNumberFormat('0.000000');
for(let j=0;j<metricKeys.length;j++){
 if(/_(tp|fp|fn|tn)$/.test(metricKeys[j]))per.getRange(`${col(j+6)}2:${col(j+6)}${perEnd}`).setNumberFormat(metricKeys[j].startsWith('baseline_')?'#,##0':'#,##0.00');
}
per.getRange(`A2:B${perEnd}`).format.columnWidth=17;
per.getRange(`C2:F${perEnd}`).format.columnWidth=12;
per.freezePanes.freezeRows(1);per.freezePanes.freezeColumns(2);
per.tables.add(`A1:Y${perEnd}`,true,'PerImageResults');
per.getRange(`Y2:Y${perEnd}`).conditionalFormats.add('cellIs',{operator:'lessThan',formula:0,format:{font:{color:'#9C0006'}}});
baseStyle(summary,27,'M');summary.tabColor='#203864';
summary.getRange('A2').values=[['Paired test experiment — CHASE-DB1 and HRF']];summary.getRange('A2').format.font={name:'Arial',size:14,bold:true};
summary.getRange('A3').values=[['New experiment. Baseline and refinement comparisons use the same images and scoring within each dataset.']];
summary.getRange('A3').format.font={name:'Arial',size:10,italic:true};
summary.getRange('A5:M5').values=[['Dataset','DDIM steps','Images','Noise seeds','Base Dice','Base CI low','Base CI high','Ref Dice','Ref CI low','Ref CI high','Ref − base','Delta CI low','Delta CI high']];header(summary,'A5:M5');
for(let i=0;i<data.stats.length;i++){
 const r=data.stats[i],row=i+6;
 summary.getRange(`A${row}:M${row}`).values=[[r.dataset,r.steps,r.n,20,null,r.ci[0][0],r.ci[1][0],null,r.ci[0][1],r.ci[1][1],null,r.ci[0][2],r.ci[1][2]]];
 for(const [dest,src] of [['E','G'],['H','P'],['K','Y']])summary.getRange(`${dest}${row}`).formulas=[[`=AVERAGEIFS('Per image'!$${src}$2:$${src}$${perEnd},'Per image'!$A$2:$A$${perEnd},$A${row},'Per image'!$E$2:$E$${perEnd},$B${row})`]];
}
summary.getRange('A6:A9').format.columnWidth=17;
summary.getRange('B6:D9').format.columnWidth=11;
summary.getRange('E6:M9').setNumberFormat('0.000000');
summary.getRange('K6:M9').conditionalFormats.add('cellIs',{operator:'lessThan',formula:0,format:{font:{color:'#9C0006'}}});
const notes=[
 'Delta = refinement Dice minus baseline Dice. Negative values indicate lower Dice after refinement.',
 'Confidence level: 95%. Percentile bootstrap: 50,000 resamples of seed-averaged per-image results.',
 'CHASE: four subjects; bootstrap retains both eyes of each subject. HRF: resampling within h / dr / g strata.',
 'Intervals are exploratory and conditional on one training run. HRF patient independence could not be verified.',
 'Baseline checkpoints were reused and reevaluated. CHASE used its saved refiner; HRF refiner was newly trained.',
 'These results do not reproduce the deleted manuscript experiments or establish their confidence intervals.',
 'Threshold 0.5; image size 512 × 512; FP32. CHASE: whole image. HRF: annotated field of view.',
 'Per image: refinement metrics and pixel counts are means over 20 noise seeds; mean counts may be fractional.',
 'All seeds: 800 complete paired records with Dice, IoU, sensitivity, specificity, accuracy and TP / FP / FN / TN.',
 'No AUC or latency was measured in this paired rerun; none is imputed.',
 'Source data: paired_rerun_CHASE_2026-10-06_17-08-15 and paired_rerun_HRF_2026-10-06_17-09-02.',
 'Sources: paired_metrics.csv and summary.json in each run; reported CI values are imported bootstrap results.'
];
notes.forEach((n,i)=>summary.getRange(`A${12+i}`).values=[[n]]);
raw.getRange(`A${rawEnd+3}`).values=[['Source: CHASE paired_metrics.csv — '+data.sources['CHASE-DB1']]];
raw.getRange(`A${rawEnd+4}`).values=[['Source: HRF paired_metrics.csv — '+data.sources.HRF]];
wb.recalculate();
const check=await wb.inspect({kind:'table',range:'Summary!A5:M9',include:'values,formulas',tableMaxRows:5,tableMaxCols:13,maxChars:2500});
console.log(check.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'formula error scan'});
console.log(errors.ndjson);
// Independently reconcile all derived image Dice means and headline results.
for(let i=0;i<unique.length;i++){
 const u=unique[i];const e=data.expected.find(r=>r.dataset===u.dataset&&r.image===u.image&&Number(r.steps)===u.steps);
 for(const [c,k] of [['G','baseline_dice'],['P','refined_dice'],['Y','delta']]){
  const v=per.getRange(`${c}${i+2}`).values[0][0];
  if(typeof v!=='number'||Math.abs(v-Number(e[k]))>1e-12)throw new Error(`Mismatch ${c}${i+2}: ${v} / ${e[k]}`);
 }
}
for(const [name,range] of [['Summary','A1:M24'],['Per image','A1:M8'],['Per image','N1:Y8'],['All seeds','A1:M8'],['All seeds','N1:Y8']]){
 const blob=await wb.render({sheetName:name,range,scale:1.4,format:'png'});
 await fs.writeFile(path.join(dir,`${name.replaceAll(' ','_')}_${range.split(':')[0]}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const file=path.join(dir,`paired_test_results_${data.stamp}.xlsx`);
await(await SpreadsheetFile.exportXlsx(wb)).save(file);
console.log('SAVED',file);
