from pathlib import Path
import sys,json,hashlib
try:
 import pandas as pd,numpy as np
except ImportError as e:
 print(e);sys.exit(3)
try:
 from scipy.special import expit
 from sklearn.metrics import average_precision_score,roc_auc_score,matthews_corrcoef,confusion_matrix
except ImportError as e:
 print(e);sys.exit(3)
B=Path(__file__).resolve().parents[1];A=B/'reference_results';O=Path(sys.argv[1]);checks=[];recomputed=[]
def read(p):return pd.read_csv(p,sep='\t',float_precision='round_trip')
def metric(y,p,t=.5):
 y=np.asarray(y,int);p=np.asarray(p,float);h=p>=t;tn,fp,fn,tp=confusion_matrix(y,h,labels=[0,1]).ravel();q=np.clip(p,1e-15,1-1e-15)
 return dict(n=len(y),positive=int(y.sum()),ACC=float((y==h).mean()),MCC=matthews_corrcoef(y,h),Brier=float(np.mean((p-y)**2)),log_loss=float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q))),AP=average_precision_score(y,p),AUROC=roc_auc_score(y,p),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp))
def compare(name,row,y,p,t=.5):
 m=metric(y,p,t);recomputed.append(dict(check=name,**m));errs={k:abs(float(row[k])-v) for k,v in m.items() if k in row and pd.notna(row[k])};checks.append(dict(check=name,max_error=max(errs.values()),passed=max(errs.values())<1e-10))
for r in range(5,9):
 d=next(A.glob(f'R{r}_*RESULTS'));o=read(d/'oof_predictions.tsv');manifest=read(d/'development_manifest.tsv')
 for fn in ['development_metrics.tsv','fold_metrics.tsv','subset_metrics.tsv']:
  for i,row in read(d/fn).iterrows():
   z=o[o.arm==row.arm]
   if 'fold' in row:z=z[z.fold==row.fold]
   if 'subset' in row:z=z[z.clean_id.isin(manifest.loc[manifest[row['subset']].astype(bool),'clean_id'])]
   col='probability_uncalibrated' if row['mode']=='raw' else 'probability_crossfit_Platt'
   compare(f'R{r}/{fn}/{i}',row,z.label,z[col])
 for c in json.loads((d/'crossfit_calibrators.json').read_text()):
  z=o[(o.arm==c['arm'])&(o.fold==c['fold'])];err=float(np.max(abs(expit(c['coefficient']*z.student_raw_logit+c['intercept'])-z.probability_crossfit_Platt)));checks.append(dict(check=f'R{r}/Platt/{c["arm"]}/{c["fold"]}',max_error=err,passed=err<1e-12))
 for arm,z in o.groupby('arm'):
  checks.append(dict(check=f'R{r}/fold_integrity/{arm}',passed=bool(len(z)==10567 and z.clean_id.nunique()==10567 and z.groupby('homology_group').fold.nunique().max()==1)))
for r in [3,9]:
 d=next(A.glob(f'R{r}_*RESULTS'));d=d/'results' if r==3 else d;o=read(d/'record_predictions.tsv')
 for i,row in read(d/'metrics.tsv').iterrows():
  z=o[o.dataset==row.dataset]
  if row['subset']!='all':
   col=row['subset'].replace('historical_','')+'_independent'
   if col not in z:continue
   z=z[z[col].astype(bool)]
  system={'H0_frozen':'ESMCHalo','DeepSaltPro_raw':'DeepSaltPro'}.get(row.system,row.system)
  if system not in z: raise ValueError(system)
  compare(f'R{r}/metrics/{i}',row,z.label,z[system],row.get('threshold',.5))
 if r==9:
  for prefix in ['H0','Mixup']:
   err=float(abs(o[[f'{prefix}_fold_{i}_logit' for i in range(5)]].mean(axis=1)-o[f'{prefix}_mean_logit']).max());checks.append(dict(check=f'R9/{prefix}_mean',max_error=err,passed=err<1e-12))
  for name,p in [('E0_mean_logit',(o.H0_mean_logit+o.Mixup_mean_logit)/2),('E0_raw',expit(o.E0_mean_logit)),('E0_Platt',expit(1.7473426969146193*o.E0_mean_logit-.07031368943079835))]:
   err=float(abs(o[name]-p).max());checks.append(dict(check=f'R9/{name}',max_error=err,passed=err<1e-12))
  for dataset,z in o.groupby('dataset'):
   h=(z.ESMCHalo>=.5)==z.label;e=(z.E0_Platt>=.5)==z.label;print('R9 errors',dataset,'fixed',int((~h&e).sum()),'new',int((h&~e).sum()))
 # duplicates across datasets
 print('R',r,'records',len(o),'unique_sequence',o.sequence_sha256.nunique())
d=A/'R4_label_shift_v1_output_RESULTS';o=read(d/'development_predictions.tsv')
for i,row in read(d/'development_metrics.tsv').iterrows():
 z=o[(o.fold==row.fold)&(o.target_ratio==row.target_ratio)&(o.system==row.system)]
 for prefix,col in [('before_','probability'),('after_','adapted_probability')]:
  rr={k[len(prefix):]:v for k,v in row.items() if k.startswith(prefix)};compare(f'R4/development/{i}/{prefix}',rr,z.label,z[col])
o=read(d/'challenge_predictions_evaluated.tsv')
for i,row in read(d/'challenge_metrics.tsv').iterrows():
 z=o[o.dataset==row.dataset];col=row.system+('_adapted' if row['mode']=='label_shift_adapted' else '');compare(f'R4/challenge/{i}',row,z.label,z[col])

# R1 is a recomputation of recorded timings, not a new speed benchmark.
d=next((B/'historical_project').glob('R1_end_to_end_*'));summary=json.loads((d/'SUMMARY.json').read_text());v={}
for system in ['ESMCHalo','DeepSaltPro']:
 v[system]=[json.loads((d/f'repeat{i}_{system}/TIMING.json').read_text())['wall_seconds'] for i in [1,2,3]]
 for k,value in [('median_seconds',np.median(v[system])),('min_seconds',min(v[system])),('max_seconds',max(v[system])),('throughput_at_median',1024/np.median(v[system]))]:
  checks.append(dict(check=f'R1/{system}/{k}',passed=abs(value-summary[system][k])<1e-10))
checks.append(dict(check='R1/paired_ratio',passed=abs(np.median(np.array(v['DeepSaltPro'])/v['ESMCHalo'])-summary['median_paired_time_ratio'])<1e-10))
(O/'metric_checks.json').write_text(json.dumps(checks,indent=2,default=lambda x:x.item()));pd.DataFrame(recomputed).to_csv(O/'recomputed_metrics.tsv',sep='\t',index=False)
print('Checks',len(checks),'recomputed metric rows',len(recomputed),'failed',sum(not c['passed'] for c in checks))
sys.exit(1 if any(not c['passed'] for c in checks) else 0)
