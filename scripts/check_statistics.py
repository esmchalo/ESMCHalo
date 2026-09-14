#!/usr/bin/env python3
"""Replay historical result statistics, never execute the original reveal entry."""
from pathlib import Path
import sys,json,importlib.util
try:
 import numpy as np,pandas as pd
 from scipy.stats import binomtest
except ImportError as e:print(e);sys.exit(3)
B=Path(__file__).resolve().parents[1];O=Path(sys.argv[1]);H=B/'historical_project';checks=[]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def read(p):return pd.read_csv(p,sep='\t',float_precision='round_trip')
def equal(name,actual,reference,cols,tol=1e-9):
 assert len(actual)==len(reference),(name,len(actual),len(reference))
 for c in cols:
  err=float(np.max(np.abs(actual[c].to_numpy(float)-reference[c].to_numpy(float))))
  checks.append(dict(check=name+'/'+c,max_absolute_error=err,tolerance=tol,passed=bool(np.isfinite(err) and err<=tol)))
common=load('v207_metric_functions',next(H.glob('v2_07_execution_package/*/v2_07_common.py')))
d=H/'ESMCHalo_v2_optimization/V2_07_blind_evaluation/V2_07_one_time_blind_evaluation_v1';df=read(d/'blind_predictions_with_frozen_labels.tsv')
y=df.label.to_numpy(int);a=df.esmchalo_probability_calibrated.to_numpy(float);b=df.deepsaltpro_probability_mean.to_numpy(float)
ma=common.metric_dict(y,a,.5);mb=common.metric_dict(y,b,.5)
actual=pd.DataFrame([{'model':'ESMCHalo_v2',**ma},{'model':'DeepSaltPro',**mb}]);ref=read(d/'blind_metrics.tsv');equal('V207_metrics',actual,ref,[c for c in ref if c!='model']);actual.to_csv(O/'holdout_metrics_recomputed.tsv',sep='\t',index=False)
ok_a=(a>=.5)==y;ok_b=(b>=.5)==y;n1=int((ok_a&~ok_b).sum());n2=int((~ok_a&ok_b).sum());mc={'ESMCHalo_correct_DeepSaltPro_wrong':n1,'ESMCHalo_wrong_DeepSaltPro_correct':n2,'discordant':n1+n2,'exact_two_sided_p':float(binomtest(min(n1,n2),n1+n2,.5).pvalue) if n1+n2 else 1.}
r=json.loads((d/'mcnemar_exact.json').read_text());checks.append(dict(check='V207_McNemar',passed=all(abs(mc[k]-r[k])<1e-12 for k in r)));(O/'holdout_mcnemar_recomputed.json').write_text(json.dumps(mc,indent=2))
names=['ap','auroc','mcc','accuracy','balanced_accuracy','sensitivity','specificity','ppv','f1','brier','ece_10_bins','log_loss']
groups=df.homology_group.astype(str).to_numpy();unique=np.unique(groups);lookup={g:np.flatnonzero(groups==g) for g in unique};rng=np.random.default_rng(20260903);rows=[];attempts=0
while len(rows)<5000:
 attempts+=1
 if attempts>100000:raise RuntimeError('Too few valid bootstrap replicates')
 ix=np.concatenate([lookup[g] for g in rng.choice(unique,size=len(unique),replace=True)])
 if len(np.unique(y[ix]))!=2:continue
 aa=common.metric_dict(y[ix],a[ix],.5);bb=common.metric_dict(y[ix],b[ix],.5);rows.append([aa[k]-bb[k] for k in names])
 if len(rows)%1000==0:print('Holdout bootstrap',len(rows),'/5000',flush=True)
values=np.array(rows);summary=[]
for j,k in enumerate(names):
 v=values[:,j];summary.append(dict(metric=k,observed_delta_ESMCHalo_minus_DeepSaltPro=ma[k]-mb[k],bootstrap_mean=v.mean(),ci_2_5=np.quantile(v,.025),median=np.quantile(v,.5),ci_97_5=np.quantile(v,.975),fraction_gt_zero=np.mean(v>0)))
actual=pd.DataFrame(summary);ref=read(d/'paired_group_bootstrap_summary.tsv');assert actual.metric.tolist()==ref.metric.tolist();equal('V207_intervals',actual,ref,list(actual.columns[1:]));actual.to_csv(O/'holdout_intervals_recomputed.tsv',sep='\t',index=False)
# Existing challenge paired intervals: exact sequence grouping, not homology grouping.
r3=load('challenge_evaluation',H/'R3_existing_challenge_v1/evaluate_challenge.py');d=B/'reference_results/R3_existing_challenge_v1_output_RESULTS/results';frame=read(d/'record_predictions.tsv');allrows=[];mcrows=[]
for dataset,z in frame.groupby('dataset',sort=True):
 for comparator in ['DeepSaltPro','DeepSaltPro_R2_Platt']:
  print('Challenge bootstrap',dataset,comparator,'5000',flush=True)
  allrows.extend(dict(dataset=dataset,subset='all',**r) for r in r3.bootstrap(z,comparator,5000,20260913))
 ca=(z.ESMCHalo>=.5)==z.label;cb=(z.DeepSaltPro>=.5)==z.label;x=int((ca&~cb).sum());y2=int((~ca&cb).sum());mcrows.append(dict(dataset=dataset,esmchalo_only_correct=x,deepsaltpro_only_correct=y2,exact_two_sided_p=float(binomtest(x,x+y2,.5).pvalue) if x+y2 else 1.))
actual=pd.DataFrame(allrows);ref=read(d/'paired_intervals.tsv');keys=['dataset','subset','comparator','metric'];actual=actual.sort_values(keys).reset_index(drop=True);ref=ref.sort_values(keys).reset_index(drop=True);assert actual[keys].equals(ref[keys]);equal('R3_intervals',actual,ref,[c for c in actual if c not in keys]);actual.to_csv(O/'challenge_intervals_recomputed.tsv',sep='\t',index=False)
actual=pd.DataFrame(mcrows);ref=read(d/'mcnemar.tsv');equal('R3_McNemar',actual,ref,[c for c in ref if c!='dataset']);actual.to_csv(O/'challenge_mcnemar_recomputed.tsv',sep='\t',index=False)
(O/'statistics_checks.json').write_text(json.dumps(checks,indent=2));print('Statistics checks',len(checks),'failed',sum(not c['passed'] for c in checks));sys.exit(1 if any(not c['passed'] for c in checks) else 0)
