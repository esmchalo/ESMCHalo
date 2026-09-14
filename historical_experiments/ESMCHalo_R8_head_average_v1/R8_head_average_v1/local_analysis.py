"""Reproduce already-completed diagnostic only; no challenge tuning, no training."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from calibration_core import fit_platt,metrics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--r5',type=Path,required=True);ap.add_argument('--r3',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
 p=pd.read_csv(args.r5/'oof_predictions.tsv',sep='\t',float_precision='round_trip');a=p[p.arm=='H0'].sort_values('clean_id');b=p[p.arm=='Mixup'].sort_values('clean_id')
 if not a.clean_id.is_unique or len(a)!=10567 or list(a.clean_id)!=list(b.clean_id):raise ValueError('OOF identity mismatch')
 for c in ['label','fold','homology_group']:
  if not np.array_equal(a[c],b[c]):raise ValueError('OOF mapping mismatch '+c)
 y=a.label.to_numpy();f=a.fold.to_numpy();z=(a.student_raw_logit.to_numpy()+b.student_raw_logit.to_numpy())/2;cal=np.empty(len(y));params=[]
 for k in range(5):
  v=f==k;c,d=fit_platt(z[~v],y[~v]);cal[v]=expit(c*z[v]+d);params.append(dict(fold=k,coefficient=c,intercept=d))
 table=a[['clean_id','label','fold','homology_group']].copy();table['mean_raw_logit']=z;table['probability_raw']=expit(z);table['probability_crossfit_Platt']=cal;table.to_csv(out/'fixed_logit_midpoint_OOF.tsv',sep='\t',index=False)
 overall=[];folds=[]
 for name,raw,cp in [('H0',a.probability_uncalibrated.to_numpy(),a.probability_crossfit_Platt.to_numpy()),('logit_midpoint_diagnostic',expit(z),cal)]:
  for mode,q in [('raw',raw),('crossfit_Platt',cp)]:
   overall.append(dict(system=name,mode=mode,**metrics(y,q)))
   for k in range(5):v=f==k;folds.append(dict(system=name,mode=mode,fold=k,**metrics(y[v],q[v])))
 pd.DataFrame(overall).to_csv(out/'fixed_logit_midpoint_metrics.tsv',sep='\t',index=False);pd.DataFrame(folds).to_csv(out/'fixed_logit_midpoint_fold_metrics.tsv',sep='\t',index=False)
 base=metrics(y,a.probability_crossfit_Platt.to_numpy());cand=metrics(y,cal);delta={k:float(cand[k]-base[k]) for k in ['ACC','MCC','Brier','log_loss','AP','AUROC']}
 (out/'diagnostic_status.json').write_text(json.dumps(dict(status='COMPLETED_LOCAL_DIAGNOSTIC',training_performed=False,weights_searched=False,weight=.5,calibrators=params,delta=delta,original_minimum_gain_met=bool(delta['log_loss']<=-.005 or delta['MCC']>=.010),note='Prediction average is NOT weight average; no S0 performance claimed.'),indent=2)+'\n')
 r=pd.read_csv(args.r3/'results/record_predictions.tsv',sep='\t');seq=pd.read_csv(args.r3/'inputs/sequences.tsv',sep='\t');r=r.merge(seq[['clean_id','sequence','length']],on='clean_id',validate='many_to_one')
 for feature,aa in [('acidic_DE','DE'),('basic_KR','KR'),('hydrophobic_AVILMFWY','AVILMFWY')]:r[feature]=r.sequence.map(lambda x:sum(x.count(a) for a in aa)/len(x))
 rows=[];profiles=[]
 for dataset,g in r.groupby('dataset'):
  e=(g.ESMCHalo>=.5)==g.label;d=(g.DeepSaltPro>=.5)==g.label
  for name,mask in [('both_wrong',~e&~d),('ESMCHalo_only_wrong',~e&d),('DSP_only_wrong',e&~d),('both_correct',e&d)]:
   t=g[mask];rows.append(dict(dataset=dataset,category=name,n=len(t),negative=int((t.label==0).sum()),positive=int((t.label==1).sum())))
   for label in [0,1]:
    tt=t[t.label==label];profiles.append(dict(dataset=dataset,category=name,label=label,n=len(tt),**{c:float(tt[c].median()) if len(tt) else None for c in ['length','acidic_DE','basic_KR','hydrophobic_AVILMFWY']}))
 pd.DataFrame(rows).to_csv(out/'challenge_error_overlap.tsv',sep='\t',index=False);pd.DataFrame(profiles).to_csv(out/'challenge_sequence_profiles.tsv',sep='\t',index=False)
 print('Local diagnostics completed; no model training or challenge candidate evaluation')
if __name__=='__main__':main()
