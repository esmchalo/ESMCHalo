"""Evaluate fixed existing challenges, without fitting or threshold selection."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,matthews_corrcoef

HERE=Path(__file__).resolve().parent

def restore(records,predictions):
    if not predictions.clean_id.is_unique:
        raise ValueError('Duplicate prediction IDs')
    if set(predictions.clean_id)!=set(records.clean_id):
        raise ValueError('Prediction IDs do not exactly match unique input IDs')
    p=predictions.probability.to_numpy(float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probability')
    return records.clean_id.map(predictions.set_index('clean_id').probability).to_numpy(float)

def metrics(y,p,threshold=.5):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    q=np.clip(p,1e-15,1-1e-15)
    bins=np.minimum((p*10).astype(int),9)
    ece=sum(np.mean(bins==i)*abs(np.mean(p[bins==i])-np.mean(y[bins==i])) for i in range(10) if np.any(bins==i))
    both=len(np.unique(y))==2
    return dict(n=len(y),positive=int(y.sum()),AP=float(average_precision_score(y,p)) if both else float('nan'),
        AUROC=float(roc_auc_score(y,p)) if both else float('nan'),ACC=float(np.mean(pred==y)),
        MCC=float(matthews_corrcoef(y,pred)),sensitivity=float(tp/(tp+fn)) if tp+fn else float('nan'),
        specificity=float(tn/(tn+fp)) if tn+fp else float('nan'),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
        Brier=float(np.mean((p-y)**2)),log_loss=float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))),ECE=float(ece))

def bootstrap(frame,reference,repeats=5000,seed=20260913):
    rng=np.random.default_rng(seed)
    groups=[np.asarray(v) for v in frame.reset_index(drop=True).groupby('sequence_sha256',sort=True).indices.values()]
    keys=['AP','AUROC','ACC','MCC','Brier','log_loss','ECE']
    y=frame.label.to_numpy(int);a=frame.ESMCHalo.to_numpy(float);b=frame[reference].to_numpy(float)
    values=[];invalid=0
    for _ in range(repeats):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(y[ix]))!=2:
            invalid+=1;continue
        ma=metrics(y[ix],a[ix]);mb=metrics(y[ix],b[ix]);values.append([ma[k]-mb[k] for k in keys])
    if not values: raise ValueError('No valid bootstrap replicates')
    ci=np.quantile(values,[.025,.975],axis=0);ma=metrics(y,a);mb=metrics(y,b)
    return [dict(comparator=reference,metric=k,difference=ma[k]-mb[k],lower=float(ci[0,j]),upper=float(ci[1,j]),
        groups=len(groups),repeats=repeats,valid_repeats=len(values),invalid_repeats=invalid,seed=seed) for j,k in enumerate(keys)]

def evaluate(work,repeats=5000):
    records=pd.read_csv(HERE/'inputs/records.tsv',sep='\t',dtype={'record_id':str,'clean_id':str})
    for name in ['ESMCHalo','DeepSaltPro']:
        records[name]=restore(records,pd.read_csv(work/(name+'_predictions.tsv'),sep='\t'))
    cal=json.loads((HERE/'inputs/R2_analysis.json').read_text())
    records['DeepSaltPro_R2_Platt']=expit(cal['coefficient']*logit(records.DeepSaltPro.to_numpy(float))+cal['intercept'])
    if not np.isfinite(records.DeepSaltPro_R2_Platt).all(): raise ValueError('Invalid R2 transform')
    result=work/'results';result.mkdir(exist_ok=False)
    for name in ['ESMCHalo','DeepSaltPro','DeepSaltPro_R2_Platt']:
        records[name+'_prediction_at_0_5']=(records[name]>=.5).astype(int)
    records.to_csv(result/'record_predictions.tsv',sep='\t',index=False)
    all_metrics=[];intervals=[];mcnemar=[]
    for dataset,df in records.groupby('dataset',sort=True):
        print('Evaluating '+dataset,flush=True)
        for subset,flag in [('all',None),('historical_strict40','strict40_independent'),('historical_strict25','strict25_independent')]:
            part=df if flag is None else df[df[flag].astype(str).str.lower()=='true']
            for name in ['ESMCHalo','DeepSaltPro','DeepSaltPro_R2_Platt','final_v3_calibrated_probability']:
                threshold=.47 if name=='final_v3_calibrated_probability' else .5
                all_metrics.append(dict(dataset=dataset,subset=subset,system=name,threshold=threshold,**metrics(part.label,part[name],threshold)))
        for comparator in ['DeepSaltPro','DeepSaltPro_R2_Platt']:
            intervals.extend(dict(dataset=dataset,subset='all',**r) for r in bootstrap(df,comparator,repeats))
        ok_a=(df.ESMCHalo>=.5)==df.label;ok_b=(df.DeepSaltPro>=.5)==df.label
        b=int((ok_a&~ok_b).sum());c=int((~ok_a&ok_b).sum())
        mcnemar.append(dict(dataset=dataset,esmchalo_only_correct=b,deepsaltpro_only_correct=c,
             exact_two_sided_p=float(binomtest(b,b+c,.5).pvalue) if b+c else 1.0))
    for name,data in [('metrics.tsv',all_metrics),('paired_intervals.tsv',intervals),('mcnemar.tsv',mcnemar)]:
        pd.DataFrame(data).to_csv(result/name,sep='\t',index=False)
    (result/'analysis.json').write_text(json.dumps(dict(status='COMPLETED',records=len(records),unique_sequences=records.clean_id.nunique(),
        r2_coefficient=cal['coefficient'],r2_intercept=cal['intercept'],repeats=repeats,seed=20260913,ece_bins=10,
        limitations=['Existing public challenges previously evaluated with final-v3; not first project-wide blind evaluation.',
        'Historical homology flags are reused and not revalidated against current training records.',
        'Bootstrap groups exact sequences only; no taxonomic or homology independence claim.',
        'R2 Platt is retrospective sensitivity analysis, with parameters fitted previously; no challenge fitting.',
        'No causal distillation claim. final-v3 uses historical threshold 0.47.'],
        inference='ESMCHalo: mean of five raw logits then frozen Platt; DSP: mean of five probabilities.'),indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--repeats',type=int,default=5000)
    args=ap.parse_args();evaluate(args.work,args.repeats)
