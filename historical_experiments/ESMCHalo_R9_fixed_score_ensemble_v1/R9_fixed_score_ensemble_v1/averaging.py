"""Fixed matched-fold weight average. No optimizer and no parameter fitting."""
from pathlib import Path
import json
import numpy as np
import torch
import legacy_v205 as old

def merge_checkpoints(a,b,fold):
    for ck,arm in [(a,'H0'),(b,'Mixup')]:
        if ck.get('arm')!=arm or ck.get('outer_fold')!=fold or ck.get('canonical_mean_only') is not True or ck.get('lora') is not False:
            raise ValueError('Source checkpoint identity/architecture mismatch')
    for k in ['scaler_mean','scaler_scale']:
        aa=np.asarray(a[k]);bb=np.asarray(b[k])
        if aa.shape!=bb.shape or not np.array_equal(aa,bb):raise ValueError('Preprocessing differs: '+k)
        if not np.isfinite(aa).all():raise ValueError('Nonfinite scaler')
    if not (np.asarray(a['scaler_scale'])>0).all():raise ValueError('Invalid scale')
    sa=a['model_state_dict'];sb=b['model_state_dict']
    if sa.keys()!=sb.keys():raise ValueError('Parameter names differ')
    state={}
    for k,va in sa.items():
        vb=sb[k]
        if va.shape!=vb.shape or va.dtype!=vb.dtype or not va.is_floating_point():raise ValueError('Incompatible parameter '+k)
        if not bool(torch.isfinite(va).all() and torch.isfinite(vb).all()):raise ValueError('Nonfinite parameter '+k)
        state[k]=.5*va+.5*vb
    return dict(module='R8',arm='S0',outer_fold=fold,model_state_dict=state,scaler_mean=np.asarray(a['scaler_mean']).copy(),scaler_scale=np.asarray(a['scaler_scale']).copy(),canonical_mean_only=True,lora=False,training_variant='fixed_matched_fold_weight_average',parent_arms=['H0','Mixup'],parent_weights=[.5,.5],trained=False)

def standardized(x,ck):
    scaler=old.StandardScaler();scaler.mean_=np.asarray(ck['scaler_mean']);scaler.scale_=np.asarray(ck['scaler_scale']);scaler.var_=scaler.scale_**2;scaler.n_features_in_=x.shape[1]
    return scaler.transform(np.asarray(x,dtype=np.float32),copy=True).astype(np.float32)

def score(ck,x,device):
    model=old.Student(x.shape[1],.2).to(device);model.load_state_dict(ck['model_state_dict']);model.eval()
    return old.predict(model,standardized(x,ck),device,512)

def run_fold(arm,fold,x,master,protocol,device,out,root,h0,mix):
    if arm!='S0':raise ValueError('Unexpected averaging candidate')
    directory=out/'partial'/arm;directory.mkdir(parents=True,exist_ok=True)
    sources={}
    for name,entry in protocol['source_checkpoints'][str(fold)].items():
        path=root/entry['path']
        if old.sha256(path)!=entry['sha256']:raise ValueError('Source checkpoint hash mismatch '+str(path))
        sources[name]=torch.load(path,map_location='cpu',weights_only=False)
    ck=merge_checkpoints(sources['H0'],sources['Mixup'],fold)
    mask=master.fold==fold;ids=master.loc[mask,'clean_id'];xx=x[mask]
    errors={}
    for name,ref in [('H0',h0),('Mixup',mix)]:
        z,p=score(sources[name],xx,device);expected=ref.set_index('clean_id').loc[ids]
        err=float(np.max(np.abs(z-expected.student_raw_logit.to_numpy())))
        # Numeric tolerance is a reproducibility check, not a performance gate.
        if err>2e-5:raise ValueError(f'Source prediction reproduction failed {name} fold {fold}: {err}')
        errors[name]=err
    z,p=score(ck,xx,device)
    if not np.isfinite(z).all() or not np.isfinite(p).all():raise ValueError('Nonfinite averaged-model prediction')
    result=master.loc[mask,['clean_id','label','homology_group','fold','teacher_logit']].copy();result.insert(0,'arm',arm)
    result['student_raw_logit']=z;result['probability_uncalibrated']=p;result['prediction_at_0_5']=(p>=.5).astype(int)
    cp=directory/f'fold_{fold}.pt';tsv=directory/f'fold_{fold}.tsv';torch.save(ck,cp);result.to_csv(tsv,sep='\t',index=False)
    old.atomic_json(directory/f'fold_{fold}.json',dict(status='PASS',arm=arm,outer_fold=fold,training_performed=False,source_checkpoints=protocol['source_checkpoints'][str(fold)],source_raw_logit_max_abs_error=errors,checkpoint_sha256=old.sha256(cp),result_sha256=old.sha256(tsv)))
    print(f'S0 fold={fold} averaged and predicted; no training',flush=True)
    return result
