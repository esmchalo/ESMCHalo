"""One locked CPU-only label-shift feasibility study. No encoder/head training."""
import argparse,json,hashlib,traceback,zipfile,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from core import fit_platt,adapt,metrics,gate
HERE=Path(__file__).resolve().parent
H0='ESMCHalo_v2_optimization/V2_05_hard_negative_and_peft/V2_05A_hard_sample_weighting_ablation_v1/all_candidate_oof_predictions.tsv'
DSP='ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof/deepsaltpro_nested_oof_full_v1/deepsaltpro_oof_predictions.tsv'
HASHES={H0:'d8d60797edf7f1864120005258e6e52847e655728aaa263406831617f7fc7a86',DSP:'e3715f16f7f32fb0bd28f48250c5c0bf748f8666762ba52e2a5d853d508915b2'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def load(root,out):
    for name,h in HASHES.items():
        p=root/name
        if not p.is_file():raise FileNotFoundError('Missing existing server input: '+str(p))
        if sha(p)!=h:raise ValueError('Frozen input hash mismatch: '+str(p))
    s=pd.read_csv(root/H0,sep='\t');s=s[s.arm=='H0'].copy();t=pd.read_csv(root/DSP,sep='\t')
    if len(s)!=10567 or len(t)!=10567 or not s.clean_id.is_unique or not t.clean_id.is_unique or set(s.clean_id)!=set(t.clean_id):raise ValueError('OOF member mismatch')
    s=s.sort_values('clean_id').reset_index(drop=True);t=t.set_index('clean_id').loc[s.clean_id]
    for a,b in [('label','label'),('fold','oof_fold'),('homology_group','cluster_id')]:
        if not np.array_equal(s[a].astype(str).to_numpy(),t[b].astype(str).to_numpy()):raise ValueError('OOF alignment mismatch: '+a)
    if set(s.fold)!={0,1,2,3,4} or s.groupby('homology_group').fold.nunique().max()!=1:raise ValueError('Invalid fold/group mapping')
    if set(s.label)!={0,1}:raise ValueError('Invalid labels')
    d=s[['clean_id','label','fold','homology_group']].copy();d['ESMCHalo']=s.student_raw_logit.to_numpy();d['DeepSaltPro']=t.raw_logit.to_numpy()
    if not np.isfinite(d[['ESMCHalo','DeepSaltPro']]).all().all():raise ValueError('Invalid OOF logits')
    d.to_csv(out/'development_inputs.tsv',sep='\t',index=False);write(out/'INPUT_BINDING.json',HASHES);return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/home/lvfang/ESMC_halophile'));ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    exit_code=0
    try:
        # Verify locked protocol and package before reading development inputs.
        for line in (HERE/'SHA256SUMS').read_text().splitlines():
            h,n=line.split('  ',1)
            if sha(HERE/n)!=h:raise ValueError('Package changed: '+n)
        protocol=json.loads((HERE/'PROTOCOL.json').read_text());write(out/'PROTOCOL.json',protocol);write(out/'STATUS.json',{'status':'RUNNING','stage':'development'})
        d=load(args.root,out);systems=['ESMCHalo','DeepSaltPro'];rows=[];members=[];fold_parameters=[];preds=[]
        for f in range(5):
            train=d[d.fold!=f];valid=d[d.fold==f];ytrain=train.label.to_numpy();spec={}
            for name in systems:
                a,b=fit_platt(train[name].to_numpy(),ytrain)
                prior=float(expit(a*train[name].to_numpy()+b).mean())
                spec[name]=(a,b,prior);fold_parameters.append(dict(fold=f,system=name,coefficient=float(a),intercept=float(b),source_prior=prior,empirical_prior=float(ytrain.mean())))
            for k,positive in enumerate([150,250,350]):
                rng=np.random.default_rng(20260914+100*f+k)
                ix=np.concatenate([rng.choice(valid.index[valid.label==1],positive,replace=False),rng.choice(valid.index[valid.label==0],500-positive,replace=False)]);rng.shuffle(ix);batch=d.loc[ix]
                members.extend(dict(fold=f,target_ratio=positive/500,clean_id=v) for v in batch.clean_id)
                for name in systems:
                    a,b,prior=spec[name];p=expit(a*batch[name].to_numpy()+b)
                    q,info=adapt(p,prior)
                    before=metrics(batch.label,p);after=metrics(batch.label,q)
                    if abs(before['AP']-after['AP'])>1e-12 or abs(before['AUROC']-after['AUROC'])>1e-12:raise ValueError('Ranking changed')
                    row=dict(fold=f,target_ratio=positive/500,system=name,**info,**{'before_'+k:v for k,v in before.items()},**{'after_'+k:v for k,v in after.items()});rows.append(row)
                    preds.extend(dict(fold=f,target_ratio=positive/500,system=name,clean_id=cid,label=int(y),probability=float(v),adapted_probability=float(w)) for cid,y,v,w in zip(batch.clean_id,batch.label,p,q))
                    print(f'DEV fold={f} positive={positive}/500 {name} pi_hat={info["estimated_prior"]:.5f} done',flush=True)
        pd.DataFrame(rows).to_csv(out/'development_metrics.tsv',sep='\t',index=False);pd.DataFrame(members).to_csv(out/'simulation_members.tsv',sep='\t',index=False);pd.DataFrame(preds).to_csv(out/'development_predictions.tsv',sep='\t',index=False);write(out/'fold_calibrators.json',fold_parameters)
        gates={name:gate([r for r in rows if r['system']==name]) for name in systems};passed=all(v['passed'] for v in gates.values());write(out/'GATE.json',dict(systems=gates,passed=passed,rule='Both systems pass their 15-condition mean gate; no selective adaptation. Boundary estimate causes gate failure.'))
        if not passed:
            write(out/'STATUS.json',dict(status='COMPLETED',decision='STOP_DEVELOPMENT_GATE_NOT_PASSED',challenge_adaptation_run=False));print('STOP: development gate not passed; challenge files not read.',flush=True)
        else:
            # Label-free challenge input read only after the development gate.
            c=pd.read_csv(HERE/'inputs/challenge_scores.tsv',sep='\t');params=json.loads((HERE/'inputs/source_calibrators.json').read_text());adapted=[];details=[]
            for dataset,batch in c.groupby('dataset',sort=True):
                output=batch[['record_id','dataset']].copy()
                for name in systems:
                    a,b=params[name];source_prior=float(expit(a*d[name].to_numpy()+b).mean());p=batch[name].to_numpy();q,info=adapt(p,source_prior)
                    if info['boundary']:raise RuntimeError('Challenge estimate hit boundary: '+dataset+' '+name)
                    output[name]=p;output[name+'_adapted']=q;details.append(dict(dataset=dataset,system=name,**info));print('CHALLENGE '+dataset+' '+name+' predicted',flush=True)
                adapted.append(output)
            allp=pd.concat(adapted,ignore_index=True);allp.to_csv(out/'challenge_predictions_LABEL_FREE.tsv',sep='\t',index=False);write(out/'challenge_prior_estimates.json',details)
            # Predictions are saved before reading any challenge labels.
            labels=pd.read_csv(HERE/'inputs/challenge_labels.tsv',sep='\t');ev=allp.merge(labels,on=['record_id','dataset'],validate='one_to_one');mr=[]
            if len(ev)!=1680 or ev.label.isna().any():raise ValueError('Challenge evaluation mapping mismatch')
            for dataset,batch in ev.groupby('dataset'):
                for name in systems:
                    for mode in ['', '_adapted']:mr.append(dict(dataset=dataset,system=name,mode='source_calibrated' if mode=='' else 'label_shift_adapted',**metrics(batch.label,batch[name+mode])))
            ev.to_csv(out/'challenge_predictions_evaluated.tsv',sep='\t',index=False);pd.DataFrame(mr).to_csv(out/'challenge_metrics.tsv',sep='\t',index=False)
            write(out/'STATUS.json',dict(status='COMPLETED',decision='DEVELOPMENT_PASS_CHALLENGE_EVALUATED',challenge_adaptation_run=True,note='Retrospective exploratory batch adaptation; original frozen model unchanged.'))
    except Exception:
        exit_code=1;err=traceback.format_exc();print(err,flush=True);write(out/'STATUS.json',dict(status='FAILED',traceback=err))
    target=out.with_name(out.name+'_RESULTS.zip')
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    print('RESULTS: '+str(target),flush=True);return exit_code
if __name__=='__main__':sys.exit(main())
