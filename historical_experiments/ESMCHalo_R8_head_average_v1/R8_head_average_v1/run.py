"""R8 development-only selection followed by one retrospective challenge evaluation."""
import argparse,json,sys,zipfile,traceback,fcntl,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.special import expit
import legacy_v205 as old
import averaging as tr
from calibration_core import fit_platt,metrics
HERE=Path(__file__).resolve().parent
H0_REL='ESMCHalo_v2_optimization/V2_05_hard_negative_and_peft/V2_05A_hard_sample_weighting_ablation_v1'
H0_HASH='d8d60797edf7f1864120005258e6e52847e655728aaa263406831617f7fc7a86'

def summarize(oof,master,out):
    summary=[];fold_rows=[];subset_rows=[];result=[];params=[]
    ids_strict=set(master.loc[master.strict25,'clean_id']);ids_clean=set(master.loc[master.clean_core,'clean_id'])
    for arm,g in oof.groupby('arm',sort=True):
        g=g.sort_values('clean_id').copy();z=g.student_raw_logit.to_numpy();y=g.label.to_numpy();fold=g.fold.to_numpy();cal=np.empty(len(g))
        for f in range(5):
            mask=fold!=f;a,b=fit_platt(z[mask],y[mask]);cal[~mask]=expit(a*z[~mask]+b);params.append(dict(arm=arm,fold=f,coefficient=float(a),intercept=float(b)))
        g['probability_crossfit_Platt']=cal;result.append(g)
        for mode,p in [('raw',g.probability_uncalibrated.to_numpy()),('crossfit_Platt',cal)]:
            summary.append(dict(arm=arm,mode=mode,**metrics(y,p)))
            for f in range(5):fold_rows.append(dict(arm=arm,mode=mode,fold=f,**metrics(y[fold==f],p[fold==f])))
            for subset,ids in [('strict25',ids_strict),('clean_core',ids_clean)]:
                mask=g.clean_id.isin(ids).to_numpy();subset_rows.append(dict(arm=arm,mode=mode,subset=subset,**metrics(y[mask],p[mask])))
    overall=pd.DataFrame(summary);foldtab=pd.DataFrame(fold_rows);sub=pd.DataFrame(subset_rows);r=pd.concat(result)
    for n,d in [('development_metrics.tsv',overall),('fold_metrics.tsv',foldtab),('subset_metrics.tsv',sub),('oof_predictions.tsv',r)]:d.to_csv(out/n,sep='\t',index=False)
    old.atomic_json(out/'crossfit_calibrators.json',params)
    return overall,foldtab,sub,r

def select(overall,folds,sub):
    gates=[];ix=overall.set_index(['arm','mode']);sidx=sub.set_index(['arm','mode','subset']);fix=folds.set_index(['arm','mode','fold'])
    for arm in ['S0']:
        a=ix.loc[(arm,'crossfit_Platt')];b=ix.loc[('H0','crossfit_Platt')];d={k:float(a[k]-b[k]) for k in ['ACC','MCC','Brier','log_loss']}
        guards={}
        for mode in ['raw','crossfit_Platt']:
            for k in ['AP','AUROC']:guards[mode+'_'+k]=bool(ix.loc[(arm,mode),k]-ix.loc[('H0',mode),k]>=-.001)
        for subset in ['strict25','clean_core']:
            for k in ['AP','AUROC']:guards[subset+'_'+k]=bool(sidx.loc[(arm,'raw',subset),k]-sidx.loc[('H0','raw',subset),k]>=-.001)
        count=sum(fix.loc[(arm,'crossfit_Platt',f),'log_loss']<fix.loc[('H0','crossfit_Platt',f),'log_loss'] or fix.loc[(arm,'crossfit_Platt',f),'MCC']>fix.loc[('H0','crossfit_Platt',f),'MCC'] for f in range(5))
        passed=all(guards.values()) and d['ACC']>=0 and d['MCC']>=0 and d['Brier']<=0 and d['log_loss']<=0 and (d['log_loss']<=-.005 or d['MCC']>=.010) and count>=3
        gates.append(dict(arm=arm,passed=bool(passed),delta=d,guards=guards,improving_folds=int(count)))
    eligible=[g['arm'] for g in gates if g['passed']]
    chosen=sorted(eligible,key=lambda arm:(ix.loc[(arm,'crossfit_Platt'),'log_loss'],-ix.loc[(arm,'crossfit_Platt'),'MCC'],-ix.loc[(arm,'raw'),'AP'],['S0'].index(arm)))[0] if eligible else 'H0'
    return dict(selected_arm=chosen,eligible_arms=eligible,gates=gates,decision='NEW_CANDIDATE_SELECTED' if eligible else 'NO_CANDIDATE_SELECTED')

def evaluate_selected(arm,r,root,out,device):
    g=r[r.arm==arm];a,b=fit_platt(g.student_raw_logit.to_numpy(),g.label.to_numpy());old.atomic_json(out/'selected_calibrator.json',dict(coefficient=float(a),intercept=float(b),training_records=len(g),arm=arm))
    feature_dir=root/'R3_existing_challenge_v1_output/esmc';rows=pd.read_csv(feature_dir/'rows.tsv',sep='\t',usecols=['clean_id']);ids=pd.read_csv(HERE/'inputs/R3_sequences.tsv',sep='\t',usecols=['clean_id']).clean_id.tolist()
    if not rows.clean_id.is_unique or set(rows.clean_id)!=set(ids) or len(ids)!=1675:raise ValueError('R3 cached feature ID mismatch')
    matrix=np.load(feature_dir/'embeddings.npy',mmap_mode='r',allow_pickle=False)
    if matrix.shape!=(1675,1152):raise ValueError('R3 embedding shape mismatch')
    order=pd.Series(np.arange(len(rows)),index=rows.clean_id).loc[ids].to_numpy();x=np.asarray(matrix[order],dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Invalid cached features')
    old.atomic_json(out/'challenge_cache_binding.json',dict(rows_sha256=old.sha256(feature_dir/'rows.tsv'),matrix_sha256=old.sha256(feature_dir/'embeddings.npy'),matrix_shape=list(matrix.shape)))
    logits=[]
    for f in range(5):
        ck=torch.load(out/'partial'/arm/f'fold_{f}.pt',map_location='cpu',weights_only=False)
        if ck['arm']!=arm or ck['outer_fold']!=f:raise ValueError('Selected checkpoint mismatch')
        model=old.Student(1152,.2).to(device);model.load_state_dict(ck['model_state_dict']);model.eval()
        z=(x-np.asarray(ck['scaler_mean'],np.float32))/np.asarray(ck['scaler_scale'],np.float32)
        raw,_=old.predict(model,z,device,512);logits.append(raw);del model
    avg=np.vstack(logits).mean(axis=0);p=expit(a*avg+b)
    if not np.isfinite(p).all():raise ValueError('Invalid selected probabilities')
    table=pd.DataFrame(dict(clean_id=ids,mean_raw_logit=avg,probability_uncalibrated=expit(avg),probability_calibrated=p))
    for f,z in enumerate(logits):table[f'fold_{f}_raw_logit']=z
    table.to_csv(out/'selected_challenge_predictions_LABEL_FREE.tsv',sep='\t',index=False)
    # Original labels and comparator outputs enter only after candidate and predictions are fixed.
    ref=pd.read_csv(HERE/'inputs/R3_reference_records.tsv',sep='\t');ev=ref.merge(table,on='clean_id',validate='many_to_one',how='left')
    if len(ev)!=1680 or ev.probability_calibrated.isna().any() or not ev.record_id.is_unique:raise ValueError('Original record restoration failed')
    ev.to_csv(out/'challenge_predictions_evaluated.tsv',sep='\t',index=False);mr=[]
    for dataset,df in ev.groupby('dataset'):
        for subset,flag in [('all',None),('historical_strict40','strict40_independent'),('historical_strict25','strict25_independent')]:
            t=df if flag is None else df[df[flag].astype(str).str.lower()=='true']
            for name,col in [(arm+'_raw','probability_uncalibrated'),(arm+'_Platt','probability_calibrated'),('ESMCHalo_frozen','ESMCHalo'),('DeepSaltPro_raw','DeepSaltPro'),('DeepSaltPro_R2_Platt','DeepSaltPro_R2_Platt')]:mr.append(dict(dataset=dataset,subset=subset,system=name,**metrics(t.label,t[col].to_numpy())))
    pd.DataFrame(mr).to_csv(out/'challenge_metrics.tsv',sep='\t',index=False)

def write_contrasts(overall,folds,out):
    for table,name,groupcols in [(overall,'supervision_contrasts.tsv',['mode']),(folds,'fold_supervision_contrasts.tsv',['mode','fold'])]:
        rows=[]
        for key,g in table.groupby(groupcols):
            key=key if isinstance(key,tuple) else (key,)
            ix=g.set_index('arm');r=dict(zip(groupcols,key));r['contrast']='S0-H0'
            r.update({k:float(ix.loc['S0',k]-ix.loc['H0',k]) for k in ['ACC','MCC','Brier','log_loss','AP','AUROC']});rows.append(r)
        pd.DataFrame(rows).to_csv(out/name,sep='\t',index=False)

def teacher_diagnostics(r,out):
    rows=[];transitions=[]
    for arm,g in r.groupby('arm'):
        agree=(g.teacher_logit>=0)==(g.label>=.5)
        for name,mask in [('teacher_agrees',agree),('teacher_disagrees',~agree)]:
            t=g[mask]
            for mode,col in [('raw','probability_uncalibrated'),('crossfit_Platt','probability_crossfit_Platt')]:
                if t.label.nunique()!=2:raise ValueError('Diagnostic subset missing a class')
                rows.append(dict(arm=arm,subset=name,mode=mode,**metrics(t.label.to_numpy(),t[col].to_numpy())))
    a=r[r.arm=='S0'].set_index('clean_id');b=r[r.arm=='H0'].set_index('clean_id').loc[a.index]
    for mode,col in [('raw','probability_uncalibrated'),('crossfit_Platt','probability_crossfit_Platt')]:
        ca=(a[col]>=.5)==a.label;cb=(b[col]>=.5)==b.label;agree=(a.teacher_logit>=0)==a.label
        for name,mask in [('all',np.ones(len(a),dtype=bool)),('teacher_agrees',agree),('teacher_disagrees',~agree)]:
            transitions.append(dict(mode=mode,subset=name,n=int(np.sum(mask)),corrected=int((ca&~cb&mask).sum()),introduced=int((~ca&cb&mask).sum())))
    pd.DataFrame(rows).to_csv(out/'teacher_agreement_metrics.tsv',sep='\t',index=False)
    pd.DataFrame(transitions).to_csv(out/'paired_error_transitions.tsv',sep='\t',index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/home/lvfang/ESMC_halophile'));ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args();out=args.output
    if out.exists() and not args.resume:raise FileExistsError('Output already exists; no duplicate start. Use --resume only after examining a failed/interrupted run.')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);code=0
    try:
        for line in (HERE/'SHA256SUMS').read_text().splitlines():
            h,n=line.split('  ',1)
            if old.sha256(HERE/n)!=h:raise ValueError('Execution package changed: '+n)
        protocol=json.loads((HERE/'PROTOCOL.json').read_text())
        if (out/'PROTOCOL.json').exists() and json.loads((out/'PROTOCOL.json').read_text())!=protocol:raise ValueError('Resume protocol mismatch')
        if (out/'STATUS.json').exists() and json.loads((out/'STATUS.json').read_text()).get('status')=='COMPLETED':raise RuntimeError('Already completed; no rerun')
        old.atomic_json(out/'PROTOCOL.json',protocol)
        if (HERE/'runtime_tests.log').is_file():
            (out/'runtime_tests.log').write_bytes((HERE/'runtime_tests.log').read_bytes())
        old.atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='input_binding'))
        torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        device=torch.device('cuda:0')
        for f in ['rows.tsv','embeddings.npy']:
            if not (args.root/'R3_existing_challenge_v1_output/esmc'/f).is_file():raise FileNotFoundError('Missing existing R3 cache: '+f)
        old.atomic_json(out/'RUNTIME.json',dict(torch=torch.__version__,python=sys.version,gpu=torch.cuda.get_device_name(0)))
        access=[];x,master,strict,clean,flags,unused,binding=old.load_inputs(args,protocol,access);master['strict25']=strict;master['clean_core']=clean;master.to_csv(out/'development_manifest.tsv',sep='\t',index=False);old.atomic_json(out/'INPUT_BINDING.json',binding)
        hpath=args.root/H0_REL/'all_candidate_oof_predictions.tsv'
        if old.sha256(hpath)!=H0_HASH:raise ValueError('H0 OOF hash mismatch')
        h0=pd.read_csv(hpath,sep='\t');h0=h0[h0.arm=='H0'].copy();repro=json.loads((args.root/H0_REL/'H0_REPRODUCTION_AUDIT.json').read_text())
        if not repro.get('passed') or len(h0)!=10567 or not h0.clean_id.is_unique:raise ValueError('H0 reference failed')
        aligned=h0.set_index('clean_id').loc[master.clean_id]
        for col in ['label','fold','homology_group']:
            if not np.array_equal(aligned[col].astype(str).to_numpy(),master[col].astype(str).to_numpy()):raise ValueError('H0 '+col+' mismatch')
        np.testing.assert_allclose(expit(h0.student_raw_logit),h0.probability_uncalibrated,atol=2e-15,rtol=0)
        old.atomic_json(out/'H0_REFERENCE.json',dict(oof_sha256=H0_HASH,existing_reproduction_audit=repro,retrained=False,original_seed=protocol['seed'],fold_seed_offset=protocol['k2_seed_offset']))
        r5dir=args.root/protocol['reference_directory']
        if old.sha256(r5dir/'oof_predictions.tsv')!=protocol['R5_oof_sha256'] or old.sha256(r5dir/'PROTOCOL.json')!=protocol['R5_protocol_sha256']:raise ValueError('R5 reference hash mismatch')
        rp=json.loads((r5dir/'PROTOCOL.json').read_text())
        for key in ['seed','k2_seed_offset','student']:
            if rp[key]!=protocol[key]:raise ValueError('R5 configuration mismatch '+key)
        r5=pd.read_csv(r5dir/'oof_predictions.tsv',sep='\t',float_precision='round_trip')
        mix=r5[r5.arm=='Mixup'].set_index('clean_id').loc[master.clean_id]
        if len(mix)!=10567 or not mix.index.is_unique:raise ValueError('Mixup reference IDs invalid')
        for col in ['label','fold','homology_group']:
            if not np.array_equal(mix[col].astype(str).to_numpy(),master[col].astype(str).to_numpy()):raise ValueError('Mixup reference mismatch '+col)
        # Source parameters must use identical preprocessing and corresponding folds.
        frames=[h0]
        for arm in ['S0']:
            for f in range(5):
                old.atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='head_averaging_and_oof_prediction',arm=arm,fold=f));print(f'START {arm} fold={f}',flush=True)
                frames.append(tr.run_fold(arm,f,x,master,protocol,device,out,args.root,h0,mix.reset_index()));print(f'DONE {arm} fold={f}',flush=True)
        oof=pd.concat(frames,ignore_index=True)
        if len(oof)!=2*10567 or oof.duplicated(['arm','clean_id']).any():raise ValueError('OOF assembly mismatch')
        old.atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='development_calibration_selection'));overall,folds,sub,r=summarize(oof,master,out);write_contrasts(overall,folds,out);teacher_diagnostics(r,out);selection=select(overall,folds,sub);old.atomic_json(out/'SELECTION.json',selection);print('DEVELOPMENT SELECTION: '+selection['selected_arm'],flush=True)
        if selection['selected_arm']!='H0':
            old.atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='selected_challenge_evaluation'));evaluate_selected(selection['selected_arm'],r,args.root,out,device)
        old.atomic_json(out/'STATUS.json',dict(status='COMPLETED',decision=selection['decision'],selected_arm=selection['selected_arm'],challenge_evaluated=selection['selected_arm']!='H0',note='No candidate selected is a screen result, not a conclusion that all model improvement is exhausted.'))
    except Exception:
        code=1;error=traceback.format_exc();print(error,flush=True);old.atomic_json(out/'STATUS.json',dict(status='FAILED',traceback=error))
    target=out.with_name(out.name+'_RESULTS.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.name.startswith('.'):z.write(p,p.relative_to(out))
    print('RESULTS: '+str(target),flush=True);return code
if __name__=='__main__':sys.exit(main())
