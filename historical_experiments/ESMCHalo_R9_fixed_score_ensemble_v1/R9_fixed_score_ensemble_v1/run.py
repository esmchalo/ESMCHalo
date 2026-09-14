"""Fixed ten-head exploratory challenge evaluation. No training or model selection."""
import argparse,json,zipfile,traceback,fcntl,sys
from pathlib import Path
import pandas as pd,numpy as np,torch
from scipy.special import expit
import legacy_v205 as old
from averaging import score
from calibration_core import fit_platt,metrics
HERE=Path(__file__).resolve().parent

def restore(ref,unique):
 if not ref.record_id.is_unique or not unique.clean_id.is_unique:raise ValueError('Duplicate record or unique-sequence IDs')
 merged=ref.merge(unique,on='clean_id',validate='many_to_one',how='left')
 if len(merged)!=len(ref) or merged.E0_Platt.isna().any():raise ValueError('Missing prediction or restoration failure')
 return merged

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/home/lvfang/ESMC_halophile'));ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
 if out.exists():raise FileExistsError('Output exists; do not duplicate or overwrite this fixed evaluation')
 out.mkdir(parents=True);lock=(out/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);exitcode=0
 try:
  for line in (HERE/'SHA256SUMS').read_text().splitlines():
   h,n=line.split('  ',1)
   if old.sha256(HERE/n)!=h:raise ValueError('Package hash mismatch '+n)
  protocol=json.loads((HERE/'PROTOCOL.json').read_text());old.atomic_json(out/'PROTOCOL.json',protocol)
  old.atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='development_score_calibration',production_model='H0',exploratory=True))
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent environment fallback')
  torch.set_num_threads(4);device=torch.device('cuda:0');old.atomic_json(out/'RUNTIME.json',dict(torch=torch.__version__,python=sys.version,gpu=torch.cuda.get_device_name(0),training_performed=False))
  r5dir=args.root/protocol['reference_directory']
  for name,key in [('oof_predictions.tsv','R5_oof_sha256'),('PROTOCOL.json','R5_protocol_sha256')]:
   if old.sha256(r5dir/name)!=protocol[key]:raise ValueError('R5 reference changed '+name)
  oof=pd.read_csv(r5dir/'oof_predictions.tsv',sep='\t',float_precision='round_trip');h=oof[oof.arm=='H0'].sort_values('clean_id');m=oof[oof.arm=='Mixup'].sort_values('clean_id')
  if len(h)!=10567 or not h.clean_id.is_unique or list(h.clean_id)!=list(m.clean_id):raise ValueError('Development identity mismatch')
  for c in ['label','fold','homology_group']:
   if not np.array_equal(h[c],m[c]):raise ValueError('Development metadata mismatch '+c)
  z=.5*h.student_raw_logit.to_numpy()+.5*m.student_raw_logit.to_numpy();a,b=fit_platt(z,h.label.to_numpy());old.atomic_json(out/'E0_calibrator.json',dict(coefficient=float(a),intercept=float(b),records=10567,fit='development fixed logit midpoint OOF only',label_source='development',source_oof_sha256=protocol['R5_oof_sha256']))
  # Fit only on development; no challenge labels used in calibration or weights.
  expected=pd.read_csv(HERE/'inputs/R3_sequences.tsv',sep='\t').clean_id.tolist();fd=args.root/'R3_existing_challenge_v1_output/esmc';rows=pd.read_csv(fd/'rows.tsv',sep='\t',usecols=['clean_id']);matrix=np.load(fd/'embeddings.npy',mmap_mode='r',allow_pickle=False)
  if len(expected)!=1675 or len(set(expected))!=1675 or not rows.clean_id.is_unique or set(rows.clean_id)!=set(expected) or matrix.shape!=(1675,1152):raise ValueError('Challenge cache shape or ID mismatch')
  order=pd.Series(np.arange(len(rows)),index=rows.clean_id).loc[expected].to_numpy();x=np.asarray(matrix[order],dtype=np.float32)
  if not np.isfinite(x).all():raise ValueError('Nonfinite cached embeddings')
  old.atomic_json(out/'CACHE_BINDING.json',dict(rows_sha256=old.sha256(fd/'rows.tsv'),embeddings_sha256=old.sha256(fd/'embeddings.npy')))
  unique=pd.DataFrame({'clean_id':expected});hs=[];ms=[];sources=[]
  for f in range(5):
   for name,dest in [('H0',hs),('Mixup',ms)]:
    entry=protocol['source_checkpoints'][str(f)][name];path=args.root/entry['path']
    if old.sha256(path)!=entry['sha256']:raise ValueError('Source model hash mismatch '+str(path))
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('arm')!=name or ck.get('outer_fold')!=f or ck.get('canonical_mean_only') is not True or ck.get('lora') is not False:raise ValueError('Source identity mismatch')
    old.atomic_json(out/'STATUS.json',dict(status='RUNNING',stage='cached_embedding_head_inference',arm=name,fold=f,production_model='H0',exploratory=True));print(f'START {name} fold={f}',flush=True)
    zz,_=score(ck,x,device)
    if not np.isfinite(zz).all():raise ValueError('Nonfinite source score')
    dest.append(zz);unique[f'{name}_fold_{f}_logit']=zz;sources.append(entry);print(f'DONE {name} fold={f}',flush=True)
  hz=np.vstack(hs).mean(axis=0);mz=np.vstack(ms).mean(axis=0);ez=.5*hz+.5*mz
  unique['H0_mean_logit']=hz;unique['Mixup_mean_logit']=mz;unique['E0_mean_logit']=ez;unique['E0_raw']=expit(ez);unique['E0_Platt']=expit(a*ez+b)
  unique.to_csv(out/'unique_predictions_LABEL_FREE.tsv',sep='\t',index=False);old.atomic_json(out/'SOURCE_BINDING.json',sources)
  ref=pd.read_csv(HERE/'inputs/R3_reference_records.tsv',sep='\t');ev=restore(ref,unique)
  if len(ev)!=1680:raise ValueError('Expected 1680 original challenge records')
  hp=expit(1.65445192210991*ev.H0_mean_logit.to_numpy()-.059280024051954906);diff=float(np.max(abs(hp-ev.ESMCHalo.to_numpy())))
  if diff>2e-6:raise ValueError(f'H0 challenge reference mismatch: {diff}')
  old.atomic_json(out/'H0_REPRODUCTION.json',dict(max_probability_abs_error=diff,tolerance=2e-6,passed=True))
  ev.to_csv(out/'record_predictions.tsv',sep='\t',index=False);metrics_rows=[];deltas=[];transitions=[]
  for dataset,g in ev.groupby('dataset'):
   for subset,flag in [('all',None),('historical_strict40','strict40_independent'),('historical_strict25','strict25_independent')]:
    t=g if flag is None else g[g[flag].astype(str).str.lower()=='true'];computed={}
    for name,col in [('E0_raw','E0_raw'),('E0_Platt','E0_Platt'),('H0_frozen','ESMCHalo'),('DeepSaltPro_raw','DeepSaltPro'),('DeepSaltPro_R2_Platt','DeepSaltPro_R2_Platt')]:
     met=metrics(t.label.to_numpy(),t[col].to_numpy());computed[name]=met;metrics_rows.append(dict(dataset=dataset,subset=subset,system=name,**met))
    for baseline,col in [('H0_frozen','ESMCHalo'),('DeepSaltPro_raw','DeepSaltPro'),('DeepSaltPro_R2_Platt','DeepSaltPro_R2_Platt')]:
     delta={k:float(computed['E0_Platt'][k]-computed[baseline][k]) for k in ['ACC','MCC','Brier','log_loss','AP','AUROC']};deltas.append(dict(dataset=dataset,subset=subset,contrast='E0_Platt-'+baseline,**delta))
     ca=(t.E0_Platt>=.5)==t.label;cb=(t[col]>=.5)==t.label;transitions.append(dict(dataset=dataset,subset=subset,baseline=baseline,corrected=int((ca&~cb).sum()),introduced=int((~ca&cb).sum())))
  for name,data in [('metrics.tsv',metrics_rows),('paired_point_differences.tsv',deltas),('paired_error_transitions.tsv',transitions)]:pd.DataFrame(data).to_csv(out/name,sep='\t',index=False)
  old.atomic_json(out/'STATUS.json',dict(status='COMPLETED',exploratory=True,production_model='H0',automatic_replacement=False,development_original_gate_passed=False,challenge_evaluated=True,records=1680,unique_sequences=1675,heads=10,training_performed=False))
 except Exception:
  exitcode=1;error=traceback.format_exc();print(error,flush=True);old.atomic_json(out/'STATUS.json',dict(status='FAILED',traceback=error))
 target=out.with_name(out.name+'_RESULTS.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(out.rglob('*')):
   if f.is_file() and not f.name.startswith('.'):z.write(f,f.relative_to(out))
 print('RESULTS: '+str(target),flush=True);return exitcode
if __name__=='__main__':sys.exit(main())
