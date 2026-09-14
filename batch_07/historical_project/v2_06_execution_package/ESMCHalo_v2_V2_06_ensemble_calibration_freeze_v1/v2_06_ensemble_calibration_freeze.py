#!/usr/bin/env python3
"""V2-06: freeze five K2/H0 students and development-only calibration."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, re, shutil, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

EXPECTED = {
    "folds":"68119ccd8a9c03244c2b23356b52bc9c20a4c1ba6319970917a8bca53863205d",
    "registry":"d02f0cb31d7b599d1de4780f61d3bda6b26390671dc42053ef1ba1c8dc2b0ee5",
    "v205_done":"279641e6b5488d762d286b50b3c3cffbdd874925b5db421139fb803a26fa57d8",
    "v205_selection":"f5a151eccf08fad3d98f868eedaa2e7d0673f950b9b022597b62d127f42ebe73",
    "v205_reproduction":"79543e0c70a3b71fa8d6caab5f9d788d23c25cdcdb60a7d4761874cb123db497",
    "v205_oof":"d8d60797edf7f1864120005258e6e52847e655728aaa263406831617f7fc7a86",
    "v205_binding":"46c1fcc0d67d5d4bdbe243f724df0d8eea36c2b8accc9cb4d9d19c3ce1a38614",
    "v205_access":"d5793cea32c9e69b5c7bb0f90e4bc80214ed3491abc7d9149c3bd21593ebdd81",
    "member_0":"57a12bba50e19bebc94cb839ba05639276b4b004892283efb867adf5c5480ae4",
    "member_1":"1e49d4f98d7ba8e2a567e9a7bf1298db43cbde20389d7b7243a41ce753c2c9bf",
    "member_2":"5685ca6f2ec06e4ac19c44cdbf8d13347d2ab00646a17a995b366e279ec3d023",
    "member_3":"c40651770ae2f689415cefdd516e7f2a74a60efd5c2f70b26a798c460f1606a2",
    "member_4":"6c290a6b7f5de70c5ff9052df2565486008f54056dccbb84d521216d545c524a",
}
FORBIDDEN=("blind","test200","external","challenge")
IDS=("clean_id","id","protein_id","sample_id","record_id")
FOLDS=("fold","outer_fold","fold_id","oof_fold")
GROUPS=("homology_group","strict40_group","cluster_id","group","mmseqs40_cluster")
STRICT25=("strict25","is_strict25","strict25_member","strict25_subset","strict25_eval","strict25_diagnostic")
TIERS=("audit_tier","quality_tier","tier","audit_category","audit_level","training_tier")


def now(): return datetime.now(timezone.utc).isoformat()
def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8");os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def guarded(path,role,log):
    p=path.expanduser().resolve(); hits=[x for x in FORBIDDEN if x in str(p).lower()]
    if hits: raise RuntimeError(f"Forbidden path for {role}: {hits}: {p}")
    if not p.is_file(): raise FileNotFoundError(f"Missing {role}: {p}")
    log.append({"role":role,"path":str(p),"opened_at_utc":now()});return p
def pick(df,names,role):
    low={str(c).lower():str(c) for c in df.columns}
    for n in names:
        if n.lower() in low:return low[n.lower()]
    raise ValueError(f"Cannot resolve {role}; columns={list(df.columns)}")
def binary(s,role):
    m={"0":0,"1":1,"false":0,"true":1,"no":0,"yes":1,"n":0,"y":1};out=[]
    for x in s:
        k=str(x).strip().lower()
        if k in m:out.append(m[k])
        else:
            try:v=int(float(k))
            except Exception as e:raise ValueError(f"Non-binary {role}: {x!r}") from e
            if v not in (0,1):raise ValueError(f"Non-binary {role}: {x!r}")
            out.append(v)
    return np.asarray(out,np.int64)
def sigmoid(x):return 1/(1+np.exp(-np.clip(np.asarray(x,float),-50,50)))
def ece(y,p,bins=15):
    y=np.asarray(y,int);p=np.asarray(p,float);edges=np.linspace(0,1,bins+1);total=0.0
    idx=np.minimum(np.digitize(p,edges[1:-1],right=False),bins-1)
    for b in range(bins):
        mask=idx==b
        if mask.any():total+=mask.mean()*abs(float(y[mask].mean())-float(p[mask].mean()))
    return float(total)
def prob_metrics(y,p,bins=15):
    y=np.asarray(y,int);p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    return {"n":len(y),"brier":float(brier_score_loss(y,p)),"ece":ece(y,p,bins),
            "log_loss":float(log_loss(y,p,labels=[0,1])),"auroc":float(roc_auc_score(y,p)),"ap":float(average_precision_score(y,p))}
def class_metrics(y,p,t):
    y=np.asarray(y,int);pred=np.asarray(p)>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"n":len(y),"threshold":float(t),"accuracy":float(accuracy_score(y,pred)),"mcc":float(matthews_corrcoef(y,pred)),
            "sensitivity":float(recall_score(y,pred,zero_division=0)),"specificity":float(tn/(tn+fp)),
            "ppv":float(precision_score(y,pred,zero_division=0)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}


def fit_calibrator(name,raw,y,seed):
    if name=="C0":return {"name":"C0","type":"identity"}
    if name=="C1":
        x=np.asarray(raw,float);target=np.asarray(y,float)
        def objective(theta):
            z=theta[0]*x+theta[1]
            return float(np.mean(np.logaddexp(0,z)-target*z))
        def gradient(theta):
            z=theta[0]*x+theta[1];residual=sigmoid(z)-target
            return np.asarray([np.mean(residual*x),np.mean(residual)])
        result=minimize(objective,np.asarray([1.0,0.0]),jac=gradient,method="L-BFGS-B",
                        options={"maxiter":2000,"ftol":1e-15,"gtol":1e-10})
        if not result.success or not np.all(np.isfinite(result.x)):raise RuntimeError(f"Platt fit failed: {result.message}")
        return {"name":"C1","type":"platt","coefficient":float(result.x[0]),"intercept":float(result.x[1]),
                "optimizer":"scipy_L-BFGS-B","objective":"unregularized_binary_log_loss","seed_unused_deterministic":int(seed)}
    if name=="C2":
        model=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(raw,y)
        return {"name":"C2","type":"isotonic","x_thresholds":model.X_thresholds_.astype(float).tolist(),"y_thresholds":model.y_thresholds_.astype(float).tolist()}
    raise ValueError(name)
def apply_calibrator(spec,raw):
    if spec["type"]=="identity":return sigmoid(raw)
    if spec["type"]=="platt":return sigmoid(float(spec["coefficient"])*np.asarray(raw)+float(spec["intercept"]))
    if spec["type"]=="isotonic":return np.interp(raw,np.asarray(spec["x_thresholds"]),np.asarray(spec["y_thresholds"]))
    raise ValueError(spec["type"])


def load_inputs(a,protocol,log):
    root=a.root.resolve();v205=root/"ESMCHalo_v2_optimization/V2_05_hard_negative_and_peft/V2_05A_hard_sample_weighting_ablation_v1"
    paths={
      "folds":root/"ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof/group_folds_v2.tsv",
      "registry":root/"ESMCHalo_v2_optimization/V2_02_label_audit/V2_02B_audit_lock_v1/V2_02_LOCKED_TRAINING_REGISTRY.tsv",
      "v205_done":v205/"V2_05A_DONE.json","v205_selection":v205/"selected_hard_sample_strategy.json",
      "v205_reproduction":v205/"H0_REPRODUCTION_AUDIT.json","v205_oof":v205/"all_candidate_oof_predictions.tsv",
      "v205_binding":v205/"V2_05A_BINDING_AUDIT.json","v205_access":v205/"DATA_ACCESS_AUDIT.json",
      **{f"member_{f}":v205/f"partial/H0/fold_{f}.pt" for f in range(5)}}
    safe={}; hashes={}
    for role,path in paths.items():
        p=guarded(path,role,log);actual=sha(p)
        if actual!=EXPECTED[role]:raise RuntimeError(f"SHA256 mismatch {role}: expected={EXPECTED[role]} actual={actual}")
        safe[role]=p;hashes[role]={"path":str(p),"sha256":actual,"status":"PASS"}
    done=json.loads(safe["v205_done"].read_text());sel=json.loads(safe["v205_selection"].read_text());rep=json.loads(safe["v205_reproduction"].read_text())
    if done.get("status")!="PASS_SCREEN_COMPLETE_NO_STABLE_GAIN" or done.get("selected_arm")!="H0":raise RuntimeError("V2-05A completion gate failed")
    if sel.get("decision")!="STOP_HARD_SAMPLE_NO_STABLE_GAIN" or sel.get("fallback")!="KEEP_K2_UNWEIGHTED":raise RuntimeError("V2-05A selection gate failed")
    if rep.get("status")!="PASS_H0_REPRODUCTION" or rep.get("passed") is not True:raise RuntimeError("H0 reproduction gate failed")
    for obj in (done,sel):
        for k in ("blind_labels_read","test200_labels_read","external_challenge_labels_read","lora"):
            if obj.get(k) is not False:raise RuntimeError(f"Upstream boundary gate failed: {k}")
    oof=pd.read_csv(safe["v205_oof"],sep="\t");oof=oof[oof.arm.astype(str)=="H0"].copy()
    req={"clean_id","label","homology_group","fold","student_raw_logit","probability_uncalibrated"}
    if not req.issubset(oof.columns) or len(oof)!=protocol["records_expected"] or not oof.clean_id.is_unique:raise ValueError("H0 OOF structure failed")
    oof=oof.sort_values("clean_id").reset_index(drop=True)
    if set(oof.fold)!=set(protocol["folds_expected"]) or oof.groupby("homology_group").fold.nunique().max()!=1:raise ValueError("OOF fold/group gate failed")
    if not np.allclose(sigmoid(oof.student_raw_logit),oof.probability_uncalibrated,rtol=0,atol=2e-15):raise ValueError("Raw-logit probability binding failed")
    folds=pd.read_csv(safe["folds"],sep="\t");fid=pick(folds,IDS,"fold ID");fcol=pick(folds,FOLDS,"fold");gcol=pick(folds,GROUPS,"group");scol=pick(folds,STRICT25,"strict25")
    fm=folds.set_index(folds[fid].astype(str));lookup=oof.clean_id
    if not set(lookup).issubset(fm.index):raise ValueError("Fold table misses H0 IDs")
    if not np.array_equal(oof.fold.to_numpy(int),pd.to_numeric(fm.loc[lookup,fcol]).to_numpy(int)):raise ValueError("Fold mismatch")
    if not np.array_equal(oof.homology_group.astype(str).to_numpy(),fm.loc[lookup,gcol].astype(str).to_numpy()):raise ValueError("Group mismatch")
    oof["strict25_member"]=binary(fm.loc[lookup,scol],"strict25")
    registry=pd.read_csv(safe["registry"],sep="\t");rid=pick(registry,IDS,"registry ID");tier=pick(registry,TIERS,"audit tier")
    rm=registry.set_index(registry[rid].astype(str));norm=rm.loc[lookup,tier].astype(str).map(lambda s:re.sub(r"[^a-z0-9]+","_",s.lower()).strip("_"))
    oof["clean_core_member"]=norm.map(lambda s:s in {"a","a_clean","a_clean_core"} or (s.startswith("a_") and "clean" in s)).to_numpy(int)
    if int(oof.strict25_member.sum())!=10458 or int(oof.clean_core_member.sum())!=7934:raise ValueError("Diagnostic subset count mismatch")
    return oof,safe,hashes


def calibration_screen(oof,protocol):
    rows=[];foldrows=[];models={};pred=pd.DataFrame({"clean_id":oof.clean_id,"label":oof.label,"homology_group":oof.homology_group,"fold":oof.fold,
        "strict25_member":oof.strict25_member,"clean_core_member":oof.clean_core_member,"student_raw_logit":oof.student_raw_logit})
    for name in ("C0","C1","C2"):
        p=np.empty(len(oof),float);models[name]={}
        for fold in range(5):
            train=oof.fold!=fold;valid=oof.fold==fold;spec=fit_calibrator(name,oof.loc[train,"student_raw_logit"].to_numpy(),oof.loc[train,"label"].to_numpy(),protocol["seed"]+fold)
            p[valid]=apply_calibrator(spec,oof.loc[valid,"student_raw_logit"].to_numpy());models[name][str(fold)]=spec
            foldrows.append({"candidate":name,"fold":fold,**prob_metrics(oof.loc[valid,"label"],p[valid],protocol["calibration"]["ece_bins"])})
        pred[f"probability_{name}"]=p;rows.append({"candidate":name,**prob_metrics(oof.label,p,protocol["calibration"]["ece_bins"])})
    return pred,pd.DataFrame(rows),pd.DataFrame(foldrows),models


def subset_table(pred,mask,bins):
    rows=[]
    for name in ("C0","C1","C2"):rows.append({"candidate":name,**prob_metrics(pred.loc[mask,"label"],pred.loc[mask,f"probability_{name}"],bins)})
    return pd.DataFrame(rows)


def calibration_gates(overall,folds,strict):
    o=overall.set_index("candidate");s=strict.set_index("candidate");sd=folds.groupby("candidate")[["brier","ece"]].std(ddof=1);rows=[]
    for name in ("C0","C1","C2"):
        db=float(o.loc[name,"brier"]-o.loc["C0","brier"]);de=float(o.loc[name,"ece"]-o.loc["C0","ece"])
        dap=float(o.loc[name,"ap"]-o.loc["C0","ap"]);dau=float(o.loc[name,"auroc"]-o.loc["C0","auroc"])
        gain=db<=-.002 or de<=-.010;disc=dap>=-.001 and dau>=-.001
        sb=float(s.loc[name,"brier"]-s.loc["C0","brier"]);se=float(s.loc[name,"ece"]-s.loc["C0","ece"]);sg=sb<=.002 and se<=.010
        bl=max(float(sd.loc["C0","brier"])+.002,1.25*float(sd.loc["C0","brier"]));el=max(float(sd.loc["C0","ece"])+.010,1.25*float(sd.loc["C0","ece"]))
        stable=float(sd.loc[name,"brier"])<=bl and float(sd.loc[name,"ece"])<=el
        rows.append({"candidate":name,"delta_brier":db,"delta_ece":de,"delta_log_loss":float(o.loc[name,"log_loss"]-o.loc["C0","log_loss"]),
          "delta_ap":dap,"delta_auroc":dau,"gain_gate":gain,"discrimination_guard":disc,"strict25_delta_brier":sb,"strict25_delta_ece":se,
          "strict25_guard":sg,"fold_sd_brier":float(sd.loc[name,"brier"]),"fold_sd_brier_limit":bl,"fold_sd_ece":float(sd.loc[name,"ece"]),
          "fold_sd_ece_limit":el,"fold_stability_gate":stable,"passes_all_calibration_gates":name!="C0" and gain and disc and sg and stable})
    return pd.DataFrame(rows)


def best_threshold(y,p,protocol):
    g=protocol["threshold"]["candidate_grid"];grid=np.arange(g["minimum"],g["maximum"]+g["step"]/2,g["step"])
    values=np.asarray([matthews_corrcoef(y,p>=t) for t in grid]);best=values.max();ix=np.flatnonzero(np.isclose(values,best,rtol=0,atol=1e-12))
    choices=grid[ix];order=np.lexsort((choices,np.abs(choices-.5)));return float(choices[order[0]]),float(best)


def threshold_screen(pred,selected,protocol):
    p=pred[f"probability_{selected}"].to_numpy();y=pred.label.to_numpy(int);thresholds=[];cross=np.zeros(len(pred),int)
    for fold in range(5):
        train=pred.fold!=fold;valid=pred.fold==fold;t,m=best_threshold(y[train],p[train],protocol);cross[valid]=p[valid]>=t
        thresholds.append({"fold":fold,"selection_train_records":int(train.sum()),"validation_records":int(valid.sum()),"selected_threshold":t,"selection_train_mcc":m})
    fixed=class_metrics(y,p,.5);cross_mcc=float(matthews_corrcoef(y,cross));cross_sens=float(recall_score(y,cross,zero_division=0));vals=np.array([x["selected_threshold"] for x in thresholds])
    delta=cross_mcc-fixed["mcc"];sens_delta=cross_sens-fixed["sensitivity"];spread=float(vals.max()-vals.min())
    passed=delta>=protocol["threshold"]["gain_gate"]["minimum_cross_fitted_delta_mcc"] and sens_delta>=-protocol["threshold"]["gain_gate"]["maximum_sensitivity_drop"] and spread<=protocol["threshold"]["gain_gate"]["maximum_threshold_range"]
    final=float(np.median(vals)) if passed else .5
    audit={"status":"PASS_THRESHOLD_SCREEN","selected_calibrator":selected,"fold_thresholds":thresholds,"fixed_0_5_metrics":fixed,
      "cross_fitted_threshold_metrics":{"mcc":cross_mcc,"sensitivity":cross_sens,"delta_mcc":delta,"sensitivity_delta":sens_delta},
      "threshold_min":float(vals.min()),"threshold_max":float(vals.max()),"threshold_range":spread,"threshold_gate_passed":passed,
      "decision":"KEEP_CROSSFIT_THRESHOLD" if passed else "KEEP_FIXED_0_5","deployment_threshold":final,
      "deployment_threshold_descriptive_oof_metrics":class_metrics(y,p,final),
      "note":"Cross-fitted fold-threshold metrics are the unbiased threshold-selection diagnostic; deployment-threshold OOF metrics are descriptive only."}
    out=pred[["clean_id","label","homology_group","fold"]].copy();out["selected_crossfit_probability"]=p
    out["fold_selected_threshold"]=out.fold.map({x["fold"]:x["selected_threshold"] for x in thresholds});out["crossfit_threshold_prediction"]=cross;out["fixed_0_5_prediction"]=(p>=.5).astype(int)
    return audit,out


def group_bootstrap(pred,selected,reps,seed):
    groups=list(pred.groupby("homology_group",sort=False).indices.values());rng=np.random.default_rng(seed);rows=[]
    for rep in range(reps):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);s=pred.iloc[ix];y=s.label.to_numpy(int)
        a=prob_metrics(y,s[f"probability_{selected}"].to_numpy());b=prob_metrics(y,s.probability_C0.to_numpy())
        rows.append({"replicate":rep,"delta_brier":a["brier"]-b["brier"],"delta_ece":a["ece"]-b["ece"],"delta_log_loss":a["log_loss"]-b["log_loss"]})
    d=pd.DataFrame(rows);summary=[]
    for metric in ("delta_brier","delta_ece","delta_log_loss"):
        v=d[metric].to_numpy();summary.append({"metric":metric,"replicates":len(v),"mean":float(v.mean()),"ci_2_5":float(np.quantile(v,.025)),"median":float(np.median(v)),"ci_97_5":float(np.quantile(v,.975)),"fraction_lt_zero":float((v<0).mean())})
    return pd.DataFrame(summary)


def threshold_bootstrap(tpred,reps,seed):
    groups=list(tpred.groupby("homology_group",sort=False).indices.values());rng=np.random.default_rng(seed);vals=[]
    for _ in range(reps):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);s=tpred.iloc[ix];y=s.label
        vals.append(matthews_corrcoef(y,s.crossfit_threshold_prediction)-matthews_corrcoef(y,s.fixed_0_5_prediction))
    v=np.asarray(vals);return {"replicates":len(v),"mean_delta_mcc":float(v.mean()),"ci_2_5":float(np.quantile(v,.025)),"median":float(np.median(v)),"ci_97_5":float(np.quantile(v,.975)),"fraction_gt_zero":float((v>0).mean())}


def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True);return p.parse_args()


def main():
    a=parse_args();out=a.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=True);lock=(out/".V2_06.lock").open("a+")
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print("Another V2-06 process holds the lock",file=sys.stderr);return 3
    lock.seek(0);lock.truncate();lock.write(f"pid={os.getpid()} started={now()}\n");lock.flush();log=[];started=time.time()
    atomic_json(out/"STATUS.json",{"module":"V2-06","status":"STARTED","blind_labels_read":False,"lora":False})
    try:
        for role,path in (("root",a.root),("output",out),("protocol",a.protocol)):
            if any(x in str(path.resolve()).lower() for x in FORBIDDEN):raise RuntimeError(f"Forbidden {role} path: {path}")
        pp=guarded(a.protocol,"protocol",log);protocol=json.loads(pp.read_text())
        if protocol.get("module")!="V2-06" or protocol.get("protocol_status")!="LOCKED_BEFORE_EXECUTION":raise RuntimeError("Protocol gate failed")
        atomic_json(out/"V2_06_PROTOCOL_LOCKED.json",protocol);atomic_json(out/"STATUS.json",{"module":"V2-06","status":"PREFLIGHT_BINDING","blind_labels_read":False,"lora":False})
        oof,safe,hashes=load_inputs(a,protocol,log)
        binding={"status":"PASS","records":len(oof),"positive":int(oof.label.sum()),"negative":int((1-oof.label).sum()),"fold_sizes":{str(k):int(v) for k,v in oof.fold.value_counts().sort_index().items()},
          "unique_strict40_groups":int(oof.homology_group.nunique()),"maximum_folds_per_group":int(oof.groupby("homology_group").fold.nunique().max()),"strict25_records":int(oof.strict25_member.sum()),
          "clean_core_records":int(oof.clean_core_member.sum()),"source_arm":"H0","source_equivalence":"V2-04 K2 exact reproduction","input_hashes":hashes,
          "deployment_ensemble_dev_metrics_computed":False,"blind_test200_external_labels_read":False,"model_retraining":False,"lora":False}
        atomic_json(out/"V2_06_BINDING_AUDIT.json",binding);atomic_json(out/"INPUT_SHA256.json",{"protocol":{"path":str(pp),"sha256":sha(pp)},**hashes})
        atomic_json(out/"RUNTIME.json",{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__})
        atomic_json(out/"STATUS.json",{"module":"V2-06","status":"CALIBRATION_SCREEN","blind_labels_read":False,"lora":False})
        pred,overall,foldm,foldmodels=calibration_screen(oof,protocol);strict=subset_table(pred,pred.strict25_member.astype(bool),15);clean=subset_table(pred,pred.clean_core_member.astype(bool),15)
        gates=calibration_gates(overall,foldm,strict);eligible=gates[gates.passes_all_calibration_gates.astype(bool)].candidate.tolist()
        if eligible:
            complexity={"C1":1,"C2":2};rank=overall[overall.candidate.isin(eligible)].copy();rank["complexity"]=rank.candidate.map(complexity)
            selected=str(rank.sort_values(["brier","ece","log_loss","complexity"],ascending=True,kind="mergesort").iloc[0].candidate);decision="KEEP_CALIBRATOR"
        else:selected="C0";decision="KEEP_UNCALIBRATED_IDENTITY"
        pred.to_csv(out/"crossfit_calibrated_oof_predictions.tsv",sep="\t",index=False);overall.to_csv(out/"calibration_candidate_metrics.tsv",sep="\t",index=False);foldm.to_csv(out/"calibration_fold_metrics.tsv",sep="\t",index=False)
        strict.to_csv(out/"calibration_strict25_metrics.tsv",sep="\t",index=False);clean.to_csv(out/"calibration_clean_core_metrics.tsv",sep="\t",index=False);gates.to_csv(out/"calibration_gain_gates.tsv",sep="\t",index=False)
        atomic_json(out/"crossfit_calibrator_models.json",foldmodels)
        final_cal=fit_calibrator(selected,oof.student_raw_logit.to_numpy(),oof.label.to_numpy(),protocol["seed"]+60000)
        atomic_json(out/"selected_calibrator.json",{"status":"PASS_CALIBRATION_SCREEN","decision":decision,"selected_candidate":selected,"eligible_candidates":eligible,"selection_priority":protocol["calibration"]["selection_priority"],"final_fit_records":len(oof),"final_calibrator":final_cal})
        group_bootstrap(pred,selected,protocol["bootstrap"]["replicates"],protocol["seed"]+61000).to_csv(out/"selected_calibration_group_bootstrap.tsv",sep="\t",index=False)
        ta,tpred=threshold_screen(pred,selected,protocol);atomic_json(out/"threshold_crossfit_audit.json",ta);tpred.to_csv(out/"threshold_crossfit_oof_predictions.tsv",sep="\t",index=False)
        atomic_json(out/"threshold_group_bootstrap.json",threshold_bootstrap(tpred,protocol["bootstrap"]["replicates"],protocol["seed"]+62000))
        freeze=out/"frozen_ensemble";members=freeze/"members";members.mkdir(parents=True,exist_ok=True);manifest=[]
        for fold in range(5):
            target=members/f"fold_{fold}.pt";shutil.copy2(safe[f"member_{fold}"],target)
            if sha(target)!=EXPECTED[f"member_{fold}"]:raise RuntimeError("Frozen member copy hash mismatch")
            manifest.append({"fold":fold,"file":str(target.relative_to(out)),"sha256":sha(target),"aggregation_weight":0.2})
        atomic_json(freeze/"final_calibrator.json",final_cal);atomic_json(freeze/"final_threshold.json",{"deployment_threshold":ta["deployment_threshold"],"decision":ta["decision"],"probability_comparison":">="})
        helper=Path(__file__).with_name("frozen_ensemble_inference.py");shutil.copy2(helper,freeze/"inference.py")
        final_manifest={"status":"FROZEN_FOR_V2_07","representation":"canonical_mean_1152","student":"K2_H0_unweighted","ensemble_members":manifest,"aggregation":"mean_raw_logit",
          "calibrator":{"candidate":selected,"file":"frozen_ensemble/final_calibrator.json","sha256":sha(freeze/"final_calibrator.json")},
          "threshold":{"value":ta["deployment_threshold"],"file":"frozen_ensemble/final_threshold.json","sha256":sha(freeze/"final_threshold.json")},
          "inference_script":{"file":"frozen_ensemble/inference.py","sha256":sha(freeze/"inference.py")},"DeepSaltPro_required_at_inference":False,
          "development_ensemble_performance_claimed":False,"model_retraining":False,"blind_labels_read":False,"test200_labels_read":False,"external_challenge_labels_read":False,"lora":False,
          "immutability":"Do not modify models, aggregation, calibrator, or threshold after V2-07 begins."}
        atomic_json(out/"FINAL_FREEZE_MANIFEST.json",final_manifest)
        atomic_json(out/"DATA_ACCESS_AUDIT.json",{"status":"PASS_ALLOWED_INPUTS_ONLY","accessed_inputs":log,"forbidden_tokens":list(FORBIDDEN),"blind_labels_read":False,"test200_labels_read":False,
          "external_challenge_labels_read":False,"DeepSaltPro_loaded_for_inference":False,"model_retraining":False,"hard_sample_weighting":False,"candidate_negative_pool_used":False,"lora_or_peft":False})
        done={"module":"V2-06","status":"PASS_ENSEMBLE_CALIBRATION_FREEZE","started_at_utc":datetime.fromtimestamp(started,timezone.utc).isoformat(),"finished_at_utc":now(),"elapsed_seconds":round(time.time()-started,3),
          "records":len(oof),"ensemble_members":5,"aggregation":"mean_raw_logit","selected_calibrator":selected,"calibration_decision":decision,"deployment_threshold":ta["deployment_threshold"],"threshold_decision":ta["decision"],
          "development_ensemble_performance_claimed":False,"model_retraining":False,"blind_labels_read":False,"test200_labels_read":False,"external_challenge_labels_read":False,"lora":False,"next_gate":"V2-07 one-time blind evaluation only after review"}
        atomic_json(out/"V2_06_DONE.json",done);atomic_json(out/"STATUS.json",done);return 0
    except Exception as e:
        fail={"module":"V2-06","status":"FAILED","updated_at_utc":now(),"error_type":type(e).__name__,"error":str(e),"blind_labels_read":False,"lora":False}
        atomic_json(out/"STATUS.json",fail);atomic_json(out/"DATA_ACCESS_AUDIT.json",{"status":"FAILED_DURING_ALLOWED_INPUT_PROCESSING","accessed_inputs":log,"blind_labels_read":False,"test200_labels_read":False,"external_challenge_labels_read":False,"lora_or_peft":False});traceback.print_exc();return 1


if __name__=="__main__":raise SystemExit(main())
