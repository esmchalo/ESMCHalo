#!/usr/bin/env python3
"""Apply the frozen V2-06 five-student ensemble to canonical-mean features."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1152,256),nn.GELU(),nn.Dropout(.2),
                                     nn.Linear(256,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,1))
    def forward(self,x): return self.network(x).squeeze(-1)


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-50,50)))


def calibrate(raw, spec):
    kind=spec["type"]
    if kind=="identity": return sigmoid(raw)
    if kind=="platt": return sigmoid(float(spec["coefficient"])*raw+float(spec["intercept"]))
    if kind=="isotonic":
        return np.interp(raw,np.asarray(spec["x_thresholds"],float),np.asarray(spec["y_thresholds"],float))
    raise ValueError(f"Unknown calibrator: {kind}")


@torch.no_grad()
def score(checkpoint, x, device, batch_size):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if ck.get("arm")!="H0" or ck.get("canonical_mean_only") is not True or ck.get("lora") is not False:
        raise RuntimeError(f"Checkpoint contract failed: {checkpoint}")
    model=Student().to(device); model.load_state_dict(ck["model_state_dict"]); model.eval()
    mean=np.asarray(ck["scaler_mean"],np.float32); scale=np.asarray(ck["scaler_scale"],np.float32)
    z=(x-mean)/scale; parts=[]
    for b in DataLoader(torch.from_numpy(z.astype(np.float32)),batch_size=batch_size,shuffle=False):
        parts.append(model(b.to(device)).float().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--freeze-dir",type=Path,required=True)
    p.add_argument("--features",type=Path,required=True);p.add_argument("--rows",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=512)
    a=p.parse_args(); d=a.freeze_dir.resolve(); x=np.load(a.features,allow_pickle=False)
    rows=pd.read_csv(a.rows,sep="\t"); id_col=next((c for c in ("clean_id","id","protein_id","sample_id") if c in rows.columns),None)
    if id_col is None or x.shape!=(len(rows),1152) or not np.all(np.isfinite(x)): raise ValueError("Feature/row binding failed")
    device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    members=sorted((d/"members").glob("fold_*.pt"))
    if len(members)!=5: raise RuntimeError("Frozen ensemble must contain five members")
    raw=np.mean(np.vstack([score(c,np.asarray(x,np.float32),device,a.batch_size) for c in members]),axis=0)
    cal=json.loads((d/"final_calibrator.json").read_text()); threshold=json.loads((d/"final_threshold.json").read_text())
    prob=calibrate(raw,cal); t=float(threshold["deployment_threshold"])
    out=pd.DataFrame({"clean_id":rows[id_col].astype(str),"ensemble_mean_raw_logit":raw,
                      "probability_calibrated":prob,"prediction":(prob>=t).astype(int)})
    out.to_csv(a.output,sep="\t",index=False)


if __name__=="__main__": main()
