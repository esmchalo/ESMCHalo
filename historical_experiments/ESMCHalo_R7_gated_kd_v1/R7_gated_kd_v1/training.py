"""R7 train loops derived from V2-05A; same scaling, splits, optimizer, seeds and early stopping."""
from legacy_v205 import *

def training_loss(model,x,y,u,arm,rng):
    if arm!='G0': raise ValueError('Unknown R7 arm '+arm)
    if x.shape[0]==0 or y.shape!=u.shape or not bool(torch.isfinite(u).all()):
        raise ValueError('Invalid G0 training batch')
    z=model(x)
    gate=((u>=0)==(y>=.5)).to(z.dtype)
    hard=F.binary_cross_entropy_with_logits(z,y,reduction='none')
    soft=F.binary_cross_entropy_with_logits(z/2,torch.sigmoid(u/2),reduction='none')
    return (.5*hard+2*gate*soft).mean()


def choose_epoch(x: np.ndarray, y: np.ndarray, teacher: np.ndarray, weights: np.ndarray,
                 x_valid: np.ndarray, y_valid: np.ndarray, arm: str, protocol: dict[str, Any],
                 device: torch.device, seed: int) -> tuple[int, float, list[dict[str, Any]]]:
    spec = protocol["student"]; seed_all(seed)
    mix_rng = np.random.default_rng(seed + 910000)
    model = Student(x.shape[1], float(spec["dropout"])).to(device); opt = optimizer(model, spec)
    dl = loader(x, y, teacher, weights, int(spec["batch_size"]), True, seed)
    best, best_epoch, stale, history = -math.inf, 1, 0, []
    for epoch in range(1, int(spec["maximum_epochs"]) + 1):
        model.train(); total = 0.0; batches = 0
        for bx, by, bt, bw in dl:
            bx, by, bt, bw = (v.to(device, non_blocking=True) for v in (bx, by, bt, bw))
            opt.zero_grad(set_to_none=True)
            loss = training_loss(model, bx, by, bt, arm, mix_rng)
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"Non-finite loss {arm}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip_norm"])); opt.step()
            total += float(loss.detach().cpu()); batches += 1
        _, p = predict(model, x_valid, device, int(spec["batch_size"]) * 2)
        ap = float(average_precision_score(y_valid, p))
        history.append({"epoch": epoch, "mean_training_loss": total / max(batches, 1), "inner_ap": ap})
        print(f"{arm} inner epoch={epoch} AP={ap:.6f} loss={total/max(batches,1):.6f}", flush=True)
        if ap > best + 1e-6: best, best_epoch, stale = ap, epoch, 0
        else: stale += 1
        if stale >= int(spec["early_stopping_patience"]): break
    return best_epoch, best, history


def fit_epochs(x: np.ndarray, y: np.ndarray, teacher: np.ndarray, weights: np.ndarray,
               arm: str, protocol: dict[str, Any], device: torch.device, epochs: int, seed: int) -> Student:
    spec = protocol["student"]; seed_all(seed)
    mix_rng = np.random.default_rng(seed + 910000)
    model = Student(x.shape[1], float(spec["dropout"])).to(device); opt = optimizer(model, spec)
    dl = loader(x, y, teacher, weights, int(spec["batch_size"]), True, seed)
    for epoch in range(epochs):
        model.train()
        for bx, by, bt, bw in dl:
            bx, by, bt, bw = (v.to(device, non_blocking=True) for v in (bx, by, bt, bw))
            opt.zero_grad(set_to_none=True); loss = training_loss(model, bx, by, bt, arm, mix_rng)
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"Non-finite loss {arm}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip_norm"])); opt.step()
        print(f"{arm} outer epoch={epoch+1}/{epochs} completed", flush=True)
    return model


def run_fold(arm: str, fold: int, x: np.ndarray, master: pd.DataFrame, flags: pd.DataFrame,
             protocol: dict[str, Any], device: torch.device, out: Path) -> pd.DataFrame:
    directory = out / "partial" / arm; directory.mkdir(parents=True, exist_ok=True)
    tsv, meta, ckpt = directory/f"fold_{fold}.tsv", directory/f"fold_{fold}.json", directory/f"fold_{fold}.pt"
    if tsv.is_file() and meta.is_file() and ckpt.is_file():
        m = json.loads(meta.read_text())
        if m.get("status") == "PASS" and m.get("result_sha256") == sha256(tsv) and m.get("checkpoint_sha256") == sha256(ckpt):
            reused = read_tsv(tsv)
            if len(reused) == int((master.fold == fold).sum()) and reused.clean_id.is_unique:
                progress(out/"PROGRESS.tsv", {"utc": now(), "arm": arm, "fold": fold, "state": "REUSED_VALID_CHECKPOINT"}); return reused
    inner = (fold + 1) % 5; select_train = (master.fold != fold) & (master.fold != inner)
    inner_valid, outer_train, outer_valid = master.fold == inner, master.fold != fold, master.fold == fold
    if set(master.loc[outer_train, "homology_group"]) & set(master.loc[outer_valid, "homology_group"]): raise RuntimeError("Outer leakage")
    weights = np.ones(len(master), dtype=np.float32)
    ss = StandardScaler(); xs = ss.fit_transform(x[select_train]).astype(np.float32); xv = ss.transform(x[inner_valid]).astype(np.float32)
    seed = int(protocol["seed"]) + fold * 100 + int(protocol["k2_seed_offset"])
    epochs, best, history = choose_epoch(xs, master.loc[select_train,"label"].to_numpy(np.float32),
        master.loc[select_train,"teacher_logit"].to_numpy(np.float32), weights[select_train], xv,
        master.loc[inner_valid,"label"].to_numpy(np.int64), arm, protocol, device, seed)
    sf = StandardScaler(); xt = sf.fit_transform(x[outer_train]).astype(np.float32); xo = sf.transform(x[outer_valid]).astype(np.float32)
    model = fit_epochs(xt, master.loc[outer_train,"label"].to_numpy(np.float32),
        master.loc[outer_train,"teacher_logit"].to_numpy(np.float32), weights[outer_train], arm,
        protocol, device, epochs, seed + 50000)
    raw, p = predict(model, xo, device, int(protocol["student"]["batch_size"]) * 2)
    result = master.loc[outer_valid, ["clean_id","label","homology_group","fold","teacher_logit"]].copy()
    result.insert(0, "arm", arm); result["student_raw_logit"] = raw; result["probability_uncalibrated"] = p
    result["prediction_at_0_5"] = (p >= 0.5).astype(int); result.to_csv(tsv, sep="\t", index=False)
    torch.save({"module":"R7","arm":arm,"outer_fold":fold,"inner_fold":inner,"selected_epochs":epochs,
                "model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "scaler_mean":sf.mean_.astype(np.float64),"scaler_scale":sf.scale_.astype(np.float64),
                "canonical_mean_only":True,"k2_fixed":False,"lora":False,"upweight":1.0,"training_variant":arm,"teacher_supervision_used":True,"teacher_gate":"training_label_agreement"}, ckpt)
    atomic_json(meta, {"status":"PASS","arm":arm,"outer_fold":fold,"inner_fold":inner,
        "outer_train_records":int(outer_train.sum()),"outer_validation_records":int(outer_valid.sum()),
        "weighted_outer_train_records":int((weights[outer_train] > 1).sum()),"selected_epochs":epochs,
        "inner_training_gate_on":int(((master.loc[select_train,"teacher_logit"]>=0)==(master.loc[select_train,"label"]>=.5)).sum()),
        "inner_training_records":int(select_train.sum()),
        "outer_training_gate_on":int(((master.loc[outer_train,"teacher_logit"]>=0)==(master.loc[outer_train,"label"]>=.5)).sum()),
        "gate_labels_scope":"current_training_partition_only",
        "best_inner_ap":best,"epoch_history":history,"result_sha256":sha256(tsv),"checkpoint_sha256":sha256(ckpt),
        "outer_shared_groups":0,"validation_flags_used_for_training":False})
    progress(out/"PROGRESS.tsv", {"utc":now(),"arm":arm,"fold":fold,"state":"PASS","selected_epochs":epochs,"best_inner_ap":best})
    return result

