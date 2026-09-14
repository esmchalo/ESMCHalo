"""Compatibility adapter to the original extractor; no alternate backend."""
from pathlib import Path
import json, hashlib
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from historical_esmc_extractor import extract_short_batch, extract_long_sequence

class ESMCEmbedder:
    def __init__(self, device_name, dtype_name, allow_model_download, cache_dir, model_dir):
        if allow_model_download or cache_dir is not None:
            raise ValueError('This verification uses only the exact historical local model directory.')
        if dtype_name not in ('auto','bfloat16'):
            raise ValueError('Historical inference requires bfloat16 weights.')
        self.device = torch.device('cuda:0' if device_name=='auto' else device_name)
        if self.device.type!='cuda' or not torch.cuda.is_available():
            raise RuntimeError('Historical GPU execution required; no automatic fallback.')
        self.model_path=Path(model_dir).expanduser().resolve()
        if not self.model_path.is_dir():raise FileNotFoundError(self.model_path)
        expected=json.loads(Path(__file__).with_name('HISTORICAL_MODEL_SHA256.json').read_text())
        for name,digest in expected.items():
            p=self.model_path/name
            if not p.is_file():raise FileNotFoundError(p)
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=digest:raise RuntimeError('Historical ESMC hash mismatch: '+name)
        config=AutoConfig.from_pretrained(self.model_path,local_files_only=True)
        if int(config.d_model)!=1152:raise ValueError('Historical model dimension mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_path,local_files_only=True)
        self.model=AutoModelForMaskedLM.from_pretrained(self.model_path,local_files_only=True,dtype=torch.bfloat16)
        self.model=self.model.to(self.device).eval()
        print('Historical backend:',type(self.model).__module__,type(self.model).__name__,'parameter dtype:',next(self.model.parameters()).dtype,flush=True)
    def embed_batch(self, records):
        records=list(records)
        if len(records)!=1:raise ValueError('This verification preserves historical batch size 1.')
        seq=records[0].sequence
        if len(seq)<=2046:
            x=extract_short_batch(self.model,self.tokenizer,[seq],self.device);n=1
        else:
            row,n=extract_long_sequence(self.model,self.tokenizer,seq,1152,2046,256,self.device);x=row[None,:]
        if x.shape!=(1,1152) or not np.isfinite(x).all():raise ValueError('Invalid historical representation')
        return x,np.array([n],dtype=np.uint16)
