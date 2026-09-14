#!/usr/bin/env python3
"""Check package bytes; this is not a scientific execution acceptance."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 failures=[];count=0
 for line in (R/'MANIFEST.sha256').read_text().splitlines():
  h,n=line.split('  ',1);p=R/n;count+=1
  if not p.is_file() or sha(p)!=h:failures.append(n)
 print(json.dumps({'scope':'listed file bytes only','files':count,'status':'PASS' if not failures else 'FAIL','failures':failures},indent=2));return bool(failures)
if __name__=='__main__':sys.exit(main())
