from pathlib import Path
import argparse,hashlib,json,sys
p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);a=p.parse_args();expected=json.loads((Path(__file__).resolve().parents[1]/'src/HISTORICAL_MODEL_SHA256.json').read_text());errors=[]
for name,h in expected.items():
 f=a.model_dir/name
 if not f.is_file():errors.append(name+': missing');continue
 d=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):d.update(b)
 if d.hexdigest()!=h:errors.append(name+': hash mismatch')
print(json.dumps({'status':'FAIL' if errors else 'PASS','errors':errors,'scope':'model assets only, not environment/backend identity'},indent=2));sys.exit(bool(errors))
