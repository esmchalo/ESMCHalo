from pathlib import Path
import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);out=a.output.resolve();here=Path(__file__).resolve().parent
for name in ['check_metrics.py','check_statistics.py']:
 subprocess.run([sys.executable,str(here/name),str(out)],check=True)
print('Result tables written to',out)
