#!/usr/bin/env python3
"""Explicit historical stage launcher. Never called by RUN_ACCEPTANCE.sh."""
import argparse,json,pathlib,subprocess,sys,shlex
B=pathlib.Path(__file__).resolve().parents[1];H=B/'historical_project'
a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=['teacher','kd','weighting','calibration']);a.add_argument('--root',type=pathlib.Path,required=True,help='Prepared historical input tree, including explicitly documented caches');a.add_argument('--output',type=pathlib.Path,required=True,help='NEW output directory; existing directory refused');a.add_argument('--device',default='cuda:0');a.add_argument('--execute',action='store_true',help='Without this flag only print the exact command');args=a.parse_args()
patterns={'teacher':('v2_01d_execution_package/*/v2_01d_nested_deepsaltpro_oof.py',None),'kd':('v2_04_execution_package_v1_1_fix/*/v2_04_deepsaltpro_oof_kd.py','V2_04_PROTOCOL.json'),'weighting':('v2_05a_execution_package/*/v2_05a_hard_sample_weighting.py','V2_05A_PROTOCOL.json'),'calibration':('v2_06_execution_package/*/v2_06_ensemble_calibration_freeze.py','V2_06_PROTOCOL.json')}
pattern,protocol=patterns[args.stage];source=next(H.glob(pattern));cmd=[sys.executable,str(source),'--root',str(args.root.resolve()),'--output',str(args.output.resolve())]
if args.stage!='calibration':cmd+=['--device',args.device]
if protocol:cmd+=['--protocol',str(source.parent/protocol)]
if args.stage=='teacher':cmd+=['--outer-folds','all','--seed','20260902','--inner-splits','4','--pca-components','512','--epochs','100','--batch-size','128','--eval-batch-size','1024','--learning-rate','4e-5','--weight-decay','1e-4','--patience','25','--min-mcc-improvement','0.005']
elif args.stage!='calibration':cmd+=['--threads','16']
print(shlex.join(cmd),flush=True)
if args.execute:
 if args.output.exists():a.error('Output already exists. This launcher never overwrites original runs.')
 if not args.root.is_dir():a.error('Input tree missing')
 sys.exit(subprocess.call(cmd))
