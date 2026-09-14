# Acceptance record

Source result archives:
- acceptance_20260914T063501_950671Z.zip
- fasta_fix_result_20260914T064837_917397Z.zip

The first acceptance passed metrics, primary statistics, seven frozen-feature references and synthetic boundary checks. Historical FASTA failed (maximum probability difference 0.01411191355); this failure remains documented in evidence/acceptance/initial.

The correction restored original model loading and original short/long extraction functions, rather than increasing tolerance. The second acceptance passed three biological sequences (maximum probability difference 8.203488999214414e-9, original tolerance 1e-4, all classes identical) and four boundary sequences (64/2046/2047/3837 residues; 1/1/2/3 windows). It does not isolate one arithmetic operation as the sole cause of the earlier difference. Synthetic boundary outputs have no independent historical numerical oracle.

Runtime: Python3.12.13; NumPy2.5.1; pandas3.0.3; SciPy1.18.0; sklearn1.9.0; torch2.13.0; transformers4.57.6; native ESMCForMaskedLM BF16; RTX4090 reporting49140MiB; driver580.178.04. These describe acceptance, not the teacher's historical software environment or an explanation of historical GPU memory configuration.

Training --help succeeded at all four main entries; five locked input hashes matched. The old training_inventory worker always returned BLOCKED for uncertified portability. That status is not a training failure or evidence that a directory marked is_directory=true was absent.

Not performed: complete fresh retraining, clean installation, public snapshot equivalence, every supplementary bootstrap interval. The rc1 refactor parameterizes the model directory and extracts the existing parser; those packaging changes received local structural/byte/metric checks, not a new GPU execution. Release descriptions must retain that distinction.
