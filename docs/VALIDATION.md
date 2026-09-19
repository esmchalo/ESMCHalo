# Acceptance record

## Historical acceptance

The earlier acceptance record preserved the initial historical FASTA mismatch and subsequent correction of the historical short/long extraction path.

The corrected historical reference acceptance reproduced three biological sequences with maximum calibrated-probability absolute difference `8.203488999214414e-09`.

All threshold-0.5 classifications matched.

Boundary sequences of lengths 64, 2046, 2047 and 3837 residues produced 1, 1, 2 and 3 windows respectively.

## v2.0.2-rc2 clean-install acceptance

Candidate 08 was validated in a newly created Python 3.12 environment.

Installation from the candidate `requirements.txt` completed successfully and `pip check` returned no broken requirements.

Validated runtime:

- Python 3.12.14
- `esm==3.4.1.post1`
- `torch==2.11.0`
- `transformers==4.57.6`
- NumPy 2.5.1
- pandas 3.0.3
- SciPy 1.18.0
- scikit-learn 1.9.0
- NVIDIA GeForce RTX 4090
- PyTorch CUDA 13.0

The historical ESMC-600M model loaded as `esm.models.esmc.EsmcForMaskedLM` with BF16 parameters.

The accepted `model.safetensors` SHA256 was:

`e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8`

## Short-reference regression

Observed absolute calibrated-probability differences:

- `HPCLAS_TRAIN_P_005098`: `4.705891232248405e-10`
- `HPCLAS_TRAIN_N_000101`: `1.095634300299353e-09`
- `HPCLAS_TRAIN_N_005021`: `8.203488999214414e-09`

Maximum absolute difference:

`8.203488999214414e-09`

Release acceptance threshold:

`<= 1e-6`

All three threshold-0.5 predictions matched exactly.

Result:

`NUMERICAL_REFERENCE: PASS`

## Boundary regression

Observed:

- 64 aa -> 1 window
- 2046 aa -> 1 window
- 2047 aa -> 2 windows
- 3837 aa -> 3 windows

Result:

`BOUNDARY_CONTRACT: PASS`

## Frozen inference assets

Candidate 07 and Candidate 08 were compared by SHA256 content.

The following files were unchanged:

- `fold_0.pt`
- `fold_1.pt`
- `fold_2.pt`
- `fold_3.pt`
- `fold_4.pt`
- `final_calibrator.json`
- `final_threshold.json`

Result:

`FROZEN_ASSETS_CONTENT_UNCHANGED: PASS`

Final status:

`CANDIDATE_08_RELEASE_ACCEPTANCE: PASS`

## Outside the scope of this acceptance

This acceptance does not establish:

- complete fresh retraining of the historical development pipeline;
- byte-for-byte equivalence between a newly downloaded public ESMC-600M snapshot and the historical accepted local asset;
- every historical supplementary bootstrap computation.
