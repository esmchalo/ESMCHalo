# ESMCHalo — 2.0.2-rc2

Pre-release candidate for halophile-associated protein prediction.

This release candidate preserves the five frozen H0 student heads, their preprocessing, Platt calibration, decision threshold 0.5, and the historical short/long ESMC extraction contract.

The principal release-engineering change is migration of the historical ESMC loading backend from the previous Biohub Transformers VCS dependency to the public PyPI package `esm==3.4.1.post1`.

No frozen student checkpoint, calibrator, threshold, long-sequence window rule, or reported model result was changed by this backend migration.

## Validated scope

A fresh Python 3.12 environment was created and the rc2 candidate requirements were installed from scratch.

Validated core runtime:

- `esm==3.4.1.post1`
- `torch==2.11.0`
- `transformers==4.57.6`
- `numpy==2.5.1`
- `pandas==3.0.3`
- `scipy==1.18.0`
- `scikit-learn==1.9.0`
- NVIDIA GeForce RTX 4090
- PyTorch CUDA 13.0

`pip check` reported no broken requirements.

Three historical biological FASTA references reproduced calibrated probabilities with a maximum absolute difference of `8.203488999214414e-09`.

All three threshold-0.5 predictions matched exactly.

Boundary behavior was reproduced as:

- 64 aa -> 1 window
- 2046 aa -> 1 window
- 2047 aa -> 2 windows
- 3837 aa -> 3 windows

The five student checkpoints, Platt calibrator and threshold file were verified byte-for-byte unchanged.

See `docs/VALIDATION.md`.

## Environment and model

Create the environment with:

```bash
conda env create -f environment.yml
conda activate esmchalo-v2.0.2-rc2
```

or:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The historical inference path requires CUDA and BF16-capable GPU execution.

The rc2 release acceptance was performed successfully with the public Biohub `esm` backend under its pure-PyTorch fallback configuration when Transformer Engine, xformers and flash-attn were absent.

Third-party ESMC weights are not bundled.

The predictor requires a local historical ESMC-600M directory matching `src/HISTORICAL_MODEL_SHA256.json`.

Check the model files with:

```bash
python scripts/check_model_files.py --model-dir /path/to/ESMC-600M
```

Run FASTA inference with:

```bash
python src/esmchalo_v2_predict.py --model-dir /path/to/ESMC-600M --input tests/historical_short.fasta --output predictions.tsv --device cuda:0 --dtype bfloat16 --batch-size 1
```

The wrapper verifies the required model assets before loading and does not silently substitute another checkpoint.

## Recompute results

```bash
python scripts/reproduce_results.py --output recomputed_results
```

This replays recorded result tables and statistics. It does not retrain models or retune thresholds.

## Training materials and data

`historical_project/` preserves original teacher/KD/weighting/calibration sources and available protocols, fold tables, training registry and teacher OOF targets.

`historical_experiments/` preserves R4-R9 execution sources.

These historical records are provenance materials and are not a certification of portable from-scratch retraining.

See `docs/TRAINING_AND_DATA.md`.

## Scientific scope

The task is prediction of association with halophilic organisms, not salt-specific activity, mutation effects or experimentally measured salt tolerance.

Teacher-target dependencies across student folds and historical evaluation-data use should be retained when interpreting the reported results.

## Remaining portability limitation

Clean installation of the public inference backend has been verified.

However, rc2 does not claim that an independently downloaded public ESMC-600M snapshot has been demonstrated byte-for-byte identical to the historical locked model directory used for acceptance.

The accepted asset hashes therefore remain enforced by `src/HISTORICAL_MODEL_SHA256.json`.

## License and citation

Original ESMCHalo code: MIT.

Original model/results/documentation: CC BY 4.0 under `LICENSE_SCOPE.md`.

Third-party rights are excluded.

Repository:

https://github.com/esmchalo/ESMCHalo
