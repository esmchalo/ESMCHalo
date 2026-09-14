# ESMCHalo — 2.0.2-rc1

Pre-release candidate for halophile-associated protein prediction. This update preserves the five frozen H0 student heads, their preprocessing, Platt calibration and decision threshold 0.5. It restores the historical ESMC extraction implementation and adds historical training materials, record-level evaluation inputs, result-recomputation scripts and scoped acceptance evidence.

**This is not a verified from-scratch training distribution.** Historical ESMC file hashes are available; their identity with an independently downloadable snapshot and a clean installation of the native ESMC backend remain unverified. No new version DOI has been assigned in this package.

## Validated scope

The antecedent server acceptance passed 378 metric/calibration checks, 38 primary statistical comparisons and seven synthetic feature predictions. With the corrected historical extraction path, three biological FASTA records matched historical probabilities with maximum absolute error 8.20349e-9 and identical classes. Four existing boundary examples passed parsing/window-count/output checks. The reorganized rc1 package retains the numerical functions but has not itself been rerun on GPU. See `evidence/VALIDATION_SCOPE.json` and `docs/VALIDATION.md`.

## Environment and model

`requirements.txt` records observed server versions; `environment.yml` is an installation recipe, not proof of successful clean installation. A compatible native `transformers.models.esmc` implementation is required. A version string alone does not establish identical backend code. CUDA and BF16 are required by the historical inference path; the default batch size is one. No backend fallback is provided by the ESMCHalo adapter.

Third-party ESMC weights are not bundled. See `docs/MODEL_ACQUISITION.md`. Given a lawfully obtained historical model directory:

```bash
python scripts/check_model_files.py --model-dir /path/to/ESMC-600M
python src/esmchalo_v2_predict.py --model-dir /path/to/ESMC-600M --input tests/historical_short.fasta --output predictions.tsv --device cuda:0 --dtype bfloat16 --batch-size 1
```

The wrapper verifies the five required model assets before loading. It does not silently substitute another checkpoint. Five-head prediction from existing canonical features remains available:

```bash
python src/frozen_ensemble_inference.py --freeze-dir artifacts/frozen_ensemble --features tests/synthetic_frozen_inference/features.npy --rows tests/synthetic_frozen_inference/rows.tsv --output feature_predictions.tsv --device cpu
```

## Recompute results

```bash
python scripts/reproduce_results.py --output recomputed_results
```

Use a new output directory. This replays R1 recorded timings, R3–R9 point metrics and the primary holdout/R3 paired statistics. It neither trains models nor retunes thresholds. Primary bootstrap computations can take several minutes or longer. Raw model inference is not required for this route.

## Training materials and data

`historical_project/` preserves original teacher/KD/weighting/calibration source and available fixed protocols, fold tables, training registry and teacher OOF targets. `historical_experiments/` preserves R4–R9 execution sources. Original absolute-path/hash bindings remain in these historical records; they are not a portable pipeline certification. See `docs/TRAINING_AND_DATA.md` for exact entry points, prerequisites and omitted raw source data.

Do not run historical entry points against the original project to overwrite completed experiments. The `scripts/train_stage.py` helper prints commands by default and only executes with an explicit `--execute`, rejecting an existing output directory. Its stages are not automatically connected into a fresh end-to-end run.

## Scientific scope

The task is prediction of association with halophilic organisms, not salt-specific activity, mutation effects or wet-lab validation. Teacher targets can create dependencies across student folds. Historical evaluation-data use must be disclosed. Existing challenges did not show consistent classification improvements. Directory names containing “blind” and historical protocol assertions are provenance, not evidence of first-ever project-wide blind evaluation.

Code: MIT for original ESMCHalo code. Original model/results/documentation: CC BY 4.0 under `LICENSE_SCOPE.md`; third-party rights are excluded. Repository: https://github.com/esmchalo/ESMCHalo . The prior v2.0.1 archive DOI is 10.5281/zenodo.22334942; it must not be cited as the DOI of this candidate.
