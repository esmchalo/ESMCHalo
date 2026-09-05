# ESMCHalo-v2 frozen inference release

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22334941.svg)](https://doi.org/10.5281/zenodo.22334941)

This release contains the five immutable V2-06 student checkpoints, equal
raw-logit ensemble inference, the frozen C1 Platt calibrator and threshold 0.5.
It contains no training code, private labels or labelled blind predictions.

`src/esmchalo_v2_predict.py` provides FASTA inference using the previously
validated canonical ESMC-600M layer-36 mean extractor, including the locked
long-sequence overlap-resolution contract. The old LR artifact is not included
or used. DeepSaltPro is not required at inference.

Verify first:

```bash
python src/verify_release.py
```

Example inference:

```bash
python src/esmchalo_v2_predict.py --input proteins.fasta --output predictions.tsv --device cuda:0
```

The threshold is frozen for this release. Users should validate calibration
and operating characteristics prospectively in materially different domains.

## Citation

Please cite the archived software release as:

> Lv, F., Jin, H., Ma, Q., Cheng, C., & Xue, C. (2026). ESMCHalo:
> homology-aware halophile-associated protein prediction with leakage-free
> knowledge distillation (Version v2.0.1) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.22334942

The version-specific DOI above identifies the immutable v2.0.1 archive. The
all-versions DOI, https://doi.org/10.5281/zenodo.22334941, resolves to the most
recent archived release.
