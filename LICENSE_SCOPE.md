# License scope

This repository uses separate licenses for software and non-code artifacts.

## MIT License

The root `LICENSE` applies to the original ESMCHalo source code, including:

- `src/**/*.py`
- `artifacts/frozen_ensemble/inference.py`
- verification and inference utilities authored for ESMCHalo

## Creative Commons Attribution 4.0 International

The following original ESMCHalo model and documentation artifacts are made
available under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
license:

- `artifacts/frozen_ensemble/members/*.pt`
- `artifacts/frozen_ensemble/final_calibrator.json`
- `artifacts/frozen_ensemble/final_threshold.json`
- `evidence/**`
- `tests/synthetic_frozen_inference/**`
- `README.md`, `MODEL_CARD.md` and project-authored documentation

Required attribution: “ESMCHalo contributors, ESMCHalo v2.0.0,” together with
the GitHub repository URL and the version-specific Zenodo DOI when available.

## Exclusions

This licensing statement does not grant rights to third-party software,
datasets or pretrained model weights. ESMC pretrained weights and HPClas source
data are not redistributed in this repository and remain governed by their
respective providers' terms.
