# Historical ESMC asset acquisition and verification

Accepted local model: ESMC-600M, 36 layers, width 1152, native ESMCForMaskedLM, loaded as BF16. Expected asset SHA256 values are in `src/HISTORICAL_MODEL_SHA256.json`. The accepted weight hash is `e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8`.

Historical source records identify https://huggingface.co/biohub/ESMC-600M and the earlier public wrapper specifies revision `a7e82012c83126b9eedb055fea9fa84b6c02f094`. These are provenance leads, not a newly verified download guarantee. The supplied server record confirms local bytes, not the remote revision from which each byte originated. Online verification of the exact remote weight identity was not completed during packaging.

To acquire a candidate snapshot, use the provider's download interface, comply with its access/license terms, retain the snapshot revision, and obtain the five named files in the checksum JSON. Run `python scripts/check_model_files.py --model-dir YOUR_DIRECTORY`. Do not proceed on a mismatch; do not rename a different checkpoint or disable the check. Matching files still require a compatible native backend and numerical acceptance. We intentionally do not provide an unverified “one-click download and reproduce” command.

For the authors, the already verified local directory can be supplied via `--model-dir`; its private absolute path is not built into the public predictor. External acquisition of that exact snapshot and clean native-backend installation remain release limitations. A complete portable inference claim requires these to be resolved; no further training or external dataset is implied.

Third-party weights are not included and are not relicensed by ESMCHalo.
