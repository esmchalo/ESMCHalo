# Historical ESMC asset acquisition and verification

The accepted historical model is ESMC-600M with 36 transformer layers and hidden width 1152.

The expected asset SHA256 values are stored in `src/HISTORICAL_MODEL_SHA256.json`.

The accepted `model.safetensors` SHA256 is:

`e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8`

Historical provenance records identify:

`https://huggingface.co/biohub/ESMC-600M`

and revision:

`a7e82012c83126b9eedb055fea9fa84b6c02f094`

These remain provenance references.

For v2.0.2-rc2, clean installation of the public inference backend was verified using:

- Python 3.12
- `esm==3.4.1.post1`
- `torch==2.11.0`
- `transformers==4.57.6`
- NVIDIA GeForce RTX 4090
- PyTorch CUDA 13.0

The accepted model was loaded using:

- `esm.models.esmc.EsmcForMaskedLM`
- `esm.models.esmc.EsmcTokenizer`

To check a local model directory:

```bash
python scripts/check_model_files.py --model-dir YOUR_DIRECTORY
```

Do not proceed if any expected hash differs.

Do not rename another checkpoint, bypass checksum verification, or silently substitute another ESMC model.

Third-party ESMC weights are not bundled or relicensed by ESMCHalo.

## Remaining limitation

The rc2 acceptance verifies the locked local model bytes, clean installation of the public `esm` backend, short-reference numerical regression and boundary behavior.

It does not establish that a newly downloaded public snapshot has independently been proven byte-for-byte identical to the historical accepted local asset.
