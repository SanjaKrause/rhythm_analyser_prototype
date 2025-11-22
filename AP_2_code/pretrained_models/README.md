# Pretrained Models

This directory contains pretrained models required for the AP2 pipeline.

## Required Models

### Spleeter 5-stem Model
The Spleeter 5-stem separation model is required for stem separation (Step 1 of the pipeline).

**Download**: The model will be automatically downloaded by Spleeter on first use.

**Manual download** (if needed):
```bash
# The model should be placed in: pretrained_models/5stems/
# Spleeter will download it automatically to its cache directory on first run
# You can also manually download from: https://github.com/deezer/spleeter
```

**Model files** (automatically downloaded):
- `model.data-00000-of-00001` (187 MB)
- `model.index`
- `model.meta`
- `checkpoint`

### Beat-Transformer Model
The Beat-Transformer model checkpoint is referenced in `config.py` as `BEAT_TRANSFORMER_CHECKPOINT`.

**Location**: Configured in your environment
**Download**: Available from the Beat-Transformer repository

## Note
Model files are excluded from git due to their large size (>100 MB GitHub limit).
The pipeline will prompt you to download them if they're missing.
