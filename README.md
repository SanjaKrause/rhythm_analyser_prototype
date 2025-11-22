# Loop Extractor Python

Standalone loop extraction tool based on the AP_2 pipeline.

## Directory Structure

```
loop_extractor_python/
├── AP_2_code/              # Main pipeline code
│   ├── main.py             # Entry point
│   ├── config.py           # Configuration
│   ├── beat_detection/     # Beat detection module
│   ├── stem_separation/    # Spleeter integration
│   ├── analysis/           # Pattern detection
│   └── utils/              # Utilities
│
└── Beat-Transformer/       # Beat detection model
    ├── code/               # Model code (DilatedTransformer)
    └── checkpoint/         # Pre-trained weights (36MB)
        └── fold_4_trf_param.pt
```

## Setup

1. Create conda environments (see main_project/ENVIRONMENTS.md)
2. Activate main environment: `conda activate loop_extractor_main`
3. Run pipeline: `cd AP_2_code && python main.py --audio input.wav --track-id 1 --output-dir output/`

## Configuration

All paths have been updated to use the new location:
- `/Users/alexk/mastab/loop_extractor_python/`

Check `AP_2_code/config.py` for all configuration options.

## Requirements

- Beat-Transformer checkpoint (36MB) - already included
- Two conda environments:
  - loop_extractor_main (main pipeline)
  - new_beatnet_env (beat detection subprocess)
