# Beat Detection Module

⚠️  **ENVIRONMENT REQUIREMENT**: `new_beatnet_env`

This module requires the `new_beatnet_env` conda environment to function.

## Installation

```bash
# Create environment
conda env create -f ../environment_new_beatnet.yml

# Activate environment
conda activate new_beatnet_env
```

## Dependencies

- torch==2.2.2
- madmom==0.16.1
- beatnet==1.1.3
- librosa==0.10.2.post1
- numpy==1.20.3

## Usage

```python
# Make sure new_beatnet_env is activated!
# conda activate new_beatnet_env

from AP_2_code.beat_detection import transformer, processors

# TODO: Add usage examples
```

## Beat-Transformer

This module integrates the Beat-Transformer model from:
https://github.com/zhaojw1998/Beat-Transformer

The model performs beat and downbeat detection on 5-stem separated audio.

## TODO

- [ ] Extract transformer model interface from beat_transformer.ipynb
- [ ] Create processors module for beat/downbeat tracking
- [ ] Add batch processing functions
- [ ] Add checkpoint management utilities
