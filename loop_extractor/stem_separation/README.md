# Stem Separation Module

⚠️  **ENVIRONMENT REQUIREMENT**: `AEinBOX`

This module requires the `AEinBOX` conda environment to function.

## Installation

```bash
# Create environment
conda create -n AEinBOX python=3.10

# Activate environment
conda activate AEinBOX

# Install dependencies
pip install -r ../requirements_AEinBOX.txt
```

## Dependencies

- spleeter (5-stem separation)
- tensorflow
- librosa
- ffmpeg-python

## Usage

```python
# Make sure AEinBOX is activated!
# conda activate AEinBOX

from AP_2_code.stem_separation import spleeter_interface

# TODO: Add usage examples
```

## Spleeter

This module uses Spleeter for 5-stem audio separation:
- vocals
- drums
- bass
- piano
- other

## Output Format

Separated stems are saved as:
- Individual WAV files (vocals.wav, drums.wav, etc.)
- Mel-spectrogram NPZ file (song_5stems.npz) for Beat-Transformer

## TODO

- [ ] Extract spleeter interface from AP2_5stemSeperation.ipynb
- [ ] Create batch processing functions
- [ ] Add audio preprocessing utilities
- [ ] Add mel-spectrogram generation utilities
