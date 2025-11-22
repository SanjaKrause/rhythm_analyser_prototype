# AP_2_code: Music Analysis Toolkit

A Python package for analyzing microtiming, tempo, and rhythm in music.
Extracted and refactored from AP2 analysis notebooks into reusable, well-documented modules.

## Overview

This package provides tools for:
- **Tempo analysis**: Global, snippet, and loop-based tempo calculations
- **RMS analysis**: Phase deviation histograms and grid alignment metrics
- **Microtiming analysis**: Deviation from expected grid positions
- **Raster plots**: Visualization of timing across bars and loops
- **Beat detection**: Integration with Beat-Transformer (separate environment)
- **Stem separation**: Integration with Spleeter (separate environment)

## Project Structure

```
AP_2_code/
├── __init__.py
├── README.md
├── analysis/              # Core analysis (base environment)
│   ├── __init__.py
│   ├── tempo.py          # Tempo calculation functions
│   ├── rms_grid_histograms.py  # RMS phase deviation analysis
│   ├── microtiming.py    # Microtiming analysis (TODO)
│   └── raster.py         # Raster plot generation (TODO)
├── beat_detection/        # Beat-Transformer (new_beatnet_env)
│   ├── __init__.py
│   └── README.md
├── stem_separation/       # Spleeter (AEinBOX env)
│   ├── __init__.py
│   └── README.md
├── utils/                 # Utility functions (base environment)
│   ├── __init__.py
│   ├── file_io.py        # File I/O utilities
│   └── data_processing.py # Data processing utilities
├── config/                # Configuration management
│   └── __init__.py
└── tests/                 # Unit tests
    └── __init__.py
```

## Installation

### Base Environment (for analysis modules)

```bash
cd /Users/SexySanja/mastab/MAIN_PROJECT/AP_2

# Install dependencies
pip install -r requirements.txt

# Or with conda
conda install numpy pandas matplotlib
```

### Beat Detection Environment (optional)

Only needed if using `beat_detection` module:

```bash
# Create environment from file
conda env create -f environment_new_beatnet.yml

# Or manually
conda create -n new_beatnet_env python=3.9
conda activate new_beatnet_env
pip install -r requirements_new_beatnet_env.txt
```

### Stem Separation Environment (optional)

Only needed if using `stem_separation` module:

```bash
# Create environment
conda create -n AEinBOX python=3.10
conda activate AEinBOX
pip install -r requirements_AEinBOX.txt
```

## Quick Start

### Tempo Analysis

```python
from AP_2_code.analysis import tempo

# Parse downbeats and calculate tempos
downbeats, time_sig = tempo.parse_corrected_downbeats('track_123_downbeats_corrected.txt')
df_bars = tempo.calculate_bar_tempos(downbeats, time_sig)

# Global tempo
global_tempo = tempo.calculate_global_tempo(df_bars)
print(f"Global tempo: {global_tempo:.2f} BPM")

# Snippet tempo
snippet_tempo = tempo.calculate_snippet_tempo(df_bars, snippet_start=10.0, snippet_end=40.0)
print(f"Snippet tempo: {snippet_tempo:.2f} BPM")

# Loop tempos
loop_tempos = tempo.calculate_loop_tempos(df_bars, pattern_len=4)
for loop in loop_tempos:
    print(f"Loop {loop['loop_index']}: {loop['avg_tempo_bpm']:.2f} BPM")
```

### RMS Analysis

```python
from AP_2_code.analysis import rms_grid_histograms
import pandas as pd

# Calculate RMS from comprehensive CSV
rms_values = rms_grid_histograms.calculate_rms_from_csv('track_123_comprehensive_phases.csv')

print(f"Uncorrected RMS: {rms_values['uncorrected_ms']:.2f} ms")
print(f"Per-snippet RMS: {rms_values['per_snippet_ms']:.2f} ms")
print(f"Drum method RMS: {rms_values['drum_ms']:.2f} ms")

# Calculate improvement
improvement = rms_grid_histograms.calculate_improvement_percentage(
    rms_values['uncorrected_ms'],
    rms_values['per_snippet_ms']
)
print(f"Improvement: {improvement:.1f}%")
```

### File I/O Utilities

```python
from AP_2_code.utils import file_io

# Find CSV files in directory
files = file_io.find_files_by_pattern(
    directory='/path/to/csvs',
    pattern=r'^(\d+)_comprehensive_phases\.csv$',
    start_id=0,
    end_id=100
)

# Load snippet offsets
offsets = file_io.load_snippet_offsets('corrected_shift_results.csv')

# Load pattern lengths
drum_lengths = file_io.load_pattern_length_map('drum_method_pattern_lengths.csv')
```

## Module Environments

| Module | Environment | Install Command |
|--------|-------------|----------------|
| `analysis.*` | Base | `pip install -r requirements.txt` |
| `utils.*` | Base | (included in base) |
| `beat_detection.*` | new_beatnet_env | `conda env create -f environment_new_beatnet.yml` |
| `stem_separation.*` | AEinBOX | See stem_separation/README.md |

## Development

### Adding New Functions

1. Choose appropriate module based on functionality
2. Add function with complete docstring (parameters, returns, examples)
3. Use type hints for all parameters
4. Add unit tests in `tests/`

### Documentation Style

We use NumPy-style docstrings:

```python
def my_function(param1: float, param2: str = 'default') -> bool:
    """
    Brief description.

    Longer description if needed.

    Parameters
    ----------
    param1 : float
        Description of param1
    param2 : str, default='default'
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> my_function(1.0, 'test')
    True
    """
    pass
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_tempo.py

# Run with coverage
pytest --cov=AP_2_code tests/
```

## Environment Reference

For detailed environment setup and troubleshooting, see:
- [ENVIRONMENTS.md](../AP_2/ENVIRONMENTS.md) - Complete environment guide
- [beat_detection/README.md](beat_detection/README.md) - Beat detection setup
- [stem_separation/README.md](stem_separation/README.md) - Stem separation setup

## Related Notebooks

Original Jupyter notebooks (in `../AP_2/`):
- `AP2_tempo_calculations.ipynb` → `analysis/tempo.py`
- `AP2_create_rasterHistograms.ipynb` → `analysis/rms_grid_histograms.py`
- `AP2_plot_microtimings_final.ipynb` → `analysis/microtiming.py` (TODO)
- `AP2_plot_raster_gridshift.ipynb` → `analysis/raster.py` (TODO)
- `beat_transformer.ipynb` → `beat_detection/` (TODO)
- `AP2_5stemSeperation.ipynb` → `stem_separation/` (TODO)

## Contributing

1. Keep functions focused and single-purpose
2. Add comprehensive docstrings with examples
3. Use type hints
4. Write unit tests
5. Follow PEP 8 style guide

## License

Part of the AP2 Analysis Project

## TODO

### High Priority
- [ ] Complete `microtiming.py` module from AP2_plot_microtimings_final.ipynb
- [ ] Complete `raster.py` module from AP2_plot_raster_gridshift.ipynb
- [ ] Add plotting functions to `rms_grid_histograms.py`

### Medium Priority
- [ ] Extract Beat-Transformer interface from beat_transformer.ipynb
- [ ] Extract Spleeter interface from AP2_5stemSeperation.ipynb
- [ ] Add comprehensive unit tests
- [ ] Add example scripts/notebooks

### Low Priority
- [ ] Add configuration management module
- [ ] Add logging utilities
- [ ] Add progress bars for batch processing
- [ ] Create CLI interface for common tasks

## Contact

For questions about this codebase, refer to the original AP2 notebooks or documentation.
