# AP2 Music Microtiming Analysis Pipeline

Complete 7-step pipeline for analyzing microtiming deviations in music.

## Pipeline Overview

```
1. Stem Separation (Spleeter)
   └─> 5 stems: vocals, drums, bass, piano, other
   └─> mel-spectrograms saved to NPZ

2. Beat Detection (Beat-Transformer + madmom)
   └─> Runs in subprocess (new_beatnet_env)
   └─> Detects beats and downbeats

3. Downbeat Correction
   └─> Algorithmic correction of double-time/half-time errors
   └─> No manual intervention required

4. Onset Detection (librosa)
   └─> Detects onsets from drum stem
   └─> Optimized settings: hop=512, delta=0.12, refinement enabled

5. Raster/Grid Calculations
   └─> Multiple correction methods:
       - Uncorrected (raw bar grid)
       - Per-snippet (global offset correction)
       - Loop-based (drum/mel/pitch - equidistant grids)
   └─> Calculates phases and grid times

6. RMS Analysis
   └─> Root Mean Square of timing deviations
   └─> Calculates RMS for all correction methods

7. Audio Examples
   └─> Generates MP3s with click tracks
   └─> One MP3 per correction method
```

## Environment Setup

**Main Environment:** `AEinBOX_13_3`
- Runs steps 1, 3, 4, 5, 6, 7
- Packages: spleeter, librosa, numpy, pandas, matplotlib

**Subprocess Environment:** `new_beatnet_env`
- Runs step 2 only
- Packages: PyTorch, madmom, beattransformer

## Usage

### Basic Usage

```bash
# Place audio in input_wav/ folder, then:
python main.py --audio input_wav/track.wav --track-id 123 --output-dir output_analysis/
```

### With Custom Files

```bash
python main.py --audio input_wav/track.wav --track-id 123 \
    --onset-file onsets/123_onsets.csv \
    --pattern-file pattern_lengths.csv \
    --snippet-file snippet_offsets.csv \
    --output-dir output_analysis/
```

### Options

- `--no-skip`: Reprocess even if output exists
- `--no-audio-examples`: Skip MP3 generation
- `--quiet`: Minimize output

## Output Structure

```
output_analysis/
└── {track_id}/
    ├── 1_stems/
    │   ├── vocals.wav
    │   ├── drums.wav
    │   ├── bass.wav
    │   ├── piano.wav
    │   ├── other.wav
    │   └── {track_id}_5stems.npz
    ├── 2_beats/
    │   └── {track_id}_output.txt
    ├── 3_corrected/
    │   └── {track_id}_downbeats_corrected.txt
    ├── 4_onsets/
    │   └── {track_id}_onsets.csv
    ├── 5_grid/
    │   └── {track_id}_comprehensive_phases.csv
    ├── 6_rms/
    │   └── {track_id}_rms_summary.json
    ├── 7_audio_examples/
    │   ├── per_snippet.mp3
    │   ├── drum_L4.mp3
    │   ├── mel_L4.mp3
    │   └── pitch_L4.mp3
    └── pipeline_results.json
```

## Module Structure

```
AP_2_code/
├── main.py                         # Pipeline orchestrator
├── config.py                       # Centralized configuration
├── stem_separation/
│   ├── __init__.py
│   └── spleeter_interface.py       # Spleeter wrapper
├── beat_detection/
│   ├── __init__.py
│   ├── run_transformer.py          # Standalone script (subprocess)
│   └── transformer.py              # Subprocess wrapper
├── analysis/
│   ├── __init__.py
│   ├── tempo.py                    # Tempo calculations
│   ├── microtiming.py              # Microtiming analysis
│   ├── correct_bars.py             # Downbeat correction
│   ├── onset_detection.py          # Onset detection (NEW)
│   ├── raster.py                   # Grid/phase calculations
│   └── rms_grid_histograms.py      # RMS analysis
└── utils/
    ├── __init__.py
    └── audio_export.py             # MP3 generation
```

## Key Features

### Automatic Onset Detection
- If no onset file is provided, automatically detects from drum stem
- Uses drum-optimized settings from notebook analysis
- Refinement and duplicate filtering included

### Multiple Correction Methods
- **Uncorrected**: Raw bar grid (baseline)
- **Per-snippet**: Global offset correction
- **Loop-based**: Equidistant grids for different pattern lengths
  - Drum loop (typically 4 bars)
  - Melody loop (typically 4 bars)
  - Pitch loop (typically 4 bars)

### Grid-Shift Approach
- Shifts grid forward by reference offset
- Maintains original onset times
- Avoids negative phase values

### Skip Existing Files
- By default, skips steps if output already exists
- Use `--no-skip` to force reprocessing
- Useful for iterative development

## Testing

Each module includes test functions:

```bash
# Test onset detection
python analysis/onset_detection.py --test

# Test raster calculations
python -c "from analysis.raster import test_basic_functions, test_phase_calculations; test_basic_functions(); test_phase_calculations()"

# Test correct_bars
python -c "from analysis.correct_bars import test_functions; test_functions()"
```

## Development Notes

- All imports use `importlib.util` pattern to avoid conflicts with config/ directory
- Type hints used throughout for clarity (e.g., `Optional[str]`)
- Comprehensive error handling and reporting
- Verbose output mode for debugging
- JSON results file for programmatic access

## Next Steps

- Add batch processing capability
- Create GUI wrapper
- Add visualization tools for phases/grids
- Implement parallel processing for multiple tracks

---

Generated: 2025-11-18
Version: 1.0
