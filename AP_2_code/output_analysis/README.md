# Analysis Output

This directory contains all output from the AP2 pipeline.

## Directory Structure

Each processed track gets its own subdirectory with the following structure:

```
output_analysis/
└── {track_id}/
    ├── 1_stems/
    │   ├── vocals.wav
    │   ├── drums.wav
    │   ├── bass.wav
    │   ├── piano.wav
    │   ├── other.wav
    │   └── {track_id}_5stems.npz          # Mel-spectrograms for beat detection
    │
    ├── 2_beats/
    │   └── {track_id}_output.txt          # Beat and downbeat times
    │
    ├── 3_corrected/
    │   ├── {track_id}_downbeats_corrected.txt   # Corrected downbeats
    │   ├── {track_id}_downbeat_tempos.csv       # Tempo analysis (optional)
    │   └── {track_id}_tempo_plots.pdf           # Tempo plots (optional)
    │
    ├── 4_onsets/
    │   └── {track_id}_onsets.csv          # Detected onsets from drum stem
    │
    ├── 5_grid/
    │   └── {track_id}_comprehensive_phases.csv  # All phase calculations
    │
    ├── 6_rms/
    │   └── {track_id}_rms_summary.json    # RMS timing deviation metrics
    │
    ├── 7_audio_examples/
    │   ├── uncorrected.mp3                # Click track - uncorrected (raw downbeats)
    │   ├── per_snippet.mp3                # Click track - per-snippet method
    │   ├── drum.mp3                       # Click track - drum loop method
    │   ├── mel.mp3                        # Click track - melody loop method
    │   ├── pitch.mp3                      # Click track - pitch loop method
    │   └── original.mp3                   # Original snippet without click track
    │
    └── pipeline_results.json              # Complete pipeline summary
```

## Key Files

### 1_stems/
- **5 stem WAV files**: Separated audio (vocals, drums, bass, piano, other)
- **NPZ file**: Mel-spectrograms used for beat detection

### 2_beats/
- **output.txt**: Beat and downbeat positions from Beat-Transformer

### 3_corrected/
- **downbeats_corrected.txt**: Corrected bar positions (fixes double-time/half-time errors)

### 4_onsets/
- **onsets.csv**: Detected onset times from drum stem (used for microtiming analysis)

### 5_grid/
- **comprehensive_phases.csv**: Complete phase calculations for all methods
  - Contains: bar numbers, tick positions, onset times, phases, grid times
  - Multiple correction methods: uncorrected, per-snippet, drum loop, melody loop, pitch loop

### 6_rms/
- **rms_summary.json**: Root Mean Square timing deviations
  - Quantitative measure of timing precision for each method

### 7_audio_examples/
- **MP3 files with click tracks**: Audible comparison of different correction methods
  - Useful for qualitative assessment of which method sounds most accurate

### pipeline_results.json
- Complete summary of pipeline execution
- Includes all file paths, parameters, and processing stats
- Useful for batch processing and error tracking

## Processing Status

To check if a track was fully processed, look for:
1. All 7 subdirectories exist
2. `pipeline_results.json` has no errors
3. Audio examples were generated (if not disabled with `--no-audio-examples`)
