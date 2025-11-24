# Loop Extractor Pipeline - Quick Start Guide

## Directory Structure

```
loop_extractor/
├── input_wav/              # Place your audio files here
├── output_analysis/        # Pipeline output organized by track ID
├── main.py                 # Main pipeline script
├── config.py              # Configuration settings
└── [modules...]
```

## Setup

1. **Activate the loop extractor environment:**
   ```bash
   conda activate loop_extractor_main
   ```

2. **Place audio file in input folder:**
   ```bash
   cp /path/to/your/track.wav input_wav/
   ```

## Usage

### Basic Usage

```bash
python main.py \
    --audio input_wav/track.wav \
    --track-id my_track_001 \
    --output-dir output_analysis/
```

### What This Does

The pipeline will create a complete analysis in 7 steps:

```
output_analysis/my_track_001/
├── 1_stems/                 # ← 5 separated stems + mel-spectrograms
├── 2_beats/                 # ← Beat and downbeat detection
├── 3_corrected/            # ← Corrected bar positions
├── 4_onsets/               # ← Onset detection from drums ✨ NEW
├── 5_grid/                 # ← Phase calculations (all methods)
├── 6_rms/                  # ← Timing deviation metrics
├── 7_audio_examples/       # ← MP3s with click tracks
└── pipeline_results.json   # ← Complete summary
```

### Advanced Options

```bash
# Skip audio example generation (faster)
python main.py \
    --audio input_wav/track.wav \
    --track-id my_track_001 \
    --output-dir output_analysis/ \
    --no-audio-examples

# Force reprocessing (ignore existing files)
python main.py \
    --audio input_wav/track.wav \
    --track-id my_track_001 \
    --output-dir output_analysis/ \
    --no-skip

# Quiet mode (minimal output)
python main.py \
    --audio input_wav/track.wav \
    --track-id my_track_001 \
    --output-dir output_analysis/ \
    --quiet
```

### Custom Parameters

```bash
# Provide custom pattern lengths and snippet offset
python main.py \
    --audio input_wav/track.wav \
    --track-id my_track_001 \
    --output-dir output_analysis/ \
    --pattern-file config/pattern_lengths.csv \
    --snippet-file config/snippet_offsets.csv
```

## Expected Output

### Console Output

```
================================================================================
Loop Extractor Pipeline - Track my_track_001
================================================================================

[1/7] Stem separation (Spleeter)...
  ✓ Stems and NPZ created

[2/7] Beat detection (Beat-Transformer)...
  ✓ Beat detection completed

[3/7] Downbeat correction...
  ✓ Corrected: 48 → 47 bars

[4/7] Onset detection from drum stem...
  ✓ Detected 1247 onsets
  ✓ Saved to: output_analysis/my_track_001/4_onsets/my_track_001_onsets.csv

[5/7] Raster/grid calculations...
  ✓ Comprehensive CSV created

[6/7] RMS histogram analysis...
  ✓ RMS calculated:
    Uncorrected: 45.23ms
    Per-snippet: 12.34ms
    Drum method: 8.91ms

[7/7] Audio examples...
  ✓ Audio examples created

================================================================================
Pipeline Summary
================================================================================
Track ID: my_track_001
Steps completed: 7
  ✓ stem_separation
  ✓ beat_detection
  ✓ correct_bars
  ✓ onset_detection
  ✓ raster
  ✓ rms_analysis
  ✓ audio_examples

✓ Pipeline completed successfully!
```

### Key Output Files

1. **`5_grid/{track_id}_comprehensive_phases.csv`**
   - Main analysis results
   - All phase calculations for every tick
   - Multiple correction methods

2. **`6_rms/{track_id}_rms_summary.json`**
   - Quantitative timing precision metrics
   - RMS values for each correction method

3. **`7_audio_examples/*.mp3`**
   - Listen to different correction methods
   - Compare timing accuracy audibly

4. **`pipeline_results.json`**
   - Complete processing summary
   - All file paths and parameters
   - Error tracking

## Troubleshooting

### Environment Issues

If beat detection fails:
```bash
# Check subprocess environment
/Users/SexySanja/miniconda3/envs/new_beatnet_env/bin/python --version
```

### Missing Beat-Transformer Checkpoint

Update `config.py` if checkpoint path is different:
```python
BEAT_TRANSFORMER_CHECKPOINT = Path("/path/to/checkpoint/fold_4_trf_param.pt")
```

### Processing Time

Typical processing time for a 3-minute track:
- Step 1 (Stems): ~30-60 seconds
- Step 2 (Beats): ~10-20 seconds
- Step 3 (Correct): <1 second
- Step 4 (Onsets): ~1-2 seconds
- Step 5 (Grid): ~1-2 seconds
- Step 6 (RMS): <1 second
- Step 7 (Audio): ~5-10 seconds

**Total: ~1-2 minutes per track**

## Next Steps

1. **Analyze results:** Open `5_grid/{track_id}_comprehensive_phases.csv` in Excel/pandas
2. **Listen to examples:** Play MP3s in `7_audio_examples/`
3. **Check metrics:** Review `6_rms/{track_id}_rms_summary.json`
4. **Visualize:** Create plots using the comprehensive CSV data

## Getting Help

- Full documentation: [PIPELINE.md](PIPELINE.md)
- Input folder help: [input_wav/README.md](input_wav/README.md)
- Output structure: [output_analysis/README.md](output_analysis/README.md)

---

**Environment:** AEinBOX_13_3 (main) + new_beatnet_env (subprocess)
**Version:** 1.0
**Updated:** 2025-11-18
