# Output Examples

This directory contains example outputs from the Loop Extractor pipeline to help you understand the structure and content of generated files.

## Output Structure

When processing an audio file, the pipeline creates the following directory structure:

```
output_dir/
└── track_name/
    ├── 1_stems/                    # Separated audio stems
    │   ├── vocals.wav
    │   ├── drums.wav
    │   ├── bass.wav
    │   ├── piano.wav
    │   ├── other.wav
    │   ├── track_name.npz          # Combined stems data
    │   └── bass_f0.csv             # Bass pitch detection (F0)
    │
    ├── 2_beats/                    # Beat and tempo analysis
    │   ├── track_name_beats.txt    # Raw beat detection
    │   ├── track_name_downbeats_corrected.txt  # Corrected downbeats
    │   └── tempo_plots/
    │       ├── track_name_tempo_comparison.pdf  # Tempo visualization
    │       └── track_name_bar_tempos.csv        # Bar-by-bar tempo data
    │
    ├── 3_grid/                     # Grid alignment and timing analysis
    │   ├── track_name_comprehensive_phases.csv  # Complete timing data
    │   ├── track_name_onsets.csv               # Detected drum hits
    │   ├── track_name_raster_comparison.png    # Timing visualization
    │   └── track_name_rms_summary.json         # Timing accuracy metrics
    │
    ├── 4_audio_examples/          # Audio examples with click tracks
    │   ├── uncorrected.mp3        # Original timing + click
    │   ├── per_snippet.mp3        # Snippet correction + click
    │   ├── drum.mp3               # Drum-based correction + click
    │   ├── mel.mp3                # Melodic correction + click
    │   └── pitch.mp3              # Pitch-based correction + click
    │
    ├── 5_midi/                    # MIDI exports
    │   ├── onset/                 # Drum hit MIDI (detailed mode)
    │   │   ├── per_snippet.mid
    │   │   ├── drum.mid
    │   │   ├── mel.mid
    │   │   └── pitch.mid
    │   └── bass_pitch/            # Bass pitch MIDI (detailed mode)
    │       ├── per_snippet_bass.mid
    │       ├── drum_bass.mid
    │       ├── mel_bass.mid
    │       └── pitch_bass.mid
    │
    ├── 6_loops/                   # Extracted audio loops
    │   ├── drum/                  # Loops using drum-based pattern detection
    │   │   ├── vocals_loop.wav
    │   │   ├── drums_loop.wav
    │   │   ├── bass_loop.wav
    │   │   ├── piano_loop.wav
    │   │   └── other_loop.wav
    │   ├── mel/                   # Loops using melodic pattern detection
    │   └── pitch/                 # Loops using pitch pattern detection
    │
    └── pipeline_results.json      # Summary of pipeline execution

```

## DAW-Ready Mode

When using `--daw-ready` flag, the output is simplified:

```
output_dir/
└── track_name/
    ├── 1_stems/          # Same as above
    ├── 2_beats/          # Beats + tempo CSV only (no plots)
    ├── 3_grid/           # Grid data only (no plots)
    ├── 5_midi/           # Only drum method (no subfolders)
    │   ├── drum.mid
    │   └── drum_bass.mid
    └── 6_loops/          # Only drum method (no subfolders)
        ├── vocals_loop.wav
        ├── drums_loop.wav
        ├── bass_loop.wav
        ├── piano_loop.wav
        └── other_loop.wav
```

## File Descriptions

### Stems (1_stems/)
- **vocals.wav, drums.wav, bass.wav, piano.wav, other.wav**: Separated audio sources
- **bass_f0.csv**: Bass pitch over time (Hz), used for bass MIDI generation

### Beats (2_beats/)
- **beats.txt**: Raw beat detection (time, beat number, downbeat flag)
- **downbeats_corrected.txt**: Corrected bar boundaries
- **bar_tempos.csv**: Tempo for each bar (BPM)
- **tempo_comparison.pdf**: Visual comparison of tempo methods

### Grid (3_grid/)
- **comprehensive_phases.csv**: Complete timing analysis with:
  - Onset times (drum hits)
  - Grid positions for each correction method
  - Phase deviations from expected positions
- **onsets.csv**: Detected drum onset times
- **raster_comparison.png**: Visual comparison of timing across methods
- **rms_summary.json**: RMS timing error metrics

### Audio Examples (4_audio_examples/)
- MP3 files with click tracks showing different correction methods
- Useful for A/B comparison of timing accuracy

### MIDI (5_midi/)
- **onset/*.mid**: Drum hit MIDI files (actual onset times)
- **bass_pitch/*.mid**: Bass melody MIDI files (F0 converted to notes)
- Each file named by correction method (per_snippet, drum, mel, pitch)

### Loops (6_loops/)
- Perfectly looping audio stems
- Extracted based on detected pattern length
- Organized by pattern detection method (drum, mel, pitch)

## Key Metrics

The pipeline calculates timing accuracy using RMS (Root Mean Square) deviation:

- **Uncorrected RMS**: Timing error without correction
- **Per-snippet RMS**: Error with simple snippet-based correction
- **Drum/Mel/Pitch RMS**: Error with pattern-specific corrections

Lower RMS values indicate better timing alignment with the detected grid.

## Typical Use Cases

1. **Music Production**: Use loops in DAW for remixing
2. **Analysis**: Study microtiming and groove characteristics
3. **Research**: Quantitative analysis of rhythmic patterns
4. **Education**: Understand beat detection and timing correction

## Citation

When using output from this pipeline in research, please cite the relevant papers listed in the main README.
