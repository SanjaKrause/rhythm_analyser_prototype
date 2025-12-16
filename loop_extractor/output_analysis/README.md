# Analysis Output

This directory contains all output from the Loop Extractor pipeline.

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
    │   ├── bass_f0.csv                        # Bass pitch tracking (F0 frequencies)
    │   └── {track_id}_5stems.npz              # Mel-spectrograms for beat detection
    │
    ├── 2_beats/
    │   └── {track_id}_output.txt              # Beat and downbeat times
    │
    ├── 3_corrected/
    │   ├── {track_id}_downbeats_corrected.txt # Corrected downbeats
    │   ├── {track_id}_downbeat_tempos.csv     # Tempo analysis (optional)
    │   └── {track_id}_tempo_plots.pdf         # Tempo plots (optional)
    │
    ├── 3.5_tempo_plots/
    │   ├── {track_id}_tempo_plots.pdf         # 8-panel tempo comparison plot
    │   └── {track_id}_bar_tempos.csv          # Bar-by-bar tempo data
    │
    ├── 4_onsets/
    │   └── {track_id}_onsets.csv              # Detected onsets from drum stem
    │
    ├── 5_grid/
    │   ├── {track_id}_comprehensive_phases.csv      # All phase calculations
    │   ├── {track_id}_raster_comparison.png         # Raster plot (drum/mel/pitch methods)
    │   └── {track_id}_raster_standard.png           # Raster plot (standard L=1/2/4 methods)
    │
    ├── 6_rms/
    │   └── {track_id}_rms_summary.json        # RMS timing deviation metrics
    │
    ├── 7_audio_examples/
    │   ├── uncorrected.mp3                    # Click track - uncorrected (raw downbeats)
    │   ├── per_snippet.mp3                    # Click track - per-snippet method
    │   ├── drum.mp3                           # Click track - drum loop method
    │   ├── mel.mp3                            # Click track - melody loop method
    │   ├── pitch.mp3                          # Click track - pitch loop method
    │   ├── standard_L1.mp3                    # Click track - standard 1-bar loop method
    │   ├── standard_L2.mp3                    # Click track - standard 2-bar loop method
    │   ├── standard_L4.mp3                    # Click track - standard 4-bar loop method
    │   └── original.mp3                       # Original snippet without click track
    │
    ├── 8_midi/                                # MIDI export (detailed analysis mode)
    │   ├── onset/                             # Onset-based MIDI (drum hits)
    │   │   ├── per_snippet.mid
    │   │   ├── drum.mid
    │   │   ├── mel.mid
    │   │   ├── pitch.mid
    │   │   ├── standard_L1.mid
    │   │   ├── standard_L2.mid
    │   │   └── standard_L4.mid
    │   └── bass_pitch/                        # Bass pitch MIDI (F0 converted to notes)
    │       ├── per_snippet.mid
    │       ├── drum.mid
    │       ├── mel.mid
    │       ├── pitch.mid
    │       ├── standard_L1.mid
    │       ├── standard_L2.mid
    │       └── standard_L4.mid
    │
    ├── 8_midi_drum/                           # MIDI export (DAW ready mode)
    │   ├── drum.mid                           # Onset-based MIDI (drum method only)
    │   └── drum_bass_pitch.mid                # Bass pitch MIDI (drum method only)
    │
    ├── 9_loops/                               # Stem loops (detailed analysis mode)
    │   ├── per_snippet/
    │   │   ├── vocals.wav
    │   │   ├── drums.wav
    │   │   ├── bass.wav
    │   │   ├── piano.wav
    │   │   └── other.wav
    │   ├── drum/
    │   │   ├── vocals.wav
    │   │   ├── drums.wav
    │   │   ├── bass.wav
    │   │   ├── piano.wav
    │   │   └── other.wav
    │   ├── mel/
    │   │   └── ... (same as above)
    │   ├── pitch/
    │   │   └── ... (same as above)
    │   ├── standard_L1/
    │   │   └── ... (same as above)
    │   ├── standard_L2/
    │   │   └── ... (same as above)
    │   └── standard_L4/
    │       └── ... (same as above)
    │
    ├── 9_loops_drum/                          # Stem loops (DAW ready mode)
    │   ├── vocals.wav                         # Drum method only
    │   ├── drums.wav
    │   ├── bass.wav
    │   ├── piano.wav
    │   └── other.wav
    │
    └── pipeline_results.json                  # Complete pipeline summary
```

---

## Detailed Subfolder Documentation

### 1_stems/

**Purpose**: Audio stem separation using Spleeter and mel-spectrogram generation for beat detection.

**Files**:
- **`vocals.wav`**: Isolated vocal track
- **`drums.wav`**: Isolated drum track (used for onset detection)
- **`bass.wav`**: Isolated bass track (used for pitch detection)
- **`piano.wav`**: Isolated piano/keyboard track
- **`other.wav`**: All other instruments (guitars, synths, etc.)
- **`bass_f0.csv`**: Bass pitch tracking data (fundamental frequency in Hz)
  - Columns: `time(s)`, `f0(Hz)`, `confidence`
  - Generated using libf0 pitch detection library
  - Used for bass pitch MIDI export
- **`{track_id}_5stems.npz`**: Mel-spectrograms for all 5 stems
  - Binary NumPy archive containing mel-spectrograms
  - Used as input for Beat-Transformer model
  - Parameters: 44.1kHz SR, 4096 FFT, 1024 hop, 128 mel bins, 30-11000 Hz

**Technical Details**:
- Separation model: Spleeter 5-stem (pre-trained by Deezer)
- Mel-spectrogram parameters configured in [config.py:58-63](../config.py#L58-L63)
- Stems are saved as 16-bit WAV files at original sample rate

**Usage**:
- Import stems directly into your DAW for remixing
- Use drum stem for rhythm analysis
- Use bass stem for harmonic analysis
- Mel-spectrograms are automatically passed to beat detection (not for user use)

---

### 2_beats/

**Purpose**: Raw beat and downbeat detection from Beat-Transformer model.

**Files**:
- **`{track_id}_output.txt`**: Beat and downbeat positions
  - Format: `time(s) beat_activation downbeat_activation beat_pos bar_number`
  - `beat_pos`: Position within bar (1 = downbeat, 2-4 = other beats)
  - `bar_number`: Sequential bar count
  - Beat-Transformer uses multi-head attention with 5-stem mel-spectrograms

**Technical Details**:
- Model: Beat-Transformer (9-layer dilated transformer)
- Post-processing: Madmom Dynamic Bayesian Network (DBN)
- DBN parameters: 55-215 BPM range, supports 3/4 and 4/4 time signatures
- Configuration: [config.py:68-91](../config.py#L68-L91)

**Important**: This file contains RAW detections and may have double-time/half-time errors. Use the corrected version from `3_corrected/` for analysis.

---

### 3_corrected/

**Purpose**: Corrected downbeat positions after fixing tempo errors.

**Files**:
- **`{track_id}_downbeats_corrected.txt`**: Corrected downbeat times
  - Format: Tab-separated with headers
  - Columns: `bar_number`, `corrected_downbeat_time(s)`, `next_downbeat_time(s)`, `bar_duration(s)`, `bar_tempo(BPM)`
  - Header includes: `# time_signature = 4` (or 3)
  - Header includes: `# dominant_tempo = 120.5` (BPM)
  - This is the **primary reference** for all downstream timing analysis

- **`{track_id}_downbeat_tempos.csv`** (optional, legacy):
  - Similar to corrected file but different format
  - May not be generated in newer pipeline versions

- **`{track_id}_tempo_plots.pdf`** (optional, legacy):
  - May contain older tempo visualizations
  - Superseded by `3.5_tempo_plots/`

**Technical Details**:
- Correction algorithm detects factor-of-two tempo errors (double/half/quadruple time)
- Uses ±10% tolerance for "usable" tempo classification
- BPM threshold: 135 (configurable in [config.py:99](../config.py#L99))
- Algorithm: [analysis/correct_bars.py](../analysis/correct_bars.py)

**Why Correction is Needed**:
Beat detection models sometimes misidentify the downbeat periodicity:
- **Half-time error**: Detects every 2 bars as 1 bar → tempo appears doubled
- **Double-time error**: Detects 2 downbeats per bar → tempo appears halved
- Correction algorithm uses median tempo and factor-of-two matching to fix these

---

### 3.5_tempo_plots/

**Purpose**: Visual comparison of tempo stability before and after correction.

**Files**:
- **`{track_id}_tempo_plots.pdf`**: 8-panel tempo comparison plot
  - **Panels 1-4**: Uncorrected tempo analysis
    - Panel 1: Full song tempo over time (bar-by-bar)
    - Panel 2: Snippet tempo over time (zoomed to analysis window)
    - Panel 3: Full song tempo histogram
    - Panel 4: Snippet tempo histogram
  - **Panels 5-8**: Corrected tempo analysis (same structure)
  - Shows tempo stability, outliers, and correction effectiveness

- **`{track_id}_bar_tempos.csv`**: Bar-by-bar tempo data
  - Columns: `bar_number`, `downbeat_time(s)`, `bar_duration(s)`, `tempo(BPM)`, `corrected`
  - `corrected`: Boolean indicating if bar was adjusted during correction
  - Used for tempo-aware MIDI export and loop extraction

**Technical Details**:
- Tempo calculated from inter-bar intervals: `tempo = (60 / interval) × time_signature`
- Snippet window: Configurable via `--manual-start` and `--manual-duration`
- Default snippet: 30 seconds (configurable in [config.py:101](../config.py#L101))

**Interpretation**:
- **Tight tempo histogram**: Good detection accuracy
- **Multiple peaks**: Indicates double/half-time errors (check if correction fixed them)
- **Scattered time series**: Unstable tempo or detection errors
- Compare uncorrected vs corrected plots to verify correction worked

**Note**: Plots are skipped in `--daw-ready` mode (only CSV is generated).

---

### 4_onsets/

**Purpose**: Detected onset times from drum stem for microtiming analysis.

**Files**:
- **`{track_id}_onsets.csv`**: Drum onset times
  - Single column: `onset_time(s)`
  - Detected using librosa's onset detection with backtracking
  - Minimum interval: 0.15s (prevents duplicate detections)
  - Parameters: 512 hop length, 22050 Hz SR, delta=0.12

**Technical Details**:
- Detection method: Spectral flux with high-frequency content
- Backtracking disabled (uses peak onset times, not energy rise)
- Refinement disabled (uses frame-accurate times)
- Algorithm: [analysis/onset_detection.py](../analysis/onset_detection.py)

**Usage**:
- These onsets are matched to the nearest grid positions in Step 5
- Phase deviations (onset time - grid time) quantify microtiming
- Only onsets within bar boundaries are used for analysis

---

### 5_grid/

**Purpose**: Comprehensive phase calculations for all correction methods and visualization.

**Files**:
- **`{track_id}_comprehensive_phases.csv`**: Core timing analysis data
  - **Basic columns**:
    - `onset_time(s)`: Original onset time from drum stem
    - `bar`: Bar number containing this onset
    - `tick_uncorrected`: Grid position within bar (0-15 for 16th note grid)

  - **Phase columns** (for each method):
    - `phase_uncorrected`: Timing deviation in phase units (-0.5 to +0.5, where 1.0 = one grid step)
    - `phase_per_snippet`: Per-snippet correction (4-bar snippet, 1 reference offset)
    - `phase_drum(L=N)`: Drum method (auto-detected pattern length N)
    - `phase_mel(L=N)`: Melody method (auto-detected pattern length N)
    - `phase_pitch(L=N)`: Pitch method (auto-detected pattern length N)
    - `phase_standard_L1(L=1)`: Standard 1-bar loop correction
    - `phase_standard_L2(L=2)`: Standard 2-bar loop correction
    - `phase_standard_L4(L=4)`: Standard 4-bar loop correction

  - **Grid time columns** (for each method):
    - `grid_time_uncorrected(s)`: Uncorrected grid position in seconds
    - `grid_time_per_snippet(s)`: Per-snippet corrected grid position
    - `grid_time_drum(L=N)(s)`: Drum method grid position
    - ... (similar for mel, pitch, standard_L1, standard_L2, standard_L4)

  - **Phase in milliseconds** (for each method):
    - `phase_uncorrected(ms)`: Phase deviation in milliseconds
    - `phase_per_snippet(ms)`: Per-snippet phase in milliseconds
    - ... (similar for all methods)

- **`{track_id}_raster_comparison.png`**: 5-panel raster plot comparing methods
  - **Panel 1**: Uncorrected (raw beat-transformer downbeats)
  - **Panel 2**: Per-snippet correction (4-bar snippet, single offset)
  - **Panel 3**: Drum method (auto-detected drum pattern length)
  - **Panel 4**: Melody method (auto-detected melodic pattern length)
  - **Panel 5**: Pitch method (auto-detected bass pitch pattern length)
  - Each panel shows phase deviations (y-axis) across bars (x-axis)
  - RMS values displayed in panel titles

- **`{track_id}_raster_standard.png`**: 5-panel raster plot for standard loop methods
  - **Panel 1**: Uncorrected (same as above)
  - **Panel 2**: Standard L=1 (1-bar loop correction)
  - **Panel 3**: Standard L=2 (2-bar loop correction)
  - **Panel 4**: Standard L=4 (4-bar loop correction)
  - **Panel 5**: Per-snippet (same as comparison plot)

**Technical Details**:
- **Grid resolution**: 16 positions per bar (16th note grid)
- **Phase calculation**: `phase = (onset_time - grid_time) / grid_step`
- **Tick assignment**: Onset matched to nearest grid position using argmin
- **Correction methods**:
  - **Uncorrected**: Uses raw downbeats from Beat-Transformer
  - **Per-snippet**: Single offset correction for 4-bar snippet
  - **Drum/Mel/Pitch**: Pattern-based correction with auto-detected loop length (L)
  - **Standard L=1/2/4**: Fixed loop length correction (1/2/4 bars per loop)
- Algorithm: [analysis/raster.py](../analysis/raster.py)

**Interpretation**:
- **Phase = 0**: Perfect grid alignment
- **Phase > 0**: Onset is late (after grid position)
- **Phase < 0**: Onset is early (before grid position)
- **Phase ±0.5**: Maximum deviation (halfway to next/previous grid position)
- **Lower RMS**: Better timing correction (method removes more systematic drift)

**Usage**:
- Use comprehensive CSV for custom microtiming analysis
- Compare raster plots visually to identify best correction method
- Phase data reveals groove characteristics (e.g., "laid back" = consistently late)

---

### 6_rms/

**Purpose**: Quantitative timing precision metrics for all correction methods.

**Files**:
- **`{track_id}_rms_summary.json`**: RMS (Root Mean Square) timing deviations
  - JSON format with key-value pairs for each method
  - **Milliseconds metrics**:
    - `uncorrected_ms`: RMS deviation in milliseconds (uncorrected)
    - `per_snippet_ms`: RMS deviation after per-snippet correction
    - `drum_ms`: RMS deviation after drum method correction
    - `mel_ms`: RMS deviation after melody method correction
    - `pitch_ms`: RMS deviation after pitch method correction
    - `standard_L1_ms`: RMS deviation after 1-bar loop correction
    - `standard_L2_ms`: RMS deviation after 2-bar loop correction
    - `standard_L4_ms`: RMS deviation after 4-bar loop correction

  - **Phase metrics** (same methods, in phase units):
    - `uncorrected_phase`, `per_snippet_phase`, `drum_phase`, etc.

**Technical Details**:
- **RMS calculation**: `RMS = sqrt(mean(phase²))`
- Phase values are dimensionless (-0.5 to +0.5 range)
- Millisecond values depend on tempo and grid resolution
- Algorithm: [analysis/rms_grid_histograms.py](../analysis/rms_grid_histograms.py)

**Interpretation**:
- **Lower RMS = better correction**: Less timing deviation from grid
- **Typical values**:
  - Uncorrected: 20-50ms (depends on beat detection accuracy)
  - Good correction: 5-15ms (systematic drift removed)
  - Excellent correction: <5ms (near-perfect alignment)
- **Comparing methods**: Lowest RMS indicates best correction for this track
- **Negative values impossible**: RMS is always ≥0 (square root of mean of squares)

**Usage**:
- Use RMS to objectively compare correction methods
- Lowest RMS method likely has best loop alignment
- RMS doesn't capture musical "feel" - use audio examples for subjective assessment

**Note**: RMS analysis is skipped in `--daw-ready` mode.

---

### 7_audio_examples/

**Purpose**: Audible comparison of correction methods using click tracks.

**Files**:
- **`original.mp3`**: Original audio snippet without click track
  - Duration: Snippet window (default 30s, or custom via `--manual-duration`)
  - Start time: Snippet offset (auto-detected or manual via `--manual-start`)

- **`uncorrected.mp3`**: Original + click track using raw downbeats
  - Click frequency: 3000 Hz (sharp, bright tone)
  - Click duration: 50ms
  - Shows Beat-Transformer output without any correction

- **`per_snippet.mp3`**: Original + click track using per-snippet correction
  - Single offset correction applied to entire 4-bar snippet

- **`drum.mp3`**: Original + click track using drum method
  - Pattern-based correction with auto-detected drum pattern length

- **`mel.mp3`**: Original + click track using melody method
  - Pattern-based correction with auto-detected melodic pattern length

- **`pitch.mp3`**: Original + click track using pitch method
  - Pattern-based correction with auto-detected bass pitch pattern length

- **`standard_L1.mp3`**: Original + click track using 1-bar loop correction
  - Fixed 1-bar pattern, 1 reference onset per bar

- **`standard_L2.mp3`**: Original + click track using 2-bar loop correction
  - Fixed 2-bar pattern, 1 reference onset per 2 bars

- **`standard_L4.mp3`**: Original + click track using 4-bar loop correction
  - Fixed 4-bar pattern, 1 reference onset per 4 bars

**Technical Details**:
- Click track generation: [utils/audio_export.py:create_audio_examples()](../utils/audio_export.py)
- Click track placed on downbeats (bar 1, position 0 of grid)
- Audio format: MP3 at 192 kbps (configurable in [config.py:122-123](../config.py#L122-L123))
- Mixing: Click track mixed with original audio at appropriate level

**Usage**:
- **Listen for alignment**: Best method has clicks perfectly aligned with downbeats
- **Detect drift**: If clicks drift out of sync, correction method failed
- **Subjective assessment**: RMS can't capture musical "feel" - trust your ears!
- **A/B comparison**: Switch between methods to hear differences

**Interpretation**:
- **Perfect alignment**: Clicks land exactly on perceived downbeats
- **Consistent drift**: Tempo detection error (clicks gradually shift early/late)
- **Erratic clicks**: Detection failure (clicks don't follow music at all)
- **Multiple valid methods**: Different methods may sound equally good

---

### 8_midi/ (Detailed Analysis Mode)

**Purpose**: MIDI export of onset times and bass pitch for DAW import and analysis.

**Subfolders**:
- **`onset/`**: Onset-based MIDI files (drum hits)
  - `per_snippet.mid`: Drum onsets using per-snippet correction
  - `drum.mid`: Drum onsets using drum method correction
  - `mel.mid`: Drum onsets using melody method correction
  - `pitch.mid`: Drum onsets using pitch method correction
  - `standard_L1.mid`: Drum onsets using 1-bar loop correction
  - `standard_L2.mid`: Drum onsets using 2-bar loop correction
  - `standard_L4.mid`: Drum onsets using 4-bar loop correction

- **`bass_pitch/`**: Bass pitch MIDI files (F0 → MIDI notes)
  - Same filename structure as onset/
  - Bass F0 frequencies converted to MIDI note numbers
  - Timing corrected using respective methods

**File Format**:
- Standard MIDI format (Type 0, single track)
- Tempo track: Bar-by-bar tempo changes from `bar_tempos.csv`
- Note: Middle C (MIDI note 60, C4) for onset MIDI
- Note: Bass pitch converted to nearest MIDI note for pitch MIDI
- Velocity: 100 (consistent across all notes)
- Duration: 50ms per note

**Technical Details**:
- Only onsets within loop boundaries are included (full bars only)
- Loop start/end calculated from snippet offset + pattern length
- Bass pitch detection: libf0 library with YIN algorithm
- F0 → MIDI conversion: `midi_note = round(69 + 12 × log2(f0 / 440))`
- Algorithm: [utils/midi_export.py](../utils/midi_export.py)

**Usage**:
- **Import into DAW**: Drag MIDI files into your DAW to trigger samples/synths
- **Timing analysis**: Compare MIDI timing with your own performance
- **Groove templates**: Use MIDI as groove template in Ableton/Logic
- **Bass line recreation**: Use bass pitch MIDI to recreate bass parts with synths

**Why Multiple Methods?**:
Different correction methods may produce different loop lengths and alignments. Try all methods to find the best one for your use case.

---

### 8_midi_drum/ (DAW Ready Mode)

**Purpose**: Simplified MIDI export for DAW-ready mode (drum method only).

**Files**:
- **`drum.mid`**: Onset-based MIDI using drum method
  - Contains drum hit onsets within detected loop boundaries
  - Same format as detailed mode but only drum method

- **`drum_bass_pitch.mid`**: Bass pitch MIDI using drum method
  - Bass F0 converted to MIDI notes
  - Timing corrected using drum method

**Difference from Detailed Mode**:
- Only drum method exported (not per_snippet, mel, pitch, standard)
- Files placed directly in `8_midi_drum/` (no subfolders)
- Faster processing for production workflows

---

### 9_loops/ (Detailed Analysis Mode)

**Purpose**: Perfectly cut stem loops for DAW remixing (all correction methods).

**Subfolders**:
Each correction method has its own subfolder containing 5 stem loops:

- **`per_snippet/`**: Stems cut using per-snippet correction
  - `vocals.wav`, `drums.wav`, `bass.wav`, `piano.wav`, `other.wav`

- **`drum/`**: Stems cut using drum method correction
  - 5 stem files (same as above)

- **`mel/`**: Stems cut using melody method correction
  - 5 stem files

- **`pitch/`**: Stems cut using pitch method correction
  - 5 stem files

- **`standard_L1/`**: Stems cut using 1-bar loop correction
  - 5 stem files

- **`standard_L2/`**: Stems cut using 2-bar loop correction
  - 5 stem files

- **`standard_L4/`**: Stems cut using 4-bar loop correction
  - 5 stem files

**File Format**:
- Default: WAV (uncompressed, same sample rate as input)
- Optional: MP3 (via `--export-format mp3`)
- Crossfade: 5ms at loop boundaries for seamless looping

**Technical Details**:
- Loop boundaries calculated from snippet offset + corrected grid times
- Only full bars included (partial bars at edges are excluded)
- Crossfade prevents clicks at loop boundaries
- Algorithm: [utils/audio_export.py:export_stem_loops()](../utils/audio_export.py)

**Usage**:
- **Import into DAW**: Drag stems into your DAW for remixing
- **Instant playback**: Loops are pre-cut to exact loop length
- **No manual editing**: Crossfades already applied
- **Compare methods**: Try different correction method folders to find best loop

**Loop Length Calculation**:
- Loop length = pattern_length × bar_duration
- Example: 4-bar pattern at 120 BPM in 4/4 = 8 seconds
- Different methods may produce different loop lengths (e.g., drum=4 bars, mel=2 bars)

---

### 9_loops_drum/ (DAW Ready Mode)

**Purpose**: Simplified stem loop export for DAW-ready mode (drum method only).

**Files**:
- **`vocals.wav`**: Vocal stem loop (drum method)
- **`drums.wav`**: Drum stem loop (drum method)
- **`bass.wav`**: Bass stem loop (drum method)
- **`piano.wav`**: Piano stem loop (drum method)
- **`other.wav`**: Other instruments stem loop (drum method)

**Difference from Detailed Mode**:
- Only drum method exported (not per_snippet, mel, pitch, standard)
- Files placed directly in `9_loops_drum/` (no subfolders)
- Faster processing for production workflows

---

### pipeline_results.json

**Purpose**: Complete summary of pipeline execution and all output file paths.

**Contents**:
- **`track_id`**: Track identifier
- **`audio_file`**: Input audio file path
- **`time_range`**: Time range used for analysis
  - `manual_start`: User-provided start time (or null)
  - `manual_duration`: User-provided duration (or null)
  - `auto_detect`: Boolean indicating if auto-detection was used
  - `actual_start`: Actual start time used (seconds)
  - `actual_duration`: Actual duration used (seconds)
  - `actual_end`: Actual end time (start + duration)
- **`steps_completed`**: List of completed pipeline steps
  - e.g., `["stem_separation", "beat_detection", "correct_bars", ...]`
- **`errors`**: List of error messages (empty if successful)
- **`stems_dir`**: Path to stems directory
- **`npz_file`**: Path to mel-spectrogram NPZ file
- **`beats_file`**: Path to raw beat detection output
- **`correction_stats`**: Statistics from downbeat correction
  - `raw_bars`: Number of bars before correction
  - `corrected_bars`: Number of bars after correction
  - `dominant_tempo`: Detected dominant tempo (BPM)
- **`tempo_plots_pdf`**: Path to tempo plots PDF
- **`tempo_csv`**: Path to bar tempos CSV
- **`onset_file`**: Path to onsets CSV
- **`num_onsets`**: Number of detected onsets
- **`pattern_lengths`**: Detected pattern lengths
  - `drum`: Drum pattern length (bars)
  - `mel`: Melody pattern length (bars)
  - `pitch`: Pitch pattern length (bars)
- **`comprehensive_csv`**: Path to comprehensive phases CSV
- **`rms_values`**: RMS timing deviation metrics
  - All RMS values for each method (ms and phase)
- **`midi_files`**: Dictionary of exported MIDI files
  - `onset`: List of onset MIDI file paths
  - `bass_pitch`: List of bass pitch MIDI file paths
- **`loop_files`**: Dictionary of exported stem loops
  - Keys: method names (e.g., "drum", "mel", "pitch")
  - Values: List of stem file paths for that method

**Usage**:
- **Batch processing**: Programmatically check which tracks succeeded/failed
- **Path lookup**: Find output files without manual navigation
- **Debug**: Check error messages if pipeline failed
- **Reproducibility**: Record exact parameters used for each track

---

## Processing Status

To check if a track was fully processed:

1. **All subdirectories exist**:
   - Detailed mode: 1_stems through 9_loops
   - DAW ready mode: 1_stems, 2_beats, 3_corrected, 3.5_tempo_plots, 4_onsets, 5_grid, 8_midi_drum, 9_loops_drum

2. **`pipeline_results.json` has no errors**:
   ```bash
   cat {track_id}/pipeline_results.json | grep "errors"
   # Should show: "errors": []
   ```

3. **Audio examples were generated** (if not disabled):
   - Check for MP3 files in `7_audio_examples/`
   - If `--no-audio-examples` was used, this folder may be empty

4. **MIDI and loops were generated**:
   - Check `8_midi/` or `8_midi_drum/` for MIDI files
   - Check `9_loops/` or `9_loops_drum/` for stem loops

---

## Output Modes

The pipeline has two output modes:

### Detailed Analysis Mode (default)

**Purpose**: Complete microtiming analysis with all correction methods.

**Outputs**:
- All subfolders: 1_stems through 9_loops
- Multiple correction methods: per_snippet, drum, mel, pitch, standard_L1, standard_L2, standard_L4
- Visualization: Tempo plots, raster plots, RMS analysis
- Audio examples: Click tracks for all methods
- MIDI: Subfolders for onset/ and bass_pitch/
- Loops: Subfolders for each correction method

**Use when**:
- Conducting research on microtiming
- Comparing different correction methods
- Need detailed visualizations and metrics
- Want to analyze groove characteristics

### DAW Ready Mode (`--daw-ready`)

**Purpose**: Fast production workflow for remixing and music production.

**Outputs**:
- Essential subfolders only: 1_stems, 2_beats, 3_corrected, 3.5_tempo_plots (CSV only), 4_onsets, 5_grid, 8_midi_drum, 9_loops_drum
- Single correction method: drum method only
- No visualization: Tempo plots, raster plots, RMS analysis skipped
- No audio examples: Click tracks skipped
- MIDI: Direct in `8_midi_drum/` (no subfolders)
- Loops: Direct in `9_loops_drum/` (no subfolders)

**Use when**:
- Remixing tracks in your DAW
- Need stems and loops quickly
- Don't need detailed analysis
- Want drum method only (most reliable for rhythm)

**Activate with**:
```bash
python main.py --audio track.wav --track-id 123 --output-dir output/ --daw-ready
```

---

## Correction Methods Explained

The pipeline includes 8 correction methods to fix timing drift:

### 1. Uncorrected
- **Description**: Raw Beat-Transformer output (no correction)
- **Use case**: Baseline for comparison
- **Pros**: Fast, no computation needed
- **Cons**: May have double/half-time errors and tempo drift

### 2. Per-Snippet
- **Description**: Single offset correction for 4-bar snippet
- **Pattern length**: 4 bars (fixed)
- **Reference offsets**: 1 per snippet
- **Use case**: Short loops with stable tempo
- **Pros**: Simple, works for most music
- **Cons**: Doesn't adapt to longer patterns

### 3. Drum Method
- **Description**: Pattern-based correction using drum onsets
- **Pattern length**: Auto-detected (typically 1-8 bars)
- **Reference offsets**: 1 per pattern repetition
- **Use case**: Rhythm-driven music
- **Pros**: Adapts to detected drum pattern
- **Cons**: Requires clear drum pattern

### 4. Melody Method (Mel)
- **Description**: Pattern-based correction using melodic onsets
- **Pattern length**: Auto-detected (typically 2-8 bars)
- **Reference offsets**: 1 per pattern repetition
- **Use case**: Melody-driven music with weak drums
- **Pros**: Works when drum pattern is ambiguous
- **Cons**: Requires clear melodic pattern

### 5. Pitch Method
- **Description**: Pattern-based correction using bass pitch
- **Pattern length**: Auto-detected (typically 1-4 bars)
- **Reference offsets**: 1 per pattern repetition
- **Use case**: Bass-driven music (funk, disco, etc.)
- **Pros**: Works when rhythm pattern is complex
- **Cons**: Requires clear bass line

### 6. Standard L=1 (1-bar loop)
- **Description**: Fixed 1-bar pattern, 1 reference onset per bar
- **Pattern length**: 1 bar (fixed)
- **Reference offsets**: 1 per bar
- **Use case**: Simple, repetitive rhythms
- **Pros**: Maximum correction granularity
- **Cons**: May over-correct if pattern is longer than 1 bar

### 7. Standard L=2 (2-bar loop)
- **Description**: Fixed 2-bar pattern, 1 reference onset per 2 bars
- **Pattern length**: 2 bars (fixed)
- **Reference offsets**: 1 per 2 bars
- **Use case**: Common pop/rock patterns (verse/chorus)
- **Pros**: Balances correction and pattern length
- **Cons**: May under-correct if pattern is 1 bar, over-correct if 4+ bars

### 8. Standard L=4 (4-bar loop)
- **Description**: Fixed 4-bar pattern, 1 reference onset per 4 bars
- **Pattern length**: 4 bars (fixed)
- **Reference offsets**: 1 per 4 bars
- **Use case**: Most Western popular music (4-bar phrases)
- **Pros**: Matches common song structure
- **Cons**: May under-correct for shorter patterns

**Which method should I use?**
- **Default**: Try drum method first (most reliable)
- **Compare**: Check RMS values in `6_rms/rms_summary.json` (lower is better)
- **Listen**: Use audio examples in `7_audio_examples/` (trust your ears!)
- **For DAW**: DAW ready mode automatically uses drum method

---

## File Size Estimates

Typical file sizes for a 3-minute track:

- **Stems (WAV)**: ~150 MB (5 stems × ~30 MB each)
- **Mel-spectrograms (NPZ)**: ~5 MB
- **Beat detection output**: <100 KB
- **CSV files**: <500 KB each
- **Tempo plots (PDF)**: ~500 KB
- **Raster plots (PNG)**: ~200 KB each
- **RMS summary (JSON)**: <10 KB
- **Audio examples (MP3)**: ~1 MB each (×8 = ~8 MB total)
- **MIDI files**: <50 KB each (×14 = ~700 KB total)
- **Stem loops (WAV)**: ~15 MB per method per stem (×5 stems × 7 methods = ~525 MB)
  - Or ~2 MB per method per stem if using MP3 export (×5 stems × 7 methods = ~70 MB)

**Total for detailed mode**: ~700 MB (WAV loops) or ~250 MB (MP3 loops)
**Total for DAW ready mode**: ~180 MB (WAV loops) or ~170 MB (MP3 loops)

---

## Advanced Usage

### Custom Time Range

Override automatic snippet detection:

```bash
# Start at 50 seconds, analyze 60 seconds
python main.py --audio track.wav --track-id 123 --output-dir output/ \
    --manual-start 50.0 --manual-duration 60.0
```

### MP3 Loop Export

Export stem loops as MP3 instead of WAV (smaller file size):

```bash
python main.py --audio track.wav --track-id 123 --output-dir output/ \
    --export-format mp3
```

### Skip Audio Examples

Save time by skipping click track generation:

```bash
python main.py --audio track.wav --track-id 123 --output-dir output/ \
    --no-audio-examples
```

### Batch Processing

Process entire folder with auto-detected track IDs:

```bash
python main.py --audio-dir input_wav/ --output-dir output/ --analyse-all
```

---

## Troubleshooting

### Missing Output Files

**Problem**: Some output files are missing.

**Solutions**:
- Check `pipeline_results.json` for errors
- Look for error messages in terminal output
- Verify input audio file is valid (not corrupted)
- Check disk space (pipeline needs ~700 MB per track)

### Poor Loop Quality

**Problem**: Loops don't align properly or have clicks.

**Solutions**:
- Try different correction methods (compare audio examples)
- Check tempo plots for tempo instability
- Verify time range includes stable, repetitive section
- Use manual time range to select better section

### High RMS Values

**Problem**: All methods have high RMS (>20ms).

**Solutions**:
- Check if music has tempo changes (pipeline assumes constant tempo)
- Verify beat detection worked (check `2_beats/` output)
- Try manual time range selection for more stable section
- Some music naturally has loose timing (not a bug!)

### MIDI Files Sound Wrong

**Problem**: MIDI doesn't match audio.

**Solutions**:
- Verify you're using correct correction method (try audio examples first)
- Check if loop boundaries are correct (partial bars may be excluded)
- Bass pitch MIDI requires good F0 detection (check `1_stems/bass_f0.csv`)
- Some instruments don't translate well to MIDI (use audio loops instead)

---

## Technical Notes

### Grid Resolution

- **16 positions per bar**: 16th note grid
- **4/4 time**: 16 positions = 16th notes
- **3/4 time**: 16 positions ≠ 16th notes (different quantization)

### Phase Units

- **Phase = 0**: Perfect alignment
- **Phase = ±0.5**: Maximum deviation (halfway to adjacent grid position)
- **Phase > 0.5 or < -0.5**: Impossible (onset would be closer to different grid position)

### Tempo Calculation

- **Bar tempo**: `tempo = (60 / bar_duration) × time_signature`
- **Example**: 2-second bar in 4/4 → `(60 / 2) × 4 = 120 BPM`

### Loop Boundaries

- **Start**: First complete bar after snippet offset
- **End**: Last complete bar before snippet end
- **Partial bars excluded**: Ensures clean loop boundaries

---

## Citation

If you use this pipeline in research, please cite the relevant papers listed in the main [README.md](../../README.md#citation).
