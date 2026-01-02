# Technical Details: Loop Extractor Pipeline

This document explains the internal workings of each step in the Loop Extractor pipeline.

**Author:** Alexander Krause, TU Berlin
**Co-Author:** Claude Code (Anthropic)

---

## Table of Contents

1. [Pipeline Flow Diagram](#pipeline-flow-diagram)
2. [Stem Separation](#1-stem-separation)
3. [Downbeat Detection](#2-downbeat-detection)
4. [Downbeat Correction](#3-downbeat-correction)
5. [Onset Detection](#4-onset-detection)
6. [Pattern Length Estimation](#5-pattern-length-estimation)
7. [Grid Creation and Correction](#6-grid-creation-and-correction)

---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LOOP EXTRACTOR PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              INPUT: Mixed Audio
                                 (WAV/MP3)
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: STEM SEPARATION (Spleeter)                                         │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  Mixed Audio ──► Spleeter 5-stem U-Net ──► 5 Stem WAVs                    │
│                        │                      │                             │
│                        │                      ├─► vocals.wav                │
│                        │                      ├─► drums.wav  ─┐             │
│                        │                      ├─► bass.wav   ─┼─► Used for │
│                        │                      ├─► piano.wav   │   analysis  │
│                        │                      └─► other.wav  ─┘             │
│                        │                                                    │
│                        ▼                                                    │
│              Mel-Spectrogram Conversion                                     │
│              (44.1kHz, 4096 FFT, 128 mels)                                 │
│                        │                                                    │
│                        ▼                                                    │
│              5-stem NPZ file (5, T, 128)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ├──────────────────────┐
                         │                      │
                         ▼                      ▼
┌────────────────────────────────────┐  ┌──────────────────────────────────┐
│ STEP 2: BEAT DETECTION             │  │ STEP 4: ONSET DETECTION          │
│ ────────────────────────────────── │  │ ──────────────────────────────── │
│                                    │  │                                  │
│  5-stem NPZ                        │  │  drums.wav                       │
│      │                             │  │      │                           │
│      ▼                             │  │      ▼                           │
│  Beat-Transformer                  │  │  Librosa HFC Onset Detection    │
│  (9-layer Dilated Transformer)     │  │  (hop=512, delta=0.12)          │
│      │                             │  │      │                           │
│      ├─► Beat activations          │  │      ▼                           │
│      └─► Downbeat activations      │  │  Onset times (CSV)              │
│           │                        │  │      │                           │
│           ▼                        │  │      │                           │
│  Madmom DBN Post-Processing        │  └──────┼───────────────────────────┘
│  (HMM, 55-215 BPM)                 │         │
│           │                        │         │
│           ▼                        │         │
│  Raw beats/downbeats               │         │
│  (beat_pos, bar_num, times)        │         │
└────────────────────────────────────┘         │
                 │                             │
                 ▼                             │
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: DOWNBEAT CORRECTION                                                │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  Raw Downbeats ──► Bar Tempo Calculation ──► Factor-of-2 Classification   │
│                         │                              │                    │
│                         │                              ├─► "normal"         │
│                         │                              ├─► "double"         │
│                         │                              └─► "half"           │
│                         ▼                                   │               │
│                   Median BPM = 120                          │               │
│                         │                                   │               │
│                         ▼                                   ▼               │
│              ┌──────────────────────────┐      Determine Dominant Pattern  │
│              │  Dominant = "normal"     │                  │                │
│              │  or "factor2 (double)"   │                  │                │
│              │  or "factor2 (half)"     │                  │                │
│              └──────────────────────────┘                  │                │
│                         │                                  │                │
│                         ▼                                  ▼                │
│              Apply Corrections:              ┌────────────────────┐        │
│              • MERGE double bars             │ If dominant=normal:│        │
│              • SPLIT half bars               │  MERGE "double"    │        │
│              • BPM threshold adjust          │  SPLIT "half"      │        │
│                         │                    └────────────────────┘        │
│                         ▼                                                   │
│              Corrected Downbeats (TXT)                                     │
│              + Time signature                                               │
│              + Usable bars mask                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         │
                         ├────────────────────────────────┐
                         │                                │
                         ▼                                ▼
┌────────────────────────────────────┐  ┌─────────────────────────────────────┐
│ STEP 5: PATTERN LENGTH DETECTION   │  │ STEP 6: GRID CREATION & CORRECTION │
│ ────────────────────────────────── │  │ ─────────────────────────────────── │
│                                    │  │                                     │
│  Inputs: Corrected downbeats,      │  │  Inputs: Downbeats, Onsets,        │
│          Onsets, drums.wav,        │  │          Pattern lengths            │
│          bass.wav                  │  │                                     │
│                                    │  │                                     │
│  ┌──────────────────────────────┐ │  │  ┌───────────────────────────────┐ │
│  │ Method 1: Drum Onset         │ │  │  │ A) UNCORRECTED GRID          │ │
│  │ ──────────────────────────── │ │  │  │ ───────────────────────────── │ │
│  │ • Binary onset vectors       │ │  │  │ • Use raw downbeats          │ │
│  │ • Circular cross-correlation │ │  │  │ • 16 ticks per bar           │ │
│  │ • Lag profile analysis       │ │  │  │ • Match onsets to grid       │ │
│  │        │                     │ │  │  │ • Calculate phases           │ │
│  │        ▼                     │ │  │  └───────────────────────────────┘ │
│  │   Best L = 4 bars            │ │  │                                     │
│  └──────────────────────────────┘ │  │  ┌───────────────────────────────┐ │
│                                    │  │  │ B) PER-SNIPPET CORRECTION    │ │
│  ┌──────────────────────────────┐ │  │  │ ───────────────────────────── │ │
│  │ Method 2: Mel-Band           │ │  │  │ • Find 1 reference offset    │ │
│  │ ──────────────────────────── │ │  │  │ • Priority: tick 0,8,4,12    │ │
│  │ • Log-mel spectrograms       │ │  │  │ • Apply to entire snippet    │ │
│  │ • Band-wise correlation      │ │  │  │ • Shift grid uniformly       │ │
│  │ • Resample to 64 frames      │ │  │  └───────────────────────────────┘ │
│  │        │                     │ │  │                                     │
│  │        ▼                     │ │  │  ┌───────────────────────────────┐ │
│  │   Best L = 2 bars            │ │  │  │ C) LOOP-BASED CORRECTION     │ │
│  └──────────────────────────────┘ │  │  │ ───────────────────────────── │ │
│                                    │  │  │ For each loop:               │ │
│  ┌──────────────────────────────┐ │  │  │ 1. Create equidistant grid   │ │
│  │ Method 3: Bass Pitch         │ │  │  │    (remove tempo drift)      │ │
│  │ ──────────────────────────── │ │  │  │ 2. Find ref offset in loop   │ │
│  │ • F0 extraction (Melodia)    │ │  │  │ 3. Shift loop grid           │ │
│  │ • Z-score normalize          │ │  │  │ 4. Calculate phases          │ │
│  │ • Beat-aligned shifts        │ │  │  │                              │ │
│  │        │                     │ │  │  │ Methods:                     │ │
│  │        ▼                     │ │  │  │ • Drum (L=auto)              │ │
│  │   Best L = 4 bars            │ │  │  │ • Mel (L=auto)               │ │
│  └──────────────────────────────┘ │  │  │ • Pitch (L=auto)             │ │
│                                    │  │  │ • Standard L=1, L=2, L=4     │ │
│  Output:                           │  │  └───────────────────────────────┘ │
│  {'drum': 4, 'mel': 2, 'pitch': 4} │  │                                     │
└────────────────────────────────────┘  │  Output:                            │
                                        │  comprehensive_phases.csv           │
                                        │  • onset_time(s)                    │
                                        │  • bar, tick                        │
                                        │  • phase_uncorrected                │
                                        │  • phase_per_snippet                │
                                        │  • phase_drum(L=4)                  │
                                        │  • phase_mel(L=2)                   │
                                        │  • phase_pitch(L=4)                 │
                                        │  • phase_standard_L1/L2/L4          │
                                        │  • grid_time_* for each method      │
                                        │  • phase_*_ms (milliseconds)        │
                                        └─────────────────────────────────────┘
                                                        │
                         ┌──────────────────────────────┼─────────────────────┐
                         │                              │                     │
                         ▼                              ▼                     ▼
┌────────────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────┐
│ RMS CALCULATION                │  │ AUDIO EXAMPLES           │  │ RASTER PLOTS       │
│ ────────────────────────────── │  │ ──────────────────────── │  │ ────────────────── │
│                                │  │                          │  │                    │
│  For each method:              │  │  For each method:        │  │  Two plots:        │
│  • RMS(phase) in phase units   │  │  • Original + Click      │  │                    │
│  • RMS(phase) in milliseconds  │  │  • Click on downbeats    │  │  1) Comparison     │
│                                │  │  • 3kHz, 50ms clicks     │  │     • Uncorrected  │
│  Quantifies correction quality │  │                          │  │     • Per-snippet  │
│  Lower RMS = better alignment  │  │  Exports:                │  │     • Drum         │
│                                │  │  • uncorrected.mp3       │  │     • Mel          │
│  Output: rms_summary.json      │  │  • per_snippet.mp3       │  │     • Pitch        │
│  {                             │  │  • drum.mp3              │  │                    │
│    "uncorrected_ms": 25.3,     │  │  • mel.mp3               │  │  2) Standard       │
│    "per_snippet_ms": 12.1,     │  │  • pitch.mp3             │  │     • Uncorrected  │
│    "drum_ms": 8.4,             │  │  • standard_L1.mp3       │  │     • Standard L=1 │
│    "standard_L1_ms": 5.2,      │  │  • standard_L2.mp3       │  │     • Standard L=2 │
│    ...                         │  │  • standard_L4.mp3       │  │     • Standard L=4 │
│  }                             │  │  • original.mp3          │  │     • Per-snippet  │
└────────────────────────────────┘  └──────────────────────────┘  │                    │
                                                                   │  Visualizes timing │
                                                                   │  deviations vs bar │
                                                                   │                    │
                                                                   │  Output:           │
                                                                   │  • raster_comparison.png │
                                                                   │  • raster_standard.png   │
                                                                   │  • microtiming_plots.pdf │
                                                                   │                          │
                                                                   │  Pattern-folded plots:   │
                                                                   │  • Uncorrected           │
                                                                   │  • Per-Snippet           │
                                                                   │  • Standard L=1/L=2/L=4  │
                                                                   └────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ ADDITIONAL OUTPUTS                                                          │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐│
│  │ TEMPO ANALYSIS      │  │ MIDI EXPORT         │  │ STEM LOOPS          ││
│  │ ─────────────────── │  │ ─────────────────── │  │ ─────────────────── ││
│  │                     │  │                     │  │                     ││
│  │ • 8-panel tempo     │  │ onset/ folder:      │  │ For each method:    ││
│  │   comparison plot   │  │ • per_snippet.mid   │  │ • vocals.wav loop   ││
│  │                     │  │ • drum.mid          │  │ • drums.wav loop    ││
│  │ • Bar-by-bar tempo  │  │ • mel.mid           │  │ • bass.wav loop     ││
│  │   CSV file          │  │ • pitch.mid         │  │ • piano.wav loop    ││
│  │                     │  │ • standard_L*.mid   │  │ • other.wav loop    ││
│  │ • Shows uncorrected │  │                     │  │                     ││
│  │   vs corrected      │  │ bass_pitch/ folder: │  │ Folders:            ││
│  │                     │  │ • Same structure    │  │ • per_snippet/      ││
│  │ Output:             │  │ • F0→MIDI notes     │  │ • drum/             ││
│  │ • tempo_plots.pdf   │  │                     │  │ • mel/              ││
│  │ • bar_tempos.csv    │  │ Tempo-aware MIDI    │  │ • pitch/            ││
│  └─────────────────────┘  │ with bar changes    │  │ • standard_L1/      ││
│                           └─────────────────────┘  │ • standard_L2/      ││
│                                                    │ • standard_L4/      ││
│                                                    │                     ││
│                                                    │ Perfect loops with  ││
│                                                    │ 5ms crossfade       ││
│                                                    └─────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘

                      FINAL OUTPUT: pipeline_results.json
                      • Complete summary of all outputs
                      • File paths, parameters, statistics
                      • Error tracking, processing status
```

### Key Data Flow

1. **Audio → Stems**: Spleeter separates into 5 instruments
2. **Stems → Mel-Specs**: Converted for beat detection input
3. **Mel-Specs → Beats**: Beat-Transformer + DBN finds beats/downbeats
4. **Beats → Corrected**: Factor-of-2 tempo errors fixed
5. **Drums → Onsets**: Precise drum hit times detected
6. **Downbeats + Stems → Pattern Length**: Three methods auto-detect loop length
7. **Downbeats + Onsets + Patterns → Phases**: Multiple correction methods applied
8. **Phases → Analysis**: RMS, plots, audio examples, MIDI, loops

### Parallel Processing

- Steps 2 and 4 run independently (can be parallelized)
- Steps 5 and 6 both depend on steps 2-4 outputs
- RMS, audio examples, and raster plots all depend on step 6

### Critical Files

- **Stems NPZ**: Input for beat detection
- **Corrected downbeats**: Foundation for all timing analysis
- **Comprehensive CSV**: Contains all phase calculations (core output)
- **RMS summary**: Quantifies correction quality

---

## 1. Stem Separation

**File:** [loop_extractor/stem_separation/spleeter_interface.py](loop_extractor/stem_separation/spleeter_interface.py)

### Purpose
Separate a mixed audio track into 5 isolated instrument stems using the Spleeter model, then convert them to mel-spectrograms for beat detection.

### Algorithm

#### Step 1.1: Audio Separation with Spleeter

**Model:** Spleeter 5-stem (pre-trained by Deezer Research)

**Input:**
- Mixed audio file (WAV, MP3, FLAC, etc.)

**Output:**
- 5 stem WAV files: `vocals.wav`, `drums.wav`, `bass.wav`, `piano.wav`, `other.wav`

**Process:**
```python
separator = Separator('spleeter:5stems')
separator.separate_to_file(audio_path, output_dir, codec='wav')
```

**Technical Details:**
- **Architecture**: U-Net-based convolutional neural network
- **Training**: Supervised learning on Deezer's internal dataset
- **Stems**:
  - `vocals`: Isolated vocal track (singing, rap, etc.)
  - `drums`: Percussion instruments (kick, snare, hi-hat, cymbals, etc.)
  - `bass`: Bass guitar, synth bass, sub-bass
  - `piano`: Piano, keyboard instruments
  - `other`: Everything else (guitars, synths, orchestral, etc.)

- **Memory Optimization**: Global separator instance is reused to avoid reloading the 200MB model for each track

#### Step 1.2: Mel-Spectrogram Conversion

**Purpose:** Convert audio stems to mel-spectrograms for Beat-Transformer input

**Process** (for each stem):
1. **Load audio** as mono at 44.1kHz sample rate
2. **Compute STFT** (Short-Time Fourier Transform):
   - FFT window: 4096 samples
   - Hop length: 1024 samples (~23ms per frame at 44.1kHz)
3. **Convert to power spectrogram**: `S = |STFT|²`
4. **Apply mel filterbank**:
   - Number of mel bins: 128
   - Frequency range: 30 - 11000 Hz
5. **Convert to dB scale**: `mel_dB = 10 × log₁₀(mel_power / max(mel_power))`

**Mathematical Details:**

Mel-scale conversion (perceptual frequency scale):
```
mel(f) = 2595 × log₁₀(1 + f/700)
```

STFT:
```
X(m,k) = Σ[n=0 to N-1] x(n) × w(n - mH) × e^(-j2πkn/N)
```
where:
- `m`: Frame index
- `k`: Frequency bin
- `H`: Hop length (1024)
- `N`: FFT size (4096)
- `w`: Window function (Hann window)

**Output:**
- NPZ file containing 5D array: `(5 stems, time_frames, 128 mel_bins)`
- Typical shape: `(5, 9294, 128)` for a 3-minute track
- File size: ~5 MB

**Why Mel-Spectrograms?**
- Mel-scale matches human auditory perception
- Multi-stem input provides better beat detection than mono
- Time-frequency representation captures rhythmic patterns
- Drums provide strong transients, vocals/bass provide melodic context

---

## 2. Downbeat Detection

**Files:**
- [loop_extractor/beat_detection/transformer.py](loop_extractor/beat_detection/transformer.py) (wrapper)
- [loop_extractor/beat_detection/run_transformer.py](loop_extractor/beat_detection/run_transformer.py) (subprocess)

### Purpose
Detect beat and downbeat positions using the Beat-Transformer neural network model.

### Algorithm

#### Step 2.1: Beat-Transformer Model

**Architecture:** 9-layer Dilated Transformer with multi-head attention

**Input:**
- 5-stem mel-spectrogram: `(5, time_frames, 128)`

**Model Parameters:**
```python
attention_length = 5        # Attention span in frames
num_instruments = 5          # 5 stems
num_tokens = 2               # beat, downbeat
model_dim = 256              # Hidden dimension
num_heads = 8                # Multi-head attention
feedforward_dim = 1024       # FFN dimension
num_layers = 9               # Transformer layers
```

**Process:**
1. **Input Embedding**: Mel-spectrograms → 256-dim embeddings
2. **Dilated Self-Attention**: Each layer uses dilated attention pattern to capture long-range rhythmic dependencies
3. **Multi-Head Attention**: 8 parallel attention heads learn different rhythmic features
4. **Output Heads**: Two separate heads predict beat and downbeat activations

**Mathematical Formulation:**

Self-Attention:
```
Attention(Q,K,V) = softmax(QK^T / √d_k) × V
```

Dilated Attention (dilation rate `d`):
```
AttendTo(position_i) = {position_j : |i - j| ≤ attention_len × d}
```

**Output:**
- Beat activation function: `beat_activation(t)` ∈ [0, 1]
- Downbeat activation function: `downbeat_activation(t)` ∈ [0, 1]

#### Step 2.2: Dynamic Bayesian Network (DBN) Post-Processing

**Library:** Madmom (Python library for music information retrieval)

**Purpose:** Convert continuous activation functions to discrete beat/downbeat times

**Algorithm:** Hidden Markov Model with tempo transition model

**Parameters:**
```python
min_bpm = 55.0              # Minimum tempo
max_bpm = 215.0             # Maximum tempo
fps = 44100 / 1024 ≈ 43.066 # Frames per second
transition_lambda = 100      # Tempo change penalty
observation_lambda = 6       # Activation threshold weight
beats_per_bar = [3, 4]      # Support 3/4 and 4/4 time
```

**State Space:**
- **State**: `(tempo, beat_position, bar_number)`
- **Tempo**: Discretized BPM values between 55-215
- **Beat position**: 1, 2, 3 (for 3/4) or 1, 2, 3, 4 (for 4/4)

**Transition Model:**
```
P(state_t | state_{t-1}) ∝ exp(-λ_trans × |tempo_t - tempo_{t-1}|)
```
- Penalizes large tempo changes
- Encourages temporal consistency

**Observation Model:**
```
P(activation | state) ∝ exp(λ_obs × activation)
```
- Higher activations → higher probability

**Viterbi Decoding:**
- Finds most likely beat/downbeat sequence given activations
- Balances activation strength with tempo smoothness

**Output File Format:**
```
downbeat_time(s)  beat_activation  downbeat_activation  beat_pos  bar_num
0.000000          0.891            0.923               1         0
0.465116          0.782            0.123               2         0
0.930233          0.801            0.098               3         0
1.395349          0.834            0.087               4         0
1.860465          0.903            0.956               1         1
...
```

Where:
- `beat_pos`: Position within bar (1 = downbeat, 2-4 = other beats)
- `bar_num`: Sequential bar count (starts at 0 for incomplete first bar)

**Why Two-Stage Detection?**
- **Neural network**: Captures complex rhythmic patterns from audio
- **DBN**: Enforces musical constraints (tempo continuity, time signature)
- Combination is more robust than either alone

---

## 3. Downbeat Correction

**File:** [loop_extractor/analysis/correct_bars.py](loop_extractor/analysis/correct_bars.py)

### Purpose
Fix factor-of-two tempo errors (double-time, half-time) in beat detection output.

### Problem Statement

Beat detection models sometimes misidentify the downbeat periodicity:

**Example Errors:**
1. **Half-time error**: Detects every 2 bars as 1 bar → tempo appears 2× too fast
   - True tempo: 120 BPM → Detected: 240 BPM
2. **Double-time error**: Detects 2 downbeats per bar → tempo appears 2× too slow
   - True tempo: 120 BPM → Detected: 60 BPM
3. **Quadruple-time error**: Detects every 4 bars as 1 bar → tempo appears 4× too fast

### Algorithm

#### Step 3.1: Extract Downbeat Times

**Input:** Beat-Transformer output file

**Process:**
```python
# Filter to complete bars (bar_num ≥ 1) and downbeats (beat_pos == 1)
mask = (bar_num >= 1) & (beat_pos == 1)
downbeat_times = df[mask]['downbeat_time(s)'].values
```

**Output:** List of downbeat times `[t₁, t₂, t₃, ..., tₙ]`

#### Step 3.2: Calculate Bar Tempos

**Formula:**
```
tempo_bpm = (60 / bar_duration) × time_signature
```

**Example** (4/4 time):
- Bar duration: 2.0 seconds
- Tempo: `(60 / 2.0) × 4 = 120 BPM`

#### Step 3.3: Classify Each Bar

**Base Tempo:** Median of all bar tempos

**Classification Function:**
```python
def classify_factor2(bpm, base, tolerance=0.10):
    """Classify tempo relative to base."""
    # Double range: base × 2 ± 10%
    if (base × 1.8) ≤ bpm ≤ (base × 2.2):
        return "double"

    # Half range: base × 0.5 ± 10%
    elif (base × 0.45) ≤ bpm ≤ (base × 0.55):
        return "half"

    # Normal range
    else:
        return "normal"
```

**Example:**
- Base tempo: 120 BPM
- Bar with 240 BPM → classified as "double"
- Bar with 60 BPM → classified as "half"
- Bar with 118 BPM → classified as "normal"

#### Step 3.4: Determine Dominant Pattern

**Counting:**
```python
n_normal = count("normal")
n_double = count("double")
n_half = count("half")
```

**Decision:**
```python
if n_normal >= (n_double + n_half):
    dominant = "normal"
else:
    dominant = "factor2"
    orientation = "double" if n_double >= n_half else "half"
```

#### Step 3.5: Apply Corrections

**Correction Logic:**

| Dominant | Bar Class | Action |
|----------|-----------|--------|
| normal   | double    | **MERGE** 2 bars → 1 bar |
| normal   | half      | **SPLIT** 1 bar → 2 bars |
| normal   | normal    | unchanged |
| factor2 (double) | normal | **SPLIT** 1 bar → 2 bars |
| factor2 (half)   | normal | **MERGE** 2 bars → 1 bar |

**Merge Operation:**
```python
# Merge bars i and i+1
new_bar_start = downbeats[i]
new_bar_end = downbeats[i+2]  # Skip middle downbeat
# Removes downbeat at position i+1
```

**Split Operation:**
```python
# Split bar i into two bars
bar_start = downbeats[i]
bar_end = downbeats[i+1]
midpoint = (bar_start + bar_end) / 2.0
# Inserts new downbeat at midpoint
```

**Example Correction Sequence:**

*Before (half-time error):*
```
Bar 0: 0.0 - 4.0s  (4.0s duration, 60 BPM)  ← classified as "half"
Bar 1: 4.0 - 8.0s  (4.0s duration, 60 BPM)  ← classified as "half"
Bar 2: 8.0 - 12.0s (4.0s duration, 60 BPM)  ← classified as "half"
```

*After (dominant="normal"):*
```
Bar 0: 0.0 - 2.0s  (2.0s duration, 120 BPM)  ← split from original bar 0
Bar 1: 2.0 - 4.0s  (2.0s duration, 120 BPM)  ← split from original bar 0
Bar 2: 4.0 - 6.0s  (2.0s duration, 120 BPM)  ← split from original bar 1
Bar 3: 6.0 - 8.0s  (2.0s duration, 120 BPM)  ← split from original bar 1
Bar 4: 8.0 - 10.0s (2.0s duration, 120 BPM)  ← split from original bar 2
Bar 5: 10.0 - 12.0s (2.0s duration, 120 BPM) ← split from original bar 2
```

#### Step 3.6: BPM Threshold Adjustment (Optional)

**Purpose:** Handle tracks with primarily high BPM (>135) that get misclassified

**Algorithm:**
```python
if count(bpm > 135) > (total_bars / 2):
    # More than half the bars are high BPM
    # Assume detection is in double-time
    high_median = median(bars with bpm > 135)
    new_base = high_median / 2.0

    # Reclassify all bars using new_base
```

**Example:**
- Detected tempos: 150, 155, 148, 152 BPM (all >135)
- Original classification: All "normal"
- Adjustment: new_base = 152 / 2 = 76 BPM
- Reclassification: All become "double"
- Correction: Merge pairs → final tempo ~75 BPM

#### Step 3.7: Usability Filtering

**Purpose:** Mark which bars have stable tempo for analysis

**Criterion:**
```python
usable = |bar_tempo - dominant_tempo| ≤ 10% × dominant_tempo
```

**Example:**
- Dominant tempo: 120 BPM
- Acceptable range: 108 - 132 BPM
- Bar with 125 BPM → usable
- Bar with 95 BPM → not usable (likely has tempo variation or error)

**Output File Format:**
```
# time_signature=4
# dominant_major=normal
# usable_base_bpm=120.50
# usable_count=42
# total_bars=45

corrected_bar_num  bar_num  corrected_downbeat_time(s)  next_downbeat_time(s)  duration_s  tempo_bpm  usable  action
1                  1        0.000000                    1.995349               1.995349    120.28     1       unchanged
2                  2        1.995349                    3.990698               1.995349    120.28     1       unchanged
3                  3|4      3.990698                    7.981395               3.990698    120.28     1       merged
4                  5        7.981395                    9.976744               1.995349    120.28     1       unchanged
...
```

**Metadata:**
- **action**: `unchanged`, `merged`, `split_A`, `split_B`, `merge_impossible_tail`
- **bar_num**: Original bar indices (pipe-separated for merges)
- **usable**: 1 if tempo is within tolerance, 0 otherwise

---

## 4. Onset Detection

**File:** [loop_extractor/analysis/onset_detection.py](loop_extractor/analysis/onset_detection.py)

### Purpose
Detect precise timing of drum hits (onsets) from the drum stem for microtiming analysis.

### Algorithm

#### Step 4.1: Load and Normalize Audio

**Input:** `drums.wav` stem from Step 1

**Process:**
```python
# Load at 22050 Hz (standard for MIR research)
y, sr = librosa.load('drums.wav', sr=22050, mono=True)

# Peak normalize
peak = max(abs(y))
y = y / peak  # Normalize to [-1, 1]
```

**Why 22050 Hz?**
- Standard sample rate for music information retrieval
- Half of CD quality (44100 Hz) → 2× faster processing
- Still captures all musical frequencies (<11kHz)
- Time resolution: 1/22050 ≈ 0.045 ms per sample

#### Step 4.2: Compute Onset Strength Envelope

**Algorithm:** High-Frequency Content (HFC) onset detection

**Process:**
```python
# STFT parameters
hop_length = 512  # ~23 ms per frame
onset_env = librosa.onset.onset_strength(y, sr, hop_length)
```

**Onset Strength Formula:**
```
onset_strength(t) = Σ_k max(0, |X(t,k)| - |X(t-1,k)|) × k
```

Where:
- `X(t,k)`: STFT magnitude at time `t`, frequency bin `k`
- `k`: Frequency bin index (higher `k` → higher frequency)
- Weighted by frequency (high frequencies → stronger contribution)

**Why HFC?**
- Drum hits have strong high-frequency transients
- Differencing `|X(t,k)| - |X(t-1,k)|` emphasizes sudden changes
- Frequency weighting reduces low-frequency noise

**Output:**
- Onset strength envelope: 1D array of onset likelihood per frame
- Typical values: 0.0 (no onset) to 1.0 (strong onset)

#### Step 4.3: Peak Picking

**Algorithm:** Threshold-based peak detection

**Parameters:**
```python
delta = 0.12          # Minimum peak height
backtrack = False     # Don't backtrack to energy rise
```

**Process:**
```python
# Find local maxima in onset envelope that exceed delta
onset_frames = librosa.onset.onset_detect(
    onset_envelope=onset_env,
    sr=sr,
    hop_length=hop_length,
    delta=delta,
    backtrack=backtrack
)

# Convert frame indices to time in seconds
onset_times = librosa.frames_to_time(onset_frames, sr, hop_length)
```

**Peak Picking Logic:**
```
peak at frame t if:
  onset_env(t) > onset_env(t-1)  AND
  onset_env(t) > onset_env(t+1)  AND
  onset_env(t) >= delta
```

**Why delta=0.12?**
- Lower values → more sensitive (detects softer hits)
- Higher values → less sensitive (only strong hits)
- 0.12 is empirically optimal for drum stems (from librosa defaults)

#### Step 4.4: Optional Refinement (Disabled by Default)

**Process** (if `refine_onsets=True`):
1. **Peak refinement:** Search ±20ms window around each onset for local maximum in waveform
2. **Duplicate filtering:** Remove onsets closer than `min_interval_s` (default 150ms)

**Why Disabled?**
- Frame-based timing is already accurate (~23ms resolution)
- Refinement can introduce false positives
- 150ms minimum interval removes valid double strokes

#### Step 4.5: Save to CSV

**Output Format:**
```csv
onset_times
0.023220
0.488372
0.953488
1.418605
...
```

**Typical Output:**
- 3-minute track → 300-500 onsets
- Depends on drum density (sparse vs. busy drumming)
- File size: <50 KB

**Time Resolution:**
- Frame-based: ~23 ms (limited by hop_length=512)
- Sample-based (if refined): ~0.045 ms
- Sufficient for microtiming analysis (human JND ~10-30ms)

---

## 5. Pattern Length Estimation

**File:** [loop_extractor/analysis/pattern_detection.py](loop_extractor/analysis/pattern_detection.py)

### Purpose
Automatically detect the repeating pattern length (1, 2, 4, or 8 bars) using three independent methods.

### Overview

**Three Methods:**
1. **Drum Onset Method**: Analyzes onset timing patterns
2. **Mel-Band Method**: Analyzes spectral patterns in drum audio
3. **Bass Pitch Method**: Analyzes pitch contour patterns in bass

**Output:** Dictionary `{'drum': 4, 'mel': 2, 'pitch': 4}`

---

### Method 1: Drum Onset Method

#### Algorithm: Circular Cross-Correlation of Binary Onset Vectors

**Step 1: Create Bar Vectors**

For each bar, create a binary vector indicating onset presence:

```python
def drum_bar_vector(onsets, bar_start, bar_end, bins=16):
    """Create binary vector for one bar."""
    vector = zeros(16)  # 16th note grid

    # Find onsets in this bar
    bar_onsets = onsets[(onsets >= bar_start) & (onsets < bar_end)]

    # Map each onset to nearest grid position
    for onset in bar_onsets:
        phase = (onset - bar_start) / (bar_end - bar_start)
        tick = floor(phase * 16)
        vector[tick] = 1.0  # Mark presence (no accumulation)

    return vector
```

**Example:**
```
Bar 0: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  ← 4 onsets
Bar 1: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  ← same pattern
Bar 2: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  ← same pattern
```

**Step 2: Compute Similarity Matrix**

For all bar pairs `(i, j)`, compute circular cross-correlation:

```python
def circular_xcorr_max(v1, v2):
    """Max correlation over all circular shifts."""
    # FFT-based circular convolution
    V1 = fft(v1)
    V2 = fft(v2)
    corr = ifft(V1 × conj(V2))

    # Normalize by vector norms
    return max(corr) / (||v1|| × ||v2||)
```

**Mathematical Formula:**
```
sim(i,j) = max_shift [ Σ_k v1[k] × v2[(k + shift) mod 16] ] / (||v1|| × ||v2||)
```

**Why Circular?**
- Allows pattern matching with phase shift (e.g., downbeat on tick 0 vs. tick 8)
- Captures rhythmic similarity regardless of starting position

**Step 3: Bar-Lag Profile**

For each lag `L` (1 to N-1 bars):

```python
lag_profile[L] = median(sim[i, i+L] for all i)
```

**Example:**
```
Lag 1: median([sim(0,1), sim(1,2), sim(2,3), ...]) = 0.45
Lag 2: median([sim(0,2), sim(1,3), sim(2,4), ...]) = 0.62
Lag 3: median([sim(0,3), sim(1,4), sim(2,5), ...]) = 0.31
Lag 4: median([sim(0,4), sim(1,5), sim(2,6), ...]) = 0.89  ← HIGH!
Lag 5: median([sim(0,5), sim(1,6), sim(2,7), ...]) = 0.28
...
```

**Interpretation:**
- High similarity at lag 4 → pattern repeats every 4 bars
- Lower similarity at other lags → pattern doesn't repeat at those intervals

**Step 4: Choose Best Power-of-2 Length**

```python
def choose_pow2_length(lags, lag_profile):
    """Select best L from [1, 2, 4, 8]."""
    candidates = [1, 2, 4, 8]

    # Find local maxima among candidates
    maxima = []
    for L in candidates:
        if L == 1:
            continue  # Skip L=1

        idx = find_index(lags, L)
        if is_local_max(lag_profile, idx):
            maxima.append((lag_profile[idx], L))

    # Return smallest L with highest similarity
    if maxima:
        best_value = max(v for v, L in maxima)
        return min(L for v, L in maxima if v == best_value)

    # Fallback: smallest L with globally highest value
    return min(L for L in candidates if max_value_at_L)
```

**Example Decision:**
```
Lag profile at power-of-2 positions:
  L=1: 0.45 (not local max)
  L=2: 0.62 (local max)
  L=4: 0.89 (local max, HIGHEST)
  L=8: 0.71 (local max)

→ Best L = 4 (highest local max)
```

---

### Method 2: Mel-Band Method

#### Algorithm: Spectral Pattern Correlation

**Step 1: Compute Log-Mel Spectrogram**

```python
# Load drums.wav at 22050 Hz
y, sr = librosa.load('drums.wav', sr=22050)

# Compute mel-spectrogram
S_mel = librosa.feature.melspectrogram(
    y=y, sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=48,
    fmin=20, fmax=10000
)

# Convert to dB and normalize
D = librosa.power_to_db(S_mel)
D_norm = (D - D.min()) / (D.max() - D.min())
```

**Output:** Normalized spectrogram `(48 mel_bins, time_frames)`

**Step 2: Extract Bar Patches**

For each bar, extract and resample spectrogram patch to fixed size:

```python
def mel_bar_patch(D, bar_start, bar_end, target_frames=64):
    """Extract and resample bar patch to fixed size."""
    # Convert times to frame indices
    fps = sr / hop_length  # Frames per second
    start_frame = int(bar_start * fps)
    end_frame = int(bar_end * fps)

    # Extract patch
    patch = D[:, start_frame:end_frame]  # Shape: (48, variable)

    # Resample to fixed width
    patch_resampled = interpolate(patch, target_width=64)

    return patch_resampled  # Shape: (48, 64)
```

**Why Resample?**
- Bars have different durations due to tempo variation
- Fixed size enables direct comparison
- 64 frames ≈ enough resolution for 4 beats

**Step 3: Compute Similarity Matrix**

For each mel band, compute circular cross-correlation over time:

```python
def mel_circular_xcorr(patch1, patch2):
    """Band-wise circular correlation."""
    M, T = patch1.shape  # (48 mel_bins, 64 frames)

    total_corr = 0.0
    total_norm = 0.0

    for m in range(M):  # For each mel band
        band1 = patch1[m]
        band2 = patch2[m]

        # Circular correlation (FFT-based)
        corr = ifft(fft(band1) × conj(fft(band2)))

        # Accumulate
        total_corr += max(corr)
        total_norm += ||band1|| × ||band2||

    return total_corr / total_norm
```

**Why Band-Wise?**
- Each frequency band captures different timbral features
- Low frequencies: bass drum kicks
- Mid frequencies: snare, toms
- High frequencies: hi-hats, cymbals
- Summing across bands = robust to partial pattern matches

**Step 4: Lag Profile and Selection**

Same as Drum Onset Method:
- Compute `lag_profile[L] = median(sim[i, i+L])`
- Select best power-of-2 length

---

### Method 3: Bass Pitch Method

#### Algorithm: F0 Contour Correlation

**Step 1: Extract F0 (Fundamental Frequency)**

```python
# Load bass.wav
x, sr = soundfile.read('bass.wav')

# Extract F0 using libf0 (Melodia algorithm)
f0, time_coefs, salience = libf0.melodia_f0(
    x, Fs=sr,
    N=2048,        # FFT size
    H=256,         # Hop length
    F_min=55,      # Lowest bass note (A1)
    F_max=1760     # Highest relevant (A6)
)
```

**Melodia Algorithm (Salamon & Gómez, 2012):**
1. Compute multi-resolution STFT
2. Extract salience function (pitch likelihood)
3. Track pitch contours across time
4. Select most salient contour as melody

**Output:**
- `f0`: Array of F0 values in Hz (unvoiced regions = 0)
- `time_coefs`: Time stamps in seconds
- Typical hop: 256 samples / 44100 Hz ≈ 5.8 ms resolution

**Step 2: Normalize Pitch Vectors**

For each bar, create normalized pitch vector:

```python
def normalize_pitch_vector(f0, times, bar_start, bar_end, bins=16):
    """Normalize pitch contour for one bar."""
    # Filter to this bar
    mask = (times >= bar_start) & (times < bar_end) & (f0 > 0)

    if not any(mask):
        return zeros(16)  # Silent bar

    f0_bar = f0[mask]
    times_bar = times[mask]

    # Interpolate to fixed grid
    grid_times = linspace(bar_start, bar_end, 16)
    f0_interp = interp(grid_times, times_bar, f0_bar)

    # Z-score normalization
    f0_norm = (f0_interp - mean(f0_interp)) / std(f0_interp)

    return f0_norm
```

**Why Normalize?**
- Removes absolute pitch (key) differences
- Captures pitch contour shape
- Z-score makes different bars comparable

**Example:**
```
Bar 0 F0: [110, 110, 165, 165, ...]  Hz  (A2, A2, E3, E3, ...)
Bar 0 Normalized: [-1.2, -1.2, 0.8, 0.8, ...]

Bar 1 F0: [110, 110, 165, 165, ...]  Hz  (same pattern)
Bar 1 Normalized: [-1.2, -1.2, 0.8, 0.8, ...]  ← matches Bar 0!
```

**Step 3: Compute Similarity with Circular Shifts**

```python
def pitch_circular_xcorr(v1, v2, tsig=4):
    """Circular correlation with beat-aligned shifts."""
    n = len(v1)  # 16
    step = n // tsig  # 4 (for 4/4 time)

    max_sim = 0.0

    # Try shifts aligned with beats (0, 4, 8, 12)
    for shift in [0, step, 2*step, 3*step]:
        v2_shifted = roll(v2, shift)
        sim = dot(v1, v2_shifted) / (||v1|| × ||v2||)
        max_sim = max(max_sim, sim)

    return max_sim
```

**Why Beat-Aligned Shifts?**
- Bass lines often repeat on beat boundaries
- Reduces computational cost (4 shifts instead of 16)
- Musically meaningful alignments

**Step 4: Lag Profile and Selection**

Same process as other methods.

---

### Combining Methods

**Independence:**
- Drum onset: Rhythmic timing patterns
- Mel-band: Timbral/spectral patterns
- Bass pitch: Harmonic/melodic patterns

**Consensus Strategy:**
```python
pattern_lengths = {
    'drum': 4,
    'mel': 2,
    'pitch': 4
}

# User can choose:
# - Trust most common value (4 in this case)
# - Use method-specific loops for specialized analysis
# - Inspect lag profiles manually
```

**Typical Results:**
- **All methods agree**: High confidence in pattern length
- **Methods disagree**: Song may have multiple pattern levels (e.g., 2-bar bass, 4-bar drums)
- **Fallback**: Use default L=4 (most Western pop music has 4-bar phrases)

---

## 6. Grid Creation and Correction

**File:** [loop_extractor/analysis/raster.py](loop_extractor/analysis/raster.py)

### Purpose
Calculate precise timing deviations (phases) of drum onsets from expected grid positions, using multiple correction methods.

### Core Concept

**Grid:** Expected timing positions based on corrected downbeats
- 16 positions per bar (16th note resolution)
- Grid positions: 0, 1, 2, ..., 15 (per bar)

**Phase:** Timing deviation of actual onset from nearest grid position
- Phase = (onset_time - grid_time) / grid_step
- Range: -0.5 to +0.5 (wraps around grid positions)
- Phase = 0 → perfect alignment
- Phase > 0 → late (onset after grid)
- Phase < 0 → early (onset before grid)

---

### Step 6.1: Uncorrected Grid

**Process:**

1. **Load Data:**
   - Corrected downbeats: `[t₁, t₂, t₃, ..., tₙ]`
   - Drum onsets: `[o₁, o₂, o₃, ..., oₘ]`

2. **For each bar `i`:**
   ```python
   bar_start = downbeats[i]
   bar_end = downbeats[i+1]
   bar_duration = bar_end - bar_start

   # Create 16 grid positions
   grid_times = [bar_start + (k/16) * bar_duration for k in range(16)]
   ```

3. **For each onset in bar:**
   ```python
   # Find nearest grid position
   phase = (onset - bar_start) / bar_duration
   nearest_tick = round(phase * 16)

   # Calculate phase deviation
   grid_time = grid_times[nearest_tick]
   phase_dev = (onset - grid_time) / (bar_duration / 16)
   ```

**Example:**
```
Bar 0: start=0.0s, end=2.0s, duration=2.0s
Grid step = 2.0 / 16 = 0.125s

Grid positions (seconds):
  Tick 0:  0.000s (downbeat)
  Tick 1:  0.125s
  Tick 2:  0.250s
  ...
  Tick 15: 1.875s

Onset at 0.132s:
  Nearest tick: 1 (grid time = 0.125s)
  Phase: (0.132 - 0.125) / 0.125 = +0.056 (late by 7ms)
```

**Problem:** Uses raw downbeats from Beat-Transformer
- May have tempo drift across bars
- No correction for systematic timing offsets

---

### Step 6.2: Per-Snippet Correction

**Purpose:** Apply single timing offset to entire snippet

**Algorithm:**

1. **Find Reference Offset:**
   ```python
   def find_reference_offset(downbeats, onsets, first_bar, num_bars=1):
       """Search for reference onset at priority ticks."""
       priority_ticks = [0, 8, 4, 12]  # Downbeat, backbeat, quarters

       for tick in priority_ticks:
           for bar in range(num_bars):
               # Calculate expected grid position
               grid_time = calculate_grid_time(downbeats, bar, tick)

               # Find nearest onset
               nearest_onset, distance = find_nearest(grid_time, onsets)

               # Check tolerance (50% of grid step)
               if distance < 0.5 * grid_step:
                   ref_offset = nearest_onset - grid_time
                   return ref_offset  # In seconds

       return 0.0  # No reference found
   ```

2. **Apply Correction:**
   ```python
   # Shift ALL grid times by reference offset
   corrected_grid_time = uncorrected_grid_time + ref_offset

   # Recalculate phases
   phase_corrected = (onset - corrected_grid_time) / grid_step
   ```

**Example:**
```
Reference found at tick 0, bar 0:
  Grid time: 0.000s
  Nearest onset: 0.015s
  Ref offset: +0.015s (15ms late)

Apply to all grids:
  Tick 0, Bar 0: 0.000s → 0.015s
  Tick 1, Bar 0: 0.125s → 0.140s
  Tick 0, Bar 1: 2.000s → 2.015s
  ...

Onset at 0.132s:
  Corrected grid (tick 1): 0.140s
  Phase: (0.132 - 0.140) / 0.125 = -0.064 (early by 8ms)
```

**Characteristics:**
- **Pros**: Simple, works for short snippets with stable tempo
- **Cons**: Doesn't adapt to tempo drift over time

---

### Step 6.3: Loop-Based Correction (Drum, Mel, Pitch, Standard L=1/2/4)

**Purpose:** Apply separate offset correction for each repeating loop

**Key Innovation: Equidistant Grid**

Instead of using detected downbeats (which may vary), create a perfectly even grid:

```python
def create_equidistant_grid(downbeats, loop_start, pattern_len):
    """Create evenly-spaced grid for one loop."""
    # Total loop duration
    loop_start_time = downbeats[loop_start]
    loop_end_time = downbeats[loop_start + pattern_len]
    loop_duration = loop_end_time - loop_start_time

    # Divide into equal sixteenths
    total_sixteenths = pattern_len * 16
    sixteenth_duration = loop_duration / total_sixteenths

    # Generate grid
    grid = [loop_start_time + k * sixteenth_duration
            for k in range(total_sixteenths + 1)]

    return grid, sixteenth_duration
```

**Example (4-bar loop):**
```
Detected downbeats:
  Bar 0: 0.000s
  Bar 1: 2.010s  (slightly late)
  Bar 2: 3.995s  (slightly early)
  Bar 3: 6.005s  (slightly late)
  Bar 4: 8.000s

Loop duration: 8.000 - 0.000 = 8.000s
Sixteenth duration: 8.000 / (4 × 16) = 0.125s

Equidistant grid:
  Tick 0:  0.000s (bar 0 start)
  Tick 1:  0.125s
  ...
  Tick 16: 2.000s (bar 1 start, CORRECTED from 2.010s)
  Tick 17: 2.125s
  ...
  Tick 32: 4.000s (bar 2 start, CORRECTED from 3.995s)
  ...
```

**Why Equidistant?**
- Assumes tempo is constant within each loop
- Removes cumulative timing errors from beat detection
- Pattern repetition means bars should have equal duration

**Algorithm:**

1. **Divide snippet into loops:**
   ```python
   num_loops = (last_bar - first_bar + 1) // pattern_len

   for loop_idx in range(num_loops):
       loop_start = first_bar + (loop_idx * pattern_len)
       process_loop(loop_start, pattern_len)
   ```

2. **For each loop, find reference offset:**
   ```python
   def find_loop_reference(downbeats, onsets, loop_start, pattern_len):
       """Find reference offset for this loop."""
       # Create equidistant grid
       grid, step = create_equidistant_grid(downbeats, loop_start, pattern_len)

       # Search priority ticks in ALL bars of loop
       priority_ticks = [0, 8, 4, 12]

       for tick in priority_ticks:
           for bar_offset in range(pattern_len):
               global_tick = (bar_offset * 16) + tick
               grid_time = grid[global_tick]

               nearest_onset, distance = find_nearest(grid_time, onsets)

               if distance < 0.5 * step:
                   ref_offset = nearest_onset - grid_time
                   return ref_offset

       return 0.0
   ```

3. **Apply correction to all onsets in loop:**
   ```python
   corrected_grid = [grid[k] + ref_offset for k in range(len(grid))]

   for onset in onsets_in_loop:
       # Find nearest corrected grid position
       nearest_tick = argmin(|onset - corrected_grid[k]|)

       # Calculate phase
       phase = (onset - corrected_grid[nearest_tick]) / step
   ```

**Example (4-bar loop, drum method):**
```
Loop 0 (bars 0-3):
  Equidistant grid created
  Reference found at tick 0: offset = +0.015s
  All grid times shifted by +0.015s
  Phases calculated for bars 0-3

Loop 1 (bars 4-7):
  New equidistant grid created
  Reference found at tick 8: offset = -0.008s
  All grid times shifted by -0.008s
  Phases calculated for bars 4-7

→ Each loop gets independent correction
→ Adapts to tempo changes between loops
```

**Methods Using Loop-Based Correction:**

1. **Drum Method**: `pattern_len` = auto-detected from drum onsets (e.g., 4)
2. **Mel Method**: `pattern_len` = auto-detected from spectrogram (e.g., 2)
3. **Pitch Method**: `pattern_len` = auto-detected from bass pitch (e.g., 4)
4. **Standard L=1**: `pattern_len` = 1 (1-bar loop, max correction granularity)
5. **Standard L=2**: `pattern_len` = 2 (2-bar loop)
6. **Standard L=4**: `pattern_len` = 4 (4-bar loop, common phrase length)

**Trade-offs:**

| Method | Pattern Length | Correction Frequency | Use Case |
|--------|----------------|---------------------|----------|
| Per-snippet | N/A (single offset) | Once per snippet | Short, stable tempo |
| Standard L=1 | 1 bar | Every bar | Maximum adaptation, may over-correct |
| Standard L=2 | 2 bars | Every 2 bars | Balanced |
| Standard L=4 | 4 bars | Every 4 bars | Common phrase length |
| Drum/Mel/Pitch | Auto (1-8 bars) | Every pattern | Adapts to song structure |

---

### Step 6.4: Output Format

**Comprehensive Phases CSV:**

```csv
onset_time(s),bar,tick_uncorrected,phase_uncorrected,grid_time_uncorrected(s),phase_uncorrected(ms),...
0.015234,0,0,0.1219,0.000000,15.234,...
0.132451,0,1,0.0596,0.125000,7.451,...
...
```

**Columns** (for each method):
- `onset_time(s)`: Original onset time
- `bar`: Bar number containing onset
- `tick_uncorrected`: Grid position (0-15)
- `phase_uncorrected`: Phase deviation (dimensionless)
- `grid_time_uncorrected(s)`: Expected grid time
- `phase_uncorrected(ms)`: Phase deviation in milliseconds
- `phase_per_snippet`: Per-snippet corrected phase
- `grid_time_per_snippet(s)`: Per-snippet corrected grid time
- `phase_drum(L=4)`: Drum method phase (L=4 pattern)
- `grid_time_drum(L=4)(s)`: Drum method grid time
- ... (similar for mel, pitch, standard_L1, standard_L2, standard_L4)

**Usage:**
- Compare phase columns to see which method provides best correction
- Lower phase deviation (closer to 0) = better grid alignment
- RMS of phases quantifies overall correction quality

---

## 7. Microtiming Deviation Plots

**File:** [loop_extractor/utils/microtiming_plots.py](loop_extractor/utils/microtiming_plots.py)

### Purpose
Visualize onset deviations from the metrical grid using pattern-folded raster plots, revealing microtiming characteristics across multiple loops and correction methods.

### Method

**Pattern Folding:**
1. **Data Organization**: Divide bars into complete loops based on pattern length (L=1, L=2, or L=4 bars)
2. **Fold into Pattern**: Map each loop's onsets to 16th-note positions within the pattern (0 to L×16)
3. **Calculate Deviations**: `deviation_ms = (onset_time - grid_time) × 1000`
4. **Visualize Multiple Loops**: Plot each loop as a colored line showing how onsets deviate from grid

**Key Features:**
- **X-axis**: 16th-note positions within pattern (0-16 for L=1, 0-32 for L=2, 0-64 for L=4)
- **Y-axis**: Deviation in milliseconds (negative = early, positive = late)
- **Multiple loops**: Each complete loop shown as separate colored line
- **Grid reference**: Horizontal red dashed line at y=0 represents perfect grid alignment
- **Bar boundaries**: Vertical gray dotted lines every 16 ticks mark bar divisions

**Implementation Details:**
```python
# Filter finite values only
mask = np.isfinite(deviations)
xs = ticks[mask]
ys = deviations[mask]

# Sort by tick position for proper line connections
order_idx = np.argsort(xs)
xs_sorted = xs[order_idx]
ys_sorted = ys[order_idx]

# Single plot call with marker parameter
ax.plot(xs_sorted, ys_sorted, linewidth=1.5, alpha=0.7, color=color,
       marker='o', markersize=4, markerfacecolor=color, markeredgecolor='none',
       label=f'Loop {i+1} (Bars {start_bar}-{end_bar})')
```

### Output

**File:** `5_grid/{track_id}_microtiming_plots.pdf`

**5 Plots (one page each):**
1. **Uncorrected**: Raw onset deviations vs uncorrected grid (L=4 folding)
2. **Per-Snippet**: Deviations after single snippet-wide offset correction (L=4 folding)
3. **Standard L=1**: Deviations with 1-bar loop correction (L=1 folding)
4. **Standard L=2**: Deviations with 2-bar loop correction (L=2 folding)
5. **Standard L=4**: Deviations with 4-bar loop correction (L=4 folding)

Each plot includes:
- RMS value in title showing overall deviation magnitude
- Legend identifying each loop by bar range
- Grid reference line (y=0)
- Bar boundary markers

### Interpretation

**What to Look For:**
- **Systematic patterns**: Consistent deviation shapes across loops reveal intentional microtiming (groove)
- **Loop consistency**: Similar shapes between loops suggest stable performance
- **Loop variation**: Different shapes indicate expressive timing changes or performance variation
- **RMS values**: Lower RMS indicates better grid alignment (more "quantized" feel)
- **Deviation magnitude**: Typical values 10-50ms; <10ms often imperceptible, >50ms noticeable

**Comparison Across Methods:**
- **Uncorrected**: Shows raw timing including tempo drift
- **Per-Snippet**: Removes global offset but keeps within-snippet timing intact
- **Standard L=1/L=2/L=4**: Different pattern lengths reveal timing at different structural levels

**Musical Insight:**
These plots reveal the "human feel" in music - the subtle timing deviations that make performances groovy. Perfectly quantized music would show all loops as straight lines at y=0, while expressive performances show characteristic deviation patterns that repeat across loops.

---

## Summary

This pipeline combines state-of-the-art machine learning (Beat-Transformer, Spleeter) with signal processing (onset detection, pitch tracking) and music theory (tempo correction, pattern detection) to provide comprehensive microtiming analysis.

**Key Innovations:**
1. **Multi-stem beat detection**: More robust than mono audio
2. **Automatic tempo error correction**: Fixes factor-of-2 errors
3. **Equidistant grid**: Removes cumulative timing errors
4. **Multiple correction methods**: Adapts to different musical patterns
5. **Pattern length estimation**: Uses three independent methods for robustness

**Applications:**
- Music information retrieval research
- Groove analysis and quantification
- DAW-ready loop extraction
- MIDI timing export
- Comparative musicology studies

---

## References

1. Zhao, J., Xia, G., & Wang, Y. (2022). **Beat Transformer: Demixed Beat and Downbeat Tracking with Dilated Self-Attention.** ISMIR 2022.

2. Hennequin, R., Khlif, A., Voituret, F., & Moussallam, M. (2020). **Spleeter: A Fast and Efficient Music Source Separation Tool with Pre-trained Models.** Journal of Open Source Software, 5(50), 2154.

3. Böck, S., Krebs, F., & Schedl, M. (2016). **Evaluating the Online Capabilities of Onset Detection Methods.** ISMIR 2012.

4. Salamon, J., & Gómez, E. (2012). **Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics.** IEEE Transactions on Audio, Speech, and Language Processing, 20(6), 1759-1770.

5. Rosenzweig, S., Schwär, S., & Müller, M. (2022). **libf0: A Python Library for Fundamental Frequency Estimation in Music Recordings.** ISMIR 2022 Late-Breaking/Demo.
