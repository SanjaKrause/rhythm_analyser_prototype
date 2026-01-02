# Loop Extractor Pipeline - Mermaid Diagram

**Author:** Alexander Krause, TU Berlin
**Co-Author:** Claude Code (Anthropic)

---

## Main Pipeline Flow

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e1f5ff','primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','secondaryColor':'#fff4e1','tertiaryColor':'#ffe1f5','clusterBkg':'#f9f9f9','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
flowchart TD
    Start([Mixed Audio<br/>WAV/MP3]) --> Step1

    subgraph Step1["<b>STEP 1: STEM SEPARATION</b>"]
        A1[Spleeter 5-stem U-Net] --> A2[5 Stem WAV Files]
        A2 --> A3[vocals.wav]
        A2 --> A4[drums.wav]
        A2 --> A5[bass.wav]
        A2 --> A6[piano.wav]
        A2 --> A7[other.wav]
        A1 --> A8[Mel-Spectrogram Conversion<br/>44.1kHz, 4096 FFT, 128 mels]
        A8 --> A9[5-stem NPZ file<br/>Shape: 5, T, 128]
    end

    Step1 --> Step2
    Step1 --> Step4

    subgraph Step2["<b>STEP 2: BEAT DETECTION</b>"]
        B1[5-stem NPZ] --> B2[Beat-Transformer<br/>9-layer Dilated Transformer]
        B2 --> B3[Beat Activations]
        B2 --> B4[Downbeat Activations]
        B3 --> B5[Madmom DBN<br/>HMM, 55-215 BPM]
        B4 --> B5
        B5 --> B6[Raw Beats/Downbeats<br/>beat_pos, bar_num, times]
    end

    subgraph Step4["<b>STEP 4: ONSET DETECTION</b>"]
        D1[drums.wav] --> D2[Librosa HFC<br/>hop=512, delta=0.12]
        D2 --> D3[Onset Times CSV]
    end

    Step2 --> Step3

    subgraph Step3["<b>STEP 3: DOWNBEAT CORRECTION</b>"]
        C1[Raw Downbeats] --> C2[Bar Tempo Calculation]
        C2 --> C3[Factor-of-2 Classification<br/>normal/double/half]
        C3 --> C4{Determine<br/>Dominant Pattern}
        C4 -->|normal| C5[MERGE double bars<br/>SPLIT half bars]
        C4 -->|factor2| C5
        C5 --> C6[BPM Threshold Adjust]
        C6 --> C7[Corrected Downbeats TXT<br/>+ Time Signature<br/>+ Usable Bars Mask]
    end

    Step3 --> Step5
    Step3 --> Step6
    Step4 --> Step5
    Step4 --> Step6

    subgraph Step5["<b>STEP 5: PATTERN LENGTH DETECTION</b>"]
        E1[Method 1: Drum Onset<br/>Binary vectors + Circular xcorr] --> E4[Best L = 4 bars]
        E2[Method 2: Mel-Band<br/>Spectrograms + Band-wise corr] --> E5[Best L = 2 bars]
        E3[Method 3: Bass Pitch<br/>F0 extraction + Z-score norm] --> E6[Best L = 4 bars]
        E4 --> E7{Pattern Lengths<br/>drum: 4, mel: 2, pitch: 4}
        E5 --> E7
        E6 --> E7
    end

    subgraph Step6["<b>STEP 6: GRID CREATION & CORRECTION</b>"]
        F1[A: Uncorrected Grid<br/>Raw downbeats, 16 ticks/bar] --> F4[comprehensive_phases.csv]
        F2[B: Per-Snippet Correction<br/>1 ref offset for snippet] --> F4
        F3[C: Loop-Based Correction<br/>Equidistant grid per loop<br/>Methods: drum, mel, pitch, L1, L2, L4] --> F4
        F4 --> F5[Columns:<br/>onset_time, bar, tick<br/>phase_*, grid_time_*<br/>for all methods]
    end

    Step5 --> Step6
    Step6 --> Analysis

    subgraph Analysis["<b>ANALYSIS & OUTPUTS</b>"]
        G1[Raster Plots<br/>2 plots: comparison & standard]
        G2[Microtiming Plots<br/>Pattern-folded deviation plots]
        G3[RMS Calculation<br/>Quantify correction quality]
        G4[Audio Examples<br/>Click tracks for each method]
        G5[Tempo Analysis<br/>8-panel plots + CSV]
        G6[MIDI Export<br/>Onset + Bass Pitch]
        G7[Stem Loops<br/>Perfect loops with crossfade]
    end

    Analysis --> Final[pipeline_results.json<br/>Complete summary]

    style Step1 fill:#e1f5ff,stroke:#000,color:#000
    style Step2 fill:#fff4e1,stroke:#000,color:#000
    style Step3 fill:#ffe1f5,stroke:#000,color:#000
    style Step4 fill:#fff4e1,stroke:#000,color:#000
    style Step5 fill:#e1ffe1,stroke:#000,color:#000
    style Step6 fill:#f5e1ff,stroke:#000,color:#000
    style Analysis fill:#ffffcc,stroke:#000,color:#000
    style Final fill:#ffcccc,stroke:#000,color:#000
```

---

## Detailed Step 3: Downbeat Correction Logic

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
flowchart TD
    Start[Raw Downbeats] --> CalcTempo[Calculate Bar Tempos<br/>tempo = 60/duration × time_sig]
    CalcTempo --> Median[Calculate Median Tempo<br/>e.g., 120 BPM]

    Median --> Classify1[Initial Classification<br/>For each bar: classify vs median]

    Classify1 --> CheckHalf{Is tempo in<br/>half range?<br/>0.45× to 0.55× median}
    CheckHalf -->|Yes| ClassHalf1[Class: half]
    CheckHalf -->|No| CheckDouble{Is tempo in<br/>double range?<br/>1.8× to 2.2× median}
    CheckDouble -->|Yes| ClassDouble1[Class: double]
    CheckDouble -->|No| ClassNormal1[Class: normal<br/>Everything else]

    ClassNormal1 --> Count1
    ClassDouble1 --> Count1
    ClassHalf1 --> Count1[Count Classifications]

    Count1 --> BPMCheck{BPM Threshold Check:<br/>Are >50% bars > 135 BPM?}
    BPMCheck -->|Yes| Rebase[Calculate new base:<br/>median high bars / 2<br/>Example: 140 → 70 BPM]
    BPMCheck -->|No| UseCounts[Use initial counts]

    Rebase --> Reclassify[RECLASSIFY ALL bars<br/>using new base]
    Reclassify --> Count2[Recount<br/>Most become double]
    Count2 --> Dominant
    UseCounts --> Dominant

    Dominant{n_normal >=<br/>n_double + n_half?}
    Dominant -->|Yes| DomNormal[Dominant = normal]
    Dominant -->|No| DomFactor{n_double >= n_half?}
    DomFactor -->|Yes| DomDouble[Dominant = factor2<br/>Orientation = double]
    DomFactor -->|No| DomHalf[Dominant = factor2<br/>Orientation = half]

    DomNormal --> ApplyNormal[MERGE double-classified bars<br/>SPLIT half-classified bars<br/>Keep normal bars unchanged]
    DomDouble --> ApplyDouble[SPLIT normal-classified bars<br/>Keep double/half bars unchanged]
    DomHalf --> ApplyHalf[MERGE normal-classified bars<br/>Keep double/half bars unchanged]

    ApplyNormal --> Output[Corrected Downbeats]
    ApplyDouble --> Output
    ApplyHalf --> Output

    style Start fill:#e1f5ff
    style BPMCheck fill:#ffcccc
    style Rebase fill:#fff4e1
    style Reclassify fill:#ffe1f5
    style Output fill:#e1ffe1
```

### How 3-Beat and 5-Beat Bars are Corrected

The downbeat correction automatically fixes bars that were misdetected as having 3 or 5 beats when the dominant time signature is 4/4:

| Misdetected Bar | Actual Beats | Duration vs 4/4 | Tempo Classification | Correction Action |
|----------------|--------------|-----------------|---------------------|-------------------|
| **3-beat bar** | 3 beats | 75% of normal | ~1.33x tempo → "double" | **MERGED** with next bar to create one 4/4 bar |
| **5-beat bar** | 5 beats | 125% of normal | ~0.8x tempo → "half" | **SPLIT** into two bars (4+4 beats) |

**Example**: If BeatTransformer detects a sequence like:
```
Bar 1: 4 beats ✓
Bar 2: 3 beats ✗ (too short)
Bar 3: 1 beat   (remainder from merge)
Bar 4: 4 beats ✓
```

After correction:
```
Bar 1: 4 beats ✓
Bar 2: 4 beats ✓ (merged Bar 2 + Bar 3)
Bar 3: 4 beats ✓
```

This tempo-based classification and correction ensures consistent 4/4 bars for downstream analysis, even when the beat detection model makes occasional errors.

---

## Detailed Step 5: Pattern Length Detection Methods

**Important**: By default, the three methods analyze only **FULL bars within the snippet boundaries** (bars where both start AND end times fall within the snippet). This ensures pattern detection focuses on the analyzed region. You can optionally set `use_all_bars=True` to analyze all corrected bars regardless of snippet boundaries.

Each method creates an N×N similarity matrix comparing every bar to every other bar, then extracts the diagonal at lag L to measure periodicity. The median similarity at each lag determines the best pattern length.

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
flowchart LR
    subgraph Method1["<b>Drum Onset Method</b>"]
        D1[Full Bars in Snippet<br/>default] --> D2[Create Binary Vectors<br/>16 positions per bar]
        D2 --> D3[Compute Similarity Matrix<br/>N×N circular xcorr]
        D3 --> D4[Bar-Lag Profile<br/>median of diagonal L]
        D4 --> D5[Choose Best Power-of-2<br/>from 1,2,4,8 bars]
        D5 --> D6[Example: L = 4 bars]
    end

    subgraph Method2["<b>Mel-Band Method</b>"]
        M1[Full Bars in Snippet<br/>default] --> M2[Log-Mel Spectrogram<br/>48 mels, 22050 Hz]
        M2 --> M3[Extract Bar Patches<br/>Resample to 64 frames]
        M3 --> M4[Band-wise Circular XCorr<br/>N×N sum across 48 bands]
        M4 --> M5[Bar-Lag Profile<br/>median of diagonal L]
        M5 --> M6[Choose Best Power-of-2<br/>from 1,2,4,8 bars]
        M6 --> M7[Example: L = 2 bars]
    end

    subgraph Method3["<b>Bass Pitch Method</b>"]
        P1[Full Bars in Snippet<br/>default] --> P2[F0 Extraction<br/>Melodia Algorithm]
        P2 --> P3[Normalize Per Bar<br/>Z-score normalization]
        P3 --> P4[Beat-aligned XCorr<br/>N×N, shifts: 0,4,8,12]
        P4 --> P5[Bar-Lag Profile<br/>median of diagonal L]
        P5 --> P6[Choose Best Power-of-2<br/>from 1,2,4,8 bars]
        P6 --> P7[Example: L = 4 bars]
    end

    D6 --> Combine{Pattern Lengths}
    M7 --> Combine
    P7 --> Combine
    Combine --> Output[drum: 4<br/>mel: 2<br/>pitch: 4]

    style Method1 fill:#e1f5ff,stroke:#000,color:#000
    style Method2 fill:#ffe1f5,stroke:#000,color:#000
    style Method3 fill:#e1ffe1,stroke:#000,color:#000
```

---

## Method 1: Drum Onset Pattern Detection (Detailed)

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
flowchart TD
    Start[Input: Onset CSV + Bar Times] --> Filter{Filter Bars}
    Filter -->|Default| FullBars[Only FULL bars<br/>bar_start >= snippet_start<br/>bar_end <= snippet_end]
    Filter -->|use_all_bars=True| AllBars[All corrected bars]

    FullBars --> LoadOnsets[Load Onset Times<br/>from CSV]
    AllBars --> LoadOnsets

    LoadOnsets --> FilterOnsets[Filter Onsets<br/>to Snippet Range]

    FilterOnsets --> CreateVectors[For Each Bar:<br/>Create Binary Vector]

    subgraph VectorCreation["<b>Binary Vector Creation</b>"]
        VC1[Initialize 16 bins<br/>zeros array] --> VC2[For each onset in bar]
        VC2 --> VC3[Calculate relative position<br/>rel = onset - bar_start / bar_duration]
        VC3 --> VC4[Convert to bin index<br/>k = floor rel * 16]
        VC4 --> VC5[Set vector at k = 1.0<br/>Presence only, no accumulation]
        VC5 --> VC6{More onsets<br/>in this bar?}
        VC6 -->|Yes| VC2
        VC6 -->|No| VC7[Binary Vector:<br/>e.g., 1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0]
    end

    CreateVectors --> VectorCreation
    VectorCreation --> SimMatrix[Build N×N<br/>Similarity Matrix]

    subgraph Similarity["<b>Circular Cross-Correlation</b>"]
        S1[For each pair i,j:<br/>vectors vi, vj] --> S2[FFT both vectors<br/>Vi = fft vi<br/>Vj = fft vj]
        S2 --> S3[Multiply in frequency<br/>Vi * conj Vj]
        S3 --> S4[Inverse FFT<br/>corr = ifft Vi * conj Vj]
        S4 --> S5[Extract beat-aligned shifts<br/>every 4 positions 0,4,8,12]
        S5 --> S6[Find maximum correlation<br/>max_corr = max corr]
        S6 --> S7[Normalize<br/>sim_i,j = max_corr / norm vi * norm vj]
    end

    SimMatrix --> Similarity
    Similarity --> LagProfile[Extract Bar-Lag Profile]

    subgraph LagExtraction["<b>Lag Profile Extraction</b>"]
        L1[For lag = 1 to N-1] --> L2[Get diagonal L<br/>sim_0,L, sim_1,L+1, ..., sim_N-L,N]
        L2 --> L3[Compute median<br/>profile_L = median diagonal_L]
        L3 --> L4{More lags?}
        L4 -->|Yes| L1
        L4 -->|No| L5[Profile:<br/>lag → similarity]
    end

    LagProfile --> LagExtraction
    LagExtraction --> Choose[Choose Best Power-of-2]

    subgraph PowerOf2["<b>Power-of-2 Selection</b>"]
        P1[Filter to L in 1,2,4,8] --> P2[Find local maxima<br/>profile_L-1 < profile_L > profile_L+1]
        P2 --> P3{Local maxima<br/>exist exclude L=1?}
        P3 -->|Yes| P4[Return smallest L<br/>with best value]
        P3 -->|No| P5[Return smallest L<br/>with globally best value]
        P4 --> P6[Best L<br/>e.g., L = 4 bars]
        P5 --> P6
    end

    Choose --> PowerOf2
    PowerOf2 --> Output[drum: L]

    style VectorCreation fill:#e1f5ff,stroke:#000,color:#000
    style Similarity fill:#ffe1f5,stroke:#000,color:#000
    style LagExtraction fill:#e1ffe1,stroke:#000,color:#000
    style PowerOf2 fill:#fff4e1,stroke:#000,color:#000
    style Output fill:#ffcccc,stroke:#000,color:#000
```

---

## Method 2: Mel-Band Pattern Detection (Detailed)

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
flowchart TD
    Start[Input: drums.wav + Bar Times] --> Filter{Filter Bars}
    Filter -->|Default| FullBars[Only FULL bars<br/>bar_start >= snippet_start<br/>bar_end <= snippet_end]
    Filter -->|use_all_bars=True| AllBars[All corrected bars]

    FullBars --> LoadAudio[Load Audio<br/>sr=22050 Hz, mono]
    AllBars --> LoadAudio

    LoadAudio --> ComputeMel[Compute Log-Mel Spectrogram]

    subgraph MelComputation["<b>Mel-Spectrogram Computation</b>"]
        M1[STFT Parameters:<br/>n_fft=2048<br/>hop=512<br/>n_mels=48] --> M2[Mel filterbank<br/>20 Hz - 10 kHz]
        M2 --> M3[Power spectrogram<br/>S_mel = mel_filter @ abs STFT squared]
        M3 --> M4[Convert to dB<br/>D = 10 * log10 S_mel]
        M4 --> M5[Min-Max Normalize<br/>D_norm = D - min / max - min]
        M5 --> M6[Result: 48 × T matrix<br/>48 mel bands, T time frames]
    end

    ComputeMel --> MelComputation
    MelComputation --> ExtractPatches[For Each Bar:<br/>Extract & Resample Patch]

    subgraph PatchExtraction["<b>Bar Patch Extraction</b>"]
        PE1[Convert bar times to frames<br/>fps = sr / hop_length<br/>a = floor t0 * fps<br/>b = ceil t1 * fps] --> PE2[Extract patch<br/>P = D_norm mels, a:b]
        PE2 --> PE3[Resample to fixed 64 frames<br/>Time warping via interpolation]
        PE3 --> PE4{Bar duration<br/>varies?}
        PE4 -->|Yes| PE5[Stretch/compress<br/>to 64 frames]
        PE4 -->|No| PE6[Already 64 frames]
        PE5 --> PE7[Result: 48 × 64 patch<br/>Normalized bar representation]
        PE6 --> PE7
    end

    ExtractPatches --> PatchExtraction
    PatchExtraction --> SimMatrix[Build N×N<br/>Similarity Matrix]

    subgraph Similarity["<b>Band-wise Circular XCorr</b>"]
        S1[For each pair i,j:<br/>patches Pi, Pj<br/>each 48 × 64] --> S2[Initialize accumulator<br/>acc = zeros 64<br/>denom = 0]
        S2 --> S3[For each mel band m=0..47]
        S3 --> S4[Get band vectors<br/>ri = Pi_m,:, qj = Pj_m,:]
        S4 --> S5[FFT both vectors<br/>Ri = fft ri<br/>Qj = fft qj]
        S5 --> S6[Circular correlation<br/>corr = ifft Ri * conj Qj]
        S6 --> S7[Accumulate<br/>acc += corr<br/>denom += norm ri * norm qj]
        S7 --> S8{More bands?}
        S8 -->|Yes| S3
        S8 -->|No| S9[Find maximum shift<br/>max_corr = max acc]
        S9 --> S10[Normalize across all bands<br/>sim_i,j = max_corr / denom]
    end

    SimMatrix --> Similarity
    Similarity --> LagProfile[Extract Bar-Lag Profile]

    subgraph LagExtraction["<b>Lag Profile Extraction</b>"]
        L1[For lag = 1 to N-1] --> L2[Get diagonal L<br/>sim_0,L, sim_1,L+1, ..., sim_N-L,N]
        L2 --> L3[Compute median<br/>profile_L = median diagonal_L]
        L3 --> L4{More lags?}
        L4 -->|Yes| L1
        L4 -->|No| L5[Profile:<br/>lag → similarity]
    end

    LagProfile --> LagExtraction
    LagExtraction --> Choose[Choose Best Power-of-2]

    subgraph PowerOf2["<b>Power-of-2 Selection</b>"]
        P1[Filter to L in 1,2,4,8] --> P2[Find local maxima<br/>profile_L-1 < profile_L > profile_L+1]
        P2 --> P3{Local maxima<br/>exist exclude L=1?}
        P3 -->|Yes| P4[Return smallest L<br/>with best value]
        P3 -->|No| P5[Return smallest L<br/>with globally best value]
        P4 --> P6[Best L<br/>e.g., L = 2 bars]
        P5 --> P6
    end

    Choose --> PowerOf2
    PowerOf2 --> Output[mel: L]

    style MelComputation fill:#e1f5ff,stroke:#000,color:#000
    style PatchExtraction fill:#ffe1f5,stroke:#000,color:#000
    style Similarity fill:#e1ffe1,stroke:#000,color:#000
    style LagExtraction fill:#fff4e1,stroke:#000,color:#000
    style PowerOf2 fill:#f5e1ff,stroke:#000,color:#000
    style Output fill:#ffcccc,stroke:#000,color:#000
```

---

## Method 3: Bass Pitch Pattern Detection (Detailed)

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
flowchart TD
    Start[Input: bass.wav + Bar Times] --> Filter{Filter Bars}
    Filter -->|Default| FullBars[Only FULL bars<br/>bar_start >= snippet_start<br/>bar_end <= snippet_end]
    Filter -->|use_all_bars=True| AllBars[All corrected bars]

    FullBars --> LoadAudio[Load Audio Snippet]
    AllBars --> LoadAudio

    LoadAudio --> F0Extract[F0 Extraction<br/>Melodia Algorithm]

    subgraph F0Extraction["<b>F0 Extraction Melodia</b>"]
        F1[Load bass.wav<br/>Snippet region only] --> F2[STFT Parameters:<br/>n_fft=2048<br/>hop=256<br/>sr=original]
        F2 --> F3[Compute Salience<br/>55 Hz - 1760 Hz]
        F3 --> F4[Salience = Sum of harmonic peaks<br/>weighted by amplitude]
        F4 --> F5[Peak extraction<br/>per time frame]
        F5 --> F6[Voicing detection<br/>threshold salience]
        F6 --> F7[Output F0 time series<br/>f0 t, unvoiced = 0 or NaN]
        F7 --> F8[Save to bass_f0.csv<br/>time, f0_hz columns]
    end

    F0Extract --> F0Extraction
    F0Extraction --> CreateVectors[For Each Bar:<br/>Create Pitch Vector]

    subgraph VectorCreation["<b>Pitch Vector Normalization</b>"]
        VC1[Filter F0 to bar range<br/>t0 <= t < t1<br/>f0 > 0, finite] --> VC2{Any pitch<br/>in this bar?}
        VC2 -->|No| VC3[Return zeros 16]
        VC2 -->|Yes| VC4[Get segment times & F0<br/>seg_t, seg_f0]
        VC4 --> VC5[Resample to 16 bins<br/>Linearly interpolate<br/>f_res = interp x_new, seg_t, seg_f0]
        VC5 --> VC6[Z-score normalize<br/>f_res -= mean f_res<br/>f_res /= std f_res + 1e-9]
        VC6 --> VC7[Result: 16-element vector<br/>Normalized pitch contour]
        VC3 --> VC7
    end

    CreateVectors --> VectorCreation
    VectorCreation --> SimMatrix[Build N×N<br/>Similarity Matrix]

    subgraph Similarity["<b>Beat-Aligned Circular XCorr</b>"]
        S1[For each pair i,j:<br/>vectors vi, vj<br/>each 16 elements] --> S2[Compute beat-aligned shifts<br/>step = 16 / time_sig<br/>shifts = 0, step, 2*step, 3*step<br/>e.g., 4/4: 0,4,8,12]
        S2 --> S3[For each shift s]
        S3 --> S4[Rotate vector<br/>vj_rot = roll vj, s]
        S4 --> S5[Compute dot product<br/>dot = vi · vj_rot]
        S5 --> S6[Normalize<br/>sim = dot / norm vi * norm vj]
        S6 --> S7{More shifts?}
        S7 -->|Yes| S3
        S7 -->|No| S8[Take maximum<br/>sim_i,j = max all shifts sim]
    end

    SimMatrix --> Similarity
    Similarity --> LagProfile[Extract Bar-Lag Profile]

    subgraph LagExtraction["<b>Lag Profile Extraction</b>"]
        L1[For lag = 1 to N-1] --> L2[Get diagonal L<br/>sim_0,L, sim_1,L+1, ..., sim_N-L,N]
        L2 --> L3[Compute median<br/>profile_L = median diagonal_L]
        L3 --> L4{More lags?}
        L4 -->|Yes| L1
        L4 -->|No| L5[Profile:<br/>lag → similarity]
    end

    LagProfile --> LagExtraction
    LagExtraction --> Choose[Choose Best Power-of-2]

    subgraph PowerOf2["<b>Power-of-2 Selection</b>"]
        P1[Filter to L in 1,2,4,8] --> P2[Find local maxima<br/>profile_L-1 < profile_L > profile_L+1]
        P2 --> P3{Local maxima<br/>exist exclude L=1?}
        P3 -->|Yes| P4[Return smallest L<br/>with best value]
        P3 -->|No| P5[Return smallest L<br/>with globally best value]
        P4 --> P6[Best L<br/>e.g., L = 4 bars]
        P5 --> P6
    end

    Choose --> PowerOf2
    PowerOf2 --> Output[pitch: L]

    style F0Extraction fill:#e1f5ff,stroke:#000,color:#000
    style VectorCreation fill:#ffe1f5,stroke:#000,color:#000
    style Similarity fill:#e1ffe1,stroke:#000,color:#000
    style LagExtraction fill:#fff4e1,stroke:#000,color:#000
    style PowerOf2 fill:#f5e1ff,stroke:#000,color:#000
    style Output fill:#ffcccc,stroke:#000,color:#000
```

---

## Pattern Detection: Similarity Matrix Visualization

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
flowchart LR
    subgraph Step1["<b>1. Build Similarity Matrix</b>"]
        S1["Compare every bar to every other bar<br/>using circular cross-correlation<br/><br/>Result: N×N matrix<br/>where matrix_i,j = similarity bar_i, bar_j<br/><br/>Diagonal = 1.0 perfect self-similarity<br/>High values = similar bars<br/>Low values = different bars"]
    end

    subgraph Step2["<b>2. Extract Diagonals</b>"]
        S2["Diagonal at lag L contains<br/>bar-to-bar similarities L bars apart<br/><br/>Lag 1: sim_0,1, sim_1,2, sim_2,3, ...<br/>Lag 2: sim_0,2, sim_1,3, sim_2,4, ...<br/>Lag 4: sim_0,4, sim_1,5, sim_2,6, ...<br/><br/>If pattern repeats every L bars,<br/>diagonal L will have HIGH values"]
    end

    subgraph Step3["<b>3. Compute Median</b>"]
        S3["For each lag L:<br/>median_L = median of diagonal_L<br/><br/>This gives the typical similarity<br/>between bars L bars apart<br/><br/>Peak at lag L means<br/>pattern length = L bars"]
    end

    subgraph Step4["<b>4. Power-of-2 Selection</b>"]
        S4["Filter to L in 1, 2, 4, 8<br/><br/>Find local maxima<br/>where median_L > median_L-1<br/>and median_L > median_L+1<br/><br/>Return smallest L with best value<br/><br/>Example: If median_4 = 0.91 is highest<br/>Result: Pattern length L = 4 bars"]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4

    style Step1 fill:#e1f5ff,stroke:#000,color:#000
    style Step2 fill:#ffe1f5,stroke:#000,color:#000
    style Step3 fill:#e1ffe1,stroke:#000,color:#000
    style Step4 fill:#fff4e1,stroke:#000,color:#000
```

### Example: 8-Bar Pattern with L=4 Repetition

**Similarity Matrix (8×8):**
```
       Bar0  Bar1  Bar2  Bar3  Bar4  Bar5  Bar6  Bar7
Bar0 │ 1.00  0.45  0.32  0.28  0.92  0.41  0.29  0.25 │
Bar1 │ 0.45  1.00  0.43  0.31  0.44  0.89  0.40  0.28 │
Bar2 │ 0.32  0.43  1.00  0.42  0.31  0.43  0.87  0.39 │
Bar3 │ 0.28  0.31  0.42  1.00  0.27  0.30  0.41  0.85 │
Bar4 │ 0.92  0.44  0.31  0.27  1.00  0.43  0.30  0.26 │
Bar5 │ 0.41  0.89  0.43  0.30  0.43  1.00  0.42  0.29 │
Bar6 │ 0.29  0.40  0.87  0.41  0.30  0.42  1.00  0.40 │
Bar7 │ 0.25  0.28  0.39  0.85  0.26  0.29  0.40  1.00 │
```

**Diagonal Extraction:**
- **Lag 1:** [0.45, 0.43, 0.42, 0.27, 0.43, 0.42, 0.40] → median = **0.42**
- **Lag 2:** [0.32, 0.31, 0.42, 0.31, 0.43, 0.87] → median = **0.37**
- **Lag 3:** [0.28, 0.31, 0.42, 0.27, 0.30, 0.41] → median = **0.31**
- **Lag 4:** [**0.92**, 0.89, **0.87**, **0.85**] → median = **0.88** ⭐ **PEAK!**
- **Lag 5:** [0.41, 0.43, 0.41] → median = **0.41**

**Result:** L = 4 bars (highest median at lag 4)

**Interpretation:**
- Bar 0 and Bar 4 are very similar (0.92)
- Bar 1 and Bar 5 are very similar (0.89)
- Bar 2 and Bar 6 are very similar (0.87)
- Bar 3 and Bar 7 are very similar (0.85)
- This indicates a **4-bar repeating pattern**

---

## Detailed Step 6: Grid Correction Methods

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
flowchart TD
    Input[Corrected Downbeats + Onsets] --> Split{Choose Correction Method}

    Split --> Uncorrected
    Split --> PerSnippet
    Split --> LoopBased

    subgraph Uncorrected["<b>A: Uncorrected Grid</b>"]
        U1[Use Raw Downbeats] --> U2[Create 16 Ticks Per Bar<br/>Based on bar duration]
        U2 --> U3[Match Onsets to<br/>Nearest Tick]
        U3 --> U4[Calculate Phase<br/>onset - grid / step]
    end

    subgraph PerSnippet["<b>B: Per-Snippet Correction</b>"]
        PS1[Find Reference Offset<br/>Priority ticks: 0,8,4,12] --> PS2{Found within<br/>50% tolerance?}
        PS2 -->|Yes| PS3[ref_offset = onset - grid]
        PS2 -->|No| PS4[ref_offset = 0]
        PS3 --> PS5[Shift ALL Grid Times<br/>grid + ref_offset]
        PS4 --> PS5
        PS5 --> PS6[Calculate Phases<br/>with corrected grid]
    end

    subgraph LoopBased["<b>C: Loop-Based Correction</b>"]
        LB1[Divide into Loops<br/>Pattern: L=1,2,4,8 bars] --> LB2[For Each Loop:<br/>Create Equidistant Grid]
        LB2 --> LB3[Duration / pattern × 16]
        LB3 --> LB4[Find Loop Ref Offset<br/>Search all bars]
        LB4 --> LB5{Found?}
        LB5 -->|Yes| LB6[Shift Loop Grid]
        LB5 -->|No| LB7[Use offset = 0]
        LB6 --> LB8[Calculate Phases]
        LB7 --> LB8
        LB8 --> LB9{More loops?}
        LB9 -->|Yes| LB2
        LB9 -->|No| LB10[Combine All Loops]
    end

    U4 --> Output
    PS6 --> Output
    LB10 --> Output[comprehensive_phases.csv<br/>All methods combined]

    style Uncorrected fill:#e1f5ff,stroke:#000,color:#000
    style PerSnippet fill:#ffe1f5,stroke:#000,color:#000
    style LoopBased fill:#e1ffe1,stroke:#000,color:#000
```

---

## Output Generation Flow

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
flowchart LR
    Input[comprehensive_phases.csv] --> Split{Generate Outputs}

    Split --> RMS
    Split --> Audio
    Split --> Plots
    Split --> Tempo
    Split --> MIDI
    Split --> Loops

    subgraph RMS["<b>RMS Calculation</b>"]
        R1[For Each Method:<br/>Calculate RMS phase] --> R2[RMS in phase units]
        R1 --> R3[RMS in milliseconds]
        R2 --> R4[rms_summary.json]
        R3 --> R4
    end

    subgraph Audio["<b>Audio Examples</b>"]
        A1[For Each Method:<br/>Create Click Track] --> A2[3kHz, 50ms clicks<br/>on downbeats]
        A2 --> A3[Mix with Original]
        A3 --> A4[8 MP3 files:<br/>uncorrected, per_snippet,<br/>drum, mel, pitch,<br/>L1, L2, L4]
        A4 --> A5[+ original.mp3]
    end

    subgraph Plots["<b>Raster & Microtiming Plots</b>"]
        P1[Plot 1: Raster Comparison<br/>5 panels] --> P4[raster_comparison.png]
        P2[Plot 2: Raster Standard<br/>5 panels] --> P5[raster_standard.png]
        P3[Plot 3: Microtiming<br/>5 pattern-folded plots] --> P6[microtiming_plots.pdf]
    end

    subgraph Tempo["<b>Tempo Analysis</b>"]
        T1[8-panel Plot<br/>Uncorrected vs Corrected] --> T3[tempo_plots.pdf]
        T2[Bar-by-bar Tempo CSV] --> T4[bar_tempos.csv]
    end

    subgraph MIDI["<b>MIDI Export</b>"]
        M1[Onset MIDI:<br/>7 files per method] --> M3[8_midi/onset/]
        M2[Bass Pitch MIDI:<br/>7 files per method] --> M4[8_midi/bass_pitch/]
    end

    subgraph Loops["<b>Stem Loops</b>"]
        L1[For Each Method:<br/>Extract Loop Range] --> L2[Apply 5ms Crossfade]
        L2 --> L3[Export 5 Stems<br/>vocals, drums, bass,<br/>piano, other]
        L3 --> L4[7 method folders<br/>× 5 stems each]
    end

    R4 --> Final
    A5 --> Final
    P4 --> Final
    P5 --> Final
    P6 --> Final
    T3 --> Final
    T4 --> Final
    M3 --> Final
    M4 --> Final
    L4 --> Final[pipeline_results.json<br/>Complete Summary]

    style RMS fill:#e1f5ff,stroke:#000,color:#000
    style Audio fill:#fff4e1,stroke:#000,color:#000
    style Plots fill:#ffe1f5,stroke:#000,color:#000
    style Tempo fill:#e1ffe1,stroke:#000,color:#000
    style MIDI fill:#f5e1ff,stroke:#000,color:#000
    style Loops fill:#ffffcc,stroke:#000,color:#000
    style Final fill:#ffcccc,stroke:#000,color:#000
```

---

## Complete Pipeline Architecture

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
graph TB
    subgraph Input["<b>Input</b>"]
        I1[Mixed Audio<br/>WAV/MP3]
    end

    subgraph Processing["<b>Core Processing Steps</b>"]
        S1[Step 1: Stem Separation<br/>Spleeter U-Net]
        S2[Step 2: Beat Detection<br/>Beat-Transformer + DBN]
        S3[Step 3: Downbeat Correction<br/>Factor-of-2 Fix]
        S4[Step 4: Onset Detection<br/>Librosa HFC]
        S5[Step 5: Pattern Detection<br/>3 Methods]
        S6[Step 6: Grid Correction<br/>Multiple Methods]
    end

    subgraph Outputs["<b>Analysis Outputs</b>"]
        O1[Raster Plots]
        O2[Microtiming Plots]
        O3[RMS Metrics]
        O4[Audio Examples]
        O5[Tempo Plots]
        O6[MIDI Files]
        O7[Stem Loops]
        O8[Results JSON]
    end

    I1 --> S1
    S1 --> S2
    S1 --> S4
    S2 --> S3
    S3 --> S5
    S3 --> S6
    S4 --> S5
    S4 --> S6
    S5 --> S6

    S6 --> O1
    S6 --> O2
    S6 --> O3
    S6 --> O4
    S6 --> O5
    S6 --> O6
    S6 --> O7

    O1 --> O8
    O2 --> O8
    O3 --> O8
    O4 --> O8
    O5 --> O8
    O6 --> O8
    O7 --> O8

    style Input fill:#e1f5ff,stroke:#000,color:#000
    style Processing fill:#ffe1f5,stroke:#000,color:#000
    style Outputs fill:#e1ffe1,stroke:#000,color:#000
```

---

## Data Dependencies

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','clusterBorder':'#000','edgeLabelBackground':'#fff'}}}%%
graph TD
    Audio[Audio File] --> Stems[5 Stems]
    Stems --> NPZ[Mel-Spec NPZ]
    Stems --> Drums[drums.wav]
    Stems --> Bass[bass.wav]

    NPZ --> Beats[Raw Beats]
    Beats --> CorrectedBeats[Corrected Downbeats]

    Drums --> Onsets[Onset Times]

    CorrectedBeats --> Patterns[Pattern Lengths]
    Onsets --> Patterns
    Drums --> Patterns
    Bass --> Patterns

    CorrectedBeats --> Phases[Phase Calculations]
    Onsets --> Phases
    Patterns --> Phases

    Phases --> Outputs[All Analysis Outputs]

    style Audio fill:#e1f5ff
    style CorrectedBeats fill:#ffe1f5
    style Phases fill:#e1ffe1
    style Outputs fill:#ffffcc
```

---

## Legend

- **Blue boxes**: Input/intermediate audio data
- **Pink boxes**: Beat/rhythm detection
- **Green boxes**: Pattern/timing analysis
- **Purple boxes**: Grid correction
- **Yellow boxes**: Final outputs
- **Red boxes**: Summary/results files

---

## Notes

- Steps 2 and 4 can run in parallel
- Step 6 produces the most important output: `comprehensive_phases.csv`
- All correction methods are calculated simultaneously and stored in one CSV
- Users can compare methods using RMS metrics or by listening to audio examples
