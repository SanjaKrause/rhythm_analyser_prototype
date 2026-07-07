# Feature Sets Exploration Report

**Date:** March 25, 2026
**Stems Analyzed:** DRUMS, BASS, OTHER, VOCALS
**Pattern Lengths:** L1 (16 positions), L2 (32 positions), L4 (64 positions)
**Ratios Compared:** 30, 40, 50, 70
**Primary Analysis:** L2 ratio50 (n=294 songs)

---

## Dataset Overview

```mermaid
flowchart LR
    subgraph Dataset Sizes
        R30[Ratio 30<br/>n=420]
        R40[Ratio 40<br/>n=415]
        R50[Ratio 50<br/>n=294]
        R70[Ratio 70<br/>n=90]
    end

    R30 --> |"Most data"| LARGE[Large Sample]
    R40 --> LARGE
    R50 --> |"Balanced"| MED[Medium Sample]
    R70 --> |"Limited"| SMALL[Small Sample]

    style R30 fill:#90EE90
    style R40 fill:#90EE90
    style R50 fill:#87CEEB
    style R70 fill:#FFB6C1
```

---

## DRIVE Correlations

### Position-Based Strength Features

```mermaid
xychart-beta
    title "DRIVE: Key Position Correlations by Ratio"
    x-axis [GP_str_5, GP_str_8, GP_str_13, GP_str_21, GP_str_29]
    y-axis "Correlation (r)" -0.3 --> 0.3
    bar [0.131, -0.178, 0.221, 0.124, 0.172]
    bar [0.134, -0.169, 0.230, 0.117, 0.177]
    bar [0.191, -0.163, 0.254, 0.193, 0.183]
    bar [0.229, -0.104, 0.208, 0.236, 0.107]
```

### Drive Feature Hierarchy

```mermaid
flowchart TD
    subgraph "DRIVE Predictors"
        direction TB

        subgraph "Syncopation Positions [POSITIVE]"
            P13["Position 13<br/>r=0.22-0.25***<br/>STRONGEST"]
            P5["Position 5<br/>r=0.13-0.23**"]
            P21["Position 21<br/>r=0.12-0.24*"]
            P29["Position 29<br/>r=0.11-0.18**"]
        end

        subgraph "On-Beat Positions [NEGATIVE]"
            P8["Position 8<br/>r=-0.10 to -0.18***"]
            P16["Position 16<br/>r=-0.08 to -0.14*"]
            P24["Position 24<br/>r=-0.13 to -0.19*"]
        end

        subgraph "Stats Features"
            PS["pulse_strength<br/>r=-0.15 to -0.25*"]
        end
    end

    P13 --> |"Best at ratio 50"| BEST50[Ratio 50 Optimal]
    PS --> |"Strongest at ratio 70"| R70[Ratio 70 Effect]

    style P13 fill:#FF6B6B,color:#fff
    style BEST50 fill:#4ECDC4
```

### Drive Summary Table

| Position | Ratio 30 | Ratio 40 | Ratio 50 | Ratio 70 | Interpretation |
|----------|----------|----------|----------|----------|----------------|
| GP_str_13 | +0.221*** | +0.230*** | **+0.254***| +0.208* | Syncopation = Drive |
| RP_str_13 | +0.223*** | +0.230*** | **+0.242***| +0.242* | Consistent across ratios |
| GP_str_5 | +0.131** | +0.134** | **+0.191***| +0.229* | Strengthens at high ratios |
| GP_str_21 | +0.124* | +0.117* | **+0.193***| +0.236* | Strengthens at high ratios |
| GP_str_8 | -0.178*** | -0.169*** | -0.163** | -0.104 | On-beat = less Drive |
| pulse_strength | -0.157** | -0.153** | -0.175** | -0.252* | Stronger pulse = less Drive |

---

## ROLL Correlations

### Roll Feature Hierarchy

```mermaid
flowchart TD
    subgraph "ROLL Predictors"
        direction TB

        subgraph "Timing Variability [STRONGEST]"
            IQR["GPB_iqr_1/8<br/>r=0.20-0.30***<br/>BEST PREDICTOR"]
        end

        subgraph "Tempo & Microtiming"
            TEMPO["mean_section_tempo<br/>r=0.18-0.29***"]
            MICRO["microtiming_degree<br/>r=0.11-0.19***"]
        end

        subgraph "Position Features"
            GP8["GP_str_8<br/>r=0.09-0.18**"]
            GP10["GP_str_10<br/>r=0.10-0.17**"]
            GP18["GP_str_18<br/>r=0.10-0.18**"]
        end

        subgraph "Negative [Less Roll]"
            GP1N["GP_str_1<br/>r=-0.15 to -0.18***"]
            GP29N["GP_str_29<br/>r=-0.11 to -0.18***"]
        end
    end

    IQR --> |"Best at ratio 30"| BEST30[Ratio 30/40 Optimal]
    TEMPO --> BEST30

    style IQR fill:#9B59B6,color:#fff
    style BEST30 fill:#3498DB,color:#fff
```

### Roll Summary Table

| Feature | Ratio 30 | Ratio 40 | Ratio 50 | Ratio 70 | Interpretation |
|---------|----------|----------|----------|----------|----------------|
| **GPB_iqr_1/8** | **+0.238***| +0.232*** | +0.203*** | +0.295** | Timing variability = Roll |
| mean_section_tempo | **+0.259***| +0.254*** | +0.176** | +0.286** | Faster = more Roll |
| microtiming_degree | +0.188*** | +0.185*** | +0.167** | +0.112 | Weakens at high ratio |
| GP_iqr_10 | +0.188*** | +0.184*** | +0.171** | +0.120 | Position 10 variability |
| GP_str_8 | +0.161*** | +0.159** | +0.176** | +0.092 | On-beat strength |
| GP_str_1 | -0.180*** | -0.173*** | -0.149* | -0.201 | Position 1 = less Roll |

---

## PULSE Correlations

### Pulse Feature Hierarchy

```mermaid
flowchart TD
    subgraph "PULSE Predictors"
        direction TB

        subgraph "Pulse Strength Measures [STRONGEST]"
            GPS["groove_pulse_strength<br/>r=0.15-0.36***"]
            PS["pulse_strength<br/>r=0.18-0.37***"]
        end

        subgraph "On-Beat Positions [Ratio 70 EFFECT]"
            GP8P["GP_str_8<br/>r=0.14-0.22***"]
            GP10P["GP_str_10<br/>r=0.12-0.37***"]
            GP28P["GP_str_28<br/>r=0.10-0.41***"]
        end

        subgraph "Negative [Less Pulse]"
            IOI["ioi_microtiming_degree<br/>r=-0.14 to -0.18**"]
            GPBIQR["GPB_iqr_1/4<br/>r=-0.14 to -0.20*"]
        end
    end

    GPS --> |"MUCH stronger at ratio 70"| R70EFF[Ratio 70 Effect]
    PS --> R70EFF
    GP28P --> |"r=0.41*** at ratio 70!"| R70EFF

    style GPS fill:#E74C3C,color:#fff
    style PS fill:#E74C3C,color:#fff
    style R70EFF fill:#F39C12,color:#fff
```

### Pulse Summary Table

| Feature | Ratio 30 | Ratio 40 | Ratio 50 | Ratio 70 | Interpretation |
|---------|----------|----------|----------|----------|----------------|
| **groove_pulse_strength** | +0.153** | +0.163*** | +0.225*** | **+0.361***| Dramatic increase at r70 |
| **pulse_strength** | +0.182*** | +0.183*** | +0.218*** | **+0.365***| Dramatic increase at r70 |
| GP_str_8 | +0.154** | +0.149** | **+0.221***| +0.136 | Best at ratio 50 |
| GP_str_10 | +0.115* | +0.120* | +0.084 | **+0.368***| Explodes at ratio 70 |
| GP_str_28 | +0.101* | +0.104* | +0.114 | **+0.406***| Strongest at ratio 70! |
| ioi_microtiming_degree | -0.143** | -0.144** | -0.174** | -0.183 | IOI variability = less Pulse |

---

## Ratio Comparison Overview

```mermaid
flowchart TB
    subgraph "Ratio Selection Guide"
        direction LR

        subgraph R30["Ratio 30 (n=420)"]
            R30A["+ Most data"]
            R30B["+ Best Roll correlations"]
            R30C["- May include loose patterns"]
        end

        subgraph R40["Ratio 40 (n=415)"]
            R40A["+ Large sample"]
            R40B["+ Similar to ratio 30"]
            R40C["~ Middle ground"]
        end

        subgraph R50["Ratio 50 (n=294)"]
            R50A["+ Best Drive correlations"]
            R50B["+ Good balance overall"]
            R50C["+ Recommended default"]
        end

        subgraph R70["Ratio 70 (n=90)"]
            R70A["+ Strongest Pulse effects"]
            R70B["+ Clear on-beat patterns"]
            R70C["- Too few songs (n=90)"]
        end
    end

    R50 --> RECOMMEND["RECOMMENDED"]

    style R50 fill:#27AE60,color:#fff
    style RECOMMEND fill:#2ECC71,color:#fff
    style R70 fill:#E74C3C,color:#fff
```

---

## Best Features Per Target

```mermaid
mindmap
  root((DRUMS L2<br/>Features))
    DRIVE
      Position 13 Syncopation
        GP_str_13 r=0.254
        RP_str_13 r=0.242
      Other Syncopations
        Position 5 r=0.191
        Position 21 r=0.193
        Position 29 r=0.183
      Stats
        pulse_strength r=-0.175
    ROLL
      Timing Variability
        GPB_iqr_1/8 r=0.238
      Tempo
        mean_section_tempo r=0.259
      Microtiming
        microtiming_degree r=0.188
    PULSE
      Pulse Measures
        groove_pulse_strength r=0.361
        pulse_strength r=0.365
      On-Beat Positions
        GP_str_8 r=0.221
        GP_str_28 r=0.406
```

---

## Conclusions

### 1. Drive is Syncopation-Driven
- **Position 13** is the universal Drive predictor across all ratios
- Syncopated positions (5, 13, 21, 29) consistently positive
- On-beat positions (8, 16, 24) consistently negative
- **Best ratio: 50** (strongest syncopation effects)

### 2. Roll is Variability-Driven
- **GPB_iqr_1/8** (eighth-note timing variability) dominates
- Faster tempo = more Roll
- More microtiming = more Roll
- **Best ratio: 30/40** (strongest correlations, most data)

### 3. Pulse is Regularity-Driven
- **pulse_strength** and **groove_pulse_strength** are key
- Effect DRAMATICALLY increases at ratio 70
- High-ratio songs have clearer pulse patterns
- **Best ratio: 50** (balance) or **70** (if sample size permits)

### 4. Ratio Recommendations

| Use Case | Recommended Ratio |
|----------|-------------------|
| General analysis | **50** |
| Drive research | **50** |
| Roll research | **30 or 40** |
| Pulse research | **50** (or 70 with caution) |
| Maximum sample size | **30** |

---

## Appendix: Significant Features Count

| Ratio | Drive (p<0.05) | Roll (p<0.05) | Pulse (p<0.05) |
|-------|----------------|---------------|----------------|
| 30 | ~35 | ~40 | ~25 |
| 40 | ~35 | ~38 | ~25 |
| 50 | 35 | 37 | 18 |
| 70 | ~20 | ~15 | ~25 |

*Note: Ratio 70 has fewer significant features due to smaller sample size (n=90)*

---

# DRUMS L1 vs L2 Comparison

## L1 Overview (16 positions = 1 bar)

### Dataset Sizes (Same as L2)

| Ratio | After DGA merge |
|-------|-----------------|
| 30    | **420** songs   |
| 40    | **415** songs   |
| 50    | **294** songs   |
| 70    | **90** songs    |

---

## L1 DRIVE Correlations

### L1 Drive Summary Table

| Feature | L1 r30 | L1 r40 | L1 r50 | L1 r70 |
|---------|--------|--------|--------|--------|
| **GP_str_13** | +0.226*** | +0.235*** | **+0.263***| +0.193 |
| **RP_str_13** | +0.235*** | +0.238*** | **+0.245***| +0.207 |
| GP_str_5 | +0.139** | +0.141** | **+0.203***| +0.177 |
| GP_str_1 | +0.153** | +0.147** | +0.175** | +0.150 |
| GP_str_8 (neg) | -0.168*** | -0.162*** | -0.135* | -0.127 |
| GP_iqr_13 | +0.186*** | +0.190*** | **+0.201***| +0.162 |

**Key Finding:** Position 13 is STRONGER in L1 (r=0.263) than L2 (r=0.254)!

---

## L1 ROLL Correlations

### L1 Roll Summary Table

| Feature | L1 r30 | L1 r40 | L1 r50 | L1 r70 |
|---------|--------|--------|--------|--------|
| **GPB_iqr_1/8** | **+0.247***| +0.244*** | +0.225*** | +0.295** |
| mean_section_tempo | **+0.258***| +0.254*** | +0.178** | +0.285** |
| microtiming_degree | +0.227*** | +0.224*** | +0.210*** | +0.162 |
| GP_str_1 (neg) | -0.171*** | -0.165*** | -0.159** | -0.130 |
| RP_str_1 (neg) | -0.186*** | -0.182*** | -0.182** | -0.144 |

**Key Finding:** GPB_iqr_1/8 remains #1 predictor, slightly stronger in L1!

---

## L1 PULSE Correlations

### L1 Pulse Summary Table

| Feature | L1 r30 | L1 r40 | L1 r50 | L1 r70 |
|---------|--------|--------|--------|--------|
| **GP_str_8** | +0.155** | +0.153** | **+0.194***| +0.138 |
| **GP_str_10** | +0.112* | +0.115* | +0.093 | **+0.359***|
| groove_pulse_strength | +0.137** | +0.137** | +0.135* | +0.245* |
| GPB_iqr_1/8 | +0.128** | +0.129** | +0.129* | +0.116 |
| RP_str_13 (neg) | -0.137** | -0.131** | -0.106 | -0.021 |
| **pulse_strength** | -- | -- | -- | -- |

---

## L1 Bug: Missing pulse_strength

```mermaid
flowchart TD
    subgraph "pulse_strength Calculation Bug"
        direction TB

        CODE["anchored_rhythm_statistics.py<br/>Line 100-107"]

        subgraph "Current Code"
            L2["if pattern_length == 2:<br/>beat_positions = [1,5,9,13,17,21,25,29]"]
            L4["elif pattern_length == 4:<br/>beat_positions = [1,5,9,13,...,57,61]"]
            L1BUG["else:<br/>beat_positions = [] ← BUG!"]
        end

        subgraph "Result"
            MISSING["L1: pulse_strength = None<br/>Shows as -- in output"]
        end

        subgraph "Fix Needed"
            FIX["Add: elif pattern_length == 1:<br/>beat_positions = [1, 5, 9, 13]"]
        end
    end

    CODE --> L2
    L2 --> L4
    L4 --> L1BUG
    L1BUG --> MISSING
    FIX --> |"Should add"| L1BUG

    style L1BUG fill:#E74C3C,color:#fff
    style MISSING fill:#E74C3C,color:#fff
    style FIX fill:#27AE60,color:#fff
```

**Location:** `loop_extractor/batch_analysis/anchored_rhythm_statistics.py` lines 100-107

**Problem:** Code only handles `pattern_length == 2` (L2) and `pattern_length == 4` (L4), but not `pattern_length == 1` (L1).

**Fix:** Add `elif pattern_length == 1: beat_positions = [1, 5, 9, 13]`

---

## L1 vs L2 Comparison

```mermaid
flowchart LR
    subgraph "Pattern Length Comparison"
        direction TB

        subgraph L1["L1 (16 positions)"]
            L1A["+ Stronger Drive correlations"]
            L1B["+ Stronger Roll correlations"]
            L1C["+ Simpler interpretation"]
            L1D["- Missing pulse_strength BUG"]
            L1E["- Less position detail"]
        end

        subgraph L2["L2 (32 positions)"]
            L2A["+ Has pulse_strength"]
            L2B["+ More position detail"]
            L2C["+ Better for Pulse research"]
            L2D["~ Slightly weaker correlations"]
        end
    end

    L1 --> |"Better for"| DRIVE["Drive & Roll"]
    L2 --> |"Better for"| PULSE["Pulse"]

    style L1D fill:#E74C3C,color:#fff
    style L2A fill:#27AE60,color:#fff
```

### Summary Comparison Table

| Aspect | L1 (16 pos) | L2 (32 pos) | Winner |
|--------|-------------|-------------|--------|
| **Drive: Position 13** | r=0.263*** | r=0.254*** | **L1** |
| **Roll: GPB_iqr_1/8** | r=0.247*** | r=0.238*** | **L1** |
| **Pulse: GP_str_8** | r=0.194*** | r=0.221*** | **L2** |
| **Pulse: pulse_strength** | MISSING (BUG) | r=0.218*** | **L2** |
| Feature count | 81 | 165 | L2 |
| Interpretability | Simpler | More detail | Depends |

### Recommendations by Pattern Length

| Use Case | Recommended |
|----------|-------------|
| Drive research | **L1** (stronger correlations) |
| Roll research | **L1** (stronger correlations) |
| Pulse research | **L2** (has pulse_strength) |
| Detailed position analysis | **L2** (32 positions) |
| Quick analysis | **L1** (simpler) |

---

## Overall Conclusions

### 1. Pattern Length Effects
- **L1 shows STRONGER correlations** for Drive and Roll than L2
- This is likely due to averaging effects (16 vs 32 positions)
- **L2 is required for Pulse research** due to pulse_strength availability

### 2. Consistent Patterns Across L1 and L2
- **Position 13 = Drive** (syncopation)
- **GPB_iqr_1/8 = Roll** (timing variability)
- **On-beat positions = Pulse**

### 3. Ratio Effects (Same for L1 and L2)
- **Ratio 50** best for Drive
- **Ratio 30/40** best for Roll
- **Ratio 70** shows strong Pulse effects but limited sample

### 4. Bug Fix Required
- `pulse_strength` not computed for L1 - needs fix in `anchored_rhythm_statistics.py`

---

# DRUMS L4 Analysis (64 positions = 4 bars)

## L4 Overview

### Dataset Sizes

| Ratio | After DGA merge |
|-------|-----------------|
| 30    | **389** songs   |
| 40    | **379** songs   |
| 50    | **292** songs   |
| 70    | **90** songs    |

*Note: L4 has slightly fewer songs than L1/L2 because 4-bar repeating patterns are less common.*

---

## L4 DRIVE Correlations

### L4 Drive Summary Table

| Feature | L4 r30 | L4 r40 | L4 r50 | L4 r70 |
|---------|--------|--------|--------|--------|
| **GP_str_13** | +0.217*** | +0.215*** | **+0.238***| +0.179 |
| **RP_str_13** | +0.219*** | +0.217*** | **+0.243***| +0.230* |
| GP_str_21 | +0.190*** | +0.176*** | **+0.227***| +0.290** |
| GP_str_33 | +0.182*** | +0.192*** | **+0.217***| +0.201 |
| GP_str_45 | +0.205*** | +0.203*** | +0.192*** | +0.171 |
| GP_str_37 | +0.153** | +0.167** | +0.158** | +0.266* |
| GP_str_29 | +0.161** | +0.157** | +0.199*** | +0.178 |
| GP_med_21 | +0.101* | +0.087 | +0.108 | **+0.344***|
| pulse_strength | -0.135** | -0.146** | -0.154** | -0.209* |

**Key Finding:** Syncopation "echoes" at positions 13, 21, 29, 33, 37, 45 (every 16 positions + offsets)

---

## L4 ROLL Correlations

### L4 Roll Summary Table

| Feature | L4 r30 | L4 r40 | L4 r50 | L4 r70 |
|---------|--------|--------|--------|--------|
| **GPB_iqr_1/8** | **+0.257***| +0.250*** | +0.222*** | +0.284** |
| mean_section_tempo | **+0.244***| +0.239*** | +0.174** | +0.282** |
| GP_str_40 | +0.189*** | +0.185*** | +0.218*** | +0.096 |
| GP_str_45 (neg) | -0.205*** | -0.194*** | -0.167** | -0.135 |
| RP_str_1 (neg) | -0.187*** | -0.191*** | -0.195*** | -0.268* |
| microtiming_degree | +0.161** | +0.167** | +0.134* | +0.124 |
| GP_med_42 | +0.193*** | +0.193*** | +0.208*** | +0.218* |

**Key Finding:** GPB_iqr_1/8 remains #1 Roll predictor, consistent across all pattern lengths!

---

## L4 PULSE Correlations

### L4 Pulse Summary Table

| Feature | L4 r30 | L4 r40 | L4 r50 | L4 r70 |
|---------|--------|--------|--------|--------|
| **pulse_strength** | +0.183*** | +0.197*** | +0.194*** | +0.294** |
| **GP_str_60** | +0.130* | +0.132** | +0.131* | **+0.387***|
| **GP_str_42** | +0.088 | +0.085 | +0.074 | **+0.352***|
| GP_str_40 | +0.143** | +0.146** | +0.160** | +0.002 |
| GP_str_10 | +0.099 | +0.096 | +0.046 | **+0.302**|
| groove_pulse_strength | +0.144** | +0.150** | +0.183** | +0.248* |
| GP_str_58 | +0.099 | +0.092 | +0.090 | **+0.291**|

**Key Finding:** At ratio 70, positions 60 (+0.387***) and 42 (+0.352***) emerge as strong Pulse predictors!

---

## L4 Syncopation Echo Pattern

```mermaid
flowchart LR
    subgraph "Syncopation Echoes (Position 13 family)"
        P13["Pos 13<br/>r=0.238***"]
        P29["Pos 29<br/>r=0.199***"]
        P45["Pos 45<br/>r=0.192***"]
        P61["Pos 61<br/>r=0.134*"]
    end

    subgraph "Syncopation Echoes (Position 21 family)"
        P21["Pos 21<br/>r=0.227***"]
        P37["Pos 37<br/>r=0.158**"]
        P53["Pos 53<br/>r=0.120*"]
    end

    subgraph "Syncopation Echoes (Position 33 family)"
        P33["Pos 33<br/>r=0.217***"]
        P49["Pos 49<br/>r=0.132*"]
    end

    P13 --> |"+16"| P29
    P29 --> |"+16"| P45
    P45 --> |"+16"| P61

    P21 --> |"+16"| P37
    P37 --> |"+16"| P53

    P33 --> |"+16"| P49

    style P13 fill:#FF6B6B,color:#fff
    style P21 fill:#FF6B6B,color:#fff
    style P33 fill:#FF6B6B,color:#fff
```

---

## L1 vs L2 vs L4 Comparison

### Summary Comparison Table

| Aspect | L1 (16 pos) | L2 (32 pos) | L4 (64 pos) | Winner |
|--------|-------------|-------------|-------------|--------|
| **Drive: Position 13** | r=0.263*** | r=0.254*** | r=0.238*** | **L1** |
| **Roll: GPB_iqr_1/8** | r=0.247*** | r=0.238*** | r=0.257*** | **L4** |
| **Pulse: pulse_strength** | MISSING | r=0.218*** | r=0.194*** | **L2** |
| **Pulse: GP_str_60** | N/A | N/A | r=0.387*** | **L4** |
| Feature count | 81 | 165 | 329 | L4 |
| Sample size (r50) | 294 | 294 | 292 | Similar |
| Syncopation detail | 1 bar | 2 bars | **4 bars** | L4 |

### Pattern Length Effects

```mermaid
flowchart TD
    subgraph "Correlation Strength by Pattern Length"
        direction TB

        subgraph DRIVE["DRIVE (Position 13)"]
            L1D["L1: r=0.263***<br/>STRONGEST"]
            L2D["L2: r=0.254***"]
            L4D["L4: r=0.238***"]
        end

        subgraph ROLL["ROLL (GPB_iqr_1/8)"]
            L1R["L1: r=0.247***"]
            L2R["L2: r=0.238***"]
            L4R["L4: r=0.257***<br/>STRONGEST"]
        end

        subgraph PULSE["PULSE (pulse_strength)"]
            L1P["L1: MISSING (BUG)"]
            L2P["L2: r=0.218***<br/>STRONGEST"]
            L4P["L4: r=0.194***"]
        end
    end

    L1D --> |"Averaging helps"| AVGHELP["Shorter patterns<br/>= more averaging<br/>= stronger signal"]
    L4R --> |"More positions"| MOREPOS["More timing data<br/>= better variability estimate"]

    style L1D fill:#27AE60,color:#fff
    style L4R fill:#27AE60,color:#fff
    style L2P fill:#27AE60,color:#fff
    style L1P fill:#E74C3C,color:#fff
```

---

## Updated Recommendations

### Pattern Length Selection Guide

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Drive research | **L1** | Strongest Position 13 correlation |
| Roll research | **L4** | Best GPB_iqr_1/8 correlation |
| Pulse research | **L2** | Has pulse_strength, best correlation |
| Syncopation detail | **L4** | Shows echo patterns across bars |
| Quick analysis | **L1** | Simplest, fewest features |
| Full position analysis | **L4** | 64 positions, most detail |

### Ratio Selection (Same for all Ls)

| Use Case | Recommended Ratio |
|----------|-------------------|
| General analysis | **50** |
| Drive research | **50** |
| Roll research | **30 or 40** |
| Pulse research | **50** (or 70 with caution) |
| Maximum sample size | **30** |

---

## Key Discoveries from L4

### 1. Syncopation Echoes
- Position 13 effect repeats every 16 positions (13, 29, 45, 61)
- Position 21 family: 21, 37, 53
- Position 33 family: 33, 49
- This confirms syncopation patterns repeat across bars

### 2. Late-Bar Pulse Effects (Ratio 70)
- **Position 60** emerges as strongest Pulse predictor (r=+0.387***)
- **Position 42** also very strong (r=+0.352***)
- These are "on-beat" positions in bars 3-4
- Suggests Pulse perception strengthened by consistent patterns across multiple bars

### 3. GPB_iqr_1/8 Universality
- Best Roll predictor in ALL pattern lengths
- L4 shows slightly stronger effect (r=0.257***) than L1/L2
- Timing variability at eighth-note level is fundamental to Roll perception

### 4. L4-Specific Findings
- More positions to show syncopation "echo" patterns
- Late-bar positions (40-63) show unique effects
- GP_med_21 becomes very strong at ratio 70 (r=+0.344***)

---

# Multi-Stem Comparison (L2 ratio50)

## Overview

Analysis of **four stems**: DRUMS, BASS, OTHER, VOCALS at L2 ratio50 (n=294 songs each).

```mermaid
flowchart TD
    subgraph "Stem Predictive Power Ranking"
        direction LR
        V["VOCALS<br/>r_max=0.406<br/>~80+ sig features"]
        D["DRUMS<br/>r_max=0.254<br/>~35 sig features"]
        B["BASS<br/>r_max=0.177<br/>~15 sig features"]
        O["OTHER<br/>r_max=0.176<br/>~5 sig features"]
    end

    V --> |">>>"| D
    D --> |">>"| B
    B --> |"≈"| O

    style V fill:#27AE60,color:#fff
    style D fill:#3498DB,color:#fff
    style B fill:#F39C12,color:#fff
    style O fill:#E74C3C,color:#fff
```

**Key Finding:** Despite worse onset detection, VOCALS show **60% stronger correlations** than DRUMS!

---

## DRIVE Correlations by Stem

### Drive Comparison Table

| Feature | DRUMS | BASS | OTHER | VOCALS | Pattern |
|---------|-------|------|-------|--------|---------|
| **pulse_strength** | **-0.175**\*\* | -0.010 | -0.013 | **+0.406**\*\*\* | DRUMS neg, VOCALS pos! |
| **GP_str_13** | **+0.254**\*\*\* | +0.067 | +0.003 | +0.115* | DRUMS-specific |
| **GP_str_20** | -0.028 | +0.074 | -0.072 | **+0.337**\*\*\* | VOCALS-specific |
| **GP_str_8** | **-0.163**\*\* | -0.010 | -0.036 | **+0.249**\*\*\* | Opposite signs! |
| total_ioi_count | +0.096 | +0.065 | -0.010 | **+0.367**\*\*\* | VOCALS only |
| microtiming_complexity | +0.053 | -0.038 | +0.104 | **+0.279**\*\*\* | VOCALS strongest |
| GP_str_21 | **+0.193**\*\*\* | +0.075 | -0.041 | **+0.194**\*\*\* | DRUMS & VOCALS |
| GP_str_11 | +0.092 | **+0.169**\*\* | -0.019 | +0.151** | BASS-specific |
| ioi_microtiming_degree | +0.103 | -0.052 | **+0.168**\*\* | -0.137* | OTHER pos, VOCALS neg |
| BP_str_3/16 | +0.001 | **+0.150**\*\* | -0.049 | -0.103 | BASS-specific |

### Drive Mechanisms by Stem

```mermaid
flowchart TD
    subgraph "DRIVE Mechanisms"
        direction TB

        subgraph DRUMS_D["DRUMS"]
            D1["Syncopation-driven"]
            D2["Position 13 = +0.254***"]
            D3["pulse_strength NEGATIVE"]
            D4["On-beat positions negative"]
        end

        subgraph VOCALS_D["VOCALS"]
            V1["Density-driven"]
            V2["ALL positions positive"]
            V3["pulse_strength POSITIVE +0.406"]
            V4["More activity = more Drive"]
        end

        subgraph BASS_D["BASS"]
            B1["Weak overall"]
            B2["Position 11 = +0.169**"]
            B3["BP_str_3/16 = +0.150**"]
            B4["No clear pattern"]
        end

        subgraph OTHER_D["OTHER"]
            O1["Nearly useless"]
            O2["Only ioi_micro works"]
            O3["~5 significant features"]
        end
    end

    DRUMS_D --> |"Opposite"| VOCALS_D

    style D2 fill:#FF6B6B,color:#fff
    style V3 fill:#27AE60,color:#fff
    style O1 fill:#E74C3C,color:#fff
```

---

## ROLL Correlations by Stem

### Roll Comparison Table

| Feature | DRUMS | BASS | OTHER | VOCALS | Pattern |
|---------|-------|------|-------|--------|---------|
| **mean_section_tempo** | **+0.176**\*\* | **+0.176**\*\* | **+0.176**\*\* | **+0.176**\*\* | **UNIVERSAL!** |
| **GPB_iqr_1/8** | **+0.203**\*\*\* | +0.075 | -0.021 | **-0.162**\*\* | DRUMS pos, VOCALS neg! |
| microtiming_degree | **+0.167**\*\* | +0.119* | +0.031 | +0.134* | DRUMS strongest |
| GP_str_22 | +0.076 | **+0.177**\*\* | +0.129* | -0.093 | BASS-specific |
| GPB_iqr_3/16 | +0.041 | **+0.159**\*\* | +0.099 | -0.095 | BASS-specific |
| GP_str_10 | **+0.165**\*\* | +0.085 | **+0.150**\*\* | **-0.189**\*\* | DRUMS/OTHER pos, VOCALS neg |
| GP_str_18 | **+0.176**\*\* | +0.075 | +0.088 | -0.135* | DRUMS pos, VOCALS neg |
| GP_str_20 | +0.020 | +0.033 | -0.044 | **-0.233**\*\*\* | VOCALS strongly negative |
| GP_iqr_10 | **+0.171**\*\* | +0.088 | +0.046 | **-0.234**\*\*\* | DRUMS pos, VOCALS neg |
| GP_iqr_31 | -0.038 | **-0.154**\*\* | +0.027 | +0.032 | BASS-specific negative |

### Roll Mechanisms by Stem

```mermaid
flowchart TD
    subgraph "ROLL Mechanisms"
        direction TB

        subgraph UNIVERSAL["UNIVERSAL (All Stems)"]
            U1["mean_section_tempo = +0.176**"]
            U2["Faster tempo = more Roll"]
        end

        subgraph DRUMS_R["DRUMS"]
            DR1["GPB_iqr_1/8 DOMINANT +0.203***"]
            DR2["Timing variability at 1/8"]
            DR3["On-beat positions positive"]
        end

        subgraph VOCALS_R["VOCALS"]
            VR1["GPB_iqr_1/8 REVERSED -0.162**"]
            VR2["Most positions NEGATIVE"]
            VR3["Vocal activity hurts Roll"]
        end

        subgraph BASS_R["BASS"]
            BR1["Position 22 specific +0.177**"]
            BR2["GPB_iqr_3/16 +0.159**"]
            BR3["Different mechanism"]
        end

        subgraph OTHER_R["OTHER"]
            OR1["Only tempo works"]
            OR2["Position features useless"]
        end
    end

    UNIVERSAL --> DRUMS_R
    UNIVERSAL --> VOCALS_R
    UNIVERSAL --> BASS_R
    UNIVERSAL --> OTHER_R

    style U1 fill:#9B59B6,color:#fff
    style DR1 fill:#3498DB,color:#fff
    style VR1 fill:#E74C3C,color:#fff
```

**Critical Finding:** `mean_section_tempo` shows IDENTICAL correlation (+0.176\*\*) across ALL FOUR stems!

---

## PULSE Correlations by Stem

### Pulse Comparison Table

| Feature | DRUMS | BASS | OTHER | VOCALS | Pattern |
|---------|-------|------|-------|--------|---------|
| **pulse_strength** | **+0.218**\*\*\* | -0.025 | +0.012 | **-0.283**\*\*\* | DRUMS pos, VOCALS neg! |
| **groove_pulse_strength** | **+0.225**\*\*\* | +0.101 | **+0.163**\*\* | **-0.203**\*\*\* | ALL significant, VOCALS opposite |
| **GP_str_8** | **+0.221**\*\*\* | -0.022 | -0.018 | **-0.243**\*\*\* | Opposite signs! |
| total_ioi_count | +0.013 | -0.021 | +0.056 | **-0.350**\*\*\* | VOCALS strong negative |
| microtiming_complexity | -0.081 | +0.010 | -0.079 | **-0.305**\*\*\* | VOCALS strongest |
| groove_ioi_pulse_strength | +0.010 | **+0.170**\*\* | -0.071 | +0.121* | BASS-specific |
| GP_med_30 | +0.017 | **+0.170**\*\* | +0.096 | -0.077 | BASS-specific |
| GP_med_1 | +0.013 | **+0.154**\*\* | -0.090 | -0.089 | BASS-specific |
| ioi_microtiming_degree | **-0.174**\*\* | +0.055 | **-0.159**\*\* | +0.106 | DRUMS/OTHER neg |
| GP_str_20 | +0.072 | -0.059 | -0.002 | **-0.263**\*\*\* | VOCALS-specific |

### Pulse Mechanisms by Stem

```mermaid
flowchart TD
    subgraph "PULSE Mechanisms"
        direction TB

        subgraph DRUMS_P["DRUMS"]
            DP1["pulse_strength POSITIVE +0.218***"]
            DP2["On-beat positions POSITIVE"]
            DP3["Consistent beats = Pulse"]
        end

        subgraph VOCALS_P["VOCALS"]
            VP1["pulse_strength NEGATIVE -0.283***"]
            VP2["ALL positions NEGATIVE"]
            VP3["Vocal activity MASKS Pulse"]
            VP4["total_ioi_count -0.350***"]
        end

        subgraph BASS_P["BASS"]
            BP1["groove_ioi_pulse_strength +0.170**"]
            BP2["GP_med positions work"]
            BP3["IOI timing mechanism"]
        end

        subgraph OTHER_P["OTHER"]
            OP1["Only groove_pulse_strength +0.163**"]
            OP2["Weak otherwise"]
        end
    end

    DRUMS_P --> |"OPPOSITE"| VOCALS_P

    style DP1 fill:#27AE60,color:#fff
    style VP1 fill:#E74C3C,color:#fff
    style VP4 fill:#E74C3C,color:#fff
```

---

## Sparsity Comparison (Non-Zero Counts)

| Position | DRUMS nz | BASS nz | OTHER nz | VOCALS nz | Notes |
|----------|----------|---------|----------|-----------|-------|
| GP_str_0 | 294 | 198 | 176 | 197 | DRUMS always has onset at 0 |
| GP_str_1 | 50 | 134 | 88 | 133 | DRUMS sparse at syncopation |
| GP_str_4 | 280 | 94 | 78 | 189 | DRUMS dense on quarter notes |
| GP_str_8 | 271 | 104 | 120 | 207 | DRUMS densest on beat 3 |
| GP_str_13 | 60 | 92 | 80 | 161 | VOCALS 3x denser than DRUMS |
| GP_str_16 | 288 | 170 | 150 | 190 | All stems have bar downbeat |

**Key Insight:** DRUMS is very sparse at syncopation positions (13, 21, 29 have ~50-70 non-zeros), while VOCALS is dense everywhere (130-210 non-zeros). This explains why syncopation matters for drums but not vocals.

---

## Summary: Stem Comparison

### Best Features Per Stem

| Stem | Best Drive Feature | Best Roll Feature | Best Pulse Feature |
|------|-------------------|-------------------|-------------------|
| **DRUMS** | GP_str_13 (+0.254) | GPB_iqr_1/8 (+0.203) | groove_pulse_str (+0.225) |
| **BASS** | GP_str_11 (+0.169) | GP_str_22 (+0.177) | groove_ioi_ps (+0.170) |
| **OTHER** | ioi_micro_deg (+0.168) | tempo (+0.176) | groove_pulse_str (+0.163) |
| **VOCALS** | pulse_strength (+0.406) | tempo (+0.176) | total_ioi_count (-0.350) |

### Mechanism Summary

```
                   DRUMS              BASS               OTHER              VOCALS
                   ─────              ────               ─────              ──────
DRIVE
  Best r           +0.254***          +0.169**           +0.168**           +0.406***
  Mechanism        Syncopation        Weak/Position 11   Weak               Density
  pulse_strength   NEGATIVE           None               None               POSITIVE

ROLL
  Best r           +0.203***          +0.177**           +0.176**           +0.176**
  Mechanism        Timing variability Position 22        Tempo only         Tempo only
  GPB_iqr_1/8      POSITIVE           None               None               NEGATIVE

PULSE
  Best r           +0.225***          +0.170**           +0.163**           -0.350***
  Mechanism        On-beat strength   IOI timing         groove_ps          Negative density
  pulse_strength   POSITIVE           None               None               NEGATIVE
```

---

## Key Discoveries from Multi-Stem Analysis

### 1. VOCALS Reverse DRUMS Patterns

```mermaid
flowchart LR
    subgraph "Feature Sign Reversal"
        direction TB

        subgraph DRUMS_SIGN["DRUMS Signs"]
            DS1["pulse_strength → Drive: NEGATIVE"]
            DS2["GPB_iqr_1/8 → Roll: POSITIVE"]
            DS3["On-beat → Pulse: POSITIVE"]
        end

        subgraph VOCALS_SIGN["VOCALS Signs (Opposite!)"]
            VS1["pulse_strength → Drive: POSITIVE"]
            VS2["GPB_iqr_1/8 → Roll: NEGATIVE"]
            VS3["On-beat → Pulse: NEGATIVE"]
        end
    end

    DS1 --> |"REVERSED"| VS1
    DS2 --> |"REVERSED"| VS2
    DS3 --> |"REVERSED"| VS3

    style DS1 fill:#3498DB,color:#fff
    style VS1 fill:#E74C3C,color:#fff
```

### 2. OTHER Stem is Nearly Useless
- Only ~5 significant features total
- Only `mean_section_tempo` and `groove_pulse_strength` work
- No position-based patterns
- Likely too heterogeneous (guitars, synths, effects mixed together)

### 3. mean_section_tempo is Universal
- Exactly **+0.176\*\*** for Roll in ALL FOUR stems
- Faster tempo = more Roll, regardless of instrument
- This is the ONLY feature that works identically everywhere

### 4. VOCALS Act as Perceptual "Noise"
- More vocal activity = **more Drive** (energy/excitement)
- More vocal activity = **less Pulse** (masks the beat)
- More vocal activity = **less Roll** (disrupts flow)
- Despite worse onset detection, vocals have STRONGEST correlations!

### 5. BASS Has Unique Mechanisms
- Position 22 specific for Roll (+0.177\*\*) - not significant in other stems
- GPB_iqr_3/16 works for Roll (+0.159\*\*)
- Pulse works through IOI timing (`groove_ioi_pulse_strength`)
- Different from drums - not syncopation-based

### 6. Stem Ranking by Predictive Power

```
VOCALS >>> DRUMS >> BASS ≈ OTHER
(r=0.41)   (r=0.25) (r=0.18) (r=0.17)
```

---

## Recommendations for Multi-Stem Analysis

### When to Use Each Stem

| Research Goal | Primary Stem | Secondary Stem | Avoid |
|---------------|--------------|----------------|-------|
| Drive research | VOCALS | DRUMS | OTHER |
| Roll research | DRUMS | BASS | OTHER |
| Pulse research | DRUMS | VOCALS (negative) | OTHER |
| Syncopation effects | DRUMS | - | VOCALS, OTHER |
| Density effects | VOCALS | - | DRUMS |
| Timing variability | DRUMS | BASS | OTHER |

### Combined Stem Features

For maximum predictive power, consider combining:
1. **DRUMS Position 13** (syncopation → Drive)
2. **VOCALS pulse_strength** (density → Drive)
3. **DRUMS GPB_iqr_1/8** (timing variability → Roll)
4. **ALL STEMS mean_section_tempo** (universal → Roll)
5. **DRUMS groove_pulse_strength** (regularity → Pulse)
6. **VOCALS total_ioi_count** (inverse density → Pulse)

### Interpretation Guidelines

| If you find... | It means... |
|----------------|-------------|
| DRUMS syncopation high | More perceived Drive |
| VOCALS dense everywhere | More perceived Drive, less Pulse |
| DRUMS on-beat consistent | More perceived Pulse |
| VOCALS on-beat active | LESS perceived Pulse (masking) |
| DRUMS timing variable (1/8) | More perceived Roll |
| Higher tempo (any stem) | More perceived Roll |
