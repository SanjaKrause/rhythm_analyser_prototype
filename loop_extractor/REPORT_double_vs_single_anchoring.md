# Double vs Single Anchoring: Comparative Analysis Report

**Date:** March 24, 2026
**Dataset:** DRUMS L2 ratio50
**Comparison:** Single anchoring (old pipeline) vs Double anchoring (new pipeline)

---

## Executive Summary

Double anchoring produces **more significant features** than single anchoring while maintaining the same top correlation strengths. The improvement is most pronounced for timing-sensitive targets (Roll: +11%, Pulse: +34%), while onset-based predictions (Drive) remain unchanged. **Recommendation: Use double anchoring.**

---

## 1. Background

### Anchoring Methods

- **Single Anchoring:** Aligns patterns to a reference point at the start of each rhythmic cycle. Timing deviations can accumulate across the pattern.

- **Double Anchoring:** Aligns patterns to reference points at both the start AND end of each rhythmic cycle. This constrains timing deviations and reduces drift accumulation.

### Feature Types Affected

| Feature Type | Description | Expected Impact |
|--------------|-------------|-----------------|
| `_str` (strength) | Onset presence/intensity at each position | Minimal - depends on detection, not alignment |
| `_med` (median) | Median timing deviation at each position | Significant - directly measures timing |
| `_iqr` (IQR) | Timing variability at each position | Significant - measures timing consistency |

---

## 2. Prediction Performance Comparison

### Top Correlations by Target

#### Drive (perceived forward momentum)

| Rank | Single Anchoring | r | Double Anchoring | r |
|------|------------------|---|------------------|---|
| 1 | GP_str_13 | +0.254*** | GP_str_13 | +0.254*** |
| 2 | RH_str_13 | +0.253*** | RH_str_13 | +0.253*** |
| 3 | RP_str_13 | +0.242*** | RP_str_13 | +0.242*** |
| 4 | GP_str_21 | +0.207*** | RP_str_5 | +0.201*** |
| 5 | RH_str_21 | +0.206*** | GP_str_21 | +0.193*** |

**Result:** Identical top predictors. Position 13 (syncopation) dominates both methods.

#### Roll (perceived rhythmic flow)

| Rank | Single Anchoring | r | Double Anchoring | r |
|------|------------------|---|------------------|---|
| 1 | GPB_iqr_1/8 | +0.212*** | BH_iqr_1/8 | +0.204*** |
| 2 | BH_iqr_1/8 | +0.183** | BP_iqr_1/8 | +0.204*** |
| 3 | BP_iqr_1/8 | +0.183** | GPB_iqr_1/8 | +0.203*** |
| 4 | GP_str_18 | +0.178** | GP_str_18 | +0.176** |
| 5 | RH_str_18 | +0.178** | mean_section_tempo | +0.176** |

**Result:** 1/8 note IQR features dominate both. Double anchoring shows more uniform correlations across IQR variants.

#### Pulse (perceived steadiness)

| Rank | Single Anchoring | r | Double Anchoring | r |
|------|------------------|---|------------------|---|
| 1 | groove_pulse_strength | +0.225*** | groove_pulse_strength | +0.225*** |
| 2 | RH_str_8 | +0.224*** | RH_str_8 | +0.222*** |
| 3 | GP_str_8 | +0.223*** | GP_str_8 | +0.221*** |
| 4 | BP_iqr_1/4 | -0.201*** | pulse_strength | +0.218*** |
| 5 | BH_iqr_1/4 | -0.201*** | BP_iqr_1/4 | -0.206*** |

**Result:** Nearly identical. Position 8 and pulse_strength features dominate both methods.

### Summary: Top Correlation Strength

| Target | Single Best r | Double Best r | Difference |
|--------|---------------|---------------|------------|
| Drive | 0.254 | 0.254 | 0.000 |
| Roll | 0.212 | 0.204 | -0.008 |
| Pulse | 0.225 | 0.225 | 0.000 |

**Conclusion:** No meaningful difference in maximum correlation strength.

---

## 3. Feature Discovery Comparison

### Significant Features Count (p < 0.05)

| Target | Single (old) | Double (new) | Change | % Improvement |
|--------|--------------|--------------|--------|---------------|
| Drive | 65 | 65 | 0 | 0% |
| Roll | 66 | 73 | +7 | **+11%** |
| Pulse | 32 | 43 | +11 | **+34%** |
| **Total** | **163** | **181** | **+18** | **+11%** |

### Breakdown by Feature Type

#### Drive Features
| Type | Single | Double | Change |
|------|--------|--------|--------|
| RH_ | 19 | 19 | 0 |
| GP_ | 19 | 19 | 0 |
| RP_ | 19 | 18 | -1 |
| BH_ | 2 | 2 | 0 |
| GPB_ | 2 | 2 | 0 |
| BP_ | 2 | 2 | 0 |
| stats | 2 | 3 | +1 |

#### Roll Features
| Type | Single | Double | Change |
|------|--------|--------|--------|
| RH_ | 19 | 20 | +1 |
| GP_ | 18 | 21 | **+3** |
| RP_ | 16 | 19 | **+3** |
| BH_ | 4 | 3 | -1 |
| GPB_ | 2 | 2 | 0 |
| BP_ | 4 | 4 | 0 |
| stats | 3 | 4 | +1 |

#### Pulse Features
| Type | Single | Double | Change |
|------|--------|--------|--------|
| RH_ | 8 | 11 | **+3** |
| GP_ | 8 | 11 | **+3** |
| RP_ | 5 | 8 | **+3** |
| BH_ | 3 | 4 | +1 |
| GPB_ | 2 | 2 | 0 |
| BP_ | 3 | 4 | +1 |
| stats | 3 | 3 | 0 |

---

## 4. Detailed Feature Analysis

### New IQR Features in Double Anchoring (Roll)

The following position-specific IQR features emerged as significant only with double anchoring:

| Feature | Double r | p-value | Interpretation |
|---------|----------|---------|----------------|
| GP_iqr_10 | +0.171 | 0.0033** | Timing variability at position 10 |
| RP_iqr_10 | +0.171 | 0.0033** | Timing variability at position 10 |
| RH_iqr_10 | +0.171 | 0.0033** | Timing variability at position 10 |
| RH_iqr_27 | +0.170 | 0.0034** | Timing variability at position 27 |
| RP_iqr_27 | +0.170 | 0.0034** | Timing variability at position 27 |
| GP_iqr_27 | +0.170 | 0.0034** | Timing variability at position 27 |

### Strengthened MED Features (Pulse)

| Feature | Single r | Double r | Change |
|---------|----------|----------|--------|
| BH_med_1/16 | -0.139* | -0.157** | Stronger |
| BP_med_1/16 | -0.139* | -0.157** | Stronger |
| GP_med_12 | n.s. | -0.145* | New |
| RP_med_12 | n.s. | -0.145* | New |

### STR Features (Unchanged)

As expected, strength features showed minimal change between anchoring methods:

- Position 13 (Drive): 0.254 in both
- Position 8 (Pulse): 0.224 vs 0.222
- Position 18 (Roll): 0.178 vs 0.176

---

## 5. Interpretation

### Why Double Anchoring Improves Timing Features

1. **Reduced Drift:** Single anchoring allows timing errors to accumulate across a pattern. Double anchoring constrains both ends, reducing systematic drift.

2. **Cleaner IQR Estimates:** With less drift contamination, the IQR (variability) measurements more accurately reflect true timing inconsistency rather than alignment artifacts.

3. **More Sensitive MED Detection:** Median timing deviations are measured more precisely when the reference frame is better defined at both boundaries.

### Why STR Features Are Unaffected

Strength features measure **whether an onset occurs** at a position, not **when exactly** it occurs. This binary/intensity information is independent of the fine-grained timing alignment that anchoring affects.

### Target-Specific Effects

| Target | Primary Features | Anchoring Effect |
|--------|------------------|------------------|
| Drive | STR (syncopation positions) | None - onset-based |
| Roll | IQR (timing variability) | Strong - timing-based |
| Pulse | STR + IQR (steadiness) | Moderate - mixed |

---

## 6. Recommendations

### Primary Recommendation

**Use double anchoring for all future analyses.**

Rationale:
- +18 additional significant features discovered
- +34% improvement in Pulse feature discovery
- +11% improvement in Roll feature discovery
- No loss in Drive prediction capability
- No loss in top correlation strength
- Theoretically more sound alignment method

### Secondary Recommendations

1. **For Drive prediction:** Either method works equally well. STR features at positions 13, 21, 5, 29 are robust predictors.

2. **For Roll/Pulse research:** Double anchoring is strongly preferred due to improved timing feature extraction.

3. **For microtiming studies:** Double anchoring provides cleaner MED and IQR measurements essential for understanding timing nuances.

4. **For backward compatibility:** If comparing to previous single-anchoring results, focus on STR features which are stable across methods.

---

## 7. Conclusions

Double anchoring represents a methodological improvement over single anchoring for rhythm feature extraction. While the maximum prediction strength remains unchanged, the method reveals more statistically significant relationships between timing features and perceptual ratings. This is particularly valuable for understanding the microtiming aspects of groove perception (Roll, Pulse) rather than just onset patterns (Drive).

The additional computational cost of double anchoring is justified by the richer feature set it produces, especially for timing-sensitive analyses.

---

## Appendix: Data Summary

- **Dataset:** DRUMS stem, L2 pattern length, ratio50
- **Songs analyzed:** ~295 (after filtering for time_signature == 4)
- **Targets:** DGA_Drive_blup, DGA_Roll_blup, DGA_Pulse_blup
- **Significance threshold:** p < 0.05
- **Correlation method:** Pearson's r
