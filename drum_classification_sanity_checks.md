# Drum Classification — Sanity Checks

How we validated that classifying **our own anchored onsets** with the DrumTranscriber
CNN (instead of letting the CNN detect onsets itself) produces trustworthy drum labels,
before wiring it into the pipeline (Step 6.2.6).

## What the step does

For each `*_L2_*_anchored.csv` in `6.2_filtered_patterns/drums`:

1. Take **our** onset times from the anchored file (no re-detection by the CNN).
2. Cut a window per onset: start **20 ms before** the onset (to catch the attack),
   cap length at **1 s**, pad symmetrically — same helpers the CNN uses internally.
3. Classify each window → `predicted_class` + `confidence`, written back into the file.
4. Write a cymbal-filtered copy to `6.3_filtered_patterns_noHats/drums`, clearing
   onsets classified as **crash / hihat_c / ride** so they read as empty grid slots.

The model has exactly **6 classes** (`crash, hihat_c, kick_drum, ride, snare, tom_h`) —
confirmed against both `dt_utils/config.py` (`LABELS_INDEX`) and the model's output
layer shape `(None, 6)`. There is **no open-hihat class**; open hihats fall into
`hihat_c` / `ride` / `crash`, all of which the noHats step removes.

## The four checks (per track)

The prototype prints these and writes a combined `drum_class_summary.csv`
(both the CNN's own detected onsets **and** our onsets, with class + confidence +
the 6 class probabilities). All are computed on the **same analysis window** — the
CNN's onset detection is restricted to `[min(our onsets), max(our onsets)]` so the two
sets are directly comparable, not full-song vs snippet.

### 1. Window fidelity — confidence comparison
Mean confidence on **our** onsets vs the CNN's **own** onsets. If our windows were
badly cut, confidence would drop. In practice ours is **equal or higher**, so feeding
our onset times does not degrade the model.

| track | our onsets | CNN's own onsets |
|-------|-----------:|-----------------:|
| STAY  | **0.721**  | 0.701 |
| HUMBLE| **0.762**  | 0.733 |

### 2. Onset alignment — are our onsets real?
Fraction of our onsets that sit within **30 ms** of an onset the CNN detected on its
own. High alignment ⇒ our onset times land on real transients.

- STAY: **135/147 (92%)**
- HUMBLE: **122/128 (95%)**

### 3. Class agreement (where both fire)
Of the aligned onsets, how often the two agree on **which drum**. This is the honest
reliability number — it is **not** expected to be ~100% because the CNN emits many
extra ambiguous onsets, but low agreement flags trouble.

- STAY: 59/135 = **43.7%**
- HUMBLE: 67/122 = **54.9%**

### 4. Musical sanity — class × grid position
Crosstab of `predicted_class` against **within-bar 16th position** (`tick_16th % 16`)
for our onsets. A real drum track should show **kick on beats 1 & 3** (pos 0, 8) and
**snare on beats 2 & 4** (pos 4, 12). This is the most decisive check.

**STAY (works — textbook backbeat):**

| pos16 | beat | dominant class |
|------:|:----:|:---------------|
| 0     | 1    | **kick ×21** |
| 4     | 2    | **snare ×13** |
| 8     | 3    | **kick ×26** |
| 12    | 4    | **snare ×18** |

**HUMBLE (fails — synthetic clap is out-of-distribution):** kick placement is sane
(pos 0/1/9/11 heavy kick) but the backbeat clap is labelled **ride**, not snare
(snare = 5 of 128 onsets). Verified by ear: the clap is misclassified as ride.

## Interpretation / limits

- **Plumbing is sound**: our onsets → windows → classifier is validated by checks 1–2
  (high confidence, 92–95% alignment). The bottleneck is the **model**, not our onsets.
- **Model limit**: DrumTranscriber was trained on acoustic kits. Tracks built on
  **synthetic claps / electronic percussion** (much of modern pop/hip-hop) can have
  their backbeat misclassified as ride/crash — the snare/backbeat then goes missing.
  A track with a suspiciously low snare count is the warning sign.
- **noHats caveat**: on clap-heavy tracks, clearing `ride` also clears mislabelled
  backbeat claps, so the noHats copy can lose the backbeat. Fine for acoustic-snare
  tracks (STAY: 37 cleared, backbeat kept), lossy for clap tracks (HUMBLE: 55 cleared).

## How to re-run the checks

Prototype (prints all four checks + writes `drum_class_summary.csv`):

```bash
conda activate loop_extractor_main
python /tmp/drumproto.py "/path/to/output/<track dir>"
```

In the pipeline the classification runs automatically as **Step 6.2.6** for the drums
stem (`loop_extractor/analysis/classify_anchored_drums.py`); the summary CSV / printed
checks are prototype-only and are the recommended spot-check when validating a new set.
