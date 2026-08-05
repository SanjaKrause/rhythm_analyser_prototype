# Double-Tempo Bug — Fix & Full Rerun Plan

Status: **2026-08-05.** Code fix DONE and verified. Cleanup + rerun NOT yet executed.

## 1. The bug (found via songs 112 / 138)

Songs detected **entirely in double time** were silently lost:

1. The 135-BPM rule (`adaptive rebasing`) correctly fires (>50 % of bars > 135 BPM)
   and sets `base_override = high_median / 2` (the true tempo).
2. All bars then classify as `double` → `double` becomes the **dominant** class.
3. `build_corrected_from_raw` only repairs the *minority* class → the double bars
   stay unchanged (~2× true tempo).
4. The usability filter is centred on `base_override` (±10 %) → **every bar
   unusable → whole song dropped** (no sections, absent from collected data).

Scan result (532 songs): **exactly 40 songs** have `usable_count = 0`, and all 40
match this pattern (override fired ⟺ broken, one-to-one — override fired in ZERO
healthy songs). 30 of the 40 are 4/4 → recoverable for the ML corpus
(~353 → ~383 songs). The lost songs cluster at slow tempi (68–93 BPM
ballads/half-time — a biased subset worth having back).

Affected song IDs:
108, 112, 133, 134, 138, 146, 157, 18, 184, 200, 214, 221, 222, 225, 237, 239,
251, 258, 28, 288, 307, 330, 355, 357, 365, 38, 382, 386, 408, 416, 42, 442,
450, 451, 494, 502, 509, 56, 78, 92

## 2. The fix (DONE)

`loop_extractor/analysis/correct_bars.py`, in `correct_downbeats()`: when
`base_override is not None`, force `dominant_major='normal'` — i.e. correct all
bars TOWARDS the override base (merge `double` bars pairwise, split `half`
bars) instead of letting the dominant-class logic keep them.

- 492 healthy songs: **provably unchanged** (their code path has
  `base_override=None`).
- Phase caveat: pairwise merging adopts the Beat Transformer's downbeat phase;
  which merged half-bar carries the true "One" is undetermined (50/50).
  Documented in a code comment; mention as limitation in the thesis.

Verified offline:
- 138 (Buy Dirt): 122 bars @178 → **61 bars @ 89.1 BPM, 61/61 usable** (was 0).
- 112 (Hrs & Hrs): 157 → 79 bars @ 69.8, 78/79 usable (odd count → 1 tail bar).

## 3. Cleanup before the batch (NOT YET DONE — run this first!)

`--reuse-existing` skips the correction step if `3_corrected/` exists, so the 40
songs MUST have their stale outputs removed first. Only two folders per song —
everything from onsets onward re-runs on every batch anyway
(`skip_existing` is hard-wired False in batch mode; only stems/beats/
correction/tempo-plots/SongFormer/snippet/pironio/spotify/yodfat are
reuse-gated, and all of those except correction+tempo-plots are
downbeat-independent → keep).

Self-contained cleanup script (identifies the 40 songs by the override marker,
deletes `3_corrected/` + `3.5_tempo_plots/`):

```python
# python cleanup_double_tempo_songs.py   (env: loop_extractor_main)
from pathlib import Path
import pandas as pd, shutil
ROOT = Path('/Volumes/PortableSSD/06_Testing/output all')
n = 0
for d in sorted(ROOT.iterdir()):
    if not d.is_dir() or d.name in ('batch_analysis', 'collected_data',
                                    '_batch_analysis', 'snippet_ratio_batch_analysis'):
        continue
    corr = d / '3_corrected'
    if not corr.exists():
        continue
    txts = [f for f in corr.glob('*.txt') if not f.name.startswith('._')]
    if not txts:
        continue
    df = pd.read_csv(txts[0], sep='\t', comment='#')
    if len(df) and str(df['bpm_threshold'].iloc[0]) != 'none':   # override fired = broken song
        for sub in ('3_corrected', '3.5_tempo_plots'):
            p = d / sub
            if p.exists():
                shutil.rmtree(p)
                print(f'deleted {d.name}/{sub}')
        n += 1
print(f'\ncleaned {n} songs (expected: 40)')
```

## 4. Batch rerun (GUI, as always)

Run the usual GUI batch (env `loop_extractor_main`, `python gui.py`):
input dir = `/Volumes/PortableSSD/mastabfiles/renamed`, output =
`/Volumes/PortableSSD/06_Testing/output all`, **reuse existing ON**,
onset mode **madmom**, anchoring **double**, snippet times automatic.
The 492 songs reuse their correction; the 40 hit the fixed code.

### Post-batch verification (before touching the ML)

- Re-scan headers: all 40 songs should now have `usable_count > 0` and
  `tempo ≈ usable_base` (e.g. 138 → ~61 bars @ ~89 BPM). No healthy song's
  `3_corrected` file may have a newer mtime than the batch start except the 40.
- Check the 40 now have `6.x` outputs and appear in
  `collected_data/drums/L2_ratio50.csv` (time_signature==4 subset: expect
  ~+30 songs vs the old 353).

## 5. Downstream cascade (order matters)

1. **Collect**: batch mode runs `collect_data` automatically at the end;
   otherwise: `conda run -n loop_extractor_main python
   loop_extractor/batch_analysis/collect_data.py
   "/Volumes/PortableSSD/06_Testing/output all" drums`
   **NEW (2026-08-05): `MIN_REPETITIONS = 3` filter added to
   `collect_section_data`** — sections with fewer than 3 loops are dropped at
   collection (histogram stats on <3 loops are unreliable: 1 loop = binary
   strength/single-value median/IQR 0; ≤2 loops = running-mean instead of
   Tukey filtering). In the old corpus this removes 22 of 352 4/4 songs
   (8× 1 loop, 14× 2 loops). Applies to ALL collected files (L1/L2/L4,
   all ratios, maxsel, noHats). Expected new corpus size: old 352 − 22
   + recovered double-tempo songs that pass both filters.
2. Copy `collected_data/drums/L2_ratio50.csv` →
   `ML_Notebooks/collected_data_july26/drums_L2_ratio50.csv`
   (back up the old one first). Same for `spotify_all.csv`, `yodfat_snippet.csv`
   if regenerated.
3. **New holdout split**: re-run `FEATURE CALCULATION & EVALUATION/
   1_stratified_holdout_split.ipynb` → new `holdout_split.csv`
   (~+30 songs; genre-stratified; verify floor/ceil-optimal as before).
   NOTE: this changes ALL folds → every ML number changes slightly.
4. **Features**: 1b (statistical, 19), 1c (yodfat), 6 (syncopation, keep the
   dedup), 3 (feature reduction → VIF), 5_pick (→ `our_features.csv`;
   the post-VIF 31 MAY shift with the new train set — check & update
   `statistical_featureset_docu.md` if the set changes).
5. **ML suite** (env `ML_notebooks`, headless via
   `caffeinate -is conda run -n ML_notebooks jupyter nbconvert --to notebook
   --execute --inplace --ExecutePreprocessor.timeout=-1 <nb>`):
   - repeated: NB4, 4b, 10, 10b, 16 + baselines 11, 12, 13
   - nested (long, overnight): NB5 (~3 h), 5b, 6, 17 (~1.5 h)
   - summaries: NB14 → NB15
   - Jupyter save-race: close/reload open notebook tabs before headless runs.
6. **Docs**: update `final_ML_results.md` (n songs, split sizes, all tables),
   `elasticnet_results.md`, `nested_exploratory_results.md` headline numbers;
   thesis §4.2 already describes the FIXED behaviour — add one sentence on the
   downbeat-phase ambiguity of merged songs as a limitation.

## 6. Expectations

- Results should move only slightly (n +~30, ~+8 %); recovered songs are slow
  ballads/half-time → watch whether Roll/Drive signal strengthens.
- The two-regime structure and all protocol decisions (fixed-default RF,
  near-ridge EN, ≤0.01 selection optimism) are expected to hold; verify each in
  the reruns rather than assuming.
