# Cleanup Plan — Legacy Outputs & Dead Code

Status: **CODE CLEANUP DONE (2026-08-20, branch `code-cleanup2`) — output-data
deletion still pending** (user: "not yet"; dry-run numbers in §1). Reference track used for checks:
`/Volumes/PortableSSD/06_Testing/output all/2_Back in Blood (feat Lil Durk) - Pooh ShiestyLil Durk/`

## Background

The correct, current histogram outputs are:

- `6.6_anchored_rhythm_histograms/<stem>/` — anchored rhythm histograms (step 7)
- `6.7_anchored_beat_histograms/<stem>/` — anchored beat histograms (step 7.1)

Their dependency chain is:

```
corrected downbeats + onset files + SongFormer sections
    └→ 6.1_anchoring          (anchoring.run_anchoring)
         └→ 6.2_filtered_patterns
               ├→ 6.6_anchored_rhythm_histograms
               └→ 6.7_anchored_beat_histograms   (also reads groove-pulse CSV from 6.6)
```

`5_grid` is a **separate branch** (raster/comprehensive-phases CSVs) and is NOT an input
to 6.6/6.7 — but it IS still used by raster plots, microtiming plots, MIDI/audio export,
`merge_plots.py`, and `lepa_data_export.py`. **Keep `5_grid`.**

---

## 1. Output folders to delete (per track, across `output all/`) — PENDING

Dry-run 2026-08-20 (`output all`, 532 tracks): 5.5_rhythm 530 dirs/16.4 GB,
5.6_statistics 530 dirs/3.7 GB, 5.7_beat_histograms 530 dirs/20.7 GB, 6_rms
532 dirs/0.5 GB, legacy onset .mid 2067, legacy bass .mid 3561 (+ `._` sidecars).
`output präsi`: only 4 empty `6_rms` dirs. No 8_midi_drum/9_loops_drum found.
Caveat: deleting legacy .mid leaves 531 tracks without MIDI until step 10 rerun.

| Folder | Verdict | Evidence |
|---|---|---|
| `5.5_rhythm/` | DELETE | Folder name appears nowhere in code. Files (`*_rhythm_histograms_with_style.*`, `*_rhythm_histograms_with_medians_and_iqr.*`, `*_groove_pulse_histograms_filtered.csv`) were produced by dead functions (see §2). Superseded by 6.6. |
| `5.6_statistics/` | DELETE | `*_rhythm_statistics_L{1,2,4}.csv` referenced nowhere. Superseded by step 7.2 (anchored statistics). |
| `5.7_beat_histograms/` | DELETE | Referenced nowhere. Superseded by 6.7. |
| `6_rms/` | DELETE | Empty on reference track. RMS step is commented out in `main.py` (~lines 1804–1826); `raster_plots.py` explicitly ignores `rms_summary_file`. |
| `8_midi/onset/*.mid` (legacy files only) | DELETE | `1bar_flexStart.mid`, `2bar_flexStart.mid`, `4bar_flexStart.mid`, `per_snippet.mid` — produced by the old flexStart/comprehensive-CSV branch. **Replaced** by the new anchored export (`SecNoX_L2_*.mid`, `SecNoX_full_*.mid` + `_gm` variants), already implemented in `midi_export.export_anchored_onset_midi` and wired into main.py step 10a. New files are already written next to the legacy ones for the reference track. |
| `8_midi/bass_pitch/*.mid` (legacy files only) | DELETE | `1bar/2bar/4bar_flexStart_bass.mid`, `per_snippet_bass.mid` — same legacy branch. **Replaced** by `midi_export.export_anchored_pitch_midi` (step 10b): `SecNoX_L2_*_bass.mid`, `SecNoX_full_*_bass.mid`, same anchored windows as the drum MIDI so they line up in a DAW. If the snippet-only `bass_f0.csv` doesn't cover the anchored windows, F0 is auto-re-extracted from `bass.wav` and cached as `1_stems/bass_f0_sections.csv` (original CSV untouched). New files already written for the reference track. |

Plan: dry-run first (list + count matching folders across all tracks), then delete after
confirmation.

## 2. Dead code to remove — DONE (2026-08-20)

All items below were applied on branch `code-cleanup2` (~2,000 lines removed;
verified: syntax, imports, --help, MIDI self-test, anchored MIDI export
byte-identical to the präsi batch output).

### `loop_extractor/analysis/anchored_rhythm_histograms.py`
Four functions with **zero callers** anywhere in the repo (these read from `5_grid` and
wrote the old 5.5 outputs), ~450 lines total:

- `extract_rhythm_histogram_from_flexstart_csv` (line ~990)
- `create_rhythm_histograms_with_style` (line ~1041)
- `extract_phase_statistics_from_csv` (line ~1344)
- `create_rhythm_histograms_with_medians_and_iqr` (line ~1426)

### `loop_extractor/config.py`
- Line ~258: stale `'rms_summary': track_dir / '6_rms' / ...` path entry
- Lines ~156, ~209: related stale comments/docstring mentions

### `loop_extractor/main.py`
- ~Lines 1804–1826: commented-out RMS step block
- Check: `rms_grid_histograms` import (line 84) becomes unused once the block is gone

### `loop_extractor/batch_analysis/merge_plots.py`
- Stale docstring lines (12–14): "rhythm histograms with style", "with medians and IQR" —
  no merge step reads these anymore (only `5_grid` among the 5.x folders is read)

### `loop_extractor/utils/midi_export.py` (after the anchored-MIDI switch + DAW-mode removal)
DAW mode was removed entirely (GUI + code, 2026-08-19), so the legacy
comprehensive-CSV MIDI branch now has NO callers at all. Dead and removable:
- `comprehensive_csv_to_onset_midi` / `comprehensive_csv_to_pitch_midi`
- `_flexstart_to_midi_onset` / `_flexstart_to_midi_pitch`
- `flexstart_to_midi`
- `f0_to_midi` (superseded by `_f0_to_note_events` + `_write_pitch_midi`,
  which align to the window start instead of the first note)

### Export format option (WAV/MP3) — REMOVED (done)
- gui.py: EXPORT FORMAT radio section removed — always WAV
- main.py: `export_format` param + `--export-format` flag removed; audio-examples
  skip-existing check fixed to look for `.wav` (was checking stale `.mp3` names)
- config.py: dead `AUDIO_EXPORT_FORMAT` constant removed
- Still in audio_export.py (now dead unless called with mp3 explicitly):
  `export_audio_to_mp3` + the `export_format` params of `create_audio_examples`
  / `export_stem_loops` and `AUDIO_EXPORT_BITRATE` — candidates for removal

### noHats variant — REMOVED (done, 2026-08-20)
Tested & dropped from the thesis (hi-hats carry signal), so the whole branch is gone:
- main.py: steps 7.6–7.8 (noHats histograms/stats) deleted; step 6.2.6 now does
  classification ONLY (predicted_class/confidence still written into 6.2 —
  needed by the GM MIDI mapping); skip-existing check now looks for the
  predicted_class column instead of the noHats folder
- classify_anchored_drums.py: cymbal-filter/copy logic removed; function renamed
  `classify_and_filter_anchored_drums` → `classify_all_anchored_drums`
- config.py: `nohats_*` path entries removed
- merge_plots.py: `merge_plots_noHats` deleted; collect_data.py: `variant`
  machinery removed (noHats_L2_ratio*.csv no longer produced)
- Existing `*_noHats` output folders (6.2/6.6/6.7/6.8) are orphaned →
  add to the §1 deletion list

### DAW mode — REMOVED (done)
- gui.py: OUTPUT MODE radio section removed (always full detailed pipeline)
- main.py: `daw_ready` parameter, `--daw-ready` flag, and all branches removed
- config.py: `8_midi_drum` / `9_loops_drum` folder switching removed
- Any existing `8_midi_drum/` or `9_loops_drum/` output folders are orphaned →
  DELETE if found in output dirs

### Backup / dated files (whole files)
- `loop_extractor/analysis/anchored_rhythm_histograms_backup.py`
- `loop_extractor/analysis/anchoring_backup.py`
- `loop_extractor/analysis/plots_anchoring_backup.py`
- `loop_extractor/utils/raster_plots_old5-1-2026.py`
  (only file besides config.py that references `6_rms`)

## 3. Explicitly KEEP

- `5_grid/` — inputs for raster plots, microtiming plots, MIDI export, audio export,
  batch merges, LEPA export
- `6.1_anchoring/`, `6.2_filtered_patterns/` — direct inputs of 6.6/6.7
- `6.6_anchored_rhythm_histograms/`, `6.7_anchored_beat_histograms/` — the correct outputs
- Step 7.2 statistics outputs (built from 6.6/6.7 CSVs)

## 4. Open questions (to confirm before executing)

- [x] RESOLVED 2026-08-20: steps 8 (audio examples → `7_audio_examples/`) and
      11 (stem loops → `9_loops/`) removed entirely from main.py, config paths,
      GUI, and merge_plots (groove-click merge). `create_audio_examples`,
      `export_stem_loops`, `apply_fade`, `export_audio_to_wav` deleted from
      audio_export.py (click-track utils kept — used by 11.2 section clicks).
      Existing `7_audio_examples/` and `9_loops/` folders in output dirs are
      now orphaned → add to the §1 deletion list.

- [ ] Delete output folders across **all** tracks in `output all/`, or only some?
- [ ] Any other output dirs (other corpora/SSDs) that should get the same cleanup?
- [ ] Remove `analysis/rms_grid_histograms.py` module entirely too, or keep it
      (currently still imported in `analysis/__init__.py` and `main.py`)?
- [ ] Git-commit the code cleanup separately from any data deletion.
