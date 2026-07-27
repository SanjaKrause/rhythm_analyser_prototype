#!/usr/bin/env python3
"""
Classify anchored drum onsets into 6 drum classes + write a cymbal-filtered copy.

Runs after Step 6.2 (Tukey filtering) for the drums stem. For each L2 anchored
pattern file in ``6.2_filtered_patterns/drums`` it:

  1. Classifies the drum at each of MY onset times (no re-detection by the CNN;
     the onset times from the anchored file are used directly) with the
     DrumTranscriber CNN and writes two columns back into the file in place:
     ``predicted_class`` and ``confidence``.

  2. Writes a cymbal-filtered copy to ``6.2_filtered_patterns_noHats/drums`` with
     the same filename. Onsets classified as a cymbal (crash / hihat_c / ride)
     have their ``onset_time``, ``phase``, ``tick_phase``, ``predicted_class`` and
     ``confidence`` cleared, so the row reads as an empty grid slot (no event) -
     matching how genuine no-onset rows look. ``grid_time`` / ``grid_phase`` are
     left intact, exactly as in real empty rows.

The classification window starts 20 ms before the onset (to capture the attack)
and is capped at 1 s. Windows are cut from the full drums.wav using the same
helpers the CNN uses internally, so fidelity matches the model's own onsets.

Environment: loop_extractor_main (numpy, pandas, librosa, tensorflow)
"""

import sys
from pathlib import Path

# Add drumtranscriber to path (same as utils/drumtranscriber_interface.py)
DRUMTRANSCRIBER_PATH = Path(__file__).parent.parent.parent / 'drumtranscriber'
sys.path.insert(0, str(DRUMTRANSCRIBER_PATH))

try:
    import numpy as np
    import pandas as pd
    import librosa
    from DrumTranscriber import DrumTranscriber
    from dt_utils.audio_utils import get_onset_samples, get_mel_spectrogram
    DRUM_CLASSIFICATION_AVAILABLE = True
except ImportError as e:  # pragma: no cover - depends on optional model deps
    DRUM_CLASSIFICATION_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# Model output classes, in the fixed column order the CNN emits (see dt_utils.config)
LABELS = ['crash', 'hihat_c', 'kick_drum', 'ride', 'snare', 'tom_h']
# Cymbals cleared in the noHats copy (the model has no separate open-hihat class;
# open hihats fall into hihat_c / ride / crash, all removed here).
CYMBAL_CLASSES = ('crash', 'hihat_c', 'ride')
ATTACK_BACKTRACK_S = 0.020   # start window 20 ms before onset -> capture attack
WINDOW_CAP_S = 1.0           # cap classification window at 1 s (always padded)


def _read_header_comments(csv_path):
    """Return the leading ``#`` metadata lines of an anchored CSV, verbatim."""
    with open(csv_path) as fh:
        return [ln for ln in fh if ln.startswith('#')]


def classify_anchored_file(csv_path, samples, sr, model,
                           nohats_path=None, remove_classes=CYMBAL_CLASSES):
    """
    Classify onsets in one anchored CSV; write ``predicted_class`` + ``confidence``
    back in place; optionally write a cymbal-filtered copy to ``nohats_path``.

    Parameters
    ----------
    csv_path : str
        Path to an L2 anchored CSV (6.2_filtered_patterns/drums).
    samples : np.ndarray
        Mono drums.wav samples at ``sr`` (full song).
    sr : int
        Sample rate.
    model : keras model
        The DrumTranscriber Keras model (``DrumTranscriber().model``).
    nohats_path : str, optional
        If given, write the cymbal-filtered copy here.
    remove_classes : tuple
        Classes cleared in the noHats copy.

    Returns
    -------
    (n_onsets, n_removed) : tuple of int
        Number of classified onsets, number cleared in the noHats copy.
    """
    header = _read_header_comments(csv_path)
    df = pd.read_csv(csv_path, comment='#')
    # idempotent re-run: drop columns we are about to (re)write
    df = df.drop(columns=[c for c in ('predicted_class', 'confidence') if c in df.columns])

    ot = df['onset_time'].values.astype(float)
    valid = ~np.isnan(ot)                       # only rows with an actual onset get classified

    probs = np.full((len(df), len(LABELS)), np.nan)
    if valid.any():
        back = int(ATTACK_BACKTRACK_S * sr)
        osamp = (ot[valid] * sr).astype(int)
        starts = np.maximum(0, osamp - back)
        nxt = np.append(osamp[1:], min(osamp[-1] + sr, len(samples)))
        ends = np.minimum(nxt, starts + int(WINDOW_CAP_S * sr))
        frames = list(zip(starts.astype(int), ends.astype(int)))
        wins = get_onset_samples(samples, sr, onset_frames=frames)
        mel = np.array([get_mel_spectrogram(w, sr) for w in wins])
        mel = np.expand_dims(mel, -1).repeat(3, axis=-1)
        probs[valid] = model.predict(mel, verbose=0)

    pdf = pd.DataFrame(probs, columns=LABELS)
    df['predicted_class'] = pdf.idxmax(axis=1).values     # NaN on non-onset rows
    df['confidence'] = pdf.max(axis=1).values

    # write class+confidence back into the 6.2 file (preserve # header)
    with open(csv_path, 'w') as fh:
        fh.writelines(header)
        df.to_csv(fh, index=False)

    n_removed = 0
    if nohats_path is not None:
        nh = df.copy()
        rm = nh['predicted_class'].isin(remove_classes)
        # clear the cymbal onsets so they read as empty grid slots (no event)
        nh.loc[rm, ['onset_time', 'phase', 'tick_phase',
                    'predicted_class', 'confidence']] = np.nan
        Path(nohats_path).parent.mkdir(parents=True, exist_ok=True)
        with open(nohats_path, 'w') as fh:
            fh.writelines(header)
            nh.to_csv(fh, index=False)
        n_removed = int(rm.sum())

    return int(valid.sum()), n_removed


def classify_and_filter_anchored_drums(drums_wav_path, filtered_dir, nohats_dir,
                                       sr=44100, remove_classes=CYMBAL_CLASSES,
                                       verbose=True):
    """
    Classify all L2 anchored drum files and produce the cymbal-filtered noHats copies.

    Parameters
    ----------
    drums_wav_path : str or Path
        Path to the isolated drums stem (1_stems/drums.wav).
    filtered_dir : str or Path
        6.2_filtered_patterns/drums (input; class columns written back here).
    nohats_dir : str or Path
        6.2_filtered_patterns_noHats/drums (output; cymbal-filtered copies).
    sr : int
        Sample rate for loading drums.wav.
    remove_classes : tuple
        Classes cleared in the noHats copies.
    verbose : bool

    Returns
    -------
    dict
        {'files', 'onsets', 'removed', 'per_file'}. Empty (files=0) if no L2 files.
    """
    if not DRUM_CLASSIFICATION_AVAILABLE:
        raise RuntimeError(f"Drum classification unavailable: {_IMPORT_ERROR}")

    filtered_dir = Path(filtered_dir)
    nohats_dir = Path(nohats_dir)
    files = sorted(p for p in filtered_dir.glob('*_L2_*_anchored.csv')
                   if not p.name.startswith('._'))
    if not files:
        if verbose:
            print("  Drum classification - SKIPPED (no L2 anchored files)")
        return {'files': 0, 'onsets': 0, 'removed': 0, 'per_file': {}}

    if not Path(drums_wav_path).exists():
        raise FileNotFoundError(f"Drums stem not found: {drums_wav_path}")

    samples, _ = librosa.load(str(drums_wav_path), sr=sr, mono=True)
    transcriber = DrumTranscriber()

    tot_onsets = tot_removed = 0
    per_file = {}
    for f in files:
        n_onsets, n_removed = classify_anchored_file(
            str(f), samples, sr, transcriber.model,
            nohats_path=str(nohats_dir / f.name),
            remove_classes=remove_classes,
        )
        tot_onsets += n_onsets
        tot_removed += n_removed
        per_file[f.name] = {'onsets': n_onsets, 'removed': n_removed}

    if verbose:
        kept = tot_onsets - tot_removed
        print(f"  ✓ Drum classification: {tot_onsets} onsets across {len(files)} L2 files "
              f"(class + confidence written back)")
        print(f"    noHats: kept {kept} kick/snare/tom, cleared {tot_removed} "
              f"cymbal onsets -> {nohats_dir}")

    return {'files': len(files), 'onsets': tot_onsets,
            'removed': tot_removed, 'per_file': per_file}
