#!/usr/bin/env python3
"""
Classify anchored drum onsets into 6 drum classes.

Runs after Step 6.2 (Tukey filtering) for the drums stem. For each L2 anchored
pattern file in ``6.2_filtered_patterns/drums`` it classifies the drum at each
of MY onset times (no re-detection by the CNN; the onset times from the anchored
file are used directly) with the DrumTranscriber CNN and writes two columns back
into the file in place: ``predicted_class`` and ``confidence``. The classes are
used e.g. by the GM drum mapping of the anchored MIDI export.

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
ATTACK_BACKTRACK_S = 0.020   # start window 20 ms before onset -> capture attack
WINDOW_CAP_S = 1.0           # cap classification window at 1 s (always padded)


def _read_header_comments(csv_path):
    """Return the leading ``#`` metadata lines of an anchored CSV, verbatim."""
    with open(csv_path) as fh:
        return [ln for ln in fh if ln.startswith('#')]


def classify_anchored_file(csv_path, samples, sr, model):
    """
    Classify onsets in one anchored CSV; write ``predicted_class`` + ``confidence``
    back in place.

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

    Returns
    -------
    n_onsets : int
        Number of classified onsets.
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

    return int(valid.sum())


def classify_all_anchored_drums(drums_wav_path, filtered_dir,
                                sr=44100, verbose=True):
    """
    Classify all L2 anchored drum files (writes class columns back into 6.2).

    Parameters
    ----------
    drums_wav_path : str or Path
        Path to the isolated drums stem (1_stems/drums.wav).
    filtered_dir : str or Path
        6.2_filtered_patterns/drums (input; class columns written back here).
    sr : int
        Sample rate for loading drums.wav.
    verbose : bool

    Returns
    -------
    dict
        {'files', 'onsets', 'per_file'}. Empty (files=0) if no L2 files.
    """
    if not DRUM_CLASSIFICATION_AVAILABLE:
        raise RuntimeError(f"Drum classification unavailable: {_IMPORT_ERROR}")

    filtered_dir = Path(filtered_dir)
    files = sorted(p for p in filtered_dir.glob('*_L2_*_anchored.csv')
                   if not p.name.startswith('._'))
    if not files:
        if verbose:
            print("  Drum classification - SKIPPED (no L2 anchored files)")
        return {'files': 0, 'onsets': 0, 'per_file': {}}

    if not Path(drums_wav_path).exists():
        raise FileNotFoundError(f"Drums stem not found: {drums_wav_path}")

    samples, _ = librosa.load(str(drums_wav_path), sr=sr, mono=True)
    transcriber = DrumTranscriber()

    tot_onsets = 0
    per_file = {}
    for f in files:
        n_onsets = classify_anchored_file(str(f), samples, sr, transcriber.model)
        tot_onsets += n_onsets
        per_file[f.name] = {'onsets': n_onsets}

    if verbose:
        print(f"  ✓ Drum classification: {tot_onsets} onsets across {len(files)} L2 files "
              f"(class + confidence written back)")

    return {'files': len(files), 'onsets': tot_onsets, 'per_file': per_file}
