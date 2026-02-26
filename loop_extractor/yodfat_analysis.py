#!/usr/bin/env python3
"""
Step 14: Yodfat Rhythmic Complexity Analysis

Calculates rhythmic complexity from audio using onset strength cross-correlation.
Based on: https://github.com/arnavlavan/Rhythmic-Complexity-from-Audio

Method:
1. Separate audio into harmonic and percussive components (HPSS)
2. Extract beats and onset envelope from percussive component
3. Calculate normalized cross-correlation for 3 segment lengths:
   - Quarter bar (1 beat)
   - Half bar (2 beats)
   - Full bar (4 beats)
4. High cross-correlation = low rhythmic complexity (more repetitive)
   Low cross-correlation = high rhythmic complexity (more varied)

Outputs 15 metrics:
- For each segment (quart, half, bar):
  - onscc_*_avg: mean cross-correlation
  - onscc_*_std: std of cross-correlation
  - onscc_*_lag_avg: mean lag
  - onscc_*_lag_med: median lag
  - onscc_*_lag_std: std of lag

Citation:
Adam Yodfat, "A Thousand Songs and a Song: Five Decades of Mizrahit and Rock Songs
in Israel - Musical Analysis", PhD Dissertation (Hebrew University of Jerusalem, 2020).
"""

import json
import gc
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Fix scipy.signal.hann deprecation issue (moved to scipy.signal.windows.hann in newer scipy)
import scipy.signal
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = scipy.signal.windows.hann

import librosa


def norm_cc(a: np.ndarray, b: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """
    Calculate normalized cross-correlation.

    Parameters
    ----------
    a : np.ndarray
        Full onset envelope array to search in
    b : np.ndarray
        Segment pattern to match
    start_idx : int
        Start frame index
    end_idx : int
        End frame index

    Returns
    -------
    np.ndarray
        Cross-correlation values for each position
    """
    res = np.zeros(end_idx + start_idx)
    b_norm = (b - np.mean(b)) / (np.std(b) + 1e-10)  # Avoid division by zero

    for i in range(start_idx, end_idx):
        vec_a = a[i:i + len(b)]
        if len(vec_a) < len(b):
            res[i - start_idx] = 0
        else:
            std_a = np.std(vec_a)
            if std_a < 1e-10:  # Avoid division by zero
                res[i - start_idx] = 0
            else:
                vec_a_norm = (vec_a - np.mean(vec_a)) / std_a
                res[i - start_idx] = np.correlate(vec_a_norm, b_norm) / len(vec_a)

    return res


def compute_segment_cc(
    oenv: np.ndarray,
    beat_frames: np.ndarray,
    seg_length: int,
    tempo: float,
    sr: int,
    hop_length: int
) -> Dict[str, float]:
    """
    Compute cross-correlation metrics for a given segment length.

    Parameters
    ----------
    oenv : np.ndarray
        Onset envelope
    beat_frames : np.ndarray
        Beat frame indices
    seg_length : int
        Segment length in beats (1, 2, or 4)
    tempo : float
        Detected tempo in BPM
    sr : int
        Sample rate
    hop_length : int
        Hop length in samples

    Returns
    -------
    dict
        Dictionary with avg, std, lag_avg, lag_med, lag_std
    """
    n_segments = int(np.ceil(len(beat_frames) / seg_length))
    onscc = np.zeros(n_segments)
    onscc_t = np.zeros(n_segments)
    onscc_lag = np.zeros(n_segments)
    cnt = 0

    for beat_idx in range(0, len(beat_frames) - 2 * seg_length, seg_length):
        frame_start = beat_frames[beat_idx]
        frame_end = beat_frames[beat_idx + seg_length]
        onset_subvec = oenv[frame_start:frame_end]

        end_frame = beat_frames[beat_idx + 2 * seg_length] + 10
        res = norm_cc(oenv, onset_subvec, frame_start, end_frame)

        if len(res) > 1:
            onscc[cnt] = np.max(res[1:])
            onscc_lag[cnt] = np.argmax(res[1:])
        onscc_t[cnt] = frame_start * (hop_length / sr)
        cnt += 1

    # Filter valid values (where time > 0)
    valid_mask = onscc_t > 0
    valid_cc = onscc[valid_mask]
    valid_lag = onscc_lag[valid_mask] * (tempo / 60 / sr * hop_length)

    if len(valid_cc) == 0:
        return {
            'avg': 0.0,
            'std': 0.0,
            'lag_avg': 0.0,
            'lag_med': 0.0,
            'lag_std': 0.0
        }

    return {
        'avg': float(np.mean(valid_cc)),
        'std': float(np.std(valid_cc)),
        'lag_avg': float(np.mean(valid_lag)),
        'lag_med': float(np.median(valid_lag)),
        'lag_std': float(np.std(valid_lag))
    }


def run_yodfat_analysis(
    audio_file: str,
    output_dir: str,
    track_id: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Yodfat rhythmic complexity analysis on an audio file.

    Parameters
    ----------
    audio_file : str
        Path to input audio file (WAV or MP3)
    output_dir : str
        Output directory for results
    track_id : str
        Track identifier for output files
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Results dictionary with all metrics
    """
    output_path = Path(output_dir) / "14_yodfat"
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        'track_id': track_id,
        'metrics': {},
        'errors': []
    }

    if verbose:
        print(f"\n[Step 14: Yodfat Rhythmic Complexity Analysis]")
        print(f"  Input: {Path(audio_file).name}")

    try:
        # Load audio
        if verbose:
            print("  Loading audio file...")
        y, sr = librosa.load(audio_file)
        y, _ = librosa.effects.trim(y)  # Trim silence
        hop_length = 512

        # Harmonic/Percussive separation
        if verbose:
            print("  Harmonic/percussive separation...")
        D = librosa.stft(y)
        gc.collect()
        harm, perc = librosa.decompose.hpss(D)
        gc.collect()
        y_percussive = librosa.istft(perc)
        gc.collect()

        # Beat tracking on percussive signal
        if verbose:
            print("  Beat tracking...")
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # Handle tempo being an array in newer librosa versions
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        track_duration = len(y) / sr

        if verbose:
            print(f"  Detected tempo: {tempo:.1f} BPM")
            print(f"  Duration: {track_duration:.1f}s")
            print(f"  Beats detected: {len(beat_frames)}")

        # Onset envelope extraction
        if verbose:
            print("  Extracting onset envelope...")
        oenv = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)

        # Calculate cross-correlation for each segment length
        if verbose:
            print("  Calculating cross-correlation for quarter bar (1 beat)...")
        quart_metrics = compute_segment_cc(oenv, beat_frames, seg_length=1,
                                           tempo=tempo, sr=sr, hop_length=hop_length)

        if verbose:
            print("  Calculating cross-correlation for half bar (2 beats)...")
        half_metrics = compute_segment_cc(oenv, beat_frames, seg_length=2,
                                          tempo=tempo, sr=sr, hop_length=hop_length)

        if verbose:
            print("  Calculating cross-correlation for full bar (4 beats)...")
        bar_metrics = compute_segment_cc(oenv, beat_frames, seg_length=4,
                                         tempo=tempo, sr=sr, hop_length=hop_length)

        # Store all metrics
        results['metrics'] = {
            'tempo': tempo,
            'duration': track_duration,
            'n_beats': len(beat_frames),
            # Quarter bar (1 beat) metrics
            'onscc_quart_avg': quart_metrics['avg'],
            'onscc_quart_std': quart_metrics['std'],
            'onscc_quart_lag_avg': quart_metrics['lag_avg'],
            'onscc_quart_lag_med': quart_metrics['lag_med'],
            'onscc_quart_lag_std': quart_metrics['lag_std'],
            # Half bar (2 beats) metrics
            'onscc_half_avg': half_metrics['avg'],
            'onscc_half_std': half_metrics['std'],
            'onscc_half_lag_avg': half_metrics['lag_avg'],
            'onscc_half_lag_med': half_metrics['lag_med'],
            'onscc_half_lag_std': half_metrics['lag_std'],
            # Full bar (4 beats) metrics
            'onscc_bar_avg': bar_metrics['avg'],
            'onscc_bar_std': bar_metrics['std'],
            'onscc_bar_lag_avg': bar_metrics['lag_avg'],
            'onscc_bar_lag_med': bar_metrics['lag_med'],
            'onscc_bar_lag_std': bar_metrics['lag_std'],
        }

        # Save results to JSON
        output_json = output_path / f"{track_id}_yodfat_metrics.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)

        results['output_json'] = str(output_json)

        if verbose:
            print(f"  Saved: {output_json.name}")
            print(f"\n  Rhythmic Complexity Metrics:")
            print(f"    Quarter bar CC avg: {quart_metrics['avg']:.4f}")
            print(f"    Half bar CC avg:    {half_metrics['avg']:.4f}")
            print(f"    Full bar CC avg:    {bar_metrics['avg']:.4f}")
            print(f"  (Higher CC = more repetitive = lower rhythmic complexity)")
            print(f"  Yodfat analysis completed")

    except Exception as e:
        error_msg = f"Yodfat analysis failed: {str(e)}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ERROR: {error_msg}")

    return results


def run_yodfat_section_analysis(
    sections_dir: str,
    output_dir: str,
    track_id: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Yodfat rhythmic complexity analysis on all section WAV files.

    Parameters
    ----------
    sections_dir : str
        Path to 9.1_sections directory containing section WAV files
    output_dir : str
        Output directory for results (14_yodfat)
    track_id : str
        Track identifier for output files
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Results dictionary with metrics for each section
    """
    sections_path = Path(sections_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all section WAV files (not macOS ._ files)
    section_wavs = sorted([
        f for f in sections_path.glob('*_section.wav')
        if not f.name.startswith('._')
    ])

    results = {
        'track_id': track_id,
        'sections': {},
        'errors': []
    }

    if verbose:
        print(f"\n[Step 15.1: Yodfat Section Analysis]")
        print(f"  Input: {sections_path.name}")
        print(f"  Found {len(section_wavs)} section files")

    for wav_path in section_wavs:
        # Extract section ID from filename
        # e.g., SecNo1_L4_chorus_0.1344_section.wav -> SecNo1_L4_chorus_0.1344
        section_id = wav_path.stem.replace('_section', '')

        if verbose:
            print(f"\n  Processing: {section_id}")

        try:
            # Load audio
            y, sr = librosa.load(str(wav_path))
            y, _ = librosa.effects.trim(y)
            hop_length = 512

            # Harmonic/Percussive separation
            D = librosa.stft(y)
            gc.collect()
            harm, perc = librosa.decompose.hpss(D)
            gc.collect()
            y_percussive = librosa.istft(perc)
            gc.collect()

            # Beat tracking on percussive signal
            tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

            # Handle tempo being an array in newer librosa versions
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo = float(tempo)

            track_duration = len(y) / sr

            # Onset envelope extraction
            oenv = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)

            # Calculate cross-correlation for each segment length
            quart_metrics = compute_segment_cc(oenv, beat_frames, seg_length=1,
                                               tempo=tempo, sr=sr, hop_length=hop_length)
            half_metrics = compute_segment_cc(oenv, beat_frames, seg_length=2,
                                              tempo=tempo, sr=sr, hop_length=hop_length)
            bar_metrics = compute_segment_cc(oenv, beat_frames, seg_length=4,
                                             tempo=tempo, sr=sr, hop_length=hop_length)

            # Store section metrics
            results['sections'][section_id] = {
                'audio_file': wav_path.name,
                'metrics': {
                    'tempo': tempo,
                    'duration': track_duration,
                    'n_beats': len(beat_frames),
                    'onscc_quart_avg': quart_metrics['avg'],
                    'onscc_quart_std': quart_metrics['std'],
                    'onscc_quart_lag_avg': quart_metrics['lag_avg'],
                    'onscc_quart_lag_med': quart_metrics['lag_med'],
                    'onscc_quart_lag_std': quart_metrics['lag_std'],
                    'onscc_half_avg': half_metrics['avg'],
                    'onscc_half_std': half_metrics['std'],
                    'onscc_half_lag_avg': half_metrics['lag_avg'],
                    'onscc_half_lag_med': half_metrics['lag_med'],
                    'onscc_half_lag_std': half_metrics['lag_std'],
                    'onscc_bar_avg': bar_metrics['avg'],
                    'onscc_bar_std': bar_metrics['std'],
                    'onscc_bar_lag_avg': bar_metrics['lag_avg'],
                    'onscc_bar_lag_med': bar_metrics['lag_med'],
                    'onscc_bar_lag_std': bar_metrics['lag_std'],
                },
                'errors': []
            }

            if verbose:
                print(f"    Tempo: {tempo:.1f} BPM, Duration: {track_duration:.1f}s, Beats: {len(beat_frames)}")
                print(f"    CC avg - quart: {quart_metrics['avg']:.4f}, half: {half_metrics['avg']:.4f}, bar: {bar_metrics['avg']:.4f}")

        except Exception as e:
            error_msg = f"{section_id}: {str(e)}"
            results['sections'][section_id] = {
                'audio_file': wav_path.name,
                'metrics': {},
                'errors': [error_msg]
            }
            if verbose:
                print(f"    ERROR: {error_msg}")

    # Save combined results
    output_json = output_path / f"{track_id}_yodfat_sections.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    results['output_json'] = str(output_json)

    if verbose:
        total_sections = len(results['sections'])
        total_errors = sum(len(s['errors']) for s in results['sections'].values())
        print(f"\n  Saved: {output_json.name}")
        print(f"  Processed {total_sections} sections ({total_errors} errors)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 14: Yodfat Rhythmic Complexity Analysis"
    )
    parser.add_argument("audio_file", help="Path to audio file (WAV or MP3)")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--track-id", required=True,
                        help="Track identifier")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")

    args = parser.parse_args()

    results = run_yodfat_analysis(
        audio_file=args.audio_file,
        output_dir=args.output_dir,
        track_id=args.track_id,
        verbose=not args.quiet
    )

    if results['errors']:
        exit(1)
