"""
Onset detection using librosa.

This module provides onset detection from audio files using librosa's
onset detection with optional refinement and filtering.

Sample Rate and Time Resolution
--------------------------------
By default, librosa.load() resamples audio to 22050 Hz. This determines the
time resolution for onset detection:

Sample rate = 22050 Hz (default):
- Time resolution: 1/22050 = 0.0454 ms per sample
- With hop_length=512: 512/22050 = 23.2 ms per frame

Sample rate = 44100 Hz (native):
- Time resolution: 1/44100 = 0.0227 ms per sample
- With hop_length=512: 512/44100 = 11.6 ms per frame (twice the precision)

Recommendation:
- Use 22050 Hz (default) for standard research practices and faster processing
- Use 44100 Hz for higher precision onset detection (pass sr=44100 to detect_onsets)
- IMPORTANT: Ensure consistency across all analysis steps - use the same sample
  rate throughout your pipeline

Environment: AEinBOX_13_3
Dependencies: librosa, numpy, pandas
"""

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# Import config
import importlib.util
_config_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("config_module", _config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config


def detect_onsets(
    audio_path: str,
    hop_length: int = 512,
    backtrack: bool = False,
    delta: float = 0.12,
    refine_onsets: bool = False,
    min_interval_s: float = 0.15,
    normalize: bool = True,
    sr: Optional[int] = None
) -> np.ndarray:
    """
    Detect onsets in an audio file using librosa.

    Parameters
    ----------
    audio_path : str
        Path to audio file (WAV, MP3, etc.)
    hop_length : int, optional
        STFT hop length in samples (default: 512)
        Lower values = higher time resolution but slower
    backtrack : bool, optional
        Backtrack detected onsets to previous local minimum (default: False)
    delta : float, optional
        Onset detection threshold (default: 0.12)
        Lower = more sensitive, higher = fewer onsets
    refine_onsets : bool, optional
        Apply peak refinement and duplicate filtering (default: False)
    min_interval_s : float, optional
        Minimum interval between onsets in seconds (default: 0.15)
        Used to filter duplicate/double onsets
    normalize : bool, optional
        Apply peak normalization to audio (default: True)
    sr : int, optional
        Target sample rate (default: librosa default 22050 Hz)

    Returns
    -------
    np.ndarray
        Onset times in seconds

    Examples
    --------
    >>> onsets = detect_onsets('track.wav')
    >>> print(f"Found {len(onsets)} onsets")
    Found 342 onsets

    >>> # More sensitive detection
    >>> onsets = detect_onsets('track.wav', delta=0.05, min_interval_s=0.02)
    >>> print(f"Found {len(onsets)} onsets")
    Found 512 onsets
    """
    # Debug: print parameters
    print(f"  [DEBUG] onset_detection parameters:")
    print(f"    hop_length={hop_length}, backtrack={backtrack}, delta={delta}")
    print(f"    refine_onsets={refine_onsets}, min_interval_s={min_interval_s}, normalize={normalize}, sr={sr}")

    # Load audio
    y, sr_loaded = librosa.load(audio_path, sr=sr)
    print(f"    Loaded with sr={sr_loaded}")

    # Peak normalization
    if normalize:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr_loaded, hop_length=hop_length)

    # Detect onset frames
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr_loaded,
        hop_length=hop_length,
        backtrack=backtrack,
        delta=delta
    )

    # Convert frames to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr_loaded, hop_length=hop_length)

    # Optional refinement
    if refine_onsets:
        onset_times = refine_and_filter_onsets(
            y, sr_loaded, onset_times, min_interval=min_interval_s
        )

    return onset_times


def refine_and_filter_onsets(
    y: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    min_interval: float = 0.15
) -> np.ndarray:
    """
    Refine onset times and filter duplicates.

    This function performs two operations:
    1. Refines timing by searching for the local peak in a small window around
       each detected onset
    2. Filters out duplicate onsets that are too close together

    Parameters
    ----------
    y : np.ndarray
        Audio waveform
    sr : int
        Sample rate
    onset_times : np.ndarray
        Initial onset times in seconds
    min_interval : float, optional
        Minimum interval between onsets in seconds (default: 0.15)

    Returns
    -------
    np.ndarray
        Refined and filtered onset times

    Examples
    --------
    >>> y, sr = librosa.load('track.wav')
    >>> initial_onsets = np.array([1.0, 1.05, 2.0, 3.0])  # 1.0 and 1.05 are duplicates
    >>> refined = refine_and_filter_onsets(y, sr, initial_onsets, min_interval=0.15)
    >>> len(refined)
    3  # Duplicate removed
    """
    # 1) Refine timing by local-peak search in a small window
    refined = []
    for t in onset_times:
        win_start = max(0.0, t - 0.02)
        win_end = t + 0.05
        i0 = int(win_start * sr)
        i1 = int(win_end * sr)
        segment = y[i0:i1]

        if segment.size > 0:
            peak_rel = np.argmax(np.abs(segment))
            refined_t = win_start + (peak_rel / sr)
            refined.append(refined_t)

    refined = np.asarray(refined)

    # 2) Enforce minimum inter-onset interval
    if refined.size == 0:
        return refined

    refined.sort()
    filtered = [refined[0]]
    for t in refined[1:]:
        if t - filtered[-1] >= min_interval:
            filtered.append(t)

    return np.asarray(filtered)


def detect_onsets_from_stem(
    stems_dir: str,
    stem_name: str = 'drums',
    **kwargs
) -> np.ndarray:
    """
    Detect onsets from a specific stem in a stem directory.

    Convenience function for detecting onsets from pre-separated stems,
    particularly useful for drum-focused onset detection.

    Parameters
    ----------
    stems_dir : str
        Directory containing separated stems
    stem_name : str, optional
        Name of stem file (default: 'drums')
        Expected format: {stem_name}.wav
    **kwargs
        Additional arguments passed to detect_onsets()

    Returns
    -------
    np.ndarray
        Onset times in seconds

    Examples
    --------
    >>> onsets = detect_onsets_from_stem('track/stems/', stem_name='drums')
    >>> print(f"Found {len(onsets)} drum onsets")
    """
    stem_path = Path(stems_dir) / f"{stem_name}.wav"

    if not stem_path.exists():
        raise FileNotFoundError(f"Stem file not found: {stem_path}")

    return detect_onsets(str(stem_path), **kwargs)


def save_onsets_csv(
    onset_times: np.ndarray,
    output_path: str
) -> Path:
    """
    Save onset times to CSV file.

    Parameters
    ----------
    onset_times : np.ndarray
        Onset times in seconds
    output_path : str
        Output CSV file path

    Returns
    -------
    Path
        Path to saved CSV file

    Examples
    --------
    >>> onsets = detect_onsets('track.wav')
    >>> save_onsets_csv(onsets, 'track_onsets.csv')
    PosixPath('track_onsets.csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({'onset_times': onset_times})
    df.to_csv(output_path, index=False)

    return output_path


def detect_and_save_onsets(
    audio_path: str,
    output_path: str,
    **kwargs
) -> Tuple[np.ndarray, Path]:
    """
    Detect onsets and save to CSV in one step.

    This is the main entry point for onset detection in the Loop Extractor pipeline.

    Parameters
    ----------
    audio_path : str
        Path to input audio file
    output_path : str
        Path for output CSV file
    **kwargs
        Additional arguments passed to detect_onsets()

    Returns
    -------
    tuple of (np.ndarray, Path)
        (onset_times, output_csv_path)

    Examples
    --------
    >>> onsets, csv_path = detect_and_save_onsets(
    ...     'track.wav',
    ...     'output/track_onsets.csv'
    ... )
    >>> print(f"Detected {len(onsets)} onsets, saved to {csv_path}")
    """
    # Detect onsets
    onset_times = detect_onsets(audio_path, **kwargs)

    # Save to CSV
    csv_path = save_onsets_csv(onset_times, output_path)

    return onset_times, csv_path


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def detect_onsets_default(audio_path: str) -> np.ndarray:
    """
    Detect onsets with default librosa settings.

    Good for general-purpose onset detection.

    Parameters
    ----------
    audio_path : str
        Path to audio file

    Returns
    -------
    np.ndarray
        Onset times in seconds
    """
    return detect_onsets(
        audio_path,
        hop_length=256,
        backtrack=False,
        delta=0.07,
        refine_onsets=False,
        normalize=True
    )


def detect_onsets_drums(audio_path: str) -> np.ndarray:
    """
    Detect onsets optimized for drum stems.

    Uses exact settings from original onset detection notebook:
    - hop_length=512: STFT hop length
    - backtrack=False: No backtracking to local minima
    - delta=0.12: Onset detection threshold
    - refine_onsets=False: No post-processing refinement
    - min_interval_s=0.15: 150ms minimum interval to filter duplicates
    - normalize=True: Peak normalization enabled

    Parameters
    ----------
    audio_path : str
        Path to audio file (preferably drum stem)

    Returns
    -------
    np.ndarray
        Onset times in seconds
    """
    return detect_onsets(
        audio_path,
        hop_length=512,
        backtrack=False,
        delta=0.12,
        refine_onsets=False,
        min_interval_s=0.15,
        normalize=True
    )


def detect_onsets_full_mix(audio_path: str) -> np.ndarray:
    """
    Detect onsets optimized for full mix audio.

    Uses exact settings from original onset detection notebook:
    - hop_length=512: STFT hop length
    - backtrack=False: No backtracking to local minima
    - delta=0.12: Onset detection threshold
    - refine_onsets=False: No post-processing refinement
    - min_interval_s=0.15: 150ms minimum interval
    - normalize=True: Peak normalization enabled

    Parameters
    ----------
    audio_path : str
        Path to full mix audio file

    Returns
    -------
    np.ndarray
        Onset times in seconds
    """
    return detect_onsets(
        audio_path,
        hop_length=512,
        backtrack=False,
        delta=0.12,
        refine_onsets=False,
        min_interval_s=0.15,
        normalize=True
    )


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_onset_detection():
    """Test onset detection with synthetic audio."""
    print("=" * 80)
    print("Testing Onset Detection Module")
    print("=" * 80)

    # Create synthetic audio with clear onsets
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create clicks at specific times
    click_times = [0.5, 1.0, 1.5, 2.0, 3.0]
    y = np.zeros_like(t)

    for click_time in click_times:
        idx = int(click_time * sr)
        # Add a short impulse
        y[idx:idx+100] = np.sin(2 * np.pi * 440 * t[idx:idx+100]) * np.exp(-np.linspace(0, 5, 100))

    # Save temporary test file
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, y, sr)

    print(f"\n[1] Testing onset detection on synthetic audio...")
    print(f"  Expected onsets at: {click_times}")

    try:
        detected = detect_onsets(
            temp_path,
            hop_length=512,
            delta=0.05,
            refine_onsets=False
        )

        print(f"  Detected onsets at: {detected.tolist()}")
        print(f"  Found {len(detected)} onsets (expected {len(click_times)})")

        # Check if we detected approximately the right number
        assert len(detected) >= len(click_times) - 1, "Should detect most onsets"
        print("  ✓ Onset detection works")

    finally:
        # Clean up
        Path(temp_path).unlink()

    print("\n" + "=" * 80)
    print("Onset detection test passed!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Onset detection using librosa')
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--preset', choices=['default', 'drums', 'full_mix'],
                       default='drums', help='Preset configuration')
    parser.add_argument('--hop-length', type=int, help='STFT hop length')
    parser.add_argument('--delta', type=float, help='Onset detection threshold')
    parser.add_argument('--min-interval', type=float, help='Minimum onset interval (s)')
    parser.add_argument('--test', action='store_true', help='Run tests')

    args = parser.parse_args()

    if args.test:
        test_onset_detection()
    else:
        # Detect based on preset or custom params
        kwargs = {}
        if args.hop_length:
            kwargs['hop_length'] = args.hop_length
        if args.delta:
            kwargs['delta'] = args.delta
        if args.min_interval:
            kwargs['min_interval_s'] = args.min_interval

        if args.preset == 'default' and not kwargs:
            onsets = detect_onsets_default(args.input)
        elif args.preset == 'drums' and not kwargs:
            onsets = detect_onsets_drums(args.input)
        elif args.preset == 'full_mix' and not kwargs:
            onsets = detect_onsets_full_mix(args.input)
        else:
            onsets = detect_onsets(args.input, **kwargs)

        # Save results
        csv_path = save_onsets_csv(onsets, args.output)
        print(f"Detected {len(onsets)} onsets")
        print(f"Saved to: {csv_path}")
