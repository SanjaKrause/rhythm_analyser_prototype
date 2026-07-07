"""
Onset detection using madmom CNNOnsetProcessor.

This module provides CNN-based onset detection using madmom's pre-trained models.
It's designed to be called as a standalone script from the main pipeline via subprocess.

Madmom CNNOnsetProcessor Architecture
--------------------------------------
The madmom onset detection uses a two-stage approach:
1. CNNOnsetProcessor: Generates an onset activation function (continuous signal 0-1)
   - Uses pre-trained convolutional neural network
   - Works on spectrograms computed from audio
   - Outputs probability of onset at each time frame

2. OnsetPeakPickingProcessor: Finds peaks in the activation function
   - Applies threshold to filter weak activations
   - Finds local maxima to determine precise onset times
   - Filters duplicates based on minimum time intervals

Key Parameters
--------------
threshold : float (default: 0.5)
    Minimum activation strength to consider as onset (range: 0.0-1.0)
    Lower = more sensitive (more onsets detected)
    Higher = less sensitive (fewer, stronger onsets only)

    Typical values:
    - 0.3: Very sensitive, detects subtle onsets (may include noise)
    - 0.5: Balanced (default, good for most cases)
    - 0.7: Conservative, only strong/clear onsets

fps : int (default: 100)
    Frames per second for onset detection
    Controls time resolution of onset activation function
    Higher = more precise timing but slower computation

smooth : float (optional)
    Smoothing window size in seconds
    Reduces noise in activation function
    Default: None (no smoothing)

pre_max : float (default: 0.03)
    Pre-maximum window size in seconds
    Used for finding local maxima
    Onset must be highest point in [t-pre_max, t+post_max]

post_max : float (default: 0.03)
    Post-maximum window size in seconds
    Used for finding local maxima

pre_avg : float (optional)
    Pre-average window size in seconds
    Used for adaptive thresholding
    Default: None

post_avg : float (optional)
    Post-average window size in seconds
    Used for adaptive thresholding
    Default: None

combine : float (default: 0.03)
    Minimum time between consecutive onsets in seconds
    Filters out duplicate detections
    Default: 30ms (prevents detection of duplicate onsets < 30ms apart)

Sensitivity Comparison
----------------------
On test track (drums.wav):
- Librosa (default):     32 onsets (spectral flux-based)
- Madmom (threshold=0.3): 90 onsets (very sensitive)
- Madmom (threshold=0.5): 68 onsets (default, more sensitive than librosa)
- Madmom (threshold=0.7): 46 onsets (closer to librosa)

Environment: new_beatnet_env (same as beat detection)
Dependencies: madmom, numpy, pandas
"""

import sys
import argparse
import numpy as np
import csv
from pathlib import Path


def detect_onsets_madmom(
    audio_path: str,
    threshold: float = 0.5,
    fps: int = 100,
    smooth: float = None,
    pre_max: float = 0.03,
    post_max: float = 0.03,
    pre_avg: float = None,
    post_avg: float = None,
    combine: float = 0.03
) -> np.ndarray:
    """
    Detect onsets using madmom CNNOnsetProcessor.

    Parameters
    ----------
    audio_path : str
        Path to audio file (WAV, MP3, etc.)
    threshold : float, optional
        Minimum activation strength (default: 0.5, range: 0.0-1.0)
        Lower = more sensitive, higher = less sensitive
    fps : int, optional
        Frames per second (default: 100)
    smooth : float, optional
        Smoothing window size in seconds (default: None)
    pre_max : float, optional
        Pre-maximum window in seconds (default: 0.03)
    post_max : float, optional
        Post-maximum window in seconds (default: 0.03)
    pre_avg : float, optional
        Pre-average window in seconds (default: None)
    post_avg : float, optional
        Post-average window in seconds (default: None)
    combine : float, optional
        Minimum time between onsets in seconds (default: 0.03)

    Returns
    -------
    np.ndarray
        Onset times in seconds

    Examples
    --------
    >>> onsets = detect_onsets_madmom('track.wav')
    >>> print(f"Found {len(onsets)} onsets")
    Found 68 onsets

    >>> # More sensitive detection
    >>> onsets = detect_onsets_madmom('track.wav', threshold=0.3)
    >>> print(f"Found {len(onsets)} onsets")
    Found 90 onsets

    >>> # More conservative detection
    >>> onsets = detect_onsets_madmom('track.wav', threshold=0.7)
    >>> print(f"Found {len(onsets)} onsets")
    Found 46 onsets
    """
    from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor

    # Debug: print parameters
    print(f"  [DEBUG] onset_detection_madmom parameters:")
    print(f"    threshold={threshold}, fps={fps}")
    print(f"    smooth={smooth}, pre_max={pre_max}, post_max={post_max}")
    print(f"    pre_avg={pre_avg}, post_avg={post_avg}, combine={combine}")

    # Stage 1: Generate onset activation function using CNN
    onset_proc = CNNOnsetProcessor()
    print(f"    Computing onset activation function...")
    activation = onset_proc(audio_path)
    print(f"    Activation shape: {activation.shape}")

    # Stage 2: Peak picking to find onset times
    # Only pass non-None parameters to avoid madmom TypeError
    peak_kwargs = {'fps': fps, 'threshold': threshold}

    if smooth is not None:
        peak_kwargs['smooth'] = smooth
    if pre_max is not None:
        peak_kwargs['pre_max'] = pre_max
    if post_max is not None:
        peak_kwargs['post_max'] = post_max
    if pre_avg is not None:
        peak_kwargs['pre_avg'] = pre_avg
    if post_avg is not None:
        peak_kwargs['post_avg'] = post_avg
    if combine is not None:
        peak_kwargs['combine'] = combine

    peak_proc = OnsetPeakPickingProcessor(**peak_kwargs)
    print(f"    Peak picking with threshold={threshold}...")
    onset_times = peak_proc(activation)

    print(f"    Detected {len(onset_times)} onsets")

    return onset_times


def save_onsets_csv(
    onset_times: np.ndarray,
    output_path: str
) -> Path:
    """
    Save onset times to CSV file.

    Output format matches librosa onset detection:
    - Single column 'onset_times'
    - Float values in seconds
    - One onset per row

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
    >>> onsets = detect_onsets_madmom('track.wav')
    >>> save_onsets_csv(onsets, 'track_onsets.csv')
    PosixPath('track_onsets.csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV without pandas dependency
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['onset_times'])  # Header
        for onset_time in onset_times:
            writer.writerow([onset_time])

    print(f"    Saved {len(onset_times)} onsets to: {output_path}")

    return output_path


def main():
    """Command-line interface for madmom onset detection."""
    parser = argparse.ArgumentParser(
        description='Onset detection using madmom CNNOnsetProcessor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default threshold
  python onset_detection_madmom.py --audio track.wav --output onsets.csv

  # More sensitive detection
  python onset_detection_madmom.py --audio track.wav --output onsets.csv --threshold 0.3

  # Less sensitive detection
  python onset_detection_madmom.py --audio track.wav --output onsets.csv --threshold 0.7

Threshold Guidelines:
  0.3 = Very sensitive (90+ onsets, may include noise)
  0.5 = Balanced (default, good for most cases)
  0.7 = Conservative (fewer, stronger onsets only)
        """
    )

    parser.add_argument(
        '--audio',
        required=True,
        help='Input audio file path'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Onset detection threshold (default: 0.5, range: 0.0-1.0). '
             'Lower = more sensitive, higher = less sensitive.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=100,
        help='Frames per second (default: 100)'
    )
    parser.add_argument(
        '--smooth',
        type=float,
        default=None,
        help='Smoothing window size in seconds (default: None)'
    )
    parser.add_argument(
        '--pre-max',
        type=float,
        default=0.03,
        help='Pre-maximum window in seconds (default: 0.03)'
    )
    parser.add_argument(
        '--post-max',
        type=float,
        default=0.03,
        help='Post-maximum window in seconds (default: 0.03)'
    )
    parser.add_argument(
        '--combine',
        type=float,
        default=0.03,
        help='Minimum time between onsets in seconds (default: 0.03)'
    )

    args = parser.parse_args()

    print(f"\n[Madmom Onset Detection]")
    print(f"  Input:  {args.audio}")
    print(f"  Output: {args.output}")
    print(f"  Threshold: {args.threshold}")

    try:
        # Detect onsets
        onset_times = detect_onsets_madmom(
            audio_path=args.audio,
            threshold=args.threshold,
            fps=args.fps,
            smooth=args.smooth,
            pre_max=args.pre_max,
            post_max=args.post_max,
            combine=args.combine
        )

        # Save to CSV
        csv_path = save_onsets_csv(onset_times, args.output)

        print(f"\n✓ Success: Detected {len(onset_times)} onsets")
        print(f"  Saved to: {csv_path}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
