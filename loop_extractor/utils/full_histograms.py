#!/usr/bin/env python3
"""
Full Song Histograms - Create histograms from the entire song instead of just the snippet.

This module processes onset detection data from the full audio file and creates
IOI (inter-onset interval) histograms similar to beat_histograms.py but using
all onsets from the complete song.

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import librosa
import warnings


def load_full_song_onsets(audio_path: str, sr: int = 22050) -> Tuple[np.ndarray, float]:
    """
    Load audio file and detect onsets across the entire song.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    sr : int
        Sample rate (default: 22050)

    Returns
    -------
    onset_times : np.ndarray
        Onset times in seconds
    bpm : float
        Estimated tempo
    """
    print(f"    Loading full audio: {Path(audio_path).name}")

    # Suppress librosa warnings about deprecated scipy functions
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)

        # Detect onsets using onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Estimate tempo using onset strength envelope
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)

    print(f"    Detected {len(onset_times)} onsets across {len(y)/sr:.1f}s")
    print(f"    Estimated BPM: {tempo:.1f}")

    return onset_times, tempo


def calculate_ioi_from_onsets(onset_times: np.ndarray, bpm: float) -> pd.DataFrame:
    """
    Calculate inter-onset intervals from onset times.

    Parameters
    ----------
    onset_times : np.ndarray
        Onset times in seconds
    bpm : float
        Tempo in BPM for tick conversion

    Returns
    -------
    pd.DataFrame
        DataFrame with IOI data
    """
    # Calculate tick duration
    bar_duration_s = 60.0 / bpm * 4  # 4 beats per bar
    tick_duration_s = bar_duration_s / 16  # 16th note duration
    tick_duration_ms = tick_duration_s * 1000

    ioi_data = []

    # Calculate IOI between consecutive onsets
    for i in range(len(onset_times) - 1):
        time1 = onset_times[i]
        time2 = onset_times[i + 1]

        # IOI in seconds and milliseconds
        ioi_s = time2 - time1
        ioi_ms = ioi_s * 1000

        # IOI in ticks (16th notes)
        ioi_ticks = ioi_s / tick_duration_s

        # Categorize IOI
        ioi_category = categorize_ioi(ioi_ticks)

        ioi_data.append({
            'onset1_time': float(time1),
            'onset2_time': float(time2),
            'ioi_s': float(ioi_s),
            'ioi_ms': float(ioi_ms),
            'ioi_ticks': float(ioi_ticks),
            'ioi_category': ioi_category
        })

    return pd.DataFrame(ioi_data)


def categorize_ioi(ioi_ticks: float) -> str:
    """
    Categorize IOI based on tick duration.

    Parameters
    ----------
    ioi_ticks : float
        Inter-onset interval in ticks (16th notes)

    Returns
    -------
    str
        IOI category (4/4, 2/4, 1/4, 3/16, 1/8, 6/16, 1/16)
    """
    # Define categories in descending order of size
    categories = [
        (16, '4/4'),
        (8, '2/4'),
        (6, '6/16'),
        (4, '1/4'),
        (3, '3/16'),
        (2, '1/8'),
        (1, '1/16'),
    ]

    # Find closest category (use rounding)
    ioi_rounded = round(ioi_ticks)
    for ticks, category in categories:
        if ioi_rounded >= ticks:
            return category

    return '1/16'  # Default to smallest


def create_full_song_ioi_histogram(
    audio_path: str,
    track_id: str,
    output_dir: str,
    bpm: Optional[float] = None
) -> dict:
    """
    Create IOI histogram from the entire song.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots
    bpm : Optional[float]
        Known BPM (if None, will be estimated)

    Returns
    -------
    dict
        Dictionary with paths to saved files and statistics
    """
    print(f"\n  [Full Song IOI Histogram] Creating histogram from entire song...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load onsets from full song
    onset_times, estimated_bpm = load_full_song_onsets(audio_path)

    # Use provided BPM or estimated BPM
    if bpm is None:
        bpm = estimated_bpm
        print(f"    Using estimated BPM: {bpm:.1f}")
    else:
        print(f"    Using provided BPM: {bpm:.1f}")

    # Calculate IOI
    df_ioi = calculate_ioi_from_onsets(onset_times, bpm)

    if df_ioi.empty:
        print(f"    ⚠️  No IOI data found")
        return {}

    # Create histogram
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle(f'Full Song IOI Histogram — {track_id}', fontsize=14, fontweight='bold', y=0.98)

    # Filter IOI values for plotting: only <= 16 ticks (one bar)
    df_ioi_filtered = df_ioi[df_ioi['ioi_ticks'] <= 16].copy()
    ioi_values_ms = df_ioi_filtered['ioi_ms'].values

    # Create histogram with automatic binning
    counts, bins, patches = ax.hist(ioi_values_ms, bins=50, color='#3498DB', alpha=0.7,
                                     edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_xlabel('Inter-Onset Interval (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{len(df_ioi_filtered)}/{len(df_ioi)} intervals (≤16 ticks) from entire song',
                fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add vertical lines at common rhythmic intervals (in ms)
    # Calculate expected IOI values for common categories
    bar_duration_s = 60.0 / bpm * 4
    tick_duration_s = bar_duration_s / 16
    tick_duration_ms = tick_duration_s * 1000

    rhythmic_intervals = {
        '1/16': 1 * tick_duration_ms,
        '1/8': 2 * tick_duration_ms,
        '3/16': 3 * tick_duration_ms,
        '1/4': 4 * tick_duration_ms,
        '6/16': 6 * tick_duration_ms,
        '2/4': 8 * tick_duration_ms,
        '4/4': 16 * tick_duration_ms,
    }

    for label, value_ms in rhythmic_intervals.items():
        if ax.get_xlim()[0] <= value_ms <= ax.get_xlim()[1]:
            ax.axvline(x=value_ms, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5, label=label)

    # Add legend for rhythmic interval lines
    ax.legend(loc='upper right', fontsize=9, title='Rhythmic Intervals')

    plt.tight_layout()

    # Save plot as PDF
    output_pdf = output_path / f'{track_id}_full_song_ioi_histogram.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf.name}")

    # Save plot as PNG
    output_png = output_path / f'{track_id}_full_song_ioi_histogram.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    # Save CSV with IOI data
    output_csv = output_path / f'{track_id}_full_song_ioi_data.csv'
    df_ioi.to_csv(output_csv, index=False)
    print(f"    Saved: {output_csv.name}")

    output_files = {
        'full_song_ioi_histogram_pdf': str(output_pdf),
        'full_song_ioi_histogram_png': str(output_png),
        'full_song_ioi_data_csv': str(output_csv)
    }

    print(f"    ✓ Processed {len(df_ioi)} total inter-onset intervals from full song")

    return output_files


def create_full_song_beat_histogram(
    audio_path: str,
    track_id: str,
    output_dir: str,
    bpm: Optional[float] = None
) -> dict:
    """
    Create beat histogram from the entire song (similar to beat_histograms.py).

    Shows IOI distribution in ticks with logarithmic x-axis.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots
    bpm : Optional[float]
        Known BPM (if None, will be estimated)

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    print(f"\n  [Full Song Beat Histogram] Creating beat histogram from entire song...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load onsets from full song
    onset_times, estimated_bpm = load_full_song_onsets(audio_path)

    # Use provided BPM or estimated BPM
    if bpm is None:
        bpm = estimated_bpm
        print(f"    Using estimated BPM: {bpm:.1f}")
    else:
        print(f"    Using provided BPM: {bpm:.1f}")

    # Calculate IOI
    df_ioi = calculate_ioi_from_onsets(onset_times, bpm)

    if df_ioi.empty:
        print(f"    ⚠️  No IOI data found")
        return {}

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle(f'Full Song Beat Histogram — {track_id}', fontsize=14, fontweight='bold', y=0.98)

    # Filter IOI values for plotting: only <= 16 ticks (one bar)
    df_ioi_filtered = df_ioi[df_ioi['ioi_ticks'] <= 16].copy()
    ioi_values = df_ioi_filtered['ioi_ticks'].values

    # Create histogram with automatic binning
    counts, bins, patches = ax.hist(ioi_values, bins=50, color='#2ECC71', alpha=0.7,
                                   edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_xlabel('IOI (16th note ticks)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{len(df_ioi_filtered)}/{len(df_ioi)} intervals (≤16 ticks) from entire song',
                fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add vertical lines at common rhythmic intervals (in ticks)
    rhythmic_intervals = {
        '1/16': 1,
        '1/8': 2,
        '3/16': 3,
        '1/4': 4,
        '6/16': 6,
        '2/4': 8,
        '4/4': 16,
    }

    for label, value_ticks in rhythmic_intervals.items():
        if ax.get_xlim()[0] <= value_ticks <= ax.get_xlim()[1]:
            ax.axvline(x=value_ticks, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5)
            # Add text label above the line
            ax.text(value_ticks, ax.get_ylim()[1] * 0.95, label,
                   ha='center', va='top', fontsize=8, color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save plot as PDF
    output_pdf = output_path / f'{track_id}_full_song_beat_histogram.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf.name}")

    # Save plot as PNG
    output_png = output_path / f'{track_id}_full_song_beat_histogram.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    output_files = {
        'full_song_beat_histogram_pdf': str(output_pdf),
        'full_song_beat_histogram_png': str(output_png)
    }

    print(f"    ✓ Processed {len(df_ioi)} total inter-onset intervals from full song")

    return output_files
