#!/usr/bin/env python3
"""
Full Song Histograms - Create histograms from the entire song instead of just the snippet.

This module reads pre-detected onset data from the 4_onsets CSV file and creates
IOI (inter-onset interval) histograms using all onsets from the complete song.
Uses tempo from the corrected downbeats file (avg_kept_corrected).

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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
    onsets_file: str,
    corrected_downbeats_file: str,
    track_id: str,
    output_dir: str
) -> dict:
    """
    Create IOI histogram from the entire song using pre-detected onsets.

    Reads ALL onset times from the entire song directly from the 4_onsets CSV
    and calculates IOI as differences between consecutive onsets. Uses tempo
    from corrected downbeats file (avg_kept_corrected) for rhythmic interval
    reference lines.

    Parameters
    ----------
    onsets_file : str
        Path to the onsets CSV file (4_onsets/{track_id}_onsets.csv).
        The entire file is read - all onsets from the full song.
    corrected_downbeats_file : str
        Path to corrected downbeats file (3_corrected/{track_id}_downbeats_corrected.txt)
        Used to extract avg_kept_corrected tempo from comments
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots

    Returns
    -------
    dict
        Dictionary with paths to saved files and statistics
    """
    print(f"\n  [Full Song IOI Histogram] Creating histogram from entire song...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read onset times from CSV
    onsets_path = Path(onsets_file)
    if not onsets_path.exists():
        print(f"    ⚠️  Onsets file not found: {onsets_file}")
        return {}

    df_onsets = pd.read_csv(onsets_path)
    if df_onsets.empty or 'onset_times' not in df_onsets.columns:
        print(f"    ⚠️  No onset data found in {onsets_file}")
        return {}

    onset_times = df_onsets['onset_times'].values

    if len(onset_times) < 2:
        print(f"    ⚠️  Need at least 2 onsets to calculate IOI")
        return {}

    # Get tempo from corrected downbeats file comments
    bpm = 120.0  # Default
    corrected_path = Path(corrected_downbeats_file)
    if corrected_path.exists():
        with open(corrected_path, 'r') as f:
            for line in f:
                if line.startswith('# avg_kept_corrected='):
                    try:
                        bpm = float(line.split('=')[1].strip())
                    except ValueError:
                        pass
                    break

    print(f"    Using tempo from corrected downbeats: {bpm:.1f} BPM")

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
    ax.set_title(f'{len(df_ioi_filtered)}/{len(df_ioi)} intervals (≤16 ticks) from entire song — Tempo: {bpm:.1f} BPM',
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
