#!/usr/bin/env python3
"""
Aggregate Statistics for Rhythm Histograms

Calculate summary statistics from rhythm histogram CSVs for each track.
Creates aggregate metrics for pattern lengths L=1, L=2 and L=4.

Output files: {track_id}_rhythm_statistics_L1.csv
             {track_id}_rhythm_statistics_L2.csv
             {track_id}_rhythm_statistics_L4.csv

Usage:
    python aggregate_statistics_rhythm_hist.py <track_root_folder> <track_id>
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_rhythm_statistics(
    medians_iqr_csv: Path,
    groove_pulse_csv: Path,
    pattern_length: int
) -> dict:
    """
    Calculate aggregate rhythm statistics for a given pattern length.

    Parameters
    ----------
    medians_iqr_csv : Path
        Path to rhythm_histograms_with_medians_and_iqr.csv
    groove_pulse_csv : Path
        Path to groove_pulse_histograms_filtered.csv
    pattern_length : int
        Pattern length (2 or 4 bars)

    Returns
    -------
    dict
        Dictionary with 4 metrics:
        - Microtiming Degree
        - Microtiming Complexity
        - Pulse Strength
        - Groove Pulse Strength
    """

    # Load medians & IQR CSV
    df_medians = pd.read_csv(medians_iqr_csv)

    # Filter to FlexStart method with matching pattern length
    df_medians = df_medians[
        (df_medians['method'].str.contains('FlexStart', case=False, na=False)) &
        (df_medians['pattern_length'] == pattern_length)
    ]

    if df_medians.empty:
        print(f"    Warning: No FlexStart data found for L={pattern_length} in medians & IQR CSV")
        return {
            'Microtiming Degree': None,
            'Microtiming Complexity': None,
            'Pulse Strength': None,
            'Groove Pulse Strength': None
        }

    # 1. Microtiming Degree: Mean of absolute relative_median_phase (non-NaN only)
    # Include values with relative_median_phase=0 (on-grid positions)
    relative_phases = df_medians['relative_median_phase'].dropna()

    if len(relative_phases) > 0:
        microtiming_degree = np.mean(np.abs(relative_phases))
    else:
        microtiming_degree = None

    # 2. Microtiming Complexity: Mean of iqr_16th (non-NaN)
    iqr_values = df_medians['iqr_16th'].dropna()

    if len(iqr_values) > 0:
        microtiming_complexity = np.mean(iqr_values)
    else:
        microtiming_complexity = None

    # 3. Pulse Strength: Mean of onset_strength at beat positions (non-NaN)
    # Beat positions for pattern length
    if pattern_length == 2:
        # Positions 1, 5, 9, 13, 17, 21, 25, 29 (1-based)
        beat_positions = [1, 5, 9, 13, 17, 21, 25, 29]
    elif pattern_length == 4:
        # Positions 1, 5, 9, 13, ... for all 4 bars
        beat_positions = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]
    else:
        beat_positions = []

    # Filter to beat positions
    df_beats = df_medians[df_medians['position'].isin(beat_positions)]
    beat_strengths = df_beats['onset_strength'].dropna()

    if len(beat_strengths) > 0:
        beat_onset_strength = np.mean(beat_strengths)
    else:
        beat_onset_strength = None

    # 4. Groove Pulse Strength: Mean of onset_strength_filtered (only > 0, non-NaN)
    df_groove = pd.read_csv(groove_pulse_csv)

    # Filter to FlexStart method with matching pattern length
    df_groove = df_groove[
        (df_groove['method'].str.contains('FlexStart', case=False, na=False)) &
        (df_groove['pattern_length'] == pattern_length)
    ]

    if df_groove.empty:
        print(f"    Warning: No FlexStart data found for L={pattern_length} in groove pulse CSV")
        groove_pulse_strength = None
    else:
        # Get filtered onset strengths (only > 0)
        filtered_strengths = df_groove['onset_strength_filtered'].dropna()
        filtered_strengths_nonzero = filtered_strengths[filtered_strengths > 0]

        if len(filtered_strengths_nonzero) > 0:
            groove_pulse_strength = np.mean(filtered_strengths_nonzero)
        else:
            groove_pulse_strength = None

    return {
        'Microtiming Degree': microtiming_degree,
        'Microtiming Complexity': microtiming_complexity,
        'Pulse Strength': beat_onset_strength,
        'Groove Pulse Strength': groove_pulse_strength
    }


def aggregate_statistics_for_track(track_root: Path, track_id: str):
    """
    Calculate and export aggregate rhythm statistics for a single track.

    Creates two CSV files in {track_root}/5.6_statistics/:
    - {track_id}_rhythm_statistics_L2.csv
    - {track_id}_rhythm_statistics_L4.csv

    Parameters
    ----------
    track_root : Path
        Root directory of the track (contains 5_grid, 7_plots, etc.)
    track_id : str
        Track identifier
    """
    print(f"\nProcessing track: {track_id}")

    # Check if required CSV files exist
    plots_dir = track_root / '5.5_rhythm'
    medians_iqr_csv = plots_dir / f'{track_id}_rhythm_histograms_with_medians_and_iqr.csv'
    groove_pulse_csv = plots_dir / f'{track_id}_groove_pulse_histograms_filtered.csv'

    if not medians_iqr_csv.exists():
        print(f"  Error: Missing file {medians_iqr_csv}")
        return

    if not groove_pulse_csv.exists():
        print(f"  Error: Missing file {groove_pulse_csv}")
        return

    # Create output directory
    stats_dir = track_root / '5.6_statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Calculate statistics for L=1, L=2 and L=4
    for pattern_length in [1, 2, 4]:
        print(f"  Calculating statistics for L={pattern_length}...")

        stats = calculate_rhythm_statistics(
            medians_iqr_csv,
            groove_pulse_csv,
            pattern_length
        )

        # Create DataFrame
        df_stats = pd.DataFrame([
            {'Metric': 'Microtiming Degree', 'Value': stats['Microtiming Degree']},
            {'Metric': 'Microtiming Complexity', 'Value': stats['Microtiming Complexity']},
            {'Metric': 'Pulse Strength', 'Value': stats['Pulse Strength']},
            {'Metric': 'Groove Pulse Strength', 'Value': stats['Groove Pulse Strength']}
        ])

        # Save CSV
        output_csv = stats_dir / f'{track_id}_rhythm_statistics_L{pattern_length}.csv'
        df_stats.to_csv(output_csv, index=False)
        print(f"    ✓ Saved: {output_csv}")

        # Print statistics
        print(f"    Statistics for L={pattern_length}:")
        for metric, value in stats.items():
            if value is not None:
                print(f"      {metric}: {value:.6f}")
            else:
                print(f"      {metric}: None (no valid data)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python aggregate_statistics_rhythm_hist.py <track_root_folder> <track_id>')
        print('Example: python aggregate_statistics_rhythm_hist.py /path/to/401_1-800-273-8255 "401_1-800-273-8255 - LogicAlessia CaraKhalid"')
        sys.exit(1)

    track_root = Path(sys.argv[1])
    track_id = sys.argv[2]

    if not track_root.exists():
        print(f'Error: Track root directory does not exist: {track_root}')
        sys.exit(1)

    aggregate_statistics_for_track(track_root, track_id)

    print("\n✓ Aggregate statistics calculation complete!")
