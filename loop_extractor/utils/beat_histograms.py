#!/usr/bin/env python3
"""
Beat Histograms - Create beat-level visualizations from inter-onset interval data.

This module reads FlexStart filtered CSV files and calculates inter-onset intervals (IOI)
between consecutive onsets, categorizing them by duration (4/4, 2/4, 1/4, 3/16, 1/8, 6/16, 1/16).

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
import sys

# Add parent directory to path to import rhythm_histograms utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.rhythm_histograms import read_filtered_csv_metadata


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


def process_bar_based_ioi(
    grid_output_dir: str,
    base_name: str,
    bpm: float,
    snippet_start_time: float
) -> pd.DataFrame:
    """
    Process bar-based IOI from FlexStart filtered CSV files.

    Reads the 3 FlexStart filtered CSVs (L=4, L=2, L=1) and calculates
    inter-onset intervals between consecutive onsets within each pattern.

    Parameters
    ----------
    grid_output_dir : str
        Directory containing the FlexStart filtered CSV files
    base_name : str
        Base filename (without extension)
    bpm : float
        Tempo in BPM for time conversion
    snippet_start_time : float
        Start time of snippet in seconds (for absolute time calculation)

    Returns
    -------
    pd.DataFrame
        DataFrame with IOI data
    """
    grid_dir = Path(grid_output_dir)

    # Pattern lengths to process
    pattern_lengths = [4, 2, 1]

    all_ioi_data = []

    # Calculate tick duration in seconds
    bar_duration = 60.0 / bpm * 4  # 4 beats per bar at BPM
    tick_duration = bar_duration / 16  # 16th note duration

    for pattern_length in pattern_lengths:
        # Find filtered CSV file - use base_name directly, it already has the track name
        filtered_csv_name = f'{base_name}_comprehensive_phases_{pattern_length}bar_flexStart_filtered.csv'
        filtered_csv_path = grid_dir / filtered_csv_name

        if not filtered_csv_path.exists():
            print(f"    Warning: FlexStart filtered CSV not found: {filtered_csv_name}")
            continue

        # Read CSV with metadata
        df, num_patterns_displayed, filtering_method = read_filtered_csv_metadata(str(filtered_csv_path))

        if df.empty:
            continue

        # Process each pattern
        for pattern_id in df['pattern_id'].unique():
            df_pattern = df[df['pattern_id'] == pattern_id].copy()

            # Sort by tick_16th to ensure correct ordering
            df_pattern = df_pattern.sort_values('tick_16th').reset_index(drop=True)

            # Calculate IOI between consecutive onsets
            for i in range(len(df_pattern) - 1):
                onset1 = df_pattern.iloc[i]
                onset2 = df_pattern.iloc[i + 1]

                # Get tick positions and phases
                tick1 = onset1['tick_16th']
                tick2 = onset2['tick_16th']
                phase1 = onset1['phase']
                phase2 = onset2['phase']

                # Calculate tick delta
                tick_delta = tick2 - tick1

                # Calculate phase difference
                phase_diff = phase2 - phase1

                # Calculate exact IOI in ticks
                ioi_exact_ticks = tick_delta + phase_diff

                # Calculate times in seconds (relative to snippet start)
                time1_rel = (tick1 + phase1) * tick_duration
                time2_rel = (tick2 + phase2) * tick_duration

                # Absolute times
                time1_abs = snippet_start_time + time1_rel
                time2_abs = snippet_start_time + time2_rel

                # Categorize IOI
                ioi_category = categorize_ioi(ioi_exact_ticks)

                # Store data
                all_ioi_data.append({
                    'method': 'Bar-based',
                    'pattern_length': pattern_length,
                    'pattern_id': pattern_id,
                    'onset1_tick': int(tick1),
                    'onset1_phase': float(phase1),
                    'onset1_time_abs': float(time1_abs),
                    'onset1_time_rel': float(time1_rel),
                    'onset2_tick': int(tick2),
                    'onset2_phase': float(phase2),
                    'onset2_time_abs': float(time2_abs),
                    'onset2_time_rel': float(time2_rel),
                    'tick_delta': int(tick_delta),
                    'phase_diff': float(phase_diff),
                    'ioi_exact_ticks': float(ioi_exact_ticks),
                    'ioi_category': ioi_category
                })

    if not all_ioi_data:
        return pd.DataFrame()

    # Create DataFrame
    df_ioi = pd.DataFrame(all_ioi_data)

    # Define category order for sorting
    category_order = ['4/4', '2/4', '1/4', '3/16', '1/8', '6/16', '1/16']
    df_ioi['ioi_category'] = pd.Categorical(df_ioi['ioi_category'], categories=category_order, ordered=True)

    # Sort by category (descending length), then by time (ascending)
    df_ioi = df_ioi.sort_values(['ioi_category', 'onset1_time_rel']).reset_index(drop=True)

    return df_ioi


def create_beat_histograms(
    grid_output_dir: str,
    base_name: str,
    track_id: str,
    output_dir: str,
    bpm: float,
    snippet_start_time: float
) -> dict:
    """
    Create beat-level histograms from inter-onset interval data.

    Parameters
    ----------
    grid_output_dir : str
        Directory containing the FlexStart filtered CSV files
    base_name : str
        Base filename (without extension)
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots
    bpm : float
        Tempo in BPM for time conversion
    snippet_start_time : float
        Start time of snippet in seconds

    Returns
    -------
    dict
        Dictionary with paths to saved files and statistics
    """
    print(f"\n  [Beat Histograms] Creating beat histograms from inter-onset intervals...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process bar-based IOI
    df_ioi = process_bar_based_ioi(grid_output_dir, base_name, bpm, snippet_start_time)

    if df_ioi.empty:
        print(f"    ⚠️  No IOI data found")
        return {}

    # Save pre-beat histogram CSV
    output_csv = output_path / f'{track_id}_pre_beat_histogram_bar_based.csv'
    df_ioi.to_csv(output_csv, index=False)
    print(f"    Saved: {output_csv}")

    print(f"    ✓ Processed {len(df_ioi)} inter-onset intervals")

    return {
        'pre_beat_histogram_csv': str(output_csv)
    }
