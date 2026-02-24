#!/usr/bin/env python3
"""
Anchored Statistics (Rhythm + Beat)

Calculate summary statistics from section-anchored data:
1. Rhythm Statistics (from 6.6): Position-based microtiming and pulse metrics
2. Beat Statistics (from 6.7): IOI-based microtiming and pulse metrics

Input:
    6.6_anchored_rhythm_histograms/{track_id}_anchored_rhythm_histograms.csv
    6.6_anchored_rhythm_histograms/{track_id}_filtered_anchored_groove_pulse_histograms.csv
    6.7_anchored_beat_histograms/{track_id}_anchored_beat_histograms.csv
    6.7_anchored_beat_histograms/{track_id}_groove_pulse_beat_histograms.csv

Output:
    6.8_anchored_statistics/{track_id}_anchored_rhythm_statistics.csv
    6.8_anchored_statistics/{track_id}_anchored_beat_statistics.csv

Rhythm Statistics Metrics (per section):
    - microtiming_degree: Mean of abs(median_tick_phase) - average deviation from grid
    - microtiming_complexity: Mean of iqr_16th - timing variability
    - pulse_strength: Mean onset_strength at beat positions (1, 5, 9, 13, ...)
    - groove_pulse_strength: Mean onset_strength_filtered where > 0

Beat Statistics Metrics (per section):
    - num_ioi_categories: Count of IOI categories passing threshold (≥10%)
    - total_ioi_count: Total IOI events from passing categories
    - ioi_microtiming_degree: Mean of abs(median_shift) for passing categories
    - ioi_microtiming_complexity: Mean of iqr_scaled for passing categories
    - groove_ioi_pulse_strength: Mean onset_strength from groove pulse beat histograms

Usage:
    python anchored_rhythm_statistics.py <track_root_folder> <track_id>

Example:
    python anchored_rhythm_statistics.py /path/to/449_Track_Name "449_Track_Name"
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_section_statistics(
    df_section: pd.DataFrame,
    df_groove_section: pd.DataFrame,
    section_id: str,
    pattern_length: int
) -> dict:
    """
    Calculate rhythm statistics for a single section.

    Parameters
    ----------
    df_section : pd.DataFrame
        Filtered dataframe for this section from anchored_rhythm_histograms.csv
    df_groove_section : pd.DataFrame
        Filtered dataframe for this section from groove_pulse_histograms.csv
    section_id : str
        Section identifier (e.g., 'SecNo1_verse_L2')
    pattern_length : int
        Pattern length (2 or 4 bars)

    Returns
    -------
    dict
        Dictionary with section info and 4 metrics:
        - Microtiming Degree
        - Microtiming Complexity
        - Pulse Strength
        - Groove Pulse Strength
    """
    # Get section metadata from first row
    sec_no = df_section['sec_no'].iloc[0]
    section_label = df_section['section_label'].iloc[0]
    num_repetitions = df_section['num_repetitions'].iloc[0]
    ratio_in_snippet = df_section['ratio_in_snippet'].iloc[0]
    mean_section_tempo = df_section['mean_section_tempo'].iloc[0] if 'mean_section_tempo' in df_section.columns else None

    # 1. Microtiming Degree: Mean of absolute median_tick_phase (non-NaN only)
    # median_tick_phase: 0 = on grid, +/- 0.5 = maximally off grid
    tick_phases = df_section['median_tick_phase'].dropna()

    if len(tick_phases) > 0:
        microtiming_degree = np.mean(np.abs(tick_phases))
    else:
        microtiming_degree = None

    # 2. Microtiming Complexity: Mean of iqr_16th (non-NaN)
    iqr_values = df_section['iqr_16th'].dropna()

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
    df_beats = df_section[df_section['position'].isin(beat_positions)]
    beat_strengths = df_beats['onset_strength'].dropna()

    if len(beat_strengths) > 0:
        pulse_strength = np.mean(beat_strengths)
    else:
        pulse_strength = None

    # 4. Groove Pulse Strength: Mean of onset_strength_filtered (only > 0, non-NaN)
    if df_groove_section is not None and not df_groove_section.empty:
        filtered_strengths = df_groove_section['onset_strength_filtered'].dropna()
        filtered_strengths_nonzero = filtered_strengths[filtered_strengths > 0]

        if len(filtered_strengths_nonzero) > 0:
            groove_pulse_strength = np.mean(filtered_strengths_nonzero)
        else:
            groove_pulse_strength = None
    else:
        groove_pulse_strength = None

    return {
        'section_id': section_id,
        'sec_no': sec_no,
        'section_label': section_label,
        'pattern_length': pattern_length,
        'num_repetitions': num_repetitions,
        'ratio_in_snippet': ratio_in_snippet,
        'mean_section_tempo': mean_section_tempo,
        'microtiming_degree': microtiming_degree,
        'microtiming_complexity': microtiming_complexity,
        'pulse_strength': pulse_strength,
        'groove_pulse_strength': groove_pulse_strength
    }


def calculate_beat_section_statistics(
    df_section: pd.DataFrame,
    df_groove_section: pd.DataFrame,
    section_id: str,
    pattern_length: int
) -> dict:
    """
    Calculate beat histogram statistics for a single section.

    Parameters
    ----------
    df_section : pd.DataFrame
        Filtered dataframe for this section from anchored_beat_histograms.csv
    df_groove_section : pd.DataFrame
        Filtered dataframe for this section from groove_pulse_beat_histograms.csv
    section_id : str
        Section identifier (e.g., 'SecNo1_verse_L2')
    pattern_length : int
        Pattern length (2 or 4 bars)

    Returns
    -------
    dict
        Dictionary with section info and 3 metrics:
        - IOI Microtiming Degree: Mean of absolute median_shift (passes_threshold only)
        - IOI Microtiming Complexity: Mean of iqr_scaled (passes_threshold only)
        - Groove IOI Pulse Strength: Mean of onset_strength from groove pulse data (passes_threshold only)
    """
    # Get section metadata from first row
    sec_no = df_section['section_no'].iloc[0]
    section_label = df_section['section_label'].iloc[0]
    num_repetitions = df_section['num_repetitions'].iloc[0] if 'num_repetitions' in df_section.columns else None
    ratio_in_snippet = df_section['ratio_in_snippet'].iloc[0] if 'ratio_in_snippet' in df_section.columns else None

    # Filter to categories that pass threshold
    df_passes = df_section[df_section['passes_threshold'] == True]

    # 1. IOI Microtiming Degree: Mean of absolute median_shift (passes_threshold only)
    if len(df_passes) > 0:
        median_shifts = df_passes['median_shift'].dropna()
        if len(median_shifts) > 0:
            ioi_microtiming_degree = np.mean(np.abs(median_shifts))
        else:
            ioi_microtiming_degree = None
    else:
        ioi_microtiming_degree = None

    # 2. IOI Microtiming Complexity: Mean of iqr_scaled (passes_threshold only)
    if len(df_passes) > 0:
        iqr_values = df_passes['iqr_scaled'].dropna()
        if len(iqr_values) > 0:
            ioi_microtiming_complexity = np.mean(iqr_values)
        else:
            ioi_microtiming_complexity = None
    else:
        ioi_microtiming_complexity = None

    # 3. Groove IOI Pulse Strength: Mean of onset_strength from groove pulse data (passes_threshold only)
    if df_groove_section is not None and not df_groove_section.empty:
        df_groove_passes = df_groove_section[df_groove_section['passes_threshold'] == True]
        if len(df_groove_passes) > 0:
            groove_strengths = df_groove_passes['onset_strength'].dropna()
            if len(groove_strengths) > 0:
                groove_ioi_pulse_strength = np.mean(groove_strengths)
            else:
                groove_ioi_pulse_strength = None
        else:
            groove_ioi_pulse_strength = None
    else:
        groove_ioi_pulse_strength = None

    # Count number of IOI categories that pass threshold
    num_ioi_categories = len(df_passes)
    total_ioi_count = df_passes['count'].sum() if len(df_passes) > 0 else 0

    return {
        'section_id': section_id,
        'sec_no': sec_no,
        'section_label': section_label,
        'pattern_length': pattern_length,
        'num_repetitions': num_repetitions,
        'ratio_in_snippet': ratio_in_snippet,
        'num_ioi_categories': num_ioi_categories,
        'total_ioi_count': total_ioi_count,
        'ioi_microtiming_degree': ioi_microtiming_degree,
        'ioi_microtiming_complexity': ioi_microtiming_complexity,
        'groove_ioi_pulse_strength': groove_ioi_pulse_strength
    }


def anchored_beat_statistics_for_track(track_root: Path, track_id: str):
    """
    Calculate and export anchored beat statistics for a single track.

    Creates CSV file in {track_root}/6.8_anchored_statistics/:
    - {track_id}_anchored_beat_statistics.csv

    Parameters
    ----------
    track_root : Path
        Root directory of the track
    track_id : str
        Track identifier
    """
    print(f"[Anchored Beat Statistics] Processing: {track_id}")

    # Check if required CSV files exist
    beat_hist_dir = track_root / '6.7_anchored_beat_histograms'
    beat_csv = beat_hist_dir / f'{track_id}_anchored_beat_histograms.csv'
    groove_beat_csv = beat_hist_dir / f'{track_id}_groove_pulse_beat_histograms.csv'

    if not beat_csv.exists():
        print(f"  Error: Missing file {beat_csv}")
        return

    # Load main beat histograms CSV
    df_beat = pd.read_csv(beat_csv)

    # Load groove pulse beat CSV if exists
    df_groove_beat = None
    if groove_beat_csv.exists():
        df_groove_beat = pd.read_csv(groove_beat_csv)

    # Get unique section_ids
    section_ids = df_beat['section_id'].unique()

    if len(section_ids) == 0:
        print(f"  Warning: No sections found")
        return

    # Calculate statistics for each section
    all_stats = []
    for section_id in section_ids:
        df_section = df_beat[df_beat['section_id'] == section_id]
        pattern_length = df_section['pattern_length'].iloc[0]

        # Get corresponding groove data
        df_groove_section = None
        if df_groove_beat is not None:
            df_groove_section = df_groove_beat[df_groove_beat['section_id'] == section_id]

        stats = calculate_beat_section_statistics(
            df_section,
            df_groove_section,
            section_id,
            pattern_length
        )
        all_stats.append(stats)

        ioi_mt_deg = f"{stats['ioi_microtiming_degree']:.4f}" if stats['ioi_microtiming_degree'] is not None else 'N/A'
        ioi_mt_cplx = f"{stats['ioi_microtiming_complexity']:.4f}" if stats['ioi_microtiming_complexity'] is not None else 'N/A'
        print(f"    {section_id}: IOI_MT_deg={ioi_mt_deg}, IOI_MT_cplx={ioi_mt_cplx}, cats={stats['num_ioi_categories']}")

    # Create output directory
    stats_dir = track_root / '6.8_anchored_statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated statistics CSV
    df_stats = pd.DataFrame(all_stats)
    output_csv = stats_dir / f'{track_id}_anchored_beat_statistics.csv'
    df_stats.to_csv(output_csv, index=False)
    print(f"    Saved: {output_csv.name}")


def anchored_statistics_for_track(track_root: Path, track_id: str):
    """
    Calculate and export anchored rhythm statistics for a single track.

    Creates CSV file in {track_root}/6.8_anchored_statistics/:
    - {track_id}_anchored_rhythm_statistics.csv

    Parameters
    ----------
    track_root : Path
        Root directory of the track
    track_id : str
        Track identifier
    """
    print(f"[Anchored Rhythm Statistics] Processing: {track_id}")

    # Check if required CSV files exist
    hist_dir = track_root / '6.6_anchored_rhythm_histograms'
    rhythm_csv = hist_dir / f'{track_id}_anchored_rhythm_histograms.csv'
    groove_csv = hist_dir / f'{track_id}_filtered_anchored_groove_pulse_histograms.csv'

    if not rhythm_csv.exists():
        print(f"  Error: Missing file {rhythm_csv}")
        return

    # Load main rhythm histograms CSV
    df_rhythm = pd.read_csv(rhythm_csv)

    # Load groove pulse CSV if exists
    df_groove = None
    if groove_csv.exists():
        df_groove = pd.read_csv(groove_csv)

    # Get unique section_ids
    section_ids = df_rhythm['section_id'].unique()

    if len(section_ids) == 0:
        print(f"  Warning: No sections found")
        return

    # Calculate statistics for each section
    all_stats = []
    for section_id in section_ids:
        df_section = df_rhythm[df_rhythm['section_id'] == section_id]
        pattern_length = df_section['pattern_length'].iloc[0]

        # Get corresponding groove data
        df_groove_section = None
        if df_groove is not None:
            df_groove_section = df_groove[df_groove['section_id'] == section_id]

        stats = calculate_section_statistics(
            df_section,
            df_groove_section,
            section_id,
            pattern_length
        )
        all_stats.append(stats)

        mt_deg = f"{stats['microtiming_degree']:.4f}" if stats['microtiming_degree'] is not None else 'N/A'
        mt_cplx = f"{stats['microtiming_complexity']:.4f}" if stats['microtiming_complexity'] is not None else 'N/A'
        print(f"    {section_id}: MT_deg={mt_deg}, MT_cplx={mt_cplx}")

    # Create output directory
    stats_dir = track_root / '6.8_anchored_statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated statistics CSV
    df_stats = pd.DataFrame(all_stats)
    output_csv = stats_dir / f'{track_id}_anchored_rhythm_statistics.csv'
    df_stats.to_csv(output_csv, index=False)
    print(f"    Saved: {output_csv.name}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python anchored_rhythm_statistics.py <track_root_folder> <track_id>')
        print('Example: python anchored_rhythm_statistics.py /path/to/track "track_id"')
        sys.exit(1)

    track_root = Path(sys.argv[1])
    track_id = sys.argv[2]

    if not track_root.exists():
        print(f'Error: Track root directory does not exist: {track_root}')
        sys.exit(1)

    # Calculate rhythm statistics from 6.6 data
    anchored_statistics_for_track(track_root, track_id)

    # Calculate beat statistics from 6.7 data
    anchored_beat_statistics_for_track(track_root, track_id)

    print("\nAnchored statistics calculation complete!")
