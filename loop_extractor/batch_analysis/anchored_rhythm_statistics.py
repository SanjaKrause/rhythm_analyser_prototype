#!/usr/bin/env python3
"""
Anchored Rhythm Statistics

Calculate summary statistics from section-anchored rhythm histogram CSVs.
Creates aggregate metrics per section for pattern lengths L=2 and L=4.

Input: 6.6_anchored_rhythm_histograms/{track_id}_anchored_rhythm_histograms.csv
       6.6_anchored_rhythm_histograms/{track_id}_filtered_anchored_groove_pulse_histograms.csv

Output: 6.8_anchored_statistics/{track_id}_anchored_rhythm_statistics.csv

Usage:
    python anchored_rhythm_statistics.py <track_root_folder> <track_id>
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

    anchored_statistics_for_track(track_root, track_id)

    print("\nAnchored rhythm statistics calculation complete!")
