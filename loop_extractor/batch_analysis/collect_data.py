#!/usr/bin/env python3
"""
Collect Data - Step 22 Batch Analysis

Aggregates all analysis data from rhythm histograms, groove pulse, rhythm patterns,
beat histograms, groove pulse beat, beat patterns, and statistics into structured CSVs.

Creates 4 output CSVs based on pattern length (L2, L4) and ratio thresholds (50%, 70%):
    - L2_ratio50.csv - L=2 sections with ratio_in_snippet > 50%
    - L2_ratio70.csv - L=2 sections with ratio_in_snippet > 70%
    - L4_ratio50.csv - L=4 sections with ratio_in_snippet > 50%
    - L4_ratio70.csv - L=4 sections with ratio_in_snippet > 70%

Column Structure:
    Metadata: song_id, song_name, sec_no, section_label, num_repetitions, ratio_in_snippet, mean_section_tempo

    Rhythm Histogram (RH): L*16 positions (0-31 for L2, 0-63 for L4)
        RH_str_0 ... RH_str_N (onset strength)
        RH_med_0 ... RH_med_N (median tick phase)
        RH_iqr_0 ... RH_iqr_N (IQR * 1.5)

    Groove Pulse (GP): L*16 positions
        GP_str_0 ... GP_str_N, GP_med_0 ... GP_med_N, GP_iqr_0 ... GP_iqr_N

    Rhythm Pattern (RP): L*16 positions
        RP_str_0 ... RP_str_N, RP_med_0 ... RP_med_N, RP_iqr_0 ... RP_iqr_N

    Beat Histogram (BH): 7 IOI categories (1/16, 1/8, 3/16, 1/4, 3/8, 1/2, 3/4)
        BH_str_1/16 ... BH_str_3/4 (onset strength)
        BH_med_1/16 ... BH_med_3/4 (median shift)
        BH_iqr_1/16 ... BH_iqr_3/4 (IQR scaled)

    Groove Pulse Beat (GPB): 7 IOI categories
        GPB_str_1/16 ... GPB_str_3/4, GPB_med_1/16 ... GPB_med_3/4, GPB_iqr_1/16 ... GPB_iqr_3/4

    Beat Pattern (BP): 7 IOI categories
        BP_str_1/16 ... BP_str_3/4, BP_med_1/16 ... BP_med_3/4, BP_iqr_1/16 ... BP_iqr_3/4

    Rhythm Statistics (from 6.8):
        microtiming_degree, microtiming_complexity, pulse_strength, groove_pulse_strength

    Beat Statistics (from 6.8):
        ioi_microtiming_degree, ioi_microtiming_complexity, groove_ioi_pulse_strength, total_ioi_count

Input:
    6.6_anchored_rhythm_histograms/{track_id}_filtered_anchored_rhythm_histograms.csv
    6.6_anchored_rhythm_histograms/{track_id}_filtered_anchored_groove_pulse_histograms.csv
    6.6_anchored_rhythm_histograms/{track_id}_filtered_anchored_rhythm_patterns.csv
    6.7_anchored_beat_histograms/{track_id}_anchored_beat_histograms.csv
    6.7_anchored_beat_histograms/{track_id}_groove_pulse_beat_histograms.csv
    6.7_anchored_beat_histograms/{track_id}_anchored_beat_patterns.csv
    6.8_anchored_statistics/{track_id}_anchored_rhythm_statistics.csv
    6.8_anchored_statistics/{track_id}_anchored_beat_statistics.csv

Output (in 22_collected_data/):
    - L2_ratio50.csv
    - L2_ratio70.csv
    - L4_ratio50.csv
    - L4_ratio70.csv

Usage:
    python collect_data.py /path/to/batch/output

Example:
    python collect_data.py "/Volumes/PortableSSD/06_Testing/new test feb20 stricter onset window/newnewnew10"
"""

import csv
import sys
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# IOI categories in order
IOI_CATEGORIES = ['1/16', '1/8', '3/16', '1/4', '3/8', '1/2', '3/4']

# Ratio thresholds
RATIO_THRESHOLDS = [0.50, 0.70]

# Pattern lengths
PATTERN_LENGTHS = [2, 4]


def extract_song_id(track_name: str) -> str:
    """Extract numeric song ID from track folder name."""
    match = re.match(r'^(\d+)', track_name)
    return match.group(1) if match else track_name


def extract_song_name(track_name: str) -> str:
    """Extract song name from track folder name (everything after the ID and underscore)."""
    match = re.match(r'^\d+_(.+)$', track_name)
    return match.group(1) if match else track_name


def read_rhythm_histogram_data(csv_path: Path, pattern_length: int) -> Dict[str, Dict]:
    """
    Read filtered rhythm histogram data and organize by section.

    Returns dict: section_id -> {metadata, positions: {pos: {strength, median, iqr}}}
    """
    sections = {}
    n_positions = pattern_length * 16

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section_id = row.get('section_id', '')
                pl = int(row.get('pattern_length', 0))

                if pl != pattern_length:
                    continue

                if section_id not in sections:
                    sections[section_id] = {
                        'sec_no': int(row.get('sec_no', 0)),
                        'section_label': row.get('section_label', ''),
                        'num_repetitions': int(row.get('num_repetitions', 0)),
                        'ratio_in_snippet': float(row.get('ratio_in_snippet', 0)),
                        'mean_section_tempo': float(row.get('mean_section_tempo', 0)),
                        'positions': {}
                    }

                pos = int(row.get('position', 0)) - 1  # Convert 1-based to 0-based
                if 0 <= pos < n_positions:
                    strength = float(row.get('onset_strength', 0) or 0)
                    median = row.get('median_tick_phase', '')
                    iqr = row.get('iqr_16th', '')

                    sections[section_id]['positions'][pos] = {
                        'strength': strength,
                        'median': float(median) if median else 0.0,
                        'iqr': float(iqr) if iqr else 0.0
                    }
    except Exception as e:
        print(f"    Warning: Could not read {csv_path}: {e}")

    return sections


def read_groove_pulse_histogram_data(csv_path: Path, pattern_length: int) -> Dict[str, Dict]:
    """
    Read filtered groove pulse histogram data and organize by section.
    Uses onset_strength_filtered column.
    """
    sections = {}
    n_positions = pattern_length * 16

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section_id = row.get('section_id', '')
                pl = int(row.get('pattern_length', 0))

                if pl != pattern_length:
                    continue

                if section_id not in sections:
                    sections[section_id] = {
                        'positions': {}
                    }

                pos = int(row.get('position', 0)) - 1
                if 0 <= pos < n_positions:
                    strength = float(row.get('onset_strength_filtered', 0) or 0)
                    median = row.get('median_tick_phase', '')
                    iqr = row.get('iqr_16th', '')

                    sections[section_id]['positions'][pos] = {
                        'strength': strength,
                        'median': float(median) if median else 0.0,
                        'iqr': float(iqr) if iqr else 0.0
                    }
    except Exception as e:
        print(f"    Warning: Could not read {csv_path}: {e}")

    return sections


def read_rhythm_pattern_data(csv_path: Path, pattern_length: int) -> Dict[str, Dict]:
    """
    Read filtered rhythm pattern data and organize by section.
    Uses pattern_value column (0, 0.5, or 1.0 based on groove pulse threshold).
    """
    sections = {}
    n_positions = pattern_length * 16

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section_id = row.get('section_id', '')
                pl = int(row.get('pattern_length', 0))

                if pl != pattern_length:
                    continue

                if section_id not in sections:
                    sections[section_id] = {
                        'positions': {}
                    }

                pos = int(row.get('position', 0)) - 1
                if 0 <= pos < n_positions:
                    # pattern_value: 0, 0.5, or 1.0 based on groove pulse threshold
                    strength = float(row.get('pattern_value', 0) or 0)
                    median = row.get('median_tick_phase', '')
                    iqr = row.get('iqr_16th', '')

                    sections[section_id]['positions'][pos] = {
                        'strength': strength,
                        'median': float(median) if median else 0.0,
                        'iqr': float(iqr) if iqr else 0.0
                    }
    except Exception as e:
        print(f"    Warning: Could not read {csv_path}: {e}")

    return sections


def read_beat_histogram_data(csv_path: Path, pattern_length: int) -> Dict[str, Dict]:
    """
    Read beat histogram data and organize by section.
    Returns dict: section_id -> {categories: {cat: {strength, median, iqr}}}
    """
    sections = {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section_id = row.get('section_id', '')
                pl = int(row.get('pattern_length', 0))

                if pl != pattern_length:
                    continue

                if section_id not in sections:
                    sections[section_id] = {
                        'categories': {}
                    }

                cat = row.get('ioi_category', '')
                if cat in IOI_CATEGORIES:
                    strength = float(row.get('onset_strength', 0) or 0)
                    median = row.get('median_shift', '')
                    iqr = row.get('iqr_scaled', '')

                    sections[section_id]['categories'][cat] = {
                        'strength': strength,
                        'median': float(median) if median else 0.0,
                        'iqr': float(iqr) if iqr else 0.0
                    }
    except Exception as e:
        print(f"    Warning: Could not read {csv_path}: {e}")

    return sections


def read_groove_pulse_beat_histogram_data(csv_path: Path, pattern_length: int) -> Dict[str, Dict]:
    """Read groove pulse beat histogram data."""
    return read_beat_histogram_data(csv_path, pattern_length)


def read_beat_pattern_data(csv_path: Path, pattern_length: int) -> Dict[str, Dict]:
    """
    Read beat pattern data and organize by section.
    Uses pattern_level column (0, 0.5, or 1.0 based on threshold).
    """
    sections = {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section_id = row.get('section_id', '')
                pl = int(row.get('pattern_length', 0))

                if pl != pattern_length:
                    continue

                if section_id not in sections:
                    sections[section_id] = {
                        'categories': {}
                    }

                cat = row.get('ioi_category', '')
                if cat in IOI_CATEGORIES:
                    # pattern_level: 0, 0.5, or 1.0 based on threshold
                    strength = float(row.get('pattern_level', 0) or 0)
                    median = row.get('median_shift', '')
                    iqr = row.get('iqr_scaled', '')

                    sections[section_id]['categories'][cat] = {
                        'strength': strength,
                        'median': float(median) if median else 0.0,
                        'iqr': float(iqr) if iqr else 0.0
                    }
    except Exception as e:
        print(f"    Warning: Could not read {csv_path}: {e}")

    return sections


def read_rhythm_statistics(csv_path: Path, pattern_length: int) -> Dict[str, Dict]:
    """
    Read rhythm statistics from 6.8.
    Returns dict: section_id -> {microtiming_degree, microtiming_complexity, pulse_strength, groove_pulse_strength}
    """
    sections = {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section_id = row.get('section_id', '')
                pl = int(row.get('pattern_length', 0))

                if pl != pattern_length:
                    continue

                sections[section_id] = {
                    'microtiming_degree': float(row.get('microtiming_degree', 0) or 0),
                    'microtiming_complexity': float(row.get('microtiming_complexity', 0) or 0),
                    'pulse_strength': float(row.get('pulse_strength', 0) or 0),
                    'groove_pulse_strength': float(row.get('groove_pulse_strength', 0) or 0),
                }
    except Exception as e:
        print(f"    Warning: Could not read {csv_path}: {e}")

    return sections


def read_beat_statistics(csv_path: Path, pattern_length: int) -> Dict[str, Dict]:
    """
    Read beat statistics from 6.8.
    Returns dict: section_id -> {ioi_microtiming_degree, ioi_microtiming_complexity, groove_ioi_pulse_strength, total_ioi_count}
    """
    sections = {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section_id = row.get('section_id', '')
                pl = int(row.get('pattern_length', 0))

                if pl != pattern_length:
                    continue

                sections[section_id] = {
                    'ioi_microtiming_degree': float(row.get('ioi_microtiming_degree', 0) or 0),
                    'ioi_microtiming_complexity': float(row.get('ioi_microtiming_complexity', 0) or 0),
                    'groove_ioi_pulse_strength': float(row.get('groove_ioi_pulse_strength', 0) or 0),
                    'total_ioi_count': int(row.get('total_ioi_count', 0) or 0),
                }
    except Exception as e:
        print(f"    Warning: Could not read {csv_path}: {e}")

    return sections


def build_column_headers(pattern_length: int) -> List[str]:
    """Build the full list of column headers for the output CSV."""
    n_positions = pattern_length * 16
    headers = []

    # Metadata columns
    headers.extend([
        'song_id', 'song_name', 'sec_no', 'section_label',
        'num_repetitions', 'ratio_in_snippet', 'mean_section_tempo'
    ])

    # Rhythm Histogram (RH) - L*16 positions
    for pos in range(n_positions):
        headers.append(f'RH_str_{pos}')
    for pos in range(n_positions):
        headers.append(f'RH_med_{pos}')
    for pos in range(n_positions):
        headers.append(f'RH_iqr_{pos}')

    # Groove Pulse (GP) - L*16 positions
    for pos in range(n_positions):
        headers.append(f'GP_str_{pos}')
    for pos in range(n_positions):
        headers.append(f'GP_med_{pos}')
    for pos in range(n_positions):
        headers.append(f'GP_iqr_{pos}')

    # Rhythm Pattern (RP) - L*16 positions
    for pos in range(n_positions):
        headers.append(f'RP_str_{pos}')
    for pos in range(n_positions):
        headers.append(f'RP_med_{pos}')
    for pos in range(n_positions):
        headers.append(f'RP_iqr_{pos}')

    # Beat Histogram (BH) - 7 IOI categories
    for cat in IOI_CATEGORIES:
        headers.append(f'BH_str_{cat}')
    for cat in IOI_CATEGORIES:
        headers.append(f'BH_med_{cat}')
    for cat in IOI_CATEGORIES:
        headers.append(f'BH_iqr_{cat}')

    # Groove Pulse Beat (GPB) - 7 IOI categories
    for cat in IOI_CATEGORIES:
        headers.append(f'GPB_str_{cat}')
    for cat in IOI_CATEGORIES:
        headers.append(f'GPB_med_{cat}')
    for cat in IOI_CATEGORIES:
        headers.append(f'GPB_iqr_{cat}')

    # Beat Pattern (BP) - 7 IOI categories
    for cat in IOI_CATEGORIES:
        headers.append(f'BP_str_{cat}')
    for cat in IOI_CATEGORIES:
        headers.append(f'BP_med_{cat}')
    for cat in IOI_CATEGORIES:
        headers.append(f'BP_iqr_{cat}')

    # Rhythm Statistics
    headers.extend([
        'microtiming_degree', 'microtiming_complexity',
        'pulse_strength', 'groove_pulse_strength'
    ])

    # Beat Statistics
    headers.extend([
        'ioi_microtiming_degree', 'ioi_microtiming_complexity',
        'groove_ioi_pulse_strength', 'total_ioi_count'
    ])

    return headers


def collect_section_data(
    track_dir: Path,
    track_name: str,
    pattern_length: int,
    ratio_threshold: float
) -> List[Dict]:
    """
    Collect all data for sections matching pattern_length and ratio_threshold.

    Returns list of row dicts ready for CSV output.
    """
    rows = []
    n_positions = pattern_length * 16

    # File paths
    rh_file = track_dir / '6.6_anchored_rhythm_histograms' / f'{track_name}_filtered_anchored_rhythm_histograms.csv'
    gp_file = track_dir / '6.6_anchored_rhythm_histograms' / f'{track_name}_filtered_anchored_groove_pulse_histograms.csv'
    rp_file = track_dir / '6.6_anchored_rhythm_histograms' / f'{track_name}_filtered_anchored_rhythm_patterns.csv'
    bh_file = track_dir / '6.7_anchored_beat_histograms' / f'{track_name}_anchored_beat_histograms.csv'
    gpb_file = track_dir / '6.7_anchored_beat_histograms' / f'{track_name}_groove_pulse_beat_histograms.csv'
    bp_file = track_dir / '6.7_anchored_beat_histograms' / f'{track_name}_anchored_beat_patterns.csv'
    rs_file = track_dir / '6.8_anchored_statistics' / f'{track_name}_anchored_rhythm_statistics.csv'
    bs_file = track_dir / '6.8_anchored_statistics' / f'{track_name}_anchored_beat_statistics.csv'

    # Read all data sources
    rh_data = read_rhythm_histogram_data(rh_file, pattern_length) if rh_file.exists() else {}
    gp_data = read_groove_pulse_histogram_data(gp_file, pattern_length) if gp_file.exists() else {}
    rp_data = read_rhythm_pattern_data(rp_file, pattern_length) if rp_file.exists() else {}
    bh_data = read_beat_histogram_data(bh_file, pattern_length) if bh_file.exists() else {}
    gpb_data = read_groove_pulse_beat_histogram_data(gpb_file, pattern_length) if gpb_file.exists() else {}
    bp_data = read_beat_pattern_data(bp_file, pattern_length) if bp_file.exists() else {}
    rs_data = read_rhythm_statistics(rs_file, pattern_length) if rs_file.exists() else {}
    bs_data = read_beat_statistics(bs_file, pattern_length) if bs_file.exists() else {}

    # Get all section IDs from rhythm histogram (primary source)
    if not rh_data:
        return rows

    song_id = extract_song_id(track_name)
    song_name = extract_song_name(track_name)

    for section_id, rh_section in rh_data.items():
        ratio = rh_section.get('ratio_in_snippet', 0)

        # Filter by ratio threshold
        if ratio <= ratio_threshold:
            continue

        row = {}

        # Metadata
        row['song_id'] = song_id
        row['song_name'] = song_name
        row['sec_no'] = rh_section.get('sec_no', 0)
        row['section_label'] = rh_section.get('section_label', '')
        row['num_repetitions'] = rh_section.get('num_repetitions', 0)
        row['ratio_in_snippet'] = ratio
        row['mean_section_tempo'] = rh_section.get('mean_section_tempo', 0)

        # Rhythm Histogram (RH)
        rh_positions = rh_section.get('positions', {})
        for pos in range(n_positions):
            pos_data = rh_positions.get(pos, {'strength': 0.0, 'median': 0.0, 'iqr': 0.0})
            row[f'RH_str_{pos}'] = pos_data['strength']
            row[f'RH_med_{pos}'] = pos_data['median']
            row[f'RH_iqr_{pos}'] = pos_data['iqr']

        # Groove Pulse (GP)
        gp_section = gp_data.get(section_id, {})
        gp_positions = gp_section.get('positions', {})
        for pos in range(n_positions):
            pos_data = gp_positions.get(pos, {'strength': 0.0, 'median': 0.0, 'iqr': 0.0})
            row[f'GP_str_{pos}'] = pos_data['strength']
            row[f'GP_med_{pos}'] = pos_data['median']
            row[f'GP_iqr_{pos}'] = pos_data['iqr']

        # Rhythm Pattern (RP)
        rp_section = rp_data.get(section_id, {})
        rp_positions = rp_section.get('positions', {})
        for pos in range(n_positions):
            pos_data = rp_positions.get(pos, {'strength': 0.0, 'median': 0.0, 'iqr': 0.0})
            row[f'RP_str_{pos}'] = pos_data['strength']
            row[f'RP_med_{pos}'] = pos_data['median']
            row[f'RP_iqr_{pos}'] = pos_data['iqr']

        # Beat Histogram (BH)
        bh_section = bh_data.get(section_id, {})
        bh_categories = bh_section.get('categories', {})
        for cat in IOI_CATEGORIES:
            cat_data = bh_categories.get(cat, {'strength': 0.0, 'median': 0.0, 'iqr': 0.0})
            row[f'BH_str_{cat}'] = cat_data['strength']
            row[f'BH_med_{cat}'] = cat_data['median']
            row[f'BH_iqr_{cat}'] = cat_data['iqr']

        # Groove Pulse Beat (GPB)
        gpb_section = gpb_data.get(section_id, {})
        gpb_categories = gpb_section.get('categories', {})
        for cat in IOI_CATEGORIES:
            cat_data = gpb_categories.get(cat, {'strength': 0.0, 'median': 0.0, 'iqr': 0.0})
            row[f'GPB_str_{cat}'] = cat_data['strength']
            row[f'GPB_med_{cat}'] = cat_data['median']
            row[f'GPB_iqr_{cat}'] = cat_data['iqr']

        # Beat Pattern (BP)
        bp_section = bp_data.get(section_id, {})
        bp_categories = bp_section.get('categories', {})
        for cat in IOI_CATEGORIES:
            cat_data = bp_categories.get(cat, {'strength': 0.0, 'median': 0.0, 'iqr': 0.0})
            row[f'BP_str_{cat}'] = cat_data['strength']
            row[f'BP_med_{cat}'] = cat_data['median']
            row[f'BP_iqr_{cat}'] = cat_data['iqr']

        # Rhythm Statistics
        rs_section = rs_data.get(section_id, {})
        row['microtiming_degree'] = rs_section.get('microtiming_degree', 0.0)
        row['microtiming_complexity'] = rs_section.get('microtiming_complexity', 0.0)
        row['pulse_strength'] = rs_section.get('pulse_strength', 0.0)
        row['groove_pulse_strength'] = rs_section.get('groove_pulse_strength', 0.0)

        # Beat Statistics
        bs_section = bs_data.get(section_id, {})
        row['ioi_microtiming_degree'] = bs_section.get('ioi_microtiming_degree', 0.0)
        row['ioi_microtiming_complexity'] = bs_section.get('ioi_microtiming_complexity', 0.0)
        row['groove_ioi_pulse_strength'] = bs_section.get('groove_ioi_pulse_strength', 0.0)
        row['total_ioi_count'] = bs_section.get('total_ioi_count', 0)

        rows.append(row)

    return rows


def create_collected_data(output_dir: Path):
    """
    Create collected data CSVs from batch processing results.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    """
    print("\n" + "=" * 80)
    print("STEP 22: COLLECT DATA")
    print("=" * 80)

    # Find all track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis',
                                          'snippet_ratio_batch_analysis', '22_collected_data']
    ])

    if not track_dirs:
        print('No track directories found!')
        return

    print(f"Found {len(track_dirs)} track directories")

    # Create output directory
    collected_dir = output_dir / '22_collected_data'
    collected_dir.mkdir(parents=True, exist_ok=True)

    # Process each combination of pattern_length and ratio_threshold
    for pattern_length in PATTERN_LENGTHS:
        for ratio_threshold in RATIO_THRESHOLDS:
            print(f"\n--- L{pattern_length} ratio>{int(ratio_threshold*100)}% ---")

            all_rows = []

            for track_dir in track_dirs:
                track_name = track_dir.name
                rows = collect_section_data(track_dir, track_name, pattern_length, ratio_threshold)

                if rows:
                    print(f"  {track_name}: {len(rows)} sections")
                    all_rows.extend(rows)

            # Sort by song_id (numeric) then sec_no
            all_rows.sort(key=lambda x: (int(x['song_id']), x['sec_no']))

            # Write CSV
            output_file = collected_dir / f'L{pattern_length}_ratio{int(ratio_threshold*100)}.csv'
            headers = build_column_headers(pattern_length)

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_rows)

            print(f"\n  Total sections: {len(all_rows)}")
            print(f"  Saved: {output_file.name}")

    print("\n" + "=" * 80)
    print("✓ All collected data CSVs created successfully!")
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python collect_data.py /path/to/batch/output')
        print('Example: python collect_data.py "/Volumes/PortableSSD/06_Testing/new test feb20"')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f'Error: Directory does not exist: {output_dir}')
        sys.exit(1)

    create_collected_data(output_dir)
