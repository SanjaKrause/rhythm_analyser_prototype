#!/usr/bin/env python3
"""
Collect Data - Step 22 Batch Analysis

Aggregates all analysis data from rhythm histograms, groove pulse, rhythm patterns,
beat histograms, groove pulse beat, beat patterns, and statistics into structured CSVs.

Creates output CSVs based on pattern length (L1, L2, L4) and ratio thresholds:

Standard ratio thresholds (50%, 70%) - all sections meeting threshold:
    - L1_ratio50.csv, L1_ratio70.csv
    - L2_ratio50.csv, L2_ratio70.csv
    - L4_ratio50.csv, L4_ratio70.csv

Max-selection ratio thresholds (30%, 40%) - only highest-ratio section per song:
    - L1_ratio30_maxsel.csv, L1_ratio40_maxsel.csv
    - L2_ratio30_maxsel.csv, L2_ratio40_maxsel.csv
    - L4_ratio30_maxsel.csv, L4_ratio40_maxsel.csv

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

Output (in collected_data/):
    - L1_ratio50.csv
    - L1_ratio70.csv
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

# For bass pitch calculations
import numpy as np

# Ratio thresholds
RATIO_THRESHOLDS = [0.50, 0.70]

# Additional ratio thresholds with max-ratio selection per song
# For these thresholds: if multiple sections per song meet criteria, keep only the one with highest ratio
RATIO_THRESHOLDS_MAX_SELECTION = [0.30, 0.40]

# Pattern lengths
PATTERN_LENGTHS = [1, 2, 4]


def extract_song_id(track_name: str) -> str:
    """Extract numeric song ID from track folder name."""
    match = re.match(r'^(\d+)', track_name)
    return match.group(1) if match else track_name


def extract_song_name(track_name: str) -> str:
    """Extract song name from track folder name (everything after the ID and underscore)."""
    match = re.match(r'^\d+_(.+)$', track_name)
    return match.group(1) if match else track_name


def read_time_signature(track_dir: Path, track_name: str) -> int:
    """
    Read time_signature from the downbeats_corrected file.

    Looks for 3_corrected/{track_name}_downbeats_corrected.txt and reads
    the # time_signature=N line from the header.

    Returns time_signature as int (e.g., 4 for 4/4 time), or 0 if not found.
    """
    corrected_file = track_dir / '3_corrected' / f'{track_name}_downbeats_corrected.txt'

    if not corrected_file.exists():
        return 0

    try:
        with open(corrected_file, 'r') as f:
            for line in f:
                if line.startswith('# time_signature='):
                    value = line.split('=')[1].strip()
                    return int(value)
                # Stop after header comments
                if not line.startswith('#'):
                    break
    except Exception:
        pass

    return 0


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


def read_bass_pitch_data(filtered_patterns_dir: Path, pattern_length: int) -> Dict[str, Dict]:
    """
    Read bass pitch (f0_hz) data from filtered pattern files.

    Aggregates f0_hz values per tick position within each section, computing
    median f0 across all bars in the section.

    Parameters
    ----------
    filtered_patterns_dir : Path
        Path to 6.2_filtered_patterns/bass/ directory
    pattern_length : int
        Pattern length (1, 2, or 4 bars)

    Returns
    -------
    Dict[str, Dict]
        section_id -> {positions: {pos: median_f0}}
    """
    sections = {}
    n_positions = pattern_length * 16

    # Find all anchored CSV files for this stem
    if not filtered_patterns_dir.exists():
        return sections

    anchored_files = sorted([f for f in filtered_patterns_dir.glob('*_anchored.csv')
                             if not f.name.startswith('._')])

    for csv_file in anchored_files:
        try:
            # Parse section info from filename
            # Format: {track_id}_sec{N}_L{L}_{label}_{ratio}_anchored.csv
            fname = csv_file.stem.replace('_anchored', '')
            parts = fname.split('_')

            # Find section pattern
            # Note: filenames use "SecNo" (capital S), not "sec"
            sec_idx = None
            for i, p in enumerate(parts):
                if p.lower().startswith('sec'):
                    sec_idx = i
                    break

            if sec_idx is None:
                continue

            # Extract L value and section label from filename
            # Filename format: SecNo{N}_L{L}_{label}_{ratio}_anchored.csv
            l_val = None
            l_idx = None
            section_label = None
            sec_no_part = parts[sec_idx]  # e.g., "SecNo1"

            for i, p in enumerate(parts[sec_idx:]):
                if p.startswith('L') and len(p) > 1 and p[1:].isdigit():
                    l_val = int(p[1:])
                    l_idx = sec_idx + i
                    # Label is next part after L
                    if l_idx + 1 < len(parts):
                        section_label = parts[l_idx + 1]
                    break

            if l_val != pattern_length:
                continue

            # Build section_id to match rhythm histogram format
            # Rhythm histogram uses: SecNo{N}_{label}_L{L}
            # NOT the filename format: SecNo{N}_L{L}_{label}_{ratio}
            if section_label:
                section_id = f"{sec_no_part}_{section_label}_L{l_val}"
            else:
                # Fallback to old format if parsing fails
                section_parts = []
                for p in parts[sec_idx:]:
                    section_parts.append(p)
                section_id = '_'.join(section_parts)

            # Read CSV and extract f0_hz per tick position
            with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
                # Skip comment lines
                lines = []
                for line in f:
                    if not line.startswith('#'):
                        lines.append(line)

            if not lines:
                continue

            # Parse CSV
            import io
            reader = csv.DictReader(io.StringIO('\n'.join(lines)))

            # Check if f0_hz column exists
            if 'f0_hz' not in reader.fieldnames:
                continue

            # Collect f0_hz values per tick position
            tick_f0_values = {pos: [] for pos in range(n_positions)}

            for row in reader:
                tick = int(row.get('tick_16th', 0))
                f0_str = row.get('f0_hz', '')

                if f0_str and f0_str != '0.0' and f0_str != '0':
                    try:
                        f0 = float(f0_str)
                        # 55.0 Hz means no pitch detected (silence/no bass)
                        if f0 > 0 and f0 != 55.0 and tick < n_positions:
                            tick_f0_values[tick].append(f0)
                    except (ValueError, TypeError):
                        pass

            # Compute median f0 per position
            positions = {}
            for pos in range(n_positions):
                if tick_f0_values[pos]:
                    positions[pos] = float(np.median(tick_f0_values[pos]))
                else:
                    positions[pos] = 0.0

            sections[section_id] = {'positions': positions}

        except Exception as e:
            continue

    return sections


def build_column_headers(pattern_length: int, stem: str = 'drums') -> List[str]:
    """Build the full list of column headers for the output CSV."""
    n_positions = pattern_length * 16
    headers = []

    # Metadata columns
    headers.extend([
        'song_id', 'song_name', 'sec_no', 'section_label',
        'num_repetitions', 'ratio_in_snippet', 'mean_section_tempo', 'time_signature'
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

    # Bass Pitch (F0) - only for bass stem, L*16 positions
    if stem == 'bass':
        for pos in range(n_positions):
            headers.append(f'F0_{pos}')

    return headers


def collect_section_data(
    track_dir: Path,
    track_name: str,
    pattern_length: int,
    ratio_threshold: float,
    stem: str = 'drums',
    variant: str = ''
) -> List[Dict]:
    """
    Collect all data for sections matching pattern_length and ratio_threshold.

    Parameters
    ----------
    track_dir : Path
        Track directory
    track_name : str
        Track name
    pattern_length : int
        Pattern length (1, 2, or 4)
    ratio_threshold : float
        Minimum ratio in snippet to include
    stem : str
        Stem name (e.g., 'drums', 'vocals', 'bass', 'piano', 'other')

    Returns list of row dicts ready for CSV output.
    """
    rows = []
    n_positions = pattern_length * 16

    # noHats variant reads the cymbal-filtered twin folders (6.x_..._noHats)
    sfx = '_noHats' if variant == 'noHats' else ''
    rh_dir = track_dir / f'6.6_anchored_rhythm_histograms{sfx}' / stem
    bt_dir = track_dir / f'6.7_anchored_beat_histograms{sfx}' / stem
    st_dir = track_dir / f'6.8_anchored_statistics{sfx}' / stem

    # File paths (stem-specific subfolders)
    rh_file = rh_dir / f'{track_name}_filtered_anchored_rhythm_histograms.csv'
    gp_file = rh_dir / f'{track_name}_filtered_anchored_groove_pulse_histograms.csv'
    rp_file = rh_dir / f'{track_name}_filtered_anchored_rhythm_patterns.csv'
    bh_file = bt_dir / f'{track_name}_anchored_beat_histograms.csv'
    gpb_file = bt_dir / f'{track_name}_groove_pulse_beat_histograms.csv'
    bp_file = bt_dir / f'{track_name}_anchored_beat_patterns.csv'
    rs_file = st_dir / f'{track_name}_anchored_rhythm_statistics.csv'
    bs_file = st_dir / f'{track_name}_anchored_beat_statistics.csv'

    # Read all data sources
    rh_data = read_rhythm_histogram_data(rh_file, pattern_length) if rh_file.exists() else {}
    gp_data = read_groove_pulse_histogram_data(gp_file, pattern_length) if gp_file.exists() else {}
    rp_data = read_rhythm_pattern_data(rp_file, pattern_length) if rp_file.exists() else {}
    bh_data = read_beat_histogram_data(bh_file, pattern_length) if bh_file.exists() else {}
    gpb_data = read_groove_pulse_beat_histogram_data(gpb_file, pattern_length) if gpb_file.exists() else {}
    bp_data = read_beat_pattern_data(bp_file, pattern_length) if bp_file.exists() else {}
    rs_data = read_rhythm_statistics(rs_file, pattern_length) if rs_file.exists() else {}
    bs_data = read_beat_statistics(bs_file, pattern_length) if bs_file.exists() else {}

    # Read bass pitch data (only for bass stem)
    f0_data = {}
    if stem == 'bass':
        filtered_patterns_dir = track_dir / '6.2_filtered_patterns' / 'bass'
        f0_data = read_bass_pitch_data(filtered_patterns_dir, pattern_length)

    # Get all section IDs from rhythm histogram (primary source)
    if not rh_data:
        return rows

    song_id = extract_song_id(track_name)
    song_name = extract_song_name(track_name)
    time_signature = read_time_signature(track_dir, track_name)

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
        row['time_signature'] = time_signature

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

        # Bass Pitch (F0) - only for bass stem
        if stem == 'bass':
            f0_section = f0_data.get(section_id, {})
            f0_positions = f0_section.get('positions', {})
            for pos in range(n_positions):
                row[f'F0_{pos}'] = f0_positions.get(pos, 0.0)

        rows.append(row)

    return rows


def detect_available_stems(track_dirs: List[Path]) -> List[str]:
    """
    Detect which stems have data by checking the first few tracks.

    Returns list of stem names that have 6.6 rhythm histogram data.
    """
    all_stems = ['vocals', 'drums', 'bass', 'piano', 'other', 'fullmix']
    found_stems = set()

    # Check first 5 tracks to detect available stems
    for track_dir in track_dirs[:5]:
        rhythm_hist_dir = track_dir / '6.6_anchored_rhythm_histograms'
        if rhythm_hist_dir.exists():
            for stem in all_stems:
                stem_dir = rhythm_hist_dir / stem
                if stem_dir.exists() and any(stem_dir.glob('*.csv')):
                    found_stems.add(stem)

    # If no stem subfolders found, assume old structure (drums only, no subfolder)
    if not found_stems:
        return ['drums']  # Backward compatibility

    return sorted(found_stems, key=lambda s: all_stems.index(s))


def create_collected_data(output_dir: Path, stem: str = 'drums', variant: str = ''):
    """
    Create collected data CSVs from batch processing results.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    stem : str
        Stem to collect data for (default: 'drums')
    variant : str
        '' for the regular 6.6/6.7/6.8 folders (all pattern lengths). 'noHats' reads
        the cymbal-filtered twin folders (6.x_..._noHats) which are L2-only, so it
        collects pattern_length == 2 only and prefixes output filenames with 'noHats_'.
    """
    fpfx = 'noHats_' if variant == 'noHats' else ''
    # noHats patterns only exist for L2, so restrict pattern lengths in that variant
    pattern_lengths = [2] if variant == 'noHats' else PATTERN_LENGTHS
    print("\n" + "=" * 80)
    print(f"STEP 22: COLLECT DATA ({stem}{'' if not variant else ' / ' + variant})")
    print("=" * 80)

    # Find all track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis',
                                          'snippet_ratio_batch_analysis', 'collected_data']
    ])

    if not track_dirs:
        print('No track directories found!')
        return

    print(f"Found {len(track_dirs)} track directories")

    # Create output directory (stem-specific)
    collected_dir = output_dir / 'collected_data' / stem
    collected_dir.mkdir(parents=True, exist_ok=True)

    # Process each combination of pattern_length and ratio_threshold
    for pattern_length in pattern_lengths:
        for ratio_threshold in RATIO_THRESHOLDS:
            print(f"\n--- L{pattern_length} ratio>{int(ratio_threshold*100)}% ---")

            all_rows = []

            for track_dir in track_dirs:
                track_name = track_dir.name
                rows = collect_section_data(track_dir, track_name, pattern_length, ratio_threshold, stem=stem, variant=variant)

                if rows:
                    print(f"  {track_name}: {len(rows)} sections")
                    all_rows.extend(rows)

            # Sort by song_id (numeric) then sec_no
            all_rows.sort(key=lambda x: (int(x['song_id']), x['sec_no']))

            # Write CSV
            output_file = collected_dir / f'{fpfx}L{pattern_length}_ratio{int(ratio_threshold*100)}.csv'
            headers = build_column_headers(pattern_length, stem=stem)

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_rows)

            print(f"\n  Total sections: {len(all_rows)}")
            print(f"  Saved: {output_file.name}")

    # Process ratio thresholds with max-ratio selection (0.30, 0.40)
    # For these: if multiple sections per song meet threshold, keep only the one with highest ratio
    for pattern_length in pattern_lengths:
        for ratio_threshold in RATIO_THRESHOLDS_MAX_SELECTION:
            print(f"\n--- L{pattern_length} ratio>{int(ratio_threshold*100)}% (max selection) ---")

            all_rows = []

            for track_dir in track_dirs:
                track_name = track_dir.name
                rows = collect_section_data(track_dir, track_name, pattern_length, ratio_threshold, stem=stem, variant=variant)

                if rows:
                    # If multiple sections for this song, keep only the one with highest ratio
                    if len(rows) > 1:
                        max_row = max(rows, key=lambda r: r['ratio_in_snippet'])
                        print(f"  {track_name}: {len(rows)} sections -> keeping max ratio ({max_row['ratio_in_snippet']:.4f})")
                        all_rows.append(max_row)
                    else:
                        print(f"  {track_name}: 1 section")
                        all_rows.extend(rows)

            # Sort by song_id (numeric) then sec_no
            all_rows.sort(key=lambda x: (int(x['song_id']), x['sec_no']))

            # Write CSV
            output_file = collected_dir / f'{fpfx}L{pattern_length}_ratio{int(ratio_threshold*100)}_maxsel.csv'
            headers = build_column_headers(pattern_length, stem=stem)

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_rows)

            print(f"\n  Total sections: {len(all_rows)} (one per song)")
            print(f"  Saved: {output_file.name}")

    print("\n" + "=" * 80)
    print("✓ All collected data CSVs created successfully!")
    print("=" * 80)


def parse_section_id(section_id: str) -> Dict:
    """
    Parse section_id to extract sec_no, pattern_length, section_label, ratio.
    Example: SecNo1_L4_chorus_0.1344 -> {sec_no: 1, pattern_length: 4, section_label: 'chorus', ratio: 0.1344}
    """
    import re
    match = re.match(r'SecNo(\d+)_L(\d+)_([^_]+)_([0-9.]+)', section_id)
    if match:
        return {
            'sec_no': int(match.group(1)),
            'pattern_length': int(match.group(2)),
            'section_label': match.group(3),
            'ratio_in_snippet': float(match.group(4))
        }
    return {'sec_no': 0, 'pattern_length': 0, 'section_label': '', 'ratio_in_snippet': 0.0}


def read_pironio_sections(json_path: Path) -> Dict[str, Dict]:
    """Read Pironio section metrics from JSON."""
    sections = {}
    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for section_id, section_data in data.get('sections', {}).items():
            metrics = section_data.get('metrics', {})
            if metrics:  # Only include if metrics exist
                sections[section_id] = metrics
    except Exception as e:
        print(f"    Warning: Could not read {json_path}: {e}")
    return sections


def read_yodfat_sections(json_path: Path) -> Dict[str, Dict]:
    """Read Yodfat section metrics from JSON."""
    sections = {}
    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for section_id, section_data in data.get('sections', {}).items():
            metrics = section_data.get('metrics', {})
            if metrics:  # Only include if metrics exist
                sections[section_id] = metrics
    except Exception as e:
        print(f"    Warning: Could not read {json_path}: {e}")
    return sections


def read_pironio_snippet(json_path: Path) -> Dict:
    """Read Pironio snippet (full track) metrics from JSON."""
    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('metrics', {})
    except Exception as e:
        print(f"    Warning: Could not read {json_path}: {e}")
    return {}


def read_yodfat_snippet(json_path: Path) -> Dict:
    """Read Yodfat snippet (full track) metrics from JSON."""
    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('metrics', {})
    except Exception as e:
        print(f"    Warning: Could not read {json_path}: {e}")
    return {}


# Pironio metrics columns (5 fast metrics - slow metrics often have errors)
PIRONIO_METRICS = [
    'viterbi_max', 'viterbi_entropy', 'peak_average', 'RNN_entropy', 'DBN_entropy',
    'neurons_cross_correlation', 'cell_states_precision', 'autocorrelation_periodicity'
]

# Yodfat metrics columns (15 metrics)
YODFAT_METRICS = [
    'tempo', 'duration', 'n_beats',
    'onscc_quart_avg', 'onscc_quart_std', 'onscc_quart_lag_avg', 'onscc_quart_lag_med', 'onscc_quart_lag_std',
    'onscc_half_avg', 'onscc_half_std', 'onscc_half_lag_avg', 'onscc_half_lag_med', 'onscc_half_lag_std',
    'onscc_bar_avg', 'onscc_bar_std', 'onscc_bar_lag_avg', 'onscc_bar_lag_med', 'onscc_bar_lag_std'
]


def collect_pironio_yodfat_data(output_dir: Path):
    """
    Collect Pironio and Yodfat section data into CSVs.

    Creates (filtered by ratio threshold 50% and 70%):
    - pironio_sections_L2_ratio50.csv, pironio_sections_L2_ratio70.csv
    - pironio_sections_L4_ratio50.csv, pironio_sections_L4_ratio70.csv
    - yodfat_sections_L2_ratio50.csv, yodfat_sections_L2_ratio70.csv
    - yodfat_sections_L4_ratio50.csv, yodfat_sections_L4_ratio70.csv
    - pironio_snippet.csv (full snippet Pironio metrics per track)
    - yodfat_snippet.csv (full snippet Yodfat metrics per track)
    """
    print("\n" + "=" * 80)
    print("COLLECTING PIRONIO & YODFAT DATA")
    print("=" * 80)

    # Find all track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis',
                                          'snippet_ratio_batch_analysis', 'collected_data']
    ])

    if not track_dirs:
        print('No track directories found!')
        return

    print(f"Found {len(track_dirs)} track directories")

    # Create output directory
    collected_dir = output_dir / 'collected_data'
    collected_dir.mkdir(parents=True, exist_ok=True)

    # Collect section data by pattern length
    pironio_sections = {1: [], 2: [], 4: []}
    yodfat_sections = {1: [], 2: [], 4: []}

    # Collect snippet data
    pironio_snippets = []
    yodfat_snippets = []

    for track_dir in track_dirs:
        track_name = track_dir.name
        song_id = extract_song_id(track_name)
        song_name = extract_song_name(track_name)

        # Pironio section file
        pironio_sections_file = track_dir / '12_pironio' / f'{track_name}_pironio_sections.json'
        if pironio_sections_file.exists():
            sections = read_pironio_sections(pironio_sections_file)
            for section_id, metrics in sections.items():
                parsed = parse_section_id(section_id)
                pl = parsed['pattern_length']
                if pl in [1, 2, 4]:
                    row = {
                        'song_id': song_id,
                        'song_name': song_name,
                        'sec_no': parsed['sec_no'],
                        'section_label': parsed['section_label'],
                        'ratio_in_snippet': parsed['ratio_in_snippet'],
                    }
                    for metric in PIRONIO_METRICS:
                        row[f'PIR_{metric}'] = metrics.get(metric, '')
                    pironio_sections[pl].append(row)

        # Yodfat section file
        yodfat_sections_file = track_dir / '14_yodfat' / f'{track_name}_yodfat_sections.json'
        if yodfat_sections_file.exists():
            sections = read_yodfat_sections(yodfat_sections_file)
            for section_id, metrics in sections.items():
                parsed = parse_section_id(section_id)
                pl = parsed['pattern_length']
                if pl in [1, 2, 4]:
                    row = {
                        'song_id': song_id,
                        'song_name': song_name,
                        'sec_no': parsed['sec_no'],
                        'section_label': parsed['section_label'],
                        'ratio_in_snippet': parsed['ratio_in_snippet'],
                    }
                    for metric in YODFAT_METRICS:
                        row[f'YOD_{metric}'] = metrics.get(metric, '')
                    yodfat_sections[pl].append(row)

        # Pironio snippet file (full track)
        pironio_snippet_file = track_dir / '12_pironio' / f'{track_name}_pironio_metrics.json'
        if pironio_snippet_file.exists():
            metrics = read_pironio_snippet(pironio_snippet_file)
            if metrics:
                row = {'song_id': song_id, 'song_name': song_name}
                for metric in PIRONIO_METRICS:
                    row[f'PIR_{metric}'] = metrics.get(metric, '')
                pironio_snippets.append(row)

        # Yodfat snippet file (full track)
        yodfat_snippet_file = track_dir / '14_yodfat' / f'{track_name}_yodfat_metrics.json'
        if yodfat_snippet_file.exists():
            metrics = read_yodfat_snippet(yodfat_snippet_file)
            if metrics:
                row = {'song_id': song_id, 'song_name': song_name}
                for metric in YODFAT_METRICS:
                    row[f'YOD_{metric}'] = metrics.get(metric, '')
                yodfat_snippets.append(row)

    # Write Pironio section CSVs (filtered by ratio threshold)
    for pl in [1, 2, 4]:
        for ratio_th in RATIO_THRESHOLDS:
            rows = [r for r in pironio_sections[pl] if r['ratio_in_snippet'] >= ratio_th]
            if rows:
                rows.sort(key=lambda x: (int(x['song_id']), x['sec_no']))
                headers = ['song_id', 'song_name', 'sec_no', 'section_label', 'ratio_in_snippet'] + \
                          [f'PIR_{m}' for m in PIRONIO_METRICS]
                output_file = collected_dir / f'pironio_sections_L{pl}_ratio{int(ratio_th*100)}.csv'
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  ✓ Saved {output_file.name}: {len(rows)} sections")

    # Write Pironio section CSVs with max-selection (30%, 40%)
    for pl in [1, 2, 4]:
        for ratio_th in RATIO_THRESHOLDS_MAX_SELECTION:
            rows = [r for r in pironio_sections[pl] if r['ratio_in_snippet'] >= ratio_th]
            if rows:
                # Group by song_id and keep only max ratio per song
                from collections import defaultdict
                by_song = defaultdict(list)
                for r in rows:
                    by_song[r['song_id']].append(r)
                rows = [max(song_rows, key=lambda x: x['ratio_in_snippet']) for song_rows in by_song.values()]
                rows.sort(key=lambda x: (int(x['song_id']), x['sec_no']))
                headers = ['song_id', 'song_name', 'sec_no', 'section_label', 'ratio_in_snippet'] + \
                          [f'PIR_{m}' for m in PIRONIO_METRICS]
                output_file = collected_dir / f'pironio_sections_L{pl}_ratio{int(ratio_th*100)}_maxsel.csv'
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  ✓ Saved {output_file.name}: {len(rows)} sections (max selection)")

    # Write Yodfat section CSVs (filtered by ratio threshold)
    for pl in [1, 2, 4]:
        for ratio_th in RATIO_THRESHOLDS:
            rows = [r for r in yodfat_sections[pl] if r['ratio_in_snippet'] >= ratio_th]
            if rows:
                rows.sort(key=lambda x: (int(x['song_id']), x['sec_no']))
                headers = ['song_id', 'song_name', 'sec_no', 'section_label', 'ratio_in_snippet'] + \
                          [f'YOD_{m}' for m in YODFAT_METRICS]
                output_file = collected_dir / f'yodfat_sections_L{pl}_ratio{int(ratio_th*100)}.csv'
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  ✓ Saved {output_file.name}: {len(rows)} sections")

    # Write Yodfat section CSVs with max-selection (30%, 40%)
    for pl in [1, 2, 4]:
        for ratio_th in RATIO_THRESHOLDS_MAX_SELECTION:
            rows = [r for r in yodfat_sections[pl] if r['ratio_in_snippet'] >= ratio_th]
            if rows:
                # Group by song_id and keep only max ratio per song
                from collections import defaultdict
                by_song = defaultdict(list)
                for r in rows:
                    by_song[r['song_id']].append(r)
                rows = [max(song_rows, key=lambda x: x['ratio_in_snippet']) for song_rows in by_song.values()]
                rows.sort(key=lambda x: (int(x['song_id']), x['sec_no']))
                headers = ['song_id', 'song_name', 'sec_no', 'section_label', 'ratio_in_snippet'] + \
                          [f'YOD_{m}' for m in YODFAT_METRICS]
                output_file = collected_dir / f'yodfat_sections_L{pl}_ratio{int(ratio_th*100)}_maxsel.csv'
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  ✓ Saved {output_file.name}: {len(rows)} sections (max selection)")

    # Write Pironio snippet CSV
    if pironio_snippets:
        pironio_snippets.sort(key=lambda x: int(x['song_id']))
        headers = ['song_id', 'song_name'] + [f'PIR_{m}' for m in PIRONIO_METRICS]
        output_file = collected_dir / 'pironio_snippet.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(pironio_snippets)
        print(f"  ✓ Saved {output_file.name}: {len(pironio_snippets)} tracks")

    # Write Yodfat snippet CSV
    if yodfat_snippets:
        yodfat_snippets.sort(key=lambda x: int(x['song_id']))
        headers = ['song_id', 'song_name'] + [f'YOD_{m}' for m in YODFAT_METRICS]
        output_file = collected_dir / 'yodfat_snippet.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(yodfat_snippets)
        print(f"  ✓ Saved {output_file.name}: {len(yodfat_snippets)} tracks")

    print("\n✓ Pironio & Yodfat data collection complete!")


# Spotify audio features to collect
SPOTIFY_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'time_signature'
]

# Path to Spotify audio features JSON
SPOTIFY_FEATURES_PATH = Path(__file__).parent.parent.parent / 'groove-data' / 'spotify' / 'spotify_audio_features.json'


def load_spotify_features() -> Dict[str, Dict]:
    """Load Spotify audio features from JSON file."""
    import json
    if not SPOTIFY_FEATURES_PATH.exists():
        print(f"  Warning: Spotify features file not found: {SPOTIFY_FEATURES_PATH}")
        return {}

    with open(SPOTIFY_FEATURES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to dict keyed by song_id (string)
    features = {}
    for song_id, song_data in data.items():
        audio_features = song_data.get('audio_features', [])
        if audio_features and len(audio_features) > 0:
            features[song_id] = audio_features[0]  # Take first result
    return features


def collect_spotify_data(output_dir: Path):
    """
    Collect Spotify audio features into CSVs.

    Creates:
    - spotify_L2_ratio50.csv, spotify_L2_ratio70.csv (songs with L2 sections meeting ratio)
    - spotify_L4_ratio50.csv, spotify_L4_ratio70.csv (songs with L4 sections meeting ratio)
    - spotify_all.csv (all songs in dataset)
    """
    print("\n" + "=" * 80)
    print("COLLECTING SPOTIFY AUDIO FEATURES")
    print("=" * 80)

    # Load Spotify features
    spotify_features = load_spotify_features()
    if not spotify_features:
        print("  No Spotify features loaded, skipping...")
        return

    print(f"  Loaded Spotify features for {len(spotify_features)} songs")

    # Find all track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis',
                                          'snippet_ratio_batch_analysis', 'collected_data']
    ])

    if not track_dirs:
        print('No track directories found!')
        return

    # Create output directory
    collected_dir = output_dir / 'collected_data'
    collected_dir.mkdir(parents=True, exist_ok=True)

    # Collect song_ids that have sections meeting each criteria
    # Key: (pattern_length, ratio_threshold) -> set of song_ids
    songs_by_criteria = {}
    for pl in [1, 2, 4]:
        for ratio_th in RATIO_THRESHOLDS + RATIO_THRESHOLDS_MAX_SELECTION:
            songs_by_criteria[(pl, ratio_th)] = set()
    all_song_ids = set()

    for track_dir in track_dirs:
        track_name = track_dir.name
        song_id = extract_song_id(track_name)
        all_song_ids.add(song_id)

        # Check Pironio sections file to get section info
        pironio_sections_file = track_dir / '12_pironio' / f'{track_name}_pironio_sections.json'
        yodfat_sections_file = track_dir / '14_yodfat' / f'{track_name}_yodfat_sections.json'

        # Try to get sections from either file
        sections_file = pironio_sections_file if pironio_sections_file.exists() else yodfat_sections_file

        if sections_file.exists():
            try:
                import json
                with open(sections_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for section_id in data.get('sections', {}).keys():
                    parsed = parse_section_id(section_id)
                    pl = parsed['pattern_length']
                    ratio = parsed['ratio_in_snippet']

                    if pl in [1, 2, 4]:
                        for ratio_th in RATIO_THRESHOLDS + RATIO_THRESHOLDS_MAX_SELECTION:
                            if ratio >= ratio_th:
                                songs_by_criteria[(pl, ratio_th)].add(song_id)
            except Exception as e:
                print(f"    Warning: Could not read {sections_file.name}: {e}")

    # Write Spotify CSVs for each L/ratio combination
    headers = ['song_id', 'song_name'] + [f'SP_{f}' for f in SPOTIFY_FEATURES]

    for pl in [1, 2, 4]:
        for ratio_th in RATIO_THRESHOLDS:
            song_ids = songs_by_criteria[(pl, ratio_th)]
            rows = []

            for song_id in sorted(song_ids, key=int):
                if song_id in spotify_features:
                    features = spotify_features[song_id]
                    row = {'song_id': song_id, 'song_name': ''}
                    for feat in SPOTIFY_FEATURES:
                        row[f'SP_{feat}'] = features.get(feat, '')
                    rows.append(row)

            if rows:
                output_file = collected_dir / f'spotify_L{pl}_ratio{int(ratio_th*100)}.csv'
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  ✓ Saved {output_file.name}: {len(rows)} songs")

    # Write Spotify CSVs for max-selection thresholds (30%, 40%)
    # Note: Spotify features are per-song, so no max-selection needed - just use songs that have qualifying sections
    for pl in [1, 2, 4]:
        for ratio_th in RATIO_THRESHOLDS_MAX_SELECTION:
            song_ids = songs_by_criteria[(pl, ratio_th)]
            rows = []

            for song_id in sorted(song_ids, key=int):
                if song_id in spotify_features:
                    features = spotify_features[song_id]
                    row = {'song_id': song_id, 'song_name': ''}
                    for feat in SPOTIFY_FEATURES:
                        row[f'SP_{feat}'] = features.get(feat, '')
                    rows.append(row)

            if rows:
                output_file = collected_dir / f'spotify_L{pl}_ratio{int(ratio_th*100)}_maxsel.csv'
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  ✓ Saved {output_file.name}: {len(rows)} songs")

    # Write spotify_all.csv with all songs
    rows = []
    # Filter to only numeric song_ids
    valid_song_ids = [sid for sid in all_song_ids if sid.isdigit()]
    for song_id in sorted(valid_song_ids, key=int):
        if song_id in spotify_features:
            features = spotify_features[song_id]
            row = {'song_id': song_id, 'song_name': ''}
            for feat in SPOTIFY_FEATURES:
                row[f'SP_{feat}'] = features.get(feat, '')
            rows.append(row)

    if rows:
        output_file = collected_dir / 'spotify_all.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  ✓ Saved {output_file.name}: {len(rows)} songs")

    print("\n✓ Spotify data collection complete!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python collect_data.py /path/to/batch/output [stem]')
        print('Example: python collect_data.py "/Volumes/PortableSSD/06_Testing/new test feb20"')
        print('Example: python collect_data.py "/path/to/output" drums')
        print('If stem is not specified, all available stems will be processed.')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f'Error: Directory does not exist: {output_dir}')
        sys.exit(1)

    # Find track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis',
                                          'snippet_ratio_batch_analysis', 'collected_data']
    ])

    # Determine which stems to process
    if len(sys.argv) >= 3:
        # Specific stem requested
        stems = [sys.argv[2]]
    else:
        # Auto-detect available stems
        stems = detect_available_stems(track_dirs)
        print(f"Detected stems: {stems}")

    # Collect data for each stem
    for stem in stems:
        create_collected_data(output_dir, stem=stem)

    # noHats variant (drums-only, L2-only)
    if 'drums' in stems:
        create_collected_data(output_dir, stem='drums', variant='noHats')

    # Pironio/Yodfat and Spotify are not stem-specific
    collect_pironio_yodfat_data(output_dir)
    collect_spotify_data(output_dir)
