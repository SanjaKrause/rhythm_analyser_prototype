#!/usr/bin/env python3
"""
Anchored Beat Histograms - Create beat-level IOI visualizations from section-anchored data.

This module reads section-anchored CSV files from 6.2_filtered_patterns and calculates
inter-onset intervals (IOI) between consecutive onsets, categorizing them by duration
(4/4, 2/4, 1/4, 3/16, 1/8, 6/16, 1/16).

Input: 6.2_filtered_patterns/SecNoX_LY_label_ratio_anchored.csv files
Output: 6.7_anchored_beat_histograms/ with IOI CSVs per section and combined histogram plots

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Set, Tuple
import re
import sys


def get_groove_pulse_positions(groove_pulse_csv: str, section_id: str) -> Set[int]:
    """
    Get positions that pass the groove pulse threshold for a given section.

    Reads the filtered groove pulse histogram CSV from 6.6 and returns
    the set of positions where onset_strength_filtered > 0.

    Parameters
    ----------
    groove_pulse_csv : str
        Path to the filtered_anchored_groove_pulse_histograms.csv file
    section_id : str
        Section identifier (e.g., 'SecNo1_verse_L2')

    Returns
    -------
    Set[int]
        Set of position indices (1-based) that pass the groove pulse threshold
    """
    csv_path = Path(groove_pulse_csv)
    if not csv_path.exists():
        return set()

    df = pd.read_csv(csv_path)

    # Filter to matching section
    df_section = df[df['section_id'] == section_id]

    if df_section.empty:
        return set()

    # Get positions where onset_strength_filtered > 0
    groove_positions = set(
        df_section[df_section['onset_strength_filtered'] > 0]['position'].astype(int).tolist()
    )

    return groove_positions


def filter_ioi_by_groove_positions(
    df_ioi: pd.DataFrame,
    groove_positions: Set[int],
    pattern_length: int
) -> pd.DataFrame:
    """
    Filter IOI data to only include IOIs where BOTH onset positions are groove pulse positions.

    Parameters
    ----------
    df_ioi : pd.DataFrame
        DataFrame with IOI data (from process_section_ioi)
    groove_positions : Set[int]
        Set of groove pulse positions (1-based, within pattern)
    pattern_length : int
        Pattern length in bars (2 or 4)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only IOIs between groove pulse positions
    """
    if df_ioi.empty or not groove_positions:
        return pd.DataFrame()

    num_positions = pattern_length * 16  # 32 for L2, 64 for L4

    def get_position_in_pattern(bar: int, tick: int) -> int:
        """Convert bar number and tick to 1-based position within pattern."""
        # bar is 0-based within pattern, tick is 0-15
        bar_in_pattern = bar % pattern_length
        return bar_in_pattern * 16 + tick + 1

    # Calculate positions for onset1 and onset2
    onset1_positions = df_ioi.apply(
        lambda row: get_position_in_pattern(row['onset1_bar'], row['onset1_tick']),
        axis=1
    )
    onset2_positions = df_ioi.apply(
        lambda row: get_position_in_pattern(row['onset2_bar'], row['onset2_tick']),
        axis=1
    )

    # Filter: both positions must be in groove_positions
    mask = onset1_positions.isin(groove_positions) & onset2_positions.isin(groove_positions)

    return df_ioi[mask].copy()


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
    categories = [
        (16, '4/4'),
        (8, '2/4'),
        (6, '6/16'),
        (4, '1/4'),
        (3, '3/16'),
        (2, '1/8'),
        (1, '1/16'),
    ]

    ioi_rounded = round(ioi_ticks)
    for ticks, category in categories:
        if ioi_rounded >= ticks:
            return category

    return '1/16'


def parse_anchored_filename(filename: str) -> Optional[Dict]:
    """
    Parse section-anchored CSV filename to extract metadata.

    Format: SecNoX_LY_label_ratio_anchored.csv
    Example: SecNo1_L2_chorus_0.5438_anchored.csv

    Returns
    -------
    dict or None
        Dictionary with section_no, pattern_length, section_label, ratio_in_snippet
    """
    pattern = r'SecNo(\d+)_L(\d+)_([^_]+)_([0-9.]+)_anchored\.csv'
    match = re.match(pattern, filename)

    if match:
        return {
            'section_no': int(match.group(1)),
            'pattern_length': int(match.group(2)),
            'section_label': match.group(3),
            'ratio_in_snippet': float(match.group(4))
        }
    return None


def read_anchored_csv_metadata(csv_path: str) -> Dict:
    """
    Read metadata from comment lines in anchored CSV file.
    """
    metadata = {}

    with open(csv_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            line = line.strip()[2:]  # Remove '# '
            if '=' in line:
                key, value = line.split('=', 1)
                try:
                    if '.' in value:
                        metadata[key] = float(value)
                    elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                        metadata[key] = int(value)
                    else:
                        metadata[key] = value
                except ValueError:
                    metadata[key] = value

    return metadata


def process_section_ioi(csv_path: str) -> pd.DataFrame:
    """
    Process IOI from a single section-anchored CSV file.

    Reads the anchored CSV and calculates inter-onset intervals between
    consecutive onsets within the section.

    Returns
    -------
    pd.DataFrame
        DataFrame with IOI data
    """
    metadata = read_anchored_csv_metadata(csv_path)
    pattern_length = metadata.get('pattern_length', 2)
    section_label = metadata.get('section_label', 'unknown')
    no_of_repetitions = metadata.get('no_of_repetitions', 0)
    ratio_in_snippet = metadata.get('ratio_in_snippet', 0.0)

    df = pd.read_csv(csv_path, comment='#')

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(['bar_number', 'tick_16th']).reset_index(drop=True)
    df = df[df['onset_time'].notna()].copy()

    if len(df) < 2:
        return pd.DataFrame()

    all_ioi_data = []

    for i in range(len(df) - 1):
        onset1 = df.iloc[i]
        onset2 = df.iloc[i + 1]

        tick1 = onset1['tick_16th']
        tick2 = onset2['tick_16th']
        bar1 = onset1['bar_number']
        bar2 = onset2['bar_number']

        # Get tick_phase if available
        tick_phase1 = onset1.get('tick_phase', 0.0) if pd.notna(onset1.get('tick_phase', np.nan)) else 0.0
        tick_phase2 = onset2.get('tick_phase', 0.0) if pd.notna(onset2.get('tick_phase', np.nan)) else 0.0

        tick1_absolute = bar1 * 16 + tick1
        tick2_absolute = bar2 * 16 + tick2

        tick_delta = tick2_absolute - tick1_absolute
        tick_phase_diff = tick_phase2 - tick_phase1
        ioi_exact_ticks = tick_delta + tick_phase_diff

        time1 = onset1['onset_time']
        time2 = onset2['onset_time']
        ioi_seconds = time2 - time1

        ioi_category = categorize_ioi(ioi_exact_ticks)

        all_ioi_data.append({
            'section_label': section_label,
            'pattern_length': pattern_length,
            'num_repetitions': no_of_repetitions,
            'ratio_in_snippet': ratio_in_snippet,
            'onset1_bar': int(bar1),
            'onset1_tick': int(tick1),
            'onset1_tick_absolute': int(tick1_absolute),
            'onset1_tick_phase': float(tick_phase1),
            'onset1_time': float(time1),
            'onset2_bar': int(bar2),
            'onset2_tick': int(tick2),
            'onset2_tick_absolute': int(tick2_absolute),
            'onset2_tick_phase': float(tick_phase2),
            'onset2_time': float(time2),
            'tick_delta': int(tick_delta),
            'tick_phase_diff': float(tick_phase_diff),
            'ioi_exact_ticks': float(ioi_exact_ticks),
            'ioi_seconds': float(ioi_seconds),
            'ioi_category': ioi_category
        })

    if not all_ioi_data:
        return pd.DataFrame()

    df_ioi = pd.DataFrame(all_ioi_data)

    category_order = ['4/4', '2/4', '1/4', '3/16', '1/8', '6/16', '1/16']
    df_ioi['ioi_category'] = pd.Categorical(df_ioi['ioi_category'], categories=category_order, ordered=True)

    return df_ioi


def create_anchored_beat_histograms(
    filtered_patterns_dir: str,
    track_id: str,
    output_dir: str,
    pulse_threshold: float = 0.1
) -> Dict:
    """
    Create beat-level histograms from section-anchored inter-onset interval data.

    Processes all anchored CSV files in 6.2_filtered_patterns directory and creates
    IOI CSVs per section and combined histogram plots.

    Parameters
    ----------
    filtered_patterns_dir : str
        Directory containing the section-anchored CSV files (6.2_filtered_patterns)
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots (6.7_anchored_beat_histograms)
    pulse_threshold : float
        Threshold for pulse filtering displayed as a red horizontal line (default 0.1 = 10%)

    Returns
    -------
    dict
        Dictionary with paths to saved files and statistics
    """
    print(f"\n  [Anchored Beat Histograms] Creating section-anchored IOI histograms...")

    filtered_dir = Path(filtered_patterns_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    anchored_files = sorted(filtered_dir.glob('SecNo*_anchored.csv'))

    if not anchored_files:
        print(f"    Warning: No anchored CSV files found in {filtered_patterns_dir}")
        return {}

    print(f"    Found {len(anchored_files)} anchored CSV files")

    # Group files by section number
    sections = {}
    for csv_file in anchored_files:
        parsed = parse_anchored_filename(csv_file.name)
        if parsed:
            sec_no = parsed['section_no']
            if sec_no not in sections:
                sections[sec_no] = {}
            # Read full metadata from anchored CSV file
            full_metadata = read_anchored_csv_metadata(str(csv_file))
            sections[sec_no][parsed['pattern_length']] = {
                'path': csv_file,
                'metadata': parsed,
                'full_metadata': full_metadata
            }

    # Process each section and pattern length, save separate CSVs
    all_ioi_data = []
    output_files = {'csv_files': []}

    for sec_no in sorted(sections.keys()):
        for pattern_length in sorted(sections[sec_no].keys()):
            csv_info = sections[sec_no][pattern_length]
            csv_path = csv_info['path']
            metadata = csv_info['metadata']

            df_ioi = process_section_ioi(str(csv_path))

            if not df_ioi.empty:
                section_label = metadata['section_label']
                # Add section_id and section_no columns right after section_label
                df_ioi.insert(0, 'section_id', f'SecNo{sec_no}_{section_label}_L{pattern_length}')
                df_ioi.insert(2, 'section_no', sec_no)
                all_ioi_data.append(df_ioi)

                # Save separate CSV for this section/pattern length
                ratio = metadata['ratio_in_snippet']
                csv_filename = f"SecNo{sec_no}_L{pattern_length}_{section_label}_{ratio:.4f}_ioi_data.csv"
                csv_output_path = output_path / csv_filename

                # Write CSV with metadata header (copied from original anchored CSV)
                full_metadata = csv_info['full_metadata']
                with open(csv_output_path, 'w') as f:
                    for key, value in full_metadata.items():
                        f.write(f"# {key}={value}\n")
                    df_ioi.to_csv(f, index=False)
                output_files['csv_files'].append(str(csv_output_path))
                print(f"    SecNo{sec_no} L{pattern_length}: {len(df_ioi)} IOIs -> {csv_filename}")

    if not all_ioi_data:
        print(f"    Warning: No IOI data found")
        return {}

    df_all = pd.concat(all_ioi_data, ignore_index=True)

    # Save aggregated IOI data CSV (all sections combined)
    aggregated_csv_path = output_path / f'{track_id}_anchored_ioi_data.csv'
    df_all.to_csv(aggregated_csv_path, index=False)
    output_files['aggregated_ioi_csv'] = str(aggregated_csv_path)
    print(f"    Saved aggregated IOI data: {aggregated_csv_path.name}")

    # Get all unique section numbers and pattern lengths
    all_sec_nos = sorted(sections.keys())
    all_pattern_lengths = sorted(set(pl for sec in sections.values() for pl in sec.keys()))

    # Create figure: rows = pattern lengths (L2 top, L4 bottom), columns = sections
    num_cols = len(all_sec_nos)
    num_rows = len(all_pattern_lengths)

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(8 * num_cols, 4 * num_rows),
                             squeeze=False)
    fig.suptitle(f'Anchored Beat Histograms (IOI) — {track_id}',
                 fontsize=14, fontweight='bold', y=0.995)

    colors = {2: '#F39C12', 4: '#2ECC71'}  # Orange for L2, Green for L4

    category_order = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']
    category_to_ticks = {
        '1/16': 1, '1/8': 2, '3/16': 3, '1/4': 4,
        '6/16': 6, '2/4': 8, '4/4': 16
    }

    # Collect all stats for aggregated CSV
    all_stats_rows = []

    for row_idx, pattern_length in enumerate(all_pattern_lengths):
        for col_idx, sec_no in enumerate(all_sec_nos):
            ax = axes[row_idx, col_idx]
            color = colors.get(pattern_length, '#3498DB')

            df_section = df_all[(df_all['section_no'] == sec_no) &
                               (df_all['pattern_length'] == pattern_length)]

            if df_section.empty:
                ax.text(0.5, 0.5, f'No L{pattern_length} data\nfor SecNo{sec_no}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'SecNo{sec_no} — L{pattern_length}', fontsize=10, fontweight='bold')
                continue

            section_label = df_section['section_label'].iloc[0]
            ratio = df_section['ratio_in_snippet'].iloc[0]
            num_reps = df_section['num_repetitions'].iloc[0]

            # Calculate statistics for each IOI category
            category_stats = {}
            for cat in category_order:
                cat_data = df_section[df_section['ioi_category'] == cat]['ioi_exact_ticks']
                nominal_ticks = category_to_ticks[cat]

                if len(cat_data) > 0:
                    count = len(cat_data)
                    median_ioi = np.median(cat_data)
                    # Calculate shifts (deviation from nominal)
                    shifts = cat_data - nominal_ticks

                    if len(cat_data) > 1:
                        q75, q25 = np.percentile(cat_data, [75, 25])
                        iqr_scaled = (q75 - q25) * 1.5
                        # IQR of shifts (same as IQR of raw values since it's a constant offset)
                        shift_q75, shift_q25 = np.percentile(shifts, [75, 25])
                        iqr_shift = shift_q75 - shift_q25
                    else:
                        iqr_scaled = 0.0
                        iqr_shift = 0.0

                    median_shift = np.median(shifts)
                    category_stats[cat] = {
                        'count': count,
                        'median': median_ioi,
                        'median_shift': median_shift,
                        'iqr_scaled': iqr_scaled,
                        'iqr_shift': iqr_shift,
                        'nominal_ticks': nominal_ticks
                    }
                else:
                    category_stats[cat] = {
                        'count': 0,
                        'median': np.nan,
                        'median_shift': np.nan,
                        'iqr_scaled': 0.0,
                        'iqr_shift': 0.0,
                        'nominal_ticks': nominal_ticks
                    }

            counts = np.array([category_stats[cat]['count'] for cat in category_order])
            medians = np.array([category_stats[cat]['median'] for cat in category_order])
            iqrs_scaled = np.array([category_stats[cat]['iqr_scaled'] for cat in category_order])

            max_count = np.max(counts) if len(counts) > 0 else 1
            onset_strength = counts / max_count if max_count > 0 else counts

            base_positions_log = np.array([np.log2(category_to_ticks[cat]) for cat in category_order])
            shifted_positions_log = base_positions_log.copy()

            for i, (cat, median_val) in enumerate(zip(category_order, medians)):
                if not np.isnan(median_val) and median_val > 0:
                    shifted_positions_log[i] = np.log2(median_val)

            bar_width = 0.15
            ax.bar(shifted_positions_log, onset_strength, width=bar_width,
                  color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add IQR error bars
            for i, (cat, shifted_log, strength, iqr_val, median_val) in enumerate(
                zip(category_order, shifted_positions_log, onset_strength, iqrs_scaled, medians)):
                if strength > 0 and iqr_val > 0 and not np.isnan(median_val):
                    error_bar_y = strength * 0.9
                    log_upper = np.log2(median_val + iqr_val / 2)
                    log_lower = np.log2(max(0.1, median_val - iqr_val / 2))
                    iqr_log = (log_upper - log_lower) / 2
                    ax.errorbar(shifted_log, error_bar_y, xerr=iqr_log, fmt='none',
                               ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

            # Add deviation labels (median shift)
            for i, (cat, shifted_log, strength, median_val) in enumerate(
                zip(category_order, shifted_positions_log, onset_strength, medians)):
                if strength > 0 and not np.isnan(median_val):
                    nominal_ticks = category_to_ticks[cat]
                    relative_deviation = median_val - nominal_ticks
                    label_text = f'{relative_deviation:.2f}'.replace('0.', '.').replace('-0.', '-.')
                    ax.text(shifted_log, strength, label_text, ha='center', va='bottom',
                           fontsize=7, fontweight='bold')

            tick_values = [1, 2, 3, 4, 6, 8, 16]
            tick_positions_log = [np.log2(v) for v in tick_values]
            tick_labels = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']

            ax.set_xticks(tick_positions_log)
            ax.set_xticklabels(tick_labels, fontsize=9)
            ax.set_ylabel('Onset Strength', fontsize=9)

            max_strength = np.max(onset_strength) if max_count > 0 else 1.0
            ax.set_ylim(0, max_strength * 1.2)

            # Add secondary y-axis for actual counts
            ax2 = ax.twinx()
            ax2.set_ylim(0, max_count * 1.2)
            ax2.set_ylabel('Count', fontsize=9)

            title = f'SecNo{sec_no} — {section_label} — L{pattern_length}'
            if num_reps:
                title += f' — {num_reps} reps'
            title += f' — {len(df_section)} IOIs'
            if ratio:
                title += f' — {ratio:.1%} of snippet'

            ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
            ax.grid(True, alpha=0.3, axis='y')

            for tick_log in tick_positions_log:
                ax.axvline(x=tick_log, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)

            # Add red horizontal line for pulse threshold
            ax.axhline(y=pulse_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f'Threshold ({pulse_threshold:.0%})')

            ax.set_xlim(-0.5, 4.5)

            if row_idx == num_rows - 1:
                ax.set_xlabel('IOI Category (16th note ticks, log scale)', fontsize=9)

            # Determine which categories pass the pulse threshold
            passes_threshold = onset_strength >= pulse_threshold

            # Save beat histogram summary CSV for this section/pattern length
            csv_stats_rows = []
            for i, cat in enumerate(category_order):
                stats = category_stats[cat]
                row_data = {
                    'ioi_category': cat,
                    'nominal_ticks': stats['nominal_ticks'],
                    'count': stats['count'],
                    'onset_strength': float(onset_strength[i]),
                    'passes_threshold': bool(passes_threshold[i]),
                    'median_ioi': stats['median'],
                    'median_shift': stats['median_shift'],
                    'iqr_shift': stats['iqr_shift'],
                    'iqr_scaled': stats['iqr_scaled']
                }
                csv_stats_rows.append(row_data)
                # Add to aggregated stats with section info
                all_stats_rows.append({
                    'section_id': f'SecNo{sec_no}_{section_label}_L{pattern_length}',
                    'section_label': section_label,
                    'section_no': sec_no,
                    'pattern_length': pattern_length,
                    'num_repetitions': num_reps,
                    'ratio_in_snippet': ratio,
                    **row_data
                })
            df_stats = pd.DataFrame(csv_stats_rows)
            stats_csv_filename = f"SecNo{sec_no}_L{pattern_length}_{section_label}_{ratio:.4f}_beat_histogram_stats.csv"
            stats_csv_path = output_path / stats_csv_filename

            # Write CSV with metadata header (copied from original anchored CSV)
            full_metadata = sections[sec_no][pattern_length]['full_metadata']
            with open(stats_csv_path, 'w') as f:
                for key, value in full_metadata.items():
                    f.write(f"# {key}={value}\n")
                f.write(f"# pulse_threshold={pulse_threshold}\n")
                df_stats.to_csv(f, index=False)
            output_files['csv_files'].append(str(stats_csv_path))

    # Save aggregated stats CSV (all sections combined)
    if all_stats_rows:
        df_all_stats = pd.DataFrame(all_stats_rows)
        aggregated_stats_csv_path = output_path / f'{track_id}_anchored_beat_histograms.csv'
        df_all_stats.to_csv(aggregated_stats_csv_path, index=False)
        output_files['aggregated_stats_csv'] = str(aggregated_stats_csv_path)
        print(f"    Saved aggregated beat histograms: {aggregated_stats_csv_path.name}")

    # Add a single legend for the entire figure describing the visual elements
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='Bar: median IOI (ticks)'),
        Line2D([0], [0], color='black', linewidth=1.5, marker='|', markersize=10, label='Error bar: IQR × 1.5'),
        Line2D([0], [0], color='none', marker='$+.02$', markersize=12, markerfacecolor='black',
               markeredgecolor='none', label='Text: median shift (ticks)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Threshold ({pulse_threshold:.0%})'),
    ]
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend at bottom
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.02))

    output_png = output_path / f'{track_id}_anchored_beat_histograms.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    output_files['png'] = str(output_png)

    print(f"    Processed {len(df_all)} total IOIs across {num_cols} sections")

    return output_files


def create_anchored_beat_histograms_all_onsets(
    filtered_patterns_dir: str,
    track_id: str,
    output_dir: str
) -> Dict:
    """
    Create beat-level scatter plots showing all individual IOIs as markers.
    """
    print(f"\n  [Anchored Beat Histograms - All Onsets] Creating individual IOI scatter plots...")

    filtered_dir = Path(filtered_patterns_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    anchored_files = sorted(filtered_dir.glob('SecNo*_anchored.csv'))

    if not anchored_files:
        print(f"    Warning: No anchored CSV files found")
        return {}

    sections = {}
    for csv_file in anchored_files:
        parsed = parse_anchored_filename(csv_file.name)
        if parsed:
            sec_no = parsed['section_no']
            if sec_no not in sections:
                sections[sec_no] = {}
            sections[sec_no][parsed['pattern_length']] = {
                'path': csv_file,
                'metadata': parsed
            }

    all_ioi_data = []
    for sec_no in sorted(sections.keys()):
        for pattern_length in sorted(sections[sec_no].keys()):
            csv_path = sections[sec_no][pattern_length]['path']
            df_ioi = process_section_ioi(str(csv_path))
            if not df_ioi.empty:
                df_ioi['section_no'] = sec_no
                all_ioi_data.append(df_ioi)

    if not all_ioi_data:
        print(f"    Warning: No IOI data found")
        return {}

    df_all = pd.concat(all_ioi_data, ignore_index=True)

    # Get all unique section numbers and pattern lengths
    all_sec_nos = sorted(sections.keys())
    all_pattern_lengths = sorted(set(pl for sec in sections.values() for pl in sec.keys()))

    # Create figure: rows = pattern lengths (L2 top, L4 bottom), columns = sections
    num_cols = len(all_sec_nos)
    num_rows = len(all_pattern_lengths)

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(8 * num_cols, 3 * num_rows),
                             squeeze=False)
    fig.suptitle(f'Anchored Beat Histograms — All IOIs — {track_id}',
                 fontsize=14, fontweight='bold', y=0.995)

    category_to_ticks = {
        '1/16': 1, '1/8': 2, '3/16': 3, '1/4': 4,
        '6/16': 6, '2/4': 8, '4/4': 16
    }

    for row_idx, pattern_length in enumerate(all_pattern_lengths):
        for col_idx, sec_no in enumerate(all_sec_nos):
            ax = axes[row_idx, col_idx]

            df_section = df_all[(df_all['section_no'] == sec_no) &
                               (df_all['pattern_length'] == pattern_length)]

            if df_section.empty:
                ax.text(0.5, 0.5, f'No L{pattern_length} data\nfor SecNo{sec_no}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'SecNo{sec_no} — L{pattern_length}', fontsize=10, fontweight='bold')
                continue

            section_label = df_section['section_label'].iloc[0]
            ratio = df_section['ratio_in_snippet'].iloc[0]
            num_reps = df_section['num_repetitions'].iloc[0]

            ioi_values = df_section['ioi_exact_ticks'].values
            ioi_categories = df_section['ioi_category'].values

            # Filter out invalid values
            valid_mask = ioi_values > 0
            ioi_values = ioi_values[valid_mask]
            ioi_categories = ioi_categories[valid_mask]

            if len(ioi_values) == 0:
                ax.text(0.5, 0.5, 'No valid IOIs',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                continue

            ioi_log = np.log2(ioi_values)

            np.random.seed(42 + sec_no + pattern_length)
            y_positions = np.random.uniform(0.3, 0.7, size=len(ioi_values))

            # Define colors for each IOI category
            category_colors = {
                '1/16': '#E74C3C',  # Red
                '1/8': '#F39C12',   # Orange
                '3/16': '#F1C40F',  # Yellow
                '1/4': '#2ECC71',   # Green
                '6/16': '#8E44AD',  # Purple
                '2/4': '#3498DB',   # Blue
                '4/4': '#2C3E50'    # Dark blue-gray
            }

            # Map categories to colors
            point_colors = [category_colors.get(cat, 'black') for cat in ioi_categories]

            ax.scatter(ioi_log, y_positions, marker='x', s=50,
                      c=point_colors, alpha=0.7, linewidths=1.5)

            tick_values = [1, 2, 3, 4, 6, 8, 16]
            tick_positions_log = [np.log2(v) for v in tick_values]
            tick_labels = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']

            ax.set_xticks(tick_positions_log)
            ax.set_xticklabels(tick_labels, fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_yticks([])

            title = f'SecNo{sec_no} — {section_label} — L{pattern_length}'
            if num_reps:
                title += f' — {num_reps} reps'
            title += f' — {len(df_section)} IOIs'
            if ratio:
                title += f' — {ratio:.1%}'

            ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
            ax.grid(True, alpha=0.3, axis='x')

            for tick_log in tick_positions_log:
                ax.axvline(x=tick_log, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)

            ax.set_xlim(-0.5, 4.5)

            if row_idx == num_rows - 1:
                ax.set_xlabel('IOI (16th note ticks, log scale)', fontsize=9)

    # Add a single legend for the entire figure
    from matplotlib.lines import Line2D
    category_colors = {
        '1/16': '#E74C3C',  # Red
        '1/8': '#F39C12',   # Orange
        '3/16': '#F1C40F',  # Yellow
        '1/4': '#2ECC71',   # Green
        '6/16': '#8E44AD',  # Purple
        '2/4': '#3498DB',   # Blue
        '4/4': '#2C3E50'    # Dark blue-gray
    }
    legend_elements = [Line2D([0], [0], marker='x', color='w', markerfacecolor=color,
                              markeredgecolor=color, markersize=8, label=cat, linewidth=0)
                       for cat, color in category_colors.items()]
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend at bottom
    fig.legend(handles=legend_elements, loc='lower center', ncol=7, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.02))

    output_png = output_path / f'{track_id}_anchored_beat_histograms_all_onsets.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    return {'png': str(output_png)}


def create_anchored_groove_pulse_beat_histograms(
    filtered_patterns_dir: str,
    track_id: str,
    output_dir: str,
    groove_pulse_csv: str,
    pulse_threshold: float = 0.1
) -> Dict:
    """
    Create beat-level histograms filtered to only include IOIs between groove pulse positions.

    Parameters
    ----------
    filtered_patterns_dir : str
        Directory containing the section-anchored CSV files (6.2_filtered_patterns)
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots (6.7_anchored_beat_histograms)
    groove_pulse_csv : str
        Path to filtered_anchored_groove_pulse_histograms.csv from 6.6
    pulse_threshold : float
        Minimum onset_strength for a category to be included (default 0.1 = 10%)
        Categories below this threshold are excluded from plots and stats.

    Returns
    -------
    dict
        Dictionary with paths to saved files and statistics
    """
    print(f"\n  [Groove Pulse Beat Histograms] Creating filtered IOI histograms...")

    filtered_dir = Path(filtered_patterns_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    anchored_files = sorted(filtered_dir.glob('SecNo*_anchored.csv'))

    if not anchored_files:
        print(f"    Warning: No anchored CSV files found in {filtered_patterns_dir}")
        return {}

    # Group files by section number
    sections = {}
    for csv_file in anchored_files:
        parsed = parse_anchored_filename(csv_file.name)
        if parsed:
            sec_no = parsed['section_no']
            if sec_no not in sections:
                sections[sec_no] = {}
            full_metadata = read_anchored_csv_metadata(str(csv_file))
            sections[sec_no][parsed['pattern_length']] = {
                'path': csv_file,
                'metadata': parsed,
                'full_metadata': full_metadata
            }

    # Process each section and pattern length
    all_ioi_data = []
    output_files = {'csv_files': []}

    for sec_no in sorted(sections.keys()):
        for pattern_length in sorted(sections[sec_no].keys()):
            csv_info = sections[sec_no][pattern_length]
            csv_path = csv_info['path']
            metadata = csv_info['metadata']

            section_label = metadata['section_label']
            section_id = f"SecNo{sec_no}_{section_label}_L{pattern_length}"

            groove_positions = get_groove_pulse_positions(groove_pulse_csv, section_id)

            if not groove_positions:
                print(f"    SecNo{sec_no} L{pattern_length}: No groove positions, skipping")
                continue

            df_ioi = process_section_ioi(str(csv_path))

            if df_ioi.empty:
                continue

            df_ioi_filtered = filter_ioi_by_groove_positions(df_ioi, groove_positions, pattern_length)

            if df_ioi_filtered.empty:
                print(f"    SecNo{sec_no} L{pattern_length}: No IOIs between groove positions")
                continue

            df_ioi_filtered.insert(0, 'section_id', section_id)
            df_ioi_filtered.insert(2, 'section_no', sec_no)
            all_ioi_data.append(df_ioi_filtered)

            ratio = metadata['ratio_in_snippet']
            csv_filename = f"SecNo{sec_no}_L{pattern_length}_{section_label}_{ratio:.4f}_groove_pulse_ioi_data.csv"
            csv_output_path = output_path / csv_filename

            full_metadata = csv_info['full_metadata']
            with open(csv_output_path, 'w') as f:
                for key, value in full_metadata.items():
                    f.write(f"# {key}={value}\n")
                f.write(f"# groove_positions={len(groove_positions)}\n")
                df_ioi_filtered.to_csv(f, index=False)
            output_files['csv_files'].append(str(csv_output_path))
            print(f"    SecNo{sec_no} L{pattern_length}: {len(df_ioi_filtered)}/{len(df_ioi)} IOIs (groove filtered) -> {csv_filename}")

    if not all_ioi_data:
        print(f"    Warning: No groove pulse IOI data found")
        return {}

    df_all = pd.concat(all_ioi_data, ignore_index=True)

    aggregated_csv_path = output_path / f'{track_id}_groove_pulse_ioi_data.csv'
    df_all.to_csv(aggregated_csv_path, index=False)
    output_files['aggregated_ioi_csv'] = str(aggregated_csv_path)
    print(f"    Saved aggregated groove pulse IOI data: {aggregated_csv_path.name}")

    all_sec_nos = sorted(sections.keys())
    all_pattern_lengths = sorted(set(pl for sec in sections.values() for pl in sec.keys()))

    num_cols = len(all_sec_nos)
    num_rows = len(all_pattern_lengths)

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(8 * num_cols, 4 * num_rows),
                             squeeze=False)
    fig.suptitle(f'Groove Pulse Beat Histograms (Filtered IOI) — {track_id}',
                 fontsize=14, fontweight='bold', y=0.995)

    colors = {2: '#9B59B6', 4: '#1ABC9C'}

    category_order = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']
    category_to_ticks = {
        '1/16': 1, '1/8': 2, '3/16': 3, '1/4': 4,
        '6/16': 6, '2/4': 8, '4/4': 16
    }

    all_stats_rows = []

    for row_idx, pattern_length in enumerate(all_pattern_lengths):
        for col_idx, sec_no in enumerate(all_sec_nos):
            ax = axes[row_idx, col_idx]
            color = colors.get(pattern_length, '#3498DB')

            df_section = df_all[(df_all['section_no'] == sec_no) &
                               (df_all['pattern_length'] == pattern_length)]

            if df_section.empty:
                ax.text(0.5, 0.5, f'No groove pulse L{pattern_length} data\nfor SecNo{sec_no}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'SecNo{sec_no} — L{pattern_length}', fontsize=10, fontweight='bold')
                continue

            section_label = df_section['section_label'].iloc[0]
            ratio = df_section['ratio_in_snippet'].iloc[0]
            num_reps = df_section['num_repetitions'].iloc[0]

            category_stats = {}
            for cat in category_order:
                cat_data = df_section[df_section['ioi_category'] == cat]['ioi_exact_ticks']
                nominal_ticks = category_to_ticks[cat]

                if len(cat_data) > 0:
                    count = len(cat_data)
                    median_ioi = np.median(cat_data)
                    shifts = cat_data - nominal_ticks

                    if len(cat_data) > 1:
                        q75, q25 = np.percentile(cat_data, [75, 25])
                        iqr_scaled = (q75 - q25) * 1.5
                        shift_q75, shift_q25 = np.percentile(shifts, [75, 25])
                        iqr_shift = shift_q75 - shift_q25
                    else:
                        iqr_scaled = 0.0
                        iqr_shift = 0.0

                    median_shift = np.median(shifts)
                    category_stats[cat] = {
                        'count': count, 'median': median_ioi, 'median_shift': median_shift,
                        'iqr_scaled': iqr_scaled, 'iqr_shift': iqr_shift, 'nominal_ticks': nominal_ticks
                    }
                else:
                    category_stats[cat] = {
                        'count': 0, 'median': np.nan, 'median_shift': np.nan,
                        'iqr_scaled': 0.0, 'iqr_shift': 0.0, 'nominal_ticks': nominal_ticks
                    }

            counts = np.array([category_stats[cat]['count'] for cat in category_order])
            medians = np.array([category_stats[cat]['median'] for cat in category_order])
            iqrs_scaled = np.array([category_stats[cat]['iqr_scaled'] for cat in category_order])

            max_count = np.max(counts) if len(counts) > 0 else 1
            onset_strength = counts / max_count if max_count > 0 else counts

            # Apply pulse threshold - filter out categories below threshold
            pulse_mask = onset_strength >= pulse_threshold
            filtered_onset_strength = np.where(pulse_mask, onset_strength, 0)
            filtered_medians = np.where(pulse_mask, medians, np.nan)
            filtered_iqrs = np.where(pulse_mask, iqrs_scaled, 0)

            base_positions_log = np.array([np.log2(category_to_ticks[cat]) for cat in category_order])
            shifted_positions_log = base_positions_log.copy()

            for i, (cat, median_val) in enumerate(zip(category_order, filtered_medians)):
                if not np.isnan(median_val) and median_val > 0:
                    shifted_positions_log[i] = np.log2(median_val)

            bar_width = 0.15
            ax.bar(shifted_positions_log, filtered_onset_strength, width=bar_width,
                  color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

            for i, (cat, shifted_log, strength, iqr_val, median_val) in enumerate(
                zip(category_order, shifted_positions_log, filtered_onset_strength, filtered_iqrs, filtered_medians)):
                if strength > 0 and iqr_val > 0 and not np.isnan(median_val):
                    error_bar_y = strength * 0.9
                    log_upper = np.log2(median_val + iqr_val / 2)
                    log_lower = np.log2(max(0.1, median_val - iqr_val / 2))
                    iqr_log = (log_upper - log_lower) / 2
                    ax.errorbar(shifted_log, error_bar_y, xerr=iqr_log, fmt='none',
                               ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

            for i, (cat, shifted_log, strength, median_val) in enumerate(
                zip(category_order, shifted_positions_log, filtered_onset_strength, filtered_medians)):
                if strength > 0 and not np.isnan(median_val):
                    nominal_ticks = category_to_ticks[cat]
                    relative_deviation = median_val - nominal_ticks
                    label_text = f'{relative_deviation:.2f}'.replace('0.', '.').replace('-0.', '-.')
                    ax.text(shifted_log, strength, label_text, ha='center', va='bottom',
                           fontsize=7, fontweight='bold')

            tick_values = [1, 2, 3, 4, 6, 8, 16]
            tick_positions_log = [np.log2(v) for v in tick_values]
            tick_labels = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']

            ax.set_xticks(tick_positions_log)
            ax.set_xticklabels(tick_labels, fontsize=9)
            ax.set_ylabel('Onset Strength', fontsize=9)

            max_strength = np.max(onset_strength) if max_count > 0 else 1.0
            ax.set_ylim(0, max_strength * 1.2)

            ax2 = ax.twinx()
            ax2.set_ylim(0, max_count * 1.2)
            ax2.set_ylabel('Count', fontsize=9)

            title = f'SecNo{sec_no} — {section_label} — L{pattern_length} (Groove Pulse)'
            if num_reps:
                title += f' — {num_reps} reps'
            title += f' — {len(df_section)} IOIs'
            if ratio:
                title += f' — {ratio:.1%}'

            ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
            ax.grid(True, alpha=0.3, axis='y')

            for tick_log in tick_positions_log:
                ax.axvline(x=tick_log, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)

            ax.set_xlim(-0.5, 4.5)

            if row_idx == num_rows - 1:
                ax.set_xlabel('IOI Category (16th note ticks, log scale)', fontsize=9)

            # Save groove pulse beat histogram stats CSV
            csv_stats_rows = []
            for i, cat in enumerate(category_order):
                stats = category_stats[cat]
                passes_threshold = bool(pulse_mask[i])
                row_data = {
                    'ioi_category': cat,
                    'nominal_ticks': stats['nominal_ticks'],
                    'count': stats['count'],
                    'onset_strength': float(onset_strength[i]),
                    'passes_threshold': passes_threshold,
                    'median_ioi': stats['median'] if passes_threshold else np.nan,
                    'median_shift': stats['median_shift'] if passes_threshold else np.nan,
                    'iqr_shift': stats['iqr_shift'] if passes_threshold else np.nan,
                    'iqr_scaled': stats['iqr_scaled'] if passes_threshold else 0.0
                }
                csv_stats_rows.append(row_data)
                all_stats_rows.append({
                    'section_id': f'SecNo{sec_no}_{section_label}_L{pattern_length}',
                    'section_label': section_label,
                    'section_no': sec_no,
                    'pattern_length': pattern_length,
                    'num_repetitions': num_reps,
                    'ratio_in_snippet': ratio,
                    **row_data
                })

            df_stats = pd.DataFrame(csv_stats_rows)
            stats_csv_filename = f"SecNo{sec_no}_L{pattern_length}_{section_label}_{ratio:.4f}_groove_pulse_beat_histogram_stats.csv"
            stats_csv_path = output_path / stats_csv_filename

            full_metadata = sections[sec_no][pattern_length]['full_metadata']
            with open(stats_csv_path, 'w') as f:
                for key, value in full_metadata.items():
                    f.write(f"# {key}={value}\n")
                f.write(f"# pulse_threshold={pulse_threshold}\n")
                df_stats.to_csv(f, index=False)
            output_files['csv_files'].append(str(stats_csv_path))

    if all_stats_rows:
        df_all_stats = pd.DataFrame(all_stats_rows)
        aggregated_stats_csv_path = output_path / f'{track_id}_groove_pulse_beat_histograms.csv'
        df_all_stats.to_csv(aggregated_stats_csv_path, index=False)
        output_files['aggregated_stats_csv'] = str(aggregated_stats_csv_path)
        print(f"    Saved aggregated groove pulse beat histograms: {aggregated_stats_csv_path.name}")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='Bar: median IOI (ticks)'),
        Line2D([0], [0], color='black', linewidth=1.5, marker='|', markersize=10, label='Error bar: IQR × 1.5'),
        Line2D([0], [0], color='none', marker='$+.02$', markersize=12, markerfacecolor='black',
               markeredgecolor='none', label='Text: median shift (ticks)'),
    ]
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.02))

    output_png = output_path / f'{track_id}_groove_pulse_beat_histograms.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    output_files['png'] = str(output_png)
    print(f"    Processed {len(df_all)} groove pulse IOIs across {num_cols} sections")

    return output_files


def create_anchored_groove_pulse_beat_histograms_all_onsets(
    filtered_patterns_dir: str,
    track_id: str,
    output_dir: str,
    groove_pulse_csv: str
) -> Dict:
    """
    Create scatter plots showing all individual IOIs filtered by groove pulse positions.
    """
    print(f"\n  [Groove Pulse Beat Histograms - All Onsets] Creating filtered IOI scatter plots...")

    filtered_dir = Path(filtered_patterns_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    anchored_files = sorted(filtered_dir.glob('SecNo*_anchored.csv'))

    if not anchored_files:
        return {}

    sections = {}
    for csv_file in anchored_files:
        parsed = parse_anchored_filename(csv_file.name)
        if parsed:
            sec_no = parsed['section_no']
            if sec_no not in sections:
                sections[sec_no] = {}
            sections[sec_no][parsed['pattern_length']] = {
                'path': csv_file,
                'metadata': parsed
            }

    all_ioi_data = []
    for sec_no in sorted(sections.keys()):
        for pattern_length in sorted(sections[sec_no].keys()):
            csv_path = sections[sec_no][pattern_length]['path']
            metadata = sections[sec_no][pattern_length]['metadata']

            section_label = metadata['section_label']
            section_id = f"SecNo{sec_no}_{section_label}_L{pattern_length}"

            groove_positions = get_groove_pulse_positions(groove_pulse_csv, section_id)

            if not groove_positions:
                continue

            df_ioi = process_section_ioi(str(csv_path))
            if df_ioi.empty:
                continue

            df_ioi_filtered = filter_ioi_by_groove_positions(df_ioi, groove_positions, pattern_length)

            if not df_ioi_filtered.empty:
                df_ioi_filtered['section_no'] = sec_no
                all_ioi_data.append(df_ioi_filtered)

    if not all_ioi_data:
        print(f"    Warning: No groove pulse IOI data found")
        return {}

    df_all = pd.concat(all_ioi_data, ignore_index=True)

    all_sec_nos = sorted(sections.keys())
    all_pattern_lengths = sorted(set(pl for sec in sections.values() for pl in sec.keys()))

    num_cols = len(all_sec_nos)
    num_rows = len(all_pattern_lengths)

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(8 * num_cols, 3 * num_rows),
                             squeeze=False)
    fig.suptitle(f'Groove Pulse Beat Histograms — All IOIs — {track_id}',
                 fontsize=14, fontweight='bold', y=0.995)

    category_to_ticks = {
        '1/16': 1, '1/8': 2, '3/16': 3, '1/4': 4,
        '6/16': 6, '2/4': 8, '4/4': 16
    }

    for row_idx, pattern_length in enumerate(all_pattern_lengths):
        for col_idx, sec_no in enumerate(all_sec_nos):
            ax = axes[row_idx, col_idx]

            df_section = df_all[(df_all['section_no'] == sec_no) &
                               (df_all['pattern_length'] == pattern_length)]

            if df_section.empty:
                ax.text(0.5, 0.5, f'No groove pulse L{pattern_length} data\nfor SecNo{sec_no}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'SecNo{sec_no} — L{pattern_length}', fontsize=10, fontweight='bold')
                continue

            section_label = df_section['section_label'].iloc[0]
            ratio = df_section['ratio_in_snippet'].iloc[0]
            num_reps = df_section['num_repetitions'].iloc[0]

            ioi_values = df_section['ioi_exact_ticks'].values
            ioi_categories = df_section['ioi_category'].values

            valid_mask = ioi_values > 0
            ioi_values = ioi_values[valid_mask]
            ioi_categories = ioi_categories[valid_mask]

            if len(ioi_values) == 0:
                ax.text(0.5, 0.5, 'No valid IOIs',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                continue

            ioi_log = np.log2(ioi_values)

            np.random.seed(42 + sec_no + pattern_length)
            y_positions = np.random.uniform(0.3, 0.7, size=len(ioi_values))

            category_colors = {
                '1/16': '#E74C3C', '1/8': '#F39C12', '3/16': '#F1C40F', '1/4': '#2ECC71',
                '6/16': '#8E44AD', '2/4': '#3498DB', '4/4': '#2C3E50'
            }

            point_colors = [category_colors.get(cat, 'black') for cat in ioi_categories]

            ax.scatter(ioi_log, y_positions, marker='x', s=50,
                      c=point_colors, alpha=0.7, linewidths=1.5)

            tick_values = [1, 2, 3, 4, 6, 8, 16]
            tick_positions_log = [np.log2(v) for v in tick_values]
            tick_labels = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']

            ax.set_xticks(tick_positions_log)
            ax.set_xticklabels(tick_labels, fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_yticks([])

            title = f'SecNo{sec_no} — {section_label} — L{pattern_length} (Groove Pulse)'
            if num_reps:
                title += f' — {num_reps} reps'
            title += f' — {len(df_section)} IOIs'
            if ratio:
                title += f' — {ratio:.1%}'

            ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
            ax.grid(True, alpha=0.3, axis='x')

            for tick_log in tick_positions_log:
                ax.axvline(x=tick_log, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)

            ax.set_xlim(-0.5, 4.5)

            if row_idx == num_rows - 1:
                ax.set_xlabel('IOI (16th note ticks, log scale)', fontsize=9)

    from matplotlib.lines import Line2D
    category_colors = {
        '1/16': '#E74C3C', '1/8': '#F39C12', '3/16': '#F1C40F', '1/4': '#2ECC71',
        '6/16': '#8E44AD', '2/4': '#3498DB', '4/4': '#2C3E50'
    }
    legend_elements = [Line2D([0], [0], marker='x', color='w', markerfacecolor=color,
                              markeredgecolor=color, markersize=8, label=cat, linewidth=0)
                       for cat, color in category_colors.items()]
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    fig.legend(handles=legend_elements, loc='lower center', ncol=7, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.02))

    output_png = output_path / f'{track_id}_groove_pulse_beat_histograms_all_onsets.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    return {'png': str(output_png)}


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python anchored_beat_histograms.py <filtered_patterns_dir> <track_id> <output_dir> [groove_pulse_csv]')
        print('')
        print('Arguments:')
        print('  filtered_patterns_dir: 6.2_filtered_patterns directory')
        print('  track_id: Track identifier for plot titles')
        print('  output_dir: 6.7_anchored_beat_histograms directory')
        print('  groove_pulse_csv: (optional) Path to filtered_anchored_groove_pulse_histograms.csv from 6.6')
        sys.exit(1)

    filtered_patterns_dir = sys.argv[1]
    track_id = sys.argv[2]
    output_dir = sys.argv[3]

    create_anchored_beat_histograms(filtered_patterns_dir, track_id, output_dir)
    create_anchored_beat_histograms_all_onsets(filtered_patterns_dir, track_id, output_dir)

    # Run groove pulse filtered versions if groove_pulse_csv is provided
    if len(sys.argv) >= 5:
        groove_pulse_csv = sys.argv[4]
        create_anchored_groove_pulse_beat_histograms(filtered_patterns_dir, track_id, output_dir, groove_pulse_csv)
        create_anchored_groove_pulse_beat_histograms_all_onsets(filtered_patterns_dir, track_id, output_dir, groove_pulse_csv)
