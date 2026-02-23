#!/usr/bin/env python3
"""
Anchored Rhythm Histograms - Create rhythm pattern visualizations from section-anchored data.

This module reads anchored CSV files from 6.1_anchoring or 6.2_filtered_patterns
and creates rhythm pattern histograms showing binary patterns where onset strength
>= 50% is displayed as 1.0 and < 50% as 0.5.

Step 7 in the pipeline.

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List
import re
import sys


def parse_csv_metadata(csv_path: Path) -> Dict[str, str]:
    """
    Parse metadata from comment lines at the top of CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file

    Returns
    -------
    Dict[str, str]
        Dictionary of metadata key-value pairs
    """
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#'):
                match = re.match(r'^#\s*(\w+)=(.+)$', line.strip())
                if match:
                    metadata[match.group(1)] = match.group(2)
            else:
                break
    return metadata


def extract_section_info(filename: str) -> Dict:
    """
    Extract section information from anchored CSV filename.

    Parameters
    ----------
    filename : str
        Filename like "SecNo1_L2_pre-chorus_0.2972_anchored.csv"

    Returns
    -------
    Dict with sec_num, pattern_len, section_label
    """
    base = filename.replace('_anchored.csv', '')
    parts = base.split('_')

    sec_num = parts[0] if parts else ''
    pattern_len = 2

    if len(parts) > 1 and parts[1].startswith('L'):
        try:
            pattern_len = int(parts[1][1:])
        except ValueError:
            pass

    section_label = ''
    if len(parts) > 2:
        section_label = '_'.join(parts[2:-1]) if len(parts) > 3 else parts[2]

    return {
        'sec_num': sec_num,
        'pattern_len': pattern_len,
        'section_label': section_label
    }


def calculate_onset_strength_per_position(
    df: pd.DataFrame,
    pattern_len: int
) -> Dict[int, float]:
    """
    Calculate onset strength (ratio of patterns with onset) for each 16th note position.

    Parameters
    ----------
    df : pd.DataFrame
        Anchored CSV data with columns: bar_number, tick_16th, onset_time, phase
    pattern_len : int
        Pattern length in bars (2 or 4)

    Returns
    -------
    Dict[int, float]
        Dictionary mapping position (1-based) to onset strength (0-1)
    """
    num_positions = pattern_len * 16

    # Filter to rows with actual onsets
    onset_data = df[df['phase'].notna()].copy()

    if len(onset_data) == 0:
        return {}

    # Get number of complete patterns
    n_bars = int(df['bar_number'].max()) + 1
    n_patterns = n_bars // pattern_len

    if n_patterns == 0:
        return {}

    # Calculate onset strength for each position
    strengths = {}
    for pos in range(num_positions):
        bar_in_pattern = pos // 16
        tick_16th = pos % 16

        # Count how many patterns have an onset at this position
        onset_count = 0
        for pattern_idx in range(n_patterns):
            bar_number = pattern_idx * pattern_len + bar_in_pattern
            bar_tick_data = onset_data[
                (onset_data['bar_number'] == bar_number) &
                (onset_data['tick_16th'] == tick_16th)
            ]
            if len(bar_tick_data) > 0:
                onset_count += 1

        strengths[pos + 1] = onset_count / n_patterns  # 1-based position

    return strengths


def calculate_median_phase_per_position(
    df: pd.DataFrame,
    pattern_len: int
) -> Dict[int, float]:
    """
    Calculate median phase for each 16th note position across all patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Anchored CSV data
    pattern_len : int
        Pattern length in bars

    Returns
    -------
    Dict[int, float]
        Dictionary mapping position (1-based) to median phase (0-1)
    """
    num_positions = pattern_len * 16
    onset_data = df[df['phase'].notna()].copy()

    if len(onset_data) == 0:
        return {}

    n_bars = int(df['bar_number'].max()) + 1
    n_patterns = n_bars // pattern_len

    if n_patterns == 0:
        return {}

    median_phases = {}
    for pos in range(num_positions):
        bar_in_pattern = pos // 16
        tick_16th = pos % 16

        phases = []
        for pattern_idx in range(n_patterns):
            bar_number = pattern_idx * pattern_len + bar_in_pattern
            bar_tick_data = onset_data[
                (onset_data['bar_number'] == bar_number) &
                (onset_data['tick_16th'] == tick_16th)
            ]
            if len(bar_tick_data) > 0:
                phases.append(bar_tick_data['phase'].iloc[0])

        if phases:
            median_phases[pos + 1] = np.median(phases)

    return median_phases


def calculate_iqr_per_position(
    df: pd.DataFrame,
    pattern_len: int
) -> Dict[int, float]:
    """
    Calculate IQR of phases for each 16th note position (in 16th note units).

    Parameters
    ----------
    df : pd.DataFrame
        Anchored CSV data
    pattern_len : int
        Pattern length in bars

    Returns
    -------
    Dict[int, float]
        Dictionary mapping position (1-based) to IQR in 16th note units
    """
    num_positions = pattern_len * 16
    onset_data = df[df['phase'].notna()].copy()

    if len(onset_data) == 0:
        return {}

    n_bars = int(df['bar_number'].max()) + 1
    n_patterns = n_bars // pattern_len

    if n_patterns == 0:
        return {}

    iqr_values = {}
    for pos in range(num_positions):
        bar_in_pattern = pos // 16
        tick_16th = pos % 16

        phases = []
        for pattern_idx in range(n_patterns):
            bar_number = pattern_idx * pattern_len + bar_in_pattern
            bar_tick_data = onset_data[
                (onset_data['bar_number'] == bar_number) &
                (onset_data['tick_16th'] == tick_16th)
            ]
            if len(bar_tick_data) > 0:
                phases.append(bar_tick_data['phase'].iloc[0])

        if len(phases) >= 2:
            q75, q25 = np.percentile(phases, [75, 25])
            iqr = q75 - q25
            iqr_16th = iqr * 16  # Convert to 16th note units
            iqr_values[pos + 1] = iqr_16th

    return iqr_values


def plot_rhythm_pattern_subplot(
    ax: plt.Axes,
    section_data: Dict,
    color: str
) -> None:
    """
    Plot a single rhythm pattern subplot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    section_data : Dict
        Section data including df, metadata, pattern_len, etc.
    color : str
        Color for the bars
    """
    df = section_data['df']
    pattern_len = section_data['pattern_len']
    metadata = section_data.get('metadata', {})
    sec_num = section_data.get('sec_num', '')
    section_label = section_data.get('section_label', '')

    num_positions = pattern_len * 16

    # Calculate onset strength for each position
    strengths = calculate_onset_strength_per_position(df, pattern_len)
    median_phases = calculate_median_phase_per_position(df, pattern_len)
    iqr_values = calculate_iqr_per_position(df, pattern_len)

    if not strengths:
        ax.text(0.5, 0.5, 'No onset data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title(f'{sec_num} L{pattern_len} {section_label}\n(No data)',
                     fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Convert to pattern values (binary-ish: >50% -> 1.0, 0<x<=50% -> 0.5)
    pattern_values = np.zeros(num_positions)
    position_exists = np.zeros(num_positions, dtype=bool)

    for pos, strength in strengths.items():
        idx = pos - 1
        if strength > 0.5:
            pattern_values[idx] = 1.0
            position_exists[idx] = True
        elif strength > 0:
            pattern_values[idx] = 0.5
            position_exists[idx] = True

    # X-axis: base positions (1-based)
    base_positions = np.arange(1, num_positions + 1)

    # Calculate shifted positions based on median phase
    shifted_positions = base_positions.copy().astype(float)
    median_phases_array = np.full(num_positions, np.nan)
    iqr_array = np.full(num_positions, np.nan)

    for pos, median_phase in median_phases.items():
        idx = pos - 1
        median_phases_array[idx] = median_phase
        bar_number = idx // 16
        shifted_positions[idx] = bar_number * 16 + (median_phase * 16) + 1

    for pos, iqr in iqr_values.items():
        iqr_array[pos - 1] = iqr

    # Plot bars
    bar_width = 0.8
    for i in range(num_positions):
        if position_exists[i] and pattern_values[i] > 0:
            ax.bar(shifted_positions[i], pattern_values[i], width=bar_width,
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add error bars (IQR)
    for i in range(num_positions):
        if not np.isnan(iqr_array[i]) and iqr_array[i] > 0 and pattern_values[i] > 0:
            error_bar_y = pattern_values[i] * 0.9
            ax.errorbar(shifted_positions[i], error_bar_y,
                        xerr=iqr_array[i], fmt='none',
                        ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

    # Add relative median phase labels
    for i in range(num_positions):
        if pattern_values[i] > 0 and not np.isnan(median_phases_array[i]):
            tick_within_bar = i % 16
            grid_phase = tick_within_bar / 16.0
            phase_diff = median_phases_array[i] - grid_phase
            relative_phase = phase_diff / (1.0 / 16.0)

            label_text = f'{relative_phase:.2f}'.replace('0.', '.').replace('-0.', '-.')
            ax.text(shifted_positions[i], pattern_values[i], label_text,
                    ha='center', va='bottom', fontsize=6, rotation=0)

    # Set axis limits
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0.5, 1.0])
    ax.set_xlim(0.5, num_positions + 0.5)
    ax.set_xticks(base_positions)
    ax.set_xticklabels(base_positions, fontsize=8)

    # Add vertical center lines (blue, bar height)
    y_limits = ax.get_ylim()
    y_range = y_limits[1] - y_limits[0]
    for i in range(num_positions):
        if pattern_values[i] > 0 and not np.isnan(median_phases_array[i]):
            ymax_fraction = (pattern_values[i] - y_limits[0]) / y_range
            ax.axvline(x=shifted_positions[i], ymin=0, ymax=ymax_fraction,
                       color='blue', linestyle='-', linewidth=1.5, alpha=0.7, zorder=10)

    # Add bar boundary lines (red dashed)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=10)
    for bar_idx in range(1, pattern_len):
        ax.axvline(x=bar_idx * 16 + 1, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.5, zorder=10)

    # Add grid lines at expected positions (gray dotted)
    for i in range(num_positions):
        ax.axvline(x=base_positions[i], color='gray', linestyle=':',
                   linewidth=0.8, alpha=0.4, zorder=1)

    ax.grid(True, alpha=0.3, axis='y')

    # Build title
    n_bars = int(df['bar_number'].max()) + 1 if 'bar_number' in df.columns else 0
    n_patterns = n_bars // pattern_len
    n_reps = metadata.get('no_of_repetitions', str(n_patterns))
    occupied = int(np.sum(position_exists))

    title = f'{sec_num} L{pattern_len} {section_label}'
    title += f'\n{n_reps} reps | {occupied}/{num_positions} positions'

    ax.set_title(title, fontsize=9, fontweight='bold', pad=10)


def create_anchored_rhythm_histograms(
    anchoring_dir: str,
    output_dir: Optional[str] = None,
    track_id: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Create rhythm pattern histograms from section-anchored data.

    Creates a combined figure with all sections showing rhythm patterns
    using the L2/L4 two-row layout aligned by section number.

    Parameters
    ----------
    anchoring_dir : str
        Path to 6.1_anchoring or 6.2_filtered_patterns directory
    output_dir : str, optional
        Output directory (defaults to anchoring_dir)
    track_id : str, optional
        Track identifier for filenames
    verbose : bool
        Print progress messages

    Returns
    -------
    Dict
        Dictionary with output file paths
    """
    anchoring_path = Path(anchoring_dir)
    output_path = Path(output_dir) if output_dir else anchoring_path

    if not anchoring_path.exists():
        if verbose:
            print(f"  Directory not found: {anchoring_dir}")
        return {}

    # Find all anchored CSV files
    anchored_files = sorted([
        f for f in anchoring_path.glob("*_anchored.csv")
        if not f.name.startswith('._')
    ])

    if not anchored_files:
        if verbose:
            print(f"  No anchored CSV files found")
        return {}

    # Extract track_id from directory name if not provided
    if track_id is None:
        track_id = anchoring_path.parent.name

    if verbose:
        print(f"  Found {len(anchored_files)} anchored sections")

    # Load all section data
    sections_data = []
    for csv_file in anchored_files:
        try:
            metadata = parse_csv_metadata(csv_file)
            df = pd.read_csv(csv_file, comment='#', encoding='utf-8', encoding_errors='replace')

            section_info = extract_section_info(csv_file.name)

            sections_data.append({
                'csv_file': csv_file,
                'df': df,
                'metadata': metadata,
                'sec_num': section_info['sec_num'],
                'pattern_len': section_info['pattern_len'],
                'section_label': section_info['section_label']
            })

            if verbose:
                print(f"    Loaded: {csv_file.name}")

        except Exception as e:
            if verbose:
                print(f"    Error loading {csv_file.name}: {e}")

    if not sections_data:
        if verbose:
            print("  No valid section data found")
        return {}

    # Group by pattern length (L2 and L4)
    l2_by_sec = {}
    l4_by_sec = {}
    for s in sections_data:
        sec_num = s['sec_num']
        if s['pattern_len'] == 2:
            l2_by_sec[sec_num] = s
        elif s['pattern_len'] == 4:
            l4_by_sec[sec_num] = s

    all_sec_nums = sorted(set(l2_by_sec.keys()) | set(l4_by_sec.keys()))
    n_cols = max(len(all_sec_nums), 1)
    n_rows = 2

    if verbose:
        print(f"  Section numbers: {all_sec_nums}")
        print(f"  L2 sections: {list(l2_by_sec.keys())}")
        print(f"  L4 sections: {list(l4_by_sec.keys())}")

    # Colors for L2 and L4
    l2_color = '#F39C12'  # Orange
    l4_color = '#2ECC71'  # Green

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Row 0: L2 sections
    for col, sec_num in enumerate(all_sec_nums):
        ax = axes[0, col]
        if sec_num in l2_by_sec:
            plot_rhythm_pattern_subplot(ax, l2_by_sec[sec_num], l2_color)
        else:
            section_label = l4_by_sec.get(sec_num, {}).get('section_label', '')
            ax.text(0.5, 0.5, 'No L2 data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{sec_num} L2 {section_label}\n(No L2 data)',
                         fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    # Row 1: L4 sections
    for col, sec_num in enumerate(all_sec_nums):
        ax = axes[1, col]
        if sec_num in l4_by_sec:
            plot_rhythm_pattern_subplot(ax, l4_by_sec[sec_num], l4_color)
        else:
            section_label = l2_by_sec.get(sec_num, {}).get('section_label', '')
            ax.text(0.5, 0.5, 'No L4 data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{sec_num} L4 {section_label}\n(No L4 data)',
                         fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    # Add labels
    for col in range(n_cols):
        axes[1, col].set_xlabel('16th Note Position', fontsize=10, fontweight='bold')
    for row in range(n_rows):
        axes[row, 0].set_ylabel('Pattern Level', fontsize=10, fontweight='bold')

    fig.suptitle(f'{track_id} - Anchored Rhythm Patterns\nRow 1: L2 | Row 2: L4',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()

    # Save PNG
    output_path.mkdir(parents=True, exist_ok=True)
    output_png = output_path / f'{track_id}_anchored_rhythm_patterns.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')

    if verbose:
        print(f"  Saved: {output_png.name}")

    plt.close()

    return {
        'anchored_rhythm_patterns_png': str(output_png)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python anchored_rhythm_histograms.py <anchoring_dir> [track_id]")
        print("Example: python anchored_rhythm_histograms.py /path/to/6.1_anchoring")
        sys.exit(1)

    anchoring_dir = sys.argv[1]
    track_id = sys.argv[2] if len(sys.argv) > 2 else None

    create_anchored_rhythm_histograms(anchoring_dir, track_id=track_id)
