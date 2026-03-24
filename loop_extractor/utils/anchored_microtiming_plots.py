#!/usr/bin/env python3
"""
Anchored Microtiming deviation plots - Pattern-folded raster visualization.

NEW METHOD: Uses per-section and per-stem anchored CSVs from 6.2_filtered_patterns.

This module creates pattern-folded microtiming plots showing onset deviations
from the metrical grid, using the new section-anchored correction method.

Each plot shows:
- Pattern-folded 16th-note positions (x-axis)
- Deviation in milliseconds (y-axis)
- Multiple loops as colored lines (pattern repetitions)

Input files: 6.2_filtered_patterns/{stem}/SecNo{N}_L{L}_{section}_{ratio}_anchored.csv

Environment: AEinBOX_13_3
Dependencies: matplotlib, pandas, numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import re
import glob as glob_module


def parse_anchored_csv_metadata(csv_path: Path) -> Dict:
    """
    Parse metadata from anchored CSV header comments.

    Parameters
    ----------
    csv_path : Path
        Path to anchored CSV file

    Returns
    -------
    dict
        Metadata extracted from header comments
    """
    metadata = {}
    with open(csv_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            # Parse key=value from comment lines
            line = line.strip('# \n')
            if '=' in line:
                key, value = line.split('=', 1)
                metadata[key.strip()] = value.strip()
    return metadata


def calculate_deviation_ms(df: pd.DataFrame, phase_col: str, grid_col: str) -> np.ndarray:
    """
    Calculate deviation in milliseconds from phase and grid time columns.

    Parameters
    ----------
    df : pd.DataFrame
        Anchored CSV dataframe
    phase_col : str
        Column name for phase data
    grid_col : str
        Column name for grid time data

    Returns
    -------
    np.ndarray
        Deviation in milliseconds for each onset
    """
    # Get onset times and grid times (using anchored columns)
    onset_times = df['onset_time'].values
    grid_times = df[grid_col].values

    # Calculate deviation: onset_time - grid_time, convert to ms
    deviation_ms = (onset_times - grid_times) * 1000.0

    return deviation_ms


def fold_into_pattern_anchored(df: pd.DataFrame, pattern_length: int) -> Tuple[List, List, List]:
    """
    Fold bar data into pattern repeats (loops) for anchored CSVs.

    Parameters
    ----------
    df : pd.DataFrame
        Anchored CSV dataframe with columns: bar_number, tick_16th, onset_time, grid_time, pattern_index
    pattern_length : int
        Pattern length in bars (1, 2, or 4)

    Returns
    -------
    tuple
        (loop_data, tick_positions, bar_ranges)
        - loop_data: List of arrays, one per loop, containing deviation_ms values
        - tick_positions: List of 16th-note positions for each data point
        - bar_ranges: List of (start_bar, end_bar) tuples for each loop
    """
    # Filter to rows with actual onsets (non-null onset_time)
    df_onsets = df[df['onset_time'].notna()].copy()

    if len(df_onsets) == 0:
        return [], [], []

    # Get unique pattern indices (loops)
    pattern_indices = sorted(df_onsets['pattern_index'].unique())

    if len(pattern_indices) == 0:
        return [], [], []

    loop_data = []
    tick_positions = []
    bar_ranges = []

    # Extract data for each pattern repetition (loop)
    for pattern_idx in pattern_indices:
        loop_df = df_onsets[df_onsets['pattern_index'] == pattern_idx]

        if len(loop_df) == 0:
            continue

        # Calculate deviations using anchored onset_time and grid_time
        onset_times = loop_df['onset_time'].values
        grid_times = loop_df['grid_time'].values
        deviation_ms = (onset_times - grid_times) * 1000.0

        # Calculate 16th-note tick positions within pattern (1-based indexing)
        # For L=1: bar_number IS the pattern index, so % 1 = 0 (all fold to same position)
        # For L>1: bar_number is 0 to L-1 within each pattern, % L keeps it the same
        ticks = loop_df['tick_16th'].values
        bar_in_pattern = loop_df['bar_number'].values % pattern_length
        tick_positions_pattern = bar_in_pattern * 16 + ticks + 1

        # Get bar range (global bar numbers if available)
        if 'bar_number_global' in loop_df.columns:
            bar_global = loop_df['bar_number_global'].values
            start_bar = int(bar_global.min())
            end_bar = int(bar_global.max())
        else:
            start_bar = int(pattern_idx * pattern_length)
            end_bar = start_bar + pattern_length - 1

        loop_data.append(deviation_ms)
        tick_positions.append(tick_positions_pattern)
        bar_ranges.append((start_bar, end_bar))

    return loop_data, tick_positions, bar_ranges


def plot_pattern_folded(ax, loop_data: List, tick_positions: List, bar_ranges: List,
                        pattern_length: int, title: str, show_xlabel: bool = False):
    """
    Create a pattern-folded plot with multiple loops.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    loop_data : list
        List of deviation arrays, one per loop
    tick_positions : list
        List of tick position arrays, one per loop
    bar_ranges : list
        List of (start_bar, end_bar) tuples
    pattern_length : int
        Pattern length in bars
    title : str
        Plot title
    show_xlabel : bool
        Whether to show x-axis label
    """
    if len(loop_data) == 0:
        ax.text(0.5, 0.5, 'No data available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return None

    # Color palette for loops
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(loop_data), 1)))

    # Plot each loop
    for i, (deviations, ticks, (start_bar, end_bar)) in enumerate(zip(loop_data, tick_positions, bar_ranges)):
        # Only plot finite values (filter out NaN/inf)
        mask = np.isfinite(deviations)
        if mask.any():
            xs = ticks[mask]
            ys = deviations[mask]

            # Sort by tick position to ensure consecutive events are connected
            order_idx = np.argsort(xs)
            xs_sorted = xs[order_idx]
            ys_sorted = ys[order_idx]

            # Plot line connecting all points in this loop (single call with marker parameter)
            ax.plot(xs_sorted, ys_sorted, linewidth=1.5, alpha=0.7, color=colors[i],
                   marker='o', markersize=4, markerfacecolor=colors[i], markeredgecolor='none',
                   label=f'Loop {i+1} (Bars {start_bar}-{end_bar})')

    # Calculate and plot mean/median deviation curves (dashed lines)
    # Collect all (tick, deviation) pairs across all loops
    all_ticks = []
    all_devs = []
    for deviations, ticks in zip(loop_data, tick_positions):
        mask = np.isfinite(deviations)
        if mask.any():
            all_ticks.extend(ticks[mask])
            all_devs.extend(deviations[mask])

    if all_ticks:
        # Group by tick position and calculate mean/median
        tick_to_devs = {}
        for t, d in zip(all_ticks, all_devs):
            if t not in tick_to_devs:
                tick_to_devs[t] = []
            tick_to_devs[t].append(d)

        sorted_ticks = sorted(tick_to_devs.keys())
        mean_devs = [np.mean(tick_to_devs[t]) for t in sorted_ticks]
        median_devs = [np.median(tick_to_devs[t]) for t in sorted_ticks]

        # Plot mean curve (dashed red line)
        ax.plot(sorted_ticks, mean_devs, linewidth=2.5, linestyle='--', color='red',
               marker='s', markersize=5, markerfacecolor='red', markeredgecolor='none',
               alpha=0.9, label='Mean', zorder=10)

        # Plot median curve (dashed blue line)
        ax.plot(sorted_ticks, median_devs, linewidth=2.5, linestyle='--', color='blue',
               marker='D', markersize=5, markerfacecolor='blue', markeredgecolor='none',
               alpha=0.9, label='Median', zorder=10)

    # Add horizontal zero line (grid)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='Grid')

    # Add vertical lines for bar boundaries (1-based: at 17, 33, 49, etc.)
    for bar_idx in range(1, pattern_length):
        ax.axvline(x=bar_idx * 16 + 1, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Calculate RMS
    all_deviations = np.concatenate(loop_data)
    finite_deviations = all_deviations[np.isfinite(all_deviations)]
    if len(finite_deviations) > 0:
        rms_ms = np.sqrt(np.mean(finite_deviations**2))
    else:
        rms_ms = 0.0

    # Formatting
    ax.set_ylabel('Deviation (ms)', fontsize=10)
    ax.set_title(f'{title} (RMS: {rms_ms:.2f} ms)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    # Set x-axis limits (1-based indexing)
    ax.set_xlim(0, pattern_length * 16 + 2)

    # Set x-axis ticks at every 16th-note position (1-based: 1 to pattern_length*16)
    ax.set_xticks(range(1, pattern_length * 16 + 1))
    ax.tick_params(axis='x', labelsize=8)

    if show_xlabel:
        ax.set_xlabel('16th index within pattern (folded)', fontsize=10)

    return rms_ms


def create_anchored_microtiming_plots(
    filtered_patterns_dir: str,
    track_id: str,
    stem: str
) -> Optional[str]:
    """
    Create pattern-folded microtiming plots for all section/L combinations in ONE file.

    NEW METHOD: Uses anchored CSVs from 6.2_filtered_patterns/{stem}/ folder.
    Creates ONE PDF/PNG with all sections as subplots.

    Parameters
    ----------
    filtered_patterns_dir : str
        Path to 6.2_filtered_patterns folder (e.g., track_dir/6.2_filtered_patterns)
    track_id : str
        Track identifier for plot title
    stem : str
        Stem name (e.g., 'drums', 'bass', 'vocals', 'piano', 'other')

    Returns
    -------
    str or None
        Path to saved PDF file, or None if no data
    """
    filtered_patterns_dir = Path(filtered_patterns_dir)
    stem_dir = filtered_patterns_dir / stem

    if not stem_dir.exists():
        print(f"  [Anchored Microtiming] Stem directory not found: {stem_dir}")
        return None

    # Find all anchored CSV files: SecNo{N}_L{L}_{section}_{ratio}_anchored.csv
    anchored_csvs = sorted(stem_dir.glob('SecNo*_L*_*_anchored.csv'))

    if not anchored_csvs:
        print(f"  [Anchored Microtiming] No anchored CSVs found in {stem_dir}")
        return None

    print(f"\n  [Anchored Microtiming] Creating plots for {stem} ({len(anchored_csvs)} files)...")

    # Collect all plot data first
    plot_data_list = []

    for csv_path in anchored_csvs:
        # Parse filename: SecNo2_L4_chorus_0.7040_anchored.csv
        filename = csv_path.stem  # SecNo2_L4_chorus_0.7040_anchored
        match = re.match(r'SecNo(\d+)_L(\d+)_([^_]+)_([0-9.]+)_anchored', filename)

        if not match:
            print(f"    Skipping unrecognized file: {filename}")
            continue

        sec_no = int(match.group(1))
        pattern_length = int(match.group(2))
        section_label = match.group(3)
        ratio = float(match.group(4))

        # Read metadata from CSV header
        metadata = parse_anchored_csv_metadata(csv_path)

        # Read CSV data (skip comment lines)
        df = pd.read_csv(csv_path, comment='#')

        if len(df) == 0:
            print(f"    {filename}: No data")
            continue

        # Fold into pattern
        loop_data, tick_positions, bar_ranges = fold_into_pattern_anchored(df, pattern_length)

        if len(loop_data) == 0:
            print(f"    {filename}: No onset data")
            continue

        # Store plot data
        plot_data_list.append({
            'filename': filename,
            'sec_no': sec_no,
            'pattern_length': pattern_length,
            'section_label': section_label,
            'ratio': ratio,
            'metadata': metadata,
            'loop_data': loop_data,
            'tick_positions': tick_positions,
            'bar_ranges': bar_ranges
        })

    if not plot_data_list:
        print(f"  [Anchored Microtiming] No valid data for {stem}")
        return None

    # Create figure with all subplots
    n_plots = len(plot_data_list)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    fig.suptitle(f'Anchored Microtiming Plots — {track_id} — {stem.capitalize()}', fontsize=14, fontweight='bold')

    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]

    # Plot each section
    for idx, data in enumerate(plot_data_list):
        ax = axes[idx]

        # Build title with metadata
        tempo = data['metadata'].get('mean_section_tempo', '?')
        n_reps = data['metadata'].get('no_of_repetitions', len(data['loop_data']))
        title = f"{data['section_label'].capitalize()} (Sec#{data['sec_no']}) L={data['pattern_length']} — {tempo} BPM, {n_reps} reps, ratio={data['ratio']:.2f}"

        # Plot
        show_xlabel = (idx == n_plots - 1)
        rms_ms = plot_pattern_folded(
            ax, data['loop_data'], data['tick_positions'], data['bar_ranges'],
            data['pattern_length'], title, show_xlabel=show_xlabel
        )

        if rms_ms is not None:
            print(f"    {data['filename']}: {len(data['loop_data'])} loops, L={data['pattern_length']}, RMS = {rms_ms:.2f} ms")
        else:
            print(f"    {data['filename']}: No data")

    plt.tight_layout()

    # Save in stem folder - ONE file per stem
    output_pdf = stem_dir / f'{track_id}_anchored_microtiming_plots.pdf'
    output_png = stem_dir / f'{track_id}_anchored_microtiming_plots.png'

    plt.savefig(output_pdf, bbox_inches='tight')
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [Anchored Microtiming] Saved: {output_pdf.name} ({n_plots} subplots)")
    return str(output_pdf)


def create_all_anchored_microtiming_plots(
    filtered_patterns_dir: str,
    track_id: str,
    stems: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Create anchored microtiming plots for all stems.

    Parameters
    ----------
    filtered_patterns_dir : str
        Path to 6.2_filtered_patterns folder
    track_id : str
        Track identifier
    stems : list, optional
        List of stems to process. If None, auto-detect from directory.

    Returns
    -------
    dict
        Dictionary mapping stem name to saved PDF path (one file per stem)
    """
    filtered_patterns_dir = Path(filtered_patterns_dir)

    # Auto-detect stems if not provided
    if stems is None:
        stems = []
        for subdir in filtered_patterns_dir.iterdir():
            if subdir.is_dir() and subdir.name in ['drums', 'bass', 'vocals', 'piano', 'other']:
                stems.append(subdir.name)
        stems = sorted(stems)

    if not stems:
        print(f"  [Anchored Microtiming] No stem directories found in {filtered_patterns_dir}")
        return {}

    results = {}
    for stem in stems:
        saved_file = create_anchored_microtiming_plots(
            str(filtered_patterns_dir), track_id, stem
        )
        if saved_file:
            results[stem] = saved_file

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python anchored_microtiming_plots.py <filtered_patterns_dir> <track_id> [stem]')
        print('Example: python anchored_microtiming_plots.py /path/to/6.2_filtered_patterns "Track Name" drums')
        sys.exit(1)

    filtered_patterns_dir = sys.argv[1]
    track_id = sys.argv[2]
    stem = sys.argv[3] if len(sys.argv) > 3 else None

    if stem:
        create_anchored_microtiming_plots(filtered_patterns_dir, track_id, stem)
    else:
        create_all_anchored_microtiming_plots(filtered_patterns_dir, track_id)
