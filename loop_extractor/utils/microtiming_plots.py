#!/usr/bin/env python3
"""
Microtiming deviation plots - Pattern-folded raster visualization (AP2 style).

This module creates pattern-folded microtiming plots showing onset deviations
from the metrical grid, similar to the AP2 notebook visualization.

Each plot shows:
- Pattern-folded 16th-note positions (x-axis)
- Deviation in milliseconds (y-axis)
- Multiple loops as colored lines
- Corrected and uncorrected versions

Environment: AEinBOX_13_3
Dependencies: matplotlib, pandas, numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def calculate_deviation_ms(df: pd.DataFrame, phase_col: str, grid_col: str) -> np.ndarray:
    """
    Calculate deviation in milliseconds from phase and grid time columns.

    Parameters
    ----------
    df : pd.DataFrame
        Comprehensive phases dataframe
    phase_col : str
        Column name for phase data
    grid_col : str
        Column name for grid time data

    Returns
    -------
    np.ndarray
        Deviation in milliseconds for each onset
    """
    # Get onset times and grid times
    onset_times = df['onset_time_uncorrected'].values
    grid_times = df[grid_col].values

    # Calculate deviation: onset_time - grid_time, convert to ms
    deviation_ms = (onset_times - grid_times) * 1000.0

    return deviation_ms


def fold_into_pattern(df: pd.DataFrame, pattern_length: int, phase_col: str, grid_col: str) -> Tuple[List, List, List]:
    """
    Fold bar data into pattern repeats (loops).

    Parameters
    ----------
    df : pd.DataFrame
        Comprehensive phases dataframe
    pattern_length : int
        Pattern length in bars (1, 2, or 4)
    phase_col : str
        Column name for phase data
    grid_col : str
        Column name for grid time data

    Returns
    -------
    tuple
        (loop_data, tick_positions, bar_ranges)
        - loop_data: List of arrays, one per loop, containing deviation_ms values
        - tick_positions: List of 16th-note positions for each data point
        - bar_ranges: List of (start_bar, end_bar) tuples for each loop
    """
    # Filter to rows with onsets (non-null phase)
    df_onsets = df[df[phase_col].notna()].copy()

    if len(df_onsets) == 0:
        return [], [], []

    # Get unique bars
    bars = df_onsets['bar_number'].unique()
    num_bars = len(bars)

    # Calculate number of complete loops
    num_loops = num_bars // pattern_length

    if num_loops == 0:
        return [], [], []

    loop_data = []
    tick_positions = []
    bar_ranges = []

    # Extract data for each loop
    for loop_idx in range(num_loops):
        start_bar = loop_idx * pattern_length
        end_bar = start_bar + pattern_length - 1

        # Get data for this loop
        loop_mask = (df_onsets['bar_number'] >= start_bar) & (df_onsets['bar_number'] <= end_bar)
        loop_df = df_onsets[loop_mask]

        if len(loop_df) == 0:
            continue

        # Calculate deviations
        onset_times = loop_df['onset_time_uncorrected'].values
        grid_times = loop_df[grid_col].values
        deviation_ms = (onset_times - grid_times) * 1000.0

        # Calculate 16th-note tick positions within pattern (1-based indexing)
        ticks = loop_df['tick_16th'].values
        bar_in_pattern = loop_df['bar_number'].values - start_bar
        tick_positions_pattern = bar_in_pattern * 16 + ticks + 1

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
        return

    # Color palette for loops
    colors = plt.cm.tab10(np.linspace(0, 1, len(loop_data)))

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

    # Add horizontal zero line (grid)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Grid')

    # Add vertical lines for bar boundaries (1-based: at 17, 33, 49, etc.)
    for bar_idx in range(1, pattern_length):
        ax.axvline(x=bar_idx * 16 + 1, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Calculate RMS
    all_deviations = np.concatenate(loop_data)
    rms_ms = np.sqrt(np.mean(all_deviations**2))

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


def create_microtiming_plots(
    comprehensive_csv: str,
    track_id: str,
    output_dir: str,
    snippet_info: Optional[Dict] = None
) -> str:
    """
    Create pattern-folded microtiming plots (AP2 style).

    Creates 5 pattern-folded plots showing onset deviations:
    1. Uncorrected - raw onset deviations
    2. Per-snippet - snippet-average correction
    3. Standard L=1 - 1-bar loop correction
    4. Standard L=2 - 2-bar loop correction
    5. Standard L=4 - 4-bar loop correction

    Each plot shows multiple loops as colored lines, folded into the pattern length.

    Parameters
    ----------
    comprehensive_csv : str
        Path to comprehensive phases CSV file
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots
    snippet_info : dict, optional
        Snippet information (not currently used in pattern-folded plots)

    Returns
    -------
    str
        Path to saved PDF file
    """
    print(f"\n  [Microtiming Plots] Creating pattern-folded raster plots...")

    # Load comprehensive CSV
    df = pd.read_csv(comprehensive_csv)

    # Define methods with their pattern lengths
    methods = [
        ('Uncorrected', 'phase_uncorrected', 'grid_time_uncorrected', 4),
        ('Per-Snippet', 'phase_per_snippet', 'grid_time_per_snippet', 4),
        ('Standard L=1', 'phase_standard_L1(L=1)', 'grid_time_standard_L1(L=1)', 1),
        ('Standard L=2', 'phase_standard_L2(L=2)', 'grid_time_standard_L2(L=2)', 2),
        ('Standard L=4', 'phase_standard_L4(L=4)', 'grid_time_standard_L4(L=4)', 4),
    ]

    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 18))
    fig.suptitle(f'Microtiming Deviation Plots — {track_id}', fontsize=14, fontweight='bold')

    # Plot each method
    for idx, (method_title, phase_col, grid_col, pattern_length) in enumerate(methods):
        ax = axes[idx]

        # Check if columns exist
        if phase_col not in df.columns or grid_col not in df.columns:
            ax.text(0.5, 0.5, f'No data for {method_title}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{method_title}', fontsize=11, fontweight='bold')
            print(f"    {method_title}: No data (missing columns)")
            continue

        # Fold data into pattern
        loop_data, tick_positions, bar_ranges = fold_into_pattern(
            df, pattern_length, phase_col, grid_col
        )

        # Plot pattern-folded data
        show_xlabel = (idx == len(methods) - 1)
        rms_ms = plot_pattern_folded(
            ax, loop_data, tick_positions, bar_ranges,
            pattern_length, method_title, show_xlabel
        )

        if rms_ms is not None:
            print(f"    {method_title}: {len(loop_data)} loops, L={pattern_length}, RMS = {rms_ms:.2f} ms")
        else:
            print(f"    {method_title}: No data")

    plt.tight_layout()

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / f'{track_id}_microtiming_plots.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf}")

    output_png = output_dir / f'{track_id}_microtiming_plots.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png}")

    plt.close()

    return str(output_pdf)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python microtiming_plots.py <comprehensive_csv> <track_id> <output_dir>')
        sys.exit(1)

    comprehensive_csv = sys.argv[1]
    track_id = sys.argv[2]
    output_dir = sys.argv[3]

    create_microtiming_plots(comprehensive_csv, track_id, output_dir)
