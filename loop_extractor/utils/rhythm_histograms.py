#!/usr/bin/env python3
"""
Rhythm Histograms - Create rhythm histograms showing onset distributions.

This module creates rhythm histograms showing the distribution of onsets
across 16th-note positions within patterns for different correction methods.

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def extract_rhythm_histogram(
    df: pd.DataFrame,
    phase_col: str,
    pattern_length: int
) -> np.ndarray:
    """
    Extract rhythm histogram from comprehensive phases dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Comprehensive phases dataframe
    phase_col : str
        Column name for phase data (e.g., 'phase_uncorrected', 'phase_standard_L1(L=1)')
    pattern_length : int
        Pattern length in bars (1, 2, or 4)

    Returns
    -------
    np.ndarray
        Histogram counts for each 16th-note position (length: pattern_length * 16)
    """
    # Filter to rows with actual onsets (non-null phase)
    df_onsets = df[df[phase_col].notna()].copy()

    if len(df_onsets) == 0:
        return np.zeros(pattern_length * 16)

    # Calculate 16th-note position within pattern for each onset
    # tick_16th is 0-15 within each bar
    # bar_number indicates which bar
    ticks = df_onsets['tick_16th'].values
    bars = df_onsets['bar_number'].values

    # Position within pattern (0-based for indexing)
    bar_in_pattern = bars % pattern_length
    position_in_pattern = bar_in_pattern * 16 + ticks

    # Create histogram
    histogram = np.zeros(pattern_length * 16)
    for pos in position_in_pattern:
        if 0 <= pos < len(histogram):
            histogram[int(pos)] += 1

    return histogram


def create_rhythm_histograms(
    comprehensive_csv: str,
    track_id: str,
    output_dir: str
) -> dict:
    """
    Create rhythm histograms for a single track.

    Creates 5 rhythm histograms showing onset distributions:
    1. Uncorrected
    2. Per-snippet
    3. FlexStart Pattern Length 4 (1-64 positions)
    4. FlexStart Pattern Length 2 (1-32 positions)
    5. FlexStart Pattern Length 1 (1-16 positions)

    Parameters
    ----------
    comprehensive_csv : str
        Path to comprehensive phases CSV file
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    import json

    print(f"\n  [Rhythm Histograms] Creating rhythm histograms...")

    # Load comprehensive CSV
    df = pd.read_csv(comprehensive_csv)

    # Try to load loop count information from pipeline_results.json
    loop_counts = {}
    try:
        comprehensive_path = Path(comprehensive_csv)
        track_dir = comprehensive_path.parent.parent
        json_file = track_dir / 'pipeline_results.json'

        if json_file.exists():
            with open(json_file, 'r') as f:
                results = json.load(f)
                if 'snippet_info' in results and 'num_complete_loops' in results['snippet_info']:
                    loop_counts = results['snippet_info']['num_complete_loops']
    except Exception as e:
        print(f"    Warning: Could not load loop counts from JSON: {e}")

    # Define methods with their pattern lengths and corresponding loop count keys
    methods = [
        ('Uncorrected', 'phase_uncorrected', 4, None),
        ('Per-Snippet', 'phase_per_snippet', 4, None),
        ('FlexStart Pattern Length 4', 'phase_4bar_pattern_flexStart', 4, 'mel'),
        ('FlexStart Pattern Length 2', 'phase_2bar_pattern_flexStart', 2, 'lepa'),
        ('FlexStart Pattern Length 1', 'phase_1bar_pattern_flexStart', 1, 'aicc'),
    ]

    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 18))
    fig.suptitle(f'Rhythm Histograms — {track_id}', fontsize=14, fontweight='bold')

    # Define colors
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    # Create histograms for each method
    for idx, ((method_title, phase_col, pattern_length, loop_key), color) in enumerate(zip(methods, colors)):
        ax = axes[idx]

        # Check if columns exist
        if phase_col not in df.columns:
            ax.text(0.5, 0.5, f'No data for {method_title}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{method_title}', fontsize=11, fontweight='bold')
            print(f"    {method_title}: No data (missing column)")
            continue

        # Extract histogram
        hist = extract_rhythm_histogram(df, phase_col, pattern_length)
        num_positions = pattern_length * 16

        # X-axis: 16th-note positions (1-based for display)
        positions = np.arange(1, num_positions + 1)

        # Create bar plot
        ax.bar(positions, hist, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Onset Count', fontsize=10, fontweight='bold')

        # Build title with loop count if available
        title = f'{method_title} (L={pattern_length}, {num_positions} positions)'
        if loop_key and loop_key in loop_counts:
            num_loops = loop_counts[loop_key]
            title += f' — {num_loops} loops'

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical lines at bar boundaries (centered on bar beginnings)
        # First line at position 1 (start of pattern)
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=10)
        # Subsequent lines at each bar beginning (every 16 positions)
        for bar_idx in range(1, pattern_length):
            ax.axvline(x=bar_idx * 16 + 1, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5, zorder=10)

        # Add vertical grid lines at expected 16th note positions (gray)
        for i in range(num_positions):
            ax.axvline(x=positions[i], color='gray', linestyle=':',
                      linewidth=0.8, alpha=0.4, zorder=1)

        # Add vertical lines at center of each bar position (blue, bar height)
        y_limits = ax.get_ylim()
        y_range = y_limits[1] - y_limits[0]
        for i in range(num_positions):
            if hist[i] > 0:  # Only draw line if there are onsets at this position
                # Calculate ymax as fraction of axes height
                ymax_fraction = (hist[i] - y_limits[0]) / y_range
                ax.axvline(x=positions[i], ymin=0, ymax=ymax_fraction,
                          color='blue', linestyle='-', linewidth=1.5, alpha=0.7, zorder=10)

        # Set x-axis limits and ticks
        ax.set_xlim(0, num_positions + 1)
        ax.set_xticks(positions)
        ax.tick_params(axis='x', labelsize=7, rotation=90)

        # Calculate statistics
        total_onsets = int(np.sum(hist))
        occupied_positions = int(np.sum(hist > 0))
        max_count = int(np.max(hist)) if total_onsets > 0 else 0

        # Add statistics text box with loop count
        stats_text = f'Total: {total_onsets}\n'
        stats_text += f'Occupied: {occupied_positions}/{num_positions}\n'
        stats_text += f'Max: {max_count}'

        if loop_key and loop_key in loop_counts:
            stats_text += f'\nLoops: {loop_counts[loop_key]}'

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

        # Only show x-axis label on the bottom plot
        if idx == len(methods) - 1:
            ax.set_xlabel('16th-note position within pattern', fontsize=10, fontweight='bold')

        loop_info = f", {loop_counts[loop_key]} loops" if loop_key and loop_key in loop_counts else ""
        print(f"    {method_title}: {total_onsets} onsets, {occupied_positions}/{num_positions} positions{loop_info}")

    plt.tight_layout()

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / f'{track_id}_rhythm_histograms.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf}")

    output_png = output_dir / f'{track_id}_rhythm_histograms.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png}")

    # Also save histogram data as CSV
    csv_data = []
    for method_title, phase_col, pattern_length, loop_key in methods:
        if phase_col in df.columns:
            hist = extract_rhythm_histogram(df, phase_col, pattern_length)
            num_positions = pattern_length * 16
            num_loops = loop_counts.get(loop_key, None) if loop_key else None

            for pos_idx in range(num_positions):
                csv_data.append({
                    'method': method_title,
                    'pattern_length': pattern_length,
                    'num_loops': num_loops,
                    'position': pos_idx + 1,  # 1-based
                    'count': int(hist[pos_idx])
                })

    if csv_data:
        df_out = pd.DataFrame(csv_data)
        output_csv = output_dir / f'{track_id}_rhythm_histograms.csv'
        df_out.to_csv(output_csv, index=False)
        print(f"    Saved: {output_csv}")

    plt.close()

    return {
        'pdf': str(output_pdf),
        'png': str(output_png),
        'csv': str(output_csv) if csv_data else None
    }


def extract_rhythm_histogram_from_flexstart_csv(
    flexstart_csv: str,
    pattern_length: int
) -> np.ndarray:
    """
    Extract rhythm histogram from flexStart filtered CSV.

    Parameters
    ----------
    flexstart_csv : str
        Path to flexStart filtered CSV file (e.g., *_4bar_flexStart_filtered.csv)
    pattern_length : int
        Pattern length in bars (1, 2, or 4)

    Returns
    -------
    np.ndarray
        Histogram counts for each 16th-note position (length: pattern_length * 16)
    """
    try:
        df = pd.read_csv(flexstart_csv)

        # Filter to rows with actual onsets (non-null onset_time)
        df_onsets = df[df['onset_time'].notna()].copy()

        if len(df_onsets) == 0:
            return np.zeros(pattern_length * 16)

        # Calculate 16th-note position within pattern for each onset
        # tick_16th is 0-15 within each bar
        # bar_number indicates which bar
        ticks = df_onsets['tick_16th'].values
        bars = df_onsets['bar_number'].values

        # Position within pattern (0-based for indexing)
        bar_in_pattern = bars % pattern_length
        position_in_pattern = bar_in_pattern * 16 + ticks

        # Create histogram
        histogram = np.zeros(pattern_length * 16)
        for pos in position_in_pattern:
            if 0 <= pos < len(histogram):
                histogram[int(pos)] += 1

        return histogram

    except Exception as e:
        print(f"    Warning: Could not process {flexstart_csv}: {e}")
        return np.zeros(pattern_length * 16)


def create_rhythm_histograms_with_style(
    grid_output_dir: str,
    base_name: str,
    track_id: str,
    output_dir: str
) -> dict:
    """
    Create rhythm histograms using filtered flexStart CSVs.

    Creates 3 rhythm histograms showing onset distributions from filtered flexStart data:
    1. FlexStart Pattern Length 4 (1-64 positions) - from *_4bar_flexStart_filtered.csv
    2. FlexStart Pattern Length 2 (1-32 positions) - from *_2bar_flexStart_filtered.csv
    3. FlexStart Pattern Length 1 (1-16 positions) - from *_1bar_flexStart_filtered.csv

    Parameters
    ----------
    grid_output_dir : str
        Directory containing the filtered flexStart CSV files
    base_name : str
        Base filename (without extension)
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    import json

    print(f"\n  [Rhythm Histograms with Style] Creating rhythm histograms from filtered flexStart CSVs...")

    grid_dir = Path(grid_output_dir)

    # Try to load loop count information from pipeline_results.json
    loop_counts = {}
    try:
        track_dir = Path(grid_output_dir).parent
        json_file = track_dir / 'pipeline_results.json'

        if json_file.exists():
            with open(json_file, 'r') as f:
                results = json.load(f)
                if 'snippet_info' in results and 'num_complete_loops' in results['snippet_info']:
                    loop_counts = results['snippet_info']['num_complete_loops']
    except Exception as e:
        print(f"    Warning: Could not load loop counts from JSON: {e}")

    # Add _comprehensive_phases prefix to match the actual filtered CSV filenames
    full_base_name = f'{base_name}_comprehensive_phases'

    # Define methods with their pattern lengths and corresponding loop count keys
    methods = [
        ('Per-Snippet L=4', f'{full_base_name}_4bar_flexStart_filtered.csv', 4, None, True),
        ('Per-Snippet L=2', f'{full_base_name}_2bar_flexStart_filtered.csv', 2, None, True),
        ('FlexStart Pattern Length 4', f'{full_base_name}_4bar_flexStart_filtered.csv', 4, 'mel', False),
        ('FlexStart Pattern Length 2', f'{full_base_name}_2bar_flexStart_filtered.csv', 2, 'lepa', False),
        ('FlexStart Pattern Length 1', f'{full_base_name}_1bar_flexStart_filtered.csv', 1, 'aicc', False),
    ]

    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle(f'Rhythm Histograms (Filtered FlexStart) — {track_id}', fontsize=14, fontweight='bold')

    # Define colors
    colors = ['#3498DB', '#E67E22', '#2ECC71', '#F39C12', '#9B59B6']

    # Create histograms for each method
    for idx, ((method_title, csv_filename, pattern_length, loop_key, is_per_snippet), color) in enumerate(zip(methods, colors)):
        ax = axes[idx]

        # Check if CSV file exists
        csv_path = grid_dir / csv_filename
        if not csv_path.exists():
            ax.text(0.5, 0.5, f'No data for {method_title}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{method_title}', fontsize=11, fontweight='bold')
            print(f"    {method_title}: No data (missing file: {csv_filename})")
            continue

        # Extract histogram
        hist = extract_rhythm_histogram_from_flexstart_csv(str(csv_path), pattern_length)
        num_positions = pattern_length * 16

        # Calculate onset strength (relative counts)
        total_counts = np.sum(hist)
        onset_strength = hist / total_counts if total_counts > 0 else hist

        # Calculate number of patterns used
        # For both Per-Snippet and FlexStart: read CSV and count unique patterns based on bar_number modulo pattern_length
        # This is more reliable than relying on loop_counts which may not always be available
        num_patterns = None
        try:
            df_csv = pd.read_csv(csv_path)
            min_bar = df_csv['bar_number'].min()
            pattern_indices = (df_csv['bar_number'] - min_bar) // pattern_length
            num_patterns = len(pattern_indices.unique())
        except Exception as e:
            print(f"    Warning: Could not count patterns for {method_title}: {e}")
            # Fallback to loop_counts for FlexStart methods if CSV reading fails
            if not is_per_snippet and loop_key and loop_key in loop_counts:
                num_patterns = loop_counts[loop_key]

        # X-axis: 16th-note positions (1-based for display)
        positions = np.arange(1, num_positions + 1)

        # Create bar plot with onset strength (left y-axis)
        ax.bar(positions, onset_strength, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Onset Strength', fontsize=10, fontweight='bold')

        # Adjust left y-axis scale based on data
        max_strength = np.max(onset_strength) if total_counts > 0 else 1.0
        ax.set_ylim(0, max_strength * 1.1)  # Add 10% padding

        # Create second y-axis for counts (right side)
        ax2 = ax.twinx()
        ax2.set_ylabel('Onset Count', fontsize=10, fontweight='bold', rotation=270, labelpad=15)

        # Adjust right y-axis scale based on data
        max_count = np.max(hist) if total_counts > 0 else 1.0
        ax2.set_ylim(0, max_count * 1.1)  # Add 10% padding

        # Build title with pattern count
        title = f'{method_title} (L={pattern_length}, {num_positions} positions)'
        if num_patterns is not None:
            title += f' — {num_patterns} patterns'

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical lines at bar boundaries (centered on bar beginnings)
        # First line at position 1 (start of pattern)
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=10)
        # Subsequent lines at each bar beginning (every 16 positions)
        for bar_idx in range(1, pattern_length):
            ax.axvline(x=bar_idx * 16 + 1, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5, zorder=10)

        # Add vertical grid lines at expected 16th note positions (gray)
        for i in range(num_positions):
            ax.axvline(x=positions[i], color='gray', linestyle=':',
                      linewidth=0.8, alpha=0.4, zorder=1)

        # Add vertical lines at center of each bar position (blue, bar height)
        y_limits = ax.get_ylim()
        y_range = y_limits[1] - y_limits[0]
        for i in range(num_positions):
            if hist[i] > 0:  # Only draw line if there are onsets at this position
                # Calculate ymax as fraction of axes height
                ymax_fraction = (hist[i] - y_limits[0]) / y_range
                ax.axvline(x=positions[i], ymin=0, ymax=ymax_fraction,
                          color='blue', linestyle='-', linewidth=1.5, alpha=0.7, zorder=10)

        # Set x-axis limits and ticks
        ax.set_xlim(0, num_positions + 1)
        ax.set_xticks(positions)
        ax.tick_params(axis='x', labelsize=7, rotation=90)

        # Calculate statistics
        total_onsets = int(np.sum(hist))
        occupied_positions = int(np.sum(hist > 0))
        max_count = int(np.max(hist)) if total_onsets > 0 else 0

        # Add statistics text box with pattern count
        stats_text = f'Total: {total_onsets}\n'
        stats_text += f'Occupied: {occupied_positions}/{num_positions}\n'
        stats_text += f'Max: {max_count}'

        if num_patterns is not None:
            stats_text += f'\nPatterns: {num_patterns}'

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

        # Only show x-axis label on the bottom plot
        if idx == len(methods) - 1:
            ax.set_xlabel('16th-note position within pattern', fontsize=10, fontweight='bold')

        pattern_info = f", {num_patterns} patterns" if num_patterns is not None else ""
        print(f"    {method_title}: {total_onsets} onsets, {occupied_positions}/{num_positions} positions{pattern_info}")

    plt.tight_layout()

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / f'{track_id}_rhythm_histograms_with_style.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf}")

    output_png = output_dir / f'{track_id}_rhythm_histograms_with_style.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png}")

    # Also save histogram data as CSV
    csv_data = []
    for method_title, csv_filename, pattern_length, loop_key, is_per_snippet in methods:
        csv_path = grid_dir / csv_filename
        if csv_path.exists():
            hist = extract_rhythm_histogram_from_flexstart_csv(str(csv_path), pattern_length)
            num_positions = pattern_length * 16

            # Calculate number of patterns
            num_patterns = None
            if is_per_snippet:
                try:
                    df_csv = pd.read_csv(csv_path)
                    min_bar = df_csv['bar_number'].min()
                    pattern_indices = (df_csv['bar_number'] - min_bar) // pattern_length
                    num_patterns = len(pattern_indices.unique())
                except Exception:
                    pass
            else:
                num_patterns = loop_counts.get(loop_key, None) if loop_key else None

            # Calculate onset strength
            total_counts = np.sum(hist)
            onset_strength = hist / total_counts if total_counts > 0 else hist

            for pos_idx in range(num_positions):
                csv_data.append({
                    'method': method_title,
                    'pattern_length': pattern_length,
                    'num_patterns': num_patterns,
                    'position': pos_idx + 1,  # 1-based
                    'count': int(hist[pos_idx]),
                    'onset_strength': float(onset_strength[pos_idx])
                })

    if csv_data:
        df_out = pd.DataFrame(csv_data)
        output_csv = output_dir / f'{track_id}_rhythm_histograms_with_style.csv'
        df_out.to_csv(output_csv, index=False)
        print(f"    Saved: {output_csv}")

    plt.close()

    return {
        'pdf': str(output_pdf),
        'png': str(output_png),
        'csv': str(output_csv) if csv_data else None
    }


def extract_phase_statistics_from_csv(
    csv_path: str,
    phase_column: str,
    pattern_length: int
) -> tuple:
    """
    Extract phase statistics (median, IQR in 16th notes) from CSV with specified phase column.

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    phase_column : str
        Name of the phase column to use (e.g., 'phase', 'phase_per_snippet')
    pattern_length : int
        Pattern length in bars (1, 2, or 4)

    Returns
    -------
    tuple
        (histogram, median_phases, iqr_16th, raw_iqr_phases) where:
        - histogram: onset counts per position
        - median_phases: median phase value per position (0.0-1.0 within bar)
        - iqr_16th: IQR in 16th note units (for error bars)
        - raw_iqr_phases: raw IQR in phase units (0.0-1.0, no transformations)
    """
    try:
        df = pd.read_csv(csv_path)

        # Filter to rows with actual onsets (non-null phase in the specified column)
        df_onsets = df[df[phase_column].notna()].copy()

        num_positions = pattern_length * 16
        histogram = np.zeros(num_positions)
        median_phases = np.full(num_positions, np.nan)
        iqr_16th = np.full(num_positions, np.nan)
        raw_iqr_phases = np.full(num_positions, np.nan)

        if len(df_onsets) == 0:
            return histogram, median_phases, iqr_16th, raw_iqr_phases

        # Calculate 16th-note position within pattern for each onset
        ticks = df_onsets['tick_16th'].values
        bars = df_onsets['bar_number'].values
        phases = df_onsets[phase_column].values

        # Position within pattern (0-based for indexing)
        bar_in_pattern = bars % pattern_length
        position_in_pattern = bar_in_pattern * 16 + ticks

        # Group phases by position and calculate statistics
        for pos in range(num_positions):
            # Get all phases for this position
            mask = position_in_pattern == pos
            phases_at_pos = phases[mask]

            if len(phases_at_pos) > 0:
                histogram[pos] = len(phases_at_pos)
                median_phases[pos] = np.median(phases_at_pos)

                # Calculate IQR in 16th note units
                if len(phases_at_pos) > 1:
                    q75, q25 = np.percentile(phases_at_pos, [75, 25])
                    iqr_phase = q75 - q25  # IQR in phase units (0.0-1.0)
                    raw_iqr_phases[pos] = iqr_phase

                    # Convert IQR from phase (0.0-1.0 within bar) to 16th note units
                    # 1.0 phase = 16 sixteenth notes within a bar
                    # Multiply by 1.5 to get better error bar representation
                    iqr_16th[pos] = iqr_phase * 16 * 1.5
                else:
                    raw_iqr_phases[pos] = 0.0
                    iqr_16th[pos] = 0.0

        return histogram, median_phases, iqr_16th, raw_iqr_phases

    except Exception as e:
        print(f"    Warning: Could not process {csv_path}: {e}")
        num_positions = pattern_length * 16
        return np.zeros(num_positions), np.full(num_positions, np.nan), np.full(num_positions, np.nan), np.full(num_positions, np.nan)


def create_rhythm_histograms_with_medians_and_iqr(
    grid_output_dir: str,
    base_name: str,
    track_id: str,
    output_dir: str,
    groove_pulse_threshold: float = 0.2
) -> dict:
    """
    Create rhythm histograms with median phase shifts and sqrt(IQR)/1.5 error bars.

    Shows onset strength with bars shifted by median phase, sqrt(IQR)/1.5 error bars,
    and groove pulse threshold line.

    Parameters
    ----------
    grid_output_dir : str
        Directory containing the filtered flexStart CSV files
    base_name : str
        Base filename (without extension)
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots
    groove_pulse_threshold : float
        Threshold multiplier for groove pulse line (default 0.2)

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    import json

    print(f"\n  [Rhythm Histograms with Medians and IQR] Creating from filtered flexStart CSVs...")

    grid_dir = Path(grid_output_dir)

    # Try to load loop count information
    loop_counts = {}
    try:
        track_dir = Path(grid_output_dir).parent
        json_file = track_dir / 'pipeline_results.json'

        if json_file.exists():
            with open(json_file, 'r') as f:
                results = json.load(f)
                if 'snippet_info' in results and 'num_complete_loops' in results['snippet_info']:
                    loop_counts = results['snippet_info']['num_complete_loops']
    except Exception as e:
        print(f"    Warning: Could not load loop counts from JSON: {e}")

    # Read time signature from corrected downbeats file
    time_signature = None
    try:
        track_dir = Path(grid_output_dir).parent
        # Look for corrected downbeats file in 3_corrected directory
        corrected_dir = track_dir / '3_corrected'
        if corrected_dir.exists():
            corrected_files = list(corrected_dir.glob('*_downbeats_corrected.txt'))
            if corrected_files:
                corrected_file = corrected_files[0]
                with open(corrected_file, 'r') as f:
                    for line in f:
                        if line.startswith('# time_signature='):
                            time_signature = int(line.split('=')[1].strip())
                            break
    except Exception as e:
        print(f"    Warning: Could not load time signature: {e}")

    # Add _comprehensive_phases prefix
    full_base_name = f'{base_name}_comprehensive_phases'

    # Define methods: (title, csv_filename, pattern_length, phase_column, loop_key, is_per_snippet)
    # Per-Snippet methods use comprehensive CSV with phase_per_snippet column
    # FlexStart methods use filtered CSVs with phase column
    methods = [
        ('Per-Snippet L=4', f'{full_base_name}.csv', 4, 'phase_per_snippet', None, True),
        ('Per-Snippet L=2', f'{full_base_name}.csv', 2, 'phase_per_snippet', None, True),
        ('FlexStart Pattern Length 4', f'{full_base_name}_4bar_flexStart_filtered.csv', 4, 'phase', 'mel', False),
        ('FlexStart Pattern Length 2', f'{full_base_name}_2bar_flexStart_filtered.csv', 2, 'phase', 'lepa', False),
        ('FlexStart Pattern Length 1', f'{full_base_name}_1bar_flexStart_filtered.csv', 1, 'phase', 'aicc', False),
    ]

    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle(f'Rhythm Histograms with Median Phase & IQR — {track_id}', fontsize=14, fontweight='bold')

    # Define colors
    colors = ['#3498DB', '#E67E22', '#2ECC71', '#F39C12', '#9B59B6']

    # CSV data storage
    csv_data = []

    # Create histograms for each method
    for idx, ((method_title, csv_filename, pattern_length, phase_column, loop_key, is_per_snippet), color) in enumerate(zip(methods, colors)):
        ax = axes[idx]

        # Check if CSV file exists
        csv_path = grid_dir / csv_filename
        if not csv_path.exists():
            ax.text(0.5, 0.5, f'No data for {method_title}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{method_title}', fontsize=11, fontweight='bold')
            print(f"    {method_title}: No data (missing file: {csv_filename})")
            continue

        # Extract statistics using specified phase column
        hist, median_phases, iqr_16th, raw_iqr_phases = extract_phase_statistics_from_csv(str(csv_path), phase_column, pattern_length)
        num_positions = pattern_length * 16

        # Calculate onset strength (normalize to max)
        max_count = np.max(hist)
        onset_strength = hist / max_count if max_count > 0 else hist

        # Calculate number of patterns
        num_patterns = None
        try:
            df_csv = pd.read_csv(csv_path)
            min_bar = df_csv['bar_number'].min()
            pattern_indices = (df_csv['bar_number'] - min_bar) // pattern_length
            num_patterns = len(pattern_indices.unique())
        except Exception as e:
            print(f"    Warning: Could not count patterns for {method_title}: {e}")
            if not is_per_snippet and loop_key and loop_key in loop_counts:
                num_patterns = loop_counts[loop_key]

        # X-axis: base positions (1-based)
        base_positions = np.arange(1, num_positions + 1)

        # Convert median phases to x-axis positions
        # Phase is per-bar (0.0-1.0), so we need to convert to position within pattern
        # IMPORTANT: Use float array to preserve decimal positions!
        shifted_positions = base_positions.copy().astype(float)

        # DEBUG: Print some median phase values
        print(f"\n  DEBUG {method_title}: First 10 median_phases:")
        for i in range(min(10, num_positions)):
            if not np.isnan(median_phases[i]):
                print(f"    Position {i}: median_phase={median_phases[i]:.4f}")

        for i in range(num_positions):
            if not np.isnan(median_phases[i]):
                bar_number = i // 16  # which bar (0, 1, 2, 3...)
                phase_within_bar = median_phases[i]  # 0.0-1.0 within that bar
                # Convert to x-position (1-based)
                shifted_positions[i] = bar_number * 16 + (phase_within_bar * 16) + 1

                # DEBUG: Print calculation for first few positions
                if i < 5:
                    print(f"    Position {i}: bar={bar_number}, phase={phase_within_bar:.4f}, x_pos={shifted_positions[i]:.4f}")

        # Plot bars at shifted positions
        bar_width = 0.8
        bars = ax.bar(shifted_positions, onset_strength, width=bar_width,
                     color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add error bars (IQR in 16th note units) positioned 10% below bar top
        for i in range(num_positions):
            if not np.isnan(iqr_16th[i]) and iqr_16th[i] > 0 and onset_strength[i] > 0:
                # Position error bar at 90% of bar height
                error_bar_y = onset_strength[i] * 0.9
                ax.errorbar(shifted_positions[i], error_bar_y,
                           xerr=iqr_16th[i], fmt='none',
                           ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

        # Add relative median phase value labels on top of bars (horizontal)
        for i in range(num_positions):
            if onset_strength[i] > 0 and not np.isnan(median_phases[i]):
                # Calculate grid_phase for this position
                tick_within_bar = i % 16  # 0-15
                grid_phase = tick_within_bar / 16.0  # Expected phase (0.0-1.0)

                # Calculate relative phase: -1.0 to +1.0
                # -1.0 = halfway to previous tick, 0.0 = on grid, +1.0 = halfway to next tick
                phase_diff = median_phases[i] - grid_phase
                relative_phase = phase_diff / (1.0 / 16.0)  # Normalize by step size

                # Format without leading zero (e.g., .34 instead of 0.34)
                label_text = f'{relative_phase:.2f}'.replace('0.', '.').replace('-0.', '-.')
                ax.text(shifted_positions[i], onset_strength[i], label_text,
                       ha='center', va='bottom', fontsize=6, rotation=0)

        ax.set_ylabel('Onset Strength', fontsize=10, fontweight='bold')

        # Adjust left y-axis scale based on data
        max_strength = np.max(onset_strength) if max_count > 0 else 1.0
        ax.set_ylim(0, max_strength * 1.2)  # Extra padding for labels

        # Create second y-axis for counts (right side)
        ax2 = ax.twinx()
        ax2.set_ylabel('Onset Count', fontsize=10, fontweight='bold', rotation=270, labelpad=15)

        # Adjust right y-axis scale to match left axis
        ax2.set_ylim(0, max_count * 1.2)  # Match padding

        # Add horizontal threshold line at groove_pulse_threshold * max_onset_strength
        threshold_value = groove_pulse_threshold * max_strength
        ax.axhline(y=threshold_value, color='red', linestyle='--',
                  linewidth=2, alpha=0.7, label=f'Groove Pulse Threshold ({groove_pulse_threshold})')

        # Build title with pattern count
        title = f'{method_title} (L={pattern_length}, {num_positions} positions)'
        if num_patterns is not None:
            title += f' — {num_patterns} patterns'

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical lines at bar boundaries (centered on bar beginnings)
        # First line at position 1 (start of pattern)
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=10)
        # Subsequent lines at each bar beginning (every 16 positions)
        for bar_idx in range(1, pattern_length):
            ax.axvline(x=bar_idx * 16 + 1, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5, zorder=10)

        # Add vertical grid lines at expected 16th note positions (gray)
        for i in range(num_positions):
            ax.axvline(x=base_positions[i], color='gray', linestyle=':',
                      linewidth=0.8, alpha=0.4, zorder=1)

        # Add vertical lines at center of each bar (shifted positions, blue, bar height)
        y_limits = ax.get_ylim()
        y_range = y_limits[1] - y_limits[0]
        for i in range(num_positions):
            if onset_strength[i] > 0 and not np.isnan(median_phases[i]):
                # Calculate ymax as fraction of axes height
                ymax_fraction = (onset_strength[i] - y_limits[0]) / y_range
                ax.axvline(x=shifted_positions[i], ymin=0, ymax=ymax_fraction,
                          color='blue', linestyle='-', linewidth=1.5, alpha=0.7, zorder=10)

        # Set x-axis limits and ticks (keep at integer positions)
        ax.set_xlim(0, num_positions + 1)
        ax.set_xticks(base_positions)
        ax.tick_params(axis='x', labelsize=7, rotation=90)

        # Add legend
        ax.legend(loc='upper right', fontsize=8)

        # Calculate statistics
        total_onsets = int(np.sum(hist))
        occupied_positions = int(np.sum(hist > 0))
        max_count = int(np.max(hist)) if total_onsets > 0 else 0

        # Add statistics text box
        stats_text = f'Total: {total_onsets}\n'
        stats_text += f'Occupied: {occupied_positions}/{num_positions}\n'
        stats_text += f'Max: {max_count}'

        if num_patterns is not None:
            stats_text += f'\nPatterns: {num_patterns}'

        if time_signature is not None:
            stats_text += f'\nTime Sig: {time_signature}/4'

        ax.text(0.02, 0.97, stats_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

        # Only show x-axis label on the bottom plot
        if idx == len(methods) - 1:
            ax.set_xlabel('16th-note position within pattern', fontsize=10, fontweight='bold')

        pattern_info = f", {num_patterns} patterns" if num_patterns is not None else ""
        print(f"    {method_title}: {total_onsets} onsets, {occupied_positions}/{num_positions} positions{pattern_info}")

        # Store CSV data
        for pos_idx in range(num_positions):
            # Calculate relative_median_phase
            relative_median_phase = None
            if not np.isnan(median_phases[pos_idx]):
                tick_within_bar = pos_idx % 16  # 0-15
                grid_phase = tick_within_bar / 16.0  # Expected phase (0.0-1.0)
                phase_diff = median_phases[pos_idx] - grid_phase
                relative_median_phase = phase_diff / (1.0 / 16.0)  # Normalize by step size

            csv_data.append({
                'method': method_title,
                'pattern_length': pattern_length,
                'num_patterns': num_patterns,
                'position': pos_idx + 1,  # 1-based
                'count': int(hist[pos_idx]),
                'onset_strength': float(onset_strength[pos_idx]),
                'median_phase': float(median_phases[pos_idx]) if not np.isnan(median_phases[pos_idx]) else None,
                'relative_median_phase': float(relative_median_phase) if relative_median_phase is not None else None,
                'iqr_phase': float(raw_iqr_phases[pos_idx]) if not np.isnan(raw_iqr_phases[pos_idx]) else None,
                'iqr_16th': float(iqr_16th[pos_idx]) if not np.isnan(iqr_16th[pos_idx]) else None
            })

    plt.tight_layout()

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / f'{track_id}_rhythm_histograms_with_medians_and_iqr.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf}")

    output_png = output_dir / f'{track_id}_rhythm_histograms_with_medians_and_iqr.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png}")

    # Save CSV
    if csv_data:
        df_out = pd.DataFrame(csv_data)
        output_csv = output_dir / f'{track_id}_rhythm_histograms_with_medians_and_iqr.csv'
        df_out.to_csv(output_csv, index=False)
        print(f"    Saved: {output_csv}")

    plt.close()

    return {
        'pdf': str(output_pdf),
        'png': str(output_png),
        'csv': str(output_csv) if csv_data else None
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python rhythm_histograms.py <comprehensive_csv> <track_id> <output_dir>')
        sys.exit(1)

    comprehensive_csv = sys.argv[1]
    track_id = sys.argv[2]
    output_dir = sys.argv[3]

    create_rhythm_histograms(comprehensive_csv, track_id, output_dir)
