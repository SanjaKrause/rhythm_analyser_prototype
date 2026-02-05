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


def process_pattern_based_ioi(
    grid_output_dir: str,
    base_name: str,
    bpm: float,
    snippet_start_time: float
) -> pd.DataFrame:
    """
    Process pattern-based IOI from FlexStart filtered CSV files.

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

        # Read CSV data (skip comment lines starting with #)
        df = pd.read_csv(filtered_csv_path, comment='#')

        if df.empty:
            continue

        # Get metadata from CSV header
        num_patterns_displayed, num_patterns_total, filtering_method = read_filtered_csv_metadata(str(filtered_csv_path))

        # Sort by bar_number and tick_16th to ensure correct ordering
        df = df.sort_values(['bar_number', 'tick_16th']).reset_index(drop=True)

        # Filter to only rows with onsets (non-null tick_phase)
        df = df[df['tick_phase'].notna()].copy()

        if df.empty:
            continue

        # Calculate IOI between consecutive onsets
        for i in range(len(df) - 1):
            onset1 = df.iloc[i]
            onset2 = df.iloc[i + 1]

            # Get tick positions and tick_phases (phase within the 16th note, 0.0-1.0)
            tick1 = onset1['tick_16th']
            tick2 = onset2['tick_16th']
            bar1 = onset1['bar_number']
            bar2 = onset2['bar_number']
            tick_phase1 = onset1['tick_phase']  # Fractional position within 16th note
            tick_phase2 = onset2['tick_phase']

            # Calculate absolute tick positions (accounting for bar crossings)
            tick1_absolute = bar1 * 16 + tick1
            tick2_absolute = bar2 * 16 + tick2

            # Calculate tick delta (can span multiple bars)
            tick_delta = tick2_absolute - tick1_absolute

            # Calculate tick_phase difference (fractional ticks)
            tick_phase_diff = tick_phase2 - tick_phase1

            # Calculate exact IOI in ticks
            ioi_exact_ticks = tick_delta + tick_phase_diff

            # Calculate times in seconds (relative to snippet start)
            time1_rel = (tick1 + tick_phase1) * tick_duration
            time2_rel = (tick2 + tick_phase2) * tick_duration

            # Absolute times
            time1_abs = snippet_start_time + time1_rel
            time2_abs = snippet_start_time + time2_rel

            # Categorize IOI
            ioi_category = categorize_ioi(ioi_exact_ticks)

            # Store data
            all_ioi_data.append({
                'method': 'Pattern-based',
                'pattern_length': pattern_length,
                'onset1_bar': int(bar1),
                'onset1_tick': int(tick1),
                'onset1_tick_absolute': int(tick1_absolute),
                'onset1_tick_phase': float(tick_phase1),
                'onset1_time_abs': float(time1_abs),
                'onset1_time_rel': float(time1_rel),
                'onset2_bar': int(bar2),
                'onset2_tick': int(tick2),
                'onset2_tick_absolute': int(tick2_absolute),
                'onset2_tick_phase': float(tick_phase2),
                'onset2_time_abs': float(time2_abs),
                'onset2_time_rel': float(time2_rel),
                'tick_delta': int(tick_delta),
                'tick_phase_diff': float(tick_phase_diff),
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

    # Process pattern-based IOI
    df_ioi = process_pattern_based_ioi(grid_output_dir, base_name, bpm, snippet_start_time)

    if df_ioi.empty:
        print(f"    ⚠️  No IOI data found")
        return {}

    # Save separate CSV files for each pattern length
    output_files = {}
    pattern_lengths = [4, 2, 1]

    for pattern_length in pattern_lengths:
        df_pattern = df_ioi[df_ioi['pattern_length'] == pattern_length].copy()

        if not df_pattern.empty:
            output_csv = output_path / f'{track_id}_pre_beat_histogram_L{pattern_length}.csv'
            df_pattern.to_csv(output_csv, index=False)
            print(f"    Saved: {output_csv.name} ({len(df_pattern)} intervals)")
            output_files[f'L{pattern_length}'] = str(output_csv)

    # Create histogram visualizations
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Beat Histograms (IOI) — {track_id}', fontsize=14, fontweight='bold', y=0.995)

    # Define colors for each pattern length
    colors = ['#2ECC71', '#F39C12', '#9B59B6']  # Green, Orange, Purple

    # Define IOI categories in ascending order (smallest to largest)
    category_order = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']

    # Map categories to tick values for logarithmic positioning
    category_to_ticks = {
        '1/16': 1,
        '1/8': 2,
        '3/16': 3,
        '1/4': 4,
        '6/16': 6,
        '2/4': 8,
        '4/4': 16
    }

    for idx, (pattern_length, color) in enumerate(zip(pattern_lengths, colors)):
        ax = axes[idx]
        df_pattern = df_ioi[df_ioi['pattern_length'] == pattern_length].copy()

        if df_pattern.empty:
            ax.text(0.5, 0.5, f'No data for L={pattern_length}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Pattern Length L={pattern_length}', fontsize=11, fontweight='bold')
            continue

        # Calculate statistics for each IOI category
        category_stats = {}
        for cat in category_order:
            cat_data = df_pattern[df_pattern['ioi_category'] == cat]['ioi_exact_ticks']

            if len(cat_data) > 0:
                count = len(cat_data)
                mean_ioi = np.mean(cat_data)

                # Calculate IQR with 1.5 scaling factor (similar to rhythm histograms)
                if len(cat_data) > 1:
                    q75, q25 = np.percentile(cat_data, [75, 25])
                    iqr_raw = q75 - q25
                    # Apply scaling factor: multiply by 1.5 for better error bar representation
                    iqr_scaled = iqr_raw * 1.5
                else:
                    iqr_raw = 0.0
                    iqr_scaled = 0.0

                category_stats[cat] = {
                    'count': count,
                    'mean': mean_ioi,
                    'iqr_scaled': iqr_scaled
                }
            else:
                category_stats[cat] = {
                    'count': 0,
                    'mean': np.nan,
                    'iqr_scaled': 0.0
                }

        # Extract arrays for plotting
        counts = np.array([category_stats[cat]['count'] for cat in category_order])
        means = np.array([category_stats[cat]['mean'] for cat in category_order])
        iqrs_scaled = np.array([category_stats[cat]['iqr_scaled'] for cat in category_order])

        # Calculate onset strength (normalize to max count)
        max_count = np.max(counts) if len(counts) > 0 else 1
        onset_strength = counts / max_count if max_count > 0 else counts

        # Calculate base positions in log space (nominal tick values)
        base_positions_log = np.array([np.log2(category_to_ticks[cat]) for cat in category_order])

        # Calculate shifted positions based on mean IOI (similar to rhythm histograms median phase shifts)
        shifted_positions_log = base_positions_log.copy()
        for i, (cat, mean_val) in enumerate(zip(category_order, means)):
            if not np.isnan(mean_val) and mean_val > 0:
                # Shift position to actual mean IOI in log space
                shifted_positions_log[i] = np.log2(mean_val)

        # Create bar plot with shifted logarithmic x-positioning
        bar_width = 0.15  # Width in log space
        bars = ax.bar(shifted_positions_log, onset_strength, width=bar_width, color=color, alpha=0.7,
                     edgecolor='black', linewidth=0.5)

        # Add horizontal IQR error bars (positioned at 90% of bar height)
        for i, (cat, shifted_log, strength, iqr_val, mean_val) in enumerate(zip(category_order, shifted_positions_log, onset_strength, iqrs_scaled, means)):
            if strength > 0 and iqr_val > 0 and not np.isnan(mean_val):
                # Position error bar at 90% of bar height
                error_bar_y = strength * 0.9

                # Convert IQR from tick units to log space
                # Calculate log distance for ±IQR/2 around mean
                log_upper = np.log2(mean_val + iqr_val / 2)
                log_lower = np.log2(max(0.1, mean_val - iqr_val / 2))  # Prevent log(0)
                iqr_log = (log_upper - log_lower) / 2

                ax.errorbar(shifted_log, error_bar_y,
                           xerr=iqr_log, fmt='none',
                           ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

        # Add relative deviation labels on top of bars (similar to rhythm histograms)
        for i, (cat, shifted_log, strength, mean_val) in enumerate(zip(category_order, shifted_positions_log, onset_strength, means)):
            if strength > 0 and not np.isnan(mean_val):
                # Calculate relative deviation from nominal tick value
                nominal_ticks = category_to_ticks[cat]
                relative_deviation = mean_val - nominal_ticks

                # Format without leading zero (e.g., .34 instead of 0.34)
                label_text = f'{relative_deviation:.2f}'.replace('0.', '.').replace('-0.', '-.')
                ax.text(shifted_log, strength, label_text, ha='center', va='bottom',
                       fontsize=7, fontweight='bold')

        # Formatting with logarithmic x-axis
        tick_values = [1, 2, 3, 4, 6, 8, 16]
        tick_positions_log = [np.log2(v) for v in tick_values]
        tick_labels = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']

        ax.set_xticks(tick_positions_log)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel('Onset Strength', fontsize=10, fontweight='bold')

        # Adjust left y-axis scale based on data
        max_strength = np.max(onset_strength) if max_count > 0 else 1.0
        ax.set_ylim(0, max_strength * 1.2)  # Extra padding for labels

        # Create second y-axis for counts (right side)
        ax2 = ax.twinx()
        ax2.set_ylabel('Onset Count', fontsize=10, fontweight='bold', rotation=270, labelpad=15)

        # Adjust right y-axis scale to match left axis
        ax2.set_ylim(0, max_count * 1.2)  # Match padding

        ax.set_title(f'Pattern Length L={pattern_length} ({len(df_pattern)} intervals)',
                    fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical gray grid lines at nominal category positions (log scale)
        for tick_log in tick_positions_log:
            ax.axvline(x=tick_log, color='gray', linestyle=':',
                      linewidth=0.8, alpha=0.4, zorder=1)

        # Add vertical blue lines at center of each bar (shifted positions, bar height)
        y_limits = ax.get_ylim()
        y_range = y_limits[1] - y_limits[0]
        for i, (cat, shifted_log, strength) in enumerate(zip(category_order, shifted_positions_log, onset_strength)):
            if strength > 0 and not np.isnan(means[i]):
                # Calculate ymax as fraction of axes height
                ymax_fraction = (strength - y_limits[0]) / y_range
                ax.axvline(x=shifted_log, ymin=0, ymax=ymax_fraction,
                          color='blue', linestyle='-', linewidth=1.5, alpha=0.7, zorder=10)

        # Set x-axis limits with padding in log space
        ax.set_xlim(-0.5, 4.5)  # log2(1) = 0, log2(16) = 4

        # Only show x-label on bottom subplot
        if idx == len(pattern_lengths) - 1:
            ax.set_xlabel('IOI Category (16th note ticks, log scale)', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save plot as PDF
    output_pdf = output_path / f'{track_id}_beat_histograms.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf.name}")

    # Save plot as PNG
    output_png = output_path / f'{track_id}_beat_histograms.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    output_files['beat_histogram_pdf'] = str(output_pdf)
    output_files['beat_histogram_png'] = str(output_png)

    print(f"    ✓ Processed {len(df_ioi)} total inter-onset intervals")

    return output_files


def create_beat_histograms_all_onsets(
    grid_output_dir: str,
    base_name: str,
    track_id: str,
    output_dir: str,
    bpm: float,
    snippet_start_time: float
) -> dict:
    """
    Create beat-level histograms showing all individual onsets as markers.

    Each IOI is plotted as an 'x' marker at its exact value in log space.

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
    print(f"\n  [Beat Histograms - All Onsets] Creating individual onset plots...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process pattern-based IOI
    df_ioi = process_pattern_based_ioi(grid_output_dir, base_name, bpm, snippet_start_time)

    if df_ioi.empty:
        print(f"    ⚠️  No IOI data found")
        return {}

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Beat Histograms — All Onsets (IOI) — {track_id}', fontsize=14, fontweight='bold', y=0.995)

    # Define colors for each pattern length
    colors = ['#2ECC71', '#F39C12', '#9B59B6']  # Green, Orange, Purple

    # Define IOI categories for x-axis reference
    category_order = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']
    category_to_ticks = {
        '1/16': 1,
        '1/8': 2,
        '3/16': 3,
        '1/4': 4,
        '6/16': 6,
        '2/4': 8,
        '4/4': 16
    }

    pattern_lengths = [4, 2, 1]

    for idx, (pattern_length, color) in enumerate(zip(pattern_lengths, colors)):
        ax = axes[idx]
        df_pattern = df_ioi[df_ioi['pattern_length'] == pattern_length].copy()

        if df_pattern.empty:
            ax.text(0.5, 0.5, f'No data for L={pattern_length}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Pattern Length L={pattern_length}', fontsize=11, fontweight='bold')
            continue

        # Get all IOI values and categories
        ioi_values = df_pattern['ioi_exact_ticks'].values
        ioi_log = np.log2(ioi_values)

        # Create y-positions: jittered slightly for visibility
        np.random.seed(42)  # Reproducible jitter
        y_positions = np.random.uniform(0.4, 0.6, size=len(ioi_values))

        # Plot each onset as an 'x' marker
        ax.scatter(ioi_log, y_positions, marker='x', s=50,
                  color='black', alpha=0.5, linewidths=1.5)

        # Formatting with logarithmic x-axis
        tick_values = [1, 2, 3, 4, 6, 8, 16]
        tick_positions_log = [np.log2(v) for v in tick_values]
        tick_labels = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']

        ax.set_xticks(tick_positions_log)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel('Onset Density', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([])  # Hide y-axis ticks (density visualization)

        ax.set_title(f'Pattern Length L={pattern_length} ({len(df_pattern)} intervals)',
                    fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x')

        # Add vertical gray grid lines at nominal category positions (log scale)
        for tick_log in tick_positions_log:
            ax.axvline(x=tick_log, color='gray', linestyle=':',
                      linewidth=0.8, alpha=0.4, zorder=1)

        # Set x-axis limits with padding in log space
        ax.set_xlim(-0.5, 4.5)  # log2(1) = 0, log2(16) = 4

        # Only show x-label on bottom subplot
        if idx == len(pattern_lengths) - 1:
            ax.set_xlabel('IOI Category (16th note ticks, log scale)', fontsize=10, fontweight='bold')

        print(f"    Pattern Length L={pattern_length}: {len(df_pattern)} onsets")

    plt.tight_layout()

    # Save plot as PDF
    output_pdf = output_path / f'{track_id}_beat_histograms_all_onsets.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf.name}")

    # Save plot as PNG
    output_png = output_path / f'{track_id}_beat_histograms_all_onsets.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    output_files = {
        'beat_histogram_all_onsets_pdf': str(output_pdf),
        'beat_histogram_all_onsets_png': str(output_png)
    }

    print(f"    ✓ Processed {len(df_ioi)} total inter-onset intervals")

    return output_files


def create_simple_ioi_histogram(
    grid_output_dir: str,
    base_name: str,
    track_id: str,
    output_dir: str,
    bpm: float,
    snippet_start_time: float
) -> dict:
    """
    Create simple IOI histogram showing distribution of all inter-onset intervals in milliseconds.

    Finds all onsets in the snippet and plots IOI values on x-axis (ms) with counts on y-axis.

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
    print(f"\n  [Simple IOI Histogram] Creating simple IOI distribution plot...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process pattern-based IOI (get all IOI data)
    df_ioi = process_pattern_based_ioi(grid_output_dir, base_name, bpm, snippet_start_time)

    if df_ioi.empty:
        print(f"    ⚠️  No IOI data found")
        return {}

    # Convert IOI from ticks to milliseconds
    # Calculate tick duration in ms
    bar_duration_s = 60.0 / bpm * 4  # 4 beats per bar at BPM
    tick_duration_s = bar_duration_s / 16  # 16th note duration in seconds
    tick_duration_ms = tick_duration_s * 1000  # Convert to milliseconds

    df_ioi['ioi_ms'] = df_ioi['ioi_exact_ticks'] * tick_duration_ms

    # Create histogram
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle(f'Simple IOI Histogram — {track_id}', fontsize=14, fontweight='bold', y=0.98)

    # Get all IOI values in milliseconds
    ioi_values_ms = df_ioi['ioi_ms'].values

    # Create histogram with automatic binning
    counts, bins, patches = ax.hist(ioi_values_ms, bins=50, color='#3498DB', alpha=0.7,
                                     edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_xlabel('Inter-Onset Interval (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{len(df_ioi)} intervals from all pattern lengths',
                fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add vertical lines at common rhythmic intervals (in ms)
    # Calculate expected IOI values for common categories
    rhythmic_intervals = {
        '1/16': 1 * tick_duration_ms,
        '1/8': 2 * tick_duration_ms,
        '3/16': 3 * tick_duration_ms,
        '1/4': 4 * tick_duration_ms,
        '6/16': 6 * tick_duration_ms,
        '2/4': 8 * tick_duration_ms,
        '4/4': 16 * tick_duration_ms,
    }

    for label, value_ms in rhythmic_intervals.items():
        if ax.get_xlim()[0] <= value_ms <= ax.get_xlim()[1]:
            ax.axvline(x=value_ms, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5, label=label)

    # Add legend for rhythmic interval lines
    ax.legend(loc='upper right', fontsize=9, title='Rhythmic Intervals')

    plt.tight_layout()

    # Save plot as PDF
    output_pdf = output_path / f'{track_id}_simple_ioi_histogram.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf.name}")

    # Save plot as PNG
    output_png = output_path / f'{track_id}_simple_ioi_histogram.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    # Save CSV with IOI in milliseconds
    output_csv = output_path / f'{track_id}_simple_ioi_data.csv'
    df_ioi[['pattern_length', 'ioi_exact_ticks', 'ioi_ms', 'ioi_category',
            'onset1_time_rel', 'onset2_time_rel']].to_csv(output_csv, index=False)
    print(f"    Saved: {output_csv.name}")

    output_files = {
        'simple_ioi_histogram_pdf': str(output_pdf),
        'simple_ioi_histogram_png': str(output_png),
        'simple_ioi_data_csv': str(output_csv)
    }

    print(f"    ✓ Processed {len(df_ioi)} total inter-onset intervals")

    return output_files


def create_simple_beat_histograms(
    output_dir: str,
    track_id: str,
    bpm: float
) -> dict:
    """
    Create simple beat histograms from pre_beat_histogram CSV files.

    Reads the L4, L2, L1 CSV files and creates 3 subplots showing IOI distribution
    in ticks for each pattern length.

    Parameters
    ----------
    output_dir : str
        Directory containing the pre_beat_histogram CSV files
    track_id : str
        Track identifier for plot title
    bpm : float
        Tempo in BPM (for reference, not used in plotting)

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    print(f"\n  [Simple Beat Histograms] Creating simple beat histograms from IOI data...")

    output_path = Path(output_dir)

    # Check if pre_beat_histogram files exist
    pattern_lengths = [4, 2, 1]
    csv_files = {}

    for L in pattern_lengths:
        csv_path = output_path / f'{track_id}_pre_beat_histogram_L{L}.csv'
        if csv_path.exists():
            csv_files[L] = csv_path
        else:
            print(f"    Warning: {csv_path.name} not found")

    if not csv_files:
        print(f"    ⚠️  No pre_beat_histogram CSV files found")
        return {}

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Simple Beat Histograms — {track_id}', fontsize=14, fontweight='bold', y=0.995)

    # Define colors for each pattern length
    colors = ['#2ECC71', '#F39C12', '#9B59B6']  # Green, Orange, Purple

    for idx, (L, color) in enumerate(zip(pattern_lengths, colors)):
        ax = axes[idx]

        if L not in csv_files:
            ax.text(0.5, 0.5, f'No data for L={L}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Pattern Length L={L}', fontsize=11, fontweight='bold')
            continue

        # Read CSV
        df = pd.read_csv(csv_files[L])

        if df.empty:
            ax.text(0.5, 0.5, f'Empty data for L={L}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Pattern Length L={L}', fontsize=11, fontweight='bold')
            continue

        # Get IOI values in ticks
        ioi_values = df['ioi_exact_ticks'].values

        # Create histogram with automatic binning
        counts, bins, patches = ax.hist(ioi_values, bins=50, color=color, alpha=0.7,
                                       edgecolor='black', linewidth=0.5)

        # Formatting
        ax.set_xlabel('IOI (16th note ticks)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title(f'Pattern Length L={L} ({len(df)} intervals)',
                    fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical lines at common rhythmic intervals (in ticks)
        rhythmic_intervals = {
            '1/16': 1,
            '1/8': 2,
            '3/16': 3,
            '1/4': 4,
            '6/16': 6,
            '2/4': 8,
            '4/4': 16,
        }

        for label, value_ticks in rhythmic_intervals.items():
            if ax.get_xlim()[0] <= value_ticks <= ax.get_xlim()[1]:
                ax.axvline(x=value_ticks, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.5)
                # Add text label above the line
                ax.text(value_ticks, ax.get_ylim()[1] * 0.95, label,
                       ha='center', va='top', fontsize=8, color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        print(f"    Pattern Length L={L}: {len(df)} intervals")

    plt.tight_layout()

    # Save plot as PDF
    output_pdf = output_path / f'{track_id}_simple_beat_histograms.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf.name}")

    # Save plot as PNG
    output_png = output_path / f'{track_id}_simple_beat_histograms.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png.name}")

    plt.close()

    output_files = {
        'simple_beat_histograms_pdf': str(output_pdf),
        'simple_beat_histograms_png': str(output_png)
    }

    total_intervals = sum(len(pd.read_csv(csv_files[L])) for L in csv_files.keys())
    print(f"    ✓ Processed {total_intervals} total inter-onset intervals")

    return output_files
