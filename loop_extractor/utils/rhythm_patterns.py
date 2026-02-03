#!/usr/bin/env python3
"""
Rhythm Patterns - Create binary rhythm pattern visualizations from groove pulse data.

This module reads groove pulse CSV files and creates rhythm pattern histograms showing
binary patterns where onset strength >= 50% is displayed as 1.0 and < 50% as 0.5.

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add parent directory to path to import rhythm_histograms utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.rhythm_histograms import read_filtered_csv_metadata


def read_groove_pulse_csv(csv_path: str) -> pd.DataFrame:
    """
    Read groove pulse CSV file.

    Parameters
    ----------
    csv_path : str
        Path to groove pulse CSV file

    Returns
    -------
    pd.DataFrame
        Groove pulse data with columns: bar_in_pattern, tick_16th, onset_strength_filtered, etc.
    """
    return pd.read_csv(csv_path)


def create_rhythm_pattern_histograms(
    grid_output_dir: str,
    base_name: str,
    track_id: str,
    output_dir: str
) -> dict:
    """
    Create rhythm pattern histograms from groove pulse data.

    Reads groove pulse CSV files and creates binary pattern visualizations where:
    - onset_strength_filtered >= 50% -> bar height 1.0
    - onset_strength_filtered < 50% -> bar height 0.5

    Parameters
    ----------
    grid_output_dir : str
        Directory containing the groove pulse CSV files
    base_name : str
        Base filename (without extension)
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots

    Returns
    -------
    dict
        Dictionary with paths to saved files and statistics
    """
    import json

    print(f"\n  [Rhythm Patterns] Creating rhythm pattern histograms from groove pulse data...")

    grid_dir = Path(grid_output_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get full base name with comprehensive_phases prefix
    full_base_name = base_name.replace('comprehensive_phases', 'comprehensive_phases')
    if not full_base_name.startswith('comprehensive_phases'):
        full_base_name = f'comprehensive_phases_{full_base_name}'

    # Try to load time signature from pipeline_results.json
    time_signature = None
    try:
        comprehensive_path = grid_dir / f'{base_name}.csv'
        track_dir = comprehensive_path.parent.parent
        json_file = track_dir / 'pipeline_results.json'

        if json_file.exists():
            with open(json_file, 'r') as f:
                results = json.load(f)
                if 'downbeat_correction' in results and 'time_signature' in results['downbeat_correction']:
                    time_signature = results['downbeat_correction']['time_signature']
    except Exception as e:
        print(f"    Warning: Could not load time signature from JSON: {e}")

    # Try to load loop counts for pattern count display
    loop_counts = {}
    try:
        comprehensive_path = grid_dir / f'{base_name}.csv'
        track_dir = comprehensive_path.parent.parent
        json_file = track_dir / 'pipeline_results.json'

        if json_file.exists():
            with open(json_file, 'r') as f:
                results = json.load(f)
                if 'snippet_info' in results and 'num_complete_loops' in results['snippet_info']:
                    loop_counts = results['snippet_info']['num_complete_loops']
    except Exception as e:
        print(f"    Warning: Could not load loop counts from JSON: {e}")

    # Read the combined groove pulse CSV file (it's in output_dir, not grid_dir)
    groove_pulse_csv = output_path / f'{track_id}_groove_pulse_histograms_filtered.csv'

    if not groove_pulse_csv.exists():
        print(f"    ⚠️  Groove pulse CSV not found: {groove_pulse_csv.name}")
        print(f"    Looked in: {groove_pulse_csv}")
        return {}

    df_groove_all = pd.read_csv(groove_pulse_csv)

    # Define methods: Only FlexStart L=4, L=2, L=1
    methods = [
        ('Rhythm Pattern L=4', 4, 'FlexStart Pattern Length 4', 'mel'),
        ('Rhythm Pattern L=2', 2, 'FlexStart Pattern Length 2', 'lepa'),
        ('Rhythm Pattern L=1', 1, 'FlexStart Pattern Length 1', 'aicc'),
    ]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Rhythm Patterns — {track_id}', fontsize=14, fontweight='bold', y=0.995)

    # Define colors (match groove pulse histogram colors for L=4, L=2, L=1)
    colors = ['#2ECC71', '#F39C12', '#9B59B6']

    # CSV data storage
    csv_data = []

    # Create histograms for each method
    for idx, ((method_title, pattern_length, method_name, loop_key), color) in enumerate(zip(methods, colors)):
        ax = axes[idx]

        # Filter groove pulse data for this method
        df_groove = df_groove_all[df_groove_all['method'] == method_name].copy()

        if df_groove.empty:
            ax.text(0.5, 0.5, f'No data for {method_title}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{method_title}', fontsize=11, fontweight='bold')
            print(f"    {method_title}: No data (method not found: {method_name})")
            continue

        # Check required columns
        required_cols = ['position', 'onset_strength_filtered']
        if not all(col in df_groove.columns for col in required_cols):
            ax.text(0.5, 0.5, f'Missing required columns',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{method_title}', fontsize=11, fontweight='bold')
            print(f"    {method_title}: Missing required columns")
            continue

        num_positions = pattern_length * 16

        # Initialize arrays for pattern data
        pattern_values = np.zeros(num_positions)  # Binary values: 1.0, 0.5, or 0
        position_exists = np.zeros(num_positions, dtype=bool)  # Track which positions have data
        median_phases = np.full(num_positions, np.nan)  # Median phases for position shifting
        iqr_16th = np.full(num_positions, np.nan)  # IQR for error bars

        # Fill pattern data from groove pulse CSV
        for _, row in df_groove.iterrows():
            pos_1based = int(row['position'])  # Position is 1-based in CSV
            pos = pos_1based - 1  # Convert to 0-indexed
            onset_strength = row['onset_strength_filtered']

            if 0 <= pos < num_positions:
                position_exists[pos] = True

                # Get median phase and IQR if available
                if 'median_phase' in row and pd.notna(row['median_phase']):
                    median_phases[pos] = row['median_phase']
                if 'iqr_16th' in row and pd.notna(row['iqr_16th']):
                    iqr_16th[pos] = row['iqr_16th']

                # Apply threshold: > 50% -> 1.0, 0 < strength <= 50% -> 0.5, strength = 0 -> 0
                if onset_strength > 0.5:
                    pattern_values[pos] = 1.0
                elif onset_strength > 0:
                    pattern_values[pos] = 0.5
                else:
                    pattern_values[pos] = 0.0

        # Get pattern count from groove pulse CSV data
        num_patterns_displayed = None
        num_patterns_total = None

        # Get from first row of this method's data
        if not df_groove.empty and 'num_patterns_displayed' in df_groove.columns and 'num_patterns_total' in df_groove.columns:
            first_row = df_groove.iloc[0]
            if pd.notna(first_row['num_patterns_displayed']):
                num_patterns_displayed = int(first_row['num_patterns_displayed'])
            if pd.notna(first_row['num_patterns_total']):
                num_patterns_total = int(first_row['num_patterns_total'])

        # Try to get filtering method from filtered CSV metadata
        filtering_method = None
        filtered_csv_name = f'{full_base_name}_{pattern_length}bar_flexStart_filtered.csv'
        filtered_csv_path = grid_dir / filtered_csv_name
        if filtered_csv_path.exists():
            _, _, filtering_method = read_filtered_csv_metadata(str(filtered_csv_path))

        # X-axis: base positions (1-based)
        base_positions = np.arange(1, num_positions + 1)

        # Convert median phases to shifted x-axis positions
        shifted_positions = base_positions.copy().astype(float)

        for i in range(num_positions):
            if not np.isnan(median_phases[i]):
                bar_number = i // 16  # which bar (0, 1, 2, 3...)
                phase_within_bar = median_phases[i]  # 0.0-1.0 within that bar
                # Convert to x-position (1-based)
                shifted_positions[i] = bar_number * 16 + (phase_within_bar * 16) + 1

        # Plot bars at shifted positions (only where data exists and pattern_value > 0)
        bar_width = 0.8
        for i in range(num_positions):
            if position_exists[i] and pattern_values[i] > 0:
                ax.bar(shifted_positions[i], pattern_values[i], width=bar_width,
                      color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add error bars (IQR in 16th note units) positioned 10% below bar top
        for i in range(num_positions):
            if not np.isnan(iqr_16th[i]) and iqr_16th[i] > 0 and pattern_values[i] > 0:
                # Position error bar at 90% of bar height
                error_bar_y = pattern_values[i] * 0.9
                ax.errorbar(shifted_positions[i], error_bar_y,
                           xerr=iqr_16th[i], fmt='none',
                           ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

        # Add relative median phase value labels on top of bars
        for i in range(num_positions):
            if pattern_values[i] > 0 and not np.isnan(median_phases[i]):
                # Calculate grid_phase for this position
                tick_within_bar = i % 16  # 0-15
                grid_phase = tick_within_bar / 16.0  # Expected phase (0.0-1.0)

                # Calculate relative phase: -1.0 to +1.0
                phase_diff = median_phases[i] - grid_phase
                relative_phase = phase_diff / (1.0 / 16.0)  # Normalize by step size

                # Format without leading zero (e.g., .34 instead of 0.34)
                label_text = f'{relative_phase:.2f}'.replace('0.', '.').replace('-0.', '-.')
                ax.text(shifted_positions[i], pattern_values[i], label_text,
                       ha='center', va='bottom', fontsize=6, rotation=0)

        ax.set_ylabel('Pattern Level', fontsize=10, fontweight='bold')

        # Set y-axis limits with padding for labels
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0.5, 1.0])

        # Add vertical lines at center of each bar (shifted positions, blue, bar height)
        y_limits = ax.get_ylim()
        y_range = y_limits[1] - y_limits[0]
        for i in range(num_positions):
            if pattern_values[i] > 0 and not np.isnan(median_phases[i]):
                # Calculate ymax as fraction of axes height
                ymax_fraction = (pattern_values[i] - y_limits[0]) / y_range
                ax.axvline(x=shifted_positions[i], ymin=0, ymax=ymax_fraction,
                          color='blue', linestyle='-', linewidth=1.5, alpha=0.7, zorder=10)

        # Build title with pattern count, time signature, and occupied positions
        occupied_positions = int(np.sum(position_exists))
        title = f'{method_title} (L={pattern_length}, {num_positions} positions)'

        if num_patterns_displayed is not None and num_patterns_total is not None:
            title += f' — {num_patterns_displayed}/{num_patterns_total} repetitions'
            if filtering_method:
                if 'Tukey' in filtering_method:
                    title += ' (Tukey)'
                elif 'running mean' in filtering_method:
                    title += ' (Running Mean)'

        # Add time signature if available
        if time_signature is not None:
            title += f' — Time Signature {time_signature}/4'

        # Add occupied positions
        title += f' — Pos {occupied_positions}/{num_positions}'

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical lines at bar boundaries (centered on bar beginnings)
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=10)
        for bar_idx in range(1, pattern_length):
            ax.axvline(x=bar_idx * 16 + 1, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5, zorder=10)

        # Add vertical grid lines at expected 16th note positions (gray)
        for i in range(num_positions):
            ax.axvline(x=base_positions[i], color='gray', linestyle=':',
                      linewidth=0.8, alpha=0.4, zorder=1)

        # Set x-axis limits and ticks
        ax.set_xlim(0.5, num_positions + 0.5)
        ax.set_xticks(base_positions)
        ax.set_xticklabels(base_positions, fontsize=8)

        # Only show x-label on bottom subplot
        if idx == len(methods) - 1:
            ax.set_xlabel('16th Note Position in Pattern (1-based)', fontsize=10, fontweight='bold')

        # Store CSV data
        for i in range(num_positions):
            if position_exists[i]:
                # Calculate relative median phase if median_phase exists
                relative_median_phase = None
                if not np.isnan(median_phases[i]):
                    tick_within_bar = i % 16  # 0-15
                    grid_phase = tick_within_bar / 16.0  # Expected phase (0.0-1.0)
                    phase_diff = median_phases[i] - grid_phase
                    relative_median_phase = phase_diff / (1.0 / 16.0)  # Normalize by step size

                csv_data.append({
                    'method': method_title,
                    'pattern_length': pattern_length,
                    'num_patterns_displayed': num_patterns_displayed,
                    'num_patterns_total': num_patterns_total,
                    'position': i + 1,  # 1-based
                    'pattern_value': float(pattern_values[i]),
                    'median_phase': float(median_phases[i]) if not np.isnan(median_phases[i]) else None,
                    'relative_median_phase': float(relative_median_phase) if relative_median_phase is not None else None,
                    'iqr_16th': float(iqr_16th[i]) if not np.isnan(iqr_16th[i]) else None
                })

        print(f"    {method_title}: {occupied_positions} positions")

    plt.tight_layout()

    # Save plot as PDF
    output_pdf = output_path / f'{base_name}_rhythm_patterns.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf}")

    # Save plot as PNG
    output_png = output_path / f'{base_name}_rhythm_patterns.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png}")

    plt.close()

    # Save CSV
    if csv_data:
        df_out = pd.DataFrame(csv_data)
        output_csv = output_path / f'{track_id}_rhythm_patterns.csv'
        df_out.to_csv(output_csv, index=False)
        print(f"    Saved: {output_csv}")

    print(f"    ✓ Saved rhythm pattern histograms to {output_pdf.name} and {output_png.name}")

    return {
        'rhythm_pattern_histogram_pdf': str(output_pdf),
        'rhythm_pattern_histogram_png': str(output_png),
        'rhythm_pattern_csv': str(output_csv) if csv_data else None
    }
