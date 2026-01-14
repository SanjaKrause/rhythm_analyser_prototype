#!/usr/bin/env python3
"""
Groove Pulse and Statistics - Create groove pulse histograms with filtered onsets.

This module filters out onsets below the groove pulse threshold, recalculates onset strengths
from remaining onsets, and creates "groove pulse histograms" showing only the core rhythmic
pattern without weak onsets.

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path to import rhythm_histograms
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.rhythm_histograms import extract_phase_statistics_from_csv


def create_groove_pulse_histograms(
    grid_output_dir: str,
    base_name: str,
    track_id: str,
    output_dir: str,
    groove_pulse_threshold: float = 0.2
) -> dict:
    """
    Create groove pulse histograms by filtering out weak onsets below threshold.

    Filters onset strengths below groove_pulse_threshold, recalculates onset strengths
    from remaining onsets, and creates histograms with median phase shifts and IQR error bars.

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
        Threshold multiplier for filtering weak onsets (default 0.2)

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    import json

    print(f"\n  [Groove Pulse Histograms] Creating from filtered flexStart CSVs...")

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
    methods = [
        ('Per-Snippet L=4', f'{full_base_name}.csv', 4, 'phase_per_snippet', None, True),
        ('Per-Snippet L=2', f'{full_base_name}.csv', 2, 'phase_per_snippet', None, True),
        ('FlexStart Pattern Length 4', f'{full_base_name}_4bar_flexStart_filtered.csv', 4, 'phase', 'mel', False),
        ('FlexStart Pattern Length 2', f'{full_base_name}_2bar_flexStart_filtered.csv', 2, 'phase', 'lepa', False),
        ('FlexStart Pattern Length 1', f'{full_base_name}_1bar_flexStart_filtered.csv', 1, 'phase', 'aicc', False),
    ]

    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle(f'Groove Pulse Histograms (Filtered) — {track_id}', fontsize=14, fontweight='bold')

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

        # Apply groove pulse threshold filter
        max_strength = np.max(onset_strength) if max_count > 0 else 1.0
        threshold_value = groove_pulse_threshold * max_strength

        # Filter: keep only positions above threshold
        filtered_mask = onset_strength >= threshold_value
        filtered_hist = np.where(filtered_mask, hist, 0)
        filtered_max_count = np.max(filtered_hist)

        # Recalculate onset strengths from filtered onsets (normalize to max)
        filtered_onset_strength = filtered_hist / filtered_max_count if filtered_max_count > 0 else filtered_hist

        # Filter median phases and IQR to match (set to NaN where filtered out)
        filtered_median_phases = np.where(filtered_mask, median_phases, np.nan)
        filtered_iqr_16th = np.where(filtered_mask, iqr_16th, np.nan)
        filtered_raw_iqr_phases = np.where(filtered_mask, raw_iqr_phases, np.nan)

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
        shifted_positions = base_positions.copy().astype(float)

        for i in range(num_positions):
            if not np.isnan(filtered_median_phases[i]):
                bar_number = i // 16  # which bar (0, 1, 2, 3...)
                phase_within_bar = filtered_median_phases[i]  # 0.0-1.0 within that bar
                # Convert to x-position (1-based)
                shifted_positions[i] = bar_number * 16 + (phase_within_bar * 16) + 1

        # Plot bars at shifted positions (only for filtered positions)
        bar_width = 0.8
        for i in range(num_positions):
            if filtered_onset_strength[i] > 0:
                ax.bar(shifted_positions[i], filtered_onset_strength[i], width=bar_width,
                      color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add error bars (IQR in 16th note units) positioned 10% below bar top
        for i in range(num_positions):
            if not np.isnan(filtered_iqr_16th[i]) and filtered_iqr_16th[i] > 0 and filtered_onset_strength[i] > 0:
                # Position error bar at 90% of bar height
                error_bar_y = filtered_onset_strength[i] * 0.9
                ax.errorbar(shifted_positions[i], error_bar_y,
                           xerr=filtered_iqr_16th[i], fmt='none',
                           ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

        # Add relative median phase value labels on top of bars
        for i in range(num_positions):
            if filtered_onset_strength[i] > 0 and not np.isnan(filtered_median_phases[i]):
                # Calculate grid_phase for this position
                tick_within_bar = i % 16  # 0-15
                grid_phase = tick_within_bar / 16.0  # Expected phase (0.0-1.0)

                # Calculate relative phase: -1.0 to +1.0
                phase_diff = filtered_median_phases[i] - grid_phase
                relative_phase = phase_diff / (1.0 / 16.0)  # Normalize by step size

                label_text = f'{relative_phase:.2f}'
                ax.text(shifted_positions[i], filtered_onset_strength[i], label_text,
                       ha='center', va='bottom', fontsize=6, rotation=0)

        ax.set_ylabel('Onset Strength (Filtered)', fontsize=10, fontweight='bold')

        # Adjust left y-axis scale based on filtered data
        max_filtered_strength = np.max(filtered_onset_strength) if filtered_max_count > 0 else 1.0
        ax.set_ylim(0, max_filtered_strength * 1.2)  # Extra padding for labels

        # Create second y-axis for counts (right side)
        ax2 = ax.twinx()
        ax2.set_ylabel('Onset Count (Filtered)', fontsize=10, fontweight='bold', rotation=270, labelpad=15)

        # Adjust right y-axis scale to match left axis
        ax2.set_ylim(0, filtered_max_count * 1.2)  # Match padding

        # Build title with pattern count
        title = f'{method_title} (L={pattern_length}, {num_positions} positions)'
        if num_patterns is not None:
            title += f' — {num_patterns} patterns'

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical lines at bar boundaries (centered on bar beginnings)
        # First line at position 1 (start of pattern)
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        # Subsequent lines at each bar beginning (every 16 positions)
        for bar_idx in range(1, pattern_length):
            ax.axvline(x=bar_idx * 16 + 1, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5)

        # Set x-axis limits and ticks (keep at integer positions)
        ax.set_xlim(0, num_positions + 1)
        ax.set_xticks(base_positions)
        ax.tick_params(axis='x', labelsize=7, rotation=90)

        # Calculate statistics
        total_onsets_original = int(np.sum(hist))
        total_onsets_filtered = int(np.sum(filtered_hist))
        occupied_positions = int(np.sum(filtered_hist > 0))
        max_count_stat = int(filtered_max_count) if filtered_max_count > 0 else 0
        num_filtered_out = total_onsets_original - total_onsets_filtered

        # Add statistics text box
        stats_text = f'Original: {total_onsets_original}\n'
        stats_text += f'Filtered: {total_onsets_filtered}\n'
        stats_text += f'Removed: {num_filtered_out}\n'
        stats_text += f'Occupied: {occupied_positions}/{num_positions}\n'
        stats_text += f'Max: {max_count_stat}\n'
        stats_text += f'Threshold: {groove_pulse_threshold}'

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
        print(f"    {method_title}: {total_onsets_filtered}/{total_onsets_original} onsets (removed {num_filtered_out}), {occupied_positions}/{num_positions} positions{pattern_info}")

        # Store CSV data
        for pos_idx in range(num_positions):
            # Calculate relative_median_phase
            relative_median_phase = None
            if not np.isnan(filtered_median_phases[pos_idx]):
                tick_within_bar = pos_idx % 16  # 0-15
                grid_phase = tick_within_bar / 16.0  # Expected phase (0.0-1.0)
                phase_diff = filtered_median_phases[pos_idx] - grid_phase
                relative_median_phase = phase_diff / (1.0 / 16.0)  # Normalize by step size

            csv_data.append({
                'method': method_title,
                'pattern_length': pattern_length,
                'num_patterns': num_patterns,
                'position': pos_idx + 1,  # 1-based
                'count_original': int(hist[pos_idx]),
                'count_filtered': int(filtered_hist[pos_idx]),
                'onset_strength_original': float(onset_strength[pos_idx]),
                'onset_strength_filtered': float(filtered_onset_strength[pos_idx]),
                'median_phase': float(filtered_median_phases[pos_idx]) if not np.isnan(filtered_median_phases[pos_idx]) else None,
                'relative_median_phase': float(relative_median_phase) if relative_median_phase is not None else None,
                'iqr_phase': float(filtered_raw_iqr_phases[pos_idx]) if not np.isnan(filtered_raw_iqr_phases[pos_idx]) else None,
                'iqr_16th': float(filtered_iqr_16th[pos_idx]) if not np.isnan(filtered_iqr_16th[pos_idx]) else None,
                'threshold': threshold_value
            })

    plt.tight_layout()

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / f'{track_id}_groove_pulse_histograms.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"    Saved: {output_pdf}")

    output_png = output_dir / f'{track_id}_groove_pulse_histograms.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png}")

    # Save CSV
    if csv_data:
        df_out = pd.DataFrame(csv_data)
        output_csv = output_dir / f'{track_id}_groove_pulse_histograms.csv'
        df_out.to_csv(output_csv, index=False)
        print(f"    Saved: {output_csv}")

    plt.close()

    return {
        'pdf': str(output_pdf),
        'png': str(output_png),
        'csv': str(output_csv) if csv_data else None
    }


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python groove_pulse_and_statistics.py <grid_output_dir> <base_name> <track_id> <output_dir> [threshold]')
        sys.exit(1)

    grid_output_dir = sys.argv[1]
    base_name = sys.argv[2]
    track_id = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else grid_output_dir
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2

    create_groove_pulse_histograms(grid_output_dir, base_name, track_id, output_dir, threshold)
