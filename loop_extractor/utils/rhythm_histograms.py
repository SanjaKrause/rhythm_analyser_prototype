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
    3. Standard L=1 (1-16 positions)
    4. Standard L=2 (1-32 positions)
    5. Standard L=4 (1-64 positions)

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
        ('Standard L=1', 'phase_standard_L1(L=1)', 1, 'aicc'),
        ('Standard L=2', 'phase_standard_L2(L=2)', 2, 'lepa'),
        ('Standard L=4', 'phase_standard_L4(L=4)', 4, 'mel'),
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

        # Add vertical lines at bar boundaries (every 16 positions)
        for bar_idx in range(1, pattern_length):
            ax.axvline(x=bar_idx * 16 + 0.5, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5)

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


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python rhythm_histograms.py <comprehensive_csv> <track_id> <output_dir>')
        sys.exit(1)

    comprehensive_csv = sys.argv[1]
    track_id = sys.argv[2]
    output_dir = sys.argv[3]

    create_rhythm_histograms(comprehensive_csv, track_id, output_dir)
