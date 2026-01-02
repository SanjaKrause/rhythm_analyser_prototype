#!/usr/bin/env python3
"""
Rhythm Histograms - Create rhythm-style histograms from batch processing results.

This script reads comprehensive_phases.csv files from batch processing and creates
5 rhythm histograms showing the distribution of onsets across 16th-note positions for:
1. Uncorrected method
2. Per-snippet method
3. Standard L=1 method (1-16 positions)
4. Standard L=2 method (1-32 positions)
5. Standard L=4 method (1-64 positions)

Each histogram shows which 16th-note positions have onsets, aggregated across all loops.

Usage:
    python rhythm_histograms.py /path/to/batch/output
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_rhythm_histogram(
    comprehensive_csv: str,
    phase_col: str,
    pattern_length: int
) -> np.ndarray:
    """
    Extract rhythm histogram from comprehensive CSV.

    Parameters
    ----------
    comprehensive_csv : str
        Path to comprehensive phases CSV file
    phase_col : str
        Column name for phase data (e.g., 'phase_uncorrected', 'phase_standard_L1(L=1)')
    pattern_length : int
        Pattern length in bars (1, 2, or 4)

    Returns
    -------
    np.ndarray
        Histogram counts for each 16th-note position (length: pattern_length * 16)
    """
    try:
        df = pd.read_csv(comprehensive_csv)

        # Filter to rows with actual onsets (non-null phase)
        df_onsets = df[df[phase_col].notna()].copy()

        if len(df_onsets) == 0:
            return np.zeros(pattern_length * 16)

        # Calculate 16th-note position within pattern for each onset
        # tick_16th is 0-15 within each bar
        # bar_number indicates which bar
        ticks = df_onsets['tick_16th'].values
        bars = df_onsets['bar_number'].values

        # Position within pattern (0-based)
        bar_in_pattern = bars % pattern_length
        position_in_pattern = bar_in_pattern * 16 + ticks

        # Create histogram (convert to 1-based for display, but use 0-based for counting)
        histogram = np.zeros(pattern_length * 16)
        for pos in position_in_pattern:
            if 0 <= pos < len(histogram):
                histogram[int(pos)] += 1

        return histogram

    except Exception as e:
        print(f"  Warning: Could not process {comprehensive_csv}: {e}")
        return np.zeros(pattern_length * 16)


def create_rhythm_histograms(output_dir: Path):
    """
    Create rhythm histograms showing onset distributions across 16th-note positions.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    """
    print("\n" + "=" * 80)
    print("RHYTHM HISTOGRAMS")
    print("=" * 80)

    # Find all track directories
    track_dirs = sorted([d for d in output_dir.iterdir()
                        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis']])

    if not track_dirs:
        print('No track directories found!')
        return

    print(f"Found {len(track_dirs)} track directories")

    # Initialize histogram accumulators
    methods = [
        ('Uncorrected', 'phase_uncorrected', 4),
        ('Per-Snippet', 'phase_per_snippet', 4),
        ('Standard L=1', 'phase_standard_L1(L=1)', 1),
        ('Standard L=2', 'phase_standard_L2(L=2)', 2),
        ('Standard L=4', 'phase_standard_L4(L=4)', 4),
    ]

    histograms = {}
    for method_name, phase_col, pattern_length in methods:
        histograms[method_name] = np.zeros(pattern_length * 16)

    # Accumulate histograms from all tracks (read from 5.5_rhythm folder CSV files)
    tracks_processed = 0
    for track_dir in track_dirs:
        rhythm_csv = track_dir / '5.5_rhythm' / f'{track_dir.name}_rhythm_histograms.csv'

        if rhythm_csv.exists():
            try:
                df_rhythm = pd.read_csv(rhythm_csv)

                # Aggregate counts for each method
                for method_name, phase_col, pattern_length in methods:
                    method_data = df_rhythm[df_rhythm['method'] == method_name]
                    if len(method_data) > 0:
                        # Add counts to histogram
                        for _, row in method_data.iterrows():
                            pos = int(row['position']) - 1  # Convert to 0-based index
                            if 0 <= pos < len(histograms[method_name]):
                                histograms[method_name][pos] += row['count']

                tracks_processed += 1
                if tracks_processed % 10 == 0:
                    print(f"  Processed {tracks_processed} tracks...")

            except Exception as e:
                print(f"  Warning: Could not process {track_dir.name}: {e}")

    print(f"Successfully processed {tracks_processed} tracks")

    if tracks_processed == 0:
        print("No valid data found!")
        return

    # Create batch_analysis directory
    batch_dir = output_dir / 'batch_analysis'
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 5 subplots in a column (similar to microtiming plots)
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle(f'Rhythm Histograms — Onset Distribution Across {tracks_processed} Tracks',
                 fontsize=16, fontweight='bold')

    # Define colors
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    # Create histograms for each method
    for idx, ((method_title, phase_col, pattern_length), color) in enumerate(zip(methods, colors)):
        ax = axes[idx]

        hist = histograms[method_title]
        num_positions = pattern_length * 16

        # X-axis: 16th-note positions (1-based for display)
        positions = np.arange(1, num_positions + 1)

        # Create bar plot
        ax.bar(positions, hist, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Onset Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{method_title} (L={pattern_length}, {num_positions} positions)',
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical lines at bar boundaries (every 16 positions)
        for bar_idx in range(1, pattern_length):
            ax.axvline(x=bar_idx * 16 + 0.5, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.5)

        # Set x-axis limits and ticks
        ax.set_xlim(0, num_positions + 1)

        # Show all tick positions
        ax.set_xticks(positions)
        ax.tick_params(axis='x', labelsize=7, rotation=90)

        # Calculate statistics
        total_onsets = int(np.sum(hist))
        occupied_positions = int(np.sum(hist > 0))
        max_count = int(np.max(hist))

        # Add statistics text box
        stats_text = f'Total onsets: {total_onsets}\n'
        stats_text += f'Occupied positions: {occupied_positions}/{num_positions}\n'
        stats_text += f'Max count: {max_count}'

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

        # Only show x-axis label on the bottom plot
        if idx == len(methods) - 1:
            ax.set_xlabel('16th-note position within pattern', fontsize=11, fontweight='bold')

        # Print statistics to console
        print(f"\n{method_title}:")
        print(f"  Total onsets: {total_onsets}")
        print(f"  Occupied positions: {occupied_positions}/{num_positions}")
        print(f"  Max count at one position: {max_count}")

    plt.tight_layout()

    # Save figure
    output_file = batch_dir / 'rhythm_histograms.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved rhythm histograms: {output_file}")

    # Also save as PDF
    output_pdf = batch_dir / 'rhythm_histograms.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved rhythm histograms: {output_pdf}")

    plt.close()

    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python rhythm_histograms.py /path/to/batch/output')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f'Error: Directory does not exist: {output_dir}')
        sys.exit(1)

    create_rhythm_histograms(output_dir)
