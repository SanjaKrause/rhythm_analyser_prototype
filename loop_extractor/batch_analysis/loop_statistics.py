#!/usr/bin/env python3
"""
Loop Statistics - Create histograms from batch processing results.

This script reads all pipeline_results.json files from batch processing and creates
6 histograms showing the distribution of:
1. Number of bars per song
2. Number of complete loops for drum method
3. Number of complete loops for mel method
4. Number of complete loops for pitch method
5. Number of complete loops for lepa (BIC) method
6. Number of complete loops for aicc (AICc) method

Usage:
    python loop_statistics.py /path/to/batch/output
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def create_loop_statistics(output_dir: Path):
    """
    Create histograms summarizing bars and loop distributions across all tracks.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    """
    print("\n" + "=" * 80)
    print("LOOP STATISTICS")
    print("=" * 80)

    # Collect data from all tracks
    num_bars_data = []
    num_loops_data = {
        'drum': [],
        'mel': [],
        'pitch': [],
        'lepa': [],
        'aicc': []
    }

    # Find all pipeline_results.json files
    track_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis']])

    if not track_dirs:
        print('No track directories found!')
        return

    print(f"Found {len(track_dirs)} track directories")

    # Read snippet info from each track
    tracks_processed = 0
    for track_dir in track_dirs:
        results_file = track_dir / 'pipeline_results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)

                if 'snippet_info' in results:
                    snippet_info = results['snippet_info']

                    # Get number of bars
                    if 'num_full_bars' in snippet_info:
                        num_bars_data.append(snippet_info['num_full_bars'])

                    # Get number of complete loops for each method
                    if 'num_complete_loops' in snippet_info:
                        loops = snippet_info['num_complete_loops']
                        for method in ['drum', 'mel', 'pitch', 'lepa', 'aicc']:
                            if method in loops:
                                num_loops_data[method].append(loops[method])

                    tracks_processed += 1
            except Exception as e:
                print(f"  Warning: Could not read {results_file}: {e}")

    print(f"Successfully read snippet info from {tracks_processed} tracks")

    if tracks_processed == 0:
        print("No valid data found!")
        return

    # Create batch_analysis directory
    batch_dir = output_dir / 'batch_analysis'
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 6 subplots (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bar and Loop Statistics Across All Tracks', fontsize=16, fontweight='bold')

    # Define histogram settings
    methods = ['drum', 'mel', 'pitch', 'lepa', 'aicc']
    method_titles = [
        'Drum Onset Method',
        'Mel-Band Method',
        'Bass Pitch Method',
        'Lepa Style (BIC) Method',
        'Lepa Style (AICc) Method'
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3']

    # Plot 1: Number of bars per song (inside timerange/snippet)
    ax = axes[0, 0]
    if num_bars_data:
        ax.hist(num_bars_data, bins=20, color='#6C5CE7', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Bars', fontsize=11)
        ax.set_ylabel('Number of Tracks', fontsize=11)
        ax.set_title(f'Number of Bars per Song (inside timerange/snippet)\n(n={len(num_bars_data)} tracks)',
                     fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text
        mean_bars = np.mean(num_bars_data)
        median_bars = np.median(num_bars_data)
        ax.text(0.98, 0.97, f'Mean: {mean_bars:.1f}\nMedian: {median_bars:.1f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        print(f"\nNumber of Bars:")
        print(f"  Mean: {mean_bars:.1f}")
        print(f"  Median: {median_bars:.1f}")
        print(f"  Min: {min(num_bars_data)}")
        print(f"  Max: {max(num_bars_data)}")
    else:
        ax.text(0.5, 0.5, 'No data available',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12)
        ax.set_title('Number of Bars per Song (inside timerange/snippet)', fontsize=12, fontweight='bold')

    # Plots 2-6: Number of complete loops for each method
    for idx, (method, title, color) in enumerate(zip(methods, method_titles, colors)):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        ax = axes[row, col]

        if num_loops_data[method]:
            ax.hist(num_loops_data[method], bins=20, color=color, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Complete Loops', fontsize=11)
            ax.set_ylabel('Number of Tracks', fontsize=11)
            ax.set_title(f'{title}\n(n={len(num_loops_data[method])} tracks)',
                        fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistics text
            mean_loops = np.mean(num_loops_data[method])
            median_loops = np.median(num_loops_data[method])
            ax.text(0.98, 0.97, f'Mean: {mean_loops:.1f}\nMedian: {median_loops:.1f}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            print(f"\n{title}:")
            print(f"  Mean loops: {mean_loops:.1f}")
            print(f"  Median loops: {median_loops:.1f}")
            print(f"  Min loops: {min(num_loops_data[method])}")
            print(f"  Max loops: {max(num_loops_data[method])}")
        else:
            ax.text(0.5, 0.5, 'No data available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_file = batch_dir / 'loop_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved loop statistics: {output_file}")

    # Also save as PDF
    output_pdf = batch_dir / 'loop_statistics.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved loop statistics: {output_pdf}")

    plt.close()

    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python loop_statistics.py /path/to/batch/output')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f'Error: Directory does not exist: {output_dir}')
        sys.exit(1)

    create_loop_statistics(output_dir)
