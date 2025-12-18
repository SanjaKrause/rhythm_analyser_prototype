#!/usr/bin/env python3
"""
Pattern Length Summary - Create pie charts from batch processing results.

This script reads all pipeline_results.json files from batch processing and creates
5 pie charts showing the distribution of pattern lengths for each detection method:
- Drum method
- Mel method
- Pitch method
- Lepa (BIC) method
- Lepa (AICc) method

Usage:
    python pattern_length_summary.py /path/to/batch/output
"""

import json
import sys
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


def create_pattern_length_summary(output_dir: Path):
    """
    Create pie charts summarizing pattern length distributions across all tracks.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    """
    print("\n" + "=" * 80)
    print("PATTERN LENGTH SUMMARY")
    print("=" * 80)

    # Collect pattern lengths from all tracks
    pattern_lengths_data = {
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

    # Read pattern lengths from each track
    tracks_processed = 0
    for track_dir in track_dirs:
        results_file = track_dir / 'pipeline_results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)

                if 'pattern_lengths' in results:
                    pl = results['pattern_lengths']
                    for method in ['drum', 'mel', 'pitch', 'lepa', 'aicc']:
                        if method in pl:
                            pattern_lengths_data[method].append(pl[method])

                    tracks_processed += 1
            except Exception as e:
                print(f"  Warning: Could not read {results_file}: {e}")

    print(f"Successfully read pattern lengths from {tracks_processed} tracks")

    if tracks_processed == 0:
        print("No valid data found!")
        return

    # Create batch_analysis directory
    batch_dir = output_dir / 'batch_analysis'
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 5 subplots (2x3 grid, one empty)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Pattern Length Distribution Across All Tracks', fontsize=16, fontweight='bold')

    methods = ['drum', 'mel', 'pitch', 'lepa', 'aicc']
    method_titles = [
        'Drum Onset Method',
        'Mel-Band Method',
        'Bass Pitch Method',
        'Lepa Style (BIC) Method',
        'Lepa Style (AICc) Method'
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3']

    for idx, (method, title) in enumerate(zip(methods, method_titles)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Count pattern length occurrences
        if pattern_lengths_data[method]:
            counter = Counter(pattern_lengths_data[method])

            # Sort by pattern length
            sorted_data = sorted(counter.items())
            labels = [f'L={pl}' for pl, _ in sorted_data]
            sizes = [count for _, count in sorted_data]

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=[colors[idx % len(colors)]] * len(sizes),
                textprops={'fontsize': 11}
            )

            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)

            # Add title with total count
            ax.set_title(f'{title}\n(n={len(pattern_lengths_data[method])} tracks)',
                        fontsize=12, fontweight='bold', pad=10)

            # Print summary to console
            print(f"\n{title}:")
            for pl, count in sorted_data:
                percentage = (count / len(pattern_lengths_data[method])) * 100
                print(f"  L={pl}: {count} tracks ({percentage:.1f}%)")
        else:
            ax.text(0.5, 0.5, 'No data available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')

    # Hide the last subplot (bottom right) since we only have 5 methods
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save figure
    output_file = batch_dir / 'pattern_length_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved pattern length summary: {output_file}")

    # Also save as PDF
    output_pdf = batch_dir / 'pattern_length_summary.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved pattern length summary: {output_pdf}")

    plt.close()

    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python pattern_length_summary.py /path/to/batch/output')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f'Error: Directory does not exist: {output_dir}')
        sys.exit(1)

    create_pattern_length_summary(output_dir)
