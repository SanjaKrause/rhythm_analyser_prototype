#!/usr/bin/env python3
"""
Snippet Ratio Diagrams - Batch analysis for section coverage ratios.

This script analyzes all *_anchored_rhythm_histograms.csv files from batch processing
and creates a bar plot showing the percentage of songs meeting various ratio_in_snippet
conditions.

Conditions (in order):
    k: 1 section > 95%
    a: 1 section > 90%
    b: 1 section > 80%
    c: 1 section > 70%
    d: 1 section > 60%
    e: 1 section > 50%
    f: 2 sections both > 40%
    g: 2 sections both > 30%
    h: 1 section > 30% AND 1 section > 40%
    j: No section > 30%
    i: 3+ sections > 20% each

Output:
    - snippet_ratio_batch_analysis/snippet_ratio_diagram.png (bar plot + conditions table)
    - snippet_ratio_batch_analysis/snippet_ratio_diagram.pdf
    - snippet_ratio_batch_analysis/snippet_ratio_results.csv (song IDs per condition)

Usage:
    python snippet_ratio_diagrams.py /path/to/batch/output
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


# Condition definitions with labels
CONDITIONS = [
    ('k', '1 section > 95%'),
    ('a', '1 section > 90%'),
    ('b', '1 section > 80%'),
    ('c', '1 section > 70%'),
    ('d', '1 section > 60%'),
    ('e', '1 section > 50%'),
    ('f', '2 sections both > 40%'),
    ('g', '2 sections both > 30%'),
    ('h', '1 section > 30% AND 1 section > 40%'),
    ('j', 'No section > 30%'),
    ('i', '3+ sections > 20% each'),
]


def get_unique_section_ratios(csv_file: Path) -> list:
    """
    Read CSV file and extract unique section ratios.

    Each section_id has the same ratio_in_snippet for all its rows,
    so we only need to get unique values.

    Parameters
    ----------
    csv_file : Path
        Path to the *_anchored_rhythm_histograms.csv file

    Returns
    -------
    list
        List of unique ratio_in_snippet values (as floats)
    """
    section_ratios = {}

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section_id = row.get('section_id', '')
                if section_id and section_id not in section_ratios:
                    ratio_str = row.get('ratio_in_snippet', '')
                    if ratio_str:
                        try:
                            section_ratios[section_id] = float(ratio_str)
                        except ValueError:
                            pass
    except Exception as e:
        print(f"  Warning: Could not read {csv_file}: {e}")
        return []

    return list(section_ratios.values())


def check_condition(ratios: list, condition_letter: str) -> bool:
    """
    Check if a song's section ratios meet a given condition.

    Parameters
    ----------
    ratios : list
        List of ratio_in_snippet values for a song
    condition_letter : str
        The condition letter to check (k, a, b, c, d, e, f, g, h, j, i)

    Returns
    -------
    bool
        True if the condition is met
    """
    if not ratios:
        return False

    if condition_letter == 'k':
        # 1 section > 95%
        return any(r > 0.95 for r in ratios)
    elif condition_letter == 'a':
        # 1 section > 90%
        return any(r > 0.90 for r in ratios)
    elif condition_letter == 'b':
        # 1 section > 80%
        return any(r > 0.80 for r in ratios)
    elif condition_letter == 'c':
        # 1 section > 70%
        return any(r > 0.70 for r in ratios)
    elif condition_letter == 'd':
        # 1 section > 60%
        return any(r > 0.60 for r in ratios)
    elif condition_letter == 'e':
        # 1 section > 50%
        return any(r > 0.50 for r in ratios)
    elif condition_letter == 'f':
        # 2 sections both > 40%
        count = sum(1 for r in ratios if r > 0.40)
        return count >= 2
    elif condition_letter == 'g':
        # 2 sections both > 30%
        count = sum(1 for r in ratios if r > 0.30)
        return count >= 2
    elif condition_letter == 'h':
        # 1 section > 30% AND a different section > 40%
        # Need at least one > 40%, and at least one OTHER section > 30%
        sections_above_40 = [r for r in ratios if r > 0.40]
        sections_above_30 = [r for r in ratios if r > 0.30]
        # Must have at least 2 sections > 30% total (since >40 implies >30)
        # AND at least one must be > 40%
        return len(sections_above_40) >= 1 and len(sections_above_30) >= 2
    elif condition_letter == 'j':
        # No section > 30%
        return all(r <= 0.30 for r in ratios)
    elif condition_letter == 'i':
        # 3+ sections > 20% each
        count = sum(1 for r in ratios if r > 0.20)
        return count >= 3
    else:
        return False


def create_snippet_ratio_diagrams(output_dir: Path, stem: str = 'drums'):
    """
    Create snippet ratio diagrams from batch processing results.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    stem : str
        Stem to analyze (default: 'drums')
    """
    print("\n" + "=" * 80)
    print(f"SNIPPET RATIO DIAGRAMS ({stem})")
    print("=" * 80)

    # Find all track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis', 'snippet_ratio_batch_analysis']
    ])

    if not track_dirs:
        print('No track directories found!')
        return

    print(f"Found {len(track_dirs)} track directories")

    # Collect data from all tracks
    songs_data = {}  # song_name -> list of ratios

    for track_dir in track_dirs:
        # Look for *_anchored_rhythm_histograms.csv in 6.6_anchored_rhythm_histograms/{stem} folder
        hist_dir = track_dir / '6.6_anchored_rhythm_histograms' / stem
        if not hist_dir.exists():
            continue

        # Find the main (non-filtered) anchored rhythm histograms file
        csv_files = list(hist_dir.glob('*_anchored_rhythm_histograms.csv'))
        # Exclude filtered versions
        csv_files = [f for f in csv_files if not f.name.startswith('._') and 'filtered' not in f.name.lower()]

        if csv_files:
            csv_file = csv_files[0]
            ratios = get_unique_section_ratios(csv_file)
            if ratios:
                songs_data[track_dir.name] = ratios

    print(f"Successfully read ratio data from {len(songs_data)} tracks")

    if not songs_data:
        print("No valid data found!")
        return

    total_songs = len(songs_data)

    # Evaluate all conditions for each song
    condition_results = defaultdict(list)  # condition_letter -> list of song names

    for song_name, ratios in songs_data.items():
        for letter, _ in CONDITIONS:
            if check_condition(ratios, letter):
                condition_results[letter].append(song_name)

    # Calculate percentages and counts
    condition_percentages = []
    condition_counts = []
    condition_letters = []

    for letter, _ in CONDITIONS:
        count = len(condition_results[letter])
        percentage = (count / total_songs) * 100 if total_songs > 0 else 0
        condition_percentages.append(percentage)
        condition_counts.append(count)
        condition_letters.append(letter)

    # Create output directory (stem-specific)
    batch_dir = output_dir / 'snippet_ratio_batch_analysis' / stem
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 2 subplots (bar plot on top, table on bottom)
    fig = plt.figure(figsize=(14, 10))

    # Create grid spec for layout
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

    # Top subplot: Bar chart
    ax1 = fig.add_subplot(gs[0])

    x = np.arange(len(CONDITIONS))
    bars = ax1.bar(x, condition_percentages, color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)

    # Add percentage labels on bars
    for i, (bar, pct, count) in enumerate(zip(bars, condition_percentages, condition_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Condition', fontsize=12)
    ax1.set_ylabel('Percentage of Songs (%)', fontsize=12)
    ax1.set_title(f'Section Coverage Analysis\n(n={total_songs} songs)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(condition_letters, fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(condition_percentages) * 1.15 if condition_percentages else 100)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add second y-axis for absolute counts
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylim(0, max(condition_counts) * 1.15 if condition_counts else total_songs)
    ax1_twin.set_ylabel('Number of Songs', fontsize=12)

    # Sync the twin axis with percentage axis
    ax1_twin.set_ylim(0, total_songs * (ax1.get_ylim()[1] / 100))

    # Bottom subplot: Conditions table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    # Create table data
    table_data = []
    for (letter, description), count, pct in zip(CONDITIONS, condition_counts, condition_percentages):
        table_data.append([letter, description, f'{count}', f'{pct:.1f}%'])

    # Create table
    table = ax2.table(
        cellText=table_data,
        colLabels=['ID', 'Condition', 'Count', 'Percentage'],
        colWidths=[0.08, 0.52, 0.15, 0.15],
        loc='center',
        cellLoc='left'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors for readability
    for i in range(1, len(CONDITIONS) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')

    # Adjust layout (use subplots_adjust instead of tight_layout for table compatibility)
    plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.35)

    # Save figure
    output_file = batch_dir / 'snippet_ratio_diagram.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved diagram: {output_file}")

    # Save as PDF
    output_pdf = batch_dir / 'snippet_ratio_diagram.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved diagram: {output_pdf}")

    plt.close()

    # Save results CSV with one row per song per condition
    results_file = batch_dir / 'snippet_ratio_results.csv'
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['condition', 'description', 'count', 'percentage', 'song_id'])

        # Write one row per song per condition
        for (letter, description), count, pct in zip(CONDITIONS, condition_counts, condition_percentages):
            for song_id in sorted(condition_results[letter]):
                writer.writerow([letter, description, count, f'{pct:.1f}', song_id])

    print(f"Saved results: {results_file}")

    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for (letter, description), count, pct in zip(CONDITIONS, condition_counts, condition_percentages):
        print(f"  {letter}: {description:<35} {count:>4} songs ({pct:>5.1f}%)")

    print("=" * 80)


def detect_available_stems(track_dirs):
    """Detect which stems have data."""
    all_stems = ['vocals', 'drums', 'bass', 'piano', 'other', 'fullmix']
    found_stems = set()

    for track_dir in track_dirs[:5]:
        rhythm_hist_dir = track_dir / '6.6_anchored_rhythm_histograms'
        if rhythm_hist_dir.exists():
            for stem in all_stems:
                stem_dir = rhythm_hist_dir / stem
                if stem_dir.exists() and any(stem_dir.glob('*.csv')):
                    found_stems.add(stem)

    if not found_stems:
        return ['drums']

    return sorted(found_stems, key=lambda s: all_stems.index(s))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python snippet_ratio_diagrams.py /path/to/batch/output [stem]')
        print('If stem is not specified, all available stems will be processed.')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f'Error: Directory does not exist: {output_dir}')
        sys.exit(1)

    # Find track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis', 'snippet_ratio_batch_analysis']
    ])

    # Determine which stems to process
    if len(sys.argv) >= 3:
        stems = [sys.argv[2]]
    else:
        stems = detect_available_stems(track_dirs)
        print(f"Detected stems: {stems}")

    # Process each stem
    for stem in stems:
        create_snippet_ratio_diagrams(output_dir, stem=stem)
