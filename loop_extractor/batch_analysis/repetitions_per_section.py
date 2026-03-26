#!/usr/bin/env python3
"""
Repetitions Per Section - Step 21 Batch Analysis

Analyzes the distribution of repetitions (num_repetitions) per section from
6.6_anchored_rhythm_histograms data.

Creates histograms showing repetition counts for sections at different
ratio_in_snippet thresholds (>40%, >50%, >60%, >70%, >80%), separated by
pattern length (L1, L2 and L4).

Generates TWO sets of outputs:
1. Unfiltered (before): from *_anchored_rhythm_histograms.csv (6.1 source)
2. Filtered (after): from *_filtered_anchored_rhythm_histograms.csv (6.2 source)

Input:
    6.6_anchored_rhythm_histograms/{track_id}_anchored_rhythm_histograms.csv
    6.6_anchored_rhythm_histograms/{track_id}_filtered_anchored_rhythm_histograms.csv

Output (in snippet_ratio_batch_analysis/):
    - repetitions_per_section_unfiltered.csv / .png / .pdf
    - repetitions_per_section_filtered.csv / .png / .pdf

Usage:
    python repetitions_per_section.py /path/to/batch/output

Example:
    python repetitions_per_section.py "/Volumes/PortableSSD/06_Testing/new test feb20 stricter onset window/new 50"
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


# Ratio thresholds to analyze
RATIO_THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.80]


def collect_repetition_data(output_dir: Path, use_filtered: bool = False, stem: str = 'drums') -> list:
    """
    Collect repetition data from all tracks' 6.6_anchored_rhythm_histograms.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    use_filtered : bool
        If True, read from *_filtered_anchored_rhythm_histograms.csv (6.2 source)
        If False, read from *_anchored_rhythm_histograms.csv (6.1 source)
    stem : str
        Stem to collect data for (default: 'drums')

    Returns
    -------
    list
        List of dicts with keys: track_id, section_id, pattern_length,
        num_repetitions, ratio_in_snippet
    """
    # Find all track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis', 'snippet_ratio_batch_analysis']
    ])

    if not track_dirs:
        print('No track directories found!')
        return []

    all_sections = []

    for track_dir in track_dirs:
        # Look for *_anchored_rhythm_histograms.csv in 6.6_anchored_rhythm_histograms/{stem} folder
        hist_dir = track_dir / '6.6_anchored_rhythm_histograms' / stem
        if not hist_dir.exists():
            continue

        if use_filtered:
            # Find the filtered anchored rhythm histograms file
            csv_files = list(hist_dir.glob('*_filtered_anchored_rhythm_histograms.csv'))
            csv_files = [f for f in csv_files if not f.name.startswith('._')]
        else:
            # Find the unfiltered anchored rhythm histograms file
            csv_files = list(hist_dir.glob('*_anchored_rhythm_histograms.csv'))
            csv_files = [f for f in csv_files
                         if not f.name.startswith('._') and 'filtered' not in f.name.lower()]

        if not csv_files:
            continue

        csv_file = csv_files[0]

        # Read unique section data
        section_data = {}  # section_id -> {pattern_length, num_repetitions, ratio_in_snippet}

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    section_id = row.get('section_id', '')
                    if section_id and section_id not in section_data:
                        try:
                            pattern_length = int(row.get('pattern_length', 0))
                            num_repetitions = int(row.get('num_repetitions', 0))
                            ratio_in_snippet = float(row.get('ratio_in_snippet', 0))

                            section_data[section_id] = {
                                'track_id': track_dir.name,
                                'section_id': section_id,
                                'pattern_length': pattern_length,
                                'num_repetitions': num_repetitions,
                                'ratio_in_snippet': ratio_in_snippet
                            }
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            print(f"  Warning: Could not read {csv_file}: {e}")
            continue

        all_sections.extend(section_data.values())

    return all_sections


def create_repetitions_plot(all_sections: list, batch_dir: Path, suffix: str, title_suffix: str):
    """
    Create repetitions per section plot and save CSV/PNG/PDF.

    Parameters
    ----------
    all_sections : list
        List of section data dicts
    batch_dir : Path
        Output directory
    suffix : str
        File suffix (e.g., '_unfiltered' or '_filtered')
    title_suffix : str
        Title suffix (e.g., ' (Before Filtering)' or ' (After Filtering)')
    """
    # First, compute all repetition counts to include in CSV
    csv_rows = []
    for section in sorted(all_sections, key=lambda x: (x['track_id'], x['section_id'])):
        csv_rows.append(section)

    # Save raw CSV data with normalized strength
    csv_file = batch_dir / f'repetitions_per_section{suffix}.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['track_id', 'section_id', 'pattern_length',
                                               'num_repetitions', 'ratio_in_snippet'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved raw data: {csv_file.name}")

    # Create figure with 5 rows (ratio thresholds) x 3 columns (L1, L2, L4)
    fig, axes = plt.subplots(5, 3, figsize=(15, 14))
    fig.suptitle(f'Repetitions Per Section by Ratio Threshold{title_suffix}',
                 fontsize=14, fontweight='bold', y=0.98)

    pattern_lengths = [1, 2, 4]
    col_labels = ['L1 (1-bar)', 'L2 (2-bar)', 'L4 (4-bar)']

    for row_idx, threshold in enumerate(RATIO_THRESHOLDS):
        for col_idx, (pattern_len, col_label) in enumerate(zip(pattern_lengths, col_labels)):
            ax = axes[row_idx, col_idx]

            # Filter sections for this pattern length and threshold
            filtered = [s for s in all_sections
                       if s['pattern_length'] == pattern_len
                       and s['ratio_in_snippet'] > threshold]

            if not filtered:
                ax.text(0.5, 0.5, '0 complete patterns', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 1)
            else:
                # Count repetitions
                rep_counts = defaultdict(int)
                for s in filtered:
                    rep_counts[s['num_repetitions']] += 1

                # Create histogram bars
                reps = sorted(rep_counts.keys())
                counts = [rep_counts[r] for r in reps]
                max_count = max(counts) if counts else 1

                # Normalize to get "Repetition Strength" (0-1)
                strengths = [c / max_count for c in counts]

                # Plot normalized strength on left y-axis
                bars = ax.bar(reps, strengths, color='#4ECDC4', edgecolor='black', linewidth=0.5, alpha=0.8)

                # Set x-axis to show all integer ticks
                max_rep = max(reps) if reps else 10
                ax.set_xlim(0.5, max_rep + 0.5)
                ax.set_xticks(range(1, max_rep + 1))
                ax.set_ylim(0, 1.15)

                # Create secondary y-axis for counts (right side)
                ax2 = ax.twinx()
                ax2.set_ylim(0, max_count * 1.15)
                if col_idx == 2:  # Only show count label on right column
                    ax2.set_ylabel('Count', fontsize=8, color='gray')
                ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)

                # Add count labels on bars
                for bar, count, strength in zip(bars, counts, strengths):
                    if count > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., strength + 0.03,
                               f'{count}', ha='center', va='bottom', fontsize=8)

            # Labels
            if row_idx == 4:  # Bottom row
                ax.set_xlabel('Repetitions', fontsize=9)
            if row_idx == 0:  # Top row - column headers
                ax.set_title(col_label, fontsize=11, fontweight='bold')
            if col_idx == 0:  # Left column - ratio labels and strength axis
                ax.set_ylabel(f'ratio>{int(threshold*100)}%\nRep. Strength', fontsize=9)

            # Add section count as text in corner
            n_sections = len(filtered)
            ax.text(0.98, 0.95, f'n={n_sections}', ha='right', va='top',
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save figure
    output_png = batch_dir / f'repetitions_per_section{suffix}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"  Saved diagram: {output_png.name}")

    # Save as PDF
    output_pdf = batch_dir / f'repetitions_per_section{suffix}.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"  Saved diagram: {output_pdf.name}")

    plt.close()

    # Print summary statistics
    print(f"\n  Summary{title_suffix}:")
    for pattern_len in pattern_lengths:
        print(f"    L{pattern_len}:")
        for threshold in RATIO_THRESHOLDS:
            filtered = [s for s in all_sections
                       if s['pattern_length'] == pattern_len
                       and s['ratio_in_snippet'] > threshold]

            if filtered:
                reps = [s['num_repetitions'] for s in filtered]
                n_tukey = sum(1 for r in reps if r > 2)
                n_running = sum(1 for r in reps if r <= 2)
                print(f"      ratio>{int(threshold*100)}%: {len(filtered):3d} sections | "
                      f"Tukey(>2): {n_tukey:3d} | Running(<=2): {n_running:3d} | "
                      f"mean={np.mean(reps):.1f}, median={np.median(reps):.0f}")
            else:
                print(f"      ratio>{int(threshold*100)}%:   0 sections")


def create_repetitions_diagrams(output_dir: Path, stem: str = 'drums'):
    """
    Create repetitions per section diagrams from batch processing results.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    stem : str
        Stem to analyze (default: 'drums')
    """
    print("\n" + "=" * 80)
    print(f"STEP 21: REPETITIONS PER SECTION ANALYSIS ({stem})")
    print("=" * 80)

    # Create output directory (stem-specific)
    batch_dir = output_dir / 'snippet_ratio_batch_analysis' / stem
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Find number of track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis', 'snippet_ratio_batch_analysis']
    ])
    print(f"Found {len(track_dirs)} track directories")

    # 1. Unfiltered (before filtering) - from 6.1 source
    print("\n--- UNFILTERED (Before Filtering) ---")
    unfiltered_sections = collect_repetition_data(output_dir, use_filtered=False, stem=stem)
    if unfiltered_sections:
        print(f"Collected {len(unfiltered_sections)} sections")
        create_repetitions_plot(unfiltered_sections, batch_dir, '_unfiltered', ' (Before Filtering)')
    else:
        print("No unfiltered data found!")

    # 2. Filtered (after filtering) - from 6.2 source
    print("\n--- FILTERED (After Filtering) ---")
    filtered_sections = collect_repetition_data(output_dir, use_filtered=True, stem=stem)
    if filtered_sections:
        print(f"Collected {len(filtered_sections)} sections")
        create_repetitions_plot(filtered_sections, batch_dir, '_filtered', ' (After Filtering)')
    else:
        print("No filtered data found!")

    print("\n" + "=" * 80)


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
        print('Usage: python repetitions_per_section.py /path/to/batch/output [stem]')
        print('Example: python repetitions_per_section.py "/Volumes/PortableSSD/06_Testing/new test feb20"')
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
        create_repetitions_diagrams(output_dir, stem=stem)
