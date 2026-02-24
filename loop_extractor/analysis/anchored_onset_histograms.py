"""
Step 6.4: Anchored onset histograms.

This module creates bar plots showing the number of onsets per pattern and per bar
for section-anchored data from the 6.2_filtered_patterns step.

Output goes to 6.4_onset_histograms folder as combined subplot figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re


def parse_csv_metadata(csv_path: Path) -> Dict[str, Any]:
    """
    Parse metadata from comment lines at the top of CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file

    Returns
    -------
    Dict[str, Any]
        Dictionary of metadata key-value pairs
    """
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#'):
                # Parse "# key=value" format
                match = re.match(r'^#\s*(\w+)=(.+)$', line.strip())
                if match:
                    key = match.group(1)
                    value = match.group(2)
                    # Try to convert to numeric
                    try:
                        if '.' in value:
                            metadata[key] = float(value)
                        else:
                            metadata[key] = int(value)
                    except ValueError:
                        metadata[key] = value
            else:
                break  # Stop at first non-comment line
    return metadata


def extract_pattern_len_from_filename(filename: str) -> int:
    """
    Extract pattern length from anchored CSV filename.

    Filename format: SecNo1_L2_pre-chorus_0.2972_anchored.csv
    The L{N} part indicates pattern length.

    Parameters
    ----------
    filename : str
        The filename (without path)

    Returns
    -------
    int
        Pattern length (1, 2, or 4)
    """
    match = re.search(r'_L(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 4  # Default to 4 if not found


def get_section_data(anchored_csv: Path) -> Optional[Dict[str, Any]]:
    """
    Load and process data from an anchored CSV file.

    Parameters
    ----------
    anchored_csv : Path
        Path to the *_anchored.csv file

    Returns
    -------
    Dict or None
        Dictionary with processed data, or None if failed
    """
    if not anchored_csv.exists():
        return None

    # Parse metadata
    metadata = parse_csv_metadata(anchored_csv)
    pattern_len = int(metadata.get('pattern_length', extract_pattern_len_from_filename(anchored_csv.name)))

    # Read CSV data
    df = pd.read_csv(anchored_csv, comment='#')

    if df.empty or 'bar_number' not in df.columns:
        return None

    # Calculate loop index (which pattern each row belongs to)
    min_bar = df['bar_number'].min()
    df['loop_index'] = (df['bar_number'] - min_bar) // pattern_len

    # Count onsets per pattern (where onset_time is not NaN)
    pattern_onset_counts = df.groupby('loop_index')['onset_time'].apply(
        lambda x: x.notna().sum()
    ).to_dict()

    # Count onsets per bar
    bar_onset_counts = df.groupby('bar_number')['onset_time'].apply(
        lambda x: x.notna().sum()
    ).to_dict()

    # Get global bar numbers for each pattern
    pattern_global_bars = {}
    if 'bar_number_global' in df.columns:
        for loop_idx in pattern_onset_counts.keys():
            loop_rows = df[df['loop_index'] == loop_idx]
            if len(loop_rows) > 0:
                global_bars = sorted(loop_rows['bar_number_global'].unique())
                pattern_global_bars[loop_idx] = f"{min(global_bars)}-{max(global_bars)}"

    # Get global bar numbers for each bar
    bar_global_nums = {}
    if 'bar_number_global' in df.columns:
        for bar_num in bar_onset_counts.keys():
            bar_rows = df[df['bar_number'] == bar_num]
            if len(bar_rows) > 0:
                bar_global_nums[bar_num] = int(bar_rows['bar_number_global'].iloc[0])

    # Parse section info from filename
    filename_stem = anchored_csv.stem.replace('_anchored', '')
    parts = filename_stem.split('_')
    sec_num = parts[0] if parts else ''
    pattern_info = parts[1] if len(parts) > 1 else f'L{pattern_len}'
    section_label = metadata.get('section_label', '')

    return {
        'csv_file': anchored_csv,
        'metadata': metadata,
        'pattern_len': pattern_len,
        'pattern_onset_counts': pattern_onset_counts,
        'bar_onset_counts': bar_onset_counts,
        'pattern_global_bars': pattern_global_bars,
        'bar_global_nums': bar_global_nums,
        'sec_num': sec_num,
        'pattern_info': pattern_info,
        'section_label': section_label,
    }


def plot_onsets_per_pattern_subplot(ax: plt.Axes, section_data: Dict[str, Any]) -> plt.Axes:
    """
    Plot onsets per pattern on a given axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    section_data : Dict
        Section data from get_section_data()

    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    pattern_onset_counts = section_data['pattern_onset_counts']
    pattern_global_bars = section_data['pattern_global_bars']
    metadata = section_data['metadata']
    pattern_len = section_data['pattern_len']

    if not pattern_onset_counts:
        ax.text(0.5, 0.5, 'No pattern data', ha='center', va='center', transform=ax.transAxes)
        return ax

    # Data for plotting
    pattern_nums = sorted(pattern_onset_counts.keys())
    onset_counts = [pattern_onset_counts[p] for p in pattern_nums]

    # Bar plot
    bar_width = 0.8
    x_positions = list(range(len(pattern_nums)))
    bars = ax.bar(x_positions, onset_counts, width=bar_width,
                  color='steelblue', edgecolor='black', alpha=0.8)

    # Add onset count labels on bars
    for bar, count in zip(bars, onset_counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Primary x-axis (pattern number)
    ax.set_xlabel('Pattern Index', fontsize=9)
    ax.set_ylabel('Onsets', fontsize=9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(p) for p in pattern_nums], fontsize=8)

    # Secondary x-axis (global bar ranges)
    if pattern_global_bars:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([pattern_global_bars.get(p, '') for p in pattern_nums], fontsize=7)
        ax2.set_xlabel('Global Bars', fontsize=8)

    # Add horizontal lines for filtering thresholds
    # For filtered data (6.2): thresholds from metadata
    # For unfiltered data (6.1): calculate thresholds on-the-fly based on n_reps
    filtering_method = str(metadata.get('filtering_method', ''))

    if filtering_method:
        # Use thresholds from metadata (filtered 6.2 data)
        if 'Tukey' in filtering_method:
            lower_bound = metadata.get('filter_lower_bound')
            upper_bound = metadata.get('filter_upper_bound')
            median_val = metadata.get('filter_median')

            if lower_bound is not None:
                ax.axhline(y=lower_bound, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
            if upper_bound is not None:
                ax.axhline(y=upper_bound, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
            if median_val is not None:
                ax.axhline(y=median_val, color='green', linestyle='-', linewidth=1.2, alpha=0.8)
    elif len(onset_counts) >= 2:
        # Calculate thresholds on-the-fly for unfiltered data (6.1)
        # Use hybrid approach: Tukey for n_reps > 2, running mean for n_reps <= 2
        n_reps = len(onset_counts)
        no_of_repetitions_TH = 2
        running_mean_threshold = 0.5  # Same as filter_anchored_patterns.py

        if n_reps > no_of_repetitions_TH:
            # TUKEY METHOD: IQR-based bounds
            onset_counts_array = np.array(onset_counts)
            q1 = np.percentile(onset_counts_array, 25)
            median_val = np.median(onset_counts_array)
            q3 = np.percentile(onset_counts_array, 75)
            iqr = q3 - q1
            iqr_multiplier = 1.5
            lower_bound = q1 - (iqr_multiplier * iqr)
            upper_bound = q3 + (iqr_multiplier * iqr)

            ax.axhline(y=lower_bound, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
            ax.axhline(y=upper_bound, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
            ax.axhline(y=median_val, color='green', linestyle='-', linewidth=1.2, alpha=0.8)
        else:
            # RUNNING MEAN METHOD: threshold * mean (first pattern always kept)
            # Show threshold line based on mean of all patterns
            mean_onsets = np.mean(onset_counts)
            threshold_line = running_mean_threshold * mean_onsets

            ax.axhline(y=threshold_line, color='orange', linestyle=':', linewidth=1.2, alpha=0.8)
            ax.axhline(y=mean_onsets, color='green', linestyle='-', linewidth=1.2, alpha=0.8)

    # Build title
    sec_num = section_data['sec_num']
    pattern_info = section_data['pattern_info']
    section_label = section_data['section_label']
    n_reps = metadata.get('no_of_repetitions', len(pattern_nums))

    title = f'{sec_num} {pattern_info} {section_label}\nL={pattern_len}, n_reps={n_reps}'
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    return ax


def plot_onsets_per_bar_subplot(ax: plt.Axes, section_data: Dict[str, Any]) -> plt.Axes:
    """
    Plot onsets per bar on a given axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    section_data : Dict
        Section data from get_section_data()

    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    bar_onset_counts = section_data['bar_onset_counts']
    bar_global_nums = section_data['bar_global_nums']
    metadata = section_data['metadata']
    pattern_len = section_data['pattern_len']

    if not bar_onset_counts:
        ax.text(0.5, 0.5, 'No bar data', ha='center', va='center', transform=ax.transAxes)
        return ax

    # Data for plotting
    bar_nums = sorted(bar_onset_counts.keys())
    onset_counts = [bar_onset_counts[b] for b in bar_nums]

    # Bar plot
    bar_width = 0.8
    x_positions = list(range(len(bar_nums)))
    bars = ax.bar(x_positions, onset_counts, width=bar_width,
                  color='steelblue', edgecolor='black', alpha=0.8)

    # Add onset count labels on bars (only if not too many bars)
    if len(bar_nums) <= 20:
        for bar, count in zip(bars, onset_counts):
            height = bar.get_height()
            ax.annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    # Primary x-axis (section-relative bar number)
    ax.set_xlabel('Bar Index', fontsize=9)
    ax.set_ylabel('Onsets', fontsize=9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(b) for b in bar_nums], fontsize=7)

    # Secondary x-axis (global bar numbers)
    if bar_global_nums:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([str(bar_global_nums.get(b, '')) for b in bar_nums], fontsize=6)
        ax2.set_xlabel('Global Bar', fontsize=8)

    # Add horizontal lines for per-bar thresholds
    filtering_method = str(metadata.get('filtering_method', ''))

    if 'Tukey' in filtering_method:
        lower_bound = metadata.get('filter_lower_bound')
        upper_bound = metadata.get('filter_upper_bound')
        median_val = metadata.get('filter_median')

        if lower_bound is not None:
            per_bar_lower = lower_bound / pattern_len
            ax.axhline(y=per_bar_lower, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
        if upper_bound is not None:
            per_bar_upper = upper_bound / pattern_len
            ax.axhline(y=per_bar_upper, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
        if median_val is not None:
            per_bar_median = median_val / pattern_len
            ax.axhline(y=per_bar_median, color='green', linestyle='-', linewidth=1.2, alpha=0.8)

    # Build title
    sec_num = section_data['sec_num']
    pattern_info = section_data['pattern_info']
    section_label = section_data['section_label']
    n_reps = metadata.get('no_of_repetitions', '?')

    title = f'{sec_num} {pattern_info} {section_label}\n{len(bar_nums)} bars'
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    return ax


def create_combined_onset_histograms(
    filtered_dir: str,
    output_dir: Optional[str] = None,
    track_id: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Optional[str]]:
    """
    Create combined onset histogram plots for all anchored CSVs.

    Creates two PNG files with 2 rows (L2 sections in row 1, L4 sections in row 2):
    - {track_id}_onsets_per_pattern.png: All sections' onsets per pattern
    - {track_id}_onsets_per_bar.png: All sections' onsets per bar

    Parameters
    ----------
    filtered_dir : str
        Path to 6.2_filtered_patterns directory
    output_dir : str, optional
        Output directory. If None, creates 6.4_onset_histograms sibling folder
    track_id : str, optional
        Track identifier for filenames. If None, extracted from parent folder
    verbose : bool
        Print progress messages

    Returns
    -------
    Dict[str, Optional[str]]
        Dictionary with 'per_pattern' and 'per_bar' output paths
    """
    filtered_path = Path(filtered_dir)

    if not filtered_path.exists():
        if verbose:
            print(f"  Filtered directory not found: {filtered_dir}")
        return {'per_pattern': None, 'per_bar': None}

    # Determine output directory
    if output_dir is None:
        output_path = filtered_path.parent / '6.4_onset_histograms'
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine track_id
    if track_id is None:
        track_id = filtered_path.parent.name

    # Find all anchored CSV files (exclude macOS resource fork files)
    anchored_files = sorted([
        f for f in filtered_path.glob("*_anchored.csv")
        if not f.name.startswith('._')
    ])

    if not anchored_files:
        if verbose:
            print(f"  No anchored CSV files found in {filtered_dir}")
        return {'per_pattern': None, 'per_bar': None}

    if verbose:
        print(f"\nStep 6.4: Creating anchored onset histograms")
        print(f"  Input: {filtered_path}")
        print(f"  Output: {output_path}")
        print(f"  Found {len(anchored_files)} anchored files")

    # Load data for all sections
    sections_data = []
    for csv_file in anchored_files:
        data = get_section_data(csv_file)
        if data:
            sections_data.append(data)
            if verbose:
                print(f"    Loaded: {csv_file.name}")

    if not sections_data:
        if verbose:
            print("  No valid section data found")
        return {'per_pattern': None, 'per_bar': None}

    # Build dictionaries mapping sec_num to section data for L2 and L4
    l2_by_sec = {}
    l4_by_sec = {}
    for s in sections_data:
        sec_num = s['sec_num']  # e.g., "SecNo1", "SecNo2"
        if s['pattern_len'] == 2:
            l2_by_sec[sec_num] = s
        elif s['pattern_len'] == 4:
            l4_by_sec[sec_num] = s

    # Get all unique section numbers, sorted
    all_sec_nums = sorted(set(l2_by_sec.keys()) | set(l4_by_sec.keys()))
    n_cols = max(len(all_sec_nums), 1)
    n_rows = 2  # Always 2 rows: L2 and L4

    if verbose:
        print(f"  Section numbers: {all_sec_nums}")
        print(f"  L2 sections: {list(l2_by_sec.keys())}")
        print(f"  L4 sections: {list(l4_by_sec.keys())}")

    results = {'per_pattern': None, 'per_bar': None}

    # --- Create onsets per pattern figure ---
    fig_pattern, axes_pattern = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_cols == 1:
        axes_pattern = axes_pattern.reshape(-1, 1)

    # Row 0: L2 sections (matched by sec_num)
    for col, sec_num in enumerate(all_sec_nums):
        ax = axes_pattern[0, col]
        if sec_num in l2_by_sec:
            plot_onsets_per_pattern_subplot(ax, l2_by_sec[sec_num])
        else:
            # Get section label from L4 if available for better placeholder title
            section_label = l4_by_sec.get(sec_num, {}).get('section_label', '')
            ax.text(0.5, 0.5, '0 Complete Patterns', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{sec_num} L2 {section_label}\n(No L2 data)', fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    # Row 1: L4 sections (matched by sec_num)
    for col, sec_num in enumerate(all_sec_nums):
        ax = axes_pattern[1, col]
        if sec_num in l4_by_sec:
            plot_onsets_per_pattern_subplot(ax, l4_by_sec[sec_num])
        else:
            # Get section label from L2 if available for better placeholder title
            section_label = l2_by_sec.get(sec_num, {}).get('section_label', '')
            ax.text(0.5, 0.5, '0 Complete Patterns', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{sec_num} L4 {section_label}\n(No L4 data)', fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    fig_pattern.suptitle(f'{track_id} - Onsets per Pattern\nRow 1: L2 | Row 2: L4',
                         fontsize=12, fontweight='bold')
    fig_pattern.tight_layout()

    pattern_output = output_path / f'{track_id}_onsets_per_pattern.png'
    fig_pattern.savefig(pattern_output, dpi=150, bbox_inches='tight')
    plt.close(fig_pattern)
    results['per_pattern'] = str(pattern_output)

    if verbose:
        print(f"  Saved: {pattern_output.name}")

    # --- Create onsets per bar figure ---
    fig_bar, axes_bar = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_cols == 1:
        axes_bar = axes_bar.reshape(-1, 1)

    # Row 0: L2 sections (matched by sec_num)
    for col, sec_num in enumerate(all_sec_nums):
        ax = axes_bar[0, col]
        if sec_num in l2_by_sec:
            plot_onsets_per_bar_subplot(ax, l2_by_sec[sec_num])
        else:
            section_label = l4_by_sec.get(sec_num, {}).get('section_label', '')
            ax.text(0.5, 0.5, '0 Complete Patterns', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{sec_num} L2 {section_label}\n(No L2 data)', fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    # Row 1: L4 sections (matched by sec_num)
    for col, sec_num in enumerate(all_sec_nums):
        ax = axes_bar[1, col]
        if sec_num in l4_by_sec:
            plot_onsets_per_bar_subplot(ax, l4_by_sec[sec_num])
        else:
            section_label = l2_by_sec.get(sec_num, {}).get('section_label', '')
            ax.text(0.5, 0.5, '0 Complete Patterns', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{sec_num} L4 {section_label}\n(No L4 data)', fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    fig_bar.suptitle(f'{track_id} - Onsets per Bar\nRow 1: L2 | Row 2: L4',
                     fontsize=12, fontweight='bold')
    fig_bar.tight_layout()

    bar_output = output_path / f'{track_id}_onsets_per_bar.png'
    fig_bar.savefig(bar_output, dpi=150, bbox_inches='tight')
    plt.close(fig_bar)
    results['per_bar'] = str(bar_output)

    if verbose:
        print(f"  Saved: {bar_output.name}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python anchored_onset_histograms.py <filtered_dir> [track_id]")
        print("Example: python anchored_onset_histograms.py /path/to/6.2_filtered_patterns")
        sys.exit(1)

    filtered_dir = sys.argv[1]
    track_id = sys.argv[2] if len(sys.argv) > 2 else None

    create_combined_onset_histograms(filtered_dir, track_id=track_id)
