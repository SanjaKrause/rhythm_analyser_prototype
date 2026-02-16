"""
Filter loops based on onset count using Tukey's method (IQR outlier detection).

This module filters flexStart pattern CSVs based on onset density,
removing loops that are outliers according to Tukey's method.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple


def filter_loops_by_onset_count_tukey(
    input_csv: str,
    output_csv: str,
    pattern_len: int,
    iqr_multiplier: float = 1.5,
    threshold: float = 0.5,
    no_of_repetitions_TH: int = 2,
    snippet_offset: float = None
) -> pd.DataFrame:
    """
    Filter loops using hybrid approach based on pattern count.

    HYBRID FILTERING STRATEGY:
    - If patterns <= no_of_repetitions_TH: Use percentage-based method (running mean)
      - First loop (reference loop) is always kept
      - Remove loops with < threshold * running_mean onset count
    - If patterns > no_of_repetitions_TH: Use Tukey's method (IQR outlier detection)
      - ALL loops treated equally (including first loop)
      - Remove outliers based on IQR bounds

    Tukey's method (when used):
    - Calculate Q1 (25th percentile) and Q3 (75th percentile) of onset counts
    - IQR = Q3 - Q1
    - Lower bound = Q1 - (iqr_multiplier * IQR)
    - Upper bound = Q3 + (iqr_multiplier * IQR)
    - Remove patterns with onset counts outside these bounds

    Parameters
    ----------
    input_csv : str
        Path to input flexStart CSV
    output_csv : str
        Path to output filtered CSV
    pattern_len : int
        Pattern length in bars (4, 2, or 1)
    iqr_multiplier : float
        IQR multiplier for outlier detection (default 1.5)
        - 1.5 is standard for outliers
        - 3.0 is for extreme outliers
    threshold : float
        Threshold ratio for percentage-based method (default 0.5 = 50%)
    no_of_repetitions_TH : int
        Threshold for number of repetitions (default 2)
        - If patterns <= this value: use running mean method
        - If patterns > this value: use Tukey method

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with 'pattern_removed' column
    """
    df = pd.read_csv(input_csv, comment='#')

    # Calculate loop index (which loop each row belongs to)
    min_bar = df['bar_number'].min()
    df['loop_index'] = (df['bar_number'] - min_bar) // pattern_len

    # Count onsets per loop (where onset_time is not NaN)
    loop_onset_counts = df.groupby('loop_index')['onset_time'].apply(
        lambda x: x.notna().sum()
    ).to_dict()

    total_patterns = len(loop_onset_counts)
    loops_to_keep = set()
    loops_to_remove = set()

    # HYBRID APPROACH: Choose filtering method based on pattern count
    if total_patterns <= no_of_repetitions_TH:
        # RUNNING MEAN METHOD: Percentage-based filtering with running mean
        # First loop (reference loop) is always kept
        loops_to_keep.add(0)
        onset_counts_so_far = []

        for loop_idx in sorted(loop_onset_counts.keys()):
            current_count = loop_onset_counts[loop_idx]

            if loop_idx == 0:
                # First loop - add to history but don't check threshold
                onset_counts_so_far.append(current_count)
                continue

            # Calculate mean of previous loops (excluding loop 0)
            if len(onset_counts_so_far) > 1:
                mean_onset_count = np.mean(onset_counts_so_far[1:])
            else:
                # Only loop 0 so far, use it as baseline
                mean_onset_count = onset_counts_so_far[0]

            # Check threshold
            if current_count >= threshold * mean_onset_count:
                loops_to_keep.add(loop_idx)
            else:
                loops_to_remove.add(loop_idx)

            # Add current to history for next iterations
            onset_counts_so_far.append(current_count)

        filtering_method = f"running mean (threshold={threshold})"
        q1 = q3 = iqr = lower_bound = upper_bound = np.nan

    else:
        # TUKEY'S METHOD: IQR-based outlier detection
        # ALL loops treated equally (no special treatment of loop 0)
        onset_counts_array = np.array(list(loop_onset_counts.values()))

        # Calculate Tukey bounds and median
        q1 = np.percentile(onset_counts_array, 25)
        median_onsets = np.median(onset_counts_array)
        q3 = np.percentile(onset_counts_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)

        for loop_idx, count in loop_onset_counts.items():
            # Check if within bounds
            if lower_bound <= count <= upper_bound:
                loops_to_keep.add(loop_idx)
            else:
                loops_to_remove.add(loop_idx)

        filtering_method = f"Tukey (IQR multiplier={iqr_multiplier})"

    # Add 'pattern_removed' column to track which bars were removed
    df['pattern_removed'] = df['loop_index'].apply(lambda x: x in loops_to_remove)

    # Filter dataframe to keep only non-removed patterns
    df_filtered = df[df['loop_index'].isin(loops_to_keep)].copy()

    # Drop the temporary loop_index column from filtered data
    df_filtered = df_filtered.drop(columns=['loop_index'])

    # Save filtered CSV with metadata in header
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total_patterns = len(loop_onset_counts)
    kept_patterns = len(loops_to_keep)
    removed_patterns = len(loops_to_remove)

    # Get removed pattern indices for metadata
    removed_indices = sorted(list(loops_to_remove))

    # Calculate pattern boundary times (first and last complete pattern)
    if not df_filtered.empty and 'grid_time' in df_filtered.columns:
        # Find the first and last complete pattern boundaries
        min_bar = df_filtered['bar_number'].min()
        max_bar = df_filtered['bar_number'].max()
        num_complete_patterns = (max_bar - min_bar + 1) // pattern_len
        last_complete_pattern_bar = min_bar + (num_complete_patterns * pattern_len) - 1

        # Get start time of first pattern (first bar, tick_16th=0)
        first_pattern_start = df_filtered[
            (df_filtered['bar_number'] == min_bar) &
            (df_filtered['tick_16th'] == 0)
        ]
        pattern_start_time = first_pattern_start['grid_time'].min() if not first_pattern_start.empty else None

        # Get end time of last complete pattern (next bar after last complete pattern, tick_16th=0)
        next_bar_after_last = last_complete_pattern_bar + 1
        next_bar_start = df_filtered[
            (df_filtered['bar_number'] == next_bar_after_last) &
            (df_filtered['tick_16th'] == 0)
        ]
        if not next_bar_start.empty:
            pattern_end_time = next_bar_start['grid_time'].min()
        else:
            # If no next bar, use the max grid_time from last complete pattern bar
            last_complete_bar_data = df_filtered[df_filtered['bar_number'] == last_complete_pattern_bar]
            pattern_end_time = last_complete_bar_data['grid_time'].max() if not last_complete_bar_data.empty else None
    else:
        pattern_start_time = None
        pattern_end_time = None

    # Write metadata as comments, then the CSV data
    with open(output_csv, 'w') as f:
        f.write(f"# filtering_method={filtering_method}\n")
        f.write(f"# patterns_displayed={kept_patterns}\n")
        f.write(f"# patterns_total={total_patterns}\n")
        if pattern_start_time is not None and pattern_end_time is not None:
            f.write(f"# pattern_start_time={pattern_start_time:.6f}\n")
            f.write(f"# pattern_end_time={pattern_end_time:.6f}\n")
            if snippet_offset is not None:
                f.write(f"# pattern_start_time_relative={pattern_start_time - snippet_offset:.6f}\n")
                f.write(f"# pattern_end_time_relative={pattern_end_time - snippet_offset:.6f}\n")
        if total_patterns <= no_of_repetitions_TH:
            # Running mean method metadata
            f.write(f"# threshold={threshold}\n")
            f.write(f"# no_of_repetitions_TH={no_of_repetitions_TH}\n")
        else:
            # Tukey's method metadata
            f.write(f"# iqr_multiplier={iqr_multiplier}\n")
            f.write(f"# no_of_repetitions_TH={no_of_repetitions_TH}\n")
            f.write(f"# q1={q1:.2f}\n")
            f.write(f"# q3={q3:.2f}\n")
            f.write(f"# iqr={iqr:.2f}\n")
            f.write(f"# lower_bound={lower_bound:.2f}\n")
            f.write(f"# upper_bound={upper_bound:.2f}\n")
            f.write(f"# median={median_onsets:.2f}\n")
        f.write(f"# removed_pattern_indices={','.join(map(str, removed_indices))}\n")
        # Write the CSV content
        df_filtered.to_csv(f, index=False)

    # Print results
    print(f"  ✓ Filtered using {filtering_method}")
    if total_patterns <= no_of_repetitions_TH:
        print(f"    Threshold: {threshold} (running mean method, applies when patterns ≤ {no_of_repetitions_TH})")
    else:
        print(f"    Onset count bounds: [{lower_bound:.1f}, {upper_bound:.1f}] (Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f})")
        print(f"    Tukey method applies when patterns > {no_of_repetitions_TH}")
    print(f"    Kept {kept_patterns}/{total_patterns} patterns (removed {removed_patterns})")
    if removed_indices:
        print(f"    Removed pattern indices: {removed_indices}")
    print(f"    Input:  {len(df)} rows → Output: {len(df_filtered)} rows")

    return df_filtered


def filter_all_flexstart_patterns(
    output_dir: Path,
    base_name: str,
    iqr_multiplier: float = 1.5,
    threshold: float = 0.5,
    no_of_repetitions_TH: int = 2,
    snippet_offset: float = None
):
    """
    Filter all three flexStart pattern CSVs (4bar, 2bar, 1bar) using hybrid filtering.

    Uses running mean method for patterns ≤ no_of_repetitions_TH,
    and Tukey's method for patterns > no_of_repetitions_TH.

    Parameters
    ----------
    output_dir : Path
        Directory containing the flexStart CSVs
    base_name : str
        Base filename (without extension)
    iqr_multiplier : float
        IQR multiplier for Tukey outlier detection (default 1.5)
    threshold : float
        Threshold ratio for running mean method (default 0.5 = 50%)
    no_of_repetitions_TH : int
        Threshold for number of repetitions (default 2)
        - If patterns ≤ this value: use running mean method
        - If patterns > this value: use Tukey method
    """
    print(f"\nFiltering flexStart patterns using hybrid method:")
    print(f"  Running mean (threshold={threshold}) for patterns ≤ {no_of_repetitions_TH}")
    print(f"  Tukey (IQR multiplier={iqr_multiplier}) for patterns > {no_of_repetitions_TH}")

    # Process each pattern
    for pattern_name, pattern_len in [('4bar', 4), ('2bar', 2), ('1bar', 1)]:
        input_file = output_dir / f"{base_name}_{pattern_name}_flexStart.csv"

        if not input_file.exists():
            print(f"  ! Skipping {pattern_name}: input file not found")
            continue

        output_file = output_dir / f"{base_name}_{pattern_name}_flexStart_filtered.csv"

        print(f"\n  Processing {pattern_name} pattern:")
        filter_loops_by_onset_count_tukey(
            str(input_file),
            str(output_file),
            pattern_len,
            iqr_multiplier,
            threshold,
            no_of_repetitions_TH,
            snippet_offset
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python filter_bars_and_onsets.py <output_dir> <base_name> [iqr_multiplier] [threshold] [no_of_repetitions_TH]")
        print("Example: python filter_bars_and_onsets.py /path/to/output songname 1.5 0.5 2")
        print("\nParameters:")
        print("  iqr_multiplier: IQR multiplier for Tukey method (default 1.5)")
        print("    1.5 = standard outlier detection")
        print("    3.0 = extreme outlier detection (more conservative)")
        print("  threshold: Threshold ratio for running mean method (default 0.5 = 50%)")
        print("  no_of_repetitions_TH: Repetition threshold (default 2)")
        print("    If patterns ≤ this value: use running mean method")
        print("    If patterns > this value: use Tukey method")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    base_name = sys.argv[2]
    iqr_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    no_of_repetitions_TH = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    filter_all_flexstart_patterns(output_dir, base_name, iqr_multiplier, threshold, no_of_repetitions_TH)
