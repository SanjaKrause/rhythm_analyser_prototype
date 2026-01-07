"""
Filter loops based on onset count threshold.

This module filters flexStart pattern CSVs based on onset density,
removing loops that have significantly fewer onsets than the running mean.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set


def filter_loops_by_onset_count(
    input_csv: str,
    output_csv: str,
    pattern_len: int,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Filter loops based on onset count relative to running mean of previous loops.

    The first loop (reference loop) is always kept. For each subsequent loop,
    we calculate the mean onset count of all previous loops (excluding the first)
    and remove the current loop if its onset count is less than threshold * mean.

    Parameters
    ----------
    input_csv : str
        Path to input flexStart CSV
    output_csv : str
        Path to output filtered CSV
    pattern_len : int
        Pattern length in bars (4, 2, or 1)
    threshold : float
        Threshold ratio (default 0.5 = 50%)

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    df = pd.read_csv(input_csv)

    # Calculate loop index (which loop each row belongs to)
    min_bar = df['bar_number'].min()
    df['loop_index'] = (df['bar_number'] - min_bar) // pattern_len

    # Count onsets per loop (where onset_time is not NaN)
    loop_onset_counts = df.groupby('loop_index')['onset_time'].apply(
        lambda x: x.notna().sum()
    ).to_dict()

    # Calculate which loops to keep
    loops_to_keep = set()
    loops_to_keep.add(0)  # Always keep first loop (reference loop)

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

        # Add current to history for next iterations
        onset_counts_so_far.append(current_count)

    # Filter dataframe
    df_filtered = df[df['loop_index'].isin(loops_to_keep)].copy()
    df_filtered = df_filtered.drop(columns=['loop_index'])

    # Save filtered CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)

    removed_loops = len(loop_onset_counts) - len(loops_to_keep)
    print(f"  ✓ Filtered: kept {len(loops_to_keep)}/{len(loop_onset_counts)} loops (removed {removed_loops})")
    print(f"    Input:  {len(df)} rows → Output: {len(df_filtered)} rows")

    return df_filtered


def filter_all_flexstart_patterns(
    output_dir: Path,
    base_name: str,
    threshold: float = 0.5
):
    """
    Filter all three flexStart pattern CSVs (4bar, 2bar, 1bar).

    Parameters
    ----------
    output_dir : Path
        Directory containing the flexStart CSVs
    base_name : str
        Base filename (without extension)
    threshold : float
        Threshold ratio (default 0.5 = 50%)
    """
    print(f"\nFiltering flexStart patterns with {threshold*100:.0f}% onset threshold...")

    # Process each pattern
    for pattern_name, pattern_len in [('4bar', 4), ('2bar', 2), ('1bar', 1)]:
        input_file = output_dir / f"{base_name}_{pattern_name}_flexStart.csv"

        if not input_file.exists():
            print(f"  ! Skipping {pattern_name}: input file not found")
            continue

        output_file = output_dir / f"{base_name}_{pattern_name}_flexStart_filtered.csv"

        print(f"\n  Processing {pattern_name} pattern:")
        filter_loops_by_onset_count(
            str(input_file),
            str(output_file),
            pattern_len,
            threshold
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python filter_bars_and_onsets.py <output_dir> <base_name> [threshold]")
        print("Example: python filter_bars_and_onsets.py /path/to/output songname 0.5")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    base_name = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    filter_all_flexstart_patterns(output_dir, base_name, threshold)
