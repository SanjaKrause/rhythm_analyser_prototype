"""
Filter anchored patterns based on onset count using Tukey's method (IQR outlier detection).

This module filters section-anchored CSV files from step 6.1 based on onset density,
removing loops that are outliers according to Tukey's method.

Step 6.2 in the pipeline: Filters patterns within each anchored section file.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re


def parse_csv_metadata(csv_path: Path) -> Dict[str, str]:
    """
    Parse metadata from comment lines at the top of CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file

    Returns
    -------
    Dict[str, str]
        Dictionary of metadata key-value pairs
    """
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#'):
                # Parse "# key=value" format
                match = re.match(r'^#\s*(\w+)=(.+)$', line.strip())
                if match:
                    metadata[match.group(1)] = match.group(2)
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


def filter_anchored_csv(
    input_csv: str,
    output_csv: str,
    pattern_len: int,
    iqr_multiplier: float = 1.5,
    threshold: float = 0.5,
    no_of_repetitions_TH: int = 2
) -> pd.DataFrame:
    """
    Filter anchored patterns using hybrid approach based on pattern count.

    FILTERING ORDER:
    1. First, remove patterns without a reference onset (not in reference_onsets CSV)
    2. Then apply Tukey/running mean filtering on remaining patterns

    HYBRID FILTERING STRATEGY (step 2):
    - If patterns <= no_of_repetitions_TH: Use percentage-based method (running mean)
      - First loop (reference loop) is always kept
      - Remove loops with < threshold * running_mean onset count
    - If patterns > no_of_repetitions_TH: Use Tukey's method (IQR outlier detection)
      - ALL loops treated equally (including first loop)
      - Remove outliers based on IQR bounds

    Parameters
    ----------
    input_csv : str
        Path to input anchored CSV (from step 6.1)
    output_csv : str
        Path to output filtered CSV
    pattern_len : int
        Pattern length in bars (4, 2, or 1)
    iqr_multiplier : float
        IQR multiplier for outlier detection (default 1.5)
    threshold : float
        Threshold ratio for percentage-based method (default 0.5 = 50%)
    no_of_repetitions_TH : int
        Threshold for number of repetitions (default 2)

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    input_path = Path(input_csv)

    # Parse original metadata
    original_metadata = parse_csv_metadata(input_path)

    # Read CSV (skip comment lines)
    df = pd.read_csv(input_csv, comment='#')

    if df.empty:
        print(f"    ! Empty CSV: {input_path.name}")
        return df

    # Calculate loop index (which loop each row belongs to)
    # bar_number in anchored CSVs is 0-based within the section
    min_bar = df['bar_number'].min()
    df['loop_index'] = (df['bar_number'] - min_bar) // pattern_len

    # =========================================================================
    # STEP 1: Filter out patterns without reference onset
    # =========================================================================
    # Load reference onsets CSV to check which patterns have a reference onset
    ref_csv_path = input_path.parent / input_path.name.replace('_anchored.csv', '_reference_onsets.csv')

    loops_without_ref = set()
    if ref_csv_path.exists():
        ref_df = pd.read_csv(ref_csv_path)
        # bar_number in ref_onsets is the starting bar of each pattern (0-based within section)
        # Convert to loop_index
        if 'bar_number' in ref_df.columns:
            ref_loop_indices = set(ref_df['bar_number'] // pattern_len)
            all_loop_indices = set(df['loop_index'].unique())
            loops_without_ref = all_loop_indices - ref_loop_indices

    # Count onsets per loop (where onset_time is not NaN)
    loop_onset_counts = df.groupby('loop_index')['onset_time'].apply(
        lambda x: x.notna().sum()
    ).to_dict()

    # Remove loops without reference onset from consideration
    for loop_idx in loops_without_ref:
        if loop_idx in loop_onset_counts:
            del loop_onset_counts[loop_idx]

    total_patterns_original = len(set(df['loop_index'].unique()))
    total_patterns = len(loop_onset_counts)  # After reference filtering
    loops_to_keep = set()
    loops_to_remove = set(loops_without_ref)  # Start with loops without ref onset

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
        q1 = q3 = iqr = lower_bound = upper_bound = median_onsets = np.nan

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

    # Filter dataframe to keep only non-removed patterns
    df_filtered = df[df['loop_index'].isin(loops_to_keep)].copy()

    # Drop the temporary loop_index column from filtered data
    df_filtered = df_filtered.drop(columns=['loop_index'])

    # Save filtered CSV with combined metadata
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    kept_patterns = len(loops_to_keep)
    removed_patterns = len(loops_to_remove)
    removed_indices = sorted(list(loops_to_remove))
    removed_no_ref_indices = sorted(list(loops_without_ref))

    # Write combined metadata (original + filtering) then CSV data
    with open(output_csv, 'w') as f:
        # Write original metadata first
        for key, value in original_metadata.items():
            f.write(f"# {key}={value}\n")

        # Write filtering metadata
        f.write(f"# filtering_method={filtering_method}\n")
        f.write(f"# patterns_displayed={kept_patterns}\n")
        f.write(f"# patterns_total={total_patterns_original}\n")
        f.write(f"# patterns_with_ref_onset={total_patterns}\n")
        f.write(f"# removed_no_ref_onset={','.join(map(str, removed_no_ref_indices))}\n")
        if total_patterns <= no_of_repetitions_TH:
            f.write(f"# filter_threshold={threshold}\n")
        else:
            f.write(f"# filter_iqr_multiplier={iqr_multiplier}\n")
            f.write(f"# filter_q1={q1:.2f}\n")
            f.write(f"# filter_q3={q3:.2f}\n")
            f.write(f"# filter_iqr={iqr:.2f}\n")
            f.write(f"# filter_lower_bound={lower_bound:.2f}\n")
            f.write(f"# filter_upper_bound={upper_bound:.2f}\n")
            f.write(f"# filter_median={median_onsets:.2f}\n")
        f.write(f"# removed_pattern_indices={','.join(map(str, removed_indices))}\n")

        # Write the CSV content
        df_filtered.to_csv(f, index=False)

    return df_filtered


def filter_all_anchored_patterns(
    anchoring_dir: Path,
    output_dir: Path = None,
    iqr_multiplier: float = 1.5,
    threshold: float = 0.5,
    no_of_repetitions_TH: int = 2,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Filter all anchored CSVs in the 6.1_anchoring directory.

    Processes all *_anchored.csv files, creating *_filtered.csv output in 6.2_filtered_patterns.

    Parameters
    ----------
    anchoring_dir : Path
        Path to 6.1_anchoring directory
    output_dir : Path, optional
        Path to output directory (default: sibling 6.2_filtered_patterns folder)
    iqr_multiplier : float
        IQR multiplier for Tukey outlier detection (default 1.5)
    threshold : float
        Threshold ratio for running mean method (default 0.5 = 50%)
    no_of_repetitions_TH : int
        Threshold for number of repetitions (default 2)
    verbose : bool
        Print progress information

    Returns
    -------
    Dict[str, str]
        Mapping of input file names to output file paths
    """
    anchoring_path = Path(anchoring_dir)

    if not anchoring_path.exists():
        if verbose:
            print(f"  ! Anchoring directory not found: {anchoring_dir}")
        return {}

    # Create output directory (6.2_filtered_patterns as sibling to 6.1_anchoring)
    if output_dir is None:
        output_path = anchoring_path.parent / '6.2_filtered_patterns'
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all anchored CSV files (exclude macOS resource fork files)
    anchored_files = sorted([
        f for f in anchoring_path.glob("*_anchored.csv")
        if not f.name.startswith('._')
    ])

    if not anchored_files:
        if verbose:
            print(f"  ! No anchored CSV files found in {anchoring_dir}")
        return {}

    if verbose:
        print(f"\nStep 6.2: Filtering anchored patterns")
        print(f"  Input:  {anchoring_path}")
        print(f"  Output: {output_path}")
        print(f"  Running mean (threshold={threshold}) for patterns ≤ {no_of_repetitions_TH}")
        print(f"  Tukey (IQR multiplier={iqr_multiplier}) for patterns > {no_of_repetitions_TH}")
        print(f"  Found {len(anchored_files)} anchored files")

    results = {}

    for input_file in anchored_files:
        # Extract pattern length from filename (L1, L2, L4)
        pattern_len = extract_pattern_len_from_filename(input_file.name)

        # Output filename: replace _anchored.csv with _filtered.csv
        output_filename = input_file.name.replace('_anchored.csv', '_filtered.csv')
        output_file = output_path / output_filename

        if verbose:
            print(f"\n  Processing: {input_file.name} (L={pattern_len})")

        df_filtered = filter_anchored_csv(
            str(input_file),
            str(output_file),
            pattern_len,
            iqr_multiplier,
            threshold,
            no_of_repetitions_TH
        )

        if not df_filtered.empty:
            # Read back the metadata to report
            metadata = parse_csv_metadata(output_file)
            patterns_displayed = metadata.get('patterns_displayed', '?')
            patterns_total = metadata.get('patterns_total', '?')
            method = metadata.get('filtering_method', '?')

            if verbose:
                print(f"    ✓ Kept {patterns_displayed}/{patterns_total} patterns ({method})")

            results[input_file.name] = str(output_file)
        else:
            if verbose:
                print(f"    ! No patterns kept")

    if verbose:
        print(f"\n  Filtered {len(results)} files -> {output_path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python filter_anchored_patterns.py <anchoring_dir> [iqr_multiplier] [threshold] [no_of_repetitions_TH]")
        print("Example: python filter_anchored_patterns.py /path/to/6.1_anchoring 1.5 0.5 2")
        print("\nParameters:")
        print("  iqr_multiplier: IQR multiplier for Tukey method (default 1.5)")
        print("    1.5 = standard outlier detection")
        print("    3.0 = extreme outlier detection (more conservative)")
        print("  threshold: Threshold ratio for running mean method (default 0.5 = 50%)")
        print("  no_of_repetitions_TH: Repetition threshold (default 2)")
        print("    If patterns ≤ this value: use running mean method")
        print("    If patterns > this value: use Tukey method")
        sys.exit(1)

    anchoring_dir = Path(sys.argv[1])
    iqr_multiplier = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    no_of_repetitions_TH = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    filter_all_anchored_patterns(
        anchoring_dir=anchoring_dir,
        output_dir=None,  # Use default sibling folder
        iqr_multiplier=iqr_multiplier,
        threshold=threshold,
        no_of_repetitions_TH=no_of_repetitions_TH
    )
