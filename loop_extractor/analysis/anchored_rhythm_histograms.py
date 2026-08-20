#!/usr/bin/env python3
"""
Anchored Rhythm Histograms - Create rhythm histograms from section-anchored data.

This module creates rhythm histograms showing the distribution of onsets
across 16th-note positions within patterns using section-anchored data from
6.2_filtered_patterns folder.

Layout: 2 rows (L2 patterns, L4 patterns) x N columns (sections by SecNo)
CSV output: aggregated by section_name + pattern_length (e.g., "SecNo1_chorus_L2")

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import re


def parse_anchored_filename(filename: str) -> Optional[Dict]:
    """
    Parse section-anchored CSV filename to extract metadata.

    Filename pattern: SecNo{N}_L{length}_{section_label}_{ratio}_anchored.csv

    Parameters
    ----------
    filename : str
        Filename to parse (e.g., "SecNo1_L2_pre-chorus_0.2972_anchored.csv")

    Returns
    -------
    dict or None
        Dictionary with keys: sec_no, pattern_length, section_label, ratio
        Returns None if filename doesn't match pattern
    """
    # Pattern: SecNo{N}_L{length}_{section_label}_{ratio}_anchored.csv
    pattern = r'^SecNo(\d+)_L(\d+)_(.+)_(\d+\.\d+)_anchored\.csv$'
    match = re.match(pattern, filename)

    if match:
        return {
            'sec_no': int(match.group(1)),
            'pattern_length': int(match.group(2)),
            'section_label': match.group(3),
            'ratio': float(match.group(4)),
            'section_id': f"SecNo{match.group(1)}_{match.group(3)}_L{match.group(2)}"
        }
    return None


def read_anchored_csv_metadata(csv_path: str) -> Dict:
    """
    Read metadata from section-anchored CSV header comments.

    Parameters
    ----------
    csv_path : str
        Path to anchored CSV file

    Returns
    -------
    dict
        Dictionary with metadata from header comments
    """
    metadata = {}
    try:
        with open(csv_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    # Parse comment line: # key=value
                    line = line[1:].strip()  # Remove # prefix
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Convert to appropriate type
                        if key in ['pattern_length', 'no_of_repetitions', 'anchor_bar_global',
                                   'pattern_start_bar_global', 'section_start_bar_global',
                                   'snippet_start_bar_global', 'patterns_kept', 'bars_kept',
                                   'no_of_repetitions_before']:
                            try:
                                metadata[key] = int(value)
                            except ValueError:
                                metadata[key] = value
                        elif key in ['section_start_absolute', 'section_duration',
                                     'ratio_in_snippet', 'ratio_outside_snippet',
                                     'snippet_start', 'snippet_end', 'filter_threshold']:
                            try:
                                metadata[key] = float(value)
                            except ValueError:
                                metadata[key] = value
                        else:
                            metadata[key] = value
                else:
                    # Stop at first non-comment line (header)
                    break
    except Exception as e:
        print(f"    Warning: Could not read metadata from {csv_path}: {e}")

    return metadata


def discover_anchored_csvs(anchoring_dir: str) -> Dict[int, Dict[int, List[Path]]]:
    """
    Discover all anchored CSV files and organize by section number and pattern length.

    Parameters
    ----------
    anchoring_dir : str
        Directory containing anchored CSV files (6.2_filtered_patterns)

    Returns
    -------
    dict
        Nested dict: {sec_no: {pattern_length: [list of csv paths]}}
    """
    anchoring_path = Path(anchoring_dir)
    result = {}

    for csv_file in anchoring_path.glob("SecNo*_anchored.csv"):
        parsed = parse_anchored_filename(csv_file.name)
        if parsed:
            sec_no = parsed['sec_no']
            pattern_length = parsed['pattern_length']

            if sec_no not in result:
                result[sec_no] = {}
            if pattern_length not in result[sec_no]:
                result[sec_no][pattern_length] = []

            result[sec_no][pattern_length].append(csv_file)

    return result


def extract_rhythm_histogram_from_anchored(
    csv_path: str,
    pattern_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract rhythm histogram and phase statistics from anchored CSV.

    Parameters
    ----------
    csv_path : str
        Path to anchored CSV file
    pattern_length : int
        Pattern length in bars (2 or 4)

    Returns
    -------
    tuple
        (histogram, median_phases, iqr_16th, raw_iqr_phases, onset_counts) where:
        - histogram: onset strength per position (ratio of patterns with onset, 0-1)
        - median_phases: median tick_phase value per position
        - iqr_16th: IQR in 16th note units (for error bars)
        - raw_iqr_phases: raw IQR in tick_phase units
        - onset_counts: raw count of onsets at each position
    """
    try:
        df = pd.read_csv(csv_path, comment='#')

        # Filter to rows with actual onsets (non-null onset_time)
        df_onsets = df[df['onset_time'].notna()].copy()

        num_positions = pattern_length * 16
        histogram = np.zeros(num_positions)
        onset_counts = np.zeros(num_positions, dtype=int)
        median_phases = np.full(num_positions, np.nan)
        iqr_16th = np.full(num_positions, np.nan)
        raw_iqr_phases = np.full(num_positions, np.nan)

        if len(df_onsets) == 0:
            return histogram, median_phases, iqr_16th, raw_iqr_phases, onset_counts

        # Get unique repetition count (based on bar_number patterns)
        # Each pattern has pattern_length bars, repetitions are bar_number // pattern_length
        df['repetition_idx'] = df['bar_number'] // pattern_length
        num_repetitions = df['repetition_idx'].nunique()

        # Calculate 16th-note position within pattern for each row
        ticks = df['tick_16th'].values
        bars = df['bar_number'].values

        # Position within pattern (0-based for indexing)
        bar_in_pattern = bars % pattern_length
        position_in_pattern = bar_in_pattern * 16 + ticks

        df['position'] = position_in_pattern

        # For onset strength: count how many repetitions have an onset at each position
        for pos in range(num_positions):
            pos_df = df_onsets[df_onsets['tick_16th'].values + (df_onsets['bar_number'].values % pattern_length) * 16 == pos]

            if len(pos_df) > 0:
                # Store raw count
                onset_counts[pos] = len(pos_df)
                # Onset strength = ratio of patterns that have onset at this position
                histogram[pos] = len(pos_df) / num_repetitions if num_repetitions > 0 else 0

                # Get tick_phase values for this position
                if 'tick_phase' in pos_df.columns:
                    tick_phases = pos_df['tick_phase'].dropna().values
                    if len(tick_phases) > 0:
                        median_phases[pos] = np.median(tick_phases)

                        if len(tick_phases) > 1:
                            q75, q25 = np.percentile(tick_phases, [75, 25])
                            iqr = q75 - q25
                            raw_iqr_phases[pos] = iqr
                            # Convert IQR to 16th note units (tick_phase is in 16th units)
                            iqr_16th[pos] = iqr * 1.5  # Scale for visibility
                        else:
                            raw_iqr_phases[pos] = 0.0
                            iqr_16th[pos] = 0.0

        return histogram, median_phases, iqr_16th, raw_iqr_phases, onset_counts

    except Exception as e:
        print(f"    Warning: Could not process {csv_path}: {e}")
        num_positions = pattern_length * 16
        return np.zeros(num_positions), np.full(num_positions, np.nan), np.full(num_positions, np.nan), np.full(num_positions, np.nan)


def create_anchored_rhythm_histograms(
    anchoring_dir: str,
    track_id: str,
    output_dir: str,
    groove_pulse_threshold: float = 0.2,
    verbose: bool = True
) -> dict:
    """
    Create rhythm histograms from section-anchored data.

    Layout: 2 rows (L2 patterns top, L4 patterns bottom) x N columns (sections by SecNo)
    CSV output: aggregated by section_id (e.g., "SecNo1_chorus_L2")

    Parameters
    ----------
    anchoring_dir : str
        Directory containing anchored CSV files (6.2_filtered_patterns)
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots
    groove_pulse_threshold : float
        Threshold for groove pulse line (default 0.2)

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    if verbose:
        print(f"\n  [Anchored Rhythm Histograms] Creating from section-anchored data...")

    # Discover all anchored CSV files
    anchored_csvs = discover_anchored_csvs(anchoring_dir)

    if not anchored_csvs:
        if verbose:
            print(f"    No anchored CSV files found in {anchoring_dir}")
        return {'pdf': None, 'png': None, 'csv': None}

    # Get all unique section numbers and pattern lengths
    all_sec_nos = sorted(anchored_csvs.keys())
    all_pattern_lengths = set()
    for sec_data in anchored_csvs.values():
        all_pattern_lengths.update(sec_data.keys())
    all_pattern_lengths = sorted(all_pattern_lengths)

    if verbose:
        print(f"    Found sections: {all_sec_nos}")
        print(f"    Found pattern lengths: {['L' + str(pl) for pl in all_pattern_lengths]}")

    # Create figure: 2 rows (L2 top, L4 bottom) x N columns (sections)
    num_cols = len(all_sec_nos)
    num_rows = len(all_pattern_lengths)

    if num_rows == 0 or num_cols == 0:
        if verbose:
            print(f"    No data to plot")
        return {'pdf': None, 'png': None, 'csv': None}

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)
    fig.suptitle(f'Anchored Rhythm Histograms — {track_id}', fontsize=14, fontweight='bold', y=0.995)

    # Define colors by pattern length
    colors = {2: '#3498DB', 4: '#2ECC71'}

    # CSV data storage
    csv_data = []

    # Create plots
    for row_idx, pattern_length in enumerate(all_pattern_lengths):
        for col_idx, sec_no in enumerate(all_sec_nos):
            ax = axes[row_idx, col_idx]

            # Check if we have data for this combination
            if sec_no not in anchored_csvs or pattern_length not in anchored_csvs[sec_no]:
                ax.text(0.5, 0.5, f'No L{pattern_length} data\nfor SecNo{sec_no}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'SecNo{sec_no} — L{pattern_length}', fontsize=10, fontweight='bold')
                continue

            csv_files = anchored_csvs[sec_no][pattern_length]
            if not csv_files:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            # Use first CSV (should typically be one per section/pattern_length)
            csv_path = csv_files[0]
            parsed = parse_anchored_filename(csv_path.name)
            metadata = read_anchored_csv_metadata(str(csv_path))

            section_label = parsed['section_label'] if parsed else 'unknown'
            section_id = parsed['section_id'] if parsed else f'SecNo{sec_no}_L{pattern_length}'
            num_repetitions = metadata.get('no_of_repetitions', 0)
            filtering_method = metadata.get('filtering_method', '')
            # ratio_in_snippet: fraction of snippet covered by this section (duration_inside_snippet / snippet_duration)
            ratio_in_snippet = metadata.get('ratio_in_snippet', parsed.get('ratio') if parsed else None)
            # mean_section_tempo: average tempo across all patterns in this section
            mean_section_tempo = metadata.get('mean_section_tempo', None)

            # Extract histogram and statistics
            hist, median_phases, iqr_16th, raw_iqr_phases, onset_counts = extract_rhythm_histogram_from_anchored(
                str(csv_path), pattern_length
            )
            num_positions = pattern_length * 16

            # X-axis: base positions (1-based)
            base_positions = np.arange(1, num_positions + 1)

            # Calculate shifted positions based on median tick_phase
            shifted_positions = base_positions.copy().astype(float)
            for i in range(num_positions):
                if not np.isnan(median_phases[i]):
                    # tick_phase is already relative offset in 16th units
                    shifted_positions[i] = base_positions[i] + median_phases[i]

            # Plot bars at shifted positions
            color = colors.get(pattern_length, '#9B59B6')
            bar_width = 0.8
            ax.bar(shifted_positions, hist, width=bar_width,
                  color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add error bars (IQR in 16th note units)
            for i in range(num_positions):
                if not np.isnan(iqr_16th[i]) and iqr_16th[i] > 0 and hist[i] > 0:
                    error_bar_y = hist[i] * 0.9
                    ax.errorbar(shifted_positions[i], error_bar_y,
                               xerr=iqr_16th[i], fmt='none',
                               ecolor='black', capsize=2, capthick=1, linewidth=1)

            # Add relative tick_phase labels on top of bars
            for i in range(num_positions):
                if hist[i] > 0 and not np.isnan(median_phases[i]):
                    # Format without leading zero
                    label_text = f'{median_phases[i]:.2f}'.replace('0.', '.').replace('-0.', '-.')
                    ax.text(shifted_positions[i], hist[i], label_text,
                           ha='center', va='bottom', fontsize=5, rotation=0)

            ax.set_ylabel('Onset Strength', fontsize=9, fontweight='bold')

            # Adjust y-axis scale
            max_strength = np.max(hist) if np.any(hist > 0) else 1.0
            ax.set_ylim(0, max_strength * 1.2)

            # Add secondary y-axis for counts
            # Scale proportionally to left axis: if max_strength = 0.5 and num_repetitions = 4,
            # the right axis should show 0 to 2 (= 0.5 * 4), not 0 to 4
            ax2 = ax.twinx()
            max_count = num_repetitions if num_repetitions > 0 else 1
            ax2.set_ylim(0, max_strength * max_count * 1.2)
            ax2.set_ylabel('Count', fontsize=9, fontweight='bold', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')

            # Add horizontal threshold line
            threshold_value = groove_pulse_threshold
            ax.axhline(y=threshold_value, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.7)

            # Calculate statistics
            occupied_positions = int(np.sum(hist > 0))

            # Build title
            title = f'SecNo{sec_no} — {section_label} — L{pattern_length}'
            if num_repetitions:
                title += f' — {num_repetitions} reps'
            title += f' — {occupied_positions}/{num_positions} pos'
            # Add ratio_in_snippet (fraction of snippet covered by this section)
            if ratio_in_snippet is not None:
                title += f' — {ratio_in_snippet:.1%} of snippet'

            ax.set_title(title, fontsize=9, fontweight='bold', pad=5)
            ax.grid(True, alpha=0.3, axis='y')

            # Add vertical lines at bar boundaries
            ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=10)
            for bar_idx in range(1, pattern_length):
                ax.axvline(x=bar_idx * 16 + 1, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.5, zorder=10)

            # Add vertical grid lines at expected 16th note positions
            for i in range(num_positions):
                ax.axvline(x=base_positions[i], color='gray', linestyle=':',
                          linewidth=0.6, alpha=0.4, zorder=1)

            # Set x-axis limits and ticks
            ax.set_xlim(0, num_positions + 1)
            ax.set_xticks(base_positions)
            ax.tick_params(axis='x', labelsize=6, rotation=90)

            # Only show x-axis label on bottom row
            if row_idx == num_rows - 1:
                ax.set_xlabel('16th-note position', fontsize=8, fontweight='bold')

            if verbose:
                print(f"    {section_id}: {occupied_positions}/{num_positions} positions, {num_repetitions} reps")

            # Store CSV data
            for pos_idx in range(num_positions):
                csv_data.append({
                    'section_id': section_id,
                    'sec_no': sec_no,
                    'section_label': section_label,
                    'pattern_length': pattern_length,
                    'num_repetitions': num_repetitions,
                    'ratio_in_snippet': ratio_in_snippet,
                    'mean_section_tempo': mean_section_tempo,
                    'position': pos_idx + 1,  # 1-based
                    'onset_count': int(onset_counts[pos_idx]),
                    'onset_strength': float(hist[pos_idx]),
                    'median_tick_phase': float(median_phases[pos_idx]) if not np.isnan(median_phases[pos_idx]) else None,
                    'iqr_tick_phase': float(raw_iqr_phases[pos_idx]) if not np.isnan(raw_iqr_phases[pos_idx]) else None,
                    'iqr_16th': float(iqr_16th[pos_idx]) if not np.isnan(iqr_16th[pos_idx]) else None
                })

    plt.tight_layout()

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / f'{track_id}_anchored_rhythm_histograms.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    if verbose:
        print(f"    Saved: {output_pdf}")

    output_png = output_dir / f'{track_id}_anchored_rhythm_histograms.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"    Saved: {output_png}")

    # Save CSV
    output_csv = None
    if csv_data:
        df_out = pd.DataFrame(csv_data)
        output_csv = output_dir / f'{track_id}_anchored_rhythm_histograms.csv'
        df_out.to_csv(output_csv, index=False)
        if verbose:
            print(f"    Saved: {output_csv}")

    plt.close()

    return {
        'pdf': str(output_pdf),
        'png': str(output_png),
        'csv': str(output_csv) if output_csv else None
    }


def create_anchored_groove_pulse_histograms(
    rhythm_histograms_csv: str,
    track_id: str,
    output_dir: str,
    groove_pulse_threshold: float = 0.2,
    verbose: bool = True
) -> dict:
    """
    Create groove pulse histograms by filtering weak onsets from rhythm histogram data.

    Reads the anchored rhythm histograms CSV output and filters onset strengths
    below groove_pulse_threshold, then displays only strong onsets.
    Layout: 2 rows (L2 patterns top, L4 patterns bottom) x N columns (sections by SecNo)

    Parameters
    ----------
    rhythm_histograms_csv : str
        Path to the anchored rhythm histograms CSV (output from create_anchored_rhythm_histograms)
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots
    groove_pulse_threshold : float
        Threshold for filtering weak onsets (default 0.2)
    verbose : bool
        Whether to print progress messages (default True)

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    if verbose:
        print(f"\n  [Anchored Groove Pulse Histograms] Creating from rhythm histograms CSV...")

    # Read the rhythm histograms CSV
    csv_path = Path(rhythm_histograms_csv)
    if not csv_path.exists():
        if verbose:
            print(f"    Error: Rhythm histograms CSV not found: {rhythm_histograms_csv}")
        return {'pdf': None, 'png': None, 'csv': None}

    df = pd.read_csv(csv_path)

    # Get unique section_ids and pattern_lengths
    all_sec_nos = sorted(df['sec_no'].unique())
    all_pattern_lengths = sorted(df['pattern_length'].unique())

    if verbose:
        print(f"    Found sections: {all_sec_nos}")
        print(f"    Found pattern lengths: {['L' + str(pl) for pl in all_pattern_lengths]}")

    # Create figure: 2 rows (L2 top, L4 bottom) x N columns (sections)
    num_cols = len(all_sec_nos)
    num_rows = len(all_pattern_lengths)

    if num_rows == 0 or num_cols == 0:
        if verbose:
            print(f"    No data to plot")
        return {'pdf': None, 'png': None, 'csv': None}

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)
    fig.suptitle(f'Anchored Groove Pulse Histograms (Filtered ≥{groove_pulse_threshold}) — {track_id}',
                 fontsize=14, fontweight='bold', y=0.995)

    # Define colors by pattern length
    colors = {2: '#E67E22', 4: '#9B59B6'}

    # CSV data storage
    csv_data = []

    # Create plots
    for row_idx, pattern_length in enumerate(all_pattern_lengths):
        for col_idx, sec_no in enumerate(all_sec_nos):
            ax = axes[row_idx, col_idx]

            # Get data for this section and pattern length
            section_df = df[(df['sec_no'] == sec_no) & (df['pattern_length'] == pattern_length)]

            if len(section_df) == 0:
                ax.text(0.5, 0.5, f'No L{pattern_length} data\nfor SecNo{sec_no}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'SecNo{sec_no} — L{pattern_length}', fontsize=10, fontweight='bold')
                continue

            # Get section info from first row
            section_label = section_df['section_label'].iloc[0]
            section_id = section_df['section_id'].iloc[0]
            num_repetitions = section_df['num_repetitions'].iloc[0]
            ratio_in_snippet = section_df['ratio_in_snippet'].iloc[0] if 'ratio_in_snippet' in section_df.columns else None
            mean_section_tempo = section_df['mean_section_tempo'].iloc[0] if 'mean_section_tempo' in section_df.columns else None
            num_positions = pattern_length * 16

            # Extract arrays from dataframe
            onset_strength = section_df['onset_strength'].values
            onset_counts = section_df['onset_count'].values if 'onset_count' in section_df.columns else np.zeros(len(section_df), dtype=int)
            median_phases = section_df['median_tick_phase'].values
            iqr_16th = section_df['iqr_16th'].values

            # Apply groove pulse threshold filter
            filtered_mask = onset_strength >= groove_pulse_threshold
            filtered_hist = np.where(filtered_mask, onset_strength, 0)
            filtered_counts = np.where(filtered_mask, onset_counts, 0)
            filtered_median_phases = np.where(filtered_mask, median_phases, np.nan)
            filtered_iqr_16th = np.where(filtered_mask, iqr_16th, np.nan)

            # X-axis: base positions (1-based)
            base_positions = np.arange(1, num_positions + 1)

            # Calculate shifted positions based on median tick_phase
            shifted_positions = base_positions.copy().astype(float)
            for i in range(num_positions):
                if not np.isnan(filtered_median_phases[i]):
                    shifted_positions[i] = base_positions[i] + filtered_median_phases[i]

            # Plot bars at shifted positions
            color = colors.get(pattern_length, '#9B59B6')
            bar_width = 0.8
            ax.bar(shifted_positions, filtered_hist, width=bar_width,
                  color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add error bars (IQR in 16th note units)
            for i in range(num_positions):
                if not np.isnan(filtered_iqr_16th[i]) and filtered_iqr_16th[i] > 0 and filtered_hist[i] > 0:
                    error_bar_y = filtered_hist[i] * 0.9
                    ax.errorbar(shifted_positions[i], error_bar_y,
                               xerr=filtered_iqr_16th[i], fmt='none',
                               ecolor='black', capsize=2, capthick=1, linewidth=1)

            # Add relative tick_phase labels on top of bars
            for i in range(num_positions):
                if filtered_hist[i] > 0 and not np.isnan(filtered_median_phases[i]):
                    label_text = f'{filtered_median_phases[i]:.2f}'.replace('0.', '.').replace('-0.', '-.')
                    ax.text(shifted_positions[i], filtered_hist[i], label_text,
                           ha='center', va='bottom', fontsize=5, rotation=0)

            ax.set_ylabel('Onset Strength (Filtered)', fontsize=9, fontweight='bold')

            # Adjust y-axis scale
            max_strength = np.max(filtered_hist) if np.any(filtered_hist > 0) else 1.0
            ax.set_ylim(0, max(max_strength * 1.2, 1.0))

            # Add secondary y-axis for counts
            # Scale proportionally to left axis: if max_strength = 0.5 and num_repetitions = 4,
            # the right axis should show 0 to 2 (= 0.5 * 4), not 0 to 4
            ax2 = ax.twinx()
            max_count = num_repetitions if num_repetitions > 0 else 1
            ax2.set_ylim(0, max_strength * max_count * 1.2)
            ax2.set_ylabel('Count', fontsize=9, fontweight='bold', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')

            # Calculate statistics
            original_positions = int(np.sum(onset_strength > 0))
            filtered_positions = int(np.sum(filtered_hist > 0))

            # Build title
            title = f'SecNo{sec_no} — {section_label} — L{pattern_length}'
            if num_repetitions:
                title += f' — {num_repetitions} reps'
            title += f' — {filtered_positions}/{original_positions} pos (≥{groove_pulse_threshold})'
            # Add ratio_in_snippet (fraction of snippet covered by this section)
            if ratio_in_snippet is not None and not pd.isna(ratio_in_snippet):
                title += f' — {ratio_in_snippet:.1%} of snippet'

            ax.set_title(title, fontsize=9, fontweight='bold', pad=5)
            ax.grid(True, alpha=0.3, axis='y')

            # Add vertical lines at bar boundaries
            ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=10)
            for bar_idx in range(1, pattern_length):
                ax.axvline(x=bar_idx * 16 + 1, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.5, zorder=10)

            # Add vertical grid lines at expected 16th note positions
            for i in range(num_positions):
                ax.axvline(x=base_positions[i], color='gray', linestyle=':',
                          linewidth=0.6, alpha=0.4, zorder=1)

            # Set x-axis limits and ticks
            ax.set_xlim(0, num_positions + 1)
            ax.set_xticks(base_positions)
            ax.tick_params(axis='x', labelsize=6, rotation=90)

            # Only show x-axis label on bottom row
            if row_idx == num_rows - 1:
                ax.set_xlabel('16th-note position', fontsize=8, fontweight='bold')

            if verbose:
                print(f"    {section_id}: {filtered_positions}/{original_positions} positions (filtered), {num_repetitions} reps")

            # Store CSV data
            for pos_idx in range(num_positions):
                iqr_tick_val = section_df['iqr_tick_phase'].values[pos_idx] if 'iqr_tick_phase' in section_df.columns else None
                csv_data.append({
                    'section_id': section_id,
                    'sec_no': sec_no,
                    'section_label': section_label,
                    'pattern_length': pattern_length,
                    'num_repetitions': num_repetitions,
                    'ratio_in_snippet': ratio_in_snippet,
                    'mean_section_tempo': mean_section_tempo,
                    'position': pos_idx + 1,
                    'onset_count_original': int(onset_counts[pos_idx]),
                    'onset_count_filtered': int(filtered_counts[pos_idx]),
                    'onset_strength_original': float(onset_strength[pos_idx]),
                    'onset_strength_filtered': float(filtered_hist[pos_idx]),
                    'median_tick_phase': float(filtered_median_phases[pos_idx]) if not np.isnan(filtered_median_phases[pos_idx]) else None,
                    'iqr_tick_phase': float(iqr_tick_val) if iqr_tick_val is not None and not pd.isna(iqr_tick_val) else None,
                    'iqr_16th': float(filtered_iqr_16th[pos_idx]) if not np.isnan(filtered_iqr_16th[pos_idx]) else None,
                    'threshold': groove_pulse_threshold
                })

    plt.tight_layout()

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / f'{track_id}_anchored_groove_pulse_histograms.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    if verbose:
        print(f"    Saved: {output_pdf}")

    output_png = output_dir / f'{track_id}_anchored_groove_pulse_histograms.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"    Saved: {output_png}")

    # Save CSV
    output_csv = None
    if csv_data:
        df_out = pd.DataFrame(csv_data)
        output_csv = output_dir / f'{track_id}_anchored_groove_pulse_histograms.csv'
        df_out.to_csv(output_csv, index=False)
        if verbose:
            print(f"    Saved: {output_csv}")

    plt.close()

    return {
        'pdf': str(output_pdf),
        'png': str(output_png),
        'csv': str(output_csv) if output_csv else None
    }


def create_anchored_rhythm_patterns(
    groove_pulse_csv: str,
    track_id: str,
    output_dir: str,
    binary_threshold: float = 0.5,
    verbose: bool = True
) -> dict:
    """
    Create binary rhythm pattern visualizations from groove pulse data.

    Reads the anchored groove pulse histograms CSV output and creates binary patterns:
    - onset_strength_filtered >= binary_threshold -> bar height 1.0
    - 0 < onset_strength_filtered < binary_threshold -> bar height 0.5
    - onset_strength_filtered == 0 -> no bar

    Layout: 2 rows (L2 patterns top, L4 patterns bottom) x N columns (sections by SecNo)

    Parameters
    ----------
    groove_pulse_csv : str
        Path to the anchored groove pulse histograms CSV
    track_id : str
        Track identifier for plot title
    output_dir : str
        Output directory for saving plots
    binary_threshold : float
        Threshold for full vs half pattern level (default 0.5)
    verbose : bool
        Whether to print progress messages (default True)

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    if verbose:
        print(f"\n  [Anchored Rhythm Patterns] Creating from groove pulse CSV...")

    # Read the groove pulse CSV
    csv_path = Path(groove_pulse_csv)
    if not csv_path.exists():
        if verbose:
            print(f"    Error: Groove pulse CSV not found: {groove_pulse_csv}")
        return {'pdf': None, 'png': None, 'csv': None}

    df = pd.read_csv(csv_path)

    # Get unique section_ids and pattern_lengths
    all_sec_nos = sorted(df['sec_no'].unique())
    all_pattern_lengths = sorted(df['pattern_length'].unique())

    if verbose:
        print(f"    Found sections: {all_sec_nos}")
        print(f"    Found pattern lengths: {['L' + str(pl) for pl in all_pattern_lengths]}")

    # Create figure: 2 rows (L2 top, L4 bottom) x N columns (sections)
    num_cols = len(all_sec_nos)
    num_rows = len(all_pattern_lengths)

    if num_rows == 0 or num_cols == 0:
        if verbose:
            print(f"    No data to plot")
        return {'pdf': None, 'png': None, 'csv': None}

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows), squeeze=False)
    fig.suptitle(f'Anchored Rhythm Patterns (Binary: ≥{binary_threshold}=1.0) — {track_id}',
                 fontsize=14, fontweight='bold', y=0.995)

    # Define colors by pattern length
    colors = {2: '#2ECC71', 4: '#F39C12'}

    # CSV data storage
    csv_data = []

    # Create plots
    for row_idx, pattern_length in enumerate(all_pattern_lengths):
        for col_idx, sec_no in enumerate(all_sec_nos):
            ax = axes[row_idx, col_idx]

            # Get data for this section and pattern length
            section_df = df[(df['sec_no'] == sec_no) & (df['pattern_length'] == pattern_length)]

            if len(section_df) == 0:
                ax.text(0.5, 0.5, f'No L{pattern_length} data\nfor SecNo{sec_no}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'SecNo{sec_no} — L{pattern_length}', fontsize=10, fontweight='bold')
                continue

            # Get section info from first row
            section_label = section_df['section_label'].iloc[0]
            section_id = section_df['section_id'].iloc[0]
            num_repetitions = section_df['num_repetitions'].iloc[0]
            ratio_in_snippet = section_df['ratio_in_snippet'].iloc[0] if 'ratio_in_snippet' in section_df.columns else None
            mean_section_tempo = section_df['mean_section_tempo'].iloc[0] if 'mean_section_tempo' in section_df.columns else None
            num_positions = pattern_length * 16

            # Extract arrays from dataframe
            onset_strength = section_df['onset_strength_filtered'].values
            median_phases = section_df['median_tick_phase'].values
            iqr_16th = section_df['iqr_16th'].values

            # Binarization from RAW onset counts (per section x pattern-length group).
            # Single events (count == 1) are dropped as noise; the strong/weak split
            # sits at ((maxCount - 1) / 2) + 1 == (maxCount + 1) / 2 of the raw counts.
            # Uses onset_count_original (unfiltered), so this floor replaces the relative
            # groove-pulse filter for the pattern stage. Falls back to the old
            # strength-based split for older CSVs without raw counts.
            use_counts = 'onset_count_original' in section_df.columns
            if use_counts:
                onset_counts = section_df['onset_count_original'].values
                max_count = onset_counts.max() if len(onset_counts) else 0
                strong_threshold = (max_count + 1) / 2

            # Create binary pattern values
            pattern_values = np.zeros(num_positions)
            for i in range(num_positions):
                if use_counts:
                    # Count-based ternary: drop singletons, split at (maxCount + 1) / 2
                    count = onset_counts[i]
                    if count <= 1:
                        pattern_values[i] = 0.0          # noise floor (count 0 or 1)
                    elif count > strong_threshold:
                        pattern_values[i] = 1.0          # strong
                    else:
                        pattern_values[i] = 0.5          # weak (2 <= count <= threshold)
                else:
                    # Fallback: strength-based split (>= threshold -> 1.0, 0 < s -> 0.5)
                    if onset_strength[i] >= binary_threshold:
                        pattern_values[i] = 1.0
                    elif onset_strength[i] > 0:
                        pattern_values[i] = 0.5
                    else:
                        pattern_values[i] = 0.0

            # X-axis: base positions (1-based)
            base_positions = np.arange(1, num_positions + 1)

            # Calculate shifted positions based on median tick_phase
            shifted_positions = base_positions.copy().astype(float)
            for i in range(num_positions):
                if not np.isnan(median_phases[i]):
                    shifted_positions[i] = base_positions[i] + median_phases[i]

            # Plot bars at shifted positions
            color = colors.get(pattern_length, '#9B59B6')
            bar_width = 0.8
            for i in range(num_positions):
                if pattern_values[i] > 0:
                    ax.bar(shifted_positions[i], pattern_values[i], width=bar_width,
                          color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add error bars (IQR in 16th note units)
            for i in range(num_positions):
                if not np.isnan(iqr_16th[i]) and iqr_16th[i] > 0 and pattern_values[i] > 0:
                    error_bar_y = pattern_values[i] * 0.9
                    ax.errorbar(shifted_positions[i], error_bar_y,
                               xerr=iqr_16th[i], fmt='none',
                               ecolor='black', capsize=2, capthick=1, linewidth=1)

            # Add relative tick_phase labels on top of bars
            for i in range(num_positions):
                if pattern_values[i] > 0 and not np.isnan(median_phases[i]):
                    label_text = f'{median_phases[i]:.2f}'.replace('0.', '.').replace('-0.', '-.')
                    ax.text(shifted_positions[i], pattern_values[i], label_text,
                           ha='center', va='bottom', fontsize=5, rotation=0)

            ax.set_ylabel('Pattern Level', fontsize=9, fontweight='bold')

            # Set y-axis limits and ticks
            ax.set_ylim(0, 1.3)
            ax.set_yticks([0.5, 1.0])

            # Calculate statistics
            strong_positions = int(np.sum(pattern_values == 1.0))
            weak_positions = int(np.sum(pattern_values == 0.5))
            total_positions = strong_positions + weak_positions

            # Build title
            title = f'SecNo{sec_no} — {section_label} — L{pattern_length}'
            if num_repetitions:
                title += f' — {num_repetitions} reps'
            title += f' — {strong_positions}+{weak_positions}={total_positions} pos'
            # Add ratio_in_snippet (fraction of snippet covered by this section)
            if ratio_in_snippet is not None and not pd.isna(ratio_in_snippet):
                title += f' — {ratio_in_snippet:.1%} of snippet'

            ax.set_title(title, fontsize=9, fontweight='bold', pad=5)
            ax.grid(True, alpha=0.3, axis='y')

            # Add vertical lines at bar boundaries
            ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=10)
            for bar_idx in range(1, pattern_length):
                ax.axvline(x=bar_idx * 16 + 1, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.5, zorder=10)

            # Add vertical grid lines at expected 16th note positions
            for i in range(num_positions):
                ax.axvline(x=base_positions[i], color='gray', linestyle=':',
                          linewidth=0.6, alpha=0.4, zorder=1)

            # Set x-axis limits and ticks
            ax.set_xlim(0, num_positions + 1)
            ax.set_xticks(base_positions)
            ax.tick_params(axis='x', labelsize=6, rotation=90)

            # Only show x-axis label on bottom row
            if row_idx == num_rows - 1:
                ax.set_xlabel('16th-note position', fontsize=8, fontweight='bold')

            if verbose:
                print(f"    {section_id}: {strong_positions} strong + {weak_positions} weak = {total_positions} positions")

            # Store CSV data
            for pos_idx in range(num_positions):
                iqr_tick_val = section_df['iqr_tick_phase'].values[pos_idx] if 'iqr_tick_phase' in section_df.columns else None
                csv_data.append({
                    'section_id': section_id,
                    'sec_no': sec_no,
                    'section_label': section_label,
                    'pattern_length': pattern_length,
                    'num_repetitions': num_repetitions,
                    'ratio_in_snippet': ratio_in_snippet,
                    'mean_section_tempo': mean_section_tempo,
                    'position': pos_idx + 1,
                    'onset_strength_filtered': float(onset_strength[pos_idx]),
                    'pattern_value': float(pattern_values[pos_idx]),
                    'median_tick_phase': float(median_phases[pos_idx]) if not np.isnan(median_phases[pos_idx]) else None,
                    'iqr_tick_phase': float(iqr_tick_val) if iqr_tick_val is not None and not pd.isna(iqr_tick_val) else None,
                    'iqr_16th': float(iqr_16th[pos_idx]) if not np.isnan(iqr_16th[pos_idx]) else None,
                    'binary_threshold': binary_threshold
                })

    plt.tight_layout()

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / f'{track_id}_anchored_rhythm_patterns.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    if verbose:
        print(f"    Saved: {output_pdf}")

    output_png = output_dir / f'{track_id}_anchored_rhythm_patterns.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"    Saved: {output_png}")

    # Save CSV
    output_csv = None
    if csv_data:
        df_out = pd.DataFrame(csv_data)
        output_csv = output_dir / f'{track_id}_anchored_rhythm_patterns.csv'
        df_out.to_csv(output_csv, index=False)
        if verbose:
            print(f"    Saved: {output_csv}")

    plt.close()

    return {
        'pdf': str(output_pdf),
        'png': str(output_png),
        'csv': str(output_csv) if output_csv else None
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python anchored_rhythm_histograms.py <anchoring_dir> <track_id> <output_dir>')
        print('  anchoring_dir: Directory containing anchored CSV files (6.2_filtered_patterns)')
        print('  track_id: Track identifier for plot titles')
        print('  output_dir: Output directory for saving plots')
        sys.exit(1)

    anchoring_dir = sys.argv[1]
    track_id = sys.argv[2]
    output_dir = sys.argv[3]

    create_anchored_rhythm_histograms(anchoring_dir, track_id, output_dir)
