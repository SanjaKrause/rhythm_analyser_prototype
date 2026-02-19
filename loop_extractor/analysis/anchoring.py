"""
Section-anchored onset grid calculation.

This module creates onset grids anchored to SongFormer section boundaries,
using the same per-pattern onset anchoring logic as FlexStart but starting
from the bar nearest to each section boundary.

Output folder: 6.1_anchoring
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Import config from parent directory
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

import importlib.util
spec = importlib.util.spec_from_file_location("config_module", _parent_dir / "config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config


# ============================================================================
# CONFIGURATION (standalone - no raster.py dependency)
# ============================================================================

GRID_SUBDIV_PER_BEAT = 4  # Sixteenth notes
MAX_MATCH_FRAC_BEFORE = 0.49  # Max distance before grid position (prevents overlap)
MAX_MATCH_FRAC_AFTER = 0.51  # Max distance after grid position (prevents overlap)
SEARCH_WINDOW_START_PHASE = 0.5  # Search window before 1/16th for reference onset
SEARCH_WINDOW_END_PHASE = 0.75  # Search window after 1/16th for reference onset


# ============================================================================
# INPUT PARSING (standalone - no raster.py dependency)
# ============================================================================

def parse_corrected_downbeats(corrected_file: str) -> Tuple[List[float], int]:
    """
    Parse corrected downbeats file.

    Parameters
    ----------
    corrected_file : str
        Path to corrected downbeats file

    Returns
    -------
    tuple
        (downbeat_times: List[float], time_signature: int)
    """
    # Read file to get metadata
    with open(corrected_file, 'r') as f:
        lines = f.readlines()

    # Parse metadata
    tsig = 4  # default
    for line in lines:
        if line.startswith('# time_signature='):
            tsig = int(line.split('=')[1].strip())
            break

    # Read data
    df = pd.read_csv(corrected_file, sep='\t', comment='#')

    # Extract downbeat times
    downbeat_times = df['corrected_downbeat_time(s)'].tolist()

    # Add final downbeat (end of last bar)
    if 'next_downbeat_time(s)' in df.columns:
        downbeat_times.append(df['next_downbeat_time(s)'].iloc[-1])

    return downbeat_times, tsig


def load_onsets(onset_file: str) -> np.ndarray:
    """
    Load onset times from CSV file.

    Parameters
    ----------
    onset_file : str
        Path to onset CSV file

    Returns
    -------
    np.ndarray
        Array of onset times in seconds
    """
    df = pd.read_csv(onset_file)

    # Try common column names
    if 'onset_times' in df.columns:
        return df['onset_times'].values
    elif 'onset_time' in df.columns:
        return df['onset_time'].values
    else:
        # Use first column
        return df.iloc[:, 0].values


def find_anchor_bar(
    section_start_time: float,
    downbeats: List[float],
    first_bar: int,
    last_bar: int,
    tolerance: float = None
) -> Optional[int]:
    """
    Find the bar index nearest to a section boundary.

    Parameters
    ----------
    section_start_time : float
        Absolute time of the section start
    downbeats : List[float]
        List of all downbeat times (global indexing)
    first_bar : int
        First bar index in the snippet
    last_bar : int
        Last bar index in the snippet
    tolerance : float
        Maximum fraction of bar duration for a valid anchor (default from config)

    Returns
    -------
    Optional[int]
        Bar index (global) of the nearest bar to section boundary, or None if
        no bar is within tolerance
    """
    if tolerance is None:
        tolerance = config.ANCHOR_BAR_TOLERANCE

    best_bar = None
    best_distance = float('inf')

    for bar_idx in range(first_bar, last_bar + 1):
        if bar_idx >= len(downbeats) - 1:
            break

        bar_time = downbeats[bar_idx]
        bar_duration = downbeats[bar_idx + 1] - downbeats[bar_idx]

        # Calculate distance as fraction of bar duration
        distance = abs(section_start_time - bar_time)
        distance_frac = distance / bar_duration

        if distance_frac <= tolerance and distance_frac < best_distance:
            best_distance = distance_frac
            best_bar = bar_idx

    return best_bar


def calculate_anchored_phases(
    onsets: np.ndarray,
    downbeats: List[float],
    anchor_bar: int,
    section_end_time: float,
    snippet_end_time: float,
    pattern_len: int,
    steps_per_bar: int = 16,
    search_window_start_phase: float = SEARCH_WINDOW_START_PHASE,
    search_window_end_phase: float = SEARCH_WINDOW_END_PHASE
) -> pd.DataFrame:
    """
    Calculate phases with per-pattern anchoring, starting from anchor_bar.

    Uses the same logic as FlexStart: for each L-bar pattern, find an onset
    near tick 0 and shift the grid by that offset.

    Parameters
    ----------
    onsets : np.ndarray
        Onset times
    downbeats : List[float]
        Downbeat times (global indexing)
    anchor_bar : int
        Bar index (global) to start from
    section_end_time : float
        End time of the section (absolute)
    snippet_end_time : float
        End time of the snippet (absolute)
    pattern_len : int
        Pattern length in bars (2 or 4)
    steps_per_bar : int
        Number of grid positions per bar (default 16)
    search_window_start_phase : float
        Search window before grid position for reference onset
    search_window_end_phase : float
        Search window after grid position for reference onset

    Returns
    -------
    pd.DataFrame
        Anchored onset data with columns:
        bar_number, bar_number_global, tick_16th, onset_time, phase, grid_time, grid_phase, tick_phase
    """
    rows = []

    # Determine the effective end time (minimum of section end and snippet end)
    effective_end_time = min(section_end_time, snippet_end_time)

    # Process every L-bar segment starting from anchor_bar
    segment_idx = 0
    segment_start = anchor_bar

    while segment_start + pattern_len <= len(downbeats) - 1:
        # Calculate equidistant grid for THIS segment
        segment_start_time = downbeats[segment_start]
        segment_end_time = downbeats[segment_start + pattern_len]

        # Stop if segment starts beyond effective end time
        if segment_start_time >= effective_end_time:
            break

        # Skip incomplete patterns (those that extend beyond effective end)
        if segment_end_time > effective_end_time:
            break

        segment_duration = segment_end_time - segment_start_time
        total_sixteenths = pattern_len * steps_per_bar
        sixteenth_duration = segment_duration / total_sixteenths

        # Find reference onset for THIS segment at tick 0 of segment start
        segment_grid_time = segment_start_time
        window_start = segment_grid_time - search_window_start_phase * sixteenth_duration
        window_end = segment_grid_time + search_window_end_phase * sixteenth_duration
        onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

        if len(onsets_in_window) > 0:
            # Find closest onset to grid_time
            distances = np.abs(onsets_in_window - segment_grid_time)
            min_idx = np.argmin(distances)
            nearest_onset = onsets_in_window[min_idx]
            segment_ref_offset_s = nearest_onset - segment_grid_time
        else:
            # No reference found for this segment, use 0
            segment_ref_offset_s = 0.0

        # Process all bars in this L-bar segment
        for bar_offset in range(pattern_len):
            bar_idx = segment_start + bar_offset

            if bar_idx >= len(downbeats) - 1:
                break

            # Calculate equidistant bar boundaries within the segment
            bar_sixteenth_pos = bar_offset * steps_per_bar
            equi_bar_start = segment_start_time + (bar_sixteenth_pos * sixteenth_duration)
            equi_bar_duration = steps_per_bar * sixteenth_duration
            equi_bar_end = equi_bar_start + equi_bar_duration

            # Apply segment-specific correction to grid positions
            corrected_equi_bar_start = equi_bar_start + segment_ref_offset_s

            # Filter onsets using UNCORRECTED equidistant boundaries (avoid gaps)
            bar_onsets = onsets[(onsets >= equi_bar_start) & (onsets < equi_bar_end)]

            for onset_time in bar_onsets:
                # Calculate corrected phase (relative to segment-corrected equidistant grid)
                phase = (onset_time - corrected_equi_bar_start) / equi_bar_duration

                # Assign to nearest tick
                nearest_tick = int(round(phase * steps_per_bar))
                nearest_tick = max(0, min(steps_per_bar - 1, nearest_tick))

                # Check if within tolerance (asymmetric boundaries)
                grid_time = corrected_equi_bar_start + (nearest_tick / steps_per_bar) * equi_bar_duration
                distance = abs(onset_time - grid_time)

                if onset_time < grid_time:
                    tolerance = MAX_MATCH_FRAC_BEFORE * sixteenth_duration
                else:
                    tolerance = MAX_MATCH_FRAC_AFTER * sixteenth_duration

                if distance <= tolerance:
                    # Calculate grid_phase for this tick (0.0-1.0 within bar)
                    grid_phase = nearest_tick / steps_per_bar

                    # Calculate tick_phase: phase within the 16th note (0.0-1.0)
                    tick_phase = (phase - grid_phase) * steps_per_bar

                    rows.append({
                        'bar_number': bar_offset + (segment_idx * pattern_len),
                        'bar_number_global': bar_idx,
                        'tick_16th': nearest_tick,
                        'onset_time': onset_time,
                        'phase': phase,
                        'grid_time': grid_time,
                        'grid_phase': grid_phase,
                        'tick_phase': tick_phase
                    })

        segment_idx += 1
        segment_start += pattern_len

    return pd.DataFrame(rows)


def build_complete_grid(
    df_onsets: pd.DataFrame,
    pattern_len: int,
    anchor_bar: int,
    downbeats: List[float],
    section_end_time: float,
    snippet_end_time: float,
    steps_per_bar: int = 16
) -> pd.DataFrame:
    """
    Build a complete grid with all tick positions, merging in onset data.

    Creates a row for every tick position (16 per bar), with onset_time/phase/tick_phase
    filled where onsets exist and empty otherwise.

    Parameters
    ----------
    df_onsets : pd.DataFrame
        DataFrame with onset matches from calculate_anchored_phases
    pattern_len : int
        Pattern length in bars
    anchor_bar : int
        Starting bar index (global)
    downbeats : List[float]
        Downbeat times
    section_end_time : float
        Section end time
    snippet_end_time : float
        Snippet end time
    steps_per_bar : int
        Grid positions per bar (default 16)

    Returns
    -------
    pd.DataFrame
        Complete grid with all tick positions
    """
    effective_end_time = min(section_end_time, snippet_end_time)

    # Determine how many complete patterns we have
    complete_patterns = 0
    segment_start = anchor_bar

    while segment_start + pattern_len <= len(downbeats) - 1:
        segment_end_time = downbeats[segment_start + pattern_len]
        if segment_end_time > effective_end_time:
            break
        complete_patterns += 1
        segment_start += pattern_len

    if complete_patterns == 0:
        return pd.DataFrame()

    # Build complete grid
    rows = []
    segment_idx = 0
    segment_start = anchor_bar

    while segment_idx < complete_patterns:
        segment_start_time = downbeats[segment_start]
        segment_end_time = downbeats[segment_start + pattern_len]
        segment_duration = segment_end_time - segment_start_time
        total_sixteenths = pattern_len * steps_per_bar
        sixteenth_duration = segment_duration / total_sixteenths

        for bar_offset in range(pattern_len):
            bar_idx = segment_start + bar_offset
            bar_sixteenth_pos = bar_offset * steps_per_bar
            equi_bar_start = segment_start_time + (bar_sixteenth_pos * sixteenth_duration)
            equi_bar_duration = steps_per_bar * sixteenth_duration

            for tick in range(steps_per_bar):
                grid_time = equi_bar_start + (tick / steps_per_bar) * equi_bar_duration
                grid_phase = tick / steps_per_bar
                bar_number = bar_offset + (segment_idx * pattern_len)

                rows.append({
                    'bar_number': bar_number,
                    'bar_number_global': bar_idx,
                    'tick_16th': tick,
                    'onset_time': None,
                    'phase': None,
                    'grid_time': grid_time,
                    'grid_phase': grid_phase,
                    'tick_phase': None
                })

        segment_idx += 1
        segment_start += pattern_len

    df_grid = pd.DataFrame(rows)

    # Merge onset data into grid
    if not df_onsets.empty:
        # Create key for matching
        df_grid['key'] = df_grid['bar_number'].astype(str) + '_' + df_grid['tick_16th'].astype(str)
        df_onsets_copy = df_onsets.copy()
        df_onsets_copy['key'] = df_onsets_copy['bar_number'].astype(str) + '_' + df_onsets_copy['tick_16th'].astype(str)

        # Update onset values where they exist
        for _, onset_row in df_onsets_copy.iterrows():
            mask = df_grid['key'] == onset_row['key']
            if mask.any():
                df_grid.loc[mask, 'onset_time'] = onset_row['onset_time']
                df_grid.loc[mask, 'phase'] = onset_row['phase']
                df_grid.loc[mask, 'tick_phase'] = onset_row['tick_phase']

        df_grid = df_grid.drop(columns=['key'])

    return df_grid


def run_anchoring(
    corrected_downbeats_file: str,
    onset_file: str,
    songformer_sections_csv: str,
    snippet_timings_csv: str,
    output_dir: str,
    pattern_lengths: List[int] = [2, 4],
    verbose: bool = True
) -> Dict[str, str]:
    """
    Run section-anchored grid calculation for all overlapping sections.

    Parameters
    ----------
    corrected_downbeats_file : str
        Path to corrected downbeats file
    onset_file : str
        Path to onset detection CSV
    songformer_sections_csv : str
        Path to SF_overlapping_sections.csv
    snippet_timings_csv : str
        Path to SF_snippet_timings.csv
    output_dir : str
        Output directory for anchored CSVs
    pattern_lengths : List[int]
        Pattern lengths to process (default [2, 4])
    verbose : bool
        Print progress information

    Returns
    -------
    Dict[str, str]
        Dictionary with output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Parse inputs
    downbeats, time_sig = parse_corrected_downbeats(corrected_downbeats_file)
    onsets = load_onsets(onset_file)

    # Parse snippet timings
    snippet_df = pd.read_csv(snippet_timings_csv)
    snippet_start = snippet_df['snippet_start'].iloc[0]
    snippet_end = snippet_df['snippet_end'].iloc[0]

    # Parse sections
    sections_df = pd.read_csv(songformer_sections_csv)

    # Determine first/last bar in snippet
    first_bar = None
    last_bar = None
    for i, db in enumerate(downbeats):
        if db >= snippet_start and first_bar is None:
            first_bar = i
        if db <= snippet_end:
            last_bar = i

    if first_bar is None or last_bar is None:
        if verbose:
            print("  ! No bars found within snippet bounds")
        return results

    steps_per_bar = time_sig * GRID_SUBDIV_PER_BEAT

    if verbose:
        print(f"\nSection Anchoring:")
        print(f"  Snippet: {snippet_start:.3f}s - {snippet_end:.3f}s")
        print(f"  Bars in snippet: {first_bar} - {last_bar} (global)")
        print(f"  Overlapping sections: {len(sections_df)}")

    # Process each section
    for sec_idx, section in sections_df.iterrows():
        section_num = section['section_num']
        section_start = section['start_absolute_s']
        section_duration = section['duration_s']
        section_end = section_start + section_duration
        section_label = section['label']
        ratio_in_snippet = section['ratio_in_snippet']

        # Calculate ratio_outside_snippet (portion outside / section_duration)
        # Portion outside = section_duration - (portion inside snippet)
        # Portion inside snippet = ratio_in_snippet * (snippet_end - snippet_start)
        portion_inside = ratio_in_snippet * (snippet_end - snippet_start)
        portion_outside = section_duration - portion_inside
        ratio_outside_snippet = portion_outside / section_duration if section_duration > 0 else 0

        if verbose:
            print(f"\n  Section {sec_idx + 1}: {section_label} (section_num={section_num})")
            print(f"    Time: {section_start:.3f}s - {section_end:.3f}s (duration: {section_duration:.3f}s)")
            print(f"    ratio_in_snippet: {ratio_in_snippet:.4f}, ratio_outside: {ratio_outside_snippet:.4f}")

        # Find anchor bar for this section
        anchor_bar = find_anchor_bar(section_start, downbeats, first_bar, last_bar)

        if anchor_bar is None:
            if verbose:
                print(f"    ! No anchor bar found within tolerance {config.ANCHOR_BAR_TOLERANCE}")
            continue

        if verbose:
            anchor_time = downbeats[anchor_bar]
            print(f"    Anchor bar: {anchor_bar} (time: {anchor_time:.3f}s)")

        # Process each pattern length
        for L in pattern_lengths:
            # Calculate anchored phases
            df_onsets = calculate_anchored_phases(
                onsets=onsets,
                downbeats=downbeats,
                anchor_bar=anchor_bar,
                section_end_time=section_end,
                snippet_end_time=snippet_end,
                pattern_len=L,
                steps_per_bar=steps_per_bar
            )

            # Build complete grid
            df_grid = build_complete_grid(
                df_onsets=df_onsets,
                pattern_len=L,
                anchor_bar=anchor_bar,
                downbeats=downbeats,
                section_end_time=section_end,
                snippet_end_time=snippet_end,
                steps_per_bar=steps_per_bar
            )

            if df_grid.empty:
                if verbose:
                    print(f"    L={L}: No complete patterns found")
                continue

            # Calculate complete patterns
            complete_patterns = (df_grid['bar_number'].max() + 1) // L
            bars_used = sorted(df_grid['bar_number_global'].unique())
            bars_used_str = f"{bars_used[0]}-{bars_used[-1]}" if len(bars_used) > 1 else str(bars_used[0])

            # Output filename: SecNo{n}_L{len}_{label}_{ratio}_anchored_onsets.csv
            output_filename = f"SecNo{sec_idx + 1}_L{L}_{section_label}_{ratio_in_snippet:.4f}_anchored_onsets.csv"
            output_file = output_path / output_filename

            # Write CSV with metadata header
            with open(output_file, 'w') as f:
                f.write(f"# section_label={section_label}\n")
                f.write(f"# section_start_absolute={section_start:.6f}\n")
                f.write(f"# section_duration={section_duration:.6f}\n")
                f.write(f"# ratio_in_snippet={ratio_in_snippet:.4f}\n")
                f.write(f"# ratio_outside_snippet={ratio_outside_snippet:.4f}\n")
                f.write(f"# anchor_bar_global={anchor_bar}\n")
                f.write(f"# bars_used_global={bars_used_str}\n")
                f.write(f"# complete_patterns={complete_patterns}\n")
                f.write(f"# pattern_length={L}\n")
                f.write(f"# snippet_start={snippet_start:.6f}\n")
                f.write(f"# snippet_end={snippet_end:.6f}\n")
                df_grid.to_csv(f, index=False)

            results[f"sec{sec_idx + 1}_L{L}"] = str(output_file)

            if verbose:
                print(f"    L={L}: {complete_patterns} patterns, {len(df_grid)} rows -> {output_filename}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python anchoring.py <corrected_downbeats> <onsets_csv> <sf_sections_csv> <sf_timings_csv> <output_dir>")
        print("Example: python anchoring.py downbeats.txt onsets.csv SF_overlapping_sections.csv SF_snippet_timings.csv output/")
        sys.exit(1)

    corrected_downbeats_file = sys.argv[1]
    onset_file = sys.argv[2]
    sf_sections_csv = sys.argv[3]
    sf_timings_csv = sys.argv[4]
    output_dir = sys.argv[5]

    run_anchoring(
        corrected_downbeats_file,
        onset_file,
        sf_sections_csv,
        sf_timings_csv,
        output_dir
    )
