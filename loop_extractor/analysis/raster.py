"""
Simplified raster plot generation and phase calculations.

This module calculates phase deviations of onsets from expected grid positions
using 3 correction methods:
1. Uncorrected (raw downbeat-based grid)
2. Per-snippet correction (finds first onset at 1/16th position)
3. 4-bar loop correction (equidistant grid across 4 bars)

Environment: AEinBOX_13_3 (numpy, pandas)
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
# CONFIGURATION
# ============================================================================

SNIPPET_DURATION_S = config.CORRECT_BARS_SNIPPET_DURATION_S
GRID_SUBDIV_PER_BEAT = 4  # Sixteenth notes
MAX_MATCH_FRAC_BEFORE = 0.49  # Max distance before grid position (prevents overlap)
MAX_MATCH_FRAC_AFTER = 0.51  # Max distance after grid position (prevents overlap)
SEARCH_WINDOW_START_PHASE = 0.5  # Search window before 1/16th for reference onset
SEARCH_WINDOW_END_PHASE = 0.75  # Search window after 1/16th for reference onset


# ============================================================================
# INPUT PARSING
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


def load_snippet_offset(overview_file: str, track_id: str) -> float:
    """
    Load snippet start offset for a track.

    Parameters
    ----------
    overview_file : str
        Path to overview CSV with snippet offsets
    track_id : str
        Track identifier

    Returns
    -------
    float
        Snippet offset in seconds
    """
    # Load CSV with semicolon separator
    df = pd.read_csv(overview_file, sep=';', engine='python')

    # Normalize column names (strip whitespace, lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Find ID column
    id_col = None
    for c in df.columns:
        if c in ('song_id', 'id', 'track_id') or 'id' in c:
            id_col = c
            break

    if id_col is None:
        return 0.0

    # Try to find row by numeric ID first
    track_row = df[df[id_col].astype(str) == str(track_id)]

    # If not found, try extracting numeric ID from filename
    if len(track_row) == 0 and '_' in str(track_id):
        potential_id = str(track_id).split('_')[0]
        if potential_id.isdigit():
            track_row = df[df[id_col].astype(str) == potential_id]

    if len(track_row) == 0:
        return 0.0

    # Find offset column
    def normalize_col_name(s):
        return s.replace('_', ' ').replace('  ', ' ')

    offset_col = None
    for c in df.columns:
        nc = normalize_col_name(c)
        if 'corrected' in nc and 'offset' in nc and 'ms' in nc:
            offset_col = c
            break

    # Fallback: any column with 'offset' and 'ms'
    if offset_col is None:
        for c in df.columns:
            if 'offset' in c and 'ms' in c:
                offset_col = c
                break

    if offset_col is None:
        return 0.0

    offset_ms = track_row.iloc[0][offset_col]
    offset_s = offset_ms / 1000.0
    return offset_s


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_snippet_bars(
    downbeats: List[float],
    snippet_start: float,
    snippet_duration: float = SNIPPET_DURATION_S
) -> Tuple[int, int]:
    """
    Calculate which bars fall within the snippet window.

    Parameters
    ----------
    downbeats : List[float]
        List of downbeat times
    snippet_start : float
        Snippet start time in seconds
    snippet_duration : float
        Snippet duration in seconds

    Returns
    -------
    tuple
        (first_bar_idx, last_bar_idx) - inclusive range
    """
    snippet_end = snippet_start + snippet_duration

    first_bar_idx = 0
    last_bar_idx = len(downbeats) - 2  # -2 because we need pairs

    for i, db in enumerate(downbeats[:-1]):
        if db >= snippet_start:
            first_bar_idx = i
            break

    for i in range(len(downbeats) - 2, -1, -1):
        if downbeats[i] < snippet_end:
            last_bar_idx = i
            break

    return first_bar_idx, last_bar_idx


def find_nearest_onset(target_time: float, onsets: np.ndarray) -> Tuple[float, float]:
    """
    Find onset nearest to target time.

    Parameters
    ----------
    target_time : float
        Target time in seconds
    onsets : np.ndarray
        Array of onset times

    Returns
    -------
    tuple
        (nearest_onset_time, distance)
    """
    if len(onsets) == 0:
        return np.nan, np.inf

    idx = np.argmin(np.abs(onsets - target_time))
    nearest = onsets[idx]
    distance = abs(nearest - target_time)

    return nearest, distance


# ============================================================================
# METHOD 1: UNCORRECTED
# ============================================================================

def calculate_phases_uncorrected(
    onsets: np.ndarray,
    downbeats: List[float],
    first_bar: int,
    last_bar: int,
    steps_per_bar: int,
    max_match_frac: float = None  # Deprecated - uses asymmetric tolerances
) -> pd.DataFrame:
    """
    Calculate uncorrected phases using raw downbeat-based grid.

    The downbeat times define the grid. Between two downbeats, we create
    an equidistant 16th-note grid. Each onset is assigned to the nearest
    grid position.

    Parameters
    ----------
    onsets : np.ndarray
        Onset times in seconds
    downbeats : List[float]
        Downbeat times
    first_bar : int
        First bar index
    last_bar : int
        Last bar index (inclusive)
    steps_per_bar : int
        Number of ticks per bar (e.g., 16 for 16th notes)
    max_match_frac : float
        Maximum matching distance as fraction of step duration

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_number, tick_16th, onset_time, phase_uncorrected
    """
    rows = []

    for bar_idx in range(first_bar, last_bar + 1):
        if bar_idx >= len(downbeats) - 1:
            continue

        bar_start = downbeats[bar_idx]
        bar_end = downbeats[bar_idx + 1]
        bar_duration = bar_end - bar_start

        if bar_duration <= 0:
            continue

        step_duration = bar_duration / steps_per_bar

        # Filter onsets in this bar
        bar_onsets = onsets[(onsets >= bar_start) & (onsets < bar_end)]

        for onset_time in bar_onsets:
            # Calculate phase
            phase = (onset_time - bar_start) / bar_duration

            # Assign to nearest tick
            nearest_tick = int(round(phase * steps_per_bar))
            nearest_tick = max(0, min(steps_per_bar - 1, nearest_tick))

            # Check if within tolerance (asymmetric boundaries)
            grid_time = bar_start + (nearest_tick / steps_per_bar) * bar_duration
            distance = abs(onset_time - grid_time)

            # Use asymmetric tolerance to prevent duplicate assignments at boundaries
            if onset_time < grid_time:
                tolerance = MAX_MATCH_FRAC_BEFORE * step_duration
            else:
                tolerance = MAX_MATCH_FRAC_AFTER * step_duration

            if distance <= tolerance:
                rows.append({
                    'bar_number': bar_idx - first_bar,  # Snippet-relative
                    'tick_16th': nearest_tick,
                    'onset_time': onset_time,
                    'phase_uncorrected': phase
                })

    return pd.DataFrame(rows)


# ============================================================================
# METHOD 2: PER-SNIPPET CORRECTION
# ============================================================================

def find_per_snippet_reference(
    downbeats: List[float],
    onsets: np.ndarray,
    first_bar: int,
    last_bar: int,
    steps_per_bar: int,
    search_window_start_phase: float = SEARCH_WINDOW_START_PHASE,
    search_window_end_phase: float = SEARCH_WINDOW_END_PHASE,
    max_match_frac: float = None  # Deprecated - kept for compatibility
) -> Tuple[float, int]:
    """
    Find reference offset for per-snippet correction.

    Searches for the first onset at the 1/16th position (tick 1).
    If not found in bar 0, checks bar 1, bar 2, etc.

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    onsets : np.ndarray
        Onset times
    first_bar : int
        First bar index to search
    last_bar : int
        Last bar index
    steps_per_bar : int
        Number of ticks per bar
    search_window_start_phase : float
        Search window before 1/16th (relative to step duration)
    search_window_end_phase : float
        Search window after 1/16th (relative to step duration)
    max_match_frac : float
        Maximum matching distance as fraction of step duration

    Returns
    -------
    tuple
        (ref_offset_ms, ref_bar_idx)
    """
    target_tick = 0  # 1/16th position (tick 0 = downbeat = 1/16th)

    for bar_idx in range(first_bar, last_bar + 1):
        if bar_idx >= len(downbeats) - 1:
            continue

        bar_start = downbeats[bar_idx]
        bar_end = downbeats[bar_idx + 1]
        bar_duration = bar_end - bar_start

        if bar_duration <= 0:
            continue

        step_duration = bar_duration / steps_per_bar
        grid_time = bar_start + (target_tick * step_duration)

        # Calculate search window (symmetric around target, using current bar's step duration)
        window_start = grid_time - search_window_start_phase * step_duration
        window_end = grid_time + search_window_end_phase * step_duration

        # Debug logging for first few bars
        if bar_idx <= first_bar + 2:
            print(f"\n[DEBUG] Bar {bar_idx}: bar_start={bar_start:.3f}, grid_time={grid_time:.3f}")
            print(f"  window: [{window_start:.3f}, {window_end:.3f}]")
            print(f"  step_duration={step_duration:.6f}")

        # Find onsets within window
        onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

        if bar_idx <= first_bar + 2:
            print(f"  onsets_in_window: {onsets_in_window[:5] if len(onsets_in_window) > 0 else 'none'}")

        if len(onsets_in_window) == 0:
            # No onset found, try next bar
            continue

        # Find closest onset to grid_time within the search window
        distances = np.abs(onsets_in_window - grid_time)
        min_idx = np.argmin(distances)
        nearest_onset = onsets_in_window[min_idx]
        distance = distances[min_idx]

        if bar_idx <= first_bar + 2:
            print(f"  nearest_onset={nearest_onset:.3f}, distance={distance:.6f}")

        # For reference finding: the search window IS the tolerance criterion
        # Accept the closest onset within the window (no additional max_match_frac check)
        # The onset will still be checked against max_match_frac when assigning to grid
        ref_offset_ms = (nearest_onset - grid_time) * 1000.0
        print(f"\n[DEBUG] Reference found in bar {bar_idx}: offset={ref_offset_ms:.3f}ms\n")
        return ref_offset_ms, bar_idx

    # No reference found
    return 0.0, first_bar


def calculate_phases_per_snippet(
    onsets: np.ndarray,
    downbeats: List[float],
    first_bar: int,
    last_bar: int,
    steps_per_bar: int,
    ref_offset_ms: float,
    max_match_frac: float = None  # Deprecated - uses asymmetric tolerances
) -> pd.DataFrame:
    """
    Calculate per-snippet corrected phases.

    Shifts the entire grid by the reference offset found at 1/16th position.

    Parameters
    ----------
    onsets : np.ndarray
        Onset times
    downbeats : List[float]
        Downbeat times
    first_bar : int
        First bar index
    last_bar : int
        Last bar index
    steps_per_bar : int
        Number of ticks per bar
    ref_offset_ms : float
        Reference offset in milliseconds
    max_match_frac : float
        Maximum matching distance as fraction of step duration

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_number, tick_16th, onset_time, phase_per_snippet
    """
    ref_offset_s = ref_offset_ms / 1000.0
    rows = []

    for bar_idx in range(first_bar, last_bar + 1):
        if bar_idx >= len(downbeats) - 1:
            continue

        bar_start = downbeats[bar_idx]
        bar_end = downbeats[bar_idx + 1]
        bar_duration = bar_end - bar_start

        if bar_duration <= 0:
            continue

        # Apply correction to grid positions (not boundaries)
        corrected_bar_start = bar_start + ref_offset_s
        step_duration = bar_duration / steps_per_bar

        # Filter onsets using UNCORRECTED boundaries (avoid gaps at boundaries)
        bar_onsets = onsets[(onsets >= bar_start) & (onsets < bar_end)]

        # Debug logging for bars 1-2
        if bar_idx - first_bar <= 2:
            print(f"\n[PER-SNIPPET DEBUG] Bar {bar_idx - first_bar}: corrected_bar_start={corrected_bar_start:.6f}")
            print(f"  bar_onsets: {bar_onsets[:5] if len(bar_onsets) > 0 else 'none'}")

        for onset_time in bar_onsets:
            # Calculate corrected phase (relative to corrected grid)
            phase = (onset_time - corrected_bar_start) / bar_duration

            # Assign to nearest tick based on corrected phase
            nearest_tick = int(round(phase * steps_per_bar))
            nearest_tick = max(0, min(steps_per_bar - 1, nearest_tick))

            # Check if within tolerance (asymmetric boundaries)
            grid_time = corrected_bar_start + (nearest_tick / steps_per_bar) * bar_duration
            distance = abs(onset_time - grid_time)

            # Use asymmetric tolerance to prevent duplicate assignments at boundaries
            if onset_time < grid_time:
                tolerance = MAX_MATCH_FRAC_BEFORE * step_duration
            else:
                tolerance = MAX_MATCH_FRAC_AFTER * step_duration

            if distance <= tolerance:
                rows.append({
                    'bar_number': bar_idx - first_bar,
                    'tick_16th': nearest_tick,
                    'onset_time': onset_time,
                    'phase_per_snippet': phase
                })

    return pd.DataFrame(rows)


# ============================================================================
# METHOD 3: 4-BAR LOOP CORRECTION
# ============================================================================

def find_4bar_loop_reference(
    downbeats: List[float],
    onsets: np.ndarray,
    loop_start_bar: int,
    steps_per_bar: int,
    search_window_start_phase: float = SEARCH_WINDOW_START_PHASE,
    search_window_end_phase: float = SEARCH_WINDOW_END_PHASE,
    max_match_frac: float = None  # Deprecated - kept for compatibility
) -> float:
    """
    Find reference offset for a 4-bar loop.

    Uses equidistant grid across 4 bars and searches for onset at 1/16th position
    of the loop start bar.

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    onsets : np.ndarray
        Onset times
    loop_start_bar : int
        Starting bar index of loop
    steps_per_bar : int
        Number of ticks per bar
    search_window_start_phase : float
        Search window before 1/16th
    search_window_end_phase : float
        Search window after 1/16th
    max_match_frac : float
        Maximum matching distance as fraction

    Returns
    -------
    float
        Reference offset in milliseconds
    """
    pattern_len = 4

    if loop_start_bar + pattern_len >= len(downbeats):
        return 0.0

    # Calculate equidistant grid for this 4-bar loop
    loop_start_time = downbeats[loop_start_bar]
    loop_end_time = downbeats[loop_start_bar + pattern_len]
    loop_duration = loop_end_time - loop_start_time

    total_sixteenths = pattern_len * steps_per_bar
    sixteenth_duration = loop_duration / total_sixteenths

    # Only check tick 0 (1/16th = downbeat) in the first bar of the loop
    target_tick = 0
    grid_time = loop_start_time + (target_tick * sixteenth_duration)

    if loop_start_bar == 0:
        # Cannot look at previous bar
        return 0.0

    # Get previous bar info (from actual downbeats, not equidistant)
    prev_bar_start = downbeats[loop_start_bar - 1]
    prev_bar_duration = downbeats[loop_start_bar] - prev_bar_start

    if prev_bar_duration <= 0:
        return 0.0

    # Calculate search window (symmetric around target, using current bar's step duration)
    window_start = grid_time - search_window_start_phase * sixteenth_duration
    window_end = grid_time + search_window_end_phase * sixteenth_duration

    # Find nearest onset within the search window
    onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

    if len(onsets_in_window) == 0:
        return 0.0

    # Find onset closest to grid_time
    distances = np.abs(onsets_in_window - grid_time)
    min_idx = np.argmin(distances)
    nearest_onset = onsets_in_window[min_idx]
    distance = distances[min_idx]

    # For reference finding: the search window IS the tolerance criterion
    # Accept the closest onset within the window
    ref_offset_ms = (nearest_onset - grid_time) * 1000.0
    return ref_offset_ms


def calculate_phases_4bar_loop(
    onsets: np.ndarray,
    downbeats: List[float],
    first_bar: int,
    last_bar: int,
    steps_per_bar: int,
    max_match_frac: float = None,  # Deprecated - uses asymmetric tolerances
    search_window_start_phase: float = SEARCH_WINDOW_START_PHASE,
    search_window_end_phase: float = SEARCH_WINDOW_END_PHASE
) -> pd.DataFrame:
    """
    Calculate 4-bar loop corrected phases using equidistant grid.

    Parameters
    ----------
    onsets : np.ndarray
        Onset times
    downbeats : List[float]
        Downbeat times
    first_bar : int
        First bar index
    last_bar : int
        Last bar index
    steps_per_bar : int
        Number of ticks per bar
    max_match_frac : float
        Maximum matching distance as fraction
    search_window_start_phase : float
        Search window before 1/16th
    search_window_end_phase : float
        Search window after 1/16th

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_number, tick_16th, onset_time, phase_4bar_loop
    """
    rows = []
    pattern_len = 4
    total_sixteenths = pattern_len * steps_per_bar

    # Process each 4-bar loop
    for loop_start_bar in range(first_bar, last_bar + 1, pattern_len):
        if loop_start_bar + pattern_len > len(downbeats) - 1:
            break

        # Calculate equidistant grid for this loop
        loop_start_time = downbeats[loop_start_bar]
        loop_end_time = downbeats[loop_start_bar + pattern_len]
        loop_duration = loop_end_time - loop_start_time
        sixteenth_duration = loop_duration / total_sixteenths

        # Find reference offset for this loop
        ref_offset_ms = find_4bar_loop_reference(
            downbeats, onsets, loop_start_bar, steps_per_bar,
            search_window_start_phase, search_window_end_phase, max_match_frac
        )
        ref_offset_s = ref_offset_ms / 1000.0

        # Process each bar in loop
        for bar_offset in range(pattern_len):
            bar_idx = loop_start_bar + bar_offset

            if bar_idx > last_bar or bar_idx >= len(downbeats) - 1:
                break

            # Calculate equidistant bar boundaries
            bar_sixteenth_pos = bar_offset * steps_per_bar
            equi_bar_start = loop_start_time + (bar_sixteenth_pos * sixteenth_duration)
            equi_bar_duration = steps_per_bar * sixteenth_duration
            equi_bar_end = equi_bar_start + equi_bar_duration

            # Apply correction to grid positions (not boundaries)
            corrected_equi_bar_start = equi_bar_start + ref_offset_s

            # Filter onsets using UNCORRECTED equidistant boundaries (avoid gaps)
            bar_onsets = onsets[(onsets >= equi_bar_start) & (onsets < equi_bar_end)]

            for onset_time in bar_onsets:
                # Calculate corrected phase (relative to corrected grid)
                phase = (onset_time - corrected_equi_bar_start) / equi_bar_duration

                # Assign to nearest tick
                nearest_tick = int(round(phase * steps_per_bar))
                nearest_tick = max(0, min(steps_per_bar - 1, nearest_tick))

                # Check if within tolerance (asymmetric boundaries)
                grid_time = corrected_equi_bar_start + (nearest_tick / steps_per_bar) * equi_bar_duration
                distance = abs(onset_time - grid_time)

                # Use asymmetric tolerance to prevent duplicate assignments at boundaries
                if onset_time < grid_time:
                    tolerance = MAX_MATCH_FRAC_BEFORE * sixteenth_duration
                else:
                    tolerance = MAX_MATCH_FRAC_AFTER * sixteenth_duration

                if distance <= tolerance:
                    rows.append({
                        'bar_number': bar_idx - first_bar,
                        'tick_16th': nearest_tick,
                        'onset_time': onset_time,
                        'phase_4bar_loop': phase
                    })

    return pd.DataFrame(rows)


# ============================================================================
# MAIN CSV EXPORT
# ============================================================================

def create_raster_csv(
    corrected_downbeats_file: str,
    onset_file: str,
    snippet_offset: float,
    output_file: str,
    max_match_frac: float = None,  # Deprecated - uses asymmetric tolerances
    search_window_start_phase: float = SEARCH_WINDOW_START_PHASE,
    search_window_end_phase: float = SEARCH_WINDOW_END_PHASE
) -> pd.DataFrame:
    """
    Create raster CSV with 3 correction methods.

    Parameters
    ----------
    corrected_downbeats_file : str
        Path to corrected downbeats file
    onset_file : str
        Path to onsets CSV file
    snippet_offset : float
        Snippet start offset in seconds
    output_file : str
        Output CSV path
    max_match_frac : float
        Maximum matching distance as fraction of step duration
    search_window_start_phase : float
        Search window before 1/16th for reference onset
    search_window_end_phase : float
        Search window after 1/16th for reference onset

    Returns
    -------
    pd.DataFrame
        Comprehensive DataFrame with all phases
    """
    print(f"\nCreating raster CSV...")

    # Load inputs
    print("  Loading downbeats...")
    downbeats, tsig = parse_corrected_downbeats(corrected_downbeats_file)
    steps_per_bar = tsig * GRID_SUBDIV_PER_BEAT

    print("  Loading onsets...")
    onsets = load_onsets(onset_file)

    # Calculate snippet bars
    first_bar, last_bar = calculate_snippet_bars(downbeats, snippet_offset, SNIPPET_DURATION_S)
    print(f"  Snippet covers bars {first_bar} to {last_bar}")

    # Method 1: Uncorrected
    print("  Calculating uncorrected phases...")
    df_uncorrected = calculate_phases_uncorrected(
        onsets, downbeats, first_bar, last_bar, steps_per_bar, max_match_frac
    )

    # Method 2: Per-snippet correction
    print("  Finding per-snippet reference offset...")
    ref_offset_ms, ref_bar = find_per_snippet_reference(
        downbeats, onsets, first_bar, last_bar, steps_per_bar,
        search_window_start_phase, search_window_end_phase, max_match_frac
    )
    print(f"    Reference: {ref_offset_ms:.2f}ms at bar {ref_bar}")

    print("  Calculating per-snippet phases...")
    df_per_snippet = calculate_phases_per_snippet(
        onsets, downbeats, first_bar, last_bar, steps_per_bar, ref_offset_ms, max_match_frac
    )

    # Method 3: 4-bar loop correction
    print("  Calculating 4-bar loop phases...")
    df_4bar_loop = calculate_phases_4bar_loop(
        onsets, downbeats, first_bar, last_bar, steps_per_bar,
        max_match_frac, search_window_start_phase, search_window_end_phase
    )

    # Create comprehensive grid (all bar/tick combinations)
    print("  Building comprehensive grid...")
    grid_rows = []
    for bar_idx in range(first_bar, last_bar + 1):
        for tick in range(steps_per_bar):
            grid_rows.append({
                'bar_number': bar_idx - first_bar,  # Snippet-relative
                'tick_16th': tick
            })

    df_comprehensive = pd.DataFrame(grid_rows)

    # Merge all phase data
    print("  Merging phase data...")

    # Merge uncorrected (with its onset_time)
    df_comprehensive = df_comprehensive.merge(
        df_uncorrected[['bar_number', 'tick_16th', 'onset_time', 'phase_uncorrected']].rename(
            columns={'onset_time': 'onset_time_uncorrected'}
        ),
        on=['bar_number', 'tick_16th'],
        how='left'
    )

    # Merge per-snippet (with its own onset_time - may be different!)
    df_comprehensive = df_comprehensive.merge(
        df_per_snippet[['bar_number', 'tick_16th', 'onset_time', 'phase_per_snippet']].rename(
            columns={'onset_time': 'onset_time_per_snippet'}
        ),
        on=['bar_number', 'tick_16th'],
        how='left'
    )

    # Merge 4-bar loop (with its own onset_time - may be different!)
    df_comprehensive = df_comprehensive.merge(
        df_4bar_loop[['bar_number', 'tick_16th', 'onset_time', 'phase_4bar_loop']].rename(
            columns={'onset_time': 'onset_time_4bar_loop'}
        ),
        on=['bar_number', 'tick_16th'],
        how='left'
    )

    # Add grid time and grid phase columns
    print("  Adding grid time and phase columns...")

    grid_times_uncorrected = []
    grid_times_per_snippet = []
    grid_times_4bar_loop = []
    grid_phases = []

    ref_offset_s = ref_offset_ms / 1000.0

    for _, row in df_comprehensive.iterrows():
        bar_num_abs = int(row['bar_number']) + first_bar  # Convert to absolute bar index
        tick = int(row['tick_16th'])

        # Grid phase (same for all methods)
        grid_phase = tick / steps_per_bar
        grid_phases.append(grid_phase)

        # Uncorrected grid time
        if bar_num_abs < len(downbeats) - 1:
            bar_start = downbeats[bar_num_abs]
            bar_end = downbeats[bar_num_abs + 1]
            bar_duration = bar_end - bar_start
            grid_time_uncorrected = bar_start + (tick / steps_per_bar) * bar_duration
            grid_times_uncorrected.append(grid_time_uncorrected)

            # Per-snippet corrected grid time
            grid_time_per_snippet = (bar_start + ref_offset_s) + (tick / steps_per_bar) * bar_duration
            grid_times_per_snippet.append(grid_time_per_snippet)
        else:
            grid_times_uncorrected.append(None)
            grid_times_per_snippet.append(None)

        # 4-bar loop grid time (equidistant)
        # Find which 4-bar loop this bar belongs to
        pattern_len = 4
        loop_start_bar = (bar_num_abs // pattern_len) * pattern_len

        if loop_start_bar + pattern_len <= len(downbeats) - 1:
            loop_start_time = downbeats[loop_start_bar]
            loop_end_time = downbeats[loop_start_bar + pattern_len]
            loop_duration = loop_end_time - loop_start_time
            total_sixteenths = pattern_len * steps_per_bar
            sixteenth_duration = loop_duration / total_sixteenths

            # Find loop reference offset
            loop_ref_offset_ms = find_4bar_loop_reference(
                downbeats, onsets, loop_start_bar, steps_per_bar,
                search_window_start_phase, search_window_end_phase, None
            )
            loop_ref_offset_s = loop_ref_offset_ms / 1000.0

            # Calculate equidistant grid time for this bar
            bar_offset_in_loop = bar_num_abs - loop_start_bar
            bar_sixteenth_pos = bar_offset_in_loop * steps_per_bar
            equi_bar_start = loop_start_time + (bar_sixteenth_pos * sixteenth_duration)

            grid_time_4bar = (equi_bar_start + loop_ref_offset_s) + (tick / steps_per_bar) * (steps_per_bar * sixteenth_duration)
            grid_times_4bar_loop.append(grid_time_4bar)
        else:
            grid_times_4bar_loop.append(None)

    df_comprehensive['grid_time_uncorrected'] = grid_times_uncorrected
    df_comprehensive['grid_time_per_snippet'] = grid_times_per_snippet
    df_comprehensive['grid_time_4bar_loop'] = grid_times_4bar_loop
    df_comprehensive['grid_phase'] = grid_phases

    # Prepare output path
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save reference onsets to separate CSV
    print("  Saving reference onsets...")
    ref_rows = []

    # Per-snippet reference
    if ref_bar is not None and ref_bar >= first_bar:
        # Calculate phases for reference
        if ref_bar < len(downbeats) - 1:
            bar_start = downbeats[ref_bar]
            bar_end = downbeats[ref_bar + 1]
            bar_duration = bar_end - bar_start

            # Find the actual onset at tick 0 (1/16th position)
            ref_offset_s = ref_offset_ms / 1000.0
            grid_time = bar_start  # Uncorrected 1/16th position (tick 0 = downbeat)

            # Reference onset is offset from grid
            ref_onset_time = grid_time + ref_offset_s

            # Calculate phases
            ref_phase = (ref_onset_time - bar_start) / bar_duration if bar_duration > 0 else 0.0
            grid_phase = (grid_time - bar_start) / bar_duration if bar_duration > 0 else 0.0

            ref_rows.append({
                'method': 'per_snippet',
                'bar_number': ref_bar - first_bar,
                'bar_number_global': ref_bar,
                'ref_ms': ref_offset_ms,
                'ref_phase': ref_phase,
                'grid_phase': grid_phase,
                'bar_duration': bar_duration
            })

    # 4-bar loop references (one for each 4-bar loop - each loop has its own offset)
    pattern_len = 4
    for loop_start_bar in range(first_bar, last_bar + 1, pattern_len):
        if loop_start_bar + pattern_len > len(downbeats) - 1:
            break

        # Find reference offset for THIS specific loop
        loop_ref_offset_ms = find_4bar_loop_reference(
            downbeats, onsets, loop_start_bar, steps_per_bar,
            search_window_start_phase, search_window_end_phase, None
        )

        # Calculate phases for 4-bar loop reference
        loop_start_time = downbeats[loop_start_bar]
        loop_end_time = downbeats[loop_start_bar + pattern_len]
        loop_duration = loop_end_time - loop_start_time
        total_sixteenths = pattern_len * steps_per_bar
        sixteenth_duration = loop_duration / total_sixteenths

        # Grid time at tick 0 (1/16th position) of first bar in loop
        grid_time = loop_start_time

        # Reference onset is offset from grid
        loop_ref_offset_s = loop_ref_offset_ms / 1000.0
        ref_onset_time = grid_time + loop_ref_offset_s

        # Calculate phases relative to equidistant bar
        equi_bar_start = loop_start_time
        equi_bar_duration = steps_per_bar * sixteenth_duration

        ref_phase = (ref_onset_time - equi_bar_start) / equi_bar_duration if equi_bar_duration > 0 else 0.0
        grid_phase = (grid_time - equi_bar_start) / equi_bar_duration if equi_bar_duration > 0 else 0.0

        ref_rows.append({
            'method': '4bar_loop',
            'bar_number': loop_start_bar - first_bar,
            'bar_number_global': loop_start_bar,
            'ref_ms': loop_ref_offset_ms,
            'ref_phase': ref_phase,
            'grid_phase': grid_phase,
            'bar_duration': equi_bar_duration
        })

    if ref_rows:
        df_refs = pd.DataFrame(ref_rows)
        ref_file = output_path.parent / f"{output_path.stem}_reference_onsets.csv"
        df_refs.to_csv(ref_file, index=False)
        print(f"  ✓ Reference onsets saved: {len(df_refs)} references to {ref_file.name}")

    # Save comprehensive CSV
    print(f"  Saving to {output_file}...")
    df_comprehensive.to_csv(output_file, index=False)

    print(f"  ✓ Raster CSV created: {len(df_comprehensive)} rows")

    return df_comprehensive


# Backward compatibility alias
def create_comprehensive_csv(
    corrected_downbeats_file: str,
    onset_file: str,
    pattern_lengths: Dict[str, int],  # Ignored in new version
    snippet_offset: float,
    output_file: str
) -> pd.DataFrame:
    """
    Backward compatibility wrapper for create_raster_csv.

    The pattern_lengths parameter is ignored as we now use fixed 4-bar loops.
    """
    return create_raster_csv(
        corrected_downbeats_file,
        onset_file,
        snippet_offset,
        output_file
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python raster.py <corrected_downbeats_file> <onset_file> <snippet_offset> <output_file>")
        sys.exit(1)

    corrected_downbeats_file = sys.argv[1]
    onset_file = sys.argv[2]
    snippet_offset = float(sys.argv[3])
    output_file = sys.argv[4]

    create_raster_csv(corrected_downbeats_file, onset_file, snippet_offset, output_file)
