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
IQR_MULTIPLIER_TUKEY = 1.5  # IQR multiplier for Tukey outlier detection (1.5=standard, 3.0=extreme)
RUNNING_MEAN_THRESHOLD = 0.5  # Threshold for running mean filtering (0.5 = 50%)
NO_OF_REPETITIONS_TH = 2  # Threshold for choosing filtering method (≤2: running mean, >2: Tukey)

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


def find_4bar_pattern_flexStart_reference(
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
    Find first feasible reference onset in snippet, return offset and bar.

    This is used to find the starting point for 4-bar pattern with flexible start.
    Unlike the per-snippet method, this will be used as the anchor for subsequent
    4-bar steps.

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    onsets : np.ndarray
        Onset times
    first_bar : int
        First bar in snippet
    last_bar : int
        Last bar in snippet
    steps_per_bar : int
        Number of ticks per bar
    search_window_start_phase : float
        Search window before 1/16th
    search_window_end_phase : float
        Search window after 1/16th
    max_match_frac : float
        Deprecated parameter

    Returns
    -------
    tuple
        (ref_offset_ms: float, ref_bar: int) - offset in ms and bar index
    """
    target_tick = 0  # 1/16th position (downbeat)

    # Search through snippet to find first feasible reference
    for bar_idx in range(first_bar, last_bar + 1):
        if bar_idx + 1 >= len(downbeats):
            break

        bar_start = downbeats[bar_idx]
        bar_end = downbeats[bar_idx + 1]
        bar_duration = bar_end - bar_start

        if bar_duration <= 0:
            continue

        step_duration = bar_duration / steps_per_bar
        grid_time = bar_start + (target_tick * step_duration)

        # Calculate search window (symmetric around target)
        window_start = grid_time - search_window_start_phase * step_duration
        window_end = grid_time + search_window_end_phase * step_duration

        # Find onsets within window
        onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

        if len(onsets_in_window) == 0:
            continue

        # Find closest onset to grid_time within the search window
        distances = np.abs(onsets_in_window - grid_time)
        min_idx = np.argmin(distances)
        nearest_onset = onsets_in_window[min_idx]

        # Found first reference!
        ref_offset_ms = (nearest_onset - grid_time) * 1000.0
        return ref_offset_ms, bar_idx

    # No reference found
    return 0.0, None


def calculate_phases_4bar_pattern_flexStart(
    onsets: np.ndarray,
    downbeats: List[float],
    first_bar: int,
    last_bar: int,
    steps_per_bar: int,
    ref_offset_ms: float,
    ref_bar: int,
    max_match_frac: float = None,  # Deprecated - uses asymmetric tolerances
    search_window_start_phase: float = SEARCH_WINDOW_START_PHASE,
    search_window_end_phase: float = SEARCH_WINDOW_END_PHASE
) -> pd.DataFrame:
    """
    Calculate phases with 4-bar pattern flexible start correction.

    Creates 64 equidistant points (16×4) between downbeat of bar 0 and
    downbeat of bar 4. Finds INDEPENDENT reference for each 4-bar segment
    starting from ref_bar.

    Parameters
    ----------
    onsets : np.ndarray
        Onset times
    downbeats : List[float]
        Downbeat times
    first_bar : int
        First bar in snippet
    last_bar : int
        Last bar in snippet
    steps_per_bar : int
        Number of ticks per bar
    ref_offset_ms : float
        Reference offset in milliseconds (not used - kept for compatibility)
    ref_bar : int
        Bar index where first reference was found
    max_match_frac : float
        Deprecated parameter
    search_window_start_phase : float
        Search window before 1/16th
    search_window_end_phase : float
        Search window after 1/16th

    Returns
    -------
    pd.DataFrame
        Phase data with columns: bar_number, tick_16th, onset_time, phase_4bar_pattern_flexStart
    """
    if ref_bar is None:
        return pd.DataFrame()

    rows = []
    pattern_len = 4

    # Process every 4-bar segment starting from ref_bar
    for segment_start in range(ref_bar, last_bar + 1, pattern_len):
        if segment_start + pattern_len > len(downbeats) - 1:
            break

        # Calculate equidistant grid for THIS segment
        # Grid: 64 equidistant points between bar 0 and bar 4
        segment_start_time = downbeats[segment_start]
        segment_end_time = downbeats[segment_start + pattern_len]
        segment_duration = segment_end_time - segment_start_time

        total_sixteenths = pattern_len * steps_per_bar
        sixteenth_duration = segment_duration / total_sixteenths

        # Find reference offset for THIS segment at tick 0 of segment start
        segment_grid_time = segment_start_time  # Grid time at tick 0

        # Find reference onset for THIS segment
        window_start = segment_grid_time - search_window_start_phase * sixteenth_duration
        window_end = segment_grid_time + search_window_end_phase * sixteenth_duration
        onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

        if len(onsets_in_window) > 0:
            # Find closest onset to grid_time
            distances = np.abs(onsets_in_window - segment_grid_time)
            min_idx = np.argmin(distances)
            nearest_onset = onsets_in_window[min_idx]
            segment_ref_offset_s = (nearest_onset - segment_grid_time)
        else:
            # No reference found for this segment, use 0
            segment_ref_offset_s = 0.0

        # Process all bars in this 4-bar segment
        segment_end = min(segment_start + pattern_len, last_bar + 1)

        for bar_idx in range(segment_start, segment_end):
            if bar_idx >= len(downbeats) - 1:
                break

            # Calculate equidistant bar boundaries within the segment
            bar_offset_in_segment = bar_idx - segment_start
            bar_sixteenth_pos = bar_offset_in_segment * steps_per_bar

            equi_bar_start = segment_start_time + (bar_sixteenth_pos * sixteenth_duration)
            equi_bar_duration = steps_per_bar * sixteenth_duration
            equi_bar_end = equi_bar_start + equi_bar_duration

            # Apply segment-specific correction to grid positions (not boundaries)
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
                    # This represents the fractional position within that specific tick
                    tick_phase = (phase - grid_phase) * steps_per_bar

                    rows.append({
                        'bar_number': bar_idx - first_bar,
                        'tick_16th': nearest_tick,
                        'onset_time': onset_time,
                        'phase_4bar_pattern_flexStart': phase,
                        'tick_phase_4bar_pattern_flexStart': tick_phase
                    })

    return pd.DataFrame(rows)


# ============================================================================
# GENERALIZED PATTERN FLEXSTART METHODS
# ============================================================================

def find_pattern_flexStart_reference(
    downbeats: List[float],
    onsets: np.ndarray,
    first_bar: int,
    last_bar: int,
    steps_per_bar: int,
    pattern_len: int,
    search_window_start_phase: float = SEARCH_WINDOW_START_PHASE,
    search_window_end_phase: float = SEARCH_WINDOW_END_PHASE,
    max_match_frac: float = None  # Deprecated - kept for compatibility
) -> Tuple[float, int]:
    """
    Find first feasible reference onset in snippet for any pattern length.

    This is a generalized version that works for pattern_len=1, 2, 4, etc.

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    onsets : np.ndarray
        Onset times
    first_bar : int
        First bar in snippet
    last_bar : int
        Last bar in snippet
    steps_per_bar : int
        Number of ticks per bar
    pattern_len : int
        Pattern length in bars (1, 2, 4, etc.)
    search_window_start_phase : float
        Search window before 1/16th
    search_window_end_phase : float
        Search window after 1/16th
    max_match_frac : float
        Deprecated parameter

    Returns
    -------
    tuple
        (ref_offset_ms: float, ref_bar: int) - offset in ms and bar index
    """
    target_tick = 0  # 1/16th position (downbeat)

    # Search through snippet to find first feasible reference
    for bar_idx in range(first_bar, last_bar + 1):
        if bar_idx + 1 >= len(downbeats):
            break

        bar_start = downbeats[bar_idx]
        bar_end = downbeats[bar_idx + 1]
        bar_duration = bar_end - bar_start

        if bar_duration <= 0:
            continue

        step_duration = bar_duration / steps_per_bar
        grid_time = bar_start + (target_tick * step_duration)

        # Calculate search window (symmetric around target)
        window_start = grid_time - search_window_start_phase * step_duration
        window_end = grid_time + search_window_end_phase * step_duration

        # Find onsets within window
        onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

        if len(onsets_in_window) == 0:
            continue

        # Find closest onset to grid_time within the search window
        distances = np.abs(onsets_in_window - grid_time)
        min_idx = np.argmin(distances)
        nearest_onset = onsets_in_window[min_idx]

        # Found first reference!
        ref_offset_ms = (nearest_onset - grid_time) * 1000.0
        return ref_offset_ms, bar_idx

    # No reference found
    return 0.0, None


def calculate_phases_pattern_flexStart(
    onsets: np.ndarray,
    downbeats: List[float],
    first_bar: int,
    last_bar: int,
    steps_per_bar: int,
    pattern_len: int,
    ref_offset_ms: float,
    ref_bar: int,
    phase_column_name: str,
    max_match_frac: float = None,  # Deprecated - uses asymmetric tolerances
    search_window_start_phase: float = SEARCH_WINDOW_START_PHASE,
    search_window_end_phase: float = SEARCH_WINDOW_END_PHASE
) -> pd.DataFrame:
    """
    Calculate phases with N-bar pattern flexible start correction (generalized).

    Creates 16×PatternLength equidistant points between downbeat of bar 0 and
    downbeat of bar L (where L = pattern_len). Finds INDEPENDENT reference
    for each N-bar segment starting from ref_bar.

    Parameters
    ----------
    onsets : np.ndarray
        Onset times
    downbeats : List[float]
        Downbeat times
    first_bar : int
        First bar in snippet
    last_bar : int
        Last bar in snippet
    steps_per_bar : int
        Number of ticks per bar
    pattern_len : int
        Pattern length in bars (1, 2, 4, etc.)
    ref_offset_ms : float
        Reference offset in milliseconds (not used - kept for compatibility)
    ref_bar : int
        Bar index where first reference was found
    phase_column_name : str
        Name of the phase column in output dataframe
    max_match_frac : float
        Deprecated parameter
    search_window_start_phase : float
        Search window before 1/16th
    search_window_end_phase : float
        Search window after 1/16th

    Returns
    -------
    pd.DataFrame
        Phase data with columns: bar_number, tick_16th, onset_time, phase_<method>
    """
    if ref_bar is None:
        return pd.DataFrame()

    rows = []

    # Process every N-bar segment starting from ref_bar
    for segment_start in range(ref_bar, last_bar + 1, pattern_len):
        if segment_start + pattern_len > len(downbeats) - 1:
            break

        # Calculate equidistant grid for THIS segment
        # Grid: 16×pattern_len equidistant points between bar 0 and bar L
        segment_start_time = downbeats[segment_start]
        segment_end_time = downbeats[segment_start + pattern_len]
        segment_duration = segment_end_time - segment_start_time

        total_sixteenths = pattern_len * steps_per_bar
        sixteenth_duration = segment_duration / total_sixteenths

        # Find reference offset for THIS segment at tick 0 of segment start
        segment_grid_time = segment_start_time  # Grid time at tick 0

        # Find reference onset for THIS segment
        window_start = segment_grid_time - search_window_start_phase * sixteenth_duration
        window_end = segment_grid_time + search_window_end_phase * sixteenth_duration
        onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

        if len(onsets_in_window) > 0:
            # Find closest onset to grid_time
            distances = np.abs(onsets_in_window - segment_grid_time)
            min_idx = np.argmin(distances)
            nearest_onset = onsets_in_window[min_idx]
            segment_ref_offset_s = (nearest_onset - segment_grid_time)
        else:
            # No reference found for this segment, use 0
            segment_ref_offset_s = 0.0

        # Process all bars in this N-bar segment
        segment_end = min(segment_start + pattern_len, last_bar + 1)

        for bar_idx in range(segment_start, segment_end):
            if bar_idx >= len(downbeats) - 1:
                break

            # Calculate equidistant bar boundaries within the segment
            bar_offset_in_segment = bar_idx - segment_start
            bar_sixteenth_pos = bar_offset_in_segment * steps_per_bar

            equi_bar_start = segment_start_time + (bar_sixteenth_pos * sixteenth_duration)
            equi_bar_duration = steps_per_bar * sixteenth_duration
            equi_bar_end = equi_bar_start + equi_bar_duration

            # Apply segment-specific correction to grid positions (not boundaries)
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
                    # This represents the fractional position within that specific tick
                    tick_phase = (phase - grid_phase) * steps_per_bar

                    # Generate tick_phase column name from phase column name
                    tick_phase_column_name = phase_column_name.replace('phase_', 'tick_phase_')

                    rows.append({
                        'bar_number': bar_idx - first_bar,
                        'tick_16th': nearest_tick,
                        'onset_time': onset_time,
                        phase_column_name: phase,
                        tick_phase_column_name: tick_phase
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
    search_window_end_phase: float = SEARCH_WINDOW_END_PHASE,
    snippet_duration: float = SNIPPET_DURATION_S
) -> pd.DataFrame:
    """
    Create raster CSV with 6 correction methods.

    Methods:
    1. Uncorrected: Raw downbeat grid (baseline)
    2. Per-snippet: Single global offset for entire snippet
    3. 4-bar Loop: Equidistant grid with offset per loop
    4. 4-bar Pattern FlexStart: Flexible start, independent reference every 4 bars
    5. 2-bar Pattern FlexStart: Flexible start, independent reference every 2 bars
    6. 1-bar Pattern FlexStart: Flexible start, independent reference every bar

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
    first_bar, last_bar = calculate_snippet_bars(downbeats, snippet_offset, snippet_duration)
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

    # Method 4: 4-bar pattern flexStart correction
    print("  Finding 4-bar pattern flexStart reference...")
    flexStart_ref_offset_ms, flexStart_ref_bar = find_4bar_pattern_flexStart_reference(
        downbeats, onsets, first_bar, last_bar, steps_per_bar,
        search_window_start_phase, search_window_end_phase, max_match_frac
    )
    print(f"    Reference: {flexStart_ref_offset_ms:.2f}ms at bar {flexStart_ref_bar}")

    print("  Calculating 4-bar pattern flexStart phases...")
    df_4bar_pattern_flexStart = calculate_phases_4bar_pattern_flexStart(
        onsets, downbeats, first_bar, last_bar, steps_per_bar,
        flexStart_ref_offset_ms, flexStart_ref_bar, max_match_frac,
        search_window_start_phase, search_window_end_phase
    )

    # Method 5: 2-bar pattern flexStart correction
    print("  Finding 2-bar pattern flexStart reference...")
    flexStart_2bar_ref_offset_ms, flexStart_2bar_ref_bar = find_pattern_flexStart_reference(
        downbeats, onsets, first_bar, last_bar, steps_per_bar, 2,
        search_window_start_phase, search_window_end_phase, max_match_frac
    )
    print(f"    Reference: {flexStart_2bar_ref_offset_ms:.2f}ms at bar {flexStart_2bar_ref_bar}")

    print("  Calculating 2-bar pattern flexStart phases...")
    df_2bar_pattern_flexStart = calculate_phases_pattern_flexStart(
        onsets, downbeats, first_bar, last_bar, steps_per_bar, 2,
        flexStart_2bar_ref_offset_ms, flexStart_2bar_ref_bar,
        'phase_2bar_pattern_flexStart', max_match_frac,
        search_window_start_phase, search_window_end_phase
    )

    # Method 6: 1-bar pattern flexStart correction
    print("  Finding 1-bar pattern flexStart reference...")
    flexStart_1bar_ref_offset_ms, flexStart_1bar_ref_bar = find_pattern_flexStart_reference(
        downbeats, onsets, first_bar, last_bar, steps_per_bar, 1,
        search_window_start_phase, search_window_end_phase, max_match_frac
    )
    print(f"    Reference: {flexStart_1bar_ref_offset_ms:.2f}ms at bar {flexStart_1bar_ref_bar}")

    print("  Calculating 1-bar pattern flexStart phases...")
    df_1bar_pattern_flexStart = calculate_phases_pattern_flexStart(
        onsets, downbeats, first_bar, last_bar, steps_per_bar, 1,
        flexStart_1bar_ref_offset_ms, flexStart_1bar_ref_bar,
        'phase_1bar_pattern_flexStart', max_match_frac,
        search_window_start_phase, search_window_end_phase
    )

    # Create comprehensive grid (all bar/tick combinations)
    print("  Building comprehensive grid...")
    grid_rows = []
    for bar_idx in range(first_bar, last_bar + 1):
        for tick in range(steps_per_bar):
            grid_rows.append({
                'bar_number': bar_idx - first_bar,  # Snippet-relative
                'bar_number_global': bar_idx,  # Global bar number for cross-step matching
                'tick_16th': tick
            })

    df_comprehensive = pd.DataFrame(grid_rows)

    # Deduplicate method dataframes to prevent cartesian product in merge
    # When multiple onsets are assigned to the same (bar_number, tick_16th),
    # keep only the one with the smallest absolute phase (closest to grid position)
    print("  Deduplicating method dataframes...")

    def deduplicate_by_closest_to_grid(df, phase_col):
        """
        Keep only the onset closest to grid for each (bar_number, tick_16th).

        When multiple onsets fall within tolerance of the same tick (which can
        happen with slower tempos where step duration is large), we keep the
        onset that is closest to the grid position (smallest absolute phase).
        """
        if df.empty:
            return df

        # Calculate absolute phase (distance from grid position = 0)
        df = df.copy()
        df['abs_phase'] = df[phase_col].abs()

        # Sort by absolute phase and keep only the first (closest) for each (bar, tick)
        df = df.sort_values('abs_phase').groupby(['bar_number', 'tick_16th'], as_index=False).first()

        # Remove temporary column
        df = df.drop(columns=['abs_phase'])

        return df

    df_uncorrected = deduplicate_by_closest_to_grid(df_uncorrected, 'phase_uncorrected')
    df_per_snippet = deduplicate_by_closest_to_grid(df_per_snippet, 'phase_per_snippet')
    df_4bar_loop = deduplicate_by_closest_to_grid(df_4bar_loop, 'phase_4bar_loop')
    df_4bar_pattern_flexStart = deduplicate_by_closest_to_grid(df_4bar_pattern_flexStart, 'phase_4bar_pattern_flexStart')
    df_2bar_pattern_flexStart = deduplicate_by_closest_to_grid(df_2bar_pattern_flexStart, 'phase_2bar_pattern_flexStart')
    df_1bar_pattern_flexStart = deduplicate_by_closest_to_grid(df_1bar_pattern_flexStart, 'phase_1bar_pattern_flexStart')

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

    # Merge 4-bar pattern flexStart (with its own onset_time - may be different!)
    columns_to_merge = ['bar_number', 'tick_16th', 'onset_time', 'phase_4bar_pattern_flexStart']
    if 'tick_phase_4bar_pattern_flexStart' in df_4bar_pattern_flexStart.columns:
        columns_to_merge.append('tick_phase_4bar_pattern_flexStart')
    df_comprehensive = df_comprehensive.merge(
        df_4bar_pattern_flexStart[columns_to_merge].rename(
            columns={'onset_time': 'onset_time_4bar_pattern_flexStart'}
        ),
        on=['bar_number', 'tick_16th'],
        how='left'
    )

    # Merge 2-bar pattern flexStart (with its own onset_time - may be different!)
    columns_to_merge = ['bar_number', 'tick_16th', 'onset_time', 'phase_2bar_pattern_flexStart']
    if 'tick_phase_2bar_pattern_flexStart' in df_2bar_pattern_flexStart.columns:
        columns_to_merge.append('tick_phase_2bar_pattern_flexStart')
    df_comprehensive = df_comprehensive.merge(
        df_2bar_pattern_flexStart[columns_to_merge].rename(
            columns={'onset_time': 'onset_time_2bar_pattern_flexStart'}
        ),
        on=['bar_number', 'tick_16th'],
        how='left'
    )

    # Merge 1-bar pattern flexStart (with its own onset_time - may be different!)
    columns_to_merge = ['bar_number', 'tick_16th', 'onset_time', 'phase_1bar_pattern_flexStart']
    if 'tick_phase_1bar_pattern_flexStart' in df_1bar_pattern_flexStart.columns:
        columns_to_merge.append('tick_phase_1bar_pattern_flexStart')
    df_comprehensive = df_comprehensive.merge(
        df_1bar_pattern_flexStart[columns_to_merge].rename(
            columns={'onset_time': 'onset_time_1bar_pattern_flexStart'}
        ),
        on=['bar_number', 'tick_16th'],
        how='left'
    )

    # Add grid time and grid phase columns
    print("  Adding grid time and phase columns...")

    grid_times_uncorrected = []
    grid_times_per_snippet = []
    grid_times_4bar_loop = []
    grid_times_4bar_pattern_flexStart = []
    grid_times_2bar_pattern_flexStart = []
    grid_times_1bar_pattern_flexStart = []
    grid_phases = []

    ref_offset_s = ref_offset_ms / 1000.0
    flexStart_ref_offset_s = flexStart_ref_offset_ms / 1000.0
    flexStart_2bar_ref_offset_s = flexStart_2bar_ref_offset_ms / 1000.0
    flexStart_1bar_ref_offset_s = flexStart_1bar_ref_offset_ms / 1000.0

    # Pre-calculate all reference offsets for each method to avoid recalculating in the loop
    # This ensures consistency and performance

    # 4bar_loop: one offset per loop (every 4 bars from first_bar)
    loop_ref_offsets = {}
    pattern_len = 4
    for loop_start_bar in range(first_bar, last_bar + 1, pattern_len):
        if loop_start_bar + pattern_len <= len(downbeats) - 1:
            loop_ref_offset_ms = find_4bar_loop_reference(
                downbeats, onsets, loop_start_bar, steps_per_bar,
                search_window_start_phase, search_window_end_phase, None
            )
            loop_ref_offsets[loop_start_bar] = loop_ref_offset_ms / 1000.0

    # 4bar_pattern_flexStart: one offset per segment (every 4 bars from flexStart_ref_bar)
    flexStart_4bar_ref_offsets = {}
    if flexStart_ref_bar is not None:
        for segment_start in range(flexStart_ref_bar, last_bar + 1, pattern_len):
            if segment_start + pattern_len <= len(downbeats) - 1:
                segment_start_time = downbeats[segment_start]
                segment_end_time = downbeats[segment_start + pattern_len]
                segment_duration = segment_end_time - segment_start_time
                total_sixteenths = pattern_len * steps_per_bar
                sixteenth_duration = segment_duration / total_sixteenths

                # Find reference onset for THIS segment
                segment_grid_time = segment_start_time
                window_start = segment_grid_time - search_window_start_phase * sixteenth_duration
                window_end = segment_grid_time + search_window_end_phase * sixteenth_duration
                onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

                if len(onsets_in_window) > 0:
                    distances = np.abs(onsets_in_window - segment_grid_time)
                    min_idx = np.argmin(distances)
                    nearest_onset = onsets_in_window[min_idx]
                    segment_ref_offset_s = (nearest_onset - segment_grid_time)
                else:
                    segment_ref_offset_s = 0.0

                flexStart_4bar_ref_offsets[segment_start] = segment_ref_offset_s

    # 2bar_pattern_flexStart: one offset per segment (every 2 bars from flexStart_2bar_ref_bar)
    flexStart_2bar_ref_offsets = {}
    pattern_len_2bar = 2
    if flexStart_2bar_ref_bar is not None:
        for segment_start in range(flexStart_2bar_ref_bar, last_bar + 1, pattern_len_2bar):
            if segment_start + pattern_len_2bar <= len(downbeats) - 1:
                segment_start_time = downbeats[segment_start]
                segment_end_time = downbeats[segment_start + pattern_len_2bar]
                segment_duration = segment_end_time - segment_start_time
                total_sixteenths = pattern_len_2bar * steps_per_bar
                sixteenth_duration = segment_duration / total_sixteenths

                segment_grid_time = segment_start_time
                window_start = segment_grid_time - search_window_start_phase * sixteenth_duration
                window_end = segment_grid_time + search_window_end_phase * sixteenth_duration
                onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

                if len(onsets_in_window) > 0:
                    distances = np.abs(onsets_in_window - segment_grid_time)
                    min_idx = np.argmin(distances)
                    nearest_onset = onsets_in_window[min_idx]
                    segment_ref_offset_s = (nearest_onset - segment_grid_time)
                else:
                    segment_ref_offset_s = 0.0

                flexStart_2bar_ref_offsets[segment_start] = segment_ref_offset_s

    # 1bar_pattern_flexStart: one offset per bar (every bar from flexStart_1bar_ref_bar)
    flexStart_1bar_ref_offsets = {}
    pattern_len_1bar = 1
    if flexStart_1bar_ref_bar is not None:
        for segment_start in range(flexStart_1bar_ref_bar, last_bar + 1, pattern_len_1bar):
            if segment_start + pattern_len_1bar <= len(downbeats) - 1:
                segment_start_time = downbeats[segment_start]
                segment_end_time = downbeats[segment_start + pattern_len_1bar]
                segment_duration = segment_end_time - segment_start_time
                total_sixteenths = pattern_len_1bar * steps_per_bar
                sixteenth_duration = segment_duration / total_sixteenths

                segment_grid_time = segment_start_time
                window_start = segment_grid_time - search_window_start_phase * sixteenth_duration
                window_end = segment_grid_time + search_window_end_phase * sixteenth_duration
                onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

                if len(onsets_in_window) > 0:
                    distances = np.abs(onsets_in_window - segment_grid_time)
                    min_idx = np.argmin(distances)
                    nearest_onset = onsets_in_window[min_idx]
                    segment_ref_offset_s = (nearest_onset - segment_grid_time)
                else:
                    segment_ref_offset_s = 0.0

                flexStart_1bar_ref_offsets[segment_start] = segment_ref_offset_s

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
        # Find which 4-bar loop this bar belongs to (relative to first_bar)
        pattern_len = 4
        bars_from_first = bar_num_abs - first_bar
        loop_start_bar = first_bar + (bars_from_first // pattern_len) * pattern_len

        if loop_start_bar in loop_ref_offsets:
            loop_start_time = downbeats[loop_start_bar]
            loop_end_time = downbeats[loop_start_bar + pattern_len]
            loop_duration = loop_end_time - loop_start_time
            total_sixteenths = pattern_len * steps_per_bar
            sixteenth_duration = loop_duration / total_sixteenths

            # Use pre-calculated loop reference offset
            loop_ref_offset_s = loop_ref_offsets[loop_start_bar]

            # Calculate equidistant grid time for this bar
            bar_offset_in_loop = bar_num_abs - loop_start_bar
            bar_sixteenth_pos = bar_offset_in_loop * steps_per_bar
            equi_bar_start = loop_start_time + (bar_sixteenth_pos * sixteenth_duration)

            grid_time_4bar = (equi_bar_start + loop_ref_offset_s) + (tick / steps_per_bar) * (steps_per_bar * sixteenth_duration)
            grid_times_4bar_loop.append(grid_time_4bar)
        else:
            grid_times_4bar_loop.append(None)

        # 4-bar pattern flexStart grid time (EQUIDISTANT)
        # Check if this bar is in a segment starting from flexStart_ref_bar
        if flexStart_ref_bar is not None:
            # Calculate which 4-bar segment this bar belongs to
            bars_from_ref = bar_num_abs - flexStart_ref_bar
            if bars_from_ref >= 0:
                segment_start_bar = flexStart_ref_bar + ((bar_num_abs - flexStart_ref_bar) // pattern_len) * pattern_len

                if segment_start_bar in flexStart_4bar_ref_offsets:
                    # Calculate equidistant grid for THIS segment
                    segment_start_time = downbeats[segment_start_bar]
                    segment_end_time = downbeats[segment_start_bar + pattern_len]
                    segment_duration = segment_end_time - segment_start_time
                    total_sixteenths = pattern_len * steps_per_bar
                    sixteenth_duration = segment_duration / total_sixteenths

                    # Use pre-calculated segment reference offset
                    segment_ref_offset_s = flexStart_4bar_ref_offsets[segment_start_bar]

                    # Calculate equidistant bar position within segment
                    bar_offset_in_segment = bar_num_abs - segment_start_bar
                    bar_sixteenth_pos = bar_offset_in_segment * steps_per_bar
                    equi_bar_start = segment_start_time + (bar_sixteenth_pos * sixteenth_duration)
                    equi_bar_duration = steps_per_bar * sixteenth_duration

                    # Apply correction
                    corrected_equi_bar_start = equi_bar_start + segment_ref_offset_s
                    grid_time_flexStart = corrected_equi_bar_start + (tick / steps_per_bar) * equi_bar_duration
                    grid_times_4bar_pattern_flexStart.append(grid_time_flexStart)
                else:
                    grid_times_4bar_pattern_flexStart.append(None)
            else:
                grid_times_4bar_pattern_flexStart.append(None)
        else:
            grid_times_4bar_pattern_flexStart.append(None)

        # 2-bar pattern flexStart grid time (EQUIDISTANT)
        # Check if this bar is in a segment starting from flexStart_2bar_ref_bar
        pattern_len_2bar = 2
        if flexStart_2bar_ref_bar is not None:
            # Calculate which 2-bar segment this bar belongs to
            bars_from_ref = bar_num_abs - flexStart_2bar_ref_bar
            if bars_from_ref >= 0:
                segment_start_bar = flexStart_2bar_ref_bar + ((bar_num_abs - flexStart_2bar_ref_bar) // pattern_len_2bar) * pattern_len_2bar

                if segment_start_bar in flexStart_2bar_ref_offsets:
                    # Calculate equidistant grid for THIS segment
                    segment_start_time = downbeats[segment_start_bar]
                    segment_end_time = downbeats[segment_start_bar + pattern_len_2bar]
                    segment_duration = segment_end_time - segment_start_time
                    total_sixteenths = pattern_len_2bar * steps_per_bar
                    sixteenth_duration = segment_duration / total_sixteenths

                    # Use pre-calculated segment reference offset
                    segment_ref_offset_s = flexStart_2bar_ref_offsets[segment_start_bar]

                    # Calculate equidistant bar position within segment
                    bar_offset_in_segment = bar_num_abs - segment_start_bar
                    bar_sixteenth_pos = bar_offset_in_segment * steps_per_bar
                    equi_bar_start = segment_start_time + (bar_sixteenth_pos * sixteenth_duration)
                    equi_bar_duration = steps_per_bar * sixteenth_duration

                    # Apply correction
                    corrected_equi_bar_start = equi_bar_start + segment_ref_offset_s
                    grid_time_2bar_flexStart = corrected_equi_bar_start + (tick / steps_per_bar) * equi_bar_duration
                    grid_times_2bar_pattern_flexStart.append(grid_time_2bar_flexStart)
                else:
                    grid_times_2bar_pattern_flexStart.append(None)
            else:
                grid_times_2bar_pattern_flexStart.append(None)
        else:
            grid_times_2bar_pattern_flexStart.append(None)

        # 1-bar pattern flexStart grid time (EQUIDISTANT)
        # Check if this bar is in a segment starting from flexStart_1bar_ref_bar
        pattern_len_1bar = 1
        if flexStart_1bar_ref_bar is not None:
            # Calculate which 1-bar segment this bar belongs to
            bars_from_ref = bar_num_abs - flexStart_1bar_ref_bar
            if bars_from_ref >= 0:
                # This bar is in a valid segment (every bar is a segment for 1-bar pattern)
                segment_start_bar = bar_num_abs

                if segment_start_bar in flexStart_1bar_ref_offsets:
                    # Calculate equidistant grid for THIS segment (1 bar)
                    segment_start_time = downbeats[segment_start_bar]
                    segment_end_time = downbeats[segment_start_bar + pattern_len_1bar]
                    segment_duration = segment_end_time - segment_start_time
                    total_sixteenths = pattern_len_1bar * steps_per_bar
                    sixteenth_duration = segment_duration / total_sixteenths

                    # Use pre-calculated segment reference offset
                    segment_ref_offset_s = flexStart_1bar_ref_offsets[segment_start_bar]

                    # Calculate equidistant bar position (bar 0 for 1-bar pattern)
                    bar_offset_in_segment = 0  # Always 0 for 1-bar pattern
                    bar_sixteenth_pos = bar_offset_in_segment * steps_per_bar
                    equi_bar_start = segment_start_time + (bar_sixteenth_pos * sixteenth_duration)
                    equi_bar_duration = steps_per_bar * sixteenth_duration

                    # Apply correction
                    corrected_equi_bar_start = equi_bar_start + segment_ref_offset_s
                    grid_time_1bar_flexStart = corrected_equi_bar_start + (tick / steps_per_bar) * equi_bar_duration
                    grid_times_1bar_pattern_flexStart.append(grid_time_1bar_flexStart)
                else:
                    grid_times_1bar_pattern_flexStart.append(None)
            else:
                grid_times_1bar_pattern_flexStart.append(None)
        else:
            grid_times_1bar_pattern_flexStart.append(None)

    df_comprehensive['grid_time_uncorrected'] = grid_times_uncorrected
    df_comprehensive['grid_time_per_snippet'] = grid_times_per_snippet
    df_comprehensive['grid_time_4bar_loop'] = grid_times_4bar_loop
    df_comprehensive['grid_time_4bar_pattern_flexStart'] = grid_times_4bar_pattern_flexStart
    df_comprehensive['grid_time_2bar_pattern_flexStart'] = grid_times_2bar_pattern_flexStart
    df_comprehensive['grid_time_1bar_pattern_flexStart'] = grid_times_1bar_pattern_flexStart
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

    # 4-bar pattern flexStart references (one every 4 bars from flexStart_ref_bar)
    # Each segment finds its own reference onset at its start bar
    if flexStart_ref_bar is not None:
        for segment_start in range(flexStart_ref_bar, last_bar + 1, pattern_len):
            if segment_start < len(downbeats) - 1:
                bar_start = downbeats[segment_start]
                bar_end = downbeats[segment_start + 1]
                bar_duration = bar_end - bar_start
                step_duration = bar_duration / steps_per_bar

                # Grid time at tick 0 (1/16th position) of this bar
                grid_time = bar_start

                # Find reference onset for THIS segment
                window_start = grid_time - search_window_start_phase * step_duration
                window_end = grid_time + search_window_end_phase * step_duration
                onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

                if len(onsets_in_window) > 0:
                    # Find closest onset to grid_time
                    distances = np.abs(onsets_in_window - grid_time)
                    min_idx = np.argmin(distances)
                    nearest_onset = onsets_in_window[min_idx]
                    segment_ref_offset_ms = (nearest_onset - grid_time) * 1000.0
                    segment_ref_offset_s = segment_ref_offset_ms / 1000.0
                else:
                    # No reference found for this segment, use 0
                    segment_ref_offset_ms = 0.0
                    segment_ref_offset_s = 0.0

                # Reference onset is offset from grid
                ref_onset_time = grid_time + segment_ref_offset_s

                # Calculate phases
                ref_phase = (ref_onset_time - bar_start) / bar_duration if bar_duration > 0 else 0.0
                grid_phase = (grid_time - bar_start) / bar_duration if bar_duration > 0 else 0.0

                ref_rows.append({
                    'method': '4bar_pattern_flexStart',
                    'bar_number': segment_start - first_bar,
                    'bar_number_global': segment_start,
                    'ref_ms': segment_ref_offset_ms,
                    'ref_phase': ref_phase,
                    'grid_phase': grid_phase,
                    'bar_duration': bar_duration
                })

    # 2-bar pattern flexStart references (one every 2 bars from flexStart_2bar_ref_bar)
    # Each segment finds its own reference onset at its start bar
    pattern_len_2bar = 2
    if flexStart_2bar_ref_bar is not None:
        for segment_start in range(flexStart_2bar_ref_bar, last_bar + 1, pattern_len_2bar):
            if segment_start < len(downbeats) - 1:
                bar_start = downbeats[segment_start]
                bar_end = downbeats[segment_start + 1]
                bar_duration = bar_end - bar_start
                step_duration = bar_duration / steps_per_bar

                # Grid time at tick 0 (1/16th position) of this bar
                grid_time = bar_start

                # Find reference onset for THIS segment
                window_start = grid_time - search_window_start_phase * step_duration
                window_end = grid_time + search_window_end_phase * step_duration
                onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

                if len(onsets_in_window) > 0:
                    # Find closest onset to grid_time
                    distances = np.abs(onsets_in_window - grid_time)
                    min_idx = np.argmin(distances)
                    nearest_onset = onsets_in_window[min_idx]
                    segment_ref_offset_ms = (nearest_onset - grid_time) * 1000.0
                    segment_ref_offset_s = segment_ref_offset_ms / 1000.0
                else:
                    # No reference found for this segment, use 0
                    segment_ref_offset_ms = 0.0
                    segment_ref_offset_s = 0.0

                # Reference onset is offset from grid
                ref_onset_time = grid_time + segment_ref_offset_s

                # Calculate phases
                ref_phase = (ref_onset_time - bar_start) / bar_duration if bar_duration > 0 else 0.0
                grid_phase = (grid_time - bar_start) / bar_duration if bar_duration > 0 else 0.0

                ref_rows.append({
                    'method': '2bar_pattern_flexStart',
                    'bar_number': segment_start - first_bar,
                    'bar_number_global': segment_start,
                    'ref_ms': segment_ref_offset_ms,
                    'ref_phase': ref_phase,
                    'grid_phase': grid_phase,
                    'bar_duration': bar_duration
                })

    # 1-bar pattern flexStart references (one every bar from flexStart_1bar_ref_bar)
    # Each bar finds its own reference onset at its start
    pattern_len_1bar = 1
    if flexStart_1bar_ref_bar is not None:
        for segment_start in range(flexStart_1bar_ref_bar, last_bar + 1, pattern_len_1bar):
            if segment_start < len(downbeats) - 1:
                bar_start = downbeats[segment_start]
                bar_end = downbeats[segment_start + 1]
                bar_duration = bar_end - bar_start
                step_duration = bar_duration / steps_per_bar

                # Grid time at tick 0 (1/16th position) of this bar
                grid_time = bar_start

                # Find reference onset for THIS bar
                window_start = grid_time - search_window_start_phase * step_duration
                window_end = grid_time + search_window_end_phase * step_duration
                onsets_in_window = onsets[(onsets >= window_start) & (onsets <= window_end)]

                if len(onsets_in_window) > 0:
                    # Find closest onset to grid_time
                    distances = np.abs(onsets_in_window - grid_time)
                    min_idx = np.argmin(distances)
                    nearest_onset = onsets_in_window[min_idx]
                    segment_ref_offset_ms = (nearest_onset - grid_time) * 1000.0
                    segment_ref_offset_s = segment_ref_offset_ms / 1000.0
                else:
                    # No reference found for this bar, use 0
                    segment_ref_offset_ms = 0.0
                    segment_ref_offset_s = 0.0

                # Reference onset is offset from grid
                ref_onset_time = grid_time + segment_ref_offset_s

                # Calculate phases
                ref_phase = (ref_onset_time - bar_start) / bar_duration if bar_duration > 0 else 0.0
                grid_phase = (grid_time - bar_start) / bar_duration if bar_duration > 0 else 0.0

                ref_rows.append({
                    'method': '1bar_pattern_flexStart',
                    'bar_number': segment_start - first_bar,
                    'bar_number_global': segment_start,
                    'ref_ms': segment_ref_offset_ms,
                    'ref_phase': ref_phase,
                    'grid_phase': grid_phase,
                    'bar_duration': bar_duration
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

    # Create flexStart patterns CSVs (3 separate files, one for each pattern)
    print("  Creating flexStart patterns CSVs...")
    create_flexstart_patterns_csv(
        output_file,
        output_path.parent,
        output_path.stem,
        first_bar,
        snippet_offset,
        flexStart_4bar_ref_bar=flexStart_ref_bar,
        flexStart_2bar_ref_bar=flexStart_2bar_ref_bar,
        flexStart_1bar_ref_bar=flexStart_1bar_ref_bar,
        snippet_duration=snippet_duration
    )

    # Import and run filter
    print("  Creating filtered flexStart patterns CSVs (hybrid filtering)...")
    try:
        from .filter_bars_and_onsets import filter_all_flexstart_patterns
        filter_all_flexstart_patterns(
            output_path.parent,
            output_path.stem,
            iqr_multiplier=IQR_MULTIPLIER_TUKEY,
            threshold=RUNNING_MEAN_THRESHOLD,
            no_of_repetitions_TH=NO_OF_REPETITIONS_TH,
            snippet_offset=snippet_offset
        )
    except Exception as e:
        print(f"  ! Warning: Could not create filtered CSVs: {e}")

    return df_comprehensive


def create_flexstart_patterns_csv(
    comprehensive_csv_file: str,
    output_dir: Path,
    base_name: str,
    first_bar: int,
    snippet_offset: float,
    flexStart_4bar_ref_bar: int = None,
    flexStart_2bar_ref_bar: int = None,
    flexStart_1bar_ref_bar: int = None,
    snippet_duration: float = SNIPPET_DURATION_S
):
    """
    Create 3 separate CSVs with flexStart pattern data for complete loops.

    Parameters
    ----------
    comprehensive_csv_file : str
        Path to comprehensive phases CSV file
    output_dir : Path
        Output directory
    base_name : str
        Base filename (without extension)
    first_bar : int
        First bar of snippet (global bar number)
    flexStart_4bar_ref_bar : int
        Reference bar for 4-bar pattern (global)
    flexStart_2bar_ref_bar : int
        Reference bar for 2-bar pattern (global)
    flexStart_1bar_ref_bar : int
        Reference bar for 1-bar pattern (global)
    """
    # Load comprehensive CSV
    df = pd.read_csv(comprehensive_csv_file)

    # Add global bar number column
    df['bar_number_global'] = df['bar_number'] + first_bar

    # Calculate snippet boundaries
    snippet_end = snippet_offset + snippet_duration

    # 4-bar pattern flexStart
    if flexStart_4bar_ref_bar is not None:
        ref_bar_snippet = flexStart_4bar_ref_bar - first_bar
        max_bar = df['bar_number'].max()
        last_complete_4bar = ref_bar_snippet + ((max_bar - ref_bar_snippet) // 4) * 4 + 3

        # Filter to complete patterns that are entirely within snippet bounds
        df_4bar_temp = df[
            (df['bar_number'] >= ref_bar_snippet) &
            (df['bar_number'] <= last_complete_4bar)
        ].copy()

        # Further filter to only include patterns within snippet time bounds
        # A pattern is included only if all its bars are within [snippet_offset, snippet_end]
        if not df_4bar_temp.empty and 'grid_time_4bar_pattern_flexStart' in df_4bar_temp.columns:
            # Find the last bar whose end is within snippet bounds
            # Check each 4-bar pattern and only include it if it ends before snippet_end
            valid_bars = []
            for pattern_start_bar in range(ref_bar_snippet, last_complete_4bar + 1, 4):
                pattern_end_bar = pattern_start_bar + 3
                # Get the last grid position in the pattern end bar
                pattern_last_row = df_4bar_temp[df_4bar_temp['bar_number'] == pattern_end_bar]
                if not pattern_last_row.empty:
                    pattern_end_time = pattern_last_row['grid_time_4bar_pattern_flexStart'].max()
                    if pattern_end_time <= snippet_end:
                        # This pattern fits entirely within snippet
                        valid_bars.extend(range(pattern_start_bar, pattern_end_bar + 1))

            df_4bar = df_4bar_temp[df_4bar_temp['bar_number'].isin(valid_bars)].copy()
        else:
            df_4bar = df_4bar_temp

        # Select columns (include tick_phase if available)
        columns_to_include = [
            'bar_number',
            'bar_number_global',
            'tick_16th',
            'onset_time_4bar_pattern_flexStart',
            'phase_4bar_pattern_flexStart',
            'grid_time_4bar_pattern_flexStart',
            'grid_phase'
        ]
        rename_mapping = {
            'onset_time_4bar_pattern_flexStart': 'onset_time',
            'phase_4bar_pattern_flexStart': 'phase',
            'grid_time_4bar_pattern_flexStart': 'grid_time'
        }

        # Add tick_phase column if it exists
        if 'tick_phase_4bar_pattern_flexStart' in df_4bar.columns:
            columns_to_include.append('tick_phase_4bar_pattern_flexStart')
            rename_mapping['tick_phase_4bar_pattern_flexStart'] = 'tick_phase'

        df_4bar = df_4bar[columns_to_include].rename(columns=rename_mapping)

        # Save to CSV with metadata
        output_file = output_dir / f"{base_name}_4bar_flexStart.csv"
        snippet_end = snippet_offset + snippet_duration
        with open(output_file, 'w') as f:
            f.write(f"# snippet_offset={snippet_offset:.6f}\n")
            f.write(f"# snippet_end={snippet_end:.6f}\n")
            df_4bar.to_csv(f, index=False)
        print(f"    ✓ 4-bar flexStart: {len(df_4bar)} rows → {output_file.name}")

    # 2-bar pattern flexStart
    if flexStart_2bar_ref_bar is not None:
        ref_bar_snippet = flexStart_2bar_ref_bar - first_bar
        max_bar = df['bar_number'].max()
        last_complete_2bar = ref_bar_snippet + ((max_bar - ref_bar_snippet) // 2) * 2 + 1

        # Filter to complete patterns that are entirely within snippet bounds
        df_2bar_temp = df[
            (df['bar_number'] >= ref_bar_snippet) &
            (df['bar_number'] <= last_complete_2bar)
        ].copy()

        # Further filter to only include patterns within snippet time bounds
        if not df_2bar_temp.empty and 'grid_time_2bar_pattern_flexStart' in df_2bar_temp.columns:
            valid_bars = []
            for pattern_start_bar in range(ref_bar_snippet, last_complete_2bar + 1, 2):
                pattern_end_bar = pattern_start_bar + 1
                pattern_last_row = df_2bar_temp[df_2bar_temp['bar_number'] == pattern_end_bar]
                if not pattern_last_row.empty:
                    pattern_end_time = pattern_last_row['grid_time_2bar_pattern_flexStart'].max()
                    if pattern_end_time <= snippet_end:
                        valid_bars.extend(range(pattern_start_bar, pattern_end_bar + 1))

            df_2bar = df_2bar_temp[df_2bar_temp['bar_number'].isin(valid_bars)].copy()
        else:
            df_2bar = df_2bar_temp

        # Select columns (include tick_phase if available)
        columns_to_include = [
            'bar_number',
            'bar_number_global',
            'tick_16th',
            'onset_time_2bar_pattern_flexStart',
            'phase_2bar_pattern_flexStart',
            'grid_time_2bar_pattern_flexStart',
            'grid_phase'
        ]
        rename_mapping = {
            'onset_time_2bar_pattern_flexStart': 'onset_time',
            'phase_2bar_pattern_flexStart': 'phase',
            'grid_time_2bar_pattern_flexStart': 'grid_time'
        }

        # Add tick_phase column if it exists
        if 'tick_phase_2bar_pattern_flexStart' in df_2bar.columns:
            columns_to_include.append('tick_phase_2bar_pattern_flexStart')
            rename_mapping['tick_phase_2bar_pattern_flexStart'] = 'tick_phase'

        df_2bar = df_2bar[columns_to_include].rename(columns=rename_mapping)

        output_file = output_dir / f"{base_name}_2bar_flexStart.csv"
        snippet_end = snippet_offset + snippet_duration
        with open(output_file, 'w') as f:
            f.write(f"# snippet_offset={snippet_offset:.6f}\n")
            f.write(f"# snippet_end={snippet_end:.6f}\n")
            df_2bar.to_csv(f, index=False)
        print(f"    ✓ 2-bar flexStart: {len(df_2bar)} rows → {output_file.name}")

    # 1-bar pattern flexStart
    if flexStart_1bar_ref_bar is not None:
        ref_bar_snippet = flexStart_1bar_ref_bar - first_bar

        # Filter to patterns that are entirely within snippet bounds
        df_1bar_temp = df[df['bar_number'] >= ref_bar_snippet].copy()

        # Further filter to only include bars within snippet time bounds
        if not df_1bar_temp.empty and 'grid_time_1bar_pattern_flexStart' in df_1bar_temp.columns:
            valid_bars = []
            for bar in df_1bar_temp['bar_number'].unique():
                bar_last_row = df_1bar_temp[df_1bar_temp['bar_number'] == bar]
                if not bar_last_row.empty:
                    bar_end_time = bar_last_row['grid_time_1bar_pattern_flexStart'].max()
                    if bar_end_time <= snippet_end:
                        valid_bars.append(bar)

            df_1bar = df_1bar_temp[df_1bar_temp['bar_number'].isin(valid_bars)].copy()
        else:
            df_1bar = df_1bar_temp

        # Select columns (include tick_phase if available)
        columns_to_include = [
            'bar_number',
            'bar_number_global',
            'tick_16th',
            'onset_time_1bar_pattern_flexStart',
            'phase_1bar_pattern_flexStart',
            'grid_time_1bar_pattern_flexStart',
            'grid_phase'
        ]
        rename_mapping = {
            'onset_time_1bar_pattern_flexStart': 'onset_time',
            'phase_1bar_pattern_flexStart': 'phase',
            'grid_time_1bar_pattern_flexStart': 'grid_time'
        }

        # Add tick_phase column if it exists
        if 'tick_phase_1bar_pattern_flexStart' in df_1bar.columns:
            columns_to_include.append('tick_phase_1bar_pattern_flexStart')
            rename_mapping['tick_phase_1bar_pattern_flexStart'] = 'tick_phase'

        df_1bar = df_1bar[columns_to_include].rename(columns=rename_mapping)

        output_file = output_dir / f"{base_name}_1bar_flexStart.csv"
        snippet_end = snippet_offset + snippet_duration
        with open(output_file, 'w') as f:
            f.write(f"# snippet_offset={snippet_offset:.6f}\n")
            f.write(f"# snippet_end={snippet_end:.6f}\n")
            df_1bar.to_csv(f, index=False)
        print(f"    ✓ 1-bar flexStart: {len(df_1bar)} rows → {output_file.name}")


# Backward compatibility alias
def create_comprehensive_csv(
    corrected_downbeats_file: str,
    onset_file: str,
    pattern_lengths: Dict[str, int],  # Ignored in new version
    snippet_offset: float,
    output_file: str,
    snippet_duration: float = SNIPPET_DURATION_S
) -> pd.DataFrame:
    """
    Backward compatibility wrapper for create_raster_csv.

    The pattern_lengths parameter is ignored as we now use fixed 4-bar loops.
    """
    return create_raster_csv(
        corrected_downbeats_file,
        onset_file,
        snippet_offset,
        output_file,
        snippet_duration=snippet_duration
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
