"""
Raster plot generation and phase calculations.

This module calculates phase deviations of onsets from expected grid positions
using multiple correction methods: uncorrected, per-snippet, and loop-based.

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

# Import from config.py (not config/__init__.py)
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
MAX_MATCH_FRAC = 0.5  # Max distance for onset matching (0.5 × step duration)
DEVIATION_MS_DIGITS = 2


# ============================================================================
# INPUT PARSING
# ============================================================================

def parse_corrected_downbeats(corrected_file: str) -> Tuple[List[float], int, np.ndarray]:
    """
    Parse corrected downbeats file.

    Parameters
    ----------
    corrected_file : str
        Path to corrected downbeats file

    Returns
    -------
    tuple
        (downbeat_times: List[float], time_signature: int, usable_mask: np.ndarray)

    Examples
    --------
    >>> downbeats, tsig, usable = parse_corrected_downbeats('track_corrected.txt')
    >>> tsig
    4
    >>> len(downbeats)
    84
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

    # Extract usable mask
    usable_mask = df['usable'].values if 'usable' in df.columns else np.ones(len(df), dtype=bool)

    return downbeat_times, tsig, usable_mask


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

    Examples
    --------
    >>> onsets = load_onsets('track_onsets.csv')
    >>> len(onsets)
    1234
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


def load_pattern_lengths(pattern_file: str, track_id: str) -> Dict[str, int]:
    """
    Load pattern lengths for drum, mel, and pitch methods.

    Parameters
    ----------
    pattern_file : str
        Path to pattern length summary CSV
    track_id : str
        Track identifier

    Returns
    -------
    dict
        Dictionary with keys 'drum', 'mel', 'pitch' mapping to pattern lengths

    Examples
    --------
    >>> lengths = load_pattern_lengths('pattern_summary.csv', '123')
    >>> lengths
    {'drum': 4, 'mel': 2, 'pitch': 4}
    """
    df = pd.read_csv(pattern_file)

    # Find row for this track
    track_row = df[df['song_id'].astype(str) == str(track_id)]

    if len(track_row) == 0:
        # Default to 4-bar patterns
        return {'drum': 4, 'mel': 4, 'pitch': 4}

    row = track_row.iloc[0]

    return {
        'drum': int(row.get('drum_L_pow2', 4)),
        'mel': int(row.get('mel_L_pow2', 4)),
        'pitch': int(row.get('pitch_L_pow2', 4))
    }


def load_snippet_offset(overview_file: str, track_id: str) -> float:
    """
    Load snippet start offset for a track.

    Matches the notebook logic: looks for columns with 'corrected', 'offset', and 'ms'.

    Parameters
    ----------
    overview_file : str
        Path to overview CSV with snippet offsets
    track_id : str
        Track identifier (can be numeric ID or song name)

    Returns
    -------
    float
        Snippet offset in seconds

    Examples
    --------
    >>> offset = load_snippet_offset('overview.csv', '123')
    >>> offset
    10.5
    """
    # Load CSV with semicolon separator (as in notebook)
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

    # If not found, try extracting numeric ID from filename (e.g., "0_Save Your Tears - The Weeknd" -> "0")
    if len(track_row) == 0 and '_' in str(track_id):
        potential_id = str(track_id).split('_')[0]
        if potential_id.isdigit():
            track_row = df[df[id_col].astype(str) == potential_id]

    # If still not found, try matching by song name (exact match)
    if len(track_row) == 0 and 'songname' in df.columns:
        track_row = df[df['songname'].astype(str).str.lower() == str(track_id).lower()]

    # If still not found, try partial match (in case filename has artist suffix like "Song - Artist")
    if len(track_row) == 0 and 'songname' in df.columns and ' - ' in str(track_id):
        # Extract song name before " - " (e.g., "Save Your Tears - The Weeknd" -> "Save Your Tears")
        song_part = str(track_id).split(' - ')[0]
        # Also handle "0_Save Your Tears - The Weeknd" -> "Save Your Tears"
        if '_' in song_part:
            song_part = song_part.split('_', 1)[1]
        track_row = df[df['songname'].astype(str).str.lower() == song_part.lower()]

    if len(track_row) == 0:
        return 0.0

    # Find offset column: prioritize "corrected offset ms"
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
    return offset_s  # Convert ms to seconds


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

    Examples
    --------
    >>> onsets = np.array([1.0, 2.0, 3.0])
    >>> nearest, dist = find_nearest_onset(2.1, onsets)
    >>> nearest
    2.0
    >>> dist
    0.1
    """
    if len(onsets) == 0:
        return np.nan, np.inf

    idx = np.argmin(np.abs(onsets - target_time))
    nearest = onsets[idx]
    distance = abs(nearest - target_time)

    return nearest, distance


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

    Examples
    --------
    >>> downbeats = [0, 2, 4, 6, 8, 10, 12]
    >>> first, last = calculate_snippet_bars(downbeats, 3.0, 6.0)
    >>> first, last
    (1, 4)
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


# ============================================================================
# REFERENCE OFFSET CALCULATION
# ============================================================================

def find_reference_offset_priority(
    downbeats: List[float],
    onsets: np.ndarray,
    first_bar: int,
    num_bars: int,
    steps_per_bar: int,
    max_match_frac: float = MAX_MATCH_FRAC
) -> Tuple[float, int, float, float]:
    """
    Find reference offset using priority tick positions.

    Searches for reference onset at priority ticks: [0, 8, 4, 12]
    (downbeat, half-note, quarter-notes).

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    onsets : np.ndarray
        Onset times
    first_bar : int
        First bar index to search
    num_bars : int
        Number of bars to search (typically 1 for per-snippet)
    steps_per_bar : int
        Number of ticks per bar (e.g., 16)
    max_match_frac : float
        Maximum matching distance as fraction of step duration

    Returns
    -------
    tuple
        (ref_ms, ref_bar, ref_phase, grid_phase)

    Examples
    --------
    >>> ref_ms, bar, ref_phase, grid_phase = find_reference_offset_priority(
    ...     downbeats, onsets, 0, 1, 16
    ... )
    >>> abs(ref_ms) < 100  # Reference offset within 100ms
    True
    """
    # Priority order: downbeat, half-note, quarter-notes
    ticks_priority = [0, 8, 4, 12]

    for tick in ticks_priority:
        for bar_offset in range(num_bars):
            bar_idx = first_bar + bar_offset

            if bar_idx >= len(downbeats) - 1:
                continue

            bar_start = downbeats[bar_idx]
            bar_end = downbeats[bar_idx + 1]
            bar_duration = bar_end - bar_start

            if bar_duration <= 0:
                continue

            # Calculate expected grid position
            step_duration = bar_duration / steps_per_bar
            grid_time = bar_start + (tick * step_duration)

            # Find nearest onset
            nearest_onset, distance = find_nearest_onset(grid_time, onsets)

            # Check if within tolerance
            if distance <= (max_match_frac * step_duration):
                # Found reference!
                ref_ms = (nearest_onset - grid_time) * 1000.0
                ref_phase = (nearest_onset - bar_start) / bar_duration
                grid_phase = (grid_time - bar_start) / bar_duration

                return ref_ms, bar_idx, ref_phase, grid_phase

    # No reference found, return zero offset
    return 0.0, first_bar, 0.0, 0.0


def find_reference_offset_loopwise(
    downbeats: List[float],
    onsets: np.ndarray,
    loop_start_bar: int,
    pattern_len: int,
    steps_per_bar: int,
    max_match_frac: float = MAX_MATCH_FRAC
) -> Tuple[float, int, float, float]:
    """
    Find reference offset for a loop using equidistant grid.

    Search Strategy
    ---------------
    The algorithm searches for a reference onset using a priority-based approach:
    1. Priority ticks: [0, 8, 4, 12] - checks downbeat first, then backbeat, then quarters
    2. Search scope: ALL bars within the current loop (not just the first bar)
    3. Search order: For each priority tick, check bar 0, bar 1, bar 2... until match found
    4. Matching criterion: Onset must be within tolerance of equidistant grid position

    Example for a 4-bar loop:
    - First checks: tick 0 in bars 0, 1, 2, 3
    - Then checks: tick 8 in bars 0, 1, 2, 3
    - Then checks: tick 4 in bars 0, 1, 2, 3
    - Then checks: tick 12 in bars 0, 1, 2, 3
    - Returns first match found, or (0.0, loop_start_bar, 0.0, 0.0) if no match

    Grid Correction
    ---------------
    Once a reference onset is found, its offset from the equidistant grid position
    is used to correct ALL onsets in that loop. This ensures the entire loop is
    aligned to the detected rhythmic pattern.

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    onsets : np.ndarray
        Onset times
    loop_start_bar : int
        Starting bar index of loop
    pattern_len : int
        Pattern length in bars
    steps_per_bar : int
        Number of ticks per bar
    max_match_frac : float
        Maximum matching distance as fraction

    Returns
    -------
    tuple
        (ref_ms, ref_bar, ref_phase, grid_phase)

    Examples
    --------
    >>> ref_ms, bar, ref_phase, grid_phase = find_reference_offset_loopwise(
    ...     downbeats, onsets, 0, 4, 16
    ... )
    """
    if loop_start_bar + pattern_len >= len(downbeats):
        return 0.0, loop_start_bar, 0.0, 0.0

    # Calculate equidistant grid for this loop
    loop_start_time = downbeats[loop_start_bar]
    loop_end_time = downbeats[loop_start_bar + pattern_len]
    loop_duration = loop_end_time - loop_start_time

    total_sixteenths = pattern_len * steps_per_bar
    sixteenth_duration = loop_duration / total_sixteenths

    # Priority ticks to search
    ticks_priority = [0, 8, 4, 12]

    for tick in ticks_priority:
        for bar_offset in range(pattern_len):  # Check ALL bars in the loop
            bar_idx = loop_start_bar + bar_offset

            # Skip if bar is out of bounds
            if bar_idx + 1 >= len(downbeats):
                continue

            # Calculate equidistant position
            global_tick = (bar_offset * steps_per_bar) + tick
            grid_time = loop_start_time + (global_tick * sixteenth_duration)

            # Find nearest onset
            nearest_onset, distance = find_nearest_onset(grid_time, onsets)

            # Check tolerance
            if distance <= (max_match_frac * sixteenth_duration):
                ref_ms = (nearest_onset - grid_time) * 1000.0

                # Calculate phases relative to equidistant bar
                equi_bar_start = loop_start_time + (bar_offset * steps_per_bar * sixteenth_duration)
                equi_bar_duration = steps_per_bar * sixteenth_duration

                ref_phase = (nearest_onset - equi_bar_start) / equi_bar_duration
                grid_phase = (grid_time - equi_bar_start) / equi_bar_duration

                return ref_ms, bar_idx, ref_phase, grid_phase

    return 0.0, loop_start_bar, 0.0, 0.0


# ============================================================================
# PHASE CALCULATION - UNCORRECTED
# ============================================================================

def assign_onsets_to_ticks_uncorrected(
    onsets: np.ndarray,
    downbeats: List[float],
    bar_start_idx: int,
    bar_end_idx: int,
    steps_per_bar: int
) -> Dict[Tuple[int, int], float]:
    """
    Assign onsets to tick positions using uncorrected bar grid.

    Parameters
    ----------
    onsets : np.ndarray
        Onset times in seconds
    downbeats : List[float]
        Downbeat times
    bar_start_idx : int
        First bar index
    bar_end_idx : int
        Last bar index (inclusive)
    steps_per_bar : int
        Number of ticks per bar (e.g., 16)

    Returns
    -------
    dict
        Mapping (bar_idx, tick) -> onset_time

    Examples
    --------
    >>> onsets = np.array([0.05, 1.0, 2.0])
    >>> downbeats = [0.0, 2.0, 4.0]
    >>> assignments = assign_onsets_to_ticks_uncorrected(onsets, downbeats, 0, 1, 16)
    """
    onset_map = {}

    for bar_idx in range(bar_start_idx, bar_end_idx + 1):
        if bar_idx >= len(downbeats) - 1:
            continue

        bar_start = downbeats[bar_idx]
        bar_end = downbeats[bar_idx + 1]
        bar_duration = bar_end - bar_start

        if bar_duration <= 0:
            continue

        # Filter onsets in this bar
        bar_onsets = onsets[(onsets >= bar_start) & (onsets < bar_end)]

        for onset_time in bar_onsets:
            # Calculate phase and assign to nearest tick
            phase = (onset_time - bar_start) / bar_duration
            nearest_tick = int(round(phase * steps_per_bar))
            nearest_tick = max(0, min(steps_per_bar - 1, nearest_tick))

            # Store onset (if multiple onsets per tick, keep first)
            key = (bar_idx, nearest_tick)
            if key not in onset_map:
                onset_map[key] = onset_time

    return onset_map


def calculate_phases_uncorrected(
    onset_map: Dict[Tuple[int, int], float],
    downbeats: List[float],
    steps_per_bar: int
) -> pd.DataFrame:
    """
    Calculate uncorrected phases for assigned onsets.

    Parameters
    ----------
    onset_map : dict
        Mapping (bar_idx, tick) -> onset_time
    downbeats : List[float]
        Downbeat times
    steps_per_bar : int
        Number of ticks per bar

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_number, tick_16th, onset_time, phase_uncorrected

    Examples
    --------
    >>> df = calculate_phases_uncorrected(onset_map, downbeats, 16)
    """
    rows = []

    for (bar_idx, tick), onset_time in onset_map.items():
        if bar_idx >= len(downbeats) - 1:
            continue

        bar_start = downbeats[bar_idx]
        bar_end = downbeats[bar_idx + 1]
        bar_duration = bar_end - bar_start

        if bar_duration <= 0:
            continue

        # Calculate phase
        phase = (onset_time - bar_start) / bar_duration

        # Add display shift (1/16)
        phase_shifted = phase + (1.0 / steps_per_bar)

        rows.append({
            'bar_number': bar_idx,
            'tick_16th': tick,
            'onset_time_uncorrected': onset_time,
            'phase_uncorrected': phase_shifted
        })

    return pd.DataFrame(rows)


# ============================================================================
# PHASE CALCULATION - PER-SNIPPET CORRECTION
# ============================================================================

def calculate_phases_per_snippet(
    onset_map: Dict[Tuple[int, int], float],
    downbeats: List[float],
    ref_ms: float,
    steps_per_bar: int,
    remap_ticks: bool = True
) -> pd.DataFrame:
    """
    Calculate per-snippet corrected phases.

    Uses grid-shift approach: shifts grid forward by ref_ms.

    Parameters
    ----------
    onset_map : dict
        Mapping (bar_idx, tick) -> onset_time (from uncorrected assignment)
    downbeats : List[float]
        Downbeat times
    ref_ms : float
        Reference offset in milliseconds
    steps_per_bar : int
        Number of ticks per bar
    remap_ticks : bool
        If True, reassign onsets to ticks based on corrected position (FIXED)
        If False, use uncorrected tick assignments (has BUG)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_number, tick_16th, phase_per_snippet
        (or phase_per_snippet_remapped if remap_ticks=True)

    Examples
    --------
    >>> df = calculate_phases_per_snippet(onset_map, downbeats, 50.0, 16)
    """
    ref_s = ref_ms / 1000.0
    rows = []

    if remap_ticks:
        # FIXED version: reassign onsets to ticks based on corrected position
        # Collect all onsets first
        onset_times = {}
        for (bar_idx, tick), onset_time in onset_map.items():
            if bar_idx not in onset_times:
                onset_times[bar_idx] = []
            onset_times[bar_idx].append(onset_time)

        # Reassign based on corrected grid
        for bar_idx, onsets_in_bar in onset_times.items():
            if bar_idx >= len(downbeats) - 1:
                continue

            bar_start = downbeats[bar_idx]
            bar_end = downbeats[bar_idx + 1]
            bar_duration = bar_end - bar_start

            if bar_duration <= 0:
                continue

            # Corrected bar start
            corrected_bar_start = bar_start + ref_s

            for onset_time in onsets_in_bar:
                # Calculate corrected phase
                phase = (onset_time - corrected_bar_start) / bar_duration

                # Assign to nearest tick based on corrected phase
                nearest_tick = int(round(phase * steps_per_bar))
                nearest_tick = max(0, min(steps_per_bar - 1, nearest_tick))

                # Add display shift
                phase_shifted = phase + (1.0 / steps_per_bar)

                rows.append({
                    'bar_number': bar_idx,
                    'tick_16th': nearest_tick,
                    'phase_per_snippet_remapped': phase_shifted
                })
    else:
        # BUGGY version: use uncorrected tick assignments
        for (bar_idx, tick), onset_time in onset_map.items():
            if bar_idx >= len(downbeats) - 1:
                continue

            bar_start = downbeats[bar_idx]
            bar_end = downbeats[bar_idx + 1]
            bar_duration = bar_end - bar_start

            if bar_duration <= 0:
                continue

            # Corrected bar start
            corrected_bar_start = bar_start + ref_s

            # Calculate phase with correction
            phase = (onset_time - corrected_bar_start) / bar_duration

            # Add display shift
            phase_shifted = phase + (1.0 / steps_per_bar)

            rows.append({
                'bar_number': bar_idx,
                'tick_16th': tick,
                'phase_per_snippet': phase_shifted
            })

    return pd.DataFrame(rows)


# ============================================================================
# PHASE CALCULATION - LOOP-BASED (EQUIDISTANT GRID)
# ============================================================================

def calculate_phases_loop_based(
    onsets: np.ndarray,
    downbeats: List[float],
    bar_start_idx: int,
    bar_end_idx: int,
    pattern_len: int,
    steps_per_bar: int,
    loop_ref_offsets: Dict[int, float]
) -> pd.DataFrame:
    """
    Calculate loop-based phases using equidistant grid.

    Parameters
    ----------
    onsets : np.ndarray
        Onset times
    downbeats : List[float]
        Downbeat times
    bar_start_idx : int
        First bar index
    bar_end_idx : int
        Last bar index (inclusive)
    pattern_len : int
        Pattern length in bars (e.g., 4 for 4-bar loops)
    steps_per_bar : int
        Number of ticks per bar
    loop_ref_offsets : dict
        Mapping loop_start_bar -> ref_ms for each loop

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_number, tick_16th, phase_loop

    Examples
    --------
    >>> df = calculate_phases_loop_based(onsets, downbeats, 0, 15, 4, 16, {0: 50.0, 4: 45.0})
    """
    rows = []
    total_sixteenths = pattern_len * steps_per_bar

    # Process each loop
    for loop_start_bar in range(bar_start_idx, bar_end_idx + 1, pattern_len):
        if loop_start_bar + pattern_len > len(downbeats) - 1:
            break

        # Calculate equidistant grid for this loop
        loop_start_time = downbeats[loop_start_bar]
        loop_end_time = downbeats[loop_start_bar + pattern_len]
        loop_duration = loop_end_time - loop_start_time
        sixteenth_duration = loop_duration / total_sixteenths

        # Get reference offset for this loop
        ref_tuple = loop_ref_offsets.get(loop_start_bar, (0.0, loop_start_bar, 0.0, 0.0))
        ref_ms = ref_tuple[0]
        ref_s = ref_ms / 1000.0

        # Process each bar in loop
        for bar_offset in range(pattern_len):
            bar_idx = loop_start_bar + bar_offset

            if bar_idx > bar_end_idx or bar_idx >= len(downbeats) - 1:
                break

            # Calculate equidistant bar boundaries
            bar_sixteenth_pos = bar_offset * steps_per_bar
            equi_bar_start = loop_start_time + (bar_sixteenth_pos * sixteenth_duration)
            equi_bar_duration = steps_per_bar * sixteenth_duration
            equi_bar_end = equi_bar_start + equi_bar_duration

            # Apply correction
            corrected_equi_bar_start = equi_bar_start + ref_s

            # Filter onsets in this equidistant bar
            bar_onsets = onsets[(onsets >= equi_bar_start) & (onsets < equi_bar_end)]

            for onset_time in bar_onsets:
                # Calculate corrected phase
                phase = (onset_time - corrected_equi_bar_start) / equi_bar_duration

                # Assign to nearest tick
                nearest_tick = int(round(phase * steps_per_bar))
                nearest_tick = max(0, min(steps_per_bar - 1, nearest_tick))

                # Add display shift
                phase_shifted = phase + (1.0 / steps_per_bar)

                rows.append({
                    'bar_number': bar_idx,
                    'tick_16th': nearest_tick,
                    'phase_loop': phase_shifted
                })

    return pd.DataFrame(rows)


def calculate_loop_reference_offsets(
    downbeats: List[float],
    onsets: np.ndarray,
    bar_start_idx: int,
    bar_end_idx: int,
    pattern_len: int,
    steps_per_bar: int
) -> Dict[int, Tuple[float, int, float, float]]:
    """
    Calculate reference offsets for each loop.

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    onsets : np.ndarray
        Onset times
    bar_start_idx : int
        First bar index
    bar_end_idx : int
        Last bar index
    pattern_len : int
        Pattern length in bars
    steps_per_bar : int
        Number of ticks per bar

    Returns
    -------
    dict
        Mapping loop_start_bar -> (ref_ms, bar_idx, ref_phase, grid_phase)

    Examples
    --------
    >>> offsets = calculate_loop_reference_offsets(downbeats, onsets, 0, 15, 4, 16)
    >>> offsets
    {0: (50.5, 0, 0.52, 0.0625), 4: (48.2, 4, 0.48, 0.0625), ...}
    """
    loop_offsets = {}

    for loop_start_bar in range(bar_start_idx, bar_end_idx + 1, pattern_len):
        if loop_start_bar + pattern_len > len(downbeats) - 1:
            break

        ref_ms, bar_idx, ref_phase, grid_phase = find_reference_offset_loopwise(
            downbeats,
            onsets,
            loop_start_bar,
            pattern_len,
            steps_per_bar
        )

        # Store full tuple: (ref_ms, bar_idx, ref_phase, grid_phase)
        loop_offsets[loop_start_bar] = (ref_ms, bar_idx, ref_phase, grid_phase)

    return loop_offsets


# ============================================================================
# GRID TIME CALCULATION
# ============================================================================

def calculate_grid_times(
    downbeats: List[float],
    bar_start_idx: int,
    bar_end_idx: int,
    steps_per_bar: int,
    ref_ms: float = 0.0
) -> pd.DataFrame:
    """
    Calculate grid times for per-snippet method (actual bar interpolation).

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    bar_start_idx : int
        First bar index
    bar_end_idx : int
        Last bar index
    steps_per_bar : int
        Number of ticks per bar
    ref_ms : float
        Reference offset in milliseconds

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_number, tick_16th, grid_time_per_snippet
    """
    ref_s = ref_ms / 1000.0
    rows = []

    for bar_idx in range(bar_start_idx, bar_end_idx + 1):
        if bar_idx >= len(downbeats) - 1:
            continue

        bar_start = downbeats[bar_idx]
        bar_end = downbeats[bar_idx + 1]
        bar_duration = bar_end - bar_start

        if bar_duration <= 0:
            continue

        corrected_bar_start = bar_start + ref_s

        for tick in range(steps_per_bar):
            grid_time = corrected_bar_start + (tick / steps_per_bar) * bar_duration

            rows.append({
                'bar_number': bar_idx,
                'tick_16th': tick,
                'grid_time_per_snippet': grid_time
            })

    return pd.DataFrame(rows)


def calculate_grid_times_uncorrected(
    downbeats: List[float],
    bar_start_idx: int,
    bar_end_idx: int,
    steps_per_bar: int
) -> pd.DataFrame:
    """
    Calculate uncorrected grid times (no ref_ms correction, just raw downbeats).

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    bar_start_idx : int
        First bar index
    bar_end_idx : int
        Last bar index
    steps_per_bar : int
        Number of ticks per bar

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_number, tick_16th, grid_time_uncorrected
    """
    rows = []

    for bar_idx in range(bar_start_idx, bar_end_idx + 1):
        if bar_idx >= len(downbeats) - 1:
            continue

        bar_start = downbeats[bar_idx]
        bar_end = downbeats[bar_idx + 1]
        bar_duration = bar_end - bar_start

        if bar_duration <= 0:
            continue

        for tick in range(steps_per_bar):
            grid_time = bar_start + (tick / steps_per_bar) * bar_duration

            rows.append({
                'bar_number': bar_idx,
                'tick_16th': tick,
                'grid_time_uncorrected': grid_time
            })

    return pd.DataFrame(rows)


def calculate_grid_times_loop(
    downbeats: List[float],
    bar_start_idx: int,
    bar_end_idx: int,
    pattern_len: int,
    steps_per_bar: int,
    loop_ref_offsets: Dict[int, float]
) -> pd.DataFrame:
    """
    Calculate equidistant grid times for loop-based method.

    Parameters
    ----------
    downbeats : List[float]
        Downbeat times
    bar_start_idx : int
        First bar index
    bar_end_idx : int
        Last bar index
    pattern_len : int
        Pattern length in bars
    steps_per_bar : int
        Number of ticks per bar
    loop_ref_offsets : dict
        Mapping loop_start_bar -> ref_ms

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_number, tick_16th, grid_time_loop
    """
    rows = []
    total_sixteenths = pattern_len * steps_per_bar

    for loop_start_bar in range(bar_start_idx, bar_end_idx + 1, pattern_len):
        if loop_start_bar + pattern_len > len(downbeats) - 1:
            break

        loop_start_time = downbeats[loop_start_bar]
        loop_end_time = downbeats[loop_start_bar + pattern_len]
        loop_duration = loop_end_time - loop_start_time
        sixteenth_duration = loop_duration / total_sixteenths

        ref_tuple = loop_ref_offsets.get(loop_start_bar, (0.0, loop_start_bar, 0.0, 0.0))
        ref_ms = ref_tuple[0]
        ref_s = ref_ms / 1000.0

        for bar_offset in range(pattern_len):
            bar_idx = loop_start_bar + bar_offset

            if bar_idx > bar_end_idx or bar_idx >= len(downbeats) - 1:
                break

            for tick in range(steps_per_bar):
                global_tick = (bar_offset * steps_per_bar) + tick
                grid_time = loop_start_time + (global_tick * sixteenth_duration) + ref_s

                rows.append({
                    'bar_number': bar_idx,
                    'tick_16th': tick,
                    'grid_time_loop': grid_time
                })

    return pd.DataFrame(rows)


# ============================================================================
# COMPREHENSIVE CSV EXPORT
# ============================================================================

def create_comprehensive_csv(
    corrected_downbeats_file: str,
    onset_file: str,
    pattern_lengths: Dict[str, int],
    snippet_offset: float,
    output_file: str
) -> pd.DataFrame:
    """
    Create comprehensive phases CSV with all correction methods.

    This is the main entry point for Step 4 of the AP2 pipeline.

    Parameters
    ----------
    corrected_downbeats_file : str
        Path to corrected downbeats file
    onset_file : str
        Path to onsets CSV file
    pattern_lengths : dict
        Pattern lengths for drum, mel, pitch methods
    snippet_offset : float
        Snippet start offset in seconds
    output_file : str
        Output CSV path

    Returns
    -------
    pd.DataFrame
        Comprehensive DataFrame with all phases and grid times

    Examples
    --------
    >>> df = create_comprehensive_csv(
    ...     'track_corrected.txt',
    ...     'track_onsets.csv',
    ...     {'drum': 4, 'mel': 2, 'pitch': 4},
    ...     10.5,
    ...     'track_comprehensive.csv'
    ... )
    """
    print(f"\nCreating comprehensive phases CSV...")

    # Load inputs
    print("  Loading downbeats...")
    downbeats, tsig, usable_mask = parse_corrected_downbeats(corrected_downbeats_file)
    steps_per_bar = tsig * GRID_SUBDIV_PER_BEAT

    print("  Loading onsets...")
    onsets = load_onsets(onset_file)

    # Calculate snippet bars
    first_bar, last_bar = calculate_snippet_bars(downbeats, snippet_offset, SNIPPET_DURATION_S)
    print(f"  Snippet covers bars {first_bar} to {last_bar}")

    # Find reference offset for per-snippet method
    print("  Finding per-snippet reference offset...")
    ref_ms, ref_bar, ref_phase, grid_phase = find_reference_offset_priority(
        downbeats, onsets, first_bar, 1, steps_per_bar
    )
    print(f"    Reference: {ref_ms:.2f}ms at bar {ref_bar}")

    # Assign onsets to ticks (uncorrected)
    print("  Assigning onsets to ticks...")
    onset_map = assign_onsets_to_ticks_uncorrected(onsets, downbeats, first_bar, last_bar, steps_per_bar)

    # Calculate uncorrected phases
    print("  Calculating uncorrected phases...")
    df_uncorr = calculate_phases_uncorrected(onset_map, downbeats, steps_per_bar)

    # Calculate per-snippet phases (both versions)
    print("  Calculating per-snippet phases...")
    df_snippet_remap = calculate_phases_per_snippet(onset_map, downbeats, ref_ms, steps_per_bar, remap_ticks=True)
    df_snippet_orig = calculate_phases_per_snippet(onset_map, downbeats, ref_ms, steps_per_bar, remap_ticks=False)

    # Calculate loop-based phases
    print("  Calculating loop-based phases...")
    loop_phases = {}
    for method, pattern_len in pattern_lengths.items():
        print(f"    {method} method (L={pattern_len})...")
        loop_offsets = calculate_loop_reference_offsets(downbeats, onsets, first_bar, last_bar, pattern_len, steps_per_bar)
        df_loop = calculate_phases_loop_based(onsets, downbeats, first_bar, last_bar, pattern_len, steps_per_bar, loop_offsets)
        loop_phases[method] = (df_loop, loop_offsets, pattern_len)

    # Calculate standard loop-based phases (fixed L=1, L=2, L=4)
    print("  Calculating standard loop-based phases...")
    for standard_len in [1, 2, 4]:
        method = f'standard_L{standard_len}'
        print(f"    {method} (L={standard_len})...")
        loop_offsets = calculate_loop_reference_offsets(downbeats, onsets, first_bar, last_bar, standard_len, steps_per_bar)
        df_loop = calculate_phases_loop_based(onsets, downbeats, first_bar, last_bar, standard_len, steps_per_bar, loop_offsets)
        loop_phases[method] = (df_loop, loop_offsets, standard_len)

    # Calculate grid times
    print("  Calculating grid times...")
    df_grid_uncorrected = calculate_grid_times_uncorrected(downbeats, first_bar, last_bar, steps_per_bar)
    df_grid_snippet = calculate_grid_times(downbeats, first_bar, last_bar, steps_per_bar, ref_ms)

    grid_times_loop = {}
    for method, (df_loop, loop_offsets, pattern_len) in loop_phases.items():
        df_grid_loop = calculate_grid_times_loop(downbeats, first_bar, last_bar, pattern_len, steps_per_bar, loop_offsets)
        grid_times_loop[method] = (df_grid_loop, pattern_len)

    # Create comprehensive grid (all bar/tick combinations)
    print("  Building comprehensive grid...")
    grid_rows = []
    for bar_idx in range(first_bar, last_bar + 1):
        for tick in range(steps_per_bar):
            grid_rows.append({
                'bar_number': bar_idx - first_bar,  # Snippet-relative
                'bar_number_global': bar_idx,       # Song-absolute
                'tick_16th': tick
            })

    df_comprehensive = pd.DataFrame(grid_rows)

    # Merge all phase data
    print("  Merging phase data...")

    # Merge uncorrected
    df_uncorr_merge = df_uncorr[['bar_number', 'tick_16th', 'onset_time_uncorrected', 'phase_uncorrected']].copy()
    df_uncorr_merge['bar_number'] = df_uncorr_merge['bar_number'] - first_bar
    df_comprehensive = df_comprehensive.merge(df_uncorr_merge, on=['bar_number', 'tick_16th'], how='left')

    # Merge per-snippet (remapped)
    df_snippet_remap_merge = df_snippet_remap[['bar_number', 'tick_16th', 'phase_per_snippet_remapped']].copy()
    df_snippet_remap_merge['bar_number'] = df_snippet_remap_merge['bar_number'] - first_bar
    df_comprehensive = df_comprehensive.merge(df_snippet_remap_merge, on=['bar_number', 'tick_16th'], how='left')

    # Merge per-snippet (original/buggy)
    df_snippet_orig_merge = df_snippet_orig[['bar_number', 'tick_16th', 'phase_per_snippet']].copy()
    df_snippet_orig_merge['bar_number'] = df_snippet_orig_merge['bar_number'] - first_bar
    df_comprehensive = df_comprehensive.merge(df_snippet_orig_merge, on=['bar_number', 'tick_16th'], how='left')

    # Merge loop-based phases
    for method, (df_loop, loop_offsets, pattern_len) in loop_phases.items():
        col_name = f'phase_{method}(L={pattern_len})'
        df_loop_merge = df_loop[['bar_number', 'tick_16th', 'phase_loop']].copy()
        df_loop_merge['bar_number'] = df_loop_merge['bar_number'] - first_bar
        df_loop_merge = df_loop_merge.rename(columns={'phase_loop': col_name})
        df_comprehensive = df_comprehensive.merge(df_loop_merge, on=['bar_number', 'tick_16th'], how='left')

    # Merge grid times
    print("  Merging grid times...")

    # Uncorrected grid
    df_grid_uncorrected_merge = df_grid_uncorrected[['bar_number', 'tick_16th', 'grid_time_uncorrected']].copy()
    df_grid_uncorrected_merge['bar_number'] = df_grid_uncorrected_merge['bar_number'] - first_bar
    df_comprehensive = df_comprehensive.merge(df_grid_uncorrected_merge, on=['bar_number', 'tick_16th'], how='left')

    # Per-snippet grid
    df_grid_snippet_merge = df_grid_snippet[['bar_number', 'tick_16th', 'grid_time_per_snippet']].copy()
    df_grid_snippet_merge['bar_number'] = df_grid_snippet_merge['bar_number'] - first_bar
    df_comprehensive = df_comprehensive.merge(df_grid_snippet_merge, on=['bar_number', 'tick_16th'], how='left')

    # Loop-based grids
    for method, (df_grid_loop, pattern_len) in grid_times_loop.items():
        col_name = f'grid_time_{method}(L={pattern_len})'
        df_grid_loop_merge = df_grid_loop[['bar_number', 'tick_16th', 'grid_time_loop']].copy()
        df_grid_loop_merge['bar_number'] = df_grid_loop_merge['bar_number'] - first_bar
        df_grid_loop_merge = df_grid_loop_merge.rename(columns={'grid_time_loop': col_name})
        df_comprehensive = df_comprehensive.merge(df_grid_loop_merge, on=['bar_number', 'tick_16th'], how='left')

    # Prepare output path
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save reference onsets to separate CSV
    print("  Saving reference onsets...")
    ref_rows = []

    # Per-snippet reference
    if ref_bar is not None and ref_bar >= first_bar:
        ref_rows.append({
            'method': 'per_snippet',
            'bar_number': ref_bar - first_bar,
            'bar_number_global': ref_bar,
            'ref_ms': ref_ms,
            'ref_phase': ref_phase,
            'grid_phase': grid_phase
        })

    # Loop-based references
    for method, (df_loop, loop_offsets, pattern_len) in loop_phases.items():
        for loop_start_bar, ref_tuple in loop_offsets.items():
            # Unpack the reference tuple: (ref_ms, bar_idx, ref_phase, grid_phase)
            loop_ref_ms, ref_bar_idx, ref_phase, grid_phase = ref_tuple

            # ref_bar_idx is in global coordinates, convert to snippet-relative
            if ref_bar_idx >= first_bar and ref_bar_idx <= last_bar:
                ref_rows.append({
                    'method': f'{method}(L={pattern_len})',
                    'bar_number': ref_bar_idx - first_bar,
                    'bar_number_global': ref_bar_idx,
                    'ref_ms': loop_ref_ms,
                    'ref_phase': ref_phase,
                    'grid_phase': grid_phase
                })

    if ref_rows:
        df_refs = pd.DataFrame(ref_rows)
        ref_file = output_path.parent / f"{output_path.stem}_reference_onsets.csv"
        df_refs.to_csv(ref_file, index=False)
        print(f"  ✓ Reference onsets saved: {len(df_refs)} references")

    # Save comprehensive CSV
    print(f"  Saving to {output_file}...")
    df_comprehensive.to_csv(output_file, index=False)

    print(f"  ✓ Comprehensive CSV created: {len(df_comprehensive)} rows")

    return df_comprehensive


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_basic_functions():
    """Test basic parsing and helper functions."""
    print("=" * 80)
    print("Testing Raster Module - Basic Functions")
    print("=" * 80)

    # Test find_nearest_onset
    print("\n[1] Testing find_nearest_onset...")
    onsets = np.array([1.0, 2.0, 3.5, 5.0])
    nearest, dist = find_nearest_onset(3.4, onsets)
    assert abs(nearest - 3.5) < 0.01, "Should find 3.5"
    assert abs(dist - 0.1) < 0.01, "Distance should be 0.1"
    print("  ✓ find_nearest_onset works correctly")

    # Test calculate_snippet_bars
    print("\n[2] Testing calculate_snippet_bars...")
    downbeats = [0, 2, 4, 6, 8, 10, 12]
    first, last = calculate_snippet_bars(downbeats, 3.0, 6.0)
    print(f"  Snippet 3.0-9.0s covers bars {first} to {last}")
    assert first >= 0 and last < len(downbeats), "Indices should be valid"
    print("  ✓ calculate_snippet_bars works correctly")

    # Test find_reference_offset_priority
    print("\n[3] Testing find_reference_offset_priority...")
    downbeats = [0.0, 2.0, 4.0, 6.0]
    onsets = np.array([0.05, 0.5, 1.0, 1.5, 2.05, 2.5, 3.0, 3.5])
    ref_ms, bar, ref_phase, grid_phase = find_reference_offset_priority(
        downbeats, onsets, 0, 1, 16
    )
    print(f"  Reference: {ref_ms:.2f}ms at bar {bar}")
    print(f"  Phases: ref={ref_phase:.4f}, grid={grid_phase:.4f}")
    assert abs(ref_ms - 50.0) < 10.0, "Should find ~50ms offset at downbeat"
    print("  ✓ find_reference_offset_priority works correctly")

    print("\n" + "=" * 80)
    print("All basic tests passed!")
    print("=" * 80)


def test_phase_calculations():
    """Test phase calculation functions."""
    print("=" * 80)
    print("Testing Raster Module - Phase Calculations")
    print("=" * 80)

    # Setup test data
    downbeats = [0.0, 2.0, 4.0, 6.0, 8.0]
    onsets = np.array([0.05, 0.5, 1.0, 1.5, 2.05, 2.5, 3.0, 3.5, 4.05])
    steps_per_bar = 16

    # Test 1: Assign onsets to ticks (uncorrected)
    print("\n[1] Testing assign_onsets_to_ticks_uncorrected...")
    onset_map = assign_onsets_to_ticks_uncorrected(onsets, downbeats, 0, 2, steps_per_bar)
    print(f"  Assigned {len(onset_map)} onsets to ticks")
    assert len(onset_map) > 0, "Should assign some onsets"
    print("  ✓ Onset assignment works")

    # Test 2: Calculate uncorrected phases
    print("\n[2] Testing calculate_phases_uncorrected...")
    df_uncorr = calculate_phases_uncorrected(onset_map, downbeats, steps_per_bar)
    print(f"  Calculated phases for {len(df_uncorr)} onsets")
    print(f"  Sample phases: {df_uncorr['phase_uncorrected'].head(3).tolist()}")
    assert len(df_uncorr) > 0, "Should have phases"
    assert 'phase_uncorrected' in df_uncorr.columns, "Should have phase column"
    print("  ✓ Uncorrected phase calculation works")

    # Test 3: Calculate per-snippet phases (remapped)
    print("\n[3] Testing calculate_phases_per_snippet (remapped)...")
    df_snippet = calculate_phases_per_snippet(onset_map, downbeats, 50.0, steps_per_bar, remap_ticks=True)
    print(f"  Calculated {len(df_snippet)} phases with 50ms offset")
    if len(df_snippet) > 0:
        print(f"  Sample phases: {df_snippet['phase_per_snippet_remapped'].head(3).tolist()}")
    assert 'phase_per_snippet_remapped' in df_snippet.columns, "Should have remapped phase column"
    print("  ✓ Per-snippet phase calculation (remapped) works")

    # Test 4: Calculate per-snippet phases (unmapped - buggy)
    print("\n[4] Testing calculate_phases_per_snippet (unmapped/buggy)...")
    df_snippet_bug = calculate_phases_per_snippet(onset_map, downbeats, 50.0, steps_per_bar, remap_ticks=False)
    print(f"  Calculated {len(df_snippet_bug)} phases (buggy version)")
    assert 'phase_per_snippet' in df_snippet_bug.columns, "Should have unmapped phase column"
    print("  ✓ Per-snippet phase calculation (unmapped) works")

    # Test 5: Calculate loop reference offsets
    print("\n[5] Testing calculate_loop_reference_offsets...")
    loop_offsets = calculate_loop_reference_offsets(downbeats, onsets, 0, 3, 2, steps_per_bar)
    print(f"  Found {len(loop_offsets)} loop offsets")
    for loop_start, ref_ms in loop_offsets.items():
        print(f"    Loop starting at bar {loop_start}: {ref_ms:.2f}ms")
    assert len(loop_offsets) > 0, "Should find some loop offsets"
    print("  ✓ Loop reference offset calculation works")

    # Test 6: Calculate loop-based phases
    print("\n[6] Testing calculate_phases_loop_based...")
    df_loop = calculate_phases_loop_based(onsets, downbeats, 0, 3, 2, steps_per_bar, loop_offsets)
    print(f"  Calculated {len(df_loop)} loop-based phases")
    if len(df_loop) > 0:
        print(f"  Sample phases: {df_loop['phase_loop'].head(3).tolist()}")
    assert 'phase_loop' in df_loop.columns, "Should have loop phase column"
    print("  ✓ Loop-based phase calculation works")

    print("\n" + "=" * 80)
    print("All phase calculation tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_basic_functions()
    print("\n")
    test_phase_calculations()
