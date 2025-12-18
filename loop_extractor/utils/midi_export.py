"""
MIDI export utilities for converting onset times to MIDI files.

This module creates MIDI files from detected onset times within the snippet window,
containing only full bars. This allows users to:
- Import onset timing into DAWs
- Analyze timing in MIDI-compatible software
- Compare with grid-based timing

Environment: AEinBOX_13_3
Dependencies: mido, numpy, pandas
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import sys

# Import config from parent directory
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

import importlib.util
spec = importlib.util.spec_from_file_location("config_module", _parent_dir / "config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config

try:
    import mido
    from mido import Message, MidiFile, MidiTrack
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido not available. MIDI export will be disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TEMPO = 120  # BPM
DEFAULT_NOTE = 60    # Middle C (C4)
DEFAULT_VELOCITY = 100
DEFAULT_DURATION = 0.05  # seconds (short note, 50ms)


# ============================================================================
# GRID TO MIDI CONVERSION
# ============================================================================

def filter_full_bars(
    onset_times: np.ndarray,
    bar_starts: np.ndarray,
    bar_ends: np.ndarray,
    snippet_start: float,
    snippet_end: float
) -> np.ndarray:
    """
    Filter onset times to only include full bars within snippet window.

    A bar is considered "full" if both its start and end are within
    the snippet window.

    Parameters
    ----------
    onset_times : np.ndarray
        All onset times in seconds
    bar_starts : np.ndarray
        Bar start times in seconds
    bar_ends : np.ndarray
        Bar end times in seconds
    snippet_start : float
        Snippet start time in seconds
    snippet_end : float
        Snippet end time in seconds

    Returns
    -------
    np.ndarray
        Filtered onset times containing only full bars

    Examples
    --------
    >>> onset_times = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    >>> bar_starts = np.array([10.0, 12.0])
    >>> bar_ends = np.array([12.0, 14.0])
    >>> filtered = filter_full_bars(onset_times, bar_starts, bar_ends, 9.0, 13.0)
    >>> # Only bar from 10.0-12.0 is fully within snippet
    """
    # Find bars that are fully within snippet
    full_bar_mask = (bar_starts >= snippet_start) & (bar_ends <= snippet_end)
    full_bar_starts = bar_starts[full_bar_mask]
    full_bar_ends = bar_ends[full_bar_mask]

    if len(full_bar_starts) == 0:
        return np.array([])

    # Filter onset times to only those within full bars
    filtered_times = []
    for bar_start, bar_end in zip(full_bar_starts, full_bar_ends):
        mask = (onset_times >= bar_start) & (onset_times < bar_end)
        filtered_times.append(onset_times[mask])

    if filtered_times:
        return np.concatenate(filtered_times)
    return np.array([])


def onsets_to_midi(
    onset_times: np.ndarray,
    output_path: str,
    tempo: float = DEFAULT_TEMPO,
    note: int = DEFAULT_NOTE,
    velocity: int = DEFAULT_VELOCITY,
    duration: float = DEFAULT_DURATION,
    loop_end_time: float = None
) -> Path:
    """
    Convert onset times to MIDI file.

    Creates a MIDI file with a single track containing note-on/note-off
    messages at each onset time.

    Parameters
    ----------
    onset_times : np.ndarray
        Onset times in seconds (should already be filtered to full bars)
    output_path : str
        Output MIDI file path
    tempo : float
        Tempo in BPM (default: 120)
    note : int
        MIDI note number (default: 60 = middle C)
    velocity : int
        Note velocity (default: 100)
    duration : float
        Note duration in seconds (default: 0.05 = 50ms)
    loop_end_time : float, optional
        If provided, extends MIDI file to this time (in seconds, absolute)

    Returns
    -------
    Path
        Path to created MIDI file

    Examples
    --------
    >>> onset_times = np.array([10.0, 10.5, 11.0, 11.5])
    >>> midi_path = onsets_to_midi(onset_times, 'output.mid')
    >>> # With loop end time to extend file to 4 bars
    >>> midi_path = onsets_to_midi(onset_times, 'output.mid', loop_end_time=14.0)
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido library not available. Install with: pip install mido")

    if len(onset_times) == 0:
        raise ValueError("No onset times provided")

    # Create MIDI file
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    microseconds_per_beat = int(60_000_000 / tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))

    # Calculate ticks per second
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_second = (tempo / 60) * ticks_per_beat

    # Convert duration to ticks
    duration_ticks = int(duration * ticks_per_second)

    # Sort onset times
    onset_times = np.sort(onset_times)

    # Shift times so first note starts at t=0 in MIDI file
    time_offset = onset_times[0]
    onset_times = onset_times - time_offset

    # Calculate loop end tick if provided
    loop_end_tick = None
    if loop_end_time is not None:
        loop_end_relative = loop_end_time - time_offset
        loop_end_tick = int(loop_end_relative * ticks_per_second)

    # Add note events
    last_tick = 0
    for onset_time in onset_times:
        # Calculate absolute tick time for note on
        tick_time = int(onset_time * ticks_per_second)

        # Calculate delta time from last event (must be non-negative)
        delta_time = max(0, tick_time - last_tick)

        # Add note on
        track.append(Message('note_on', note=note, velocity=velocity, time=delta_time))

        # Calculate note duration, truncating if it exceeds loop boundary
        note_duration = duration_ticks
        if loop_end_tick is not None:
            # If note would extend past loop end, truncate it
            if tick_time + duration_ticks > loop_end_tick:
                note_duration = max(0, loop_end_tick - tick_time)

        # Add note off
        track.append(Message('note_off', note=note, velocity=0, time=note_duration))

        # Update last tick to the note_off time (note_on + duration)
        # This prevents overlapping notes in the MIDI file
        last_tick = tick_time + note_duration

    # If loop_end_time provided and last note ended before loop end, add padding
    if loop_end_tick is not None and last_tick < loop_end_tick:
        delta_to_end = loop_end_tick - last_tick
        # Add a silent note (velocity 0) at the loop end
        track.append(Message('note_on', note=note, velocity=0, time=delta_to_end))
        track.append(Message('note_off', note=note, velocity=0, time=0))

    # Save MIDI file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))

    print(f"  ✓ Exported {len(onset_times)} onsets to MIDI: {output_path}")
    return output_path


def grid_times_to_midi(
    grid_times: np.ndarray,
    output_path: str,
    pattern_length: Optional[int] = None,
    num_bars: int = 1
) -> Optional[Path]:
    """
    Convert grid times to MIDI file, optionally limiting to one loop.

    Parameters
    ----------
    grid_times : np.ndarray
        Grid times in seconds
    output_path : str
        Output MIDI file path
    pattern_length : int, optional
        Number of 16th notes to export (if specified, only export first N notes)
    num_bars : int
        Ignored - kept for backwards compatibility (default: 1)

    Returns
    -------
    Path or None
        Path to created MIDI file, or None if no grid times

    Examples
    --------
    >>> grid_times = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    >>> # Export 64 notes (4 bars * 16 notes per bar)
    >>> midi_path = grid_times_to_midi(grid_times, 'output.mid', pattern_length=64)
    """
    if len(grid_times) == 0:
        return None

    # Remove NaN values
    grid_times = grid_times[~np.isnan(grid_times)]

    if len(grid_times) == 0:
        return None

    # If pattern length specified, only take first N 16th notes
    if pattern_length is not None:
        grid_times = grid_times[:pattern_length]

    # Create MIDI file
    try:
        midi_path = onsets_to_midi(grid_times, output_path)
        return midi_path
    except Exception as e:
        print(f"  ✗ Error creating MIDI: {e}")
        return None


def f0_to_midi(
    f0_csv_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    tempo: float = 120.0,
    min_note_duration: float = 0.1
) -> Optional[Path]:
    """
    Convert F0 (pitch) data from CSV to MIDI file.

    Reads bass F0 data from CSV and creates MIDI notes for pitches within
    the specified time range. F0 values are converted to MIDI note numbers,
    and consecutive frames with similar pitch are merged into single notes.

    Parameters
    ----------
    f0_csv_path : str
        Path to bass_f0.csv file with columns: time, f0_hz
    output_path : str
        Output MIDI file path
    start_time : float
        Loop start time in seconds
    end_time : float
        Loop end time in seconds
    tempo : float
        Tempo in BPM (default: 120)
    min_note_duration : float
        Minimum note duration in seconds (default: 0.1)

    Returns
    -------
    Path or None
        Path to created MIDI file, or None if no pitch data

    Examples
    --------
    >>> midi_path = f0_to_midi('1_stems/bass_f0.csv', 'pitch.mid', 10.0, 14.0, tempo=128)
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido library not available. Install with: pip install mido")

    # Load F0 data
    df = pd.read_csv(f0_csv_path)

    # Filter to time range
    mask = (df['time'] >= start_time) & (df['time'] < end_time)
    df_filtered = df[mask].copy()

    if len(df_filtered) == 0:
        return None

    # Convert F0 to MIDI note numbers (skip unvoiced regions where f0 == 0)
    df_filtered = df_filtered[df_filtered['f0_hz'] > 0].copy()

    if len(df_filtered) == 0:
        return None

    # Convert Hz to MIDI note number: note = 12 * log2(f0/440) + 69
    df_filtered['midi_note'] = 12 * np.log2(df_filtered['f0_hz'] / 440.0) + 69
    df_filtered['midi_note'] = df_filtered['midi_note'].round().astype(int)

    # Clip to valid MIDI range (0-127)
    df_filtered['midi_note'] = df_filtered['midi_note'].clip(0, 127)

    # Merge consecutive frames with same pitch into notes
    notes = []
    current_note = None
    current_start = None

    for idx, row in df_filtered.iterrows():
        if current_note is None:
            # Start new note
            current_note = row['midi_note']
            current_start = row['time']
        elif row['midi_note'] != current_note:
            # Note changed - save previous note
            notes.append({
                'note': current_note,
                'start': current_start,
                'end': row['time']
            })
            # Start new note
            current_note = row['midi_note']
            current_start = row['time']

    # Add final note
    if current_note is not None:
        notes.append({
            'note': current_note,
            'start': current_start,
            'end': end_time
        })

    # Filter out notes that are too short
    notes = [n for n in notes if (n['end'] - n['start']) >= min_note_duration]

    if len(notes) == 0:
        return None

    # Create MIDI file
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    microseconds_per_beat = int(60_000_000 / tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))

    # Calculate ticks per second
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_second = (tempo / 60) * ticks_per_beat

    # Normalize times to start at 0
    time_offset = notes[0]['start']
    for note_dict in notes:
        note_dict['start'] -= time_offset
        note_dict['end'] -= time_offset

    # Add note events
    last_tick = 0
    for note_dict in notes:
        note_num = int(note_dict['note'])  # Ensure integer for MIDI message
        start_tick = int(note_dict['start'] * ticks_per_second)
        end_tick = int(note_dict['end'] * ticks_per_second)
        duration_ticks = end_tick - start_tick

        # Note on
        delta_time = max(0, start_tick - last_tick)
        track.append(Message('note_on', note=note_num, velocity=DEFAULT_VELOCITY, time=int(delta_time)))

        # Note off
        track.append(Message('note_off', note=note_num, velocity=0, time=int(duration_ticks)))

        last_tick = end_tick

    # Save MIDI file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))

    print(f"  ✓ Exported {len(notes)} pitch notes to MIDI: {output_path.name}")
    return output_path


def comprehensive_csv_to_midi(
    comprehensive_csv_path: str,
    pattern_lengths: dict,
    output_dir: str,
    snippet_start: float,
    methods: Optional[List[str]] = None
) -> List[Path]:
    """
    Create MIDI files for each correction method from comprehensive CSV.

    Creates one MIDI file per method, containing only the first loop within
    the snippet window. Pattern length L is extracted from CSV column headers
    (e.g., 'grid_time_drum(L=4)') and represents the loop length in BARS.

    Parameters
    ----------
    comprehensive_csv_path : str
        Path to comprehensive phases CSV
    pattern_lengths : dict
        Pattern lengths for each method (e.g., {'drum': 4, 'mel': 4, 'pitch': 8})
        Values represent loop length in BARS
    output_dir : str
        Output directory for MIDI files
    snippet_start : float
        Snippet start time in seconds
    methods : List[str], optional
        Methods to export (default: ['uncorrected', 'per_snippet', 'drum', 'mel', 'pitch'])

    Returns
    -------
    List[Path]
        List of created MIDI file paths

    Examples
    --------
    >>> pattern_lengths = {'drum': 4, 'mel': 4, 'pitch': 8}  # in bars
    >>> midi_files = comprehensive_csv_to_midi(
    ...     'track_comprehensive.csv',
    ...     pattern_lengths,
    ...     'output/midi/',
    ...     snippet_start=10.0
    ... )
    """
    import re

    if methods is None:
        methods = ['uncorrected', 'per_snippet', 'drum', 'mel', 'pitch']

    # Load comprehensive CSV
    df = pd.read_csv(comprehensive_csv_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = []

    for method in methods:
        # Determine grid time column and extract L from column name
        pattern_length_bars = None

        if method == 'uncorrected':
            col_name = 'grid_time_uncorrected'
            # Use drum pattern length for uncorrected (or default 4)
            pattern_length_bars = pattern_lengths.get('drum', 4)
        elif method == 'per_snippet':
            col_name = 'grid_time_per_snippet'
            # Always use 4 bars for per_snippet (no pattern length calculated for this correction)
            pattern_length_bars = 4
        else:
            # Loop-based methods (drum, mel, pitch)
            # Find column matching method and extract L from column name
            matching_cols = [c for c in df.columns if f'grid_time_{method}' in c]
            if not matching_cols:
                print(f"  ⚠️  No grid times found for {method}, skipping")
                continue
            col_name = matching_cols[0]

            # Extract L from column name like "grid_time_drum(L=4)"
            match = re.search(r'L=(\d+)', col_name)
            if match:
                pattern_length_bars = int(match.group(1))
            else:
                # Fallback to pattern_lengths dict
                pattern_length_bars = pattern_lengths.get(method, 4)

        if col_name not in df.columns:
            print(f"  ⚠️  Column {col_name} not found, skipping {method}")
            continue

        # Get grid times (relative to snippet start)
        grid_times = df[col_name].values - snippet_start

        # Calculate number of 16th notes (L bars * 16 16th notes per bar in 4/4)
        num_16th_notes = pattern_length_bars * 16

        # Create MIDI file for first loop only
        output_file = output_dir / f"{method}.mid"
        midi_path = grid_times_to_midi(grid_times, str(output_file), pattern_length=num_16th_notes, num_bars=1)

        if midi_path:
            midi_files.append(midi_path)
            print(f"  ✓ Exported {method} MIDI ({pattern_length_bars} bars = {num_16th_notes} notes): {midi_path.name}")
        else:
            print(f"  ⚠️  No grid times for {method}")

    return midi_files


def comprehensive_csv_to_onset_midi(
    comprehensive_csv_path: str,
    bar_tempos_csv_path: str,
    output_dir: str,
    snippet_start: float,
    methods: list = None
) -> List[Path]:
    """
    Create MIDI files from onset times in comprehensive CSV.

    Extracts onset times from `onset_time_uncorrected` column and creates
    one MIDI file per method (drum, mel, pitch), using the loop length (L)
    from the column headers to determine how many bars to export.

    MIDI tempo is calculated as the average of bar tempos for the loop.

    Parameters
    ----------
    comprehensive_csv_path : str
        Path to comprehensive phases CSV
    bar_tempos_csv_path : str
        Path to bar tempos CSV (from Step 3.5)
    output_dir : str
        Output directory for MIDI files
    snippet_start : float
        Snippet start time in seconds (for time normalization)

    Returns
    -------
    List[Path]
        List of created MIDI file paths

    Examples
    --------
    >>> midi_files = comprehensive_csv_to_onset_midi(
    ...     'track_comprehensive.csv',
    ...     'track_bar_tempos.csv',
    ...     'output/midi/',
    ...     snippet_start=132.0
    ... )
    """
    import re

    # Load comprehensive CSV
    df = pd.read_csv(comprehensive_csv_path)

    # Load bar tempos CSV
    df_tempos = pd.read_csv(bar_tempos_csv_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = []

    # Get onset times (filter out NaN)
    if 'onset_time_uncorrected' not in df.columns:
        print(f"  ⚠️  No onset_time_uncorrected column found")
        return midi_files

    onset_times = pd.to_numeric(df['onset_time_uncorrected'], errors='coerce').dropna().values

    if len(onset_times) == 0:
        print(f"  ⚠️  No onset times found in comprehensive CSV")
        return midi_files

    # Process each method: per_snippet, drum, mel, pitch, standard_L1, standard_L2, standard_L4
    if methods is None:
        methods = ['per_snippet', 'drum', 'mel', 'pitch', 'standard_L1', 'standard_L2', 'standard_L4']

    for method in methods:
        # Determine pattern length
        if method == 'per_snippet':
            # per_snippet always uses 4 bars (no pattern length calculated for this correction)
            pattern_length_bars = 4
        else:
            # Find phase column with L value for loop-based methods
            matching_cols = [c for c in df.columns if f'phase_{method}(L=' in c]
            if not matching_cols:
                print(f"  ⚠️  No phase column found for {method}, skipping")
                continue

            col_name = matching_cols[0]

            # Extract L from column name like "phase_drum(L=4)"
            match = re.search(r'L=(\d+)', col_name)
            if not match:
                print(f"  ⚠️  Could not extract L value from {col_name}, skipping")
                continue

            pattern_length_bars = int(match.group(1))

        # Calculate average tempo for the first L bars (from corrected tempos)
        bar_tempos_corrected = df_tempos['tempo_corrected_bpm'].head(pattern_length_bars).dropna()
        if len(bar_tempos_corrected) > 0:
            avg_tempo = float(bar_tempos_corrected.mean())
        else:
            avg_tempo = 120.0  # Fallback to default

        # Calculate number of 16th notes (L bars * 16 16th notes per bar in 4/4)
        num_16th_notes = pattern_length_bars * 16

        # Get loop boundaries from grid times (for this method)
        grid_col_pattern = f'grid_time_{method}'
        grid_cols = [c for c in df.columns if grid_col_pattern in c]

        if not grid_cols:
            print(f"  ⚠️  No grid_time column found for {method}, skipping")
            continue

        grid_col = grid_cols[0]

        # Check if we have enough grid times
        if len(df) < num_16th_notes + 1:
            print(f"  ⚠️  Not enough data (need {num_16th_notes + 1} rows, have {len(df)})")
            continue

        # Get loop boundaries from grid times
        # Start: first 16th note (index 0)
        # End: first 16th note of the NEXT loop (index num_16th_notes)
        loop_start_time = pd.to_numeric(df[grid_col].iloc[0], errors='coerce')
        loop_end_time = pd.to_numeric(df[grid_col].iloc[num_16th_notes], errors='coerce')

        if pd.isna(loop_start_time) or pd.isna(loop_end_time):
            print(f"  ⚠️  Invalid grid times for {method}")
            continue

        # Get onset times (drum hits) that fall within the loop boundaries
        onset_times_all = pd.to_numeric(df['onset_time_uncorrected'], errors='coerce').dropna().values

        # Filter onsets to only those within the loop boundaries
        onset_times_subset = onset_times_all[
            (onset_times_all >= loop_start_time) &
            (onset_times_all < loop_end_time)
        ]

        if len(onset_times_subset) == 0:
            print(f"  ⚠️  No onsets in loop for {method}")
            continue

        # Create MIDI file with calculated average tempo
        output_file = output_dir / f"{method}.mid"
        try:
            midi_path = onsets_to_midi(
                onset_times_subset,
                str(output_file),
                tempo=avg_tempo,
                loop_end_time=loop_end_time  # Extend MIDI to full loop length
            )
            midi_files.append(midi_path)
            print(f"  ✓ Exported {method} MIDI ({pattern_length_bars} bars, {len(onset_times_subset)} onsets, {avg_tempo:.1f} BPM): {midi_path.name}")
        except Exception as e:
            print(f"  ✗ Error creating {method} MIDI: {e}")

    return midi_files


def comprehensive_csv_to_pitch_midi(
    comprehensive_csv_path: str,
    bar_tempos_csv_path: str,
    f0_csv_path: str,
    output_dir: str,
    snippet_start: float,
    methods: list = None
) -> List[Path]:
    """
    Create bass pitch MIDI files for each correction method.

    Extracts bass F0 data and creates one MIDI file per method (per_snippet,
    drum, mel, pitch), using the loop boundaries from grid times.

    Parameters
    ----------
    comprehensive_csv_path : str
        Path to comprehensive phases CSV
    bar_tempos_csv_path : str
        Path to bar tempos CSV (from Step 3.5)
    f0_csv_path : str
        Path to bass_f0.csv file with columns: time, f0_hz
    output_dir : str
        Output directory for MIDI files
    snippet_start : float
        Snippet start time in seconds (for time normalization)

    Returns
    -------
    List[Path]
        List of created MIDI file paths

    Examples
    --------
    >>> midi_files = comprehensive_csv_to_pitch_midi(
    ...     'track_comprehensive.csv',
    ...     'track_bar_tempos.csv',
    ...     '1_stems/bass_f0.csv',
    ...     'output/midi_pitch/',
    ...     snippet_start=132.0
    ... )
    """
    import re

    # Load comprehensive CSV
    df = pd.read_csv(comprehensive_csv_path)

    # Load bar tempos CSV
    df_tempos = pd.read_csv(bar_tempos_csv_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = []

    # Check if F0 CSV exists
    if not Path(f0_csv_path).exists():
        print(f"  ⚠️  Bass F0 CSV not found: {f0_csv_path}")
        return midi_files

    # Process each method: per_snippet, drum, mel, pitch, standard_L1, standard_L2, standard_L4
    if methods is None:
        methods = ['per_snippet', 'drum', 'mel', 'pitch', 'standard_L1', 'standard_L2', 'standard_L4']

    for method in methods:
        # Determine pattern length
        if method == 'per_snippet':
            # per_snippet always uses 4 bars
            pattern_length_bars = 4
        else:
            # Find phase column with L value for loop-based methods
            matching_cols = [c for c in df.columns if f'phase_{method}(L=' in c]
            if not matching_cols:
                print(f"  ⚠️  No phase column found for {method}, skipping")
                continue

            col_name = matching_cols[0]

            # Extract L from column name like "phase_drum(L=4)"
            match = re.search(r'L=(\d+)', col_name)
            if not match:
                print(f"  ⚠️  Could not extract L value from {col_name}, skipping")
                continue

            pattern_length_bars = int(match.group(1))

        # Calculate average tempo for the first L bars
        bar_tempos_corrected = df_tempos['tempo_corrected_bpm'].head(pattern_length_bars).dropna()
        if len(bar_tempos_corrected) > 0:
            avg_tempo = float(bar_tempos_corrected.mean())
        else:
            avg_tempo = 120.0  # Fallback

        # Calculate number of 16th notes
        num_16th_notes = pattern_length_bars * 16

        # Get loop boundaries from grid times
        grid_col_pattern = f'grid_time_{method}'
        grid_cols = [c for c in df.columns if grid_col_pattern in c]

        if not grid_cols:
            print(f"  ⚠️  No grid_time column found for {method}, skipping")
            continue

        grid_col = grid_cols[0]

        # Check if we have enough grid times
        if len(df) < num_16th_notes + 1:
            print(f"  ⚠️  Not enough data (need {num_16th_notes + 1} rows, have {len(df)})")
            continue

        # Get loop boundaries
        loop_start_time = pd.to_numeric(df[grid_col].iloc[0], errors='coerce')
        loop_end_time = pd.to_numeric(df[grid_col].iloc[num_16th_notes], errors='coerce')

        if pd.isna(loop_start_time) or pd.isna(loop_end_time):
            print(f"  ⚠️  Invalid grid times for {method}")
            continue

        # Create bass pitch MIDI file
        output_file = output_dir / f"{method}_bass.mid"
        try:
            midi_path = f0_to_midi(
                f0_csv_path,
                str(output_file),
                start_time=loop_start_time,
                end_time=loop_end_time,
                tempo=avg_tempo,
                min_note_duration=0.1
            )
            if midi_path:
                midi_files.append(midi_path)
                print(f"  ✓ Exported {method} bass pitch MIDI ({pattern_length_bars} bars, {avg_tempo:.1f} BPM): {midi_path.name}")
            else:
                print(f"  ⚠️  No bass pitch data for {method}")
        except Exception as e:
            print(f"  ✗ Error creating {method} bass pitch MIDI: {e}")

    return midi_files


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_midi_export():
    """Test MIDI export functionality."""
    if not MIDO_AVAILABLE:
        print("mido not available, skipping tests")
        return

    print("=" * 80)
    print("Testing MIDI Export Module")
    print("=" * 80)

    # Test 1: Filter full bars
    print("\n[1] Testing filter_full_bars...")
    onset_times = np.array([9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
    bar_starts = np.array([10.0, 12.0, 14.0])
    bar_ends = np.array([12.0, 14.0, 16.0])
    snippet_start = 9.0
    snippet_end = 13.0

    filtered = filter_full_bars(onset_times, bar_starts, bar_ends, snippet_start, snippet_end)
    print(f"  Original onset times: {onset_times}")
    print(f"  Filtered (full bars only): {filtered}")
    assert len(filtered) == 4, "Should have 4 onsets from one full bar (10-12s)"
    print("  ✓ filter_full_bars works")

    # Test 2: Basic onset to MIDI
    print("\n[2] Testing onsets_to_midi...")
    onset_times = np.array([10.0, 10.5, 11.0, 11.5])

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        temp_path = f.name

    try:
        midi_path = onsets_to_midi(onset_times, temp_path)
        assert midi_path.exists(), "MIDI file should exist"
        print(f"  Created MIDI file: {midi_path}")
        print("  ✓ onsets_to_midi works")
    finally:
        Path(temp_path).unlink(missing_ok=True)

    # Test 3: Grid times to MIDI with pattern length
    print("\n[3] Testing grid_times_to_midi with pattern length...")
    grid_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])  # 9 notes

    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        temp_path = f.name

    try:
        # Only export first 4 notes (one loop)
        midi_path = grid_times_to_midi(grid_times, temp_path, pattern_length=4)
        assert midi_path.exists(), "MIDI file should exist"
        print(f"  Created MIDI file with 4 notes (from 9 grid times)")
        print("  ✓ grid_times_to_midi with pattern length works")
    finally:
        Path(temp_path).unlink(missing_ok=True)

    print("\n" + "=" * 80)
    print("MIDI export tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_midi_export()
