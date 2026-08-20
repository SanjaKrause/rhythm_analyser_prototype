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


# ============================================================================
# ANCHORED PATTERN MIDI EXPORT
# ============================================================================

# GM drum map for the classifier's 6 output classes (see classify_anchored_drums.LABELS)
GM_DRUM_MAP = {
    'kick_drum': 36,  # Bass Drum 1
    'snare': 38,      # Acoustic Snare
    'hihat_c': 42,    # Closed Hi-Hat
    'ride': 51,       # Ride Cymbal 1
    'crash': 49,      # Crash Cymbal 1
    'tom_h': 50,      # High Tom
}
GM_FALLBACK_NOTE = 37  # Side Stick, used for onsets without a classification

import re as _re
_ANCHORED_CSV_RE = _re.compile(r'^(SecNo\d+)_L(\d+)_(.+)_(\d+\.\d+)_anchored\.csv$')


def _write_events_midi(
    events: List[Tuple[float, int, int]],
    start_time: float,
    end_time: float,
    tempo: float,
    output_path: Path,
    duration: float = DEFAULT_DURATION
) -> Path:
    """
    Write a list of (time_sec, note, velocity) events to a MIDI file.

    Times are absolute seconds; notes are placed relative to start_time so the
    MIDI file starts exactly at the pattern/section boundary (unlike
    onsets_to_midi, which shifts to the first onset). The file is padded to
    end_time so the loop length is preserved.
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido library not available. Install with: pip install mido")
    if not events:
        raise ValueError("No events provided")

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=int(60_000_000 / tempo)))

    ticks_per_second = (tempo / 60) * mid.ticks_per_beat
    duration_ticks = max(1, int(duration * ticks_per_second))
    end_tick = int((end_time - start_time) * ticks_per_second)

    # Build absolute-tick on/off events, then convert to delta times.
    # Sorting note_off before note_on at equal ticks avoids stuck notes.
    abs_events = []
    for time_sec, note, velocity in events:
        on_tick = max(0, int((time_sec - start_time) * ticks_per_second))
        off_tick = min(on_tick + duration_ticks, end_tick) if end_tick > on_tick else on_tick + duration_ticks
        abs_events.append((on_tick, 1, Message('note_on', note=note, velocity=velocity, channel=9, time=0)))
        abs_events.append((off_tick, 0, Message('note_off', note=note, velocity=0, channel=9, time=0)))
    abs_events.sort(key=lambda e: (e[0], e[1]))

    last_tick = 0
    for tick, _, msg in abs_events:
        msg.time = tick - last_tick
        track.append(msg)
        last_tick = tick

    # Pad with a silent marker note so the file spans the full pattern/section
    if last_tick < end_tick:
        track.append(Message('note_on', note=0, velocity=0, channel=9, time=end_tick - last_tick))
        track.append(Message('note_off', note=0, velocity=0, channel=9, time=0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))
    return output_path


def _load_anchored_csv(csv_path: Path) -> pd.DataFrame:
    """Load an anchored CSV (6.1/6.2), keeping only rows with a detected onset."""
    df = pd.read_csv(csv_path, comment='#')
    return df.dropna(subset=['onset_time'])


def _build_class_lookup(filtered_dir: Path, sec_prefix: str) -> dict:
    """
    Map onset_time (rounded to ms) -> predicted_class for one section, using all
    of the section's filtered anchored CSVs (L1/L2/L4). Timing is NOT taken from
    here — this is only a note-number lookup for the GM variant.
    """
    lookup = {}
    for csv_path in sorted(filtered_dir.glob(f'{sec_prefix}_L*_anchored.csv')):
        try:
            df = pd.read_csv(csv_path, comment='#')
        except Exception:
            continue
        if 'predicted_class' not in df.columns:
            continue
        df = df.dropna(subset=['onset_time', 'predicted_class'])
        for t, cls in zip(df['onset_time'], df['predicted_class']):
            lookup[round(float(t), 3)] = cls
    return lookup


def _events_from_onsets(onset_times, class_lookup: Optional[dict], gm: bool) -> List[Tuple[float, int, int]]:
    """Build (time, note, velocity) events; single-note or GM-mapped."""
    events = []
    for t in onset_times:
        if gm:
            cls = class_lookup.get(round(float(t), 3)) if class_lookup else None
            note = GM_DRUM_MAP.get(cls, GM_FALLBACK_NOTE)
        else:
            note = DEFAULT_NOTE
        events.append((float(t), note, DEFAULT_VELOCITY))
    return events


def _iter_anchored_sections(anchoring_dir: Path, filtered_dir: Path):
    """
    Yield one dict per section crossing the snippet, with the two export windows:
    - 'L2': first kept pattern of the FILTERED L2 CSV (= first 2 bars of section)
    - 'full': all bars of the UNFILTERED L1 CSV (no Tukey gaps)
    Each window is (df_onsets, start_time, end_time, tempo); 'full' may be None.
    """
    for l2_path in sorted(filtered_dir.glob('SecNo*_L2_*_anchored.csv')):
        m = _ANCHORED_CSV_RE.match(l2_path.name)
        if not m:
            continue
        sec_prefix, _, label, ratio = m.group(1), m.group(2), m.group(3), m.group(4)
        section = {'sec_prefix': sec_prefix, 'label': label, 'ratio': ratio,
                   'base': f'{sec_prefix}_{{kind}}_{label}_{ratio}', 'L2': None, 'full': None}

        df_l2 = _load_anchored_csv(l2_path)
        if len(df_l2) > 0:
            dfp = df_l2[df_l2['pattern_index'] == df_l2['pattern_index'].min()]
            section['L2'] = (dfp, float(dfp['pattern_start_time'].iloc[0]),
                             float(dfp['pattern_end_time'].iloc[0]),
                             float(dfp['local_tempo'].iloc[0]))

        l1_candidates = sorted(anchoring_dir.glob(f'{sec_prefix}_L1_*_anchored.csv'))
        if l1_candidates:
            df_l1 = _load_anchored_csv(l1_candidates[0])
            if len(df_l1) > 0:
                section['full'] = (df_l1, float(df_l1['pattern_start_time'].min()),
                                   float(df_l1['pattern_end_time'].max()),
                                   float(df_l1['local_tempo'].mean()))
        yield section


def export_anchored_onset_midi(
    anchoring_dir: str,
    filtered_dir: str,
    output_dir: str,
    verbose: bool = True
) -> List[Path]:
    """
    Export anchored drum-onset MIDI files, replacing the legacy flexStart export.

    For every section crossing the snippet (SecNoX files in 6.1/6.2):
    - L2 pattern (first 2 bars of the section) from the FILTERED L2 anchored CSV
      (6.2_filtered_patterns): first kept pattern only.
    - Full section from the UNFILTERED L1 anchored CSV (6.1_anchoring): all bars.

    Each is written twice: single-note (C4, legacy convention) and GM drum-mapped
    ('_gm' suffix) using predicted_class from the filtered CSVs. Onset timing
    always comes from the pipeline's own onset detection (onset_time column) —
    the drum classifier only selects the note number in the GM variant.

    Output naming: SecNoX_L2_<label>_<ratio>.mid, SecNoX_full_<label>_<ratio>.mid
    (+ _gm variants), matching the section file naming used in 6.x folders.

    Returns list of created MIDI file paths.
    """
    anchoring_dir = Path(anchoring_dir)
    filtered_dir = Path(filtered_dir)
    output_dir = Path(output_dir)
    created: List[Path] = []

    if not filtered_dir.exists():
        print(f"  ⚠️  Filtered patterns dir not found: {filtered_dir}")
        return created

    any_sections = False
    for section in _iter_anchored_sections(anchoring_dir, filtered_dir):
        any_sections = True
        sec_prefix = section['sec_prefix']
        class_lookup = _build_class_lookup(filtered_dir, sec_prefix)

        for kind in ('L2', 'full'):
            if section[kind] is None:
                print(f"  ⚠️  No {kind} data for {sec_prefix}")
                continue
            df, start, end, tempo = section[kind]
            try:
                for gm in (False, True):
                    events = _events_from_onsets(df['onset_time'], class_lookup, gm)
                    suffix = '_gm' if gm else ''
                    out = output_dir / f"{section['base'].format(kind=kind)}{suffix}.mid"
                    created.append(_write_events_midi(events, start, end, tempo, out))
                if verbose:
                    if kind == 'full':
                        n_unmatched = sum(
                            1 for t in df['onset_time']
                            if round(float(t), 3) not in class_lookup
                        )
                        extra = f", {n_unmatched} unclassified→side-stick" if n_unmatched else ""
                    else:
                        extra = ""
                    desc = 'L2 pattern' if kind == 'L2' else 'full section'
                    print(f"  ✓ {sec_prefix} {desc}: {len(df)} onsets, {tempo:.1f} BPM{extra}")
            except Exception as e:
                print(f"  ✗ Error exporting {kind} MIDI for {sec_prefix}: {e}")

    if not any_sections:
        print(f"  ⚠️  No filtered L2 anchored CSVs found in {filtered_dir}")
    return created


def _f0_to_note_events(df_f0: pd.DataFrame, start_time: float, end_time: float,
                       min_note_duration: float = 0.1) -> List[dict]:
    """
    Convert F0 frames within [start_time, end_time) to note dicts
    ({'note','start','end'}, absolute seconds). Consecutive frames with the
    same rounded MIDI note are merged; unvoiced (f0==0) frames end notes.
    """
    df = df_f0[(df_f0['time'] >= start_time) & (df_f0['time'] < end_time)].copy()
    df = df[df['f0_hz'] > 0]
    if len(df) == 0:
        return []

    midi_notes = (12 * np.log2(df['f0_hz'] / 440.0) + 69).round().astype(int).clip(0, 127)

    notes = []
    current_note, current_start = None, None
    for t, n in zip(df['time'], midi_notes):
        if current_note is None:
            current_note, current_start = n, t
        elif n != current_note:
            notes.append({'note': int(current_note), 'start': float(current_start), 'end': float(t)})
            current_note, current_start = n, t
    if current_note is not None:
        notes.append({'note': int(current_note), 'start': float(current_start), 'end': float(end_time)})

    return [n for n in notes if (n['end'] - n['start']) >= min_note_duration]


def _write_pitch_midi(notes: List[dict], start_time: float, end_time: float,
                      tempo: float, output_path: Path) -> Path:
    """
    Write pitch note dicts to MIDI, aligned to start_time (not the first note,
    unlike f0_to_midi) and padded to end_time, so files line up with the
    anchored drum MIDI of the same window. Channel 0 (melodic).
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido library not available. Install with: pip install mido")
    if not notes:
        raise ValueError("No notes provided")

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=int(60_000_000 / tempo)))
    ticks_per_second = (tempo / 60) * mid.ticks_per_beat
    end_tick = int((end_time - start_time) * ticks_per_second)

    last_tick = 0
    for n in notes:
        on_tick = max(0, int((n['start'] - start_time) * ticks_per_second))
        off_tick = min(int((n['end'] - start_time) * ticks_per_second), end_tick)
        track.append(Message('note_on', note=n['note'], velocity=DEFAULT_VELOCITY,
                             time=max(0, on_tick - last_tick)))
        track.append(Message('note_off', note=n['note'], velocity=0,
                             time=max(0, off_tick - on_tick)))
        last_tick = off_tick

    if last_tick < end_tick:
        track.append(Message('note_on', note=0, velocity=0, time=end_tick - last_tick))
        track.append(Message('note_off', note=0, velocity=0, time=0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))
    return output_path


def _ensure_f0_coverage(
    f0_csv_path: Path,
    need_start: float,
    need_end: float,
    verbose: bool = True,
    tolerance: float = 1.0
) -> Optional[pd.DataFrame]:
    """
    Return an F0 DataFrame covering [need_start, need_end].

    bass_f0.csv only spans the snippet window (it is written by step 5.5 pattern
    detection), but anchored section windows can start before the snippet. If
    the CSV doesn't cover the needed range, F0 is re-extracted from bass.wav
    (Melodia, cheap) for the full range and cached as bass_f0_sections.csv next
    to it. The original bass_f0.csv is left untouched — apply_drum_anchoring
    depends on its snippet-only content.
    """
    def _covers(df):
        return (df['time'].min() <= need_start + tolerance
                and df['time'].max() >= need_end - tolerance)

    if f0_csv_path.exists():
        df = pd.read_csv(f0_csv_path)
        if _covers(df):
            return df
    else:
        df = None

    # Check cache from a previous run
    cache_path = f0_csv_path.parent / 'bass_f0_sections.csv'
    if cache_path.exists():
        df_cache = pd.read_csv(cache_path)
        if _covers(df_cache):
            return df_cache

    # Extract F0 over the full needed range from bass.wav
    bass_wav = f0_csv_path.parent / 'bass.wav'
    if not bass_wav.exists():
        if verbose:
            print(f"  ⚠️  bass.wav not found, cannot extend F0 coverage — using existing CSV")
        return df

    if verbose:
        print(f"  Extracting bass F0 for anchored windows "
              f"({need_start:.1f}-{need_end:.1f}s, beyond snippet-only CSV)...")
    from analysis.pattern_detection import load_and_extract_f0
    f0, times = load_and_extract_f0(str(bass_wav), (need_start, need_end))
    df_ext = pd.DataFrame({'time': times, 'f0_hz': f0})
    df_ext.to_csv(cache_path, index=False)
    if verbose:
        print(f"  ✓ Cached extended F0: {cache_path.name}")
    return df_ext


def export_anchored_pitch_midi(
    anchoring_dir: str,
    filtered_dir: str,
    f0_csv_path: str,
    output_dir: str,
    min_note_duration: float = 0.1,
    verbose: bool = True
) -> List[Path]:
    """
    Export bass-pitch MIDI for the same anchored windows as the drum export:
    per section, L2 pattern (first 2 bars) and full section. The windows come
    from the drums anchoring (which defines the section grid); the notes come
    from bass F0. If the snippet-only bass_f0.csv doesn't cover the anchored
    windows, F0 is re-extracted from bass.wav (cached as bass_f0_sections.csv).

    Output naming: SecNoX_L2_<label>_<ratio>_bass.mid,
                   SecNoX_full_<label>_<ratio>_bass.mid
    """
    anchoring_dir = Path(anchoring_dir)
    filtered_dir = Path(filtered_dir)
    output_dir = Path(output_dir)
    f0_csv_path = Path(f0_csv_path)
    created: List[Path] = []

    if not filtered_dir.exists():
        print(f"  ⚠️  Filtered patterns dir not found: {filtered_dir}")
        return created

    sections = list(_iter_anchored_sections(anchoring_dir, filtered_dir))
    windows = [section[kind] for section in sections for kind in ('L2', 'full')
               if section[kind] is not None]
    if not windows:
        print(f"  ⚠️  No anchored section windows found in {filtered_dir}")
        return created

    df_f0 = _ensure_f0_coverage(
        f0_csv_path,
        need_start=min(w[1] for w in windows),
        need_end=max(w[2] for w in windows),
        verbose=verbose
    )
    if df_f0 is None:
        print(f"  ⚠️  No bass F0 data available: {f0_csv_path}")
        return created

    for section in sections:
        sec_prefix = section['sec_prefix']
        for kind in ('L2', 'full'):
            if section[kind] is None:
                continue
            _, start, end, tempo = section[kind]
            try:
                notes = _f0_to_note_events(df_f0, start, end, min_note_duration)
                if not notes:
                    if verbose:
                        # F0 is only extracted for the snippet window; anchored
                        # windows can start before it (sections crossing the snippet)
                        if end <= df_f0['time'].min() or start >= df_f0['time'].max():
                            print(f"  ⚠️  {sec_prefix} {kind}: window outside bass F0 coverage "
                                  f"(F0 spans snippet only: {df_f0['time'].min():.1f}-{df_f0['time'].max():.1f}s)")
                        else:
                            print(f"  ⚠️  {sec_prefix} {kind}: no voiced bass F0 in window")
                    continue
                out = output_dir / f"{section['base'].format(kind=kind)}_bass.mid"
                created.append(_write_pitch_midi(notes, start, end, tempo, out))
                if verbose:
                    desc = 'L2 pattern' if kind == 'L2' else 'full section'
                    print(f"  ✓ {sec_prefix} {desc} bass: {len(notes)} notes, {tempo:.1f} BPM")
            except Exception as e:
                print(f"  ✗ Error exporting {kind} bass MIDI for {sec_prefix}: {e}")

    return created


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
