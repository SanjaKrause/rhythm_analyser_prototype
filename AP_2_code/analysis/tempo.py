"""
Tempo calculation and analysis functions.

This module provides functions for calculating different tempo references:
- Global tempo: Average BPM across entire track
- Snippet tempo: Average BPM within snippet window
- Loop tempo: Average BPM for each loop cycle

Environment: Base (numpy, pandas)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def calculate_bar_tempos(
    downbeats: List[float],
    time_sig: int = 4
) -> pd.DataFrame:
    """
    Calculate tempo for each bar from downbeat times.

    Parameters
    ----------
    downbeats : List[float]
        List of downbeat times in seconds
    time_sig : int, default=4
        Time signature (beats per bar)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - bar_index: Bar number (0-indexed)
        - start_time: Start time of bar (s)
        - end_time: End time of bar (s)
        - duration_s: Bar duration (s)
        - tempo_bpm: Tempo in BPM

    Examples
    --------
    >>> downbeats = [0.0, 2.0, 4.0, 6.0]  # 120 BPM in 4/4
    >>> df = calculate_bar_tempos(downbeats, time_sig=4)
    >>> df['tempo_bpm'].mean()
    120.0
    """
    bars = []
    for i in range(len(downbeats) - 1):
        start = downbeats[i]
        end = downbeats[i + 1]
        duration = end - start

        if duration > 0:
            # tempo_bpm = (60 seconds/minute) / (bar_duration) * (beats/bar)
            tempo_bpm = 60.0 / duration * time_sig
            bars.append({
                'bar_index': i,
                'start_time': start,
                'end_time': end,
                'duration_s': duration,
                'tempo_bpm': tempo_bpm
            })

    return pd.DataFrame(bars)


def calculate_global_tempo(df_bars: pd.DataFrame) -> float:
    """
    Calculate average tempo across entire track.

    Parameters
    ----------
    df_bars : pd.DataFrame
        DataFrame with 'tempo_bpm' column from calculate_bar_tempos()

    Returns
    -------
    float
        Average tempo in BPM, or np.nan if no bars

    Examples
    --------
    >>> df_bars = pd.DataFrame({'tempo_bpm': [120.0, 120.5, 119.5]})
    >>> calculate_global_tempo(df_bars)
    120.0
    """
    if df_bars.empty:
        return np.nan
    return float(df_bars['tempo_bpm'].mean())


def calculate_snippet_tempo(
    df_bars: pd.DataFrame,
    snippet_start: float,
    snippet_end: float
) -> float:
    """
    Calculate average tempo within snippet window.

    Only includes bars that are fully or partially within the snippet.

    Parameters
    ----------
    df_bars : pd.DataFrame
        DataFrame with bar timing from calculate_bar_tempos()
    snippet_start : float
        Snippet start time (s)
    snippet_end : float
        Snippet end time (s)

    Returns
    -------
    float
        Average tempo in snippet (BPM), or np.nan if no bars in range

    Examples
    --------
    >>> df_bars = pd.DataFrame({
    ...     'start_time': [0.0, 2.0, 4.0],
    ...     'end_time': [2.0, 4.0, 6.0],
    ...     'tempo_bpm': [120.0, 120.0, 120.0]
    ... })
    >>> calculate_snippet_tempo(df_bars, 1.0, 5.0)
    120.0
    """
    if df_bars.empty or not (np.isfinite(snippet_start) and np.isfinite(snippet_end)):
        return np.nan

    # Get bars that overlap with snippet
    snippet_bars = df_bars[
        (df_bars['start_time'] < snippet_end) &
        (df_bars['end_time'] > snippet_start)
    ]

    if snippet_bars.empty:
        return np.nan

    return float(snippet_bars['tempo_bpm'].mean())


def calculate_loop_tempos(
    df_bars: pd.DataFrame,
    pattern_len: int
) -> List[Dict]:
    """
    Calculate average tempo for each loop cycle.

    Parameters
    ----------
    df_bars : pd.DataFrame
        DataFrame with bar timing from calculate_bar_tempos()
    pattern_len : int
        Pattern length in bars (e.g., 4 for 4-bar loops)

    Returns
    -------
    List[Dict]
        List of dicts with keys:
        - loop_index: Loop number (0-indexed)
        - start_bar: First bar in loop
        - end_bar: Last bar in loop (inclusive)
        - n_bars: Number of bars in loop
        - avg_tempo_bpm: Average tempo for this loop

    Examples
    --------
    >>> df_bars = pd.DataFrame({
    ...     'tempo_bpm': [120.0, 120.0, 120.0, 120.0, 121.0, 121.0]
    ... })
    >>> loops = calculate_loop_tempos(df_bars, pattern_len=4)
    >>> len(loops)
    2
    >>> loops[0]['avg_tempo_bpm']
    120.0
    """
    if df_bars.empty or pattern_len <= 0:
        return []

    loops = []
    n_bars = len(df_bars)
    n_loops = (n_bars + pattern_len - 1) // pattern_len

    for loop_idx in range(n_loops):
        start_bar = loop_idx * pattern_len
        end_bar = min((loop_idx + 1) * pattern_len, n_bars)
        loop_bars = df_bars.iloc[start_bar:end_bar]

        if not loop_bars.empty:
            loops.append({
                'loop_index': loop_idx,
                'start_bar': start_bar,
                'end_bar': end_bar - 1,  # Inclusive end
                'n_bars': len(loop_bars),
                'avg_tempo_bpm': float(loop_bars['tempo_bpm'].mean())
            })

    return loops


def parse_corrected_downbeats(filepath: str) -> Tuple[List[float], int]:
    """
    Parse corrected downbeat file from beat transformer output.

    Parameters
    ----------
    filepath : str
        Path to corrected downbeat file

    Returns
    -------
    downbeats : List[float]
        List of downbeat times in seconds
    time_sig : int
        Time signature (default 4)

    Examples
    --------
    >>> downbeats, time_sig = parse_corrected_downbeats('track_123_downbeats_corrected.txt')
    >>> len(downbeats) > 0
    True
    """
    downbeats = []
    time_sig = 4

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                # Parse time signature from header
                if line.startswith("# time_signature="):
                    time_sig = int(line.split("=")[1])

                # Skip header lines and empty lines
                elif not line.startswith("#") and line and "corrected_downbeat_time" not in line:
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        try:
                            # corrected_downbeat_time(s) is usually column 2
                            db_time = float(parts[2])
                            downbeats.append(db_time)
                        except ValueError:
                            continue
    except FileNotFoundError:
        pass

    return downbeats, time_sig


def analyze_track_tempo(
    downbeat_file: str,
    snippet_start: Optional[float] = None,
    snippet_duration: float = 30.0,
    pattern_lengths: Optional[Dict[str, int]] = None
) -> Dict:
    """
    Comprehensive tempo analysis for a single track.

    Parameters
    ----------
    downbeat_file : str
        Path to corrected downbeat file
    snippet_start : float, optional
        Snippet start time in seconds (None to skip snippet tempo)
    snippet_duration : float, default=30.0
        Snippet duration in seconds
    pattern_lengths : Dict[str, int], optional
        Pattern lengths for different methods, e.g.,
        {'drum': 4, 'mel': 8, 'pitch': 4}

    Returns
    -------
    Dict
        Analysis results with keys:
        - time_signature: Time signature
        - n_bars: Number of bars
        - global_tempo_bpm: Average tempo across track
        - snippet_tempo_bpm: Average tempo in snippet (if snippet_start provided)
        - loop_tempos_{method}: Loop tempo dict for each pattern length

    Examples
    --------
    >>> result = analyze_track_tempo(
    ...     'track_123_downbeats_corrected.txt',
    ...     snippet_start=10.0,
    ...     pattern_lengths={'drum': 4, 'mel': 8}
    ... )
    >>> result['global_tempo_bpm']
    120.5
    """
    # Parse downbeats
    downbeats, time_sig = parse_corrected_downbeats(downbeat_file)

    if len(downbeats) < 2:
        return {'error': 'Not enough downbeats'}

    # Calculate bar tempos
    df_bars = calculate_bar_tempos(downbeats, time_sig)

    if df_bars.empty:
        return {'error': 'No valid bars'}

    # Build result
    result = {
        'time_signature': time_sig,
        'n_bars': len(df_bars),
        'global_tempo_bpm': calculate_global_tempo(df_bars)
    }

    # Snippet tempo
    if snippet_start is not None:
        snippet_end = snippet_start + snippet_duration
        result['snippet_start_s'] = snippet_start
        result['snippet_end_s'] = snippet_end
        result['snippet_tempo_bpm'] = calculate_snippet_tempo(
            df_bars, snippet_start, snippet_end
        )

    # Loop tempos
    if pattern_lengths:
        for method, pattern_len in pattern_lengths.items():
            loop_tempos = calculate_loop_tempos(df_bars, pattern_len)
            result[f'loop_tempos_{method}'] = loop_tempos
            result[f'pattern_len_{method}'] = pattern_len

            if loop_tempos:
                avg_loop_tempo = np.mean([l['avg_tempo_bpm'] for l in loop_tempos])
                result[f'avg_loop_tempo_{method}_bpm'] = avg_loop_tempo

    return result
