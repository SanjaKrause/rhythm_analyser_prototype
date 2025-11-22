"""
File I/O utility functions.

Functions for finding, reading, and writing analysis data files.

Environment: Base (standard library, pandas)
"""

import os
import re
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path


def find_files_by_pattern(
    directory: str,
    pattern: str,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None
) -> Dict[str, str]:
    """
    Find files matching a pattern within an ID range.

    Parameters
    ----------
    directory : str
        Directory to search
    pattern : str
        Regex pattern to match filenames (should contain a capture group for ID)
    start_id : int, optional
        Minimum track ID (inclusive)
    end_id : int, optional
        Maximum track ID (inclusive)

    Returns
    -------
    Dict[str, str]
        Mapping of track_id -> file_path

    Examples
    --------
    >>> files = find_files_by_pattern(
    ...     '/path/to/csvs',
    ...     r'^(\d+)_comprehensive_phases\.csv$',
    ...     start_id=0,
    ...     end_id=10
    ... )
    >>> len(files) > 0
    True
    """
    files = {}

    if not os.path.isdir(directory):
        return files

    pattern_re = re.compile(pattern)

    for filename in os.listdir(directory):
        match = pattern_re.match(filename)
        if match:
            track_id = match.group(1)

            # Check ID range
            if start_id is not None or end_id is not None:
                try:
                    track_id_int = int(track_id)
                    if start_id is not None and track_id_int < start_id:
                        continue
                    if end_id is not None and track_id_int > end_id:
                        continue
                except ValueError:
                    continue

            files[track_id] = os.path.join(directory, filename)

    return files


def load_snippet_offsets(csv_path: str) -> Dict[str, float]:
    """
    Load snippet start times from overview CSV.

    Parameters
    ----------
    csv_path : str
        Path to overview CSV with snippet offsets

    Returns
    -------
    Dict[str, float]
        Mapping of track_id -> snippet_start_time_s

    Examples
    --------
    >>> offsets = load_snippet_offsets('corrected_shift_results.csv')
    >>> '123' in offsets
    True
    """
    if not csv_path or not os.path.isfile(csv_path):
        return {}

    df = pd.read_csv(csv_path, sep=";", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    try:
        id_col = next(c for c in df.columns if "id" in c)
        offset_col = next(c for c in df.columns if "corrected offset" in c and "ms" in c)
    except StopIteration:
        return {}

    offsets = {}
    for _, row in df.iterrows():
        if pd.isnull(row[offset_col]):
            continue
        try:
            tid_str = str(int(row[id_col]))
        except Exception:
            tid_str = str(row[id_col]).strip()

        # Convert from milliseconds to seconds
        offsets[tid_str] = float(row[offset_col]) / 1000.0

    return offsets


def load_pattern_length_map(csv_path: str) -> Dict[str, int]:
    """
    Load pattern length mapping from CSV.

    Parameters
    ----------
    csv_path : str
        Path to pattern length CSV

    Returns
    -------
    Dict[str, int]
        Mapping of track_id -> pattern_length

    Examples
    --------
    >>> lengths = load_pattern_length_map('drum_method_pattern_lengths.csv')
    >>> lengths.get('123', 4)
    4
    """
    if not os.path.isfile(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    return {
        str(row['track_id']): int(row['pattern_length'])
        for _, row in df.iterrows()
    }


def get_pattern_length(
    length_map: Dict[str, int],
    track_id: str,
    default_length: int = 4
) -> int:
    """
    Get pattern length for track with fallback to default.

    Parameters
    ----------
    length_map : Dict[str, int]
        Pattern length mapping from load_pattern_length_map()
    track_id : str
        Track ID to lookup
    default_length : int, default=4
        Default pattern length if not found

    Returns
    -------
    int
        Pattern length for this track

    Examples
    --------
    >>> length_map = {'123': 8, '456': 4}
    >>> get_pattern_length(length_map, '123')
    8
    >>> get_pattern_length(length_map, '999')
    4
    """
    return length_map.get(str(track_id), default_length)


def ensure_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Parameters
    ----------
    directory : str
        Directory path to create

    Examples
    --------
    >>> ensure_directory('/tmp/test_output')
    """
    os.makedirs(directory, exist_ok=True)


def save_dataframe_with_backup(
    df: pd.DataFrame,
    filepath: str,
    backup: bool = True
) -> None:
    """
    Save DataFrame with optional backup of existing file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str
        Output file path
    backup : bool, default=True
        Whether to create .bak file if file exists

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3]})
    >>> save_dataframe_with_backup(df, '/tmp/test.csv')
    """
    if backup and os.path.exists(filepath):
        backup_path = f"{filepath}.bak"
        os.replace(filepath, backup_path)

    df.to_csv(filepath, index=False)
