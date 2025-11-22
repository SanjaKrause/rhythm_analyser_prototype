"""
Data processing utility functions.

Common data transformation and validation functions.

Environment: Base (numpy, pandas)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def filter_by_id_range(
    track_id: str,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None
) -> bool:
    """
    Check if track ID is within specified range.

    Parameters
    ----------
    track_id : str
        Track ID to check
    start_id : int, optional
        Minimum ID (inclusive)
    end_id : int, optional
        Maximum ID (inclusive)

    Returns
    -------
    bool
        True if ID is in range or no range specified

    Examples
    --------
    >>> filter_by_id_range('123', start_id=100, end_id=200)
    True
    >>> filter_by_id_range('99', start_id=100, end_id=200)
    False
    """
    if start_id is None and end_id is None:
        return True

    try:
        tid = int(track_id)
    except ValueError:
        return False

    if start_id is not None and tid < start_id:
        return False
    if end_id is not None and tid > end_id:
        return False

    return True


def validate_downbeats(
    downbeats: List[float],
    min_length: int = 2,
    check_monotonic: bool = True
) -> Tuple[bool, str]:
    """
    Validate downbeat timing list.

    Parameters
    ----------
    downbeats : List[float]
        List of downbeat times in seconds
    min_length : int, default=2
        Minimum number of downbeats required
    check_monotonic : bool, default=True
        Whether to check if times are monotonically increasing

    Returns
    -------
    valid : bool
        True if valid
    message : str
        Error message if invalid, empty string if valid

    Examples
    --------
    >>> validate_downbeats([0.0, 2.0, 4.0])
    (True, '')
    >>> validate_downbeats([0.0])
    (False, 'Not enough downbeats: 1 < 2')
    >>> validate_downbeats([0.0, 2.0, 1.0])
    (False, 'Downbeats are not monotonically increasing')
    """
    if len(downbeats) < min_length:
        return False, f"Not enough downbeats: {len(downbeats)} < {min_length}"

    if check_monotonic:
        for i in range(1, len(downbeats)):
            if downbeats[i] <= downbeats[i-1]:
                return False, "Downbeats are not monotonically increasing"

    return True, ""


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    Safe division with default value for division by zero.

    Parameters
    ----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float, default=0.0
        Value to return if denominator is zero

    Returns
    -------
    float
        numerator / denominator, or default if denominator is zero

    Examples
    --------
    >>> safe_divide(10.0, 2.0)
    5.0
    >>> safe_divide(10.0, 0.0, default=np.nan)
    nan
    """
    if denominator == 0:
        return default
    return numerator / denominator


def normalize_phases(phases: np.ndarray) -> np.ndarray:
    """
    Normalize phase values to [0, 1) range.

    Parameters
    ----------
    phases : np.ndarray
        Phase values

    Returns
    -------
    np.ndarray
        Normalized phases in [0, 1) range

    Examples
    --------
    >>> normalize_phases(np.array([0.5, 1.5, 2.5]))
    array([0.5, 0.5, 0.5])
    """
    return np.mod(phases, 1.0)


def remove_outliers(
    data: np.ndarray,
    n_std: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers beyond n standard deviations from mean.

    Parameters
    ----------
    data : np.ndarray
        Input data
    n_std : float, default=3.0
        Number of standard deviations for outlier threshold

    Returns
    -------
    clean_data : np.ndarray
        Data with outliers removed
    mask : np.ndarray
        Boolean mask of kept values

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 100.0])
    >>> clean, mask = remove_outliers(data, n_std=2.0)
    >>> len(clean) < len(data)
    True
    """
    mean = np.mean(data)
    std = np.std(data)
    threshold = n_std * std

    mask = np.abs(data - mean) <= threshold
    return data[mask], mask


def interpolate_missing_beats(
    beat_times: List[float],
    expected_ioi: float,
    max_gap_ratio: float = 2.0
) -> List[float]:
    """
    Interpolate missing beats based on expected inter-onset interval.

    Parameters
    ----------
    beat_times : List[float]
        Detected beat times
    expected_ioi : float
        Expected inter-onset interval (seconds)
    max_gap_ratio : float, default=2.0
        Maximum gap (as multiple of expected IOI) to interpolate

    Returns
    -------
    List[float]
        Beat times with interpolated missing beats

    Examples
    --------
    >>> beats = [0.0, 0.5, 1.5, 2.0]  # Missing beat at 1.0
    >>> filled = interpolate_missing_beats(beats, expected_ioi=0.5)
    >>> len(filled) > len(beats)
    True
    """
    if len(beat_times) < 2:
        return beat_times

    filled_beats = [beat_times[0]]

    for i in range(1, len(beat_times)):
        prev_beat = beat_times[i-1]
        curr_beat = beat_times[i]
        gap = curr_beat - prev_beat

        # Check if gap is larger than expected
        if gap > expected_ioi * max_gap_ratio:
            # Interpolate missing beats
            n_missing = int(round(gap / expected_ioi)) - 1
            for j in range(1, n_missing + 1):
                interpolated_beat = prev_beat + j * expected_ioi
                filled_beats.append(interpolated_beat)

        filled_beats.append(curr_beat)

    return filled_beats
