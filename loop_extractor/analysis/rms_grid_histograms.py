"""
RMS phase deviation histogram analysis.

This module provides functions for calculating RMS phase deviations
and generating comparison histograms across different correction methods.

Environment: Base (numpy, pandas, matplotlib)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def calculate_phase_rms(
    phases: np.ndarray,
    grid_positions_per_bar: int = 16
) -> float:
    """
    Calculate RMS of phase deviations from nearest grid position in phase units.

    Parameters
    ----------
    phases : np.ndarray
        Phase values (0.0 to 1.0) within each bar
    grid_positions_per_bar : int, default=16
        Number of grid positions per bar (16 for sixteenth notes)

    Returns
    -------
    float
        RMS deviation in phase units (0.0 to 1.0)

    Examples
    --------
    >>> phases = np.array([0.0, 0.0625, 0.125])  # Perfect sixteenth notes
    >>> rms = calculate_phase_rms(phases)
    >>> rms < 0.001  # Should be near zero
    True
    """
    if len(phases) == 0:
        return 0.0

    # Create grid positions (0/16, 1/16, 2/16, ..., 15/16)
    grid_positions = np.arange(grid_positions_per_bar) / grid_positions_per_bar

    deviations = []
    for phase in phases:
        # Find nearest grid position
        nearest_idx = np.argmin(np.abs(grid_positions - phase))
        nearest_grid = grid_positions[nearest_idx]

        # Calculate deviation in phase units
        deviation = phase - nearest_grid
        deviations.append(deviation)

    # Calculate RMS
    rms = np.sqrt(np.mean(np.array(deviations)**2))
    return rms


def calculate_phase_rms_ms(
    df: pd.DataFrame,
    phase_column: str,
    grid_time_column: str = 'grid_time_per_snippet',
    grid_positions_per_bar: int = 16
) -> float:
    """
    Calculate RMS of phase deviations from nearest grid position in milliseconds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: bar_number, phase values, and grid timing columns
    phase_column : str
        Name of the phase column to analyze
    grid_time_column : str, default='grid_time_per_snippet'
        Name of the grid time column to use for bar duration calculation
        Options:
        - 'grid_time_per_snippet': for uncorrected and per-snippet methods
        - 'grid_time_drum(L=X)': for drum loop-based method
        - 'grid_time_mel(L=Y)': for melody loop-based method
        - 'grid_time_pitch(L=Z)': for pitch loop-based method
    grid_positions_per_bar : int, default=16
        Number of grid positions per bar (16 for sixteenth notes)

    Returns
    -------
    float
        RMS deviation in milliseconds

    Notes
    -----
    Each method uses its own grid for duration calculation, ensuring
    RMS values reflect the actual grid being evaluated.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'bar_number': [0, 0, 0],
    ...     'tick_16th': [0, 1, 2],
    ...     'phase_uncorrected': [0.0, 0.065, 0.130],
    ...     'grid_time_per_snippet': [0.0, 0.5, 1.0]
    ... })
    >>> rms = calculate_phase_rms_ms(df, 'phase_uncorrected')
    >>> rms > 0
    True
    """
    # Filter to rows with actual onset data
    df_with_onsets = df[df[phase_column].notna()].copy()

    if len(df_with_onsets) == 0:
        return 0.0

    # Create grid positions (0/16, 1/16, 2/16, ..., 15/16)
    grid_positions = np.arange(grid_positions_per_bar) / grid_positions_per_bar

    deviations_ms = []

    for _, row in df_with_onsets.iterrows():
        phase = row[phase_column]
        bar_idx = row['bar_number']

        # Get bar duration using the appropriate grid time column
        bar_df = df[df['bar_number'] == bar_idx]
        if len(bar_df) >= 2 and grid_time_column in df.columns:
            # Calculate bar duration from specified grid time column
            # (difference between last and first tick in bar)
            bar_duration_s = bar_df[grid_time_column].max() - bar_df[grid_time_column].min()
            # Adjust for full bar (we have 16 ticks, but duration is only 15/16 of bar)
            bar_duration_s = bar_duration_s * (grid_positions_per_bar / (grid_positions_per_bar - 1))
        else:
            # Fallback: assume typical bar duration (will be inaccurate but rare)
            bar_duration_s = 2.0  # ~120 BPM in 4/4

        # Find nearest grid position
        nearest_idx = np.argmin(np.abs(grid_positions - phase))
        nearest_grid = grid_positions[nearest_idx]

        # Calculate deviation in phase units
        deviation_phase = phase - nearest_grid

        # Convert to milliseconds using the appropriate grid's bar duration
        deviation_ms = deviation_phase * bar_duration_s * 1000.0
        deviations_ms.append(deviation_ms)

    # Calculate RMS
    rms_ms = np.sqrt(np.mean(np.array(deviations_ms)**2))
    return rms_ms


def calculate_rms_from_csv(
    csv_path: str,
    grid_positions_per_bar: int = 16
) -> Optional[dict]:
    """
    Calculate RMS values for all correction methods from comprehensive CSV.

    Parameters
    ----------
    csv_path : str
        Path to comprehensive phases CSV file
    grid_positions_per_bar : int, default=16
        Number of grid positions per bar

    Returns
    -------
    dict or None
        Dictionary with RMS values in both milliseconds and phase units:
        - uncorrected_ms, per_snippet_ms, drum_ms, mel_ms, pitch_ms
        - uncorrected_phase, per_snippet_phase, drum_phase, mel_phase, pitch_phase
        - pattern_lengths: dict with 'drum', 'mel', 'pitch' pattern lengths

    Examples
    --------
    >>> rms_values = calculate_rms_from_csv('track_123_comprehensive_phases.csv')
    >>> rms_values['uncorrected_ms'] > rms_values['per_snippet_ms']
    True
    """
    try:
        import re
        df = pd.read_csv(csv_path)

        # Extract pattern lengths from column names
        pattern_lengths = {}
        for col in df.columns:
            if 'phase_drum(L=' in col:
                pattern_lengths['drum'] = int(re.search(r'L=(\d+)', col).group(1))
            elif 'phase_mel(L=' in col:
                pattern_lengths['mel'] = int(re.search(r'L=(\d+)', col).group(1))
            elif 'phase_pitch(L=' in col:
                pattern_lengths['pitch'] = int(re.search(r'L=(\d+)', col).group(1))

        # Get column names for each method
        col_drum = [c for c in df.columns if 'phase_drum(L=' in c][0]
        col_mel = [c for c in df.columns if 'phase_mel(L=' in c][0]
        col_pitch = [c for c in df.columns if 'phase_pitch(L=' in c][0]

        # Extract phase values (only rows with actual onsets)
        phases_uncorr = df['phase_uncorrected'].dropna().values
        phases_per_snippet = df['phase_per_snippet'].dropna().values
        phases_drum = df[col_drum].dropna().values
        phases_mel = df[col_mel].dropna().values
        phases_pitch = df[col_pitch].dropna().values

        # Get grid time column names
        grid_col_drum = [c for c in df.columns if 'grid_time_drum(L=' in c][0]
        grid_col_mel = [c for c in df.columns if 'grid_time_mel(L=' in c][0]
        grid_col_pitch = [c for c in df.columns if 'grid_time_pitch(L=' in c][0]

        # Calculate RMS in both units
        rms_values = {
            # Milliseconds - use appropriate grid time column for each method
            'uncorrected_ms': calculate_phase_rms_ms(df, 'phase_uncorrected', 'grid_time_per_snippet'),
            'per_snippet_ms': calculate_phase_rms_ms(df, 'phase_per_snippet', 'grid_time_per_snippet'),
            'drum_ms': calculate_phase_rms_ms(df, col_drum, grid_col_drum),
            'mel_ms': calculate_phase_rms_ms(df, col_mel, grid_col_mel),
            'pitch_ms': calculate_phase_rms_ms(df, col_pitch, grid_col_pitch),
            # Phase units
            'uncorrected_phase': calculate_phase_rms(phases_uncorr),
            'per_snippet_phase': calculate_phase_rms(phases_per_snippet),
            'drum_phase': calculate_phase_rms(phases_drum),
            'mel_phase': calculate_phase_rms(phases_mel),
            'pitch_phase': calculate_phase_rms(phases_pitch),
            'pattern_lengths': pattern_lengths
        }

        return rms_values

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def calculate_improvement_percentage(
    baseline_rms: float,
    corrected_rms: float
) -> float:
    """
    Calculate improvement percentage between two RMS values.

    Parameters
    ----------
    baseline_rms : float
        Baseline RMS value (e.g., uncorrected)
    corrected_rms : float
        Corrected RMS value (e.g., per-snippet)

    Returns
    -------
    float
        Improvement percentage (positive means improvement)

    Examples
    --------
    >>> calculate_improvement_percentage(100.0, 50.0)
    50.0
    >>> calculate_improvement_percentage(50.0, 100.0)
    -100.0
    """
    if baseline_rms == 0:
        return 0.0
    return (baseline_rms - corrected_rms) / baseline_rms * 100.0


def create_rms_summary_dataframe(
    results: dict
) -> pd.DataFrame:
    """
    Create summary DataFrame from RMS analysis results.

    Parameters
    ----------
    results : dict
        Dictionary mapping track_id -> rms_values dict

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with RMS values and improvement statistics

    Examples
    --------
    >>> results = {
    ...     '123': {
    ...         'uncorrected_ms': 60.0,
    ...         'per_snippet_ms': 25.0,
    ...         'drum_ms': 24.0,
    ...         'pattern_lengths': {'drum': 4}
    ...     }
    ... }
    >>> df = create_rms_summary_dataframe(results)
    >>> df.loc[0, 'track_id']
    '123'
    """
    data_rows = []
    for track_id, rms_vals in results.items():
        row = {
            'track_id': track_id,
            'rms_uncorrected_ms': rms_vals['uncorrected_ms'],
            'rms_per_snippet_ms': rms_vals['per_snippet_ms'],
            'rms_drum_ms': rms_vals['drum_ms'],
            'rms_mel_ms': rms_vals['mel_ms'],
            'rms_pitch_ms': rms_vals['pitch_ms'],
            'rms_uncorrected_phase': rms_vals['uncorrected_phase'],
            'rms_per_snippet_phase': rms_vals['per_snippet_phase'],
            'rms_drum_phase': rms_vals['drum_phase'],
            'rms_mel_phase': rms_vals['mel_phase'],
            'rms_pitch_phase': rms_vals['pitch_phase'],
            'pattern_len_drum': rms_vals['pattern_lengths'].get('drum', None),
            'pattern_len_mel': rms_vals['pattern_lengths'].get('mel', None),
            'pattern_len_pitch': rms_vals['pattern_lengths'].get('pitch', None)
        }

        # Calculate improvements
        row['improvement_per_snippet_vs_uncorr'] = calculate_improvement_percentage(
            rms_vals['uncorrected_ms'], rms_vals['per_snippet_ms']
        )
        row['improvement_drum_vs_per_snippet'] = calculate_improvement_percentage(
            rms_vals['per_snippet_ms'], rms_vals['drum_ms']
        )
        row['improvement_mel_vs_per_snippet'] = calculate_improvement_percentage(
            rms_vals['per_snippet_ms'], rms_vals['mel_ms']
        )
        row['improvement_pitch_vs_per_snippet'] = calculate_improvement_percentage(
            rms_vals['per_snippet_ms'], rms_vals['pitch_ms']
        )

        data_rows.append(row)

    df = pd.DataFrame(data_rows)

    # Find best method for each track
    rms_cols = ['rms_uncorrected_ms', 'rms_per_snippet_ms', 'rms_drum_ms', 'rms_mel_ms', 'rms_pitch_ms']
    df['best_method'] = df[rms_cols].idxmin(axis=1)

    return df
