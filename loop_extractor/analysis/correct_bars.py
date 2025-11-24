"""
Automatic downbeat correction module.

This module corrects double-time and half-time detection errors in beat tracking
by analyzing tempo distributions and applying merge/split operations to bars.

Environment: AEinBOX_13_3 (numpy, pandas)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Import config for parameters
import importlib.util
_config_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("config_module", _config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

OUTLIER_PERCENT = config.CORRECT_BARS_OUTLIER_PERCENT
MULT_MATCH_TOL = config.CORRECT_BARS_MULT_MATCH_TOL
BPM_CHARTS_THRESHOLD = config.CORRECT_BARS_BPM_THRESHOLD
DOUBLE_BPM_CHART_ADJUSTED = config.CORRECT_BARS_DOUBLE_BPM_ADJUSTED
SNIPPET_DURATION_S = config.CORRECT_BARS_SNIPPET_DURATION_S
EPS = 1e-6


# ============================================================================
# INPUT PARSING FUNCTIONS
# ============================================================================

def parse_downbeats_direct_strict(beat_path: str) -> List[float]:
    """
    Extract unique downbeat times from BeatTransformer output (STRICT mode).

    Only includes downbeats from complete bars (bar_num >= 1 AND beat_pos == 1).
    Skips incomplete bars (bar_num=0).

    Parameters
    ----------
    beat_path : str
        Path to BeatTransformer output file

    Returns
    -------
    List[float]
        Sorted list of unique downbeat times in seconds

    Examples
    --------
    >>> downbeats = parse_downbeats_direct_strict('track_output.txt')
    >>> len(downbeats)
    83
    """
    df = pd.read_csv(beat_path, sep='\t')

    # Check for required columns
    if 'bar_num' not in df.columns or 'beat_pos' not in df.columns:
        raise ValueError(f"Missing required columns in {beat_path}")

    # Filter: only complete bars (bar_num >= 1) and downbeats (beat_pos == 1)
    mask = (df['bar_num'] >= 1) & (df['beat_pos'] == 1)
    downbeat_times = df.loc[mask, 'downbeat_time(s)'].values

    # Get unique sorted values
    unique_downbeats = sorted(set(downbeat_times))

    return unique_downbeats


def detect_time_signature_from_beatpos(beat_path: str) -> int:
    """
    Detect time signature from beat_pos column pattern.

    Counts the maximum beat_pos per bar and returns the most common value.

    Parameters
    ----------
    beat_path : str
        Path to BeatTransformer output file

    Returns
    -------
    int
        Detected time signature (defaults to 4 if detection fails)

    Examples
    --------
    >>> tsig = detect_time_signature_from_beatpos('track_output.txt')
    >>> tsig
    4
    """
    df = pd.read_csv(beat_path, sep='\t')

    if 'bar_num' in df.columns and 'beat_pos' in df.columns:
        # Use explicit bar_num grouping
        df_complete = df[df['bar_num'] >= 1]
        if len(df_complete) == 0:
            return 4  # Default fallback

        max_beats_per_bar = df_complete.groupby('bar_num')['beat_pos'].max()
        counts = Counter(max_beats_per_bar.values)
        most_common_tsig = counts.most_common(1)[0][0]
        return most_common_tsig

    elif 'beat_pos' in df.columns:
        # Fallback: infer bars from beat_pos resets
        beat_pos_arr = df['beat_pos'].values
        bars_beat_counts = []
        current_max = 0

        for bp in beat_pos_arr:
            if bp == 1 and current_max > 0:
                bars_beat_counts.append(current_max)
                current_max = 1
            else:
                current_max = max(current_max, bp)

        if current_max > 0:
            bars_beat_counts.append(current_max)

        if bars_beat_counts:
            counts = Counter(bars_beat_counts)
            return counts.most_common(1)[0][0]

    # Default fallback
    return 4


# ============================================================================
# DATA CONVERSION FUNCTIONS
# ============================================================================

def df_from_downbeats(downbeats: List[float], tsig: int) -> pd.DataFrame:
    """
    Convert downbeat times to bar-level DataFrame with tempo.

    Parameters
    ----------
    downbeats : List[float]
        List of downbeat timestamps
    tsig : int
        Time signature

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bar_index, start_time, end_time, duration_s, tempo_bpm

    Notes
    -----
    Tempo formula: tempo_bpm = 60.0 / duration_s * tsig

    Examples
    --------
    >>> df = df_from_downbeats([0.0, 2.0, 4.0], tsig=4)
    >>> df['tempo_bpm'].values
    array([120., 120.])
    """
    if len(downbeats) < 2:
        return pd.DataFrame(columns=['bar_index', 'start_time', 'end_time', 'duration_s', 'tempo_bpm'])

    data = []
    for i in range(len(downbeats) - 1):
        start = downbeats[i]
        end = downbeats[i + 1]
        dur = end - start
        tempo = (60.0 / dur) * tsig if dur > EPS else 0.0

        data.append({
            'bar_index': i,
            'start_time': start,
            'end_time': end,
            'duration_s': dur,
            'tempo_bpm': tempo
        })

    return pd.DataFrame(data)


# ============================================================================
# TEMPO CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_factor2(bpm: float, base: float, tol: float = MULT_MATCH_TOL) -> str:
    """
    Classify tempo relative to base as 'double', 'half', or 'normal'.

    Parameters
    ----------
    bpm : float
        Tempo to classify
    base : float
        Reference tempo
    tol : float, optional
        Tolerance (default: 0.10 = ±10%)

    Returns
    -------
    str
        'double', 'half', or 'normal'

    Examples
    --------
    >>> classify_factor2(120, 60, 0.10)
    'double'
    >>> classify_factor2(30, 60, 0.10)
    'half'
    >>> classify_factor2(60, 60, 0.10)
    'normal'
    """
    # Double range: base * 2 ± tol
    lo2 = 2.0 * (1.0 - tol) * base
    hi2 = 2.0 * (1.0 + tol) * base

    # Half range: base * 0.5 ± tol
    loh = 0.5 * (1.0 - tol) * base
    hih = 0.5 * (1.0 + tol) * base

    if loh <= bpm <= hih:
        return "half"
    elif lo2 <= bpm <= hi2:
        return "double"
    else:
        return "normal"


def mask_within_percent(arr: np.ndarray, base: float, percent: float) -> np.ndarray:
    """
    Create boolean mask for values within ±percent of base.

    Parameters
    ----------
    arr : np.ndarray
        Array of values
    base : float
        Base value
    percent : float
        Percentage tolerance

    Returns
    -------
    np.ndarray
        Boolean mask

    Examples
    --------
    >>> mask = mask_within_percent(np.array([90, 100, 110]), 100, 10)
    >>> mask
    array([ True,  True,  True])
    """
    tol = abs(percent) / 100.0
    low = base * (1.0 - tol)
    high = base * (1.0 + tol)
    return (arr >= low) & (arr <= high)


def dominant_base_bpm(
    base_median_raw: float,
    dominant_major: str,
    factor2_orientation: str,
    base_override: Optional[float] = None
) -> float:
    """
    Calculate the dominant BPM base for usability mask.

    Parameters
    ----------
    base_median_raw : float
        Raw median tempo
    dominant_major : str
        'normal' or 'factor2'
    factor2_orientation : str
        'double' or 'half'
    base_override : float, optional
        Override value from BPM threshold logic

    Returns
    -------
    float
        Dominant BPM base

    Examples
    --------
    >>> dominant_base_bpm(60, 'normal', '', None)
    60.0
    >>> dominant_base_bpm(60, 'factor2', 'double', None)
    120.0
    """
    if base_override is not None:
        return base_override

    if dominant_major == 'normal':
        return base_median_raw
    elif dominant_major == 'factor2':
        if factor2_orientation == 'double':
            return base_median_raw * 2.0
        elif factor2_orientation == 'half':
            return base_median_raw * 0.5

    return base_median_raw


# ============================================================================
# CORRECTION FUNCTIONS
# ============================================================================

def build_corrected_from_raw(
    downbeats: List[float],
    df_raw: pd.DataFrame,
    classes: List[str],
    dominant_major: str,
    factor2_orientation: str
) -> Tuple[List[float], List[Dict]]:
    """
    Build corrected downbeat timeline with per-bar metadata.

    Correction logic:
    - If dominant='normal': MERGE double bars, SPLIT half bars
    - If dominant='factor2' + orientation='double': SPLIT normal bars
    - If dominant='factor2' + orientation='half': MERGE normal bars

    Parameters
    ----------
    downbeats : List[float]
        Original downbeat times
    df_raw : pd.DataFrame
        Raw bars DataFrame
    classes : List[str]
        Classification for each bar ('normal', 'double', 'half')
    dominant_major : str
        'normal' or 'factor2'
    factor2_orientation : str
        'double' or 'half'

    Returns
    -------
    tuple
        (corrected_downbeats: List[float], corr_meta: List[Dict])

    Notes
    -----
    corr_meta contains dictionaries with keys:
    - bar_num_raw: Pipe-separated source bar indices
    - action: 'unchanged', 'merged', 'split_A', 'split_B', 'merge_impossible_tail'
    - removed_downbeat_time: Removed downbeats (for merges)
    - inserted_downbeat_time: Inserted downbeats (for splits)
    """
    corrected = []
    corr_meta = []

    i = 0
    n = len(df_raw)

    while i < n:
        cls = classes[i]
        st = df_raw.iloc[i]['start_time']
        et = df_raw.iloc[i]['end_time']

        # Determine action based on classification and dominant pattern
        if dominant_major == 'normal':
            if cls == 'double':
                action = 'merge'
            elif cls == 'half':
                action = 'split'
            else:
                action = 'unchanged'

        elif dominant_major == 'factor2':
            if cls == 'normal':
                if factor2_orientation == 'double':
                    action = 'split'
                elif factor2_orientation == 'half':
                    action = 'merge'
                else:
                    action = 'unchanged'
            else:
                action = 'unchanged'
        else:
            action = 'unchanged'

        # Execute action
        if action == 'merge':
            # Merge with next bar
            if i + 1 < n:
                et_next = df_raw.iloc[i + 1]['end_time']
                corrected.append(st)
                corr_meta.append({
                    'bar_num_raw': f"{i+1}|{i+2}",  # 1-indexed
                    'action': 'merged',
                    'removed_downbeat_time': f"{et:.6f}",
                    'inserted_downbeat_time': ''
                })
                i += 2  # Skip next bar
            else:
                # Can't merge at end
                corrected.append(st)
                corr_meta.append({
                    'bar_num_raw': str(i+1),  # 1-indexed
                    'action': 'merge_impossible_tail',
                    'removed_downbeat_time': '',
                    'inserted_downbeat_time': ''
                })
                i += 1

        elif action == 'split':
            # Split bar in half
            mid = (st + et) / 2.0
            corrected.append(st)
            corrected.append(mid)
            corr_meta.append({
                'bar_num_raw': str(i+1),  # 1-indexed
                'action': 'split_A',
                'removed_downbeat_time': '',
                'inserted_downbeat_time': f"{mid:.6f}"
            })
            corr_meta.append({
                'bar_num_raw': str(i+1),  # 1-indexed
                'action': 'split_B',
                'removed_downbeat_time': '',
                'inserted_downbeat_time': f"{mid:.6f}"
            })
            i += 1

        else:
            # Unchanged
            corrected.append(st)
            corr_meta.append({
                'bar_num_raw': str(i+1),  # 1-indexed
                'action': 'unchanged',
                'removed_downbeat_time': '',
                'inserted_downbeat_time': ''
            })
            i += 1

    # Add final downbeat
    if len(df_raw) > 0:
        corrected.append(df_raw.iloc[-1]['end_time'])

    return corrected, corr_meta


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def write_corrected_downbeats_txt(
    output_path: str,
    df_corr: pd.DataFrame,
    corr_meta: List[Dict],
    tsig: int,
    dominant_major: str,
    factor2_orientation: str,
    base_avg_corr: float,
    avg_kept_corr: float,
    usable_mask: np.ndarray,
    bpm_threshold_str: str,
    usable_base_bpm: float
) -> Path:
    """
    Write corrected downbeats to TXT file with metadata.

    Parameters
    ----------
    output_path : str
        Output file path
    df_corr : pd.DataFrame
        Corrected bars DataFrame
    corr_meta : List[Dict]
        Correction metadata
    tsig : int
        Time signature
    dominant_major : str
        Dominant pattern type
    factor2_orientation : str
        Factor-of-two orientation
    base_avg_corr : float
        Base average corrected tempo
    avg_kept_corr : float
        Average tempo of usable bars
    usable_mask : np.ndarray
        Boolean mask for usable bars
    bpm_threshold_str : str
        BPM threshold value or 'none'
    usable_base_bpm : float
        Usable base BPM

    Returns
    -------
    Path
        Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    usable_count = int(usable_mask.sum())
    total_bars = len(df_corr)

    with open(output_path, 'w') as f:
        # Write metadata
        f.write(f"# time_signature={tsig}\n")
        f.write(f"# dominant_major={dominant_major}\n")
        f.write(f"# factor2_orientation={factor2_orientation}\n")
        f.write(f"# filter_percent=±{OUTLIER_PERCENT:.3f}%\n")
        f.write(f"# base_avg_corrected={base_avg_corr:.2f}\n")
        f.write(f"# avg_kept_corrected={avg_kept_corr:.2f}\n")
        f.write(f"# usable_base_bpm={usable_base_bpm:.2f}\n")
        f.write(f"# usable_count={usable_count}\n")
        f.write(f"# total_bars={total_bars}\n")
        f.write(f"# bpm_charts_threshold={BPM_CHARTS_THRESHOLD}\n")
        f.write(f"# double_bpm_chart_adjusted={DOUBLE_BPM_CHART_ADJUSTED}\n")

        # Write header
        f.write("corrected_bar_num\tbar_num\tcorrected_downbeat_time(s)\tnext_downbeat_time(s)\t")
        f.write("duration_s\ttempo_bpm\tusable\tbpm_threshold\taction\t")
        f.write("removed_downbeat_time(s)\tinserted_downbeat_time(s)\n")

        # Write data
        for i, row in df_corr.iterrows():
            meta = corr_meta[i]
            usable_flag = 1 if usable_mask[i] else 0

            f.write(f"{i+1}\t")  # 1-indexed bar number
            f.write(f"{meta['bar_num_raw']}\t")
            f.write(f"{row['start_time']:.6f}\t")
            f.write(f"{row['end_time']:.6f}\t")
            f.write(f"{row['duration_s']:.6f}\t")
            f.write(f"{row['tempo_bpm']:.2f}\t")
            f.write(f"{usable_flag}\t")
            f.write(f"{bpm_threshold_str}\t")
            f.write(f"{meta['action']}\t")
            f.write(f"{meta['removed_downbeat_time']}\t")
            f.write(f"{meta['inserted_downbeat_time']}\n")

    return output_path


# ============================================================================
# MAIN CORRECTION PIPELINE
# ============================================================================

def correct_downbeats(
    beat_file: str,
    output_file: str,
    verbose: bool = True
) -> Dict:
    """
    Complete downbeat correction pipeline for a single track.

    Parameters
    ----------
    beat_file : str
        Path to BeatTransformer output file
    output_file : str
        Path for corrected downbeats output
    verbose : bool, optional
        Print progress (default: True)

    Returns
    -------
    dict
        Summary statistics with keys:
        - track_name: Track name
        - time_signature: Detected time signature
        - raw_bars: Number of raw bars
        - corrected_bars: Number of corrected bars
        - dominant_pattern: Dominant tempo pattern
        - usable_count: Number of usable bars
        - base_avg_corrected: Average corrected tempo
        - avg_kept_corrected: Average usable tempo

    Examples
    --------
    >>> stats = correct_downbeats('track_output.txt', 'track_corrected.txt')
    >>> stats['corrected_bars']
    45
    """
    if verbose:
        print(f"\nProcessing: {Path(beat_file).name}")

    # Parse input
    downbeats = parse_downbeats_direct_strict(beat_file)
    tsig = detect_time_signature_from_beatpos(beat_file)

    if len(downbeats) < 2:
        if verbose:
            print("  ⚠️  Insufficient downbeats, skipping")
        return None

    # Create raw bars DataFrame
    df_raw = df_from_downbeats(downbeats, tsig)

    if len(df_raw) == 0:
        if verbose:
            print("  ⚠️  No bars created, skipping")
        return None

    # Calculate base median tempo
    base_median_raw = df_raw['tempo_bpm'].median()

    # Classify each bar
    classes = [classify_factor2(bpm, base_median_raw, MULT_MATCH_TOL)
               for bpm in df_raw['tempo_bpm']]

    # Count classifications
    n_normal = classes.count('normal')
    n_double = classes.count('double')
    n_half = classes.count('half')

    # BPM threshold logic
    base_override = None
    if DOUBLE_BPM_CHART_ADJUSTED:
        high_bpm_count = (df_raw['tempo_bpm'] > BPM_CHARTS_THRESHOLD).sum()
        if high_bpm_count > len(df_raw) / 2:
            high_median = df_raw[df_raw['tempo_bpm'] > BPM_CHARTS_THRESHOLD]['tempo_bpm'].median()
            base_override = high_median / 2.0
            # Reclassify
            classes = [classify_factor2(bpm, base_override, MULT_MATCH_TOL)
                      for bpm in df_raw['tempo_bpm']]
            n_normal = classes.count('normal')
            n_double = classes.count('double')
            n_half = classes.count('half')

    # Determine dominant pattern
    if n_normal >= (n_double + n_half):
        dominant_major = 'normal'
        factor2_orientation = ''
    else:
        dominant_major = 'factor2'
        factor2_orientation = 'double' if n_double >= n_half else 'half'

    # Apply corrections
    corrected_downbeats, corr_meta = build_corrected_from_raw(
        downbeats, df_raw, classes, dominant_major, factor2_orientation
    )

    # Create corrected DataFrame
    df_corr = df_from_downbeats(corrected_downbeats, tsig)

    # Calculate usability mask
    usable_base = dominant_base_bpm(base_median_raw, dominant_major, factor2_orientation, base_override)
    usable_mask = mask_within_percent(df_corr['tempo_bpm'].values, usable_base, OUTLIER_PERCENT)

    base_avg_corr = df_corr['tempo_bpm'].mean()
    avg_kept_corr = df_corr.loc[usable_mask, 'tempo_bpm'].mean() if usable_mask.sum() > 0 else base_avg_corr

    bpm_threshold_str = str(BPM_CHARTS_THRESHOLD) if base_override is not None else 'none'

    # Write output
    write_corrected_downbeats_txt(
        output_file,
        df_corr,
        corr_meta,
        tsig,
        dominant_major,
        factor2_orientation,
        base_avg_corr,
        avg_kept_corr,
        usable_mask,
        bpm_threshold_str,
        usable_base
    )

    if verbose:
        print(f"  ✓ Corrected: {len(df_raw)} → {len(df_corr)} bars")
        print(f"  ✓ Dominant: {dominant_major} {factor2_orientation}")
        print(f"  ✓ Usable: {usable_mask.sum()}/{len(df_corr)}")
        print(f"  ✓ Saved: {output_file}")

    return {
        'track_name': Path(beat_file).stem,
        'time_signature': tsig,
        'raw_bars': len(df_raw),
        'corrected_bars': len(df_corr),
        'dominant_pattern': f"{dominant_major} {factor2_orientation}".strip(),
        'usable_count': int(usable_mask.sum()),
        'base_avg_corrected': base_avg_corr,
        'avg_kept_corrected': avg_kept_corr
    }
