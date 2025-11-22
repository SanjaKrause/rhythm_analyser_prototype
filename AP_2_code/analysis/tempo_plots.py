"""
Tempo plot generation for AP2 pipeline.

Creates 8 plots comparing uncorrected vs corrected bar tempos:
1-4: Uncorrected (full song + snippet, time series + histogram)
5-8: Corrected (full song + snippet, time series + histogram)

Also generates:
- Bar tempo CSV file
- Optional: Combined PDF with all plots

Environment: AEinBOX_13_3
Dependencies: matplotlib, numpy, pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Optional, Tuple, List
from collections import Counter


# ============================================================================
# CONFIGURATION
# ============================================================================

BIN_COUNT = 120  # Number of bins for histograms


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def detect_time_signature(beat_path: str) -> int:
    """
    Detect time signature from beat_pos column in beat file.

    Parameters
    ----------
    beat_path : str
        Path to beat transformer output file

    Returns
    -------
    int
        Time signature (beats per bar), default 4
    """
    bar_beats = {}
    with open(beat_path, 'r') as f:
        for line in f.readlines()[1:]:
            p = line.split()
            if len(p) == 5:
                pos, bar = int(p[2]), int(p[4])
                bar_beats[bar] = max(bar_beats.get(bar, 0), pos)

    if bar_beats:
        return Counter(bar_beats.values()).most_common(1)[0][0]
    return 4


def load_uncorrected_bar_tempos(beat_path: str) -> Tuple[List[float], List[float], int]:
    """
    Load bar tempos from original beat_transformer output file.

    Parameters
    ----------
    beat_path : str
        Path to beat transformer output file

    Returns
    -------
    tuple
        (bar_tempos, downbeat_times, time_signature)
    """
    sig = detect_time_signature(beat_path)

    # Extract downbeat times (beat_pos == 1)
    downbeat_times = []
    with open(beat_path, 'r') as f:
        for line in f.readlines()[1:]:
            p = line.split()
            if len(p) == 5 and int(p[2]) == 1:
                downbeat_times.append(float(p[0]))
            elif len(p) == 2 and int(p[1]) == 1:
                downbeat_times.append(float(p[0]))

    downbeat_times.sort()

    # Compute bar tempos
    intervals = [t2 - t1 for t1, t2 in zip(downbeat_times, downbeat_times[1:]) if t2 > t1]
    bar_tempos = [round((60.0 / iv) * sig, 2) for iv in intervals]

    return bar_tempos, downbeat_times, sig


def load_corrected_bar_tempos(corrected_path: str) -> Tuple[List[float], List[float], List[bool], int]:
    """
    Load bar tempos from corrected downbeat file.

    Parameters
    ----------
    corrected_path : str
        Path to corrected downbeats file

    Returns
    -------
    tuple
        (bar_tempos, downbeat_times, usable_mask, time_signature)
    """
    # Read file and extract metadata
    with open(corrected_path, 'r') as f:
        lines = f.readlines()

    # Parse time signature from header
    sig = 4
    for line in lines:
        if line.startswith('# time_signature='):
            sig = int(line.split('=')[1].strip())
            break

    # Find data start (skip comments and header)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('corrected_bar_num'):
            data_start = i + 1
            break

    # Parse data rows
    downbeat_times = []
    bar_tempos = []
    usable_mask = []

    for line in lines[data_start:]:
        parts = line.strip().split('\t')
        if len(parts) >= 7:
            downbeat_times.append(float(parts[2]))  # corrected_downbeat_time(s)
            bar_tempos.append(float(parts[5]))      # tempo_bpm
            usable_mask.append(bool(int(parts[6]))) # usable

    return bar_tempos, downbeat_times, usable_mask, sig


def filter_snippet_bars(
    bar_tempos: List[float],
    downbeat_times: List[float],
    snip_start: float,
    snip_end: float,
    usable_mask: Optional[List[bool]] = None
) -> Tuple[List[float], List[int], List[bool]]:
    """
    Filter bars to only those within the snippet time window.

    Parameters
    ----------
    bar_tempos : List[float]
        Bar tempos in BPM
    downbeat_times : List[float]
        Downbeat times in seconds
    snip_start : float
        Snippet start time in seconds
    snip_end : float
        Snippet end time in seconds
    usable_mask : List[bool], optional
        Usability mask for corrected bars

    Returns
    -------
    tuple
        (snippet_tempos, snippet_indices, snippet_usable)
    """
    snippet_tempos = []
    snippet_indices = []
    snippet_usable = []

    for i in range(len(bar_tempos)):
        if i < len(downbeat_times):
            bar_start = downbeat_times[i]
            if snip_start <= bar_start < snip_end:
                snippet_tempos.append(bar_tempos[i])
                snippet_indices.append(i)
                if usable_mask is not None and i < len(usable_mask):
                    snippet_usable.append(usable_mask[i])

    return snippet_tempos, snippet_indices, snippet_usable


def calculate_snippet_xspan(
    downbeat_times: List[float],
    snip_start: float,
    snip_end: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate x-axis span for snippet window visualization.

    Parameters
    ----------
    downbeat_times : List[float]
        Downbeat times in seconds
    snip_start : float
        Snippet start time
    snip_end : float
        Snippet end time

    Returns
    -------
    tuple
        (x_left, x_right) in bar index coordinates, or (None, None) if invalid
    """
    if not downbeat_times or snip_start is None or snip_end is None:
        return (None, None)

    x_left, x_right = None, None

    # Find left edge
    for i, db_time in enumerate(downbeat_times):
        if db_time <= snip_start:
            x_left = i
        if db_time >= snip_start and x_left is None:
            x_left = i - 0.5 if i > 0 else 0
            break

    # Find right edge
    for i, db_time in enumerate(downbeat_times):
        if db_time < snip_end:
            x_right = i
        if db_time >= snip_end:
            x_right = i + 0.5
            break

    if x_right is None and downbeat_times:
        x_right = len(downbeat_times) - 1

    if x_left is not None and x_right is not None and x_right > x_left:
        return (x_left, x_right)

    return (None, None)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_tempo_over_time(
    ax,
    bar_indices: List[int],
    bar_tempos: List[float],
    title: str,
    color: str = 'blue',
    usable_mask: Optional[List[bool]] = None,
    snippet_xspan: Optional[Tuple[float, float]] = None
):
    """
    Plot bar tempo over time (bar index on x-axis).
    Shows snippet window as shaded region if provided.
    Shows usable/non-usable bars with different styling if usable_mask provided.
    """
    if not bar_tempos:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_axis_off()
        return

    # Plot snippet window first (if provided)
    if snippet_xspan and snippet_xspan[0] is not None and snippet_xspan[1] is not None:
        ax.axvspan(snippet_xspan[0], snippet_xspan[1], alpha=0.15, color='orange', label='Snippet window')

    # If usable_mask provided, plot with usable/non-usable distinction
    if usable_mask is not None:
        # Convert to numpy arrays
        indices_arr = np.array(bar_indices)
        tempos_arr = np.array(bar_tempos)
        usable_arr = np.array(usable_mask)

        # Plot ALL bars first (faded style)
        ax.plot(indices_arr, tempos_arr, marker='o', linestyle='-', color=color,
               linewidth=1.0, markersize=3, alpha=0.35, label='corrected (all)')

        # Plot USABLE bars on top (bold style)
        if np.any(usable_arr):
            ax.plot(indices_arr[usable_arr], tempos_arr[usable_arr],
                   marker='o', linestyle='-', color=color, linewidth=1.8,
                   markersize=4, label='corrected (usable)')

            # Calculate average of usable bars only
            avg_usable = np.mean(tempos_arr[usable_arr])
            ax.axhline(avg_usable, linestyle='--', linewidth=1.0, color='red',
                      alpha=0.7, label=f'avg(corrected, usable)={avg_usable:.1f} BPM')
    else:
        # No usable mask - plot all bars normally
        ax.plot(bar_indices, bar_tempos, marker='o', linestyle='-', color=color,
               linewidth=1.5, markersize=3)

        # Add average line
        avg = np.mean(bar_tempos)
        ax.axhline(avg, linestyle='--', linewidth=1.0, color='red',
                  alpha=0.7, label=f'avg={avg:.1f} BPM')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Bar Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tempo (BPM)', fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)


def plot_tempo_histogram(
    ax,
    bar_tempos: List[float],
    bins: np.ndarray,
    title: str,
    color: str = 'skyblue',
    usable_mask: Optional[List[bool]] = None
):
    """
    Plot bar tempo histogram with dual y-axis (count and percentage).
    If usable_mask provided, only plot usable bars.
    """
    # Filter to usable bars if mask provided
    if usable_mask is not None:
        tempos_to_plot = [t for t, u in zip(bar_tempos, usable_mask) if u]
        if not tempos_to_plot:
            ax.text(0.5, 0.5, 'No usable data', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_axis_off()
            return
    else:
        tempos_to_plot = bar_tempos
        if not tempos_to_plot:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_axis_off()
            return

    counts, _, _ = ax.hist(tempos_to_plot, bins=bins, color=color, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Tempo (BPM)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Set y-axis limits
    max_count = counts.max() if counts.size else 1
    ax.set_ylim(0, max_count * 1.1)

    # Secondary axis for percentage
    ax2 = ax.twinx()
    total = counts.sum() if counts.sum() > 0 else 1
    rel = [c / total * 100 for c in counts]
    max_rel = max(rel) if rel else 100
    ax2.set_ylim(0, max_rel * 1.1)
    ax2.set_ylabel('Percent (%)', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', which='major', labelsize=10)

    # Set x-axis limits
    ax.set_xlim(bins[0], bins[-1])


# ============================================================================
# CSV EXPORT
# ============================================================================

def save_bar_tempos_csv(
    bar_tempos_uncorr: List[float],
    bar_tempos_corr: List[float],
    downbeat_times_uncorr: List[float],
    downbeat_times_corr: List[float],
    usable_mask: List[bool],
    output_path: str
) -> Path:
    """
    Save bar tempos to CSV file.

    Parameters
    ----------
    bar_tempos_uncorr : List[float]
        Uncorrected bar tempos
    bar_tempos_corr : List[float]
        Corrected bar tempos
    downbeat_times_uncorr : List[float]
        Uncorrected downbeat times
    downbeat_times_corr : List[float]
        Corrected downbeat times
    usable_mask : List[bool]
        Usability mask for corrected bars
    output_path : str
        Output CSV file path

    Returns
    -------
    Path
        Path to saved CSV file
    """
    # Create dataframe with both uncorrected and corrected data
    # Note: bar_tempos has one less element than downbeat_times (interval-based)
    max_len = max(len(downbeat_times_uncorr), len(downbeat_times_corr))

    # Pad all arrays to max_len
    def pad_list(lst, length, fill=np.nan):
        return list(lst) + [fill] * (length - len(lst))

    data = {
        'bar_num': list(range(max_len)),
        'downbeat_time_uncorrected_s': pad_list(downbeat_times_uncorr, max_len),
        'tempo_uncorrected_bpm': pad_list(bar_tempos_uncorr, max_len),
        'downbeat_time_corrected_s': pad_list(downbeat_times_corr, max_len),
        'tempo_corrected_bpm': pad_list(bar_tempos_corr, max_len),
        'usable': pad_list(usable_mask, max_len, fill=False)
    }

    df = pd.DataFrame(data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return output_path


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def create_tempo_plots(
    beats_file: str,
    corrected_downbeats_file: str,
    output_dir: str,
    track_id: str,
    snippet_start: Optional[float] = None,
    snippet_duration: float = 30.0
) -> dict:
    """
    Create 8-panel tempo plots comparing uncorrected vs corrected bar tempos.

    Also saves bar tempo CSV file.

    Parameters
    ----------
    beats_file : str
        Path to beat transformer output file
    corrected_downbeats_file : str
        Path to corrected downbeats file
    output_dir : str
        Output directory for plots and CSV
    track_id : str
        Track identifier
    snippet_start : float, optional
        Snippet start time in seconds
    snippet_duration : float
        Snippet duration in seconds (default: 30.0)

    Returns
    -------
    dict
        Dictionary with paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load uncorrected data
    uncorr_tempos, uncorr_db_times, sig_uncorr = load_uncorrected_bar_tempos(beats_file)
    uncorr_indices = list(range(len(uncorr_tempos)))

    # Load corrected data
    corr_tempos, corr_db_times, corr_usable, sig_corr = load_corrected_bar_tempos(corrected_downbeats_file)
    corr_indices = list(range(len(corr_tempos)))

    # Calculate snippet window
    snip_end = snippet_start + snippet_duration if snippet_start is not None else None

    # Calculate snippet xspan for visualization
    uncorr_xspan = calculate_snippet_xspan(uncorr_db_times, snippet_start, snip_end)
    corr_xspan = calculate_snippet_xspan(corr_db_times, snippet_start, snip_end)

    # Filter snippet data
    if snippet_start is not None and snip_end is not None:
        uncorr_snip_tempos, uncorr_snip_indices, _ = filter_snippet_bars(
            uncorr_tempos, uncorr_db_times, snippet_start, snip_end
        )
        corr_snip_tempos, corr_snip_indices, corr_snip_usable = filter_snippet_bars(
            corr_tempos, corr_db_times, snippet_start, snip_end, corr_usable
        )
    else:
        uncorr_snip_tempos, uncorr_snip_indices = [], []
        corr_snip_tempos, corr_snip_indices, corr_snip_usable = [], [], []

    # Create bin edges for histograms
    all_tempos = uncorr_tempos + corr_tempos
    if all_tempos:
        bins = np.histogram_bin_edges(all_tempos, bins=BIN_COUNT)
    else:
        bins = np.linspace(60, 180, BIN_COUNT)

    # Create 8-panel figure
    fig, axes = plt.subplots(8, 1, figsize=(10, 24))
    fig.suptitle(f"Track {track_id} - Bar Tempo Analysis (Time Sig: {sig_uncorr}/4)",
                fontsize=14, fontweight='bold')

    # UNCORRECTED plots (no usable mask for uncorrected)
    plot_tempo_over_time(axes[0], uncorr_indices, uncorr_tempos,
                        '1. Bar Tempo Over Time (Full Song) - Uncorrected',
                        color='blue', snippet_xspan=uncorr_xspan)
    plot_tempo_over_time(axes[1], uncorr_snip_indices, uncorr_snip_tempos,
                        '2. Bar Tempo Over Time (Snippet) - Uncorrected',
                        color='blue')
    plot_tempo_histogram(axes[2], uncorr_tempos, bins,
                        '3. Bar Tempo Histogram (Full Song) - Uncorrected',
                        color='skyblue')
    plot_tempo_histogram(axes[3], uncorr_snip_tempos, bins,
                        '4. Bar Tempo Histogram (Snippet) - Uncorrected',
                        color='skyblue')

    # CORRECTED plots (with usable mask)
    plot_tempo_over_time(axes[4], corr_indices, corr_tempos,
                        '5. Bar Tempo Over Time (Full Song) - Corrected',
                        color='green', usable_mask=corr_usable,
                        snippet_xspan=corr_xspan)
    plot_tempo_over_time(axes[5], corr_snip_indices, corr_snip_tempos,
                        '6. Bar Tempo Over Time (Snippet) - Corrected',
                        color='green', usable_mask=corr_snip_usable)
    plot_tempo_histogram(axes[6], corr_tempos, bins,
                        '7. Bar Tempo Histogram (Full Song) - Corrected',
                        color='lightgreen', usable_mask=corr_usable)
    plot_tempo_histogram(axes[7], corr_snip_tempos, bins,
                        '8. Bar Tempo Histogram (Snippet) - Corrected',
                        color='lightgreen', usable_mask=corr_snip_usable)

    fig.tight_layout(rect=[0, 0.01, 1, 0.99])

    # Save plot
    plot_path = output_dir / f'{track_id}_tempo_plots.pdf'
    fig.savefig(plot_path)
    plt.close(fig)

    # Save CSV
    csv_path = output_dir / f'{track_id}_bar_tempos.csv'
    save_bar_tempos_csv(
        uncorr_tempos,
        corr_tempos,
        uncorr_db_times,
        corr_db_times,
        corr_usable,
        str(csv_path)
    )

    return {
        'plot_pdf': str(plot_path),
        'csv': str(csv_path)
    }


if __name__ == "__main__":
    # Test with example files
    print("Tempo plots module loaded successfully")
