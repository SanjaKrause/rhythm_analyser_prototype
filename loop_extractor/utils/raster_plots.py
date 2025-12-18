"""
Raster plot generation for visualizing microtiming phases.

This module creates scatter plots showing onset phases across bars,
comparing different correction methods. Matches the format from AP2_plot_raster_gridshift.ipynb.

Environment: AEinBOX_13_3 (matplotlib, numpy, pandas)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
import sys

# Import config
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

import importlib.util
spec = importlib.util.spec_from_file_location("config_module", _parent_dir / "config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config


def make_bar_colors(n_bars: int) -> List[str]:
    """
    Generate a list of colors for bars.

    Parameters
    ----------
    n_bars : int
        Number of bars

    Returns
    -------
    List[str]
        List of color strings
    """
    # Use a color cycle similar to the notebook
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    return [base_colors[i % len(base_colors)] for i in range(n_bars)]


def plot_raster_single(
    ax: plt.Axes,
    df: pd.DataFrame,
    phase_column: str,
    title: str,
    track_id: str,
    rms_ms: Optional[float] = None,
    show_grid_at_32nds: bool = True,
    ref_onsets: Optional[pd.DataFrame] = None,
    method_name: Optional[str] = None
) -> plt.Axes:
    """
    Plot a single raster subplot showing onset phases across bars.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    df : pd.DataFrame
        Comprehensive phases CSV data with columns:
        - bar_number: bar index
        - phase_* : phase values (0 to 1)
        - onset_time: onset time in seconds
    phase_column : str
        Name of phase column to plot (e.g., 'phase_uncorrected')
    title : str
        Plot title
    track_id : str
        Track identifier
    rms_ms : float, optional
        RMS deviation in milliseconds (shown in title)
    show_grid_at_32nds : bool
        Show vertical grid lines at 32nd notes

    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    # Check if phase column exists
    if phase_column not in df.columns:
        ax.text(0.5, 0.5, f'Column {phase_column} not found',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax

    # Filter valid data - only rows where phase exists (where onsets were detected)
    plot_data = df[['bar_number', phase_column]].dropna()

    if len(plot_data) == 0:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax

    # Get number of bars (use full df for this, not just plot_data)
    n_bars = int(df['bar_number'].max()) + 1

    # Generate bar colors
    colors = make_bar_colors(n_bars)

    # Plot onsets for each bar with "x" markers
    for bar_idx in range(n_bars):
        bar_data = plot_data[plot_data['bar_number'] == bar_idx]
        if len(bar_data) > 0:
            phases = bar_data[phase_column].values
            color = colors[bar_idx % len(colors)]

            ax.scatter(phases, np.full(len(phases), bar_idx),
                       marker="x", s=18, linewidths=1, color=color)

    # Draw reference onset circles if provided
    if ref_onsets is not None and method_name is not None:
        # Filter references for this method
        method_refs = ref_onsets[ref_onsets['method'] == method_name]

        # Debug output
        print(f"    Method '{method_name}': found {len(method_refs)} references")

        steps_per_bar = 16  # TODO: make configurable
        phase_shift = 1.0 / steps_per_bar

        for _, ref in method_refs.iterrows():
            bar_idx = int(ref['bar_number'])
            ref_ms = ref['ref_ms']
            ref_phase = ref['ref_phase']
            grid_phase = ref['grid_phase']

            if 0 <= bar_idx < n_bars:
                # Red circle at grid position (where reference IS after correction)
                grid_phase_shifted = grid_phase + phase_shift
                ax.scatter([grid_phase_shifted], [bar_idx],
                           s=90, facecolors='none', edgecolors='red',
                           linewidths=1.5, marker='o', zorder=10)

                # Pink circle at uncorrected position (where reference WAS before correction)
                if ref_phase is not None and np.isfinite(ref_phase):
                    ref_phase_shifted = ref_phase + phase_shift
                    ax.scatter([ref_phase_shifted], [bar_idx],
                               s=60, facecolors='none', edgecolors='pink',
                               linewidths=1.2, marker='o', zorder=9, alpha=0.7)

                    # Add text label with offset time
                    if ref_ms is not None and np.isfinite(ref_ms):
                        label_x = max(ref_phase_shifted, grid_phase_shifted) + 0.02
                        ax.text(label_x, bar_idx, f'{ref_ms:.1f}ms',
                                fontsize=6, va='center', ha='left', color='red',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                          edgecolor='red', alpha=0.8, linewidth=0.5))

    # Set axis limits - extend to 17/16 to show offsets beyond bar end
    ax.set_xlim(0, 17.0/16.0)
    ax.set_ylim(-0.5, n_bars - 0.5)

    # X-axis: label every 16th (1/16 to 16/16)
    ticks_labels = np.linspace(0, 1, 17)  # 0 to 16/16
    ax.set_xticks(ticks_labels[1:])
    ax.set_xticklabels([f"{i}/16" for i in range(1, 17)])

    # Vertical grid lines
    if show_grid_at_32nds:
        # 32 background gridlines (finer grid)
        ticks_grid = np.linspace(0, 1, 33)  # 0 to 32/32
        for xg in ticks_grid[1:]:
            ax.axvline(xg, color="0.9", linewidth=0.6, zorder=0)

    # Y-axis ticks
    if n_bars > 0:
        tick_step = max(1, n_bars // 10)
        ax.set_yticks(np.arange(0, n_bars, tick_step))

    # Labels
    ax.set_xlabel("bar phase", fontsize=10)
    ax.set_ylabel("bar index (snippet)", fontsize=10)

    # Title with RMS if provided
    if rms_ms is not None:
        full_title = f"{title} | RMS: {rms_ms:.2f} ms"
    else:
        full_title = title

    ax.set_title(f"Onset raster — Track {track_id} — {full_title}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def create_raster_comparison_plot(
    csv_file: str,
    output_file: str,
    track_id: str,
    rms_values: Optional[dict] = None
):
    """
    Create 5-panel raster plot comparing all correction methods.

    Matches the format from AP2_plot_raster_gridshift.ipynb with 5 subplots:
    1. Uncorrected
    2. Per-snippet
    3. Loop-based drum
    4. Loop-based mel
    5. Loop-based pitch

    Parameters
    ----------
    csv_file : str
        Path to comprehensive phases CSV
    output_file : str
        Output PNG/PDF file path
    track_id : str
        Track identifier
    rms_values : dict, optional
        Dictionary with RMS values for each method

    Examples
    --------
    >>> create_raster_comparison_plot(
    ...     'track_comprehensive_phases.csv',
    ...     'track_raster_comparison.png',
    ...     'track_123'
    ... )
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Load reference onsets if available
    ref_onsets = None
    csv_path = Path(csv_file)
    ref_file = csv_path.parent / f"{csv_path.stem}_reference_onsets.csv"
    if ref_file.exists():
        ref_onsets = pd.read_csv(ref_file)
        print(f"  Loaded {len(ref_onsets)} reference onsets")

    # Determine number of bars for figure height
    n_bars = int(df['bar_number'].max()) + 1 if 'bar_number' in df.columns else 10
    fig_height = max(8, min(20, 0.15 * n_bars))

    # Create figure with 5 subplots
    fig = plt.figure(figsize=(12, fig_height * 2.5))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1], hspace=0.3)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # Get RMS values if provided
    rms_uncorr = rms_values.get('uncorrected_ms') if rms_values else None
    rms_per_snip = rms_values.get('per_snippet_ms') if rms_values else None
    rms_drum = rms_values.get('drum_ms') if rms_values else None
    rms_mel = rms_values.get('mel_ms') if rms_values else None
    rms_pitch = rms_values.get('pitch_ms') if rms_values else None

    # Plot 1: Uncorrected (no references)
    plot_raster_single(
        axes[0], df, 'phase_uncorrected',
        'Uncorrected', track_id, rms_ms=rms_uncorr
    )

    # Plot 2: Per-snippet
    plot_raster_single(
        axes[1], df, 'phase_per_snippet_remapped',
        'Per snippet', track_id, rms_ms=rms_per_snip,
        ref_onsets=ref_onsets, method_name='per_snippet'
    )

    # Plot 3: Loop-based drum
    drum_col = None
    for col in df.columns:
        if col.startswith('phase_drum'):
            drum_col = col
            break

    if drum_col:
        # Extract pattern length from column name if present
        # Column format: phase_drum(L=4)
        pattern_len_drum = None
        if '(L=' in drum_col and ')' in drum_col:
            try:
                pattern_len_drum = int(drum_col.split('(L=')[1].split(')')[0])
            except:
                pass

        title_drum = f"Loop-based drum (L={pattern_len_drum})" if pattern_len_drum else "Loop-based drum"
        method_name_drum = f"drum(L={pattern_len_drum})" if pattern_len_drum else "drum"
        plot_raster_single(
            axes[2], df, drum_col,
            title_drum, track_id, rms_ms=rms_drum,
            ref_onsets=ref_onsets, method_name=method_name_drum
        )
    else:
        axes[2].text(0.5, 0.5, 'No drum loop method data',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title(f"Onset raster — Track {track_id} — Loop-based drum")

    # Plot 4: Loop-based mel
    mel_col = None
    for col in df.columns:
        if col.startswith('phase_mel'):
            mel_col = col
            break

    if mel_col:
        # Extract pattern length from column name if present
        # Column format: phase_mel(L=4)
        pattern_len_mel = None
        if '(L=' in mel_col and ')' in mel_col:
            try:
                pattern_len_mel = int(mel_col.split('(L=')[1].split(')')[0])
            except:
                pass

        title_mel = f"Loop-based mel (L={pattern_len_mel})" if pattern_len_mel else "Loop-based mel"
        method_name_mel = f"mel(L={pattern_len_mel})" if pattern_len_mel else "mel"
        plot_raster_single(
            axes[3], df, mel_col,
            title_mel, track_id, rms_ms=rms_mel,
            ref_onsets=ref_onsets, method_name=method_name_mel
        )
    else:
        axes[3].text(0.5, 0.5, 'No mel loop method data',
                     ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title(f"Onset raster — Track {track_id} — Loop-based mel")

    # Plot 5: Loop-based pitch
    pitch_col = None
    for col in df.columns:
        if col.startswith('phase_pitch'):
            pitch_col = col
            break

    if pitch_col:
        # Extract pattern length from column name if present
        # Column format: phase_pitch(L=8)
        pattern_len_pitch = None
        if '(L=' in pitch_col and ')' in pitch_col:
            try:
                pattern_len_pitch = int(pitch_col.split('(L=')[1].split(')')[0])
            except:
                pass

        title_pitch = f"Loop-based pitch (L={pattern_len_pitch})" if pattern_len_pitch else "Loop-based pitch"
        method_name_pitch = f"pitch(L={pattern_len_pitch})" if pattern_len_pitch else "pitch"
        plot_raster_single(
            axes[4], df, pitch_col,
            title_pitch, track_id, rms_ms=rms_pitch,
            ref_onsets=ref_onsets, method_name=method_name_pitch
        )
    else:
        axes[4].text(0.5, 0.5, 'No pitch loop method data',
                     ha='center', va='center', transform=axes[4].transAxes)
        axes[4].set_title(f"Onset raster — Track {track_id} — Loop-based pitch")

    # Overall title
    fig.suptitle(f"Track {track_id} — Raster Plots — All Correction Methods",
                fontsize=13, fontweight="bold")
    plt.subplots_adjust(top=0.96, bottom=0.05, hspace=0.3)

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Raster comparison plot saved to {output_file}")


def create_raster_standard_plot(
    csv_file: str,
    output_file: str,
    track_id: str,
    rms_values: Optional[dict] = None
):
    """
    Create 5-panel raster plot comparing standard loop correction methods.

    Shows comparison of:
    1. Uncorrected
    2. Standard L=1 (1-bar loop)
    3. Standard L=2 (2-bar loop)
    4. Standard L=4 (4-bar loop)
    5. Per-snippet

    Parameters
    ----------
    csv_file : str
        Path to comprehensive phases CSV
    output_file : str
        Output PNG/PDF file path
    track_id : str
        Track identifier
    rms_values : dict, optional
        Dictionary with RMS values for each method

    Examples
    --------
    >>> create_raster_standard_plot(
    ...     'track_comprehensive_phases.csv',
    ...     'track_raster_standard.png',
    ...     'track_123'
    ... )
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Load reference onsets if available
    ref_onsets = None
    csv_path = Path(csv_file)
    ref_file = csv_path.parent / f"{csv_path.stem}_reference_onsets.csv"
    if ref_file.exists():
        ref_onsets = pd.read_csv(ref_file)
        print(f"  Loaded {len(ref_onsets)} reference onsets for standard plot")

    # Determine number of bars for figure height
    n_bars = int(df['bar_number'].max()) + 1 if 'bar_number' in df.columns else 10
    fig_height = max(8, min(20, 0.15 * n_bars))

    # Create figure with 5 subplots
    fig = plt.figure(figsize=(12, fig_height * 2.5))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1], hspace=0.3)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # Get RMS values if provided
    rms_uncorr = rms_values.get('uncorrected_ms') if rms_values else None
    rms_std_l1 = rms_values.get('standard_L1_ms') if rms_values else None
    rms_std_l2 = rms_values.get('standard_L2_ms') if rms_values else None
    rms_std_l4 = rms_values.get('standard_L4_ms') if rms_values else None
    rms_per_snip = rms_values.get('per_snippet_ms') if rms_values else None

    # Plot 1: Uncorrected (no references)
    plot_raster_single(
        axes[0], df, 'phase_uncorrected',
        'Uncorrected', track_id, rms_ms=rms_uncorr
    )

    # Plot 2: Standard L=1
    std_l1_col = None
    for col in df.columns:
        if col.startswith('phase_standard_L1'):
            std_l1_col = col
            break

    if std_l1_col:
        plot_raster_single(
            axes[1], df, std_l1_col,
            'Standard L=1 (1-bar loop)', track_id, rms_ms=rms_std_l1,
            ref_onsets=ref_onsets, method_name='standard_L1(L=1)'
        )
    else:
        axes[1].text(0.5, 0.5, 'No standard L=1 data',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title(f"Onset raster — Track {track_id} — Standard L=1")

    # Plot 3: Standard L=2
    std_l2_col = None
    for col in df.columns:
        if col.startswith('phase_standard_L2'):
            std_l2_col = col
            break

    if std_l2_col:
        plot_raster_single(
            axes[2], df, std_l2_col,
            'Standard L=2 (2-bar loop)', track_id, rms_ms=rms_std_l2,
            ref_onsets=ref_onsets, method_name='standard_L2(L=2)'
        )
    else:
        axes[2].text(0.5, 0.5, 'No standard L=2 data',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title(f"Onset raster — Track {track_id} — Standard L=2")

    # Plot 4: Standard L=4
    std_l4_col = None
    for col in df.columns:
        if col.startswith('phase_standard_L4'):
            std_l4_col = col
            break

    if std_l4_col:
        plot_raster_single(
            axes[3], df, std_l4_col,
            'Standard L=4 (4-bar loop)', track_id, rms_ms=rms_std_l4,
            ref_onsets=ref_onsets, method_name='standard_L4(L=4)'
        )
    else:
        axes[3].text(0.5, 0.5, 'No standard L=4 data',
                     ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title(f"Onset raster — Track {track_id} — Standard L=4")

    # Plot 5: Per-snippet
    plot_raster_single(
        axes[4], df, 'phase_per_snippet_remapped',
        'Per snippet', track_id, rms_ms=rms_per_snip,
        ref_onsets=ref_onsets, method_name='per_snippet'
    )

    # Overall title
    fig.suptitle(f"Track {track_id} — Raster Plots — Standard Loop Methods",
                fontsize=13, fontweight="bold")
    plt.subplots_adjust(top=0.96, bottom=0.05, hspace=0.3)

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Raster standard plot saved to {output_file}")


def create_all_plots(
    csv_file: str,
    output_dir: str,
    track_id: str,
    rms_summary_file: Optional[str] = None
):
    """
    Create all raster plots for a track.

    Parameters
    ----------
    csv_file : str
        Path to comprehensive phases CSV
    output_dir : str
        Output directory for plots
    track_id : str
        Track identifier
    rms_summary_file : str, optional
        Path to RMS summary JSON file

    Examples
    --------
    >>> create_all_plots(
    ...     'output/track/5_grid/track_comprehensive_phases.csv',
    ...     'output/track/5_grid/',
    ...     'track'
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating raster plots...")

    # Load RMS values if available
    rms_values = None
    if rms_summary_file and Path(rms_summary_file).exists():
        import json
        with open(rms_summary_file, 'r') as f:
            rms_values = json.load(f)

    # Try to load from same directory if not provided
    if rms_values is None:
        rms_file_in_dir = output_dir.parent / '6_rms' / f'{track_id}_rms_summary.json'
        if rms_file_in_dir.exists():
            import json
            with open(rms_file_in_dir, 'r') as f:
                rms_values = json.load(f)

    # Create 5-panel raster comparison plot (drum/mel/pitch methods)
    print("  Creating raster comparison plot...")
    create_raster_comparison_plot(
        csv_file,
        str(output_dir / f"{track_id}_raster_comparison.png"),
        track_id,
        rms_values=rms_values
    )

    # Create 5-panel standard raster plot (standard L=1, L=2, L=4)
    print("  Creating standard raster plot...")
    create_raster_standard_plot(
        csv_file,
        str(output_dir / f"{track_id}_raster_standard.png"),
        track_id,
        rms_values=rms_values
    )

    print(f"  ✓ All plots created in {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python raster_plots.py <csv_file> <output_dir> <track_id>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_dir = sys.argv[2]
    track_id = sys.argv[3]

    create_all_plots(csv_file, output_dir, track_id)
