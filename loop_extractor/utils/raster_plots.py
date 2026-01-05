"""
Simplified raster plot generation for visualizing microtiming phases.

This module creates scatter plots showing onset phases across bars,
comparing 3 correction methods:
1. Uncorrected
2. Per-snippet correction
3. 4-bar loop correction

Environment: AEinBOX_13_3 (matplotlib, numpy, pandas)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
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
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    return [base_colors[i % len(base_colors)] for i in range(n_bars)]


def plot_raster_single(
    ax: plt.Axes,
    df: pd.DataFrame,
    phase_column: str,
    title: str,
    track_id: str,
    show_grid_at_32nds: bool = True
) -> plt.Axes:
    """
    Plot a single raster subplot showing onset phases across bars.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    df : pd.DataFrame
        Raster CSV data with columns:
        - bar_number: bar index (snippet-relative)
        - tick_16th: tick position (0-15)
        - phase_* : phase values (0 to 1)
        - onset_time: onset time in seconds
    phase_column : str
        Name of phase column to plot (e.g., 'phase_uncorrected')
    title : str
        Plot title
    track_id : str
        Track identifier
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

    # Filter valid data - only rows where phase exists
    plot_data = df[['bar_number', phase_column]].dropna()

    if len(plot_data) == 0:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax

    # Get number of bars
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

    # Set axis limits - extend beyond 1.0 to show offsets beyond bar end
    ax.set_xlim(-0.0625, 1.0625)
    ax.set_ylim(-0.5, n_bars - 0.5)

    # X-axis: tick 0 = 1/16, tick 1 = 2/16, ..., tick 15 = 16/16
    # Phase values: 0, 1/16, 2/16, ..., 15/16
    ticks_positions = np.linspace(0, 15.0/16.0, 16)  # 0, 1/16, ..., 15/16
    ax.set_xticks(ticks_positions)
    ax.set_xticklabels([f"{i+1}/16" for i in range(16)])

    # Vertical grid lines at 32nd note positions
    if show_grid_at_32nds:
        # 32nd note grid: positions 0, 1/32, 2/32, ..., 30/32 (matching our 0 to 15/16 range)
        ticks_grid = np.linspace(0, 15.0/16.0, 31)  # 0 to 30/32 (= 15/16)
        for xg in ticks_grid:
            ax.axvline(xg, color="0.9", linewidth=0.6, zorder=0)

    # Y-axis ticks
    if n_bars > 0:
        tick_step = max(1, n_bars // 10)
        ax.set_yticks(np.arange(0, n_bars, tick_step))

    # Labels
    ax.set_xlabel("bar phase", fontsize=10)
    ax.set_ylabel("bar index (snippet)", fontsize=10)

    # Title
    ax.set_title(f"Onset raster — Track {track_id} — {title}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def create_raster_plot(
    csv_file: str,
    output_file: str,
    track_id: str
):
    """
    Create 3-panel raster plot comparing all correction methods.

    Panels:
    1. Uncorrected
    2. Per-snippet correction
    3. 4-bar loop correction

    Parameters
    ----------
    csv_file : str
        Path to raster CSV file
    output_file : str
        Output PNG/PDF file path
    track_id : str
        Track identifier

    Examples
    --------
    >>> create_raster_plot(
    ...     'track_raster.csv',
    ...     'track_raster_plot.png',
    ...     'track_123'
    ... )
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Determine number of bars for figure height
    n_bars = int(df['bar_number'].max()) + 1 if 'bar_number' in df.columns else 10
    fig_height = max(8, min(20, 0.15 * n_bars))

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(12, fig_height * 1.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # Plot 1: Uncorrected
    plot_raster_single(
        axes[0], df, 'phase_uncorrected',
        'Uncorrected', track_id
    )

    # Plot 2: Per-snippet correction
    plot_raster_single(
        axes[1], df, 'phase_per_snippet',
        'Per-snippet correction', track_id
    )

    # Plot 3: 4-bar loop correction
    plot_raster_single(
        axes[2], df, 'phase_4bar_loop',
        '4-bar loop correction', track_id
    )

    # Overall title
    fig.suptitle(f"Track {track_id} — Raster Plots — All Correction Methods",
                fontsize=13, fontweight="bold")
    plt.subplots_adjust(top=0.96, bottom=0.05, hspace=0.3)

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Raster plot saved to {output_file}")


def create_all_plots(
    csv_file: str,
    output_dir: str,
    track_id: str,
    rms_summary_file: Optional[str] = None
):
    """
    Create all raster plots for a track (backward compatibility wrapper).

    This function creates a single simplified raster plot with 3 methods.
    The rms_summary_file parameter is ignored in the new simplified version.

    Parameters
    ----------
    csv_file : str
        Path to raster CSV file
    output_dir : str
        Output directory for plots
    track_id : str
        Track identifier
    rms_summary_file : str, optional
        Ignored (kept for backward compatibility)

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

    # Create single raster plot with 3 methods
    output_file = output_dir / f"{track_id}_raster.png"
    create_raster_plot(csv_file, str(output_file), track_id)

    print(f"  ✓ Raster plot created in {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python raster_plots.py <csv_file> <output_file> <track_id>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = sys.argv[2]
    track_id = sys.argv[3]

    create_raster_plot(csv_file, output_file, track_id)
