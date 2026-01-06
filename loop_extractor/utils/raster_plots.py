"""
Simplified raster plot generation for visualizing microtiming phases.

This module creates scatter plots showing onset phases across bars,
comparing 4 correction methods:
1. Uncorrected
2. Per-snippet correction
3. 4-bar loop correction
4. 4-bar pattern flexStart correction

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
    rms_ms : float, optional
        RMS deviation in milliseconds (shown in title)
    show_grid_at_32nds : bool
        Show vertical grid lines at 32nd notes
    ref_onsets : pd.DataFrame, optional
        Reference onsets DataFrame for drawing red/pink circles
    method_name : str, optional
        Method name to filter reference onsets

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

    # Draw reference onset circles if provided
    if ref_onsets is not None and method_name is not None:
        # Filter references for this method
        method_refs = ref_onsets[ref_onsets['method'] == method_name]

        # Debug output
        print(f"    Method '{method_name}': found {len(method_refs)} references")

        steps_per_bar = 16  # TODO: make configurable

        for _, ref in method_refs.iterrows():
            bar_idx = int(ref['bar_number'])
            ref_ms = ref['ref_ms']
            ref_phase = ref['ref_phase']
            grid_phase = ref['grid_phase']
            bar_duration = ref.get('bar_duration', None)

            if 0 <= bar_idx < n_bars:
                # Determine if this is a "new" method (different coordinate system)
                is_new_method = 'new_per_snippet' in method_name or 'new_4bar' in method_name

                if is_new_method and bar_duration is not None and bar_duration > 0:
                    # NEW METHOD PARADIGM:
                    # ref_phase = where onset actually is (in uncorrected grid)
                    # grid_phase = where 1/16 position is in uncorrected grid (should be 0.0 for tick 0)
                    # After correction: grid shifts by ref_ms so that grid_phase aligns with onset

                    ref_s = ref_ms / 1000.0

                    # Red circle: Where the onset IS after correction (should be at 0ms = at tick 0)
                    # The target tick is 0, which in display coordinates is 0.0
                    red_phase_corrected = 0.0

                    # Pink circle: Where tick 0 WAS before correction (in corrected coordinates)
                    pink_phase_corrected = grid_phase - (ref_s / bar_duration)

                    ax.scatter([red_phase_corrected], [bar_idx],
                               s=90, facecolors='none', edgecolors='red',
                               linewidths=1.5, marker='o', zorder=10)

                    ax.scatter([pink_phase_corrected], [bar_idx],
                               s=60, facecolors='none', edgecolors='pink',
                               linewidths=1.2, marker='o', zorder=9, alpha=0.7)

                    # Add text label with offset time
                    if ref_ms is not None and np.isfinite(ref_ms):
                        label_x = max(red_phase_corrected, pink_phase_corrected) + 0.02
                        ax.text(label_x, bar_idx, f'{ref_ms:.1f}ms',
                                fontsize=6, va='center', ha='left', color='red',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                          edgecolor='red', alpha=0.8, linewidth=0.5))
                else:
                    # OLD METHOD PARADIGM (per_snippet and 4bar_loop):
                    # ref_phase = where onset actually is
                    # grid_phase = where grid position should be (should be 0.0 for tick 0)
                    # After correction: grid moves to meet onset, so they align

                    # Red circle at grid position (where reference IS after correction at tick 0)
                    ax.scatter([grid_phase], [bar_idx],
                               s=90, facecolors='none', edgecolors='red',
                               linewidths=1.5, marker='o', zorder=10)

                    # Pink circle at uncorrected position (where reference WAS before correction)
                    if ref_phase is not None and np.isfinite(ref_phase):
                        ax.scatter([ref_phase], [bar_idx],
                                   s=60, facecolors='none', edgecolors='pink',
                                   linewidths=1.2, marker='o', zorder=9, alpha=0.7)

                        # Add text label with offset time
                        if ref_ms is not None and np.isfinite(ref_ms):
                            label_x = max(ref_phase, grid_phase) + 0.02
                            ax.text(label_x, bar_idx, f'{ref_ms:.1f}ms',
                                    fontsize=6, va='center', ha='left', color='red',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                              edgecolor='red', alpha=0.8, linewidth=0.5))

    # Set axis limits - start at -1/32 (half tick before first tick), extend to 17/16 to show offsets beyond bar end
    ax.set_xlim(-1.0/32.0, 17.0/16.0)
    ax.set_ylim(-0.5, n_bars - 0.5)

    # X-axis: tick positions 0, 1/16, 2/16, ..., 15/16 labeled as 1/16, 2/16, ..., 16/16
    ticks_positions = np.linspace(0, 15.0/16.0, 16)  # 0 to 15/16
    ax.set_xticks(ticks_positions)
    ax.set_xticklabels([f"{i}/16" for i in range(1, 17)])

    # Vertical grid lines
    if show_grid_at_32nds:
        # 32 background gridlines at 32nd note positions (0, 1/32, 2/32, ..., 31/32)
        ticks_grid = np.linspace(0, 31.0/32.0, 32)  # 0 to 31/32
        for xg in ticks_grid:
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


def create_raster_plot(
    csv_file: str,
    output_file: str,
    track_id: str
):
    """
    Create 4-panel raster plot comparing all correction methods.

    Panels:
    1. Uncorrected
    2. Per-snippet correction
    3. 4-bar loop correction
    4. 4-bar pattern flexStart correction

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

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(12, fig_height * 2))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # Plot 1: Uncorrected (no reference circles)
    plot_raster_single(
        axes[0], df, 'phase_uncorrected',
        'Uncorrected', track_id
    )

    # Plot 2: Per-snippet correction
    plot_raster_single(
        axes[1], df, 'phase_per_snippet',
        'Per-snippet correction', track_id,
        ref_onsets=ref_onsets, method_name='per_snippet'
    )

    # Plot 3: 4-bar pattern correction
    plot_raster_single(
        axes[2], df, 'phase_4bar_loop',
        '4-Bar Pattern correction', track_id,
        ref_onsets=ref_onsets, method_name='4bar_loop'
    )

    # Plot 4: 4-bar pattern flexStart correction
    plot_raster_single(
        axes[3], df, 'phase_4bar_pattern_flexStart',
        '4-bar pattern flexStart correction', track_id,
        ref_onsets=ref_onsets, method_name='4bar_pattern_flexStart'
    )

    # Overall title
    fig.suptitle(f"Track {track_id} — Raster Plots — All Correction Methods",
                fontsize=13, fontweight="bold")
    plt.subplots_adjust(top=0.97, bottom=0.05, hspace=0.3)

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

    This function creates a single simplified raster plot with 4 methods.
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
