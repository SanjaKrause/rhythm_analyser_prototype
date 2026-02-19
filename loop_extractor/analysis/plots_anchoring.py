"""
Section-anchored raster plot generation.

This module creates raster plots for section-anchored onset CSVs,
with one subplot per CSV file found in the anchoring directory.

Standalone module - no raster.py dependency.

Output folder: 6.1_anchoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import sys
import re

# Import config from parent directory
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


def parse_csv_metadata(csv_path: str) -> Dict[str, str]:
    """
    Parse metadata from CSV header comments.

    Parameters
    ----------
    csv_path : str
        Path to anchoring CSV file

    Returns
    -------
    Dict[str, str]
        Dictionary of metadata key-value pairs
    """
    metadata = {}
    with open(csv_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Parse "# key=value" format
                match = re.match(r'#\s*(\w+)=(.+)', line.strip())
                if match:
                    metadata[match.group(1)] = match.group(2)
            else:
                break
    return metadata


def plot_raster_single(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    track_id: str,
    show_grid_at_32nds: bool = True,
    ref_onsets: Optional[pd.DataFrame] = None
) -> plt.Axes:
    """
    Plot a single raster subplot showing onset phases across bars.

    Uses 'phase' column from anchoring CSV.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    df : pd.DataFrame
        Anchoring CSV data with columns:
        - bar_number: bar index (pattern-relative)
        - tick_16th: tick position (0-15)
        - phase: phase values (0 to 1)
        - onset_time: onset time in seconds
    title : str
        Plot title
    track_id : str
        Track identifier
    show_grid_at_32nds : bool
        Show vertical grid lines at 32nd notes
    ref_onsets : pd.DataFrame, optional
        Reference onsets DataFrame with columns:
        - bar_number: bar index (snippet-relative)
        - ref_ms: offset in milliseconds
        - ref_phase: where onset was (as phase of bar)
        - grid_phase: where grid position is (0.0 for tick 0)
        - bar_duration: duration of the bar

    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    # Check if required columns exist
    if 'phase' not in df.columns or 'onset_time' not in df.columns:
        ax.text(0.5, 0.5, 'Required columns not found',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax

    # Filter valid data - only rows where onset_time exists (actual onsets, not empty grid rows)
    plot_data = df[['bar_number', 'phase', 'onset_time']].dropna()

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
    # Use phase which shows where the onset is relative to the corrected grid
    for bar_idx in range(n_bars):
        bar_data = plot_data[plot_data['bar_number'] == bar_idx]
        if len(bar_data) > 0:
            phases = bar_data['phase'].values
            color = colors[bar_idx % len(colors)]

            ax.scatter(phases, np.full(len(phases), bar_idx),
                       marker="x", s=18, linewidths=1, color=color)

    # Draw reference onset circles if provided
    if ref_onsets is not None and len(ref_onsets) > 0:
        for _, ref in ref_onsets.iterrows():
            bar_idx = int(ref['bar_number'])
            ref_ms = ref['ref_ms']
            ref_phase = ref['ref_phase']
            grid_phase = ref['grid_phase']
            bar_duration = ref.get('bar_duration', None)

            if 0 <= bar_idx < n_bars:
                # Section anchoring paradigm:
                # After correction, the reference onset IS at grid position (tick 0 = phase 0.0)
                # The grid was shifted by ref_ms to align with the onset

                # Red circle: Where the reference onset IS after correction (at tick 0 = 0.0)
                red_phase_corrected = grid_phase  # 0.0 for tick 0

                # Pink circle: Where tick 0 WAS before correction (in corrected coordinates)
                # The grid moved by ref_phase, so original tick 0 was at -ref_phase
                pink_phase_corrected = -ref_phase if ref_phase is not None and np.isfinite(ref_phase) else None

                # Draw red circle at corrected position
                ax.scatter([red_phase_corrected], [bar_idx],
                           s=90, facecolors='none', edgecolors='red',
                           linewidths=1.5, marker='o', zorder=10)

                # Draw pink circle at original position
                if pink_phase_corrected is not None and np.isfinite(pink_phase_corrected):
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

    ax.set_title(f"Onset raster — Track {track_id} — {title}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def create_anchoring_raster_plots(
    anchoring_dir: str,
    output_file: str,
    track_id: str
) -> str:
    """
    Create raster plots for all anchoring CSVs in a directory.

    One subplot per CSV file found.

    Parameters
    ----------
    anchoring_dir : str
        Path to 6.1_anchoring directory containing CSV files
    output_file : str
        Output PDF/PNG file path
    track_id : str
        Track identifier

    Returns
    -------
    str
        Path to output file
    """
    anchoring_path = Path(anchoring_dir)

    # Find all anchoring CSV files
    csv_files = sorted(anchoring_path.glob('SecNo*_anchored_onsets.csv'))

    if not csv_files:
        print(f"  ! No anchoring CSV files found in {anchoring_dir}")
        return None

    print(f"  Found {len(csv_files)} anchoring CSV files")

    # Determine figure size based on number of plots
    n_plots = len(csv_files)

    # Calculate figure height - each subplot needs space
    # Estimate max bars across all files for consistent sizing
    max_bars = 0
    csv_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, comment='#')
        metadata = parse_csv_metadata(str(csv_file))
        n_bars = int(df['bar_number'].max()) + 1 if 'bar_number' in df.columns else 1
        max_bars = max(max_bars, n_bars)
        csv_data.append((csv_file, df, metadata, n_bars))

    fig_height_per_plot = max(3, min(8, 0.3 * max_bars))
    fig_height = fig_height_per_plot * n_plots

    # Create figure with n subplots
    fig = plt.figure(figsize=(12, fig_height))
    gs = fig.add_gridspec(n_plots, 1, height_ratios=[1] * n_plots, hspace=0.4)
    axes = [fig.add_subplot(gs[i]) for i in range(n_plots)]

    # Plot each CSV
    for idx, (csv_file, df, metadata, n_bars) in enumerate(csv_data):
        # Build title from metadata
        section_label = metadata.get('section_label', 'unknown')
        pattern_len = metadata.get('pattern_length', '?')
        complete_patterns = metadata.get('complete_patterns', '?')
        ratio_in_snippet = metadata.get('ratio_in_snippet', '?')
        anchor_bar = metadata.get('anchor_bar_global', '?')

        title = f"Section: {section_label} | L={pattern_len} | {complete_patterns} patterns | anchor bar={anchor_bar} | ratio={ratio_in_snippet}"

        # Load reference onsets if available
        ref_onsets = None
        ref_file = csv_file.parent / csv_file.name.replace('_anchored_onsets.csv', '_reference_onsets.csv')
        if ref_file.exists():
            ref_onsets = pd.read_csv(ref_file)
            print(f"    Loaded {len(ref_onsets)} reference onsets for {csv_file.name}")

        plot_raster_single(
            axes[idx], df, title, track_id, ref_onsets=ref_onsets
        )

    # Overall title
    fig.suptitle(f"Track {track_id} — Section-Anchored Raster Plots",
                fontsize=13, fontweight="bold")
    plt.subplots_adjust(top=0.97, bottom=0.03, hspace=0.4)

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Anchoring raster plots saved to {output_file}")
    return str(output_file)


def run_anchoring_plots(
    anchoring_dir: str,
    track_id: str,
    output_file: str = None
) -> str:
    """
    Main function to create anchoring raster plots.

    Parameters
    ----------
    anchoring_dir : str
        Path to 6.1_anchoring directory
    track_id : str
        Track identifier
    output_file : str, optional
        Output file path. If None, saves to anchoring_dir/anchoring_raster.pdf

    Returns
    -------
    str
        Path to output file
    """
    if output_file is None:
        output_file = str(Path(anchoring_dir) / f'{track_id}_anchoring_raster.pdf')

    return create_anchoring_raster_plots(anchoring_dir, output_file, track_id)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plots_anchoring.py <anchoring_dir> <track_id> [output_file]")
        print("Example: python plots_anchoring.py /path/to/6.1_anchoring track_123")
        sys.exit(1)

    anchoring_dir = sys.argv[1]
    track_id = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    run_anchoring_plots(anchoring_dir, track_id, output_file)
