"""
Raster plot generation for visualizing section-anchored microtiming phases.

This module creates scatter plots showing onset phases across bars
for section-anchored data from the 6.1_anchoring step.

Environment: AEinBOX_13_3 (matplotlib, numpy, pandas)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict
import re


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


def parse_csv_metadata(csv_path: Path) -> Dict[str, str]:
    """
    Parse metadata from comment lines at the top of CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file

    Returns
    -------
    Dict[str, str]
        Dictionary of metadata key-value pairs
    """
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#'):
                # Parse "# key=value" format
                match = re.match(r'^#\s*(\w+)=(.+)$', line.strip())
                if match:
                    metadata[match.group(1)] = match.group(2)
            else:
                break  # Stop at first non-comment line
    return metadata


def plot_anchoring_raster(
    ax: plt.Axes,
    df: pd.DataFrame,
    metadata: Dict[str, str],
    ref_onsets: Optional[pd.DataFrame] = None
) -> plt.Axes:
    """
    Plot a single raster subplot showing anchored onset phases across bars.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    df : pd.DataFrame
        Anchored CSV data with columns:
        - bar_number: bar index (section-relative, 0-based)
        - bar_number_global: global bar index
        - tick_16th: tick position (0-15)
        - onset_time: onset time in seconds (NaN if no onset)
        - phase: anchored phase value (0 to 1)
    metadata : Dict[str, str]
        Metadata from CSV comments
    ref_onsets : pd.DataFrame, optional
        Reference onsets DataFrame for drawing red circles

    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    # Filter to rows with actual onsets (non-NaN phase)
    plot_data = df[df['phase'].notna()].copy()

    if len(plot_data) == 0:
        ax.text(0.5, 0.5, 'No onset data available',
                ha='center', va='center', transform=ax.transAxes)
        return ax

    # Get number of bars
    n_bars = int(df['bar_number'].max()) + 1

    # Generate bar colors
    colors = make_bar_colors(n_bars)

    # Plot onsets for each bar with "x" markers
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

            if 0 <= bar_idx < n_bars:
                # Red circle at the reference onset position (anchored to 0)
                ax.scatter([0.0], [bar_idx],
                           s=90, facecolors='none', edgecolors='red',
                           linewidths=1.5, marker='o', zorder=10)

                # Pink circle at original uncorrected position
                if ref_phase is not None and np.isfinite(ref_phase):
                    ax.scatter([ref_phase], [bar_idx],
                               s=60, facecolors='none', edgecolors='pink',
                               linewidths=1.2, marker='o', zorder=9, alpha=0.7)

                # Add text label with offset time
                if ref_ms is not None and np.isfinite(ref_ms):
                    ax.text(0.02, bar_idx, f'{ref_ms:.1f}ms',
                            fontsize=6, va='center', ha='left', color='red',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                      edgecolor='red', alpha=0.8, linewidth=0.5))

    # Set axis limits
    ax.set_xlim(-1.0/32.0, 17.0/16.0)
    ax.set_ylim(-0.5, n_bars - 0.5)

    # X-axis: tick positions
    ticks_positions = np.linspace(0, 15.0/16.0, 16)
    ax.set_xticks(ticks_positions)
    ax.set_xticklabels([f"{i}/16" for i in range(1, 17)])

    # Vertical grid lines at 32nd notes
    ticks_grid = np.linspace(0, 31.0/32.0, 32)
    for xg in ticks_grid:
        ax.axvline(xg, color="0.9", linewidth=0.6, zorder=0)

    # Y-axis ticks
    if n_bars > 0:
        tick_step = max(1, n_bars // 10)
        ax.set_yticks(np.arange(0, n_bars, tick_step))

    # Labels
    ax.set_xlabel("bar phase", fontsize=10)
    ax.set_ylabel("bar index (section)", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def create_anchoring_plot(
    anchoring_dir: Path,
    output_file: Path,
    track_id: str
):
    """
    Create a multi-panel raster plot for all anchored sections.

    Parameters
    ----------
    anchoring_dir : Path
        Path to 6.1_anchoring directory containing *_anchored.csv files
    output_file : Path
        Output PNG file path
    track_id : str
        Track identifier
    """
    # Find all anchored CSV files (exclude macOS resource fork files starting with ._)
    anchored_files = sorted([f for f in anchoring_dir.glob("*_anchored.csv") if not f.name.startswith('._')])

    if not anchored_files:
        print(f"  No anchored CSV files found in {anchoring_dir}")
        return

    n_sections = len(anchored_files)
    print(f"  Found {n_sections} anchored sections")

    # Determine figure height based on total bars
    total_bars = 0
    section_data = []

    for csv_file in anchored_files:
        # Parse metadata
        metadata = parse_csv_metadata(csv_file)

        # Load data (skip comment lines)
        df = pd.read_csv(csv_file, comment='#', encoding='utf-8', encoding_errors='replace')
        n_bars = int(df['bar_number'].max()) + 1 if 'bar_number' in df.columns else 1
        total_bars += n_bars

        # Load reference onsets if available
        ref_file = csv_file.parent / csv_file.name.replace('_anchored.csv', '_reference_onsets.csv')
        ref_onsets = None
        if ref_file.exists():
            ref_onsets = pd.read_csv(ref_file, encoding='utf-8', encoding_errors='replace')

        section_data.append({
            'csv_file': csv_file,
            'df': df,
            'metadata': metadata,
            'ref_onsets': ref_onsets,
            'n_bars': n_bars
        })

    # Calculate figure dimensions
    fig_height_per_bar = 0.35
    min_section_height = 2.5
    fig_height = max(10, sum(max(min_section_height, s['n_bars'] * fig_height_per_bar) for s in section_data))

    # Create figure with one subplot per section
    height_ratios = [max(min_section_height, s['n_bars'] * fig_height_per_bar) for s in section_data]
    fig = plt.figure(figsize=(14, fig_height))
    gs = fig.add_gridspec(n_sections, 1, height_ratios=height_ratios, hspace=0.8, top=0.92, bottom=0.05)

    for i, section in enumerate(section_data):
        ax = fig.add_subplot(gs[i])

        # Plot the raster
        plot_anchoring_raster(
            ax,
            section['df'],
            section['metadata'],
            section['ref_onsets']
        )

        # Build title from filename and metadata
        filename = section['csv_file'].stem.replace('_anchored', '')
        meta = section['metadata']

        # Extract section label from metadata (cleaner than filename)
        section_label = meta.get('section_label', '')
        pattern_len = meta.get('pattern_length', '?')
        ratio_in = meta.get('ratio_in_snippet', '?')

        # Build cleaner title: "SecNo1 L2 pre-chorus" format
        # Parse filename: SecNo1_L2_pre-chorus_0.2972 -> SecNo1 L2 pre-chorus
        parts = filename.split('_')
        sec_num = parts[0] if parts else ''  # SecNo1
        pattern_info = parts[1] if len(parts) > 1 else ''  # L2
        title = f"{sec_num} {pattern_info} {section_label}"

        # Extract key metadata for subtitle
        anchor_bar = meta.get('anchor_bar_global', '?')
        pattern_start = meta.get('pattern_start_bar_global', '?')
        snippet_start = meta.get('snippet_start_bar_global', '?')
        section_start = meta.get('section_start_bar_global', '?')
        n_reps = meta.get('no_of_repetitions', '?')

        subtitle = (f"anchor_bar={anchor_bar} | pattern_start={pattern_start} | "
                   f"snippet_start={snippet_start} | section_start={section_start} | "
                   f"pattern_len={pattern_len} | n_reps={n_reps} | ratio={ratio_in}")

        ax.set_title(f"{title}\n{subtitle}", fontsize=9, fontweight='bold', pad=10)

    # Overall title
    fig.suptitle(f"Track {track_id} — Section Anchoring Raster Plots",
                fontsize=12, fontweight="bold", y=1.02)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved anchoring raster plot to {output_file}")


def create_all_anchoring_plots(
    anchoring_dir: str,
    track_id: str
):
    """
    Create raster plots for all anchored sections in a directory.

    Parameters
    ----------
    anchoring_dir : str
        Path to 6.1_anchoring directory
    track_id : str
        Track identifier
    """
    anchoring_path = Path(anchoring_dir)

    if not anchoring_path.exists():
        print(f"  Anchoring directory not found: {anchoring_dir}")
        return

    output_file = anchoring_path / f"{track_id}_section_anchoring_raster.png"
    create_anchoring_plot(anchoring_path, output_file, track_id)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python plots_anchoring.py <anchoring_dir> <track_id>")
        print("Example: python plots_anchoring.py /path/to/6.1_anchoring track_123")
        sys.exit(1)

    anchoring_dir = sys.argv[1]
    track_id = sys.argv[2]

    create_all_anchoring_plots(anchoring_dir, track_id)
