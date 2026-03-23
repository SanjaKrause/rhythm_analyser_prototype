"""
Raster plot generation for visualizing section-anchored microtiming phases.

This module creates scatter plots showing onset phases across bars
for section-anchored data from the 6.1_anchoring step.

Three subplot types per section:
1. Double Anchored (original) - uses phase from 6.1_anchoring (reference circles shown)
2. Uncorrected - uses phase_uncorrected from 5_grid/*_comprehensive_phases.csv (step 5)
3. Per-Section Corrected - uncorrected phases shifted by one reference onset per section

Note: For the Uncorrected and Per-Section Corrected plots, this module loads data from
step 5 (5_grid/*_comprehensive_phases.csv) and matches bars using bar_number_global.
The 5_grid CSV must include the bar_number_global column (added in raster.py).

KNOWN LIMITATION - Sections Starting Before Snippet:
----------------------------------------------------
Some sections may show "No onset data available" in the Uncorrected and Per-Section
Corrected plots. This is expected behavior when:

- The section anchoring (6.1_anchoring) uses bars that start BEFORE the 30-second snippet
- The 5_grid CSV only contains bars that are WITHIN the snippet window
- When bar_number_global values don't overlap between these two data sources,
  there is no uncorrected data to display

Example:
  - Section "pre-chorus" anchored CSV has bar_number_global = 29
  - 5_grid comprehensive_phases.csv starts at bar_number_global = 30
  - Bar 29 has no uncorrected data because it's outside the snippet

The Double Anchored plot (left column) will still show data because it uses
the anchoring data directly, which includes bars before the snippet.

Environment: AEinBOX_13_3 (matplotlib, numpy, pandas)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
import glob as glob_module


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


def load_uncorrected_phases(track_dir: Path, track_id: str) -> Optional[pd.DataFrame]:
    """
    Load uncorrected phases from 5_grid/drums/*_comprehensive_phases.csv.

    Parameters
    ----------
    track_dir : Path
        Path to the track directory (parent of 6.1_anchoring)
    track_id : str
        Track identifier

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with columns: bar_number, tick_16th, onset_time_uncorrected, phase_uncorrected
        Returns None if file not found
    """
    # Find the comprehensive phases CSV in 5_grid
    grid_dir = track_dir / '5_grid'
    if not grid_dir.exists():
        return None

    # Look for *_comprehensive_phases.csv - try drums subfolder first (new structure),
    # then fall back to root 5_grid folder (old structure)
    matches = []

    # New structure: 5_grid/drums/
    drums_dir = grid_dir / 'drums'
    if drums_dir.exists():
        pattern = str(drums_dir / '*_comprehensive_phases.csv')
        matches = glob_module.glob(pattern)

    # Fall back to old structure: 5_grid/
    if not matches:
        pattern = str(grid_dir / '*_comprehensive_phases.csv')
        matches = glob_module.glob(pattern)

    if not matches:
        return None

    csv_path = Path(matches[0])

    try:
        df = pd.read_csv(csv_path)

        # Keep only the columns we need
        required_cols = ['bar_number', 'tick_16th']
        # bar_number_global is required for matching with anchoring data
        optional_cols = ['bar_number_global', 'onset_time_uncorrected', 'phase_uncorrected']

        # Check required columns exist
        if not all(col in df.columns for col in required_cols):
            return None

        # Select columns that exist
        cols_to_keep = required_cols + [c for c in optional_cols if c in df.columns]
        return df[cols_to_keep]

    except Exception as e:
        print(f"  Warning: Failed to load uncorrected phases: {e}")
        return None


def get_uncorrected_phases_for_section(
    uncorrected_df: pd.DataFrame,
    anchored_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Extract uncorrected phases for bars in an anchored section.

    Matches bars using bar_number_global from both DataFrames.
    The 5_grid CSV must have bar_number_global column (added in raster.py).

    Parameters
    ----------
    uncorrected_df : pd.DataFrame
        Full uncorrected phases from 5_grid CSV (with bar_number_global)
    anchored_df : pd.DataFrame
        Anchored section data with bar_number_global column

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with section-relative bar_number and uncorrected phases,
        or None if no matching data found
    """
    if uncorrected_df is None:
        return None

    if 'bar_number_global' not in anchored_df.columns:
        return None

    # Check if 5_grid CSV has bar_number_global (new format)
    if 'bar_number_global' not in uncorrected_df.columns:
        # Old format CSV without bar_number_global - cannot match
        return None

    # Get unique global bar numbers in this section
    global_bars = sorted(anchored_df['bar_number_global'].unique())

    # Build result rows
    result_rows = []

    for section_bar_idx, global_bar in enumerate(global_bars):
        # Get uncorrected data for this global bar using bar_number_global
        uncorr_bar_data = uncorrected_df[uncorrected_df['bar_number_global'] == global_bar]

        for _, row in uncorr_bar_data.iterrows():
            result_rows.append({
                'bar_number': section_bar_idx,  # section-relative
                'bar_number_global': global_bar,
                'tick_16th': row['tick_16th'],
                'onset_time': row.get('onset_time_uncorrected', np.nan),
                'phase': row.get('phase_uncorrected', np.nan)
            })

    if not result_rows:
        return None

    return pd.DataFrame(result_rows)


def find_section_reference_offset(
    df: pd.DataFrame,
    search_window_phase: float = 0.03  # ~half a 16th note
) -> Tuple[float, int]:
    """
    Find reference offset for per-section correction.

    Searches for the first onset near tick 0 (downbeat) position.
    If not found in bar 0, checks bar 1, bar 2, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Anchored CSV data with 'bar_number', 'tick_16th', 'phase' columns
    search_window_phase : float
        Search window around tick 0 in phase units (default ~half a 16th note)

    Returns
    -------
    Tuple[float, int]
        (reference offset in phase units, bar index where reference was found)
    """
    n_bars = int(df['bar_number'].max()) + 1 if len(df) > 0 else 0

    for bar_idx in range(n_bars):
        # Get onsets in this bar
        bar_data = df[(df['bar_number'] == bar_idx) & (df['phase'].notna())]

        if len(bar_data) == 0:
            continue

        # Look for onsets near tick 0 (phase close to 0.0)
        # Tick 0 should have phase near 0.0 (or near 1.0 if it wrapped)
        tick0_candidates = bar_data[
            (bar_data['phase'].abs() <= search_window_phase) |
            ((1.0 - bar_data['phase']).abs() <= search_window_phase)
        ]

        if len(tick0_candidates) > 0:
            # Take the one closest to phase 0
            closest_idx = tick0_candidates['phase'].abs().idxmin()
            ref_phase = tick0_candidates.loc[closest_idx, 'phase']
            return ref_phase, bar_idx

    # No reference found - return 0 offset
    return 0.0, 0


def plot_anchoring_raster(
    ax: plt.Axes,
    df: pd.DataFrame,
    metadata: Dict[str, str],
    ref_onsets: Optional[pd.DataFrame] = None,
    show_corrected: bool = False,
    ref_offset: float = 0.0,
    ref_bar: int = 0
) -> plt.Axes:
    """
    Plot a single raster subplot showing onset phases across bars.

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
        - phase: bar-normalized phase value (0 to 1)
    metadata : Dict[str, str]
        Metadata from CSV comments
    ref_onsets : pd.DataFrame, optional
        Reference onsets DataFrame for drawing red circles
    show_corrected : bool
        If True, apply per-section correction (shift phases by ref_offset)
    ref_offset : float
        Reference offset in phase units for per-section correction
    ref_bar : int
        Bar index where reference was found

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

    # Get global bar numbers for secondary y-axis (if available)
    global_bar_map = {}
    if 'bar_number_global' in df.columns:
        for bar_idx in range(n_bars):
            bar_rows = df[df['bar_number'] == bar_idx]
            if len(bar_rows) > 0:
                global_bar_map[bar_idx] = int(bar_rows['bar_number_global'].iloc[0])

    # Generate bar colors
    colors = make_bar_colors(n_bars)

    # Plot onsets for each bar with "x" markers
    for bar_idx in range(n_bars):
        bar_data = plot_data[plot_data['bar_number'] == bar_idx]
        if len(bar_data) > 0:
            phases = bar_data['phase'].values.copy()

            # Apply per-section correction if requested
            if show_corrected:
                phases = phases - ref_offset

            color = colors[bar_idx % len(colors)]
            ax.scatter(phases, np.full(len(phases), bar_idx),
                       marker="x", s=18, linewidths=1, color=color)

    # Draw reference onset circles if provided (original behavior)
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

    # Draw per-section correction reference circle (for corrected view only)
    elif show_corrected and ref_offset != 0.0:
        # Red circle at tick 0 position (where reference is after correction)
        ax.scatter([0.0], [ref_bar],
                   s=90, facecolors='none', edgecolors='red',
                   linewidths=1.5, marker='o', zorder=10)

        # Add text label with offset info
        ax.text(0.02, ref_bar, f'ref: {ref_offset:.3f}',
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

    # Y-axis ticks (left side - section-relative)
    if n_bars > 0:
        tick_step = max(1, n_bars // 10)
        ax.set_yticks(np.arange(0, n_bars, tick_step))

    # Labels
    ax.set_xlabel("bar phase", fontsize=10)
    ax.set_ylabel("bar index (section)", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Secondary y-axis on the right with global bar indices
    if global_bar_map:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())

        # Create tick positions and labels for global bars
        tick_positions = []
        tick_labels = []
        tick_step = max(1, n_bars // 10)
        for bar_idx in range(0, n_bars, tick_step):
            if bar_idx in global_bar_map:
                tick_positions.append(bar_idx)
                tick_labels.append(str(global_bar_map[bar_idx]))

        ax2.set_yticks(tick_positions)
        ax2.set_yticklabels(tick_labels)
        ax2.set_ylabel("bar index (global)", fontsize=10)

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

    # Load uncorrected phases from 5_grid
    # anchoring_dir is typically /path/to/track/6.1_anchoring/drums
    # We need to go up to track directory
    track_dir = anchoring_dir.parent.parent
    uncorrected_df = load_uncorrected_phases(track_dir, track_id)
    if uncorrected_df is not None:
        if 'bar_number_global' in uncorrected_df.columns:
            print(f"  Loaded uncorrected phases from 5_grid ({len(uncorrected_df)} rows)")
        else:
            print(f"  Warning: 5_grid CSV missing bar_number_global column (re-run step 5 to regenerate)")
            uncorrected_df = None
    else:
        print(f"  Warning: Could not load uncorrected phases from 5_grid")

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
        # Reference onsets define the anchoring grid (shared across stems from drums)
        ref_file = csv_file.parent / csv_file.name.replace('_anchored.csv', '_reference_onsets.csv')
        ref_onsets = None
        if ref_file.exists():
            ref_df = pd.read_csv(ref_file, encoding='utf-8', encoding_errors='replace')
            # Only use if it has the required columns for plotting reference circles
            required_cols = ['bar_number', 'ref_ms', 'ref_phase']
            if all(col in ref_df.columns for col in required_cols):
                ref_onsets = ref_df

        # Get uncorrected phases for this section
        uncorrected_section_df = None
        if uncorrected_df is not None:
            uncorrected_section_df = get_uncorrected_phases_for_section(uncorrected_df, df)

        section_data.append({
            'csv_file': csv_file,
            'df': df,
            'metadata': metadata,
            'ref_onsets': ref_onsets,
            'n_bars': n_bars,
            'uncorrected_df': uncorrected_section_df
        })

    # Group sections by unique section identifier (SecNo + section_label)
    # We want one row per unique section, showing all L variants in left column
    # and only ONE uncorrected/per-section pair in right column (from L1)
    unique_sections = {}  # key: (sec_no, section_label) -> list of section_data entries
    for s in section_data:
        meta = s['metadata']
        filename = s['csv_file'].stem.replace('_anchored', '')
        parts = filename.split('_')
        sec_no = parts[0] if parts else 'Unknown'
        section_label = meta.get('section_label', '')
        key = (sec_no, section_label)

        if key not in unique_sections:
            unique_sections[key] = []
        unique_sections[key].append(s)

    # Sort by sec_no
    sorted_section_keys = sorted(unique_sections.keys(), key=lambda x: x[0])
    n_unique_sections = len(sorted_section_keys)

    # Calculate figure dimensions
    # Layout: 2 columns per unique section
    #   - Left column: All L variants stacked (Double Anchored)
    #   - Right column: 2 subplots (Uncorrected + Per-Section Corrected) using L1 data
    fig_height_per_bar = 0.35
    min_section_height = 2.5

    # Calculate height for each unique section
    height_ratios = []
    for key in sorted_section_keys:
        variants = unique_sections[key]
        # Height based on total bars across all L variants
        total_bars = sum(s['n_bars'] for s in variants)
        section_height = max(min_section_height * len(variants), total_bars * fig_height_per_bar)
        height_ratios.append(section_height)

    fig_height = max(10, sum(height_ratios))

    # Create figure with 2 columns
    fig = plt.figure(figsize=(24, fig_height))
    gs = fig.add_gridspec(n_unique_sections, 2, height_ratios=height_ratios, hspace=0.4, wspace=0.12,
                          top=0.96, bottom=0.02, left=0.04, right=0.98)

    for section_idx, key in enumerate(sorted_section_keys):
        variants = unique_sections[key]
        sec_no, section_label = key

        # Find L1 variant for uncorrected/per-section plots (or first available)
        l1_variant = None
        for v in variants:
            if '_L1_' in v['csv_file'].name:
                l1_variant = v
                break
        if l1_variant is None:
            l1_variant = variants[0]

        # LEFT COLUMN: Create subgrid for all L variants
        n_variants = len(variants)
        gs_left = gs[section_idx, 0].subgridspec(n_variants, 1, hspace=0.25)

        for var_idx, section in enumerate(variants):
            filename = section['csv_file'].stem.replace('_anchored', '')
            meta = section['metadata']
            pattern_len = meta.get('pattern_length', '?')
            ratio_in = meta.get('ratio_in_snippet', '?')

            parts = filename.split('_')
            pattern_info = parts[1] if len(parts) > 1 else ''
            base_title = f"{sec_no} {pattern_info} {section_label}"

            anchor_bar = meta.get('anchor_bar_global', '?')
            n_reps = meta.get('no_of_repetitions', '?')
            subtitle = f"L={pattern_len} | reps={n_reps} | ratio={ratio_in}"

            ax_left = fig.add_subplot(gs_left[var_idx])
            plot_anchoring_raster(
                ax_left,
                section['df'],
                section['metadata'],
                section['ref_onsets']
            )
            ax_left.set_title(f"{base_title} — Double Anchored | {subtitle}",
                             fontsize=9, fontweight='bold', pad=8)

        # RIGHT COLUMN: 2 subplots (Uncorrected + Per-Section Corrected) using L1 data
        gs_right = gs[section_idx, 1].subgridspec(2, 1, hspace=0.25)

        uncorrected_section_df = l1_variant.get('uncorrected_df')
        meta = l1_variant['metadata']
        base_title = f"{sec_no} {section_label}"

        # Find per-section reference offset
        if uncorrected_section_df is not None and len(uncorrected_section_df) > 0:
            ref_offset, ref_bar = find_section_reference_offset(uncorrected_section_df)
        else:
            ref_offset, ref_bar = 0.0, 0

        # Right top: Uncorrected
        ax_right_top = fig.add_subplot(gs_right[0])
        if uncorrected_section_df is not None and len(uncorrected_section_df) > 0:
            plot_anchoring_raster(
                ax_right_top,
                uncorrected_section_df,
                meta,
                ref_onsets=None,
                show_corrected=False
            )
            ax_right_top.set_title(f"{base_title} — Uncorrected",
                                   fontsize=9, fontweight='bold', pad=8)
        else:
            ax_right_top.text(0.5, 0.5, 'No uncorrected data available',
                             ha='center', va='center', transform=ax_right_top.transAxes)
            ax_right_top.set_title(f"{base_title} — Uncorrected (no data)",
                                   fontsize=9, fontweight='bold', pad=8)

        # Right bottom: Per-Section Corrected
        ax_right_bottom = fig.add_subplot(gs_right[1])
        if uncorrected_section_df is not None and len(uncorrected_section_df) > 0:
            plot_anchoring_raster(
                ax_right_bottom,
                uncorrected_section_df,
                meta,
                ref_onsets=None,
                show_corrected=True,
                ref_offset=ref_offset,
                ref_bar=ref_bar
            )
            ax_right_bottom.set_title(f"{base_title} — Per-Section Corrected (offset={ref_offset:.3f})",
                                      fontsize=9, fontweight='bold', pad=8)
        else:
            ax_right_bottom.text(0.5, 0.5, 'No uncorrected data available',
                                ha='center', va='center', transform=ax_right_bottom.transAxes)
            ax_right_bottom.set_title(f"{base_title} — Per-Section Corrected (no data)",
                                      fontsize=9, fontweight='bold', pad=8)

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
