#!/usr/bin/env python3
"""
LEPA Data Export - Extract and export bar duration data for LEPA analysis.

This script reads comprehensive CSV files from all tracks and creates a table with:
- Bar number (Taktnummer)
- Snippet ID (track_id)
- Bar start time in ms (Taktbeginn) - with first onset at 0ms
- Bar duration in ms (Taktdauer)

Uses the per-snippet corrected grid times where the first onset is at 0ms.

Usage:
    python lepa_data_export.py /path/to/batch/output
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def export_lepa_data(output_dir: Path):
    """
    Export bar duration tables for LEPA analysis from all comprehensive CSV files.

    Creates three CSV files, one for each pattern length:
    - Pattern length 1: Each bar is its own pattern
    - Pattern length 2: Bars folded into 2-bar patterns
    - Pattern length 4: Bars folded into 4-bar patterns

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    """
    print("\n" + "=" * 80)
    print("LEPA DATA EXPORT - Bar Duration Tables (Pattern Lengths 1, 2, 4)")
    print("=" * 80)

    # Load overview CSV for snippet start times
    overview_csv_path = Path('/Users/alexk/mastab/main_project/AP_1/corrected_shift_results.csv')
    overview_df = None

    if overview_csv_path.exists():
        try:
            overview_df = pd.read_csv(overview_csv_path, sep=';')
            print(f"Loaded overview CSV with {len(overview_df)} entries")
        except Exception as e:
            print(f"Warning: Could not load overview CSV: {e}")
    else:
        print("Warning: Overview CSV not found - snippet metadata will be missing")

    # Find all track directories
    track_dirs = sorted([d for d in output_dir.iterdir()
                        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis', 'batch_processed_output_for_lepa']])

    if not track_dirs:
        print('No track directories found!')
        return

    print(f"Found {len(track_dirs)} track directories")

    # Collect bar data from all tracks for each pattern length
    bar_data_per_snippet = []  # Per-snippet (original format)
    bar_data_L1 = []  # Pattern length 1
    bar_data_L2 = []  # Pattern length 2
    bar_data_L4 = []  # Pattern length 4
    tracks_processed = 0

    for track_dir in track_dirs:
        # Look for comprehensive CSV - it has the track name as prefix
        grid_folder = track_dir / '5_grid'
        if not grid_folder.exists():
            continue

        # Find the comprehensive CSV file
        comprehensive_csvs = list(grid_folder.glob('*_comprehensive_phases.csv'))
        if not comprehensive_csvs:
            continue

        comprehensive_csv = comprehensive_csvs[0]

        try:
            df = pd.read_csv(comprehensive_csv)

            # Split track name into song_id and song_info first
            # Format: "8_Castle on the Hill - Ed Sheeran" -> song_id="8", song_info="Castle on the Hill - Ed Sheeran"
            track_name = track_dir.name
            if '_' in track_name:
                song_id, song_info = track_name.split('_', 1)
            else:
                song_id = track_name
                song_info = track_name

            # Get snippet metadata from overview CSV
            snippet_start_ms = None
            first_bar_start_ms = None

            if overview_df is not None:
                try:
                    song_id_int = int(song_id)
                    matching_row = overview_df[overview_df['song_id'] == song_id_int]

                    if not matching_row.empty:
                        snippet_start_ms = float(matching_row.iloc[0]['corrected offset (ms)'])

                        # Get first bar start time from comprehensive CSV
                        # Use the uncorrected grid time for bar 0, tick 0 (the downbeat)
                        first_bar_row = df[(df['bar_number'] == 0) & (df['tick_16th'] == 0)]
                        if not first_bar_row.empty:
                            first_bar_start_ms = float(first_bar_row.iloc[0]['grid_time_uncorrected'] * 1000.0)
                        else:
                            # Fallback: use minimum grid time
                            first_bar_start_ms = float(df['grid_time_uncorrected'].min() * 1000.0)

                except (ValueError, KeyError) as e:
                    print(f"    Warning: Could not extract metadata for {track_name}: {e}")

            # Use grid_time_per_snippet which has first onset at 0ms
            # Get unique bars and their grid times
            bars = df.groupby('bar_number').agg({
                'grid_time_per_snippet': 'first'  # Get the grid time at the start of each bar
            }).reset_index()

            # Sort by bar number
            bars = bars.sort_values('bar_number')

            # Calculate bar start times (convert to ms)
            bar_starts_ms = bars['grid_time_per_snippet'].values * 1000.0

            # Subtract the first bar's start time to make bar 0 start at 0ms
            bar_starts_ms = bar_starts_ms - bar_starts_ms[0]

            # Calculate bar durations (difference between consecutive bar starts)
            bar_durations_ms = np.diff(bar_starts_ms, append=bar_starts_ms[-1] + np.median(np.diff(bar_starts_ms)))

            # Only include songs with 12 or more bars
            num_bars = len(bars)
            if num_bars < 12:
                print(f"  Skipped: {track_dir.name} (only {num_bars} bars, need >= 12)")
                continue

            # First, collect per-snippet data (original format without pattern folding)
            for idx, row in bars.iterrows():
                bar_num = int(row['bar_number'])
                bar_start = bar_starts_ms[bars.index.get_loc(idx)]
                bar_duration = bar_durations_ms[bars.index.get_loc(idx)]

                bar_data_per_snippet.append({
                    'Taktnummer': bar_num,
                    'Song_ID': song_id,
                    'Song_Info': song_info,
                    'Snippet_Start_ms': snippet_start_ms,
                    'First_Bar_Start_ms': first_bar_start_ms,
                    'Taktbeginn_ms': bar_start,
                    'Taktdauer_ms': bar_duration
                })

            # Process for each pattern length: 1, 2, 4 with pattern-relative timing
            for pattern_length in [1, 2, 4]:
                # Determine which list to append to
                if pattern_length == 1:
                    target_list = bar_data_L1
                elif pattern_length == 2:
                    target_list = bar_data_L2
                else:  # pattern_length == 4
                    target_list = bar_data_L4

                # Create records for this track with pattern-relative timing
                for idx, row in bars.iterrows():
                    bar_num = int(row['bar_number'])
                    bar_start_absolute = bar_starts_ms[bars.index.get_loc(idx)]
                    bar_duration = bar_durations_ms[bars.index.get_loc(idx)]

                    # Calculate pattern index and position within pattern
                    pattern_index = bar_num // pattern_length
                    bar_within_pattern = bar_num % pattern_length

                    # Calculate pattern-relative bar start time
                    # Find the start of this pattern (first bar in the pattern)
                    pattern_first_bar = pattern_index * pattern_length
                    pattern_start_absolute = bar_starts_ms[bars.index.get_loc(bars.index[pattern_first_bar])]
                    bar_start_relative = bar_start_absolute - pattern_start_absolute

                    target_list.append({
                        'Taktnummer': bar_num,
                        'Pattern_Length': pattern_length,
                        'Pattern_Index': pattern_index,
                        'Bar_In_Pattern': bar_within_pattern,
                        'Song_ID': song_id,
                        'Song_Info': song_info,
                        'Snippet_Start_ms': snippet_start_ms,
                        'First_Bar_Start_ms': first_bar_start_ms,
                        'Taktbeginn_ms': bar_start_relative,
                        'Taktdauer_ms': bar_duration
                    })

            tracks_processed += 1
            print(f"  Read: {track_dir.name} ({num_bars} bars)")

        except Exception as e:
            print(f"  Warning: Could not process {comprehensive_csv}: {e}")
            import traceback
            traceback.print_exc()

    if tracks_processed == 0:
        print("No valid data found!")
        return

    print(f"\nSuccessfully processed {tracks_processed} tracks")

    # Create batch_processed_output_for_lepa directory
    lepa_dir = output_dir / 'batch_processed_output_for_lepa'
    lepa_dir.mkdir(parents=True, exist_ok=True)

    # Export per-snippet CSV (original format)
    if bar_data_per_snippet:
        result_df = pd.DataFrame(bar_data_per_snippet)
        print(f"\nPer-Snippet (original format): {len(result_df)} bars")

        output_csv = lepa_dir / 'batch_bar_durations.csv'
        result_df.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")
        print(f"  Columns: {list(result_df.columns)}")

        # Show sample of data
        print(f"  Sample data (first 10 rows):")
        print(result_df.head(10).to_string(index=False))
    else:
        print("Warning: No per-snippet data")

    # Export three CSV files, one for each pattern length
    for pattern_length, bar_data in [(1, bar_data_L1), (2, bar_data_L2), (4, bar_data_L4)]:
        if not bar_data:
            print(f"Warning: No data for pattern length {pattern_length}")
            continue

        result_df = pd.DataFrame(bar_data)
        print(f"\nPattern Length {pattern_length}: {len(result_df)} bars")

        output_csv = lepa_dir / f'batch_bar_durations_L{pattern_length}.csv'
        result_df.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")
        print(f"  Columns: {list(result_df.columns)}")

        # Show sample of data
        print(f"  Sample data (first 10 rows):")
        print(result_df.head(10).to_string(index=False))

    print("\n" + "=" * 80)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python lepa_data_export.py /path/to/batch/output')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f'Error: Directory does not exist: {output_dir}')
        sys.exit(1)

    export_lepa_data(output_dir)
