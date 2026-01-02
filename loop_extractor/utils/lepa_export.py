#!/usr/bin/env python3
"""
LEPA Export - Per-track bar duration export for LEPA analysis.

This module exports bar duration data for a single track.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def export_bar_durations(comprehensive_csv: str, track_id: str, output_dir: str):
    """
    Export bar duration table for LEPA analysis from a single track.

    Creates a CSV file with:
    - Taktnummer: Bar number (0-based)
    - Song_ID: Numeric song identifier
    - Song_Info: Song name and artist
    - Snippet_Start_ms: Uncorrected snippet start time from overview file
    - First_Bar_Start_ms: Timestamp of first bar onset (the 0ms reference)
    - Taktbeginn_ms: Bar start time in milliseconds (bar 0 starts at 0ms)
    - Taktdauer_ms: Bar duration in milliseconds

    Parameters
    ----------
    comprehensive_csv : str
        Path to the comprehensive phases CSV file
    track_id : str
        Track identifier
    output_dir : str
        Output directory for the LEPA export

    Returns
    -------
    str
        Path to the created CSV file, or None if failed
    """
    try:
        # Load comprehensive CSV
        df = pd.read_csv(comprehensive_csv)

        # Load snippet start time from overview CSV
        snippet_start_ms = None
        first_bar_start_ms = None

        from pathlib import Path as PathLib
        overview_csv_path = PathLib('/Users/alexk/mastab/main_project/AP_1/corrected_shift_results.csv')

        if overview_csv_path.exists():
            try:
                overview_df = pd.read_csv(overview_csv_path, sep=';')

                # Extract song_id from track_id (e.g., "8_Castle on the Hill - Ed Sheeran" -> 8)
                if '_' in track_id:
                    song_id_str = track_id.split('_', 1)[0]
                else:
                    song_id_str = track_id

                # Try to convert to int and match
                try:
                    song_id_int = int(song_id_str)
                    matching_row = overview_df[overview_df['song_id'] == song_id_int]

                    if not matching_row.empty:
                        snippet_start_ms = float(matching_row.iloc[0]['corrected offset (ms)'])

                        # Get first bar start time from comprehensive CSV
                        # This is the first onset time (uncorrected)
                        first_onset_time = df['onset_time'].min()
                        first_bar_start_ms = float(first_onset_time * 1000.0)

                except (ValueError, KeyError):
                    pass
            except Exception as e:
                print(f"    Warning: Could not load snippet start from overview CSV: {e}")

        if snippet_start_ms is None or first_bar_start_ms is None:
            print(f"    Warning: Could not find snippet metadata for track {track_id}")

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

        # Split track_id into song_id and song_info
        # Format: "8_Castle on the Hill - Ed Sheeran" -> song_id="8", song_info="Castle on the Hill - Ed Sheeran"
        if '_' in track_id:
            song_id, song_info = track_id.split('_', 1)
        else:
            song_id = track_id
            song_info = track_id

        # Create records for this track
        bar_data = []
        for idx, row in bars.iterrows():
            bar_num = int(row['bar_number'])
            bar_start = bar_starts_ms[bars.index.get_loc(idx)]
            bar_duration = bar_durations_ms[bars.index.get_loc(idx)]

            bar_data.append({
                'Taktnummer': bar_num,
                'Song_ID': song_id,
                'Song_Info': song_info,
                'Snippet_Start_ms': snippet_start_ms,
                'First_Bar_Start_ms': first_bar_start_ms,
                'Taktbeginn_ms': bar_start,
                'Taktdauer_ms': bar_duration
            })

        # Create DataFrame
        result_df = pd.DataFrame(bar_data)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export bar duration table
        output_csv = output_path / f'{track_id}_bar_durations.csv'
        result_df.to_csv(output_csv, index=False)

        return str(output_csv)

    except Exception as e:
        print(f"  Warning: Could not export LEPA data: {e}")
        return None
