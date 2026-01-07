#!/usr/bin/env python3
"""
LEPA Export - Per-track bar duration export for LEPA analysis.

This module exports bar duration data and audio clips for a single track.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf


def export_bar_durations(
    comprehensive_csv: str,
    track_id: str,
    output_dir: str,
    audio_file: str = None,
    drum_stem_file: str = None
):
    """
    Export bar duration table and full audio for LEPA analysis from a single track.

    Creates:
    - CSV file with bar timing data
    - Full audio WAV containing all complete bars (if audio_file provided)
    - Drum stem WAV containing all complete bars (if drum_stem_file provided)

    Audio files are trimmed to start at the first bar and end at the last complete bar,
    with 0.05s fade in/out applied.

    CSV columns:
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
    audio_file : str, optional
        Path to original audio file (for exporting full audio)
    drum_stem_file : str, optional
        Path to drum stem file (for exporting drum stem)

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
                        # Use the uncorrected grid time for bar 0, tick 0 (the downbeat)
                        first_bar_row = df[(df['bar_number'] == 0) & (df['tick_16th'] == 0)]
                        if not first_bar_row.empty:
                            first_bar_start_ms = float(first_bar_row.iloc[0]['grid_time_uncorrected'] * 1000.0)
                        else:
                            # Fallback: use minimum grid time
                            first_bar_start_ms = float(df['grid_time_uncorrected'].min() * 1000.0)

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

        # Export full audio containing only complete bars if audio files provided
        if audio_file is not None or drum_stem_file is not None:
            print(f"  Exporting full audio (complete bars only)...")

            # Calculate time range for all complete bars
            first_bar_start_s = bars['grid_time_per_snippet'].iloc[0]
            last_bar_start_s = bars['grid_time_per_snippet'].iloc[-1]
            # End of last bar = start + median bar duration
            last_bar_end_s = last_bar_start_s + np.median(np.diff(bars['grid_time_per_snippet'].values))

            # Fade duration in samples (0.05 seconds)
            sr = 44100
            fade_samples = int(0.05 * sr)

            # Export original audio
            if audio_file is not None and Path(audio_file).exists():
                print(f"    Loading original audio from {audio_file}...")
                audio_data, sr = librosa.load(audio_file, sr=sr, mono=True)

                # Extract region containing all complete bars
                start_sample = int(first_bar_start_s * sr)
                end_sample = int(last_bar_end_s * sr)
                audio_full_bars = audio_data[start_sample:end_sample]

                # Apply fade in/out
                if len(audio_full_bars) > 2 * fade_samples:
                    # Fade in (cosine curve: 0 -> 1)
                    fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
                    audio_full_bars[:fade_samples] *= fade_in

                    # Fade out (cosine curve: 1 -> 0)
                    fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))
                    audio_full_bars[-fade_samples:] *= fade_out

                # Export in same folder as CSV
                output_file = output_path / f'{song_id}_audio.wav'
                sf.write(str(output_file), audio_full_bars, sr)
                print(f"    ✓ Exported full audio: {output_file.name}")

            # Export drum stem
            if drum_stem_file is not None and Path(drum_stem_file).exists():
                print(f"    Loading drum stem from {drum_stem_file}...")
                drum_data, sr = librosa.load(drum_stem_file, sr=sr, mono=True)

                # Extract region containing all complete bars
                start_sample = int(first_bar_start_s * sr)
                end_sample = int(last_bar_end_s * sr)
                drum_full_bars = drum_data[start_sample:end_sample]

                # Apply fade in/out
                if len(drum_full_bars) > 2 * fade_samples:
                    # Fade in (cosine curve: 0 -> 1)
                    fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
                    drum_full_bars[:fade_samples] *= fade_in

                    # Fade out (cosine curve: 1 -> 0)
                    fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))
                    drum_full_bars[-fade_samples:] *= fade_out

                # Export in same folder as CSV
                output_file = output_path / f'{song_id}_drums.wav'
                sf.write(str(output_file), drum_full_bars, sr)
                print(f"    ✓ Exported drum stem: {output_file.name}")

        return str(output_csv)

    except Exception as e:
        print(f"  Warning: Could not export LEPA data: {e}")
        import traceback
        traceback.print_exc()
        return None
