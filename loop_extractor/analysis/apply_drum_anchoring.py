"""
Apply drum anchoring to other stems.

This module takes the filtered drum anchoring data and applies the same
section boundaries and grid structure to other stems (vocals, bass, piano, other).

Multi-Stem Anchoring Strategy
-----------------------------
When processing multiple stems, the drum stem serves as the "anchor" that defines
the musical structure for all other stems. This ensures consistency across stems
and enables meaningful cross-stem comparisons of microtiming.

The drum stem defines:
- Which sections exist (from SongFormer analysis)
- Which bars are kept (after Tukey IQR filtering based on drum onset counts)
- The grid times for each 16th note position (from double anchoring)
- Pattern boundaries (L=1, L=2, L=4 bar patterns)
- Local tempo values per pattern

Other stems then have their onsets mapped to this same grid structure:
1. Load the stem's onset times from 4_onsets/{stem}/
2. For each grid position from drums, search for the nearest onset within ±100ms
3. Calculate phase (deviation from grid) using local tempo
4. Write CSV with identical structure to drum CSV (same bars, grid times, metadata)

Output Structure
----------------
- 6.1_anchoring/drums/ - Full anchoring data (drums only)
- 6.2_filtered_patterns/drums/ - Filtered patterns (drums)
- 6.2_filtered_patterns/{stem}/ - Drum-anchored patterns for other stems

The non-drum stem CSVs have identical grid_time, bar_number, pattern_index etc.,
but different onset_time and phase values based on that stem's detected onsets.

Usage
-----
This module is called automatically by main.py Step 6.1-6.4 when processing
multiple stems (--all-stems flag). Drums must be processed first.

See Also
--------
- PIPELINE_DIAGRAM.md: Multi-Stem Anchoring section for detailed explanation
- config.py: STEMS order (drums first) and ONSET_STEMS configuration
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


def parse_csv_header(csv_path: Path) -> Dict[str, str]:
    """
    Parse metadata from comment lines at the top of CSV file.

    Returns dict of key=value pairs from lines starting with #
    """
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#'):
                match = re.match(r'^#\s*(\w+)=(.+)$', line.strip())
                if match:
                    metadata[match.group(1)] = match.group(2)
            else:
                break
    return metadata


def read_drum_anchored_csv(csv_path: Path) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Read a drum anchored CSV file, returning both metadata and data.

    Parameters
    ----------
    csv_path : Path
        Path to the drum anchored CSV file

    Returns
    -------
    Tuple[Dict, DataFrame]
        (metadata dict, data DataFrame)
    """
    metadata = parse_csv_header(csv_path)

    # Read the data portion (skip comment lines)
    df = pd.read_csv(csv_path, comment='#')

    return metadata, df


def load_bass_f0(stems_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load bass f0 (pitch) data from the stems directory.

    Parameters
    ----------
    stems_dir : Path
        Path to the 1_stems directory

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with 'time' and 'f0_hz' columns, or None if file doesn't exist
    """
    f0_file = stems_dir / 'bass_f0.csv'
    if not f0_file.exists():
        return None

    try:
        df = pd.read_csv(f0_file)
        if 'time' in df.columns and 'f0_hz' in df.columns:
            return df
        return None
    except Exception:
        return None


def find_nearest_f0(onset_time: float, f0_df: pd.DataFrame) -> float:
    """
    Find the nearest f0_hz value for a given onset time.

    Parameters
    ----------
    onset_time : float
        The onset time to find pitch for
    f0_df : pd.DataFrame
        DataFrame with 'time' and 'f0_hz' columns

    Returns
    -------
    float
        The nearest f0_hz value, or 0.0 if not found
    """
    if f0_df is None or len(f0_df) == 0:
        return 0.0

    times = f0_df['time'].values
    f0_values = f0_df['f0_hz'].values

    # Find nearest time
    distances = np.abs(times - onset_time)
    min_idx = np.argmin(distances)

    # Only use if within reasonable range (e.g., 50ms)
    if distances[min_idx] <= 0.05:
        f0 = f0_values[min_idx]
        # Handle NaN values
        if pd.isna(f0):
            return 0.0
        # 55.0 Hz means no pitch detected (silence/no bass)
        if f0 == 55.0:
            return 0.0
        return float(f0)

    return 0.0


def load_stem_onsets(onset_file: Path) -> np.ndarray:
    """
    Load onset times from a stem's onset CSV file.

    Parameters
    ----------
    onset_file : Path
        Path to the onset CSV file (from step 4)

    Returns
    -------
    np.ndarray
        Array of onset times in seconds
    """
    df = pd.read_csv(onset_file)

    # Handle different column names
    if 'onset_times' in df.columns:
        return df['onset_times'].values
    elif 'onset_time' in df.columns:
        return df['onset_time'].values
    else:
        # Try first column
        return df.iloc[:, 0].values


def find_nearest_onset(target_time: float, onsets: np.ndarray,
                       max_distance: float = 0.5) -> Tuple[Optional[float], Optional[float]]:
    """
    Find the nearest onset to a target grid time.

    Parameters
    ----------
    target_time : float
        The grid time to find an onset for
    onsets : np.ndarray
        Array of onset times
    max_distance : float
        Maximum allowed distance (in fraction of grid interval)

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (onset_time, phase) or (None, None) if no onset within range
    """
    if len(onsets) == 0:
        return None, None

    # Find closest onset
    distances = np.abs(onsets - target_time)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    # For now, we don't filter by max_distance - just return closest
    # The phase calculation will show how far off it is
    onset_time = onsets[min_idx]

    return onset_time, min_distance


def apply_drum_anchoring_to_stem(
    drum_filtered_dir: Path,
    stem_onset_file: Path,
    output_dir: Path,
    stem_name: str,
    verbose: bool = False,
    stems_dir: Optional[Path] = None
) -> List[Path]:
    """
    Apply drum anchoring structure to another stem's onsets.

    This reads all filtered drum CSVs and creates corresponding CSVs
    for the target stem using the same section boundaries and grid,
    but with the target stem's onsets.

    Parameters
    ----------
    drum_filtered_dir : Path
        Path to 6.2_filtered_patterns/drums/ directory
    stem_onset_file : Path
        Path to the stem's onset CSV (from step 4)
    output_dir : Path
        Output directory for the stem's filtered patterns
    stem_name : str
        Name of the stem (for logging)
    verbose : bool
        Print progress messages

    Returns
    -------
    List[Path]
        List of created CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load stem onsets
    stem_onsets = load_stem_onsets(stem_onset_file)

    if verbose:
        print(f"    Loaded {len(stem_onsets)} onsets from {stem_name}")

    # Load bass f0 data if this is the bass stem
    bass_f0_df = None
    if stem_name == 'bass' and stems_dir is not None:
        bass_f0_df = load_bass_f0(stems_dir)
        if verbose:
            if bass_f0_df is not None:
                print(f"    Loaded {len(bass_f0_df)} f0 values for bass pitch lookup")
            else:
                print(f"    No bass_f0.csv found in {stems_dir}")

    # Find all drum anchored CSVs (only _anchored.csv, not _reference_onsets.csv)
    # Exclude macOS resource fork files (._*) which cause UnicodeDecodeError
    drum_csvs = sorted([f for f in drum_filtered_dir.glob('*_anchored.csv') if not f.name.startswith('._')])

    if verbose:
        print(f"    Found {len(drum_csvs)} drum anchored CSVs to process")

    created_files = []

    for drum_csv in drum_csvs:
        try:
            # Read drum data
            metadata, drum_df = read_drum_anchored_csv(drum_csv)

            if drum_df.empty:
                continue

            # Get the grid times from drum data
            # These define the exact 16th note positions for each bar
            grid_times = drum_df['grid_time'].values
            bar_numbers = drum_df['bar_number'].values
            bar_numbers_global = drum_df['bar_number_global'].values
            tick_16ths = drum_df['tick_16th'].values
            pattern_indices = drum_df['pattern_index'].values
            pattern_start_times = drum_df['pattern_start_time'].values
            pattern_end_times = drum_df['pattern_end_time'].values
            local_tempos = drum_df['local_tempo'].values
            grid_phases = drum_df['grid_phase'].values

            # Create new dataframe for stem
            stem_rows = []

            for i in range(len(drum_df)):
                grid_time = grid_times[i]

                # Skip if grid_time is NaN
                if pd.isna(grid_time):
                    # Keep the row structure but with empty onset data
                    row = {
                        'bar_number': bar_numbers[i],
                        'bar_number_global': bar_numbers_global[i],
                        'tick_16th': tick_16ths[i],
                        'onset_time': np.nan,
                        'phase': np.nan,
                        'grid_time': grid_time,
                        'grid_phase': grid_phases[i],
                        'tick_phase': np.nan,
                        'pattern_index': pattern_indices[i],
                        'pattern_start_time': pattern_start_times[i],
                        'pattern_end_time': pattern_end_times[i],
                        'local_tempo': local_tempos[i]
                    }
                    # Add f0_hz for bass stem
                    if stem_name == 'bass':
                        row['f0_hz'] = 0.0
                    stem_rows.append(row)
                    continue

                # Find nearest onset from this stem
                # Use a window around the grid time
                # Typically, onsets should be within ~50ms of grid
                window_size = 0.1  # 100ms window

                # Filter onsets within window
                mask = (stem_onsets >= grid_time - window_size) & (stem_onsets <= grid_time + window_size)
                nearby_onsets = stem_onsets[mask]

                if len(nearby_onsets) > 0:
                    # Find closest
                    distances = np.abs(nearby_onsets - grid_time)
                    min_idx = np.argmin(distances)
                    onset_time = nearby_onsets[min_idx]

                    # Calculate phase (deviation from grid)
                    # Get local tempo to calculate beat duration
                    local_tempo = local_tempos[i]
                    if local_tempo > 0:
                        beat_duration = 60.0 / local_tempo
                        tick_duration = beat_duration / 4  # 16th note
                        phase = (onset_time - grid_time) / tick_duration
                    else:
                        phase = np.nan

                    row = {
                        'bar_number': bar_numbers[i],
                        'bar_number_global': bar_numbers_global[i],
                        'tick_16th': tick_16ths[i],
                        'onset_time': onset_time,
                        'phase': phase,
                        'grid_time': grid_time,
                        'grid_phase': grid_phases[i],
                        'tick_phase': phase,  # Same as phase for now
                        'pattern_index': pattern_indices[i],
                        'pattern_start_time': pattern_start_times[i],
                        'pattern_end_time': pattern_end_times[i],
                        'local_tempo': local_tempos[i]
                    }
                    # Add f0_hz for bass stem
                    if stem_name == 'bass':
                        row['f0_hz'] = find_nearest_f0(onset_time, bass_f0_df)
                    stem_rows.append(row)
                else:
                    # No onset found near this grid position
                    row = {
                        'bar_number': bar_numbers[i],
                        'bar_number_global': bar_numbers_global[i],
                        'tick_16th': tick_16ths[i],
                        'onset_time': np.nan,
                        'phase': np.nan,
                        'grid_time': grid_time,
                        'grid_phase': grid_phases[i],
                        'tick_phase': np.nan,
                        'pattern_index': pattern_indices[i],
                        'pattern_start_time': pattern_start_times[i],
                        'pattern_end_time': pattern_end_times[i],
                        'local_tempo': local_tempos[i]
                    }
                    # Add f0_hz for bass stem (0 when no onset)
                    if stem_name == 'bass':
                        row['f0_hz'] = 0.0
                    stem_rows.append(row)

            # Create DataFrame
            stem_df = pd.DataFrame(stem_rows)

            # Write output CSV with same metadata header
            output_csv = output_dir / drum_csv.name

            with open(output_csv, 'w', encoding='utf-8') as f:
                # Write metadata header (same as drum, but note it's derived)
                for key, value in metadata.items():
                    f.write(f'# {key}={value}\n')
                f.write(f'# stem={stem_name}\n')
                f.write(f'# derived_from_drum_anchoring=true\n')

                # Write data
                stem_df.to_csv(f, index=False)

            created_files.append(output_csv)

            # Copy the drum reference_onsets CSV (same grid structure for all stems)
            # The reference onsets define the anchoring grid which is shared across stems
            drum_ref_onsets_file = drum_csv.parent / drum_csv.name.replace('_anchored.csv', '_reference_onsets.csv')
            ref_onsets_name = drum_csv.name.replace('_anchored.csv', '_reference_onsets.csv')
            ref_onsets_path = output_dir / ref_onsets_name

            if drum_ref_onsets_file.exists():
                # Copy drum reference onsets (they define the shared grid)
                import shutil
                shutil.copy2(drum_ref_onsets_file, ref_onsets_path)
                created_files.append(ref_onsets_path)

        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to process {drum_csv.name}: {e}")
            continue

    if verbose:
        print(f"    Created {len(created_files)} files for {stem_name}")

    return created_files


def apply_drum_anchoring_to_all_stems(
    drum_filtered_dir: Path,
    stems: List[str],
    track_dir: Path,
    track_id: str,
    verbose: bool = False
) -> Dict[str, List[Path]]:
    """
    Apply drum anchoring to all non-drum stems.

    Parameters
    ----------
    drum_filtered_dir : Path
        Path to 6.2_filtered_patterns/drums/
    stems : List[str]
        List of stem names to process (should exclude 'drums')
    track_dir : Path
        Base track directory
    track_id : str
        Track identifier
    verbose : bool
        Print progress

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping stem name to list of created files
    """
    results = {}

    for stem in stems:
        if stem == 'drums':
            continue

        # Get paths
        stem_onset_file = track_dir / '4_onsets' / stem / f'{track_id}_onsets.csv'
        output_dir = track_dir / '6.2_filtered_patterns' / stem

        if not stem_onset_file.exists():
            if verbose:
                print(f"  Skipping {stem}: no onset file found")
            continue

        if verbose:
            print(f"  Applying drum anchoring to {stem}...")

        # Get stems_dir for bass f0 lookup
        stems_dir = track_dir / '1_stems'

        created_files = apply_drum_anchoring_to_stem(
            drum_filtered_dir=drum_filtered_dir,
            stem_onset_file=stem_onset_file,
            output_dir=output_dir,
            stem_name=stem,
            verbose=verbose,
            stems_dir=stems_dir
        )

        results[stem] = created_files

    return results
