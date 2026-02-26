#!/usr/bin/env python3
"""
Step 11.1: Section Extraction

Extracts audio sections from the original audio file based on filtered pattern boundaries.
Uses used_section_start and used_section_end from Step 6.2 filtered patterns.

Input: 6.2_filtered_patterns CSVs with used_section_start/used_section_end metadata
Output: 9.1_sections/ folder with extracted WAV files per section

Environment: loop_extractor_main (uses librosa, soundfile)
"""

import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_metadata(csv_path: Path) -> Dict[str, str]:
    """Read metadata from comment header lines in CSV."""
    metadata = {}
    with open(csv_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                line = line[1:].strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    metadata[key.strip()] = value.strip()
            else:
                break
    return metadata


def extract_section_audio(
    audio_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    fade_duration: float = 0.05,
    sr: int = 44100
) -> Path:
    """
    Extract a section from audio file with fade in/out.

    Parameters
    ----------
    audio_path : Path
        Path to input audio file
    output_path : Path
        Path for output WAV file
    start_time : float
        Section start time in seconds
    end_time : float
        Section end time in seconds
    fade_duration : float
        Fade in/out duration in seconds (default: 50ms)
    sr : int
        Sample rate (default: 44100)

    Returns
    -------
    Path
        Path to created section WAV file
    """
    duration = end_time - start_time

    # Load the specific portion of audio (stereo if available)
    y, file_sr = librosa.load(
        str(audio_path),
        sr=sr,
        mono=False,
        offset=start_time,
        duration=duration
    )

    # Handle mono vs stereo
    if y.ndim == 1:
        y = y.reshape(1, -1)  # Make it (1, samples) for consistent processing

    # Calculate fade samples
    fade_samples = int(fade_duration * sr)

    # Apply fade in
    if fade_samples > 0 and fade_samples < y.shape[1]:
        fade_in = np.linspace(0, 1, fade_samples)
        for ch in range(y.shape[0]):
            y[ch, :fade_samples] *= fade_in

    # Apply fade out
    if fade_samples > 0 and fade_samples < y.shape[1]:
        fade_out = np.linspace(1, 0, fade_samples)
        for ch in range(y.shape[0]):
            y[ch, -fade_samples:] *= fade_out

    # Transpose for soundfile (samples, channels)
    if y.shape[0] == 1:
        y_out = y[0]  # Mono
    else:
        y_out = y.T  # Stereo: (samples, channels)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as WAV
    sf.write(str(output_path), y_out, sr)

    return output_path


def extract_all_sections(
    filtered_dir: Path,
    audio_path: Path,
    output_dir: Optional[Path] = None,
    fade_duration: float = 0.05,
    sr: int = 44100,
    verbose: bool = True
) -> List[Dict]:
    """
    Extract all sections from filtered pattern CSVs.

    Parameters
    ----------
    filtered_dir : Path
        Path to 6.2_filtered_patterns directory
    audio_path : Path
        Path to original audio file
    output_dir : Path, optional
        Output directory (default: sibling 9.1_sections folder)
    fade_duration : float
        Fade in/out duration in seconds (default: 50ms)
    sr : int
        Sample rate (default: 44100)
    verbose : bool
        Print progress messages

    Returns
    -------
    List[Dict]
        List of extracted section info dicts
    """
    filtered_dir = Path(filtered_dir)
    audio_path = Path(audio_path)

    if output_dir is None:
        output_dir = filtered_dir.parent / '9.1_sections'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n[11.1] Section Extraction")
        print(f"  Input: {filtered_dir.name}")
        print(f"  Audio: {audio_path.name}")
        print(f"  Output: {output_dir.name}")

    # Find all anchored CSVs (not reference_onsets, not macOS ._ files)
    anchored_csvs = sorted([
        f for f in filtered_dir.glob('*.csv')
        if '_anchored' in f.name and 'reference' not in f.name and not f.name.startswith('._')
    ])

    if verbose:
        print(f"  Found {len(anchored_csvs)} filtered pattern files")

    extracted_sections = []

    for csv_path in anchored_csvs:
        # Read metadata
        metadata = read_metadata(csv_path)

        # Get used section boundaries
        used_start = metadata.get('used_section_start')
        used_end = metadata.get('used_section_end')

        if used_start is None or used_end is None:
            if verbose:
                print(f"    Skipping {csv_path.name}: missing used_section_start/end")
            continue

        try:
            start_time = float(used_start)
            end_time = float(used_end)
        except ValueError:
            if verbose:
                print(f"    Skipping {csv_path.name}: invalid time values")
            continue

        # Skip if duration is too short
        duration = end_time - start_time
        if duration < 0.1:
            if verbose:
                print(f"    Skipping {csv_path.name}: duration too short ({duration:.3f}s)")
            continue

        # Build output filename: same as input but .wav instead of .csv
        # e.g., SecNo1_L4_chorus_0.1344_anchored.csv -> SecNo1_L4_chorus_0.1344_section.wav
        output_name = csv_path.stem.replace('_anchored', '_section') + '.wav'
        output_path = output_dir / output_name

        # Extract section
        try:
            extract_section_audio(
                audio_path=audio_path,
                output_path=output_path,
                start_time=start_time,
                end_time=end_time,
                fade_duration=fade_duration,
                sr=sr
            )

            section_info = {
                'csv_file': csv_path.name,
                'output_file': output_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'section_label': metadata.get('section_label', ''),
                'pattern_length': metadata.get('pattern_length', ''),
                'no_of_repetitions': metadata.get('no_of_repetitions', ''),
            }
            extracted_sections.append(section_info)

            if verbose:
                print(f"    ✓ {output_name} ({start_time:.2f}s - {end_time:.2f}s, {duration:.2f}s)")

        except Exception as e:
            if verbose:
                print(f"    ✗ {csv_path.name}: {e}")

    if verbose:
        print(f"  Extracted {len(extracted_sections)} sections")

    return extracted_sections


def extract_sections_for_track(
    track_dir: Path,
    audio_path: Optional[Path] = None,
    fade_duration: float = 0.05,
    sr: int = 44100,
    verbose: bool = True
) -> List[Dict]:
    """
    Extract sections for a single track.

    Parameters
    ----------
    track_dir : Path
        Track output directory containing 6.2_filtered_patterns
    audio_path : Path, optional
        Path to original audio (auto-detected if not provided)
    fade_duration : float
        Fade in/out duration in seconds
    sr : int
        Sample rate
    verbose : bool
        Print progress

    Returns
    -------
    List[Dict]
        List of extracted section info
    """
    track_dir = Path(track_dir)

    # Find filtered patterns directory
    filtered_dir = track_dir / '6.2_filtered_patterns'
    if not filtered_dir.exists():
        if verbose:
            print(f"  ✗ No 6.2_filtered_patterns found in {track_dir.name}")
        return []

    # Auto-detect audio path if not provided
    if audio_path is None:
        # Look for common audio formats
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            candidates = list(track_dir.parent.glob(f"*{ext}"))
            # Also check for audio named like the track folder
            track_audio = track_dir.parent / f"{track_dir.name}{ext}"
            if track_audio.exists():
                audio_path = track_audio
                break

        if audio_path is None:
            # Check if there's a source audio reference in the directory
            # Look in stems folder for reference
            stems_dir = track_dir / 'stems'
            if stems_dir.exists():
                # Try to find original from stems metadata or nearby
                pass

            if verbose:
                print(f"  ✗ Could not find audio file for {track_dir.name}")
            return []

    return extract_all_sections(
        filtered_dir=filtered_dir,
        audio_path=audio_path,
        output_dir=track_dir / '9.1_sections',
        fade_duration=fade_duration,
        sr=sr,
        verbose=verbose
    )


if __name__ == '__main__':
    """
    Command line usage:
        python extract_sections.py <filtered_dir> <audio_path> [output_dir]

    Example:
        python extract_sections.py ./6.2_filtered_patterns ./track.wav ./9.1_sections
    """
    if len(sys.argv) < 3:
        print("Usage: python extract_sections.py <filtered_dir> <audio_path> [output_dir]")
        print("  filtered_dir: Path to 6.2_filtered_patterns directory")
        print("  audio_path: Path to original audio file")
        print("  output_dir: Optional output directory (default: sibling 9.1_sections)")
        sys.exit(1)

    filtered_dir = Path(sys.argv[1])
    audio_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    if not filtered_dir.exists():
        print(f"Error: Filtered directory not found: {filtered_dir}")
        sys.exit(1)

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    results = extract_all_sections(
        filtered_dir=filtered_dir,
        audio_path=audio_path,
        output_dir=output_dir,
        verbose=True
    )

    print(f"\nDone. Extracted {len(results)} sections.")
