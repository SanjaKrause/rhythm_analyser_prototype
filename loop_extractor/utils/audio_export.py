"""
Audio export utilities for creating example MP3s with click tracks.

This module creates demonstration MP3s that overlay click tracks on the original
audio, showing different grid correction methods.

Environment: AEinBOX_13_3 (numpy, librosa, pydub)
"""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# Import config from parent directory
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

import importlib.util
spec = importlib.util.spec_from_file_location("config_module", _parent_dir / "config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config


# ============================================================================
# CONFIGURATION
# ============================================================================

CLICK_FREQUENCY = config.CLICK_TRACK_FREQUENCY  # Hz
CLICK_DURATION = config.CLICK_TRACK_DURATION    # seconds
AUDIO_EXPORT_FORMAT = config.AUDIO_EXPORT_FORMAT
AUDIO_EXPORT_BITRATE = config.AUDIO_EXPORT_BITRATE


# ============================================================================
# CLICK TRACK GENERATION
# ============================================================================

def generate_click(
    sample_rate: int = 44100,
    frequency: float = CLICK_FREQUENCY,
    duration: float = CLICK_DURATION,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    Generate a sharp click sound (short sine burst with exponential decay).

    Uses exact parameters from AP2_create_grid_audio_examples.ipynb:
    - High frequency sine wave (3000 Hz default)
    - Exponential decay envelope for sharp, clicky attack
    - Fast decay factor = 20 for percussive sound

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz
    frequency : float
        Click frequency in Hz (default: 3000 - higher = sharper)
    duration : float
        Click duration in seconds (default: 0.05)
    amplitude : float
        Click amplitude (0.0 to 1.0, default: 1.0)

    Returns
    -------
    np.ndarray
        Click waveform

    Examples
    --------
    >>> click = generate_click(44100, 3000, 0.05)
    >>> len(click)
    2205
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, endpoint=False)

    # Generate sine wave
    click = np.sin(2 * np.pi * frequency * t)

    # Apply sharp exponential envelope for clicky sound
    # Higher decay factor (20) = faster decay = more percussive/clicky
    envelope = np.exp(-20 * t / duration)
    click *= envelope

    # Apply amplitude
    click *= amplitude

    return click.astype(np.float32)


def create_click_track(
    click_times: List[float],
    total_duration: float,
    sample_rate: int = 44100,
    click_frequency: float = CLICK_FREQUENCY,
    click_duration: float = CLICK_DURATION,
    downbeat_frequency: Optional[float] = None
) -> np.ndarray:
    """
    Create click track with clicks at specified times.

    Parameters
    ----------
    click_times : List[float]
        Times in seconds where clicks should occur
    total_duration : float
        Total duration of click track in seconds
    sample_rate : int
        Audio sample rate in Hz
    click_frequency : float
        Frequency for regular clicks in Hz
    click_duration : float
        Duration of each click in seconds
    downbeat_frequency : float, optional
        Frequency for downbeat clicks (if None, use click_frequency)

    Returns
    -------
    np.ndarray
        Click track waveform

    Examples
    --------
    >>> click_track = create_click_track([0.0, 0.5, 1.0, 1.5], 2.0)
    >>> click_track.shape
    (88200,)
    """
    # Create empty audio buffer
    total_samples = int(total_duration * sample_rate)
    click_track = np.zeros(total_samples, dtype=np.float32)

    # Generate click templates (full amplitude, volume adjusted in mixing)
    regular_click = generate_click(sample_rate, click_frequency, click_duration, amplitude=1.0)

    if downbeat_frequency is not None:
        downbeat_click = generate_click(sample_rate, downbeat_frequency, click_duration, amplitude=1.0)
    else:
        downbeat_click = regular_click

    # Add clicks at specified times
    for i, click_time in enumerate(click_times):
        sample_pos = int(click_time * sample_rate)

        # Use downbeat click for first click and every Nth click
        # (assuming clicks are at regular intervals)
        is_downbeat = (i % 4 == 0)  # Every 4th click is a downbeat
        click = downbeat_click if is_downbeat else regular_click

        # Add click to track (with bounds checking)
        end_pos = min(sample_pos + len(click), total_samples)
        click_len = end_pos - sample_pos

        if click_len > 0:
            click_track[sample_pos:end_pos] += click[:click_len]

    return click_track


def create_grid_click_track(
    grid_times: np.ndarray,
    total_duration: float,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Create click track from grid times (e.g., from comprehensive CSV).

    Parameters
    ----------
    grid_times : np.ndarray
        Array of grid times in seconds
    total_duration : float
        Total duration in seconds
    sample_rate : int
        Audio sample rate

    Returns
    -------
    np.ndarray
        Click track waveform

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'grid_time_per_snippet': [10.0, 10.5, 11.0, 11.5]})
    >>> click_track = create_grid_click_track(df['grid_time_per_snippet'].values, 30.0)
    """
    # Remove NaN values and sort
    valid_times = grid_times[~np.isnan(grid_times)]
    valid_times = np.sort(valid_times)

    return create_click_track(valid_times.tolist(), total_duration, sample_rate)


# ============================================================================
# AUDIO MIXING
# ============================================================================

def mix_audio_with_clicks(
    audio: np.ndarray,
    click_track: np.ndarray,
    click_volume_db: float = 0.0
) -> np.ndarray:
    """
    Mix original audio with click track.

    Uses exact mixing from AP2_create_grid_audio_examples.ipynb:
    - Click volume in dB (0 dB = no attenuation, -18 dB = quieter)
    - Audio at full volume
    - Normalize to prevent clipping

    Parameters
    ----------
    audio : np.ndarray
        Original audio waveform
    click_track : np.ndarray
        Click track waveform
    click_volume_db : float
        Click volume in dB (default: 0.0 = full volume, negative = quieter)

    Returns
    -------
    np.ndarray
        Mixed audio

    Examples
    --------
    >>> audio = np.random.randn(88200)
    >>> clicks = np.random.randn(88200)
    >>> mixed = mix_audio_with_clicks(audio, clicks, click_volume_db=0.0)
    >>> mixed.shape
    (88200,)
    """
    # Ensure same length
    min_len = min(len(audio), len(click_track))
    audio = audio[:min_len]
    click_track = click_track[:min_len]

    # Convert dB to linear gain
    click_gain = 10 ** (click_volume_db / 20.0)

    # Mix with click gain
    mixed = audio + (click_track * click_gain)

    # Normalize to prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 1.0:
        mixed = mixed / max_val * 0.99

    return mixed.astype(np.float32)


# ============================================================================
# MP3 EXPORT
# ============================================================================

def export_audio_to_mp3(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 44100,
    bitrate: str = AUDIO_EXPORT_BITRATE
):
    """
    Export audio to MP3 file.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    output_path : str
        Output MP3 file path
    sample_rate : int
        Audio sample rate
    bitrate : str
        MP3 bitrate (e.g., '192k')

    Examples
    --------
    >>> audio = np.random.randn(88200)
    >>> export_audio_to_mp3(audio, 'output.mp3')
    """
    try:
        from pydub import AudioSegment
        import io

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1  # mono
        )

        # Export to MP3
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_segment.export(
            str(output_path),
            format='mp3',
            bitrate=bitrate
        )

        print(f"  ✓ Exported to {output_path}")

    except ImportError:
        # Fallback: export as WAV
        print("  ⚠️  pydub not available, exporting as WAV instead")
        import soundfile as sf

        output_path = Path(output_path)
        output_path = output_path.with_suffix('.wav')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(output_path), audio, sample_rate)
        print(f"  ✓ Exported to {output_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def create_audio_examples(
    audio_file: str,
    comprehensive_csv_file: str,
    output_dir: str,
    snippet_offset: float = 0.0,
    snippet_duration: float = 30.0,
    methods: Optional[List[str]] = None
):
    """
    Create audio examples with click tracks for different correction methods.

    This is the main entry point for Step 6 of the AP2 pipeline.

    Parameters
    ----------
    audio_file : str
        Path to original audio file (WAV, MP3, etc.)
    comprehensive_csv_file : str
        Path to comprehensive phases CSV
    output_dir : str
        Output directory for MP3 files
    snippet_offset : float
        Snippet start time in seconds
    snippet_duration : float
        Snippet duration in seconds
    methods : List[str], optional
        Methods to create examples for (default: ['per_snippet', 'drum', 'mel', 'pitch'])

    Examples
    --------
    >>> create_audio_examples(
    ...     'track.wav',
    ...     'track_comprehensive.csv',
    ...     'output/audio_examples',
    ...     snippet_offset=10.0
    ... )
    """
    import pandas as pd

    if methods is None:
        methods = ['uncorrected', 'per_snippet', 'drum', 'mel', 'pitch', 'standard_L1', 'standard_L2', 'standard_L4']

    print(f"\nCreating audio examples...")

    # Load audio
    print(f"  Loading audio from {audio_file}...")
    audio, sr = librosa.load(audio_file, sr=44100, mono=True)

    # Extract snippet
    snippet_start_sample = int(snippet_offset * sr)
    snippet_end_sample = int((snippet_offset + snippet_duration) * sr)
    audio_snippet = audio[snippet_start_sample:snippet_end_sample]

    print(f"  Extracted {len(audio_snippet) / sr:.1f}s snippet")

    # Load comprehensive CSV
    print(f"  Loading grid times from {comprehensive_csv_file}...")
    df = pd.read_csv(comprehensive_csv_file)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create click tracks for each method
    for method in methods:
        print(f"\n  Creating {method} example...")

        # Determine grid time column
        if method == 'uncorrected':
            col_name = 'grid_time_uncorrected'
        elif method == 'per_snippet':
            col_name = 'grid_time_per_snippet'
        else:
            # Loop-based methods (drum, mel, pitch, standard_L1, standard_L2, standard_L4)
            # Find column matching method
            matching_cols = [c for c in df.columns if f'grid_time_{method}' in c]
            if not matching_cols:
                print(f"    ⚠️  No grid times found for {method}, skipping")
                continue
            col_name = matching_cols[0]

        if col_name not in df.columns:
            print(f"    ⚠️  Column {col_name} not found, skipping")
            continue

        # Get grid times (relative to snippet start)
        grid_times = df[col_name].values
        grid_times = grid_times - snippet_offset  # Make relative to snippet

        # Create click track
        click_track = create_grid_click_track(grid_times, snippet_duration, sr)

        # Mix with audio (0 dB = full click volume, matching notebook)
        mixed = mix_audio_with_clicks(audio_snippet, click_track, click_volume_db=0.0)

        # Export
        # Use simple naming: uncorrected.mp3, per_snippet.mp3, drum.mp3, standard_L1.mp3, etc.
        method_name = method if method in ['uncorrected', 'per_snippet'] else method
        output_file = output_dir / f"{method_name}.mp3"
        export_audio_to_mp3(mixed, str(output_file), sr)

    # Also export original snippet without clicks
    print(f"\n  Exporting original snippet...")
    output_file = output_dir / "original.mp3"
    export_audio_to_mp3(audio_snippet, str(output_file), sr)

    print(f"\n  ✓ Audio examples created in {output_dir}")

    # Clear audio from memory
    del audio, audio_snippet
    import gc
    gc.collect()


# ============================================================================
# STEM LOOP EXPORT
# ============================================================================

def apply_fade(
    audio: np.ndarray,
    fade_in_samples: int,
    fade_out_samples: int
) -> np.ndarray:
    """
    Apply fade in and fade out to prevent clicks at loop boundaries.

    Uses cosine fade for smooth transitions.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    fade_in_samples : int
        Number of samples for fade in
    fade_out_samples : int
        Number of samples for fade out

    Returns
    -------
    np.ndarray
        Audio with fades applied
    """
    audio_faded = audio.copy()

    # Fade in (cosine curve: 0 -> 1)
    if fade_in_samples > 0:
        fade_in_curve = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_in_samples)))
        audio_faded[:fade_in_samples] *= fade_in_curve

    # Fade out (cosine curve: 1 -> 0)
    if fade_out_samples > 0:
        fade_out_curve = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_out_samples)))
        audio_faded[-fade_out_samples:] *= fade_out_curve

    return audio_faded


def export_stem_loops(
    stems_dir: str,
    comprehensive_csv_path: str,
    bar_tempos_csv_path: str,
    output_dir: str,
    snippet_start: float,
    pattern_lengths: dict,
    fade_duration_ms: float = 5.0,
    export_format: str = 'wav',
    methods: list = None
) -> Dict[str, List[Path]]:
    """
    Export stem loops for each correction method.

    For each method (drum, mel, pitch), exports one loop containing L bars
    from each stem (vocals, drums, bass, piano, other). Adds short fade in/out
    to prevent clicks at loop boundaries.

    Parameters
    ----------
    stems_dir : str
        Directory containing stem WAV files
    comprehensive_csv_path : str
        Path to comprehensive phases CSV
    bar_tempos_csv_path : str
        Path to bar tempos CSV
    output_dir : str
        Output directory for loop files
    snippet_start : float
        Snippet start time in seconds
    pattern_lengths : dict
        Pattern lengths for each method in BARS (e.g., {'drum': 4, 'mel': 4, 'pitch': 8})
    fade_duration_ms : float
        Fade in/out duration in milliseconds (default: 5ms)
    export_format : str
        Export format: 'wav' or 'mp3' (default: 'wav')

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping method names to lists of exported file paths

    Examples
    --------
    >>> pattern_lengths = {'drum': 4, 'mel': 4, 'pitch': 8}
    >>> loops = export_stem_loops(
    ...     'output/track_id/1_stems',
    ...     'output/track_id/5_grid/track_id_comprehensive.csv',
    ...     'output/track_id/3_beats/track_id_bar_tempos.csv',
    ...     'output/track_id/9_loops',
    ...     snippet_start=132.0,
    ...     pattern_lengths=pattern_lengths
    ... )
    """
    import pandas as pd
    import re

    stems_dir = Path(stems_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load comprehensive CSV
    df = pd.read_csv(comprehensive_csv_path)

    # Stem names (matching Spleeter 5-stem output)
    stem_names = ['vocals', 'drums', 'bass', 'piano', 'other']

    # Methods to process (including per_snippet with fixed 4-bar loop and standard L=1, L=2, L=4)
    if methods is None:
        methods = ['per_snippet', 'drum', 'mel', 'pitch', 'standard_L1', 'standard_L2', 'standard_L4']

    exported_files = {}

    print(f"\nExporting stem loops...")

    for method in methods:
        print(f"\n  Method: {method}")

        # Determine pattern length
        if method == 'per_snippet':
            # per_snippet always uses 4 bars (no pattern length calculated for this correction)
            pattern_length_bars = 4
            print(f"    Pattern length: {pattern_length_bars} bars (fixed for per_snippet)")
        else:
            # Find phase column with L value for loop-based methods
            matching_cols = [c for c in df.columns if f'phase_{method}(L=' in c]
            if not matching_cols:
                print(f"    ⚠️  No phase column found for {method}, skipping")
                continue

            col_name = matching_cols[0]

            # Extract L from column name like "phase_drum(L=4)"
            match = re.search(r'L=(\d+)', col_name)
            if not match:
                print(f"    ⚠️  Could not extract L value from {col_name}, skipping")
                continue

            pattern_length_bars = int(match.group(1))
            print(f"    Pattern length: {pattern_length_bars} bars")

        # Calculate number of 16th notes (L bars * 16 16th notes per bar in 4/4)
        num_16th_notes = pattern_length_bars * 16

        # Get the time range for the first loop based on grid times
        # Use the corrected grid times from the comprehensive CSV
        # Loop: from first 16th note (index 0) to first 16th of next loop (index L*16)

        # Find grid time column for this method
        grid_col_pattern = f'grid_time_{method}'
        grid_cols = [c for c in df.columns if grid_col_pattern in c]

        if not grid_cols:
            print(f"    ⚠️  No grid_time column found for {method}")
            continue

        grid_col = grid_cols[0]

        # Check if we have enough grid times
        if len(df) < num_16th_notes + 1:
            print(f"    ⚠️  Not enough data (need {num_16th_notes + 1} rows, have {len(df)})")
            continue

        # Get loop boundaries from grid times
        # Start: first 16th note (index 0)
        # End: first 16th note of the NEXT loop (index num_16th_notes)
        loop_start_time = pd.to_numeric(df[grid_col].iloc[0], errors='coerce')
        loop_end_time = pd.to_numeric(df[grid_col].iloc[num_16th_notes], errors='coerce')

        if pd.isna(loop_start_time) or pd.isna(loop_end_time):
            print(f"    ⚠️  Invalid grid times")
            continue

        loop_duration = loop_end_time - loop_start_time

        print(f"    Loop: {loop_start_time:.3f}s - {loop_end_time:.3f}s ({loop_duration:.3f}s)")

        # Create method subdirectory (or use output_dir directly if only one method)
        if len(methods) == 1:
            # Single method mode (e.g., DAW ready): export directly to output_dir
            method_dir = output_dir
        else:
            # Multiple methods: create subdirectories
            method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        exported_files[method] = []

        # Export each stem
        for stem_name in stem_names:
            stem_path = stems_dir / f"{stem_name}.wav"

            if not stem_path.exists():
                print(f"      ⚠️  Stem not found: {stem_name}.wav, skipping")
                continue

            # Load stem audio
            audio, sr = librosa.load(str(stem_path), sr=44100, mono=True)

            # Extract loop
            start_sample = int(loop_start_time * sr)
            end_sample = int(loop_end_time * sr)

            if start_sample >= len(audio) or end_sample > len(audio):
                print(f"      ⚠️  Loop time out of bounds for {stem_name}, skipping")
                continue

            loop_audio = audio[start_sample:end_sample]

            # Apply fade in/out
            fade_samples = int((fade_duration_ms / 1000.0) * sr)
            loop_audio = apply_fade(loop_audio, fade_samples, fade_samples)

            # Export
            output_filename = f"{stem_name}.{export_format}"
            output_path = method_dir / output_filename

            if export_format == 'mp3':
                export_audio_to_mp3(loop_audio, str(output_path), sr)
            else:
                # Export as WAV
                import soundfile as sf
                sf.write(str(output_path), loop_audio, sr)
                print(f"      ✓ Exported: {output_filename}")

            exported_files[method].append(output_path)

            # Clear audio from memory after each stem
            del audio, loop_audio

    print(f"\n  ✓ Stem loops exported to {output_dir}")

    # Final garbage collection
    import gc
    gc.collect()

    return exported_files


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_audio_functions():
    """Test audio generation functions."""
    print("=" * 80)
    print("Testing Audio Export Module")
    print("=" * 80)

    # Test 1: Generate click
    print("\n[1] Testing generate_click...")
    click = generate_click(44100, 1000, 0.05)
    assert len(click) > 0, "Should generate click"
    assert click.dtype == np.float32, "Should be float32"
    print(f"  Generated {len(click)} samples ({len(click)/44100:.3f}s)")
    print("  ✓ Click generation works")

    # Test 2: Create click track
    print("\n[2] Testing create_click_track...")
    click_times = [0.0, 0.5, 1.0, 1.5]
    click_track = create_click_track(click_times, 2.0, 44100)
    assert len(click_track) == 2 * 44100, "Should match duration"
    print(f"  Created click track: {len(click_track)} samples")
    print("  ✓ Click track creation works")

    # Test 3: Mix audio
    print("\n[3] Testing mix_audio_with_clicks...")
    audio = np.random.randn(88200).astype(np.float32) * 0.1
    mixed = mix_audio_with_clicks(audio, click_track)
    assert len(mixed) == len(audio), "Should preserve length"
    assert np.abs(mixed).max() <= 1.0, "Should not clip"
    print(f"  Mixed audio: max={np.abs(mixed).max():.3f}")
    print("  ✓ Audio mixing works")

    print("\n" + "=" * 80)
    print("All audio export tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_audio_functions()
