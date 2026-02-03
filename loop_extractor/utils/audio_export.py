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
# WAV EXPORT
# ============================================================================

def export_audio_to_wav(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 44100
):
    """
    Export audio to WAV file.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    output_path : str
        Output WAV file path
    sample_rate : int
        Audio sample rate

    Examples
    --------
    >>> audio = np.random.randn(88200)
    >>> export_audio_to_wav(audio, 'output.wav')
    """
    import soundfile as sf

    output_path = Path(output_path)
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
    methods: Optional[List[str]] = None,
    groove_pulse_csv: Optional[str] = None,
    export_format: str = 'wav'
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
        Methods to create examples for (default: ['uncorrected', 'per_snippet',
        '4bar_pattern_flexStart', '2bar_pattern_flexStart', '1bar_pattern_flexStart'])
    groove_pulse_csv : str, optional
        Path to groove pulse filtered CSV for creating groove pulse click tracks
    export_format : str
        Export format: 'wav' or 'mp3' (default: 'wav')

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
        methods = ['uncorrected', 'per_snippet', '4bar_pattern_flexStart', '2bar_pattern_flexStart', '1bar_pattern_flexStart']

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
            # FlexStart pattern methods (4bar_pattern_flexStart, 2bar_pattern_flexStart, 1bar_pattern_flexStart)
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
        # Use simple naming: uncorrected.wav/.mp3, per_snippet.wav/.mp3, etc.
        method_name = method
        output_file = output_dir / f"{method_name}.{export_format}"

        if export_format == 'mp3':
            export_audio_to_mp3(mixed, str(output_file), sr)
        else:
            export_audio_to_wav(mixed, str(output_file), sr)

    # Create groove pulse click tracks if CSV provided
    if groove_pulse_csv and Path(groove_pulse_csv).exists():
        print(f"\n  Creating groove pulse click tracks...")
        try:
            df_groove = pd.read_csv(groove_pulse_csv)

            # Collect plot data for all pattern lengths
            plot_data = []

            # Process each FlexStart pattern length
            for pattern_length in [4, 2, 1]:
                method_name = f'groove_pulse_{pattern_length}bar'
                print(f"\n  Creating {method_name} example...")

                # Filter to FlexStart method with matching pattern length
                df_filtered = df_groove[
                    (df_groove['method'].str.contains('FlexStart', case=False, na=False)) &
                    (df_groove['pattern_length'] == pattern_length) &
                    (df_groove['onset_strength_filtered'] > 0)  # Only positions that passed threshold
                ]

                if df_filtered.empty:
                    print(f"    ⚠️  No groove pulse positions found for {pattern_length}-bar pattern, skipping")
                    continue

                # Calculate absolute times for groove pulse positions
                # Strategy: Load the filtered flexStart CSV to get timing for all repetitions

                # Load the filtered flexStart CSV for this pattern length
                pattern_csv_name = f'{Path(comprehensive_csv_file).stem}_{pattern_length}bar_flexStart_filtered.csv'
                pattern_csv_path = Path(comprehensive_csv_file).parent / pattern_csv_name

                if not pattern_csv_path.exists():
                    print(f"    ⚠️  FlexStart CSV not found: {pattern_csv_path.name}, skipping")
                    continue

                # Read metadata from CSV header
                pattern_start_time = None
                pattern_end_time = None
                pattern_start_time_relative = None
                pattern_end_time_relative = None
                with open(pattern_csv_path, 'r') as f:
                    for line in f:
                        if line.startswith('# pattern_start_time_relative='):
                            pattern_start_time_relative = float(line.split('=')[1].strip())
                        elif line.startswith('# pattern_end_time_relative='):
                            pattern_end_time_relative = float(line.split('=')[1].strip())
                        elif line.startswith('# pattern_start_time='):
                            pattern_start_time = float(line.split('=')[1].strip())
                        elif line.startswith('# pattern_end_time='):
                            pattern_end_time = float(line.split('=')[1].strip())
                        elif not line.startswith('#'):
                            break  # Stop at first data line

                # Read CSV, skipping comment lines (metadata headers starting with #)
                df_pattern = pd.read_csv(pattern_csv_path, comment='#')

                # Debug: Print boundary times
                if pattern_start_time_relative is not None and pattern_end_time_relative is not None:
                    print(f"    Pattern boundaries (from metadata): start={pattern_start_time_relative:.3f}s, end={pattern_end_time_relative:.3f}s (relative)")
                elif pattern_start_time is not None and pattern_end_time is not None:
                    # Fallback to calculating from absolute times
                    pattern_start_time_relative = pattern_start_time - snippet_offset
                    pattern_end_time_relative = pattern_end_time - snippet_offset
                    print(f"    Pattern boundaries (calculated): start={pattern_start_time_relative:.3f}s, end={pattern_end_time_relative:.3f}s (relative)")
                else:
                    print(f"    ⚠️  Warning: No pattern boundary times found in metadata")

                # Calculate bar_in_pattern from bar_number for the filtered CSV
                # bar_in_pattern = bar_number % pattern_length (0-indexed position within pattern)
                min_bar = df_pattern['bar_number'].min()
                df_pattern['bar_in_pattern'] = (df_pattern['bar_number'] - min_bar) % pattern_length

                # Get groove pulse positions (which positions in the pattern have strong groove)
                groove_positions = df_filtered['position'].values  # 1-based positions in pattern
                groove_relative_phases = df_filtered['relative_median_phase'].values  # Relative median phase (in 16th-note units, -1 to +1)

                # Convert positions to bar_in_pattern and tick_16th
                # Position maps to: bar_in_pattern = (position-1) // 16, tick = (position-1) % 16
                groove_pattern = []
                for position, relative_phase in zip(groove_positions, groove_relative_phases):
                    bar_in_pattern = (position - 1) // 16
                    tick_16th = (position - 1) % 16
                    groove_pattern.append((bar_in_pattern, tick_16th, relative_phase))

                # Check reference pattern (first occurrence) to see which positions have onsets
                # Extract first pattern: bar_in_pattern 0 to pattern_length-1
                # Note: bar_in_pattern was calculated as (bar_number - min_bar) % pattern_length
                first_pattern = df_pattern[df_pattern['bar_in_pattern'] < pattern_length].copy()

                # Build a set of (bar_in_pattern, tick_16th) tuples that have onsets in the reference
                reference_onsets = set()
                for _, row in first_pattern.iterrows():
                    if pd.notna(row.get('onset_time')):
                        reference_onsets.add((row['bar_in_pattern'], row['tick_16th']))

                # Find all matching positions in the filtered flexStart CSV
                # Filter df_pattern to only include complete patterns
                max_bar = df_pattern['bar_number'].max()
                min_bar = df_pattern['bar_number'].min()
                num_complete_patterns = (max_bar - min_bar + 1) // pattern_length
                last_complete_pattern_bar = min_bar + (num_complete_patterns * pattern_length) - 1

                # Only use bars from complete patterns
                df_complete = df_pattern[df_pattern['bar_number'] <= last_complete_pattern_bar].copy()

                # Strategy: For positions with reference onsets, use grid_time + relative_phase across all repetitions
                groove_times = []
                for bar_in_pattern, tick_16th, relative_phase in groove_pattern:
                    # Check if this position has an onset in the reference pattern
                    if (bar_in_pattern, tick_16th) not in reference_onsets:
                        continue  # Skip positions without reference onsets

                    # Find all occurrences of this position across all loop repetitions (only in complete patterns)
                    # Use modulo to match bar_in_pattern across all repetitions
                    matching_rows = df_complete[
                        (df_complete['bar_number'] % pattern_length == bar_in_pattern) &
                        (df_complete['tick_16th'] == tick_16th)
                    ]

                    for _, row in matching_rows.iterrows():
                        # Use grid_time + relative_phase to get groove pulse timing
                        if 'grid_time' in row and not np.isnan(row['grid_time']):
                            grid_time = row['grid_time']
                            current_bar_number = row['bar_number']

                            # Calculate step duration (time between 16th notes)
                            # Need to find the next 16th-note position
                            if tick_16th < 15:
                                # Next position is in the same bar
                                next_tick = tick_16th + 1
                                next_bar = current_bar_number
                            else:
                                # tick_16th == 15, next position is in the next bar
                                next_tick = 0
                                next_bar = current_bar_number + 1

                            # Find the next grid position (also from complete patterns only)
                            next_row = df_complete[
                                (df_complete['bar_number'] == next_bar) &
                                (df_complete['tick_16th'] == next_tick)
                            ]

                            if next_row.empty or 'grid_time' not in next_row.iloc[0] or np.isnan(next_row.iloc[0]['grid_time']):
                                # Can't find next grid time (probably last bar in snippet), skip this click
                                continue

                            next_grid_time = next_row.iloc[0]['grid_time']
                            step_duration = next_grid_time - grid_time

                            # Calculate groove pulse time: grid_time + (relative_phase * step_duration)
                            # relative_phase is in 16th-note units, so multiply by step_duration to get seconds
                            groove_pulse_time = grid_time + (relative_phase * step_duration)

                            # Make relative to snippet start
                            relative_time = groove_pulse_time - snippet_offset

                            # Only include if within complete pattern boundaries
                            if pattern_end_time_relative is not None:
                                if relative_time >= pattern_end_time_relative:
                                    continue  # Skip clicks at or after the last complete pattern end

                            # Filter by start boundary too
                            if pattern_start_time_relative is not None:
                                if relative_time < pattern_start_time_relative:
                                    continue  # Skip clicks before the first complete pattern start

                            # Only include if within snippet bounds
                            if 0 <= relative_time <= snippet_duration:
                                groove_times.append(relative_time)

                if not groove_times:
                    print(f"    ⚠️  Could not extract grid times for {method_name}, skipping")
                    continue

                groove_times = np.array(groove_times)
                print(f"    Generated {len(groove_times)} clicks within {num_complete_patterns} complete patterns")
                if len(groove_times) > 0:
                    print(f"    Click time range: {groove_times.min():.3f}s to {groove_times.max():.3f}s (relative to snippet)")
                    if pattern_end_time_relative is not None:
                        clicks_after_end = groove_times[groove_times >= pattern_end_time_relative]
                        if len(clicks_after_end) > 0:
                            print(f"    ⚠️  ERROR: {len(clicks_after_end)} clicks still after pattern end {pattern_end_time_relative:.3f}s! Times: {clicks_after_end[:5]}")
                    if pattern_start_time_relative is not None:
                        clicks_before_start = groove_times[groove_times < pattern_start_time_relative]
                        if len(clicks_before_start) > 0:
                            print(f"    ⚠️  ERROR: {len(clicks_before_start)} clicks before pattern start {pattern_start_time_relative:.3f}s! Times: {clicks_before_start[:5]}")

                # Export groove pulse click timings to CSV
                csv_output_file = output_dir / f"{method_name}_click_times.csv"
                df_groove_clicks = pd.DataFrame({
                    'click_time_relative': groove_times,
                    'click_time_absolute': groove_times + snippet_offset
                })
                df_groove_clicks.to_csv(csv_output_file, index=False)
                print(f"    ✓ Saved click timings to {csv_output_file.name}")

                # Calculate pattern boundaries (every pattern_length bars) for complete patterns only
                # Start boundaries only (not the end boundary)
                pattern_boundaries = []
                for i in range(num_complete_patterns):  # Only pattern starts, not the end
                    boundary_bar = min_bar + (i * pattern_length)
                    boundary_data = df_complete[
                        (df_complete['bar_number'] == boundary_bar) &
                        (df_complete['tick_16th'] == 0)
                    ]
                    if not boundary_data.empty:
                        pattern_start = boundary_data['grid_time'].min() - snippet_offset
                        if 0 <= pattern_start <= snippet_duration:
                            pattern_boundaries.append(pattern_start)

                # Add the final boundary from metadata (end of last complete pattern)
                if pattern_end_time_relative is not None:
                    if 0 <= pattern_end_time_relative <= snippet_duration:
                        pattern_boundaries.append(pattern_end_time_relative)

                # CRITICAL: Filter groove_times to ONLY include clicks within pattern boundaries
                if len(pattern_boundaries) >= 2:
                    start_boundary = pattern_boundaries[0]
                    end_boundary = pattern_boundaries[-1]
                    original_count = len(groove_times)

                    # Debug: Check first few clicks
                    if len(groove_times) > 0:
                        print(f"    Before boundary filter: first 3 clicks = {groove_times[:3]}")
                        print(f"    Boundaries: start={start_boundary:.6f}, end={end_boundary:.6f}")

                    # Include clicks at start boundary (with small tolerance for negative phase offsets),
                    # exclude clicks at or after end boundary
                    epsilon = 0.001  # 1ms tolerance for rounding errors and negative phase offsets
                    groove_times = groove_times[(groove_times >= start_boundary - epsilon) & (groove_times < end_boundary)]

                    if len(groove_times) > 0:
                        print(f"    After boundary filter: first 3 clicks = {groove_times[:3]}")

                    if original_count != len(groove_times):
                        print(f"    Filtered clicks to pattern boundaries [{start_boundary:.3f}s, {end_boundary:.3f}s): {original_count} → {len(groove_times)} clicks")

                # Store data for combined plot
                plot_data.append({
                    'method_name': method_name,
                    'pattern_length': pattern_length,
                    'groove_times': groove_times,
                    'pattern_boundaries': pattern_boundaries,
                    'num_complete_patterns': num_complete_patterns
                })

                # Create click track for groove pulse positions
                click_track = create_grid_click_track(groove_times, snippet_duration, sr)

                # Mix with audio (0 dB = full click volume)
                mixed = mix_audio_with_clicks(audio_snippet, click_track, click_volume_db=0.0)

                # Export
                output_file = output_dir / f"{method_name}.{export_format}"

                if export_format == 'mp3':
                    export_audio_to_mp3(mixed, str(output_file), sr)
                else:
                    export_audio_to_wav(mixed, str(output_file), sr)

                print(f"    ✓ Created {method_name}.{export_format} ({len(groove_times)} groove pulse positions)")

            # Create combined plot with all 3 subplots
            if plot_data:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

                for idx, data in enumerate(plot_data):
                    ax = axes[idx]
                    groove_times = data['groove_times']
                    pattern_boundaries = data['pattern_boundaries']
                    pattern_length = data['pattern_length']
                    method_name = data['method_name']
                    num_complete_patterns = data['num_complete_patterns']

                    # Plot pattern boundaries as light blue regions
                    for i in range(len(pattern_boundaries) - 1):
                        ax.axvspan(pattern_boundaries[i], pattern_boundaries[i + 1],
                                  alpha=0.15, color='blue', zorder=0)

                    # Plot ALL pattern boundaries (start and end) - more visible
                    for i, boundary in enumerate(pattern_boundaries):
                        if i == 0:
                            label = 'Pattern boundaries'
                        else:
                            label = None
                        ax.axvline(boundary, color='black', linewidth=2, linestyle='--',
                                  alpha=0.9, label=label, zorder=1)

                    # Debug: Check if any clicks are outside pattern boundaries
                    if len(pattern_boundaries) > 0:
                        last_boundary = pattern_boundaries[-1]
                        clicks_after = groove_times[groove_times > last_boundary]
                        if len(clicks_after) > 0:
                            print(f"    ⚠️  PLOT WARNING for {method_name}: {len(clicks_after)} clicks after last boundary {last_boundary:.3f}s")
                            print(f"       Click times: {clicks_after[:10]}")

                    # Plot clicks as vertical lines
                    ax.vlines(groove_times, 0, 1, colors='red', linewidth=2, alpha=0.7,
                             label='Groove pulse clicks', zorder=2)

                    # Formatting
                    ax.set_xlim(0, snippet_duration)
                    ax.set_ylim(0, 1)
                    ax.set_yticks([])
                    ax.set_title(f'{method_name} ({len(groove_times)} clicks, L={pattern_length}, {num_complete_patterns} patterns)',
                                fontsize=10, fontweight='bold')
                    ax.grid(True, axis='x', alpha=0.3)
                    if idx == 0:
                        ax.legend(loc='upper right', fontsize=8)

                # Only show x-label on bottom subplot
                axes[-1].set_xlabel('Time (seconds)', fontsize=10)

                plt.tight_layout()

                # Save combined plot
                plot_output_file = output_dir / "groove_pulse_click_times.png"
                plt.savefig(plot_output_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\n    ✓ Saved combined click timing plot to {plot_output_file.name}")

        except Exception as e:
            print(f"    ⚠️  Could not create groove pulse click tracks: {e}")
            import traceback
            traceback.print_exc()

    # Also export original snippet without clicks
    print(f"\n  Exporting original snippet...")
    output_file = output_dir / f"original.{export_format}"

    if export_format == 'mp3':
        export_audio_to_mp3(audio_snippet, str(output_file), sr)
    else:
        export_audio_to_wav(audio_snippet, str(output_file), sr)

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
    grid_output_dir: str,
    base_name: str,
    output_dir: str,
    snippet_start: float,
    pattern_lengths: dict,
    fade_duration_ms: float = 5.0,
    export_format: str = 'wav',
    methods: list = None,
    loop_start_offset_ms: float = 0.0  # Set to 0.0 - negative offset was adding silence at loop start
) -> Dict[str, List[Path]]:
    """
    Export stem loops for each correction method using filtered FlexStart CSVs.

    For each method, exports one loop containing L bars from each stem
    (vocals, drums, bass, piano, other). Adds short fade in/out to prevent
    clicks at loop boundaries.

    Uses the filtered FlexStart CSV files (*_4bar_flexStart_filtered.csv, etc.)
    which contain only the patterns that passed the onset count threshold.

    Parameters
    ----------
    stems_dir : str
        Directory containing stem WAV files
    grid_output_dir : str
        Directory containing the filtered flexStart CSV files
    base_name : str
        Base filename (e.g., 'track_id_comprehensive_phases')
    output_dir : str
        Output directory for loop files
    snippet_start : float
        Snippet start time in seconds (not used, kept for compatibility)
    pattern_lengths : dict
        Pattern lengths for each method in BARS (not used with FlexStart methods)
    fade_duration_ms : float
        Fade in/out duration in milliseconds (default: 5ms)
    export_format : str
        Export format: 'wav' or 'mp3' (default: 'wav')
    methods : list, optional
        Methods to export (default: ['4bar_flexStart', '2bar_flexStart', '1bar_flexStart'])
    loop_start_offset_ms : float
        Offset in milliseconds to apply to loop start time (default: -15.0)
        Negative values start earlier to capture drum attack transients.
        Onset detectors detect peak (~14ms late), so -15ms captures the attack start.

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping method names to lists of exported file paths

    Examples
    --------
    >>> loops = export_stem_loops(
    ...     'output/track_id/1_stems',
    ...     'output/track_id/5_grid',
    ...     'track_id_comprehensive_phases',
    ...     'output/track_id/9_loops',
    ...     snippet_start=132.0,
    ...     pattern_lengths={}
    ... )
    """
    import pandas as pd

    stems_dir = Path(stems_dir)
    grid_output_dir = Path(grid_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stem names (matching Spleeter 5-stem output)
    stem_names = ['vocals', 'drums', 'bass', 'piano', 'other']

    # Methods to process: FlexStart pattern methods using filtered CSVs
    if methods is None:
        methods = ['4bar_flexStart', '2bar_flexStart', '1bar_flexStart']

    exported_files = {}

    # Dictionary to store loop timing info for CSV export
    loop_timings = {}

    print(f"\nExporting stem loops from filtered FlexStart CSVs...")
    print(f"  Loop start offset: {loop_start_offset_ms:.1f} ms")

    for method in methods:
        print(f"\n  Method: {method}")

        # Determine pattern length and CSV filename
        if method == '4bar_flexStart':
            pattern_length_bars = 4
            csv_filename = f'{base_name}_4bar_flexStart_filtered.csv'
        elif method == '2bar_flexStart':
            pattern_length_bars = 2
            csv_filename = f'{base_name}_2bar_flexStart_filtered.csv'
        elif method == '1bar_flexStart':
            pattern_length_bars = 1
            csv_filename = f'{base_name}_1bar_flexStart_filtered.csv'
        else:
            print(f"    ⚠️  Unknown method: {method}, skipping")
            continue

        print(f"    Pattern length: {pattern_length_bars} bars")

        # Load filtered FlexStart CSV
        csv_path = grid_output_dir / csv_filename
        if not csv_path.exists():
            print(f"    ⚠️  Filtered CSV not found: {csv_filename}, skipping")
            continue

        df = pd.read_csv(csv_path, comment='#')

        # Calculate number of 16th notes (L bars * 16 16th notes per bar in 4/4)
        num_16th_notes = pattern_length_bars * 16

        # Check if we have enough grid times
        if len(df) < num_16th_notes + 1:
            print(f"    ⚠️  Not enough data (need {num_16th_notes + 1} rows, have {len(df)})")
            continue

        # Get loop boundaries from grid_time column
        # Start: first 16th note (index 0)
        # End: first 16th note of the NEXT loop (index num_16th_notes)
        loop_start_time_grid = pd.to_numeric(df['grid_time'].iloc[0], errors='coerce')
        loop_end_time_grid = pd.to_numeric(df['grid_time'].iloc[num_16th_notes], errors='coerce')

        if pd.isna(loop_start_time_grid) or pd.isna(loop_end_time_grid):
            print(f"    ⚠️  Invalid grid times")
            continue

        # Apply offset to capture attack transient (negative offset = start earlier)
        loop_start_offset_seconds = loop_start_offset_ms / 1000.0
        loop_start_time = loop_start_time_grid + loop_start_offset_seconds
        loop_end_time = loop_end_time_grid  # End time stays the same

        loop_duration = loop_end_time - loop_start_time

        print(f"    Grid times: {loop_start_time_grid:.3f}s - {loop_end_time_grid:.3f}s")
        print(f"    Actual loop (with {loop_start_offset_ms:.1f}ms offset): {loop_start_time:.3f}s - {loop_end_time:.3f}s ({loop_duration:.3f}s)")

        # Store timing info for CSV export
        loop_timings[method] = {
            'start_time': loop_start_time,
            'end_time': loop_end_time
        }

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

    # Export loop timings CSV
    if loop_timings:
        timing_csv_path = output_dir / f"{base_name}_loop_timings.csv"
        timing_df = pd.DataFrame(loop_timings)  # Methods as columns, start/end as rows
        timing_df.to_csv(timing_csv_path)
        print(f"\n  ✓ Loop timings saved to: {timing_csv_path.name}")

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
