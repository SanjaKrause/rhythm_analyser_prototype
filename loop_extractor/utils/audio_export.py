"""
Audio export utilities for creating example WAVs with click tracks.

This module creates demonstration WAVs that overlay click tracks on the original
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
