"""
Spleeter 5-stem separation interface.

This module provides functions for separating audio into 5 stems using Spleeter
and converting them to mel-spectrograms for Beat-Transformer input.

Environment: AEinBOX_13_3
Dependencies: spleeter, librosa, numpy
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Tuple
from spleeter.separator import Separator

# Import config
import importlib.util
_config_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("config_module", _config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config

# Global separator instance to avoid reloading model for each track
_separator_instance = None


def separate_stems(
    audio_path: str,
    output_dir: str,
    model: str = None
) -> Path:
    """
    Separate audio into 5 stems using Spleeter.

    Parameters
    ----------
    audio_path : str
        Path to input audio file (WAV, MP3, FLAC, etc.)
    output_dir : str
        Directory where separated stems will be saved
    model : str, optional
        Spleeter model name (default: from config)

    Returns
    -------
    Path
        Path to the directory containing separated stems

    Notes
    -----
    Creates output directory structure:
        output_dir/
        └── {track_name}/
            ├── vocals.wav
            ├── drums.wav
            ├── bass.wav
            ├── piano.wav
            └── other.wav

    Examples
    --------
    >>> stems_dir = separate_stems('track.wav', '/output/stems')
    >>> print(list(stems_dir.glob('*.wav')))
    [vocals.wav, drums.wav, bass.wav, piano.wav, other.wav]
    """
    global _separator_instance

    if model is None:
        model = config.SPLEETER_MODEL

    # Reuse global separator instance to avoid reloading model each time
    if _separator_instance is None:
        print(f"Initializing Spleeter model ({model})...")
        _separator_instance = Separator(model)

    separator = _separator_instance

    # Convert to Path objects
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Separating {audio_path.name} into 5 stems...")

    # Perform separation
    # Use custom filename_format to save stems directly to output_dir without subdirectory
    # By default Spleeter creates: output_dir/{track_name}/{instrument}.wav
    # We override this by using output_dir.parent and including output_dir.name in format
    separator.separate_to_file(
        str(audio_path),
        str(output_dir.parent),
        codec=config.SPLEETER_OUTPUT_CODEC,
        filename_format=f"{output_dir.name}/{{instrument}}.{{codec}}"
    )

    # Stems are saved directly to output_dir (no subdirectory)
    stems_dir = output_dir

    print(f"Stems saved to: {stems_dir}")

    return stems_dir


def load_stem_as_mel_spectrogram(
    stem_path: str,
    sr: int = None,
    n_fft: int = None,
    hop_length: int = None,
    n_mels: int = None,
    fmin: float = None,
    fmax: float = None
) -> np.ndarray:
    """
    Load a stem WAV file and convert to mel-spectrogram.

    Parameters
    ----------
    stem_path : str
        Path to stem WAV file
    sr : int, optional
        Sample rate (default: from config)
    n_fft : int, optional
        FFT window size (default: from config)
    hop_length : int, optional
        STFT hop length (default: from config)
    n_mels : int, optional
        Number of mel bins (default: from config)
    fmin : float, optional
        Minimum frequency in Hz (default: from config)
    fmax : float, optional
        Maximum frequency in Hz (default: from config)

    Returns
    -------
    np.ndarray
        Mel-spectrogram in dB scale, shape (n_mels, time_frames)

    Examples
    --------
    >>> mel_spec = load_stem_as_mel_spectrogram('vocals.wav')
    >>> mel_spec.shape
    (128, 9294)
    """
    # Use config defaults if not specified
    if sr is None:
        sr = config.MEL_SR
    if n_fft is None:
        n_fft = config.MEL_N_FFT
    if hop_length is None:
        hop_length = config.MEL_HOP_LENGTH
    if n_mels is None:
        n_mels = config.MEL_N_MELS
    if fmin is None:
        fmin = config.MEL_FMIN
    if fmax is None:
        fmax = config.MEL_FMAX

    # Load audio as mono
    y, _ = librosa.load(stem_path, sr=sr, mono=True)

    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Convert to power spectrogram
    S = np.abs(D) ** 2

    # Apply mel filterbank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_S = np.dot(mel_basis, S)

    # Convert to dB scale
    mel_S_db = librosa.power_to_db(mel_S, ref=np.max)

    return mel_S_db


def create_5stem_npz(
    stems_dir: str,
    output_npz_path: str,
    stems_order: list = None
) -> Path:
    """
    Convert 5 separated stems to a single NPZ file with mel-spectrograms.

    Parameters
    ----------
    stems_dir : str
        Directory containing the 5 stem WAV files
    output_npz_path : str
        Path for output NPZ file
    stems_order : list, optional
        Order of stems (default: from config)

    Returns
    -------
    Path
        Path to created NPZ file

    Notes
    -----
    Creates NPZ file with:
        - Key: 'spec'
        - Shape: (5, time_frames, n_mels)
        - Stems order: [vocals, drums, bass, piano, other]

    Examples
    --------
    >>> npz_path = create_5stem_npz('stems_dir/', 'output.npz')
    >>> data = np.load(npz_path)
    >>> data['spec'].shape
    (5, 9294, 128)
    """
    if stems_order is None:
        stems_order = config.STEMS

    stems_dir = Path(stems_dir)
    output_npz_path = Path(output_npz_path)

    # Ensure output directory exists
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting stems to mel-spectrograms...")

    # Load each stem and convert to mel-spectrogram
    mel_specs = []
    for stem_name in stems_order:
        stem_path = stems_dir / f"{stem_name}.wav"

        if not stem_path.exists():
            raise FileNotFoundError(f"Stem file not found: {stem_path}")

        print(f"  Processing {stem_name}.wav...")
        mel_spec = load_stem_as_mel_spectrogram(str(stem_path))
        mel_specs.append(mel_spec)

    # Stack all stems: (5, n_mels, time_frames)
    stacked = np.stack(mel_specs, axis=0)

    # Transpose to (5, time_frames, n_mels) to match Beat-Transformer expected input
    spec_5stems = np.transpose(stacked, (0, 2, 1))

    print(f"Final shape: {spec_5stems.shape} (5 stems, {spec_5stems.shape[1]} frames, {spec_5stems.shape[2]} mels)")

    # Save as NPZ
    np.savez(output_npz_path, spec=spec_5stems)

    print(f"Saved to: {output_npz_path}")

    # Clear mel_specs from memory immediately
    del mel_specs, stacked, spec_5stems
    import gc
    gc.collect()

    return output_npz_path


def process_audio_to_stems_and_npz(
    audio_path: str,
    stems_output_dir: str,
    npz_output_path: str
) -> Tuple[Path, Path]:
    """
    Complete pipeline: audio → 5 stems → mel-spectrogram NPZ.

    This is the main entry point for Step 1 of the Loop Extractor pipeline.

    Parameters
    ----------
    audio_path : str
        Path to input audio file
    stems_output_dir : str
        Directory where stems will be saved
    npz_output_path : str
        Path for output NPZ file

    Returns
    -------
    tuple of (Path, Path)
        (stems_directory, npz_file_path)

    Examples
    --------
    >>> stems_dir, npz_file = process_audio_to_stems_and_npz(
    ...     'track.wav',
    ...     'output/stems',
    ...     'output/track_5stems.npz'
    ... )
    >>> print(f"Stems: {stems_dir}, NPZ: {npz_file}")
    """
    # Step 1: Separate stems
    stems_dir = separate_stems(audio_path, stems_output_dir)

    # Force garbage collection after Spleeter (it holds a lot of memory)
    import gc
    gc.collect()

    # Step 2: Convert to NPZ
    npz_file = create_5stem_npz(stems_dir, npz_output_path)

    return stems_dir, npz_file


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python spleeter_interface.py <audio_file> <stems_output_dir> <npz_output_path>")
        sys.exit(1)

    audio_file = sys.argv[1]
    stems_out = sys.argv[2]
    npz_out = sys.argv[3]

    stems_dir, npz_path = process_audio_to_stems_and_npz(audio_file, stems_out, npz_out)

    print(f"\nCompleted!")
    print(f"Stems directory: {stems_dir}")
    print(f"NPZ file: {npz_path}")
