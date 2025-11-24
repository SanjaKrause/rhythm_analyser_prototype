"""
Beat-Transformer wrapper that calls beat detection via subprocess.

This module runs in AEinBOX_13_3 and calls run_transformer.py in new_beatnet_env
via subprocess to avoid madmom compatibility issues.

Environment: AEinBOX_13_3 (calls new_beatnet_env subprocess)
"""

import subprocess
from pathlib import Path
from typing import Optional
import sys

# Import config
import importlib.util
_config_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("config_module", _config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config


def detect_beats_and_downbeats(
    npz_path: str,
    output_path: str,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
    fps: Optional[float] = None,
    min_bpm: Optional[float] = None,
    max_bpm: Optional[float] = None,
    verbose: bool = True
) -> Path:
    """
    Detect beats and downbeats using Beat-Transformer via subprocess.

    This function calls run_transformer.py in the new_beatnet_env environment
    to perform beat/downbeat detection using madmom.

    Parameters
    ----------
    npz_path : str
        Path to input NPZ file with 5-stem mel-spectrograms
    output_path : str
        Path for output beat data text file
    checkpoint_path : str, optional
        Path to Beat-Transformer checkpoint (default: from config)
    device : str, optional
        Device to use ('cpu' or 'cuda'), default 'cpu'
    fps : float, optional
        Frames per second (default: from config)
    min_bpm : float, optional
        Minimum BPM (default: from config)
    max_bpm : float, optional
        Maximum BPM (default: from config)
    verbose : bool, optional
        Print subprocess output (default: True)

    Returns
    -------
    Path
        Path to output file

    Raises
    ------
    FileNotFoundError
        If checkpoint or NPZ file not found
    subprocess.CalledProcessError
        If beat detection subprocess fails

    Examples
    --------
    >>> output = detect_beats_and_downbeats(
    ...     'track_5stems.npz',
    ...     'track_output.txt'
    ... )
    >>> print(f"Beat data saved to: {output}")
    """
    # Use config defaults
    if checkpoint_path is None:
        checkpoint_path = str(config.BEAT_TRANSFORMER_CHECKPOINT)
    if fps is None:
        fps = config.DBN_FPS
    if min_bpm is None:
        min_bpm = config.DBN_MIN_BPM
    if max_bpm is None:
        max_bpm = config.DBN_MAX_BPM

    # Validate inputs
    npz_path = Path(npz_path)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {npz_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Path to run_transformer.py
    run_script = Path(__file__).parent / "run_transformer.py"

    # Build command
    cmd = [
        config.BEAT_DETECTION_PYTHON,
        str(run_script),
        '--input', str(npz_path),
        '--output', str(output_path),
        '--checkpoint', str(checkpoint_path),
        '--device', device,
        '--fps', str(fps),
        '--min-bpm', str(min_bpm),
        '--max-bpm', str(max_bpm)
    ]

    if verbose:
        print(f"\nRunning Beat-Transformer in {config.BEAT_DETECTION_ENV}...")
        print(f"Command: {' '.join(cmd[:3])} [...]")

    # Run subprocess
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=not verbose,
            text=True
        )

        if not verbose and result.stdout:
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"\nError running beat detection subprocess:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        raise

    if verbose:
        print(f"Beat detection completed: {output_path}")

    return output_path


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Beat-Transformer wrapper')
    parser.add_argument('--input', required=True, help='Input NPZ file')
    parser.add_argument('--output', required=True, help='Output text file')
    parser.add_argument('--checkpoint', help='Checkpoint path (optional)')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')

    args = parser.parse_args()

    output = detect_beats_and_downbeats(
        args.input,
        args.output,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    print(f"\nSuccess! Output: {output}")
