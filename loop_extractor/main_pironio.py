#!/usr/bin/env python3
"""
Step 12: Pironio Pulse Clarity Metrics

This module computes pulse clarity metrics from the MAIPC library
(Madmom Applied In Pulse Clarity) by Nicolás Pironio.

These metrics quantify how clear/strong the pulse is in an audio file,
useful for analyzing rhythmic clarity and beat strength.

Metrics computed:
- viterbi_max: Maximum Viterbi path probability
- viterbi_entropy: Entropy of Viterbi probabilities
- peak_average: Average peak activation values
- RNN_entropy: Entropy of RNN peak moments
- DBN_entropy: Entropy of DBN beat moments
- neurons_cross_correlation: Cross-correlation between RNN neurons
- cell_states_precision: LSTM cell state peak width analysis
- autocorrelation_periodicity: Autocorrelation of neuron outputs

Output folder: 12_pironio/

Reference:
    Pironio, N. et al. - Pulse clarity metrics from deep learning beat tracking models
    https://github.com/nPironio/maipc

Environment: Runs via subprocess in new_beatnet_env (requires madmom)
"""

import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Optional, Any

# Import config
import importlib.util
_config_path = Path(__file__).parent / "config.py"
spec = importlib.util.spec_from_file_location("config_module", _config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config


def run_pironio_analysis(
    audio_file: str,
    output_dir: str,
    track_id: Optional[str] = None,
    downbeat_model: bool = True,
    compute_slow_metrics: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Pironio pulse clarity analysis via subprocess.

    This function calls run_pironio.py in the new_beatnet_env environment
    to compute pulse clarity metrics using madmom.

    Parameters
    ----------
    audio_file : str
        Path to the audio file (WAV or MP3)
    output_dir : str
        Output directory for the track (will create 12_pironio subfolder)
    track_id : str, optional
        Track identifier (derived from filename if not provided)
    downbeat_model : bool
        Use downbeat model (default True)
    compute_slow_metrics : bool
        Compute slow RNN metrics (default True)
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results dictionary with metrics and output paths
    """
    audio_path = Path(audio_file)

    if track_id is None:
        track_id = audio_path.stem

    # Create output directory
    output_path = Path(output_dir) / '12_pironio'
    output_path.mkdir(parents=True, exist_ok=True)

    output_json = output_path / f'{track_id}_pironio_metrics.json'

    # Path to run_pironio.py script
    run_script = Path(__file__).parent / "run_pironio.py"

    # Build command
    cmd = [
        config.BEAT_DETECTION_PYTHON,  # Same Python as Beat-Transformer (has madmom)
        str(run_script),
        '--audio', str(audio_path),
        '--output', str(output_json),
        '--track-id', track_id
    ]

    if not downbeat_model:
        cmd.append('--beat-model')

    if not compute_slow_metrics:
        cmd.append('--fast-only')

    if verbose:
        print(f"\n[Step 12: Pironio Pulse Clarity] Computing metrics...")
        print(f"  Audio: {audio_path.name}")
        print(f"  Model: {'downbeat' if downbeat_model else 'beat'}")
        print(f"  Running in {config.BEAT_DETECTION_ENV}...")

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
        print(f"\nError running Pironio subprocess:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        raise

    # Load results from JSON
    if output_json.exists():
        with open(output_json, 'r') as f:
            results = json.load(f)
        results['output_json'] = str(output_json)
    else:
        results = {
            'track_id': track_id,
            'audio_path': str(audio_path),
            'metrics': {},
            'errors': ['Output JSON not created'],
            'output_json': str(output_json)
        }

    if verbose:
        num_metrics = len(results.get('metrics', {}))
        num_errors = len(results.get('errors', []))
        print(f"  ✓ Computed {num_metrics} metrics ({num_errors} errors)")

    return results


def main():
    """Command-line interface for Pironio pulse clarity analysis."""
    parser = argparse.ArgumentParser(
        description='Step 12: Compute Pironio pulse clarity metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_pironio.py audio.wav -o output/
  python main_pironio.py audio.wav -o output/ --beat-model
  python main_pironio.py audio.wav -o output/ --fast-only
        """
    )

    parser.add_argument('audio_file', help='Path to audio file (WAV or MP3)')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--track-id', help='Track identifier (default: filename)')
    parser.add_argument('--beat-model', action='store_true',
                        help='Use beat model instead of downbeat model')
    parser.add_argument('--fast-only', action='store_true',
                        help='Skip slow RNN metrics')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    results = run_pironio_analysis(
        audio_file=args.audio_file,
        output_dir=args.output_dir,
        track_id=args.track_id,
        downbeat_model=not args.beat_model,
        compute_slow_metrics=not args.fast_only,
        verbose=not args.quiet
    )

    if not args.quiet:
        print(f"\nResults saved to: {results.get('output_json', 'N/A')}")

    return 0 if not results.get('errors') else 1


if __name__ == '__main__':
    sys.exit(main())
