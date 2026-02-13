#!/usr/bin/env python3
"""
Pironio pulse clarity metrics runner - subprocess script.

This script runs in new_beatnet_env (which has madmom) and computes
all Pironio/MAIPC pulse clarity metrics for an audio file.

Environment: new_beatnet_env (same as Beat-Transformer)
"""

import sys
import json
import argparse
from pathlib import Path

# Add pironio-python to path
pironio_path = Path(__file__).parent.parent / 'pironio-python'
if pironio_path.exists():
    sys.path.insert(0, str(pironio_path))


def compute_metrics(audio_path: str, downbeat_model: bool = True, compute_slow: bool = True) -> dict:
    """
    Compute all Pironio pulse clarity metrics.

    Returns dict with metrics and any errors.
    """
    import maipc

    results = {
        'metrics': {},
        'errors': []
    }

    # Fast metrics
    fast_metrics = [
        ('viterbi_max', maipc.viterbi_max),
        ('viterbi_entropy', maipc.viterbi_entropy),
        ('peak_average', maipc.peak_average),
        ('RNN_entropy', maipc.RNN_entropy),
        ('DBN_entropy', maipc.DBN_entropy),
    ]

    # Slow metrics
    slow_metrics = [
        ('neurons_cross_correlation', maipc.neurons_cross_correlation),
        ('cell_states_precision', maipc.cell_states_precision),
        ('autocorrelation_periodicity', maipc.autocorrelation_periodicity),
    ]

    # Compute fast metrics
    for metric_name, metric_func in fast_metrics:
        try:
            print(f"    Computing {metric_name}...")
            value = metric_func(audio_path, downbeat_model=downbeat_model)
            results['metrics'][metric_name] = float(value)
            print(f"      {metric_name} = {value:.6f}")
        except Exception as e:
            error_msg = f"{metric_name}: {str(e)}"
            results['errors'].append(error_msg)
            print(f"      ERROR: {error_msg}")

    # Compute slow metrics if requested
    if compute_slow:
        print(f"    Computing slow metrics...")
        for metric_name, metric_func in slow_metrics:
            try:
                print(f"    Computing {metric_name}...")
                value = metric_func(audio_path, downbeat_model=downbeat_model)
                results['metrics'][metric_name] = float(value)
                print(f"      {metric_name} = {value:.6f}")
            except Exception as e:
                error_msg = f"{metric_name}: {str(e)}"
                results['errors'].append(error_msg)
                print(f"      ERROR: {error_msg}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute Pironio pulse clarity metrics')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument('--track-id', required=True, help='Track identifier')
    parser.add_argument('--beat-model', action='store_true', help='Use beat model instead of downbeat')
    parser.add_argument('--fast-only', action='store_true', help='Skip slow metrics')

    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_path = Path(args.output)

    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Pironio Pulse Clarity] Computing metrics...")
    print(f"  Audio: {audio_path.name}")
    print(f"  Model: {'beat' if args.beat_model else 'downbeat'}")

    # Compute metrics
    results = compute_metrics(
        str(audio_path),
        downbeat_model=not args.beat_model,
        compute_slow=not args.fast_only
    )

    # Add metadata
    results['track_id'] = args.track_id
    results['audio_path'] = str(audio_path)
    results['model'] = 'beat' if args.beat_model else 'downbeat'

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {output_path.name}")

    num_metrics = len(results['metrics'])
    num_errors = len(results['errors'])
    print(f"  Computed {num_metrics} metrics ({num_errors} errors)")

    return 0 if not results['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
