#!/usr/bin/env python3
"""
Pironio pulse clarity metrics runner - subprocess script.

This script runs in new_beatnet_env (which has madmom) and computes
all Pironio/MAIPC pulse clarity metrics for an audio file.

Can also process individual sections from 9.1_sections folder.

Environment: new_beatnet_env (same as Beat-Transformer)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

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


def compute_section_metrics(
    sections_dir: Path,
    output_dir: Path,
    track_id: str,
    downbeat_model: bool = True,
    compute_slow: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Compute Pironio metrics for all sections in a 9.1_sections folder.

    Parameters
    ----------
    sections_dir : Path
        Path to 9.1_sections directory containing section WAV files
    output_dir : Path
        Output directory for JSON results
    track_id : str
        Track identifier
    downbeat_model : bool
        Use downbeat model (True) or beat model (False)
    compute_slow : bool
        Compute slow metrics (neurons_cross_correlation, etc.)
    verbose : bool
        Print progress messages

    Returns
    -------
    Dict
        Combined results for all sections
    """
    sections_dir = Path(sections_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all section WAV files (not macOS ._ files)
    section_wavs = sorted([
        f for f in sections_dir.glob('*_section.wav')
        if not f.name.startswith('._')
    ])

    if verbose:
        print(f"\n[Pironio Section Metrics]")
        print(f"  Input: {sections_dir.name}")
        print(f"  Found {len(section_wavs)} section files")
        print(f"  Model: {'downbeat' if downbeat_model else 'beat'}")

    all_results = {
        'track_id': track_id,
        'model': 'downbeat' if downbeat_model else 'beat',
        'sections': {}
    }

    for wav_path in section_wavs:
        # Extract section ID from filename
        # e.g., SecNo1_L4_chorus_0.1344_section.wav -> SecNo1_L4_chorus_0.1344
        section_id = wav_path.stem.replace('_section', '')

        if verbose:
            print(f"\n  Processing: {section_id}")

        # Compute metrics for this section
        section_results = compute_metrics(
            str(wav_path),
            downbeat_model=downbeat_model,
            compute_slow=compute_slow
        )

        # Store results
        all_results['sections'][section_id] = {
            'audio_file': wav_path.name,
            'metrics': section_results['metrics'],
            'errors': section_results['errors']
        }

    # Save combined results
    output_file = output_dir / f'{track_id}_pironio_sections.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        total_sections = len(all_results['sections'])
        total_errors = sum(len(s['errors']) for s in all_results['sections'].values())
        print(f"\n  Saved: {output_file.name}")
        print(f"  Processed {total_sections} sections ({total_errors} total errors)")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Compute Pironio pulse clarity metrics')
    parser.add_argument('--audio', help='Path to audio file (for single file mode)')
    parser.add_argument('--output', help='Path to output JSON file (for single file mode)')
    parser.add_argument('--track-id', required=True, help='Track identifier')
    parser.add_argument('--beat-model', action='store_true', help='Use beat model instead of downbeat')
    parser.add_argument('--fast-only', action='store_true', help='Skip slow metrics')
    parser.add_argument('--sections-dir', help='Path to 9.1_sections directory (for section mode)')
    parser.add_argument('--output-dir', help='Output directory for section results')

    args = parser.parse_args()

    # Determine mode: sections or single file
    if args.sections_dir:
        # Section mode: process all sections in 9.1_sections folder
        sections_dir = Path(args.sections_dir)
        if not sections_dir.exists():
            print(f"ERROR: Sections directory not found: {sections_dir}")
            sys.exit(1)

        output_dir = Path(args.output_dir) if args.output_dir else sections_dir.parent / '12_pironio'

        results = compute_section_metrics(
            sections_dir=sections_dir,
            output_dir=output_dir,
            track_id=args.track_id,
            downbeat_model=not args.beat_model,
            compute_slow=not args.fast_only,
            verbose=True
        )

        total_errors = sum(len(s['errors']) for s in results['sections'].values())
        return 0 if total_errors == 0 else 1

    else:
        # Single file mode (original behavior)
        if not args.audio or not args.output:
            print("ERROR: --audio and --output required for single file mode")
            print("       Use --sections-dir for section mode")
            sys.exit(1)

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
