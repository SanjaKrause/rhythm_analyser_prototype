#!/usr/bin/env python3
"""
DrumTranscriber Interface - Classify drum onsets into 6 drum classes.

This module provides an interface to the DrumTranscriber model for classifying
drum hits in audio into: crash, hihat, kick_drum, ride, snare, tom.

Environment: Base (numpy, pandas, librosa, tensorflow)
"""

import sys
import os
from pathlib import Path

# Add drumtranscriber to path
DRUMTRANSCRIBER_PATH = Path(__file__).parent.parent.parent / 'drumtranscriber'
sys.path.insert(0, str(DRUMTRANSCRIBER_PATH))

# Try to import dependencies
try:
    import numpy as np
    import pandas as pd
    import librosa
    from DrumTranscriber import DrumTranscriber
    DRUMTRANSCRIBER_AVAILABLE = True
except ImportError as e:
    DRUMTRANSCRIBER_AVAILABLE = False
    # Don't print warning on import - only when function is called
    _IMPORT_ERROR = str(e)


def transcribe_drums(
    audio_path: str,
    output_dir: str,
    track_id: str,
    sr: int = 44100
) -> dict:
    """
    Transcribe drum hits in audio file using DrumTranscriber.

    Parameters
    ----------
    audio_path : str
        Path to audio file (WAV or MP3)
    output_dir : str
        Output directory for saving results
    track_id : str
        Track identifier for output files
    sr : int
        Sample rate to use for loading audio (default: 44100)

    Returns
    -------
    dict
        Dictionary with paths to output files and statistics
        {
            'predictions_csv': Path to CSV with predictions,
            'summary': Statistics dictionary
        }
    """
    if not DRUMTRANSCRIBER_AVAILABLE:
        raise RuntimeError("DrumTranscriber is not available. Check installation.")

    print(f"\n{'='*80}")
    print(f"DRUM TRANSCRIPTION - {track_id}")
    print(f"{'='*80}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    print(f"\n  Loading audio: {audio_path}")
    samples, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(samples) / loaded_sr
    print(f"    Duration: {duration:.2f}s, Sample rate: {loaded_sr}Hz")

    # Initialize transcriber
    print(f"\n  Initializing DrumTranscriber...")
    transcriber = DrumTranscriber()

    # Predict drum classes
    print(f"  Detecting and classifying drum hits...")
    predictions_df = transcriber.predict(samples, loaded_sr)

    # Add track_id column
    predictions_df.insert(0, 'track_id', track_id)

    # Sort by time
    predictions_df = predictions_df.sort_values('time').reset_index(drop=True)

    # Calculate statistics
    total_hits = len(predictions_df)

    # Get class with highest probability for each hit
    class_columns = ['crash', 'hihat_c', 'kick_drum', 'ride', 'snare', 'tom_h']
    predictions_df['predicted_class'] = predictions_df[class_columns].idxmax(axis=1)
    predictions_df['predicted_probability'] = predictions_df[class_columns].max(axis=1)

    # Count by class (using highest probability classification)
    class_counts = predictions_df['predicted_class'].value_counts().to_dict()

    # Average probabilities per class (convert to Python float for JSON serialization)
    avg_probs = {col: float(predictions_df[col].mean()) for col in class_columns}

    # Print summary
    print(f"\n  Results:")
    print(f"    Total drum hits detected: {total_hits}")
    print(f"\n    Hits per class (highest probability):")
    for drum_class in class_columns:
        count = class_counts.get(drum_class, 0)
        avg_prob = avg_probs[drum_class]
        percentage = (count / total_hits * 100) if total_hits > 0 else 0
        print(f"      {drum_class:12s}: {count:4d} hits ({percentage:5.1f}%), avg prob: {avg_prob:.3f}")

    # Save predictions CSV (full probabilities)
    output_csv = output_dir / f'{track_id}_drum_transcription.csv'
    predictions_df.to_csv(output_csv, index=False)
    print(f"\n  Saved: {output_csv}")

    # Create timeline CSV (sparse format: only predicted drum class per timestamp)
    timeline_df = pd.DataFrame()
    timeline_df['time'] = predictions_df['time']

    # Initialize all drum columns as empty strings
    for drum_class in class_columns:
        timeline_df[drum_class] = ''

    # Fill in the predicted drum class with its probability
    for idx, row in predictions_df.iterrows():
        predicted_drum = row['predicted_class']
        predicted_prob = row['predicted_probability']
        timeline_df.at[idx, predicted_drum] = f"{predicted_prob:.4f}"

    # Save timeline CSV
    timeline_csv = output_dir / f'{track_id}_drum_timeline.csv'
    timeline_df.to_csv(timeline_csv, index=False)
    print(f"  Saved: {timeline_csv}")

    # Create onset-compatible CSV (for use with onset_mode='drumtranscriber')
    # Format matches Step 4 onset detection output: single column 'onset_times'
    onsets_df = pd.DataFrame()
    onsets_df['onset_times'] = predictions_df['time']

    # Save onset-compatible CSV
    onsets_csv = output_dir / f'{track_id}_onsets.csv'
    onsets_df.to_csv(onsets_csv, index=False)
    print(f"  Saved: {onsets_csv}")

    # Create summary statistics
    summary = {
        'total_hits': total_hits,
        'duration': duration,
        'hits_per_second': total_hits / duration if duration > 0 else 0,
        'class_counts': class_counts,
        'average_probabilities': avg_probs
    }

    # Save summary JSON
    import json
    summary_json = output_dir / f'{track_id}_drum_transcription_summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_json}")

    print(f"\n{'='*80}")

    return {
        'predictions_csv': str(output_csv),
        'timeline_csv': str(timeline_csv),
        'onsets_csv': str(onsets_csv),
        'summary_json': str(summary_json),
        'summary': summary
    }


def filter_close_onsets(
    onsets_csv: str,
    tempo_csv: str,
    output_csv: str,
    min_interval_16th_fraction: float = 0.5
) -> dict:
    """
    Filter out onsets that are too close together.

    When two consecutive onsets are closer than min_interval_16th_fraction * (1/16th note),
    the FIRST onset is removed, keeping the SECOND onset.

    Parameters
    ----------
    onsets_csv : str
        Path to input onsets CSV (single column 'onset_times')
    tempo_csv : str
        Path to tempo CSV with average BPM
    output_csv : str
        Path to save filtered onsets CSV
    min_interval_16th_fraction : float
        Minimum interval as fraction of 16th note (default: 0.5 = 1/32nd note)
        Set to 0.5 to filter onsets closer than 1/32nd note
        Set to 0.25 to filter onsets closer than 1/64th note
        Set to 1.0 to filter onsets closer than 1/16th note

    Returns
    -------
    dict
        Statistics about filtering:
        {
            'original_count': int,
            'filtered_count': int,
            'removed_count': int,
            'avg_bpm': float,
            'min_interval_ms': float
        }
    """
    import pandas as pd
    import numpy as np

    # Load onsets
    onsets_df = pd.read_csv(onsets_csv)
    onset_times = onsets_df['onset_times'].values

    if len(onset_times) == 0:
        # No onsets to filter
        onsets_df.to_csv(output_csv, index=False)
        return {
            'original_count': 0,
            'filtered_count': 0,
            'removed_count': 0,
            'avg_bpm': 0,
            'min_interval_ms': 0
        }

    # Load tempo to calculate minimum interval
    tempo_df = pd.read_csv(tempo_csv)
    avg_bpm = tempo_df['avg_bpm'].iloc[0]  # Get average BPM

    # Calculate minimum interval
    # At BPM, 1 beat = 60/BPM seconds
    # 1 sixteenth note = (60/BPM) / 4 seconds
    # min_interval = fraction * (1/16th note)
    sixteenth_note_duration = (60.0 / avg_bpm) / 4.0
    min_interval = min_interval_16th_fraction * sixteenth_note_duration
    min_interval_ms = min_interval * 1000

    print(f"\n  Filtering close onsets:")
    print(f"    Average BPM: {avg_bpm:.1f}")
    print(f"    1/16th note duration: {sixteenth_note_duration*1000:.1f} ms")
    print(f"    Minimum interval: {min_interval_ms:.1f} ms ({min_interval_16th_fraction} * 1/16th)")

    # Filter onsets: remove first onset if two consecutive onsets are too close
    keep_indices = []
    i = 0
    while i < len(onset_times):
        # Check if next onset exists and is too close
        if i + 1 < len(onset_times):
            interval = onset_times[i + 1] - onset_times[i]
            if interval < min_interval:
                # Skip current onset (it's too close to the next one)
                # Don't add to keep_indices, move to next
                i += 1
                continue

        # Keep this onset
        keep_indices.append(i)
        i += 1

    # Create filtered onset times array
    filtered_times = onset_times[keep_indices]

    # Create filtered DataFrame
    filtered_df = pd.DataFrame()
    filtered_df['onset_times'] = filtered_times

    # Save filtered onsets
    filtered_df.to_csv(output_csv, index=False)

    # Statistics
    original_count = len(onset_times)
    filtered_count = len(filtered_times)
    removed_count = original_count - filtered_count

    print(f"    Original onsets: {original_count}")
    print(f"    Filtered onsets: {filtered_count}")
    print(f"    Removed: {removed_count} ({removed_count/original_count*100:.1f}%)")

    return {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'removed_count': removed_count,
        'avg_bpm': float(avg_bpm),
        'min_interval_ms': float(min_interval_ms)
    }


def transcribe_stems(
    stems_dir: str,
    output_dir: str,
    track_id: str,
    stem_names: list = None,
    sr: int = 44100
) -> dict:
    """
    Transcribe drums from individual stems.

    Particularly useful for analyzing the drums stem separately.

    Parameters
    ----------
    stems_dir : str
        Directory containing stem files
    output_dir : str
        Output directory for saving results
    track_id : str
        Track identifier
    stem_names : list, optional
        List of stem names to process (default: ['drums'])
    sr : int
        Sample rate (default: 44100)

    Returns
    -------
    dict
        Dictionary mapping stem names to their transcription results
    """
    if stem_names is None:
        stem_names = ['drums']

    stems_dir = Path(stems_dir)
    results = {}

    for stem_name in stem_names:
        stem_path = stems_dir / f'{stem_name}.wav'

        if not stem_path.exists():
            print(f"Warning: Stem not found: {stem_path}")
            continue

        # Create stem-specific output directory
        stem_output_dir = Path(output_dir) / stem_name
        stem_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = transcribe_drums(
                str(stem_path),
                str(stem_output_dir),
                f"{track_id}_{stem_name}",
                sr=sr
            )
            results[stem_name] = result
        except Exception as e:
            print(f"Error transcribing {stem_name}: {e}")
            results[stem_name] = {'error': str(e)}

    return results


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python drumtranscriber_interface.py <audio_path> <output_dir> <track_id>')
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = sys.argv[2]
    track_id = sys.argv[3]

    transcribe_drums(audio_path, output_dir, track_id)
