#!/usr/bin/env python3
"""
AP2 Analysis Pipeline - Main Orchestrator

Complete pipeline for music microtiming analysis:
1. Stem separation (Spleeter)
2. Beat detection (Beat-Transformer via subprocess)
3. Downbeat correction
3.5. Tempo plots (8-panel comparison: uncorrected vs corrected, + bar tempo CSV)
4. Onset detection (librosa, from drum stem)
4.5. Pattern length detection (drum/mel/pitch methods with circular convolution)
5. Raster/grid calculations
6. RMS histogram analysis
7. Audio example generation
8. MIDI export (actual onset times, one loop per method: drum, mel, pitch)
9. Stem loop export (WAV loops for each stem, one loop per method: drum, mel, pitch)

Environment: AEinBOX_13_3 (main)
Subprocess: new_beatnet_env (for beat detection only)

Required Pretrained Models:
    1. Spleeter 5-stem model (automatically downloaded on first use)
       - Location: Will be cached by Spleeter in system cache directory
       - Configured in: config.py (SPLEETER_MODEL = 'spleeter:5stems')

    2. Beat-Transformer checkpoint
       - Location: Configured in config.py (BEAT_TRANSFORMER_CHECKPOINT)
       - Download from: Beat-Transformer repository

    3. libf0 (for pitch detection)
       - Installation: pip install libf0
       - No separate model files needed

Usage:
    python main.py --audio track.wav --track-id 123 --output-dir output/
    python main.py --batch --input-dir audio/ --output-dir output/ --start-id 0 --end-id 100
"""

import argparse
from pathlib import Path
import sys
import json
import importlib.util
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import config
spec = importlib.util.spec_from_file_location("config_module", Path(__file__).parent / "config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config

# Import all pipeline modules
from stem_separation import spleeter_interface
from beat_detection import transformer
from analysis import correct_bars, raster, rms_grid_histograms, onset_detection, pattern_detection, tempo_plots
from utils import audio_export, raster_plots, midi_export


def run_complete_pipeline(
    audio_file: str,
    track_id: str,
    output_dir: str,
    pattern_file: Optional[str] = None,
    snippet_offset_file: Optional[str] = None,
    onset_file: Optional[str] = None,
    skip_existing: bool = True,
    create_audio_examples: bool = True,
    verbose: bool = True
) -> dict:
    """
    Run complete AP2 analysis pipeline for a single track.

    Parameters
    ----------
    audio_file : str
        Path to input audio file
    track_id : str
        Track identifier
    output_dir : str
        Base output directory
    pattern_file : str, optional
        Path to pattern lengths CSV
    snippet_offset_file : str, optional
        Path to snippet offsets CSV
    onset_file : str, optional
        Path to onsets CSV (if None, will need onset detection)
    skip_existing : bool
        Skip steps if output files already exist
    create_audio_examples : bool
        Create MP3 examples with click tracks
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Pipeline results summary

    Raises
    ------
    FileNotFoundError
        If required input files don't exist
    """
    from typing import Optional

    if verbose:
        print("=" * 80)
        print(f"AP2 Pipeline - Track {track_id}")
        print("=" * 80)

    # Validate inputs
    audio_file = Path(audio_file)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Get output paths
    paths = config.get_output_paths(track_id, Path(output_dir))

    # Create directories
    config.create_output_directories(track_id, Path(output_dir))

    results = {
        'track_id': track_id,
        'audio_file': str(audio_file),
        'steps_completed': [],
        'errors': []
    }

    # ========================================================================
    # STEP 1: STEM SEPARATION
    # ========================================================================
    try:
        if skip_existing and paths['npz_file'].exists():
            if verbose:
                print("\n[1/7] Stem separation - SKIPPED (exists)")
            results['steps_completed'].append('stem_separation_skipped')
        else:
            if verbose:
                print("\n[1/7] Stem separation (Spleeter)...")

            stems_dir, npz_file = spleeter_interface.process_audio_to_stems_and_npz(
                str(audio_file),
                str(paths['stems_dir']),
                str(paths['npz_file'])
            )

            results['stems_dir'] = str(stems_dir)
            results['npz_file'] = str(npz_file)
            results['steps_completed'].append('stem_separation')

            if verbose:
                print(f"  ✓ Stems and NPZ created")

    except Exception as e:
        error_msg = f"Step 1 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        raise

    # ========================================================================
    # STEP 2: BEAT DETECTION
    # ========================================================================
    try:
        if skip_existing and paths['beats_file'].exists():
            if verbose:
                print("\n[2/7] Beat detection - SKIPPED (exists)")
            results['steps_completed'].append('beat_detection_skipped')
        else:
            if verbose:
                print("\n[2/7] Beat detection (Beat-Transformer)...")

            beats_file = transformer.detect_beats_and_downbeats(
                str(paths['npz_file']),
                str(paths['beats_file']),
                verbose=verbose
            )

            results['beats_file'] = str(beats_file)
            results['steps_completed'].append('beat_detection')

            if verbose:
                print(f"  ✓ Beat detection completed")

    except Exception as e:
        error_msg = f"Step 2 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        raise

    # ========================================================================
    # STEP 3: CORRECT DOWNBEATS
    # ========================================================================
    try:
        if skip_existing and paths['corrected_downbeats_file'].exists():
            if verbose:
                print("\n[3/7] Downbeat correction - SKIPPED (exists)")
            results['steps_completed'].append('correct_bars_skipped')
        else:
            if verbose:
                print("\n[3/7] Downbeat correction...")

            stats = correct_bars.correct_downbeats(
                str(paths['beats_file']),
                str(paths['corrected_downbeats_file']),
                verbose=verbose
            )

            results['correction_stats'] = stats
            results['steps_completed'].append('correct_bars')

            if verbose and stats:
                print(f"  ✓ Corrected: {stats['raw_bars']} → {stats['corrected_bars']} bars")

    except Exception as e:
        error_msg = f"Step 3 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        raise

    # ========================================================================
    # STEP 3.5: TEMPO PLOTS
    # ========================================================================
    try:
        if skip_existing and paths['tempo_plots_pdf'].exists():
            if verbose:
                print("\n[3.5/7] Tempo plots - SKIPPED (exists)")
            results['steps_completed'].append('tempo_plots_skipped')
        else:
            if verbose:
                print("\n[3.5/7] Generating tempo plots...")

            # Load snippet offset
            if snippet_offset_file and Path(snippet_offset_file).exists():
                snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
            elif config.OVERVIEW_CSV.exists():
                snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
            else:
                snippet_offset = None

            # Create tempo plots
            tempo_files = tempo_plots.create_tempo_plots(
                str(paths['beats_file']),
                str(paths['corrected_downbeats_file']),
                str(paths['tempo_plots_dir']),
                track_id,
                snippet_start=snippet_offset,
                snippet_duration=config.CORRECT_BARS_SNIPPET_DURATION_S
            )

            results['tempo_plots_pdf'] = tempo_files['plot_pdf']
            results['tempo_csv'] = tempo_files['csv']
            results['steps_completed'].append('tempo_plots')

            if verbose:
                print(f"  ✓ Tempo plots created")
                print(f"    PDF: {Path(tempo_files['plot_pdf']).name}")
                print(f"    CSV: {Path(tempo_files['csv']).name}")

    except Exception as e:
        error_msg = f"Step 3.5 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue with pipeline

    # ========================================================================
    # STEP 4: ONSET DETECTION
    # ========================================================================
    try:
        # Determine onset file path
        if onset_file is None:
            onset_file = paths['onsets_file']

        if skip_existing and Path(onset_file).exists():
            if verbose:
                print("\n[4/7] Onset detection - SKIPPED (exists)")
            results['steps_completed'].append('onset_detection_skipped')
        else:
            if verbose:
                print("\n[4/7] Onset detection from drum stem...")

            # Detect onsets from drum stem (created in Step 1)
            drum_stem = paths['stems_dir'] / 'drums.wav'

            if not drum_stem.exists():
                raise FileNotFoundError(f"Drum stem not found: {drum_stem}")

            onsets, onset_file_path = onset_detection.detect_and_save_onsets(
                str(drum_stem),
                str(onset_file),
                hop_length=512,
                backtrack=False,
                delta=0.12,
                refine_onsets=False,
                min_interval_s=0.15,
                sr=22050
            )

            results['onset_file'] = str(onset_file_path)
            results['num_onsets'] = len(onsets)
            results['steps_completed'].append('onset_detection')

            if verbose:
                print(f"  ✓ Detected {len(onsets)} onsets")
                print(f"  ✓ Saved to: {onset_file_path}")

    except Exception as e:
        error_msg = f"Step 4 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        raise

    # ========================================================================
    # STEP 4.5: PATTERN LENGTH DETECTION
    # ========================================================================
    try:
        if verbose:
            print("\n[4.5/7] Pattern length detection...")

        # Load pattern lengths from file if provided
        if pattern_file and Path(pattern_file).exists():
            pattern_lengths = raster.load_pattern_lengths(pattern_file, track_id)
            if verbose:
                print(f"    Loaded from file: {pattern_lengths}")
            results['steps_completed'].append('pattern_detection_loaded')
        else:
            # Detect pattern lengths using all 3 methods
            if verbose:
                print(f"    Detecting pattern lengths using drum/mel/pitch methods...")

            # Get required paths
            drum_stem = paths['stems_dir'] / 'drums.wav'
            bass_stem = paths['stems_dir'] / 'bass.wav'

            if not drum_stem.exists():
                raise FileNotFoundError(f"Drum stem not found: {drum_stem}")
            if not bass_stem.exists():
                raise FileNotFoundError(f"Bass stem not found: {bass_stem}")

            # Load corrected downbeats to get bar times
            import pandas as pd
            df_corrected = pd.read_csv(paths['corrected_downbeats_file'], sep='\t', comment='#')
            bar_starts = df_corrected['corrected_downbeat_time(s)'].values
            bar_ends = df_corrected['next_downbeat_time(s)'].values

            # Get time signature from corrected file
            tsig = 4  # Default
            with open(paths['corrected_downbeats_file'], 'r') as f:
                for line in f:
                    if '# time_signature' in line:
                        import re
                        m = re.search(r'time_signature\s*=\s*(\d+)', line)
                        if m:
                            tsig = int(m.group(1))
                            break

            # Get snippet offset if available
            snippet = None
            if snippet_offset_file and Path(snippet_offset_file).exists():
                snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
                snippet = (snippet_offset, snippet_offset + 30.0)  # 30s snippet
            elif config.OVERVIEW_CSV.exists():
                snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                if snippet_offset > 0:
                    snippet = (snippet_offset, snippet_offset + 30.0)

            # Filter bars to snippet if provided
            if snippet:
                s0, s1 = snippet
                mask = (bar_starts < s1) & (bar_ends > s0)
                bar_starts_snip = bar_starts[mask]
                bar_ends_snip = bar_ends[mask]
            else:
                bar_starts_snip = bar_starts
                bar_ends_snip = bar_ends

            # Onset file should exist from Step 4
            if onset_file is None:
                onset_file = paths['onsets_file']

            # Run pattern detection
            pattern_lengths = pattern_detection.detect_pattern_lengths(
                onset_csv_path=str(onset_file),
                drums_wav_path=str(drum_stem),
                bass_wav_path=str(bass_stem),
                bar_starts=bar_starts_snip,
                bar_ends=bar_ends_snip,
                tsig=tsig,
                snippet=snippet
            )

            results['pattern_lengths'] = pattern_lengths
            results['steps_completed'].append('pattern_detection')

            if verbose:
                print(f"  ✓ Pattern lengths detected: {pattern_lengths}")

    except Exception as e:
        # Fall back to defaults if pattern detection fails
        pattern_lengths = {'drum': 4, 'mel': 4, 'pitch': 4}
        error_msg = f"Step 4.5 failed: {e} - using defaults {pattern_lengths}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ⚠ WARNING: {e}")
            print(f"  Using default pattern lengths: {pattern_lengths}")

    # ========================================================================
    # STEP 5: RASTER/GRID CALCULATIONS
    # ========================================================================
    try:
        if skip_existing and paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[5/7] Grid calculations - SKIPPED (exists)")
            results['steps_completed'].append('raster_skipped')
        else:
            if verbose:
                print("\n[5/7] Raster/grid calculations...")
                print(f"    Using pattern lengths: {pattern_lengths}")

            # Load snippet offset
            if snippet_offset_file and Path(snippet_offset_file).exists():
                snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
            elif config.OVERVIEW_CSV.exists():
                snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                if verbose:
                    print(f"    Using snippet offset from default overview CSV: {snippet_offset}s")
            else:
                snippet_offset = 0.0
                if verbose:
                    print(f"    Using default snippet offset: {snippet_offset}s")

            # Onset file should exist from Step 4
            if onset_file is None:
                onset_file = paths['onsets_file']

            if not Path(onset_file).exists():
                raise FileNotFoundError(f"Onset file not found (should have been created in Step 4): {onset_file}")

            # Create comprehensive CSV
            df_comp = raster.create_comprehensive_csv(
                str(paths['corrected_downbeats_file']),
                str(onset_file),
                pattern_lengths,
                snippet_offset,
                str(paths['comprehensive_csv'])
            )

            results['comprehensive_csv'] = str(paths['comprehensive_csv'])
            results['pattern_lengths'] = pattern_lengths
            results['steps_completed'].append('raster')

            if verbose:
                print(f"  ✓ Comprehensive CSV created")

    except Exception as e:
        error_msg = f"Step 5 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue to RMS if we have the CSV from before

    # ========================================================================
    # STEP 5.5: RASTER PLOTS
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[5.5/7] Raster plots - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('raster_plots_skipped')
        else:
            # Check if raster plots already exist
            grid_output_dir = paths['comprehensive_csv'].parent
            raster_files_exist = (grid_output_dir / f'{track_id}_raster_comparison.png').exists()

            if skip_existing and raster_files_exist:
                if verbose:
                    print("\n[5.5/7] Raster plots - SKIPPED (exists)")
                results['steps_completed'].append('raster_plots_skipped')
            else:
                if verbose:
                    print("\n[5.5/7] Generating raster plots...")

                raster_plots.create_all_plots(
                    str(paths['comprehensive_csv']),
                    str(grid_output_dir),
                    track_id
                )

                results['steps_completed'].append('raster_plots')

                if verbose:
                    print(f"  ✓ Raster plots created")

    except Exception as e:
        error_msg = f"Step 5.5 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue to RMS analysis

    # ========================================================================
    # STEP 6: RMS ANALYSIS
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[6/7] RMS analysis - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('rms_skipped')
        elif skip_existing and paths['rms_summary'].exists():
            if verbose:
                print("\n[6/7] RMS analysis - SKIPPED (exists)")
            results['steps_completed'].append('rms_analysis_skipped')
        else:
            if verbose:
                print("\n[6/7] RMS histogram analysis...")

            rms_values = rms_grid_histograms.calculate_rms_from_csv(
                str(paths['comprehensive_csv'])
            )

            if rms_values:
                # Save RMS summary as JSON
                with open(paths['rms_summary'], 'w') as f:
                    # Convert numpy types to Python types for JSON
                    rms_json = {k: float(v) if not isinstance(v, dict) else v
                               for k, v in rms_values.items()}
                    json.dump(rms_json, f, indent=2)

                results['rms_values'] = rms_values
                results['steps_completed'].append('rms_analysis')

                if verbose:
                    print(f"  ✓ RMS calculated:")
                    print(f"    Uncorrected: {rms_values['uncorrected_ms']:.2f}ms")
                    print(f"    Per-snippet: {rms_values['per_snippet_ms']:.2f}ms")
                    print(f"    Drum method: {rms_values['drum_ms']:.2f}ms")
            else:
                if verbose:
                    print(f"  ⚠️  RMS calculation returned no values")

    except Exception as e:
        error_msg = f"Step 6 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 7: AUDIO EXAMPLES
    # ========================================================================
    if create_audio_examples:
        try:
            if not paths['comprehensive_csv'].exists():
                if verbose:
                    print("\n[7/7] Audio examples - SKIPPED (no comprehensive CSV)")
                results['steps_completed'].append('audio_examples_skipped')
            else:
                # Check if audio examples already exist
                audio_files_exist = (
                    (paths['audio_examples_dir'] / 'drum.mp3').exists() or
                    (paths['audio_examples_dir'] / 'uncorrected.mp3').exists()
                )

                if skip_existing and audio_files_exist:
                    if verbose:
                        print("\n[7/7] Audio examples - SKIPPED (exists)")
                    results['steps_completed'].append('audio_examples_skipped')
                else:
                    if verbose:
                        print("\n[7/7] Audio examples...")

                    # Load snippet offset again
                    if snippet_offset_file and Path(snippet_offset_file).exists():
                        snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
                    elif config.OVERVIEW_CSV.exists():
                        snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                    else:
                        snippet_offset = 0.0

                    audio_export.create_audio_examples(
                        str(audio_file),
                        str(paths['comprehensive_csv']),
                        str(paths['audio_examples_dir']),
                        snippet_offset=snippet_offset,
                        snippet_duration=config.CORRECT_BARS_SNIPPET_DURATION_S
                    )

                    results['steps_completed'].append('audio_examples')

                    if verbose:
                        print(f"  ✓ Audio examples created")

        except Exception as e:
            error_msg = f"Step 7 failed: {e}"
            results['errors'].append(error_msg)
            if verbose:
                print(f"  ✗ ERROR: {e}")
    else:
        if verbose:
            print("\n[7/7] Audio examples - SKIPPED (disabled)")
        results['steps_completed'].append('audio_examples_disabled')

    # ========================================================================
    # STEP 8: MIDI EXPORT
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[8/8] MIDI export - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('midi_export_skipped')
        else:
            # Check if MIDI files already exist
            midi_files_exist = (
                (paths['midi_dir'] / 'drum.mid').exists() or
                (paths['midi_dir'] / 'mel.mid').exists() or
                (paths['midi_dir'] / 'pitch.mid').exists()
            )

            if skip_existing and midi_files_exist:
                if verbose:
                    print("\n[8/8] MIDI export - SKIPPED (exists)")
                results['steps_completed'].append('midi_export_skipped')
            else:
                if verbose:
                    print("\n[8/8] MIDI export...")

                # Load snippet offset
                if snippet_offset_file and Path(snippet_offset_file).exists():
                    snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
                elif config.OVERVIEW_CSV.exists():
                    snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                else:
                    snippet_offset = 0.0

                # Export onset MIDI files (actual drum hits within loop boundaries)
                # One file per method (per_snippet, drum, mel, pitch) with L bars each
                # Tempo calculated from average of bar tempos in loop
                if verbose:
                    print("\n  [8a] Onset-based MIDI (drum hits)...")
                midi_files_onset = midi_export.comprehensive_csv_to_onset_midi(
                    str(paths['comprehensive_csv']),
                    str(paths['tempo_csv']),
                    str(paths['midi_dir'] / 'onset'),
                    snippet_start=snippet_offset
                )

                # Export bass pitch MIDI files (F0 converted to MIDI notes)
                # One file per method (per_snippet, drum, mel, pitch) with L bars each
                if verbose:
                    print("\n  [8b] Bass pitch MIDI...")
                f0_csv_path = paths['stems_dir'] / 'bass_f0.csv'
                if f0_csv_path.exists():
                    midi_files_pitch = midi_export.comprehensive_csv_to_pitch_midi(
                        str(paths['comprehensive_csv']),
                        str(paths['tempo_csv']),
                        str(f0_csv_path),
                        str(paths['midi_dir'] / 'bass_pitch'),
                        snippet_start=snippet_offset
                    )
                else:
                    midi_files_pitch = []
                    if verbose:
                        print(f"  ⚠️  Bass F0 CSV not found, skipping bass pitch MIDI")

                # Combine all MIDI files
                midi_files = midi_files_onset + midi_files_pitch

                if midi_files:
                    results['midi_files'] = {
                        'onset': [str(f) for f in midi_files_onset],
                        'bass_pitch': [str(f) for f in midi_files_pitch]
                    }
                    results['steps_completed'].append('midi_export')
                    if verbose:
                        print(f"\n  ✓ Exported {len(midi_files_onset)} onset MIDI + {len(midi_files_pitch)} bass pitch MIDI files")
                else:
                    results['steps_completed'].append('midi_export_no_data')
                    if verbose:
                        print(f"  ⚠️  No MIDI files created")

    except Exception as e:
        error_msg = f"Step 8 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 9: STEM LOOP EXPORT
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[9/9] Stem loop export - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('loops_skipped')
        else:
            # Check if loop files already exist (check for any method subdirectory)
            loops_exist = (
                (paths['loops_dir'] / 'drum').exists() or
                (paths['loops_dir'] / 'mel').exists() or
                (paths['loops_dir'] / 'pitch').exists()
            )

            if skip_existing and loops_exist:
                if verbose:
                    print("\n[9/9] Stem loop export - SKIPPED (exists)")
                results['steps_completed'].append('loops_skipped')
            else:
                if verbose:
                    print("\n[9/9] Stem loop export...")

                # Load snippet offset
                if snippet_offset_file and Path(snippet_offset_file).exists():
                    snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
                elif config.OVERVIEW_CSV.exists():
                    snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                else:
                    snippet_offset = 0.0

                # Export stem loops (one loop per method: drum, mel, pitch)
                # Each loop contains L bars from all 5 stems (vocals, drums, bass, piano, other)
                loop_files = audio_export.export_stem_loops(
                    str(paths['stems_dir']),
                    str(paths['comprehensive_csv']),
                    str(paths['tempo_csv']),
                    str(paths['loops_dir']),
                    snippet_start=snippet_offset,
                    pattern_lengths=pattern_lengths,
                    fade_duration_ms=5.0,
                    export_format='wav'
                )

                if loop_files:
                    results['loop_files'] = {
                        method: [str(f) for f in files]
                        for method, files in loop_files.items()
                    }
                    results['steps_completed'].append('loops')
                    if verbose:
                        total_files = sum(len(files) for files in loop_files.values())
                        print(f"  ✓ Exported {total_files} stem loops across {len(loop_files)} methods")
                else:
                    results['steps_completed'].append('loops_no_data')
                    if verbose:
                        print(f"  ⚠️  No loop files created")

    except Exception as e:
        error_msg = f"Step 9 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("Pipeline Summary")
        print("=" * 80)
        print(f"Track ID: {track_id}")
        print(f"Steps completed: {len(results['steps_completed'])}")
        for step in results['steps_completed']:
            print(f"  ✓ {step}")

        if results['errors']:
            print(f"\nErrors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  ✗ {error}")

        print("=" * 80)

    return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='AP2 Music Microtiming Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single track
  python main.py --audio track.wav --track-id 123 --output-dir output/

  # Process all files in a directory
  python main.py --audio-dir input_wav/ --output-dir output/ --analyse-all

  # Process with custom files
  python main.py --audio track.wav --track-id 123 \\
      --onset-file onsets/123_onsets.csv \\
      --pattern-file pattern_lengths.csv \\
      --snippet-file snippet_offsets.csv \\
      --output-dir output/

  # Skip audio examples
  python main.py --audio track.wav --track-id 123 \\
      --output-dir output/ --no-audio-examples

Environment:
  Main: AEinBOX_13_3
  Beat detection subprocess: new_beatnet_env
"""
    )

    parser.add_argument('--audio', help='Input audio file (for single track)')
    parser.add_argument('--audio-dir', help='Input directory (for batch processing with --analyse-all)')
    parser.add_argument('--track-id', help='Track identifier (not needed with --analyse-all)')
    parser.add_argument('--output-dir', required=True, help='Output directory')

    parser.add_argument('--analyse-all', action='store_true',
                       help='Process all audio files in --audio-dir')
    parser.add_argument('--onset-file', help='Path to onsets CSV file')
    parser.add_argument('--pattern-file', help='Path to pattern lengths CSV')
    parser.add_argument('--snippet-file', help='Path to snippet offsets CSV')

    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess even if output exists')
    parser.add_argument('--no-audio-examples', action='store_true',
                       help='Skip audio example generation')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize output')

    args = parser.parse_args()

    # Validate arguments
    if args.analyse_all:
        if not args.audio_dir:
            print("ERROR: --analyse-all requires --audio-dir")
            sys.exit(1)
        if not Path(args.audio_dir).is_dir():
            print(f"ERROR: --audio-dir must be a directory: {args.audio_dir}")
            sys.exit(1)
    else:
        if not args.audio:
            print("ERROR: --audio is required (or use --analyse-all with --audio-dir)")
            sys.exit(1)
        if not args.track_id:
            print("ERROR: --track-id is required (or use --analyse-all)")
            sys.exit(1)

    # Validate environment
    validation = config.validate_environment()
    if not validation['valid']:
        print("ERROR: Environment validation failed:")
        for error in validation['errors']:
            print(f"  - {error}")
        sys.exit(1)

    # ========================================================================
    # BATCH PROCESSING MODE (--analyse-all)
    # ========================================================================
    if args.analyse_all:
        audio_dir = Path(args.audio_dir)

        # Find all WAV files in the directory
        wav_files = sorted(audio_dir.glob('*.wav'))

        if not wav_files:
            print(f"No WAV files found in {audio_dir}")
            sys.exit(1)

        print("=" * 80)
        print(f"Batch Processing Mode: {len(wav_files)} WAV files found")
        print("=" * 80)

        # Track overall results
        batch_results = {
            'total_files': len(wav_files),
            'successful': [],
            'failed': [],
            'skipped': []
        }

        # Process each file
        for i, wav_file in enumerate(wav_files, start=1):
            # Derive track_id from filename (stem without extension)
            track_id = wav_file.stem

            print(f"\n{'=' * 80}")
            print(f"Processing [{i}/{len(wav_files)}]: {track_id}")
            print(f"File: {wav_file.name}")
            print(f"{'=' * 80}")

            try:
                results = run_complete_pipeline(
                    audio_file=str(wav_file),
                    track_id=track_id,
                    output_dir=args.output_dir,
                    pattern_file=args.pattern_file,
                    snippet_offset_file=args.snippet_file,
                    onset_file=args.onset_file,
                    skip_existing=not args.no_skip,
                    create_audio_examples=not args.no_audio_examples,
                    verbose=not args.quiet
                )

                # Save results JSON
                results_file = Path(args.output_dir) / track_id / 'pipeline_results.json'
                results_file.parent.mkdir(parents=True, exist_ok=True)

                def convert_to_serializable(obj):
                    """Recursively convert numpy types to Python native types."""
                    import numpy as np

                    if isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    elif isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32)):
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)):
                        return obj.tolist()
                    elif isinstance(obj, (str, int, float, bool, type(None))):
                        return obj
                    else:
                        return str(obj)

                with open(results_file, 'w') as f:
                    results_clean = convert_to_serializable(results)
                    json.dump(results_clean, f, indent=2)

                if results['errors']:
                    batch_results['failed'].append({
                        'track_id': track_id,
                        'file': wav_file.name,
                        'errors': results['errors']
                    })
                    print(f"\n⚠️  {track_id} completed with {len(results['errors'])} errors")
                else:
                    batch_results['successful'].append({
                        'track_id': track_id,
                        'file': wav_file.name
                    })
                    print(f"\n✓ {track_id} completed successfully")

            except Exception as e:
                batch_results['failed'].append({
                    'track_id': track_id,
                    'file': wav_file.name,
                    'errors': [str(e)]
                })
                print(f"\n✗ {track_id} failed: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next file

            # Force garbage collection after each track to prevent RAM buildup
            import gc
            gc.collect()

            # Play beep sound to indicate track completion
            print('\a')  # ASCII bell character - makes system beep

        # Print batch summary
        print("\n" + "=" * 80)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total files: {batch_results['total_files']}")
        print(f"Successful: {len(batch_results['successful'])}")
        print(f"Failed: {len(batch_results['failed'])}")

        if batch_results['successful']:
            print("\n✓ Successful tracks:")
            for item in batch_results['successful']:
                print(f"  - {item['track_id']} ({item['file']})")

        if batch_results['failed']:
            print("\n✗ Failed tracks:")
            for item in batch_results['failed']:
                print(f"  - {item['track_id']} ({item['file']})")
                for error in item['errors']:
                    print(f"    • {error}")

        # Save batch results
        batch_results_file = Path(args.output_dir) / 'batch_results.json'
        with open(batch_results_file, 'w') as f:
            json.dump(batch_results, f, indent=2)

        print(f"\nBatch results saved to: {batch_results_file}")
        print("=" * 80)

        # Exit with error code if any files failed
        if batch_results['failed']:
            sys.exit(1)
        else:
            sys.exit(0)

    # ========================================================================
    # SINGLE FILE MODE
    # ========================================================================
    # Run pipeline
    try:
        results = run_complete_pipeline(
            audio_file=args.audio,
            track_id=args.track_id,
            output_dir=args.output_dir,
            pattern_file=args.pattern_file,
            snippet_offset_file=args.snippet_file,
            onset_file=args.onset_file,
            skip_existing=not args.no_skip,
            create_audio_examples=not args.no_audio_examples,
            verbose=not args.quiet
        )

        # Save results JSON
        results_file = Path(args.output_dir) / args.track_id / 'pipeline_results.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)

        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            import numpy as np

            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

        with open(results_file, 'w') as f:
            results_clean = convert_to_serializable(results)
            json.dump(results_clean, f, indent=2)

        if results['errors']:
            print(f"\n⚠️  Pipeline completed with {len(results['errors'])} errors")
            sys.exit(1)
        else:
            print("\n✓ Pipeline completed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
