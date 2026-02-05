#!/usr/bin/env python3
"""
Loop Extractor Pipeline - Main Orchestrator

Complete pipeline for music microtiming analysis and loop extraction:
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
9. Stem loop export (WAV/MP3 loops for each stem, one loop per method: drum, mel, pitch)
11. Drum transcription (DrumTranscriber CNN - 6 drum classes)

Environment: loop_extractor_main
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

    4. DrumTranscriber model (optional, for Step 11)
       - Location: drumtranscriber/model/drum_transcriber.h5
       - Download from: https://drive.google.com/file/d/1w2fIHeyr-st3sbk1PYrtGOYW6YAD1fsi/view
       - Repository: https://github.com/yoshi-man/DrumTranscriber
       - Note: Pipeline will skip Step 11 if model is not available

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
from utils import audio_export, raster_plots, midi_export, microtiming_plots, drumtranscriber_interface


def run_complete_pipeline(
    audio_file: str,
    track_id: str,
    output_dir: str,
    pattern_file: Optional[str] = None,
    snippet_offset_file: Optional[str] = None,
    onset_file: Optional[str] = None,
    onset_mode: str = 'librosa',
    onset_threshold_drumtranscriber: float = 0.5,  # Filter onsets closer than this fraction of 1/16th note (0.5 = 1/32nd)
    loop_start_offset_ms: float = 0.0,  # Loop start offset in ms (0.0 = use grid time exactly, negative was adding silence)
    skip_existing: bool = False,
    create_audio_examples: bool = True,
    daw_ready: bool = False,
    manual_start: Optional[float] = None,
    manual_duration: Optional[float] = None,
    export_format: str = 'wav',
    verbose: bool = True
) -> dict:
    """
    Run complete Loop Extractor pipeline for a single track.

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
        print(f"Loop Extractor Pipeline - Track {track_id}")
        print("=" * 80)

    # Validate inputs
    audio_file = Path(audio_file)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Get output paths
    paths = config.get_output_paths(track_id, Path(output_dir), daw_ready=daw_ready)

    # Create directories
    config.create_output_directories(track_id, Path(output_dir), daw_ready=daw_ready)

    results = {
        'track_id': track_id,
        'audio_file': str(audio_file),
        'time_range': {
            'manual_start': manual_start,
            'manual_duration': manual_duration,
            'auto_detect': manual_start is None
        },
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
        if skip_existing and paths['tempo_plots_pdf'].exists() and paths['tempo_csv'].exists():
            if verbose:
                print("\n[3.5/7] Tempo plots - SKIPPED (exists)")
            results['steps_completed'].append('tempo_plots_skipped')
        else:
            if daw_ready:
                if verbose:
                    print("\n[3.5/7] Generating tempo CSV (plots skipped in DAW mode)...")
            else:
                if verbose:
                    print("\n[3.5/7] Generating tempo plots...")

            # Load snippet offset
            if manual_start is not None:
                snippet_offset = manual_start
            elif snippet_offset_file and Path(snippet_offset_file).exists():
                snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
                # If track not found in CSV (returns 0.0), use default fallback
                if snippet_offset == 0.0:
                    snippet_offset = 30.0
                    if verbose:
                        print(f"  Track '{track_id}' not found in snippet CSV, using default: {snippet_offset}s")
            elif config.OVERVIEW_CSV.exists():
                snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                # If track not found in CSV (returns 0.0), use default fallback
                if snippet_offset == 0.0:
                    snippet_offset = 30.0
                    if verbose:
                        print(f"  Track '{track_id}' not found in snippet CSV, using default: {snippet_offset}s")
            else:
                # No CSV available, use default
                snippet_offset = 30.0
                if verbose:
                    print(f"  No snippet CSV available, using default: {snippet_offset}s")

            # Determine snippet duration
            snippet_duration = manual_duration if manual_duration is not None else config.CORRECT_BARS_SNIPPET_DURATION_S

            # Store actual time range used
            results['time_range']['actual_start'] = snippet_offset
            results['time_range']['actual_duration'] = snippet_duration
            if snippet_offset is not None:
                results['time_range']['actual_end'] = snippet_offset + snippet_duration

            # Create tempo plots (or just CSV in DAW mode)
            tempo_files = tempo_plots.create_tempo_plots(
                str(paths['beats_file']),
                str(paths['corrected_downbeats_file']),
                str(paths['tempo_plots_dir']),
                track_id,
                snippet_start=snippet_offset,
                snippet_duration=snippet_duration,
                skip_plots=daw_ready  # Skip plots in DAW mode
            )

            results['tempo_plots_pdf'] = tempo_files.get('plot_pdf')
            results['tempo_csv'] = tempo_files['csv']

            if daw_ready:
                results['steps_completed'].append('tempo_csv_only')
            else:
                results['steps_completed'].append('tempo_plots')

            if verbose:
                if daw_ready:
                    print(f"  ✓ Tempo CSV created")
                    print(f"    CSV: {Path(tempo_files['csv']).name}")
                else:
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
    # STEP 4.1: DRUM TRANSCRIPTION (if using drumtranscriber onset mode)
    # ========================================================================
    # Run DrumTranscriber before pattern detection so we can use CNN-detected onsets
    try:
        # Check if DrumTranscriber is available and onset_mode is drumtranscriber
        if onset_mode == 'drumtranscriber':
            if not drumtranscriber_interface.DRUMTRANSCRIBER_AVAILABLE:
                if verbose:
                    print("\n[4.1/7] Drum transcription - SKIPPED (DrumTranscriber not available)")
                    print("         Falling back to librosa onsets from Step 4")
                results['steps_completed'].append('drumtranscriber_unavailable')
            elif daw_ready:
                # Skip in DAW mode (not essential for loop creation)
                if verbose:
                    print("\n[4.1/7] Drum transcription - SKIPPED (DAW mode)")
                    print("         Using librosa onsets from Step 4")
                results['steps_completed'].append('drumtranscriber_skipped_daw')
            else:
                # Check if already exists
                transcription_csv = paths['drumtranscriber_dir'] / f'{track_id}_drum_transcription.csv'
                drumtranscriber_onsets_csv = paths['drumtranscriber_dir'] / f'{track_id}_onsets.csv'

                if skip_existing and transcription_csv.exists() and drumtranscriber_onsets_csv.exists():
                    if verbose:
                        print("\n[4.1/7] Drum transcription - SKIPPED (exists)")
                    results['steps_completed'].append('drumtranscriber_skipped')
                    # Override onset_file with existing drumtranscriber onsets
                    onset_file = str(drumtranscriber_onsets_csv)
                    if verbose:
                        print(f"         Using DrumTranscriber onsets: {onset_file}")
                else:
                    if verbose:
                        print("\n[4.1/7] Drum transcription...")

                    # Transcribe the drum stem (not the full mix)
                    drum_stem_path = paths['stems_dir'] / 'drums.wav'
                    transcription_results = drumtranscriber_interface.transcribe_drums(
                        str(drum_stem_path),
                        str(paths['drumtranscriber_dir']),
                        track_id,
                        sr=44100
                    )

                    results['drumtranscriber'] = {
                        'predictions_csv': transcription_results['predictions_csv'],
                        'timeline_csv': transcription_results.get('timeline_csv'),
                        'onsets_csv': transcription_results.get('onsets_csv'),
                        'summary_json': transcription_results['summary_json'],
                        'total_hits': transcription_results['summary']['total_hits']
                    }
                    results['steps_completed'].append('drumtranscriber')

                    # Override onset_file to use DrumTranscriber onsets for all downstream analysis
                    if 'onsets_csv' in transcription_results:
                        onset_file = transcription_results['onsets_csv']
                        if verbose:
                            print(f"  ✓ Transcribed {transcription_results['summary']['total_hits']} drum hits")
                            print(f"  ✓ Onset mode: Using DrumTranscriber onsets for all analysis")
        else:
            # Using librosa onset mode (default)
            if verbose:
                print(f"\n[4.1/7] Onset mode: librosa (using onsets from Step 4)")

    except Exception as e:
        error_msg = f"Step 4.1 (DrumTranscriber) failed: {e} - falling back to librosa onsets"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ⚠ WARNING: {e}")
            print(f"  Falling back to librosa onsets from Step 4")

    # ========================================================================
    # STEP 4.2: FILTER CLOSE ONSETS (if using drumtranscriber mode)
    # ========================================================================
    try:
        if onset_mode == 'drumtranscriber' and onset_file:
            # Check if tempo CSV exists (created in Step 3.5)
            if paths['tempo_csv'].exists():
                if verbose:
                    print(f"\n[4.2/7] Filtering close onsets...")

                # Create filtered onset file path
                onset_file_path = Path(onset_file)
                filtered_onset_file = onset_file_path.parent / f'{track_id}_onsets_filtered.csv'

                # Check if filtered file already exists
                if skip_existing and filtered_onset_file.exists():
                    if verbose:
                        print(f"  ✓ Filtered onsets - SKIPPED (exists)")
                    onset_file = str(filtered_onset_file)
                    results['steps_completed'].append('onset_filtering_skipped')
                else:
                    # Filter onsets that are too close together
                    filter_stats = drumtranscriber_interface.filter_close_onsets(
                        onset_file,
                        str(paths['tempo_csv']),
                        str(filtered_onset_file),
                        min_interval_16th_fraction=onset_threshold_drumtranscriber
                    )

                    # Use filtered onsets for all downstream analysis
                    onset_file = str(filtered_onset_file)
                    results['onset_filter_stats'] = filter_stats
                    results['steps_completed'].append('onset_filtering')

                    if verbose:
                        print(f"  ✓ Filtered onsets saved: {filtered_onset_file.name}")
            else:
                if verbose:
                    print(f"\n[4.2/7] Onset filtering - SKIPPED (no tempo CSV yet)")
        else:
            if verbose and onset_mode == 'drumtranscriber':
                print(f"\n[4.2/7] Onset filtering - SKIPPED (no onsets to filter)")

    except Exception as e:
        error_msg = f"Step 4.2 (Onset filtering) failed: {e} - using unfiltered onsets"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ⚠ WARNING: {e}")
            print(f"  Using unfiltered onsets")

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
            if manual_start is not None:
                snippet_offset = manual_start
                snippet_dur = manual_duration if manual_duration is not None else 30.0
                snippet = (snippet_offset, snippet_offset + snippet_dur)
            elif snippet_offset_file and Path(snippet_offset_file).exists():
                snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
                snippet = (snippet_offset, snippet_offset + 30.0)  # 30s snippet
            elif config.OVERVIEW_CSV.exists():
                snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                if snippet_offset > 0:
                    snippet = (snippet_offset, snippet_offset + 30.0)

            # Filter bars to snippet if provided
            # Only include FULL bars: bar_start >= snippet_start AND bar_end <= snippet_end
            if snippet:
                s0, s1 = snippet
                mask = (bar_starts >= s0) & (bar_ends <= s1)
                bar_starts_snippet = bar_starts[mask]
                bar_ends_snippet = bar_ends[mask]
                if verbose:
                    print(f"    Filtered to {len(bar_starts_snippet)} full bars within snippet [{s0:.2f}s - {s1:.2f}s]")
            else:
                bar_starts_snippet = bar_starts
                bar_ends_snippet = bar_ends

            # Onset file should exist from Step 4
            if onset_file is None:
                onset_file = paths['onsets_file']

            # Run pattern detection with pre-filtered bars
            # use_all_bars=True since we already filtered the bars in main.py
            pattern_detection_result = pattern_detection.detect_pattern_lengths(
                onset_csv_path=str(onset_file),
                drums_wav_path=str(drum_stem),
                bass_wav_path=str(bass_stem),
                bar_starts=bar_starts_snippet,  # Pre-filtered bars
                bar_ends=bar_ends_snippet,      # Pre-filtered bars
                tsig=tsig,
                snippet=snippet,
                use_all_bars=True  # Skip internal filtering since main.py already filtered
            )

            # Extract pattern lengths and snippet info
            pattern_lengths = {k: v for k, v in pattern_detection_result.items() if k != 'snippet_info'}
            results['pattern_lengths'] = pattern_lengths

            # Store snippet info separately
            if 'snippet_info' in pattern_detection_result:
                results['snippet_info'] = pattern_detection_result['snippet_info']

            results['steps_completed'].append('pattern_detection')

            if verbose:
                print(f"  ✓ Pattern lengths detected: {pattern_lengths}")

    except Exception as e:
        # Fall back to defaults if pattern detection fails
        pattern_lengths = {'drum': 4, 'mel': 4, 'pitch': 4, 'lepa': 4, 'aicc': 4}
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
            if manual_start is not None:
                snippet_offset = manual_start
                if verbose:
                    print(f"    Using manual snippet offset: {snippet_offset}s")
            elif snippet_offset_file and Path(snippet_offset_file).exists():
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
        if daw_ready:
            if verbose:
                print("\n[5.5/7] Raster plots - SKIPPED (DAW ready mode)")
            results['steps_completed'].append('raster_plots_skipped_daw')
        elif not paths['comprehensive_csv'].exists():
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
        # Don't raise - continue to microtiming plots

    # ========================================================================
    # STEP 5.6: MICROTIMING PLOTS
    # ========================================================================
    try:
        if daw_ready:
            if verbose:
                print("\n[5.6/7] Microtiming plots - SKIPPED (DAW ready mode)")
            results['steps_completed'].append('microtiming_plots_skipped_daw')
        elif not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[5.6/7] Microtiming plots - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('microtiming_plots_skipped')
        else:
            # Check if microtiming plots already exist
            grid_output_dir = paths['comprehensive_csv'].parent
            microtiming_files_exist = (grid_output_dir / f'{track_id}_microtiming_plots.pdf').exists()

            if skip_existing and microtiming_files_exist:
                if verbose:
                    print("\n[5.6/7] Microtiming plots - SKIPPED (exists)")
                results['steps_completed'].append('microtiming_plots_skipped')
            else:
                if verbose:
                    print("\n[5.6/7] Generating microtiming deviation plots...")

                # Get snippet info from pattern detection results
                snippet_info = results.get('snippet_info')

                microtiming_pdf = microtiming_plots.create_microtiming_plots(
                    str(paths['comprehensive_csv']),
                    track_id,
                    str(grid_output_dir),
                    snippet_info=snippet_info
                )

                results['microtiming_plots_pdf'] = microtiming_pdf
                results['steps_completed'].append('microtiming_plots')

                if verbose:
                    print(f"  ✓ Microtiming plots created")

    except Exception as e:
        error_msg = f"Step 5.6 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue to rhythm histograms

    # ========================================================================
    # STEP 5.7: RHYTHM HISTOGRAMS
    # ========================================================================
    try:
        from utils import rhythm_histograms

        if daw_ready:
            if verbose:
                print("\n[5.7/7] Rhythm histograms - SKIPPED (DAW ready mode)")
            results['steps_completed'].append('rhythm_histograms_skipped_daw')
        elif not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[5.7/7] Rhythm histograms - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('rhythm_histograms_skipped')
        else:
            # Define rhythm output directory based on the track directory
            track_dir = Path(output_dir) / track_id
            rhythm_output_dir = track_dir / '5.5_rhythm'

            # Check if rhythm histograms already exist
            rhythm_files_exist = (rhythm_output_dir / f'{track_id}_rhythm_histograms.pdf').exists()

            if skip_existing and rhythm_files_exist:
                if verbose:
                    print("\n[5.7/7] Rhythm histograms - SKIPPED (exists)")
                results['steps_completed'].append('rhythm_histograms_skipped')
            else:
                if verbose:
                    print("\n[5.7/7] Generating rhythm histograms...")

                rhythm_files = rhythm_histograms.create_rhythm_histograms(
                    str(paths['comprehensive_csv']),
                    track_id,
                    str(rhythm_output_dir)
                )

                if rhythm_files:
                    results['rhythm_histograms_pdf'] = rhythm_files.get('pdf')
                    results['rhythm_histograms_csv'] = rhythm_files.get('csv')

                results['steps_completed'].append('rhythm_histograms')

                if verbose:
                    print(f"  ✓ Rhythm histograms created")

                # Also create rhythm histograms with style (using filtered flexStart CSVs)
                try:
                    grid_output_dir = paths['comprehensive_csv'].parent
                    base_name = Path(paths['comprehensive_csv']).stem.replace('_comprehensive_phases', '')

                    rhythm_files_style = rhythm_histograms.create_rhythm_histograms_with_style(
                        str(grid_output_dir),
                        base_name,
                        track_id,
                        str(rhythm_output_dir)
                    )

                    if rhythm_files_style:
                        results['rhythm_histograms_with_style_pdf'] = rhythm_files_style.get('pdf')
                        results['rhythm_histograms_with_style_csv'] = rhythm_files_style.get('csv')

                    if verbose:
                        print(f"  ✓ Rhythm histograms with style created")

                except Exception as e:
                    if verbose:
                        print(f"  ! Warning: Could not create rhythm histograms with style: {e}")

                # Also create rhythm histograms with medians and IQR
                try:
                    grid_output_dir = paths['comprehensive_csv'].parent
                    base_name = Path(paths['comprehensive_csv']).stem.replace('_comprehensive_phases', '')

                    rhythm_files_medians = rhythm_histograms.create_rhythm_histograms_with_medians_and_iqr(
                        str(grid_output_dir),
                        base_name,
                        track_id,
                        str(rhythm_output_dir)
                    )

                    if rhythm_files_medians:
                        results['rhythm_histograms_with_medians_and_iqr_pdf'] = rhythm_files_medians.get('pdf')
                        results['rhythm_histograms_with_medians_and_iqr_csv'] = rhythm_files_medians.get('csv')

                    if verbose:
                        print(f"  ✓ Rhythm histograms with medians and IQR created")

                except Exception as e:
                    if verbose:
                        print(f"  ! Warning: Could not create rhythm histograms with medians and IQR: {e}")

                # Also create groove pulse histograms (filtered onsets)
                try:
                    from batch_analysis import groove_pulse_and_statistics

                    grid_output_dir = paths['comprehensive_csv'].parent
                    base_name = Path(paths['comprehensive_csv']).stem.replace('_comprehensive_phases', '')

                    groove_files = groove_pulse_and_statistics.create_groove_pulse_histograms(
                        str(grid_output_dir),
                        base_name,
                        track_id,
                        str(rhythm_output_dir)
                    )

                    if groove_files:
                        results['groove_pulse_histograms_pdf'] = groove_files.get('pdf')
                        results['groove_pulse_histograms_csv'] = groove_files.get('csv')

                    if verbose:
                        print(f"  ✓ Groove pulse histograms created")

                except Exception as e:
                    if verbose:
                        print(f"  ! Warning: Could not create groove pulse histograms: {e}")

                # Also create rhythm pattern histograms (binary patterns from groove pulse data)
                try:
                    from utils import rhythm_patterns

                    grid_output_dir = paths['comprehensive_csv'].parent
                    base_name = Path(paths['comprehensive_csv']).stem.replace('_comprehensive_phases', '')

                    pattern_files = rhythm_patterns.create_rhythm_pattern_histograms(
                        str(grid_output_dir),
                        base_name,
                        track_id,
                        str(rhythm_output_dir)
                    )

                    if pattern_files:
                        results['rhythm_pattern_histogram'] = pattern_files.get('rhythm_pattern_histogram')

                    if verbose:
                        print(f"  ✓ Rhythm pattern histograms created")

                except Exception as e:
                    if verbose:
                        print(f"  ! Warning: Could not create rhythm pattern histograms: {e}")

                # Create beat histograms (inter-onset interval analysis)
                try:
                    from utils import beat_histograms

                    # Create 5.7_beat_histograms folder
                    track_root = paths['comprehensive_csv'].parent.parent
                    beat_output_dir = track_root / '5.7_beat_histograms'
                    beat_output_dir.mkdir(parents=True, exist_ok=True)

                    grid_output_dir = paths['comprehensive_csv'].parent
                    base_name = Path(paths['comprehensive_csv']).stem.replace('_comprehensive_phases', '')

                    # Get BPM and snippet start time from results
                    bpm = results.get('tempo_estimation', {}).get('bpm', 120.0)
                    snippet_start_time = results.get('snippet_info', {}).get('start_time', 0.0)

                    beat_files = beat_histograms.create_beat_histograms(
                        str(grid_output_dir),
                        base_name,
                        track_id,
                        str(beat_output_dir),
                        bpm,
                        snippet_start_time
                    )

                    if beat_files:
                        results['beat_histogram'] = beat_files.get('pre_beat_histogram_csv')

                    # Create all onsets visualization
                    beat_all_onsets_files = beat_histograms.create_beat_histograms_all_onsets(
                        str(grid_output_dir),
                        base_name,
                        track_id,
                        str(beat_output_dir),
                        bpm,
                        snippet_start_time
                    )

                    # Create simple IOI histogram
                    simple_ioi_files = beat_histograms.create_simple_ioi_histogram(
                        str(grid_output_dir),
                        base_name,
                        track_id,
                        str(beat_output_dir),
                        bpm,
                        snippet_start_time
                    )

                    # Create simple beat histograms (from pre_beat_histogram CSVs)
                    simple_beat_files = beat_histograms.create_simple_beat_histograms(
                        str(beat_output_dir),
                        track_id,
                        bpm
                    )

                    if verbose:
                        print(f"  ✓ Beat histograms created")

                except Exception as e:
                    if verbose:
                        print(f"  ! Warning: Could not create beat histograms: {e}")

                # Calculate aggregate rhythm statistics
                try:
                    from batch_analysis import aggregate_statistics_rhythm_hist

                    track_root = paths['comprehensive_csv'].parent.parent

                    aggregate_statistics_rhythm_hist.aggregate_statistics_for_track(
                        track_root,
                        track_id
                    )

                    if verbose:
                        print(f"  ✓ Aggregate rhythm statistics calculated")

                except Exception as e:
                    if verbose:
                        print(f"  ! Warning: Could not calculate aggregate rhythm statistics: {e}")

    except Exception as e:
        error_msg = f"Step 5.7 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue to RMS analysis

    # ========================================================================
    # STEP 6: RMS ANALYSIS
    # ========================================================================
    try:
        if daw_ready:
            if verbose:
                print("\n[6/7] RMS analysis - SKIPPED (DAW ready mode)")
            results['steps_completed'].append('rms_skipped_daw')
        elif not paths['comprehensive_csv'].exists():
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
                    if manual_start is not None:
                        snippet_offset = manual_start
                    elif snippet_offset_file and Path(snippet_offset_file).exists():
                        snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
                    elif config.OVERVIEW_CSV.exists():
                        snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                    else:
                        snippet_offset = 0.0

                    snippet_dur = manual_duration if manual_duration is not None else config.CORRECT_BARS_SNIPPET_DURATION_S

                    # Check for groove pulse CSV
                    groove_pulse_csv = track_dir / '5.5_rhythm' / f'{track_id}_groove_pulse_histograms_filtered.csv'
                    groove_pulse_csv_str = str(groove_pulse_csv) if groove_pulse_csv.exists() else None

                    audio_export.create_audio_examples(
                        str(audio_file),
                        str(paths['comprehensive_csv']),
                        str(paths['audio_examples_dir']),
                        snippet_offset=snippet_offset,
                        snippet_duration=snippet_dur,
                        groove_pulse_csv=groove_pulse_csv_str,
                        export_format=export_format
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
    # STEP 8: LEPA DATA EXPORT
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[8/8] LEPA export - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('lepa_export_skipped')
        else:
            # Define LEPA output directory
            lepa_output_dir = Path(output_dir) / track_id / '10_output_for_lepa'

            # Check if LEPA export already exists (check for L1 file)
            lepa_file_exists = (lepa_output_dir / f'{track_id}_bar_durations_L1.csv').exists()

            if skip_existing and lepa_file_exists:
                if verbose:
                    print("\n[8/8] LEPA export - SKIPPED (exists)")
                results['steps_completed'].append('lepa_export_skipped')
            else:
                if verbose:
                    print("\n[8/8] Exporting LEPA bar duration data...")

                from utils import lepa_export

                # Get audio file paths
                drum_stem_path = paths['stems_dir'] / 'drums.wav'

                lepa_csv = lepa_export.export_bar_durations(
                    str(paths['comprehensive_csv']),
                    track_id,
                    str(lepa_output_dir),
                    audio_file=str(audio_file) if audio_file else None,
                    drum_stem_file=str(drum_stem_path) if drum_stem_path.exists() else None
                )

                if lepa_csv:
                    results['lepa_export_csv'] = lepa_csv
                    results['steps_completed'].append('lepa_export')
                    if verbose:
                        print(f"  ✓ LEPA data exported")

    except Exception as e:
        error_msg = f"Step 8 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 9: MIDI EXPORT
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[9/9] MIDI export - SKIPPED (no comprehensive CSV)")
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
                # In DAW mode: only drum method, no subfolders
                # In detailed mode: all methods (per_snippet, drum, mel, pitch) with subfolders
                if daw_ready:
                    # DAW mode: drum method onset + bass pitch, directly in midi_dir
                    if verbose:
                        print("\n  [8] MIDI export (drum method + bass pitch)...")
                    midi_files_onset = midi_export.comprehensive_csv_to_onset_midi(
                        str(paths['comprehensive_csv']),
                        str(paths['tempo_csv']),
                        str(paths['midi_dir']),
                        snippet_start=snippet_offset,
                        methods=['drum']  # Only drum method
                    )

                    # Export bass pitch MIDI in DAW mode
                    f0_csv_path = paths['stems_dir'] / 'bass_f0.csv'
                    if f0_csv_path.exists():
                        midi_files_pitch = midi_export.comprehensive_csv_to_pitch_midi(
                            str(paths['comprehensive_csv']),
                            str(paths['tempo_csv']),
                            str(f0_csv_path),
                            str(paths['midi_dir']),
                            snippet_start=snippet_offset,
                            methods=['drum']  # Only drum method
                        )
                    else:
                        midi_files_pitch = []
                        if verbose:
                            print(f"  ⚠️  Bass F0 CSV not found, skipping bass pitch MIDI")

                    # No FlexStart MIDI in DAW mode
                    midi_files_flexstart = []
                else:
                    # Detailed mode: all methods with subfolders
                    # Prepare FlexStart parameters
                    grid_output_dir = Path(paths['comprehensive_csv']).parent
                    base_name = Path(paths['comprehensive_csv']).stem.replace('_comprehensive_phases', '')

                    if verbose:
                        print("\n  [8a] Onset-based MIDI (drum hits + FlexStart grid)...")
                    midi_files_onset = midi_export.comprehensive_csv_to_onset_midi(
                        str(paths['comprehensive_csv']),
                        str(paths['tempo_csv']),
                        str(paths['midi_dir'] / 'onset'),
                        snippet_start=snippet_offset,
                        methods=['per_snippet', 'drum', 'mel', 'pitch', 'standard_L1', 'standard_L2', 'standard_L4',
                                 '1bar_flexStart', '2bar_flexStart', '4bar_flexStart'],
                        grid_output_dir=str(grid_output_dir),
                        base_name=base_name
                    )

                    # Export bass pitch MIDI files (F0 converted to MIDI notes)
                    if verbose:
                        print("\n  [8b] Bass pitch MIDI (all methods + FlexStart)...")
                    f0_csv_path = paths['stems_dir'] / 'bass_f0.csv'
                    if f0_csv_path.exists():
                        midi_files_pitch = midi_export.comprehensive_csv_to_pitch_midi(
                            str(paths['comprehensive_csv']),
                            str(paths['tempo_csv']),
                            str(f0_csv_path),
                            str(paths['midi_dir'] / 'bass_pitch'),
                            snippet_start=snippet_offset,
                            methods=['per_snippet', 'drum', 'mel', 'pitch', 'standard_L1', 'standard_L2', 'standard_L4',
                                     '1bar_flexStart', '2bar_flexStart', '4bar_flexStart'],
                            grid_output_dir=str(grid_output_dir),
                            base_name=base_name
                        )
                    else:
                        midi_files_pitch = []
                        if verbose:
                            print(f"  ⚠️  Bass F0 CSV not found, skipping bass pitch MIDI")

                    # No separate FlexStart export needed - integrated into onset and pitch exports
                    midi_files_flexstart = []

                # Combine all MIDI files
                midi_files = midi_files_onset + midi_files_pitch
                if not daw_ready:
                    midi_files += midi_files_flexstart

                if midi_files:
                    results['midi_files'] = {
                        'onset': [str(f) for f in midi_files_onset],
                        'bass_pitch': [str(f) for f in midi_files_pitch],
                        'flexstart': [str(f) for f in midi_files_flexstart] if not daw_ready else []
                    }
                    results['steps_completed'].append('midi_export')
                    if verbose:
                        if daw_ready:
                            print(f"\n  ✓ Exported {len(midi_files_onset)} onset MIDI + {len(midi_files_pitch)} bass pitch MIDI files")
                        else:
                            print(f"\n  ✓ Exported {len(midi_files_onset)} onset MIDI + {len(midi_files_pitch)} bass pitch MIDI + {len(midi_files_flexstart)} FlexStart MIDI files")
                else:
                    results['steps_completed'].append('midi_export_no_data')
                    if verbose:
                        print(f"  ⚠️  No MIDI files created")

    except Exception as e:
        error_msg = f"Step 9 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 10: STEM LOOP EXPORT
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[10/10] Stem loop export - SKIPPED (no comprehensive CSV)")
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
                    print("\n[10/10] Stem loop export - SKIPPED (exists)")
                results['steps_completed'].append('loops_skipped')
            else:
                if verbose:
                    print("\n[10/10] Stem loop export...")

                # Load snippet offset
                if snippet_offset_file and Path(snippet_offset_file).exists():
                    snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
                elif config.OVERVIEW_CSV.exists():
                    snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                else:
                    snippet_offset = 0.0

                # Export stem loops using filtered FlexStart CSVs
                # Extract base_name from comprehensive CSV path
                base_name = Path(paths['comprehensive_csv']).stem  # e.g., 'track_id_comprehensive_phases'
                grid_output_dir = Path(paths['comprehensive_csv']).parent

                # In DAW mode: only export one FlexStart method based on detected pattern length
                # In detailed mode: export all FlexStart methods (L=4, L=2, L=1)
                if daw_ready:
                    # Determine which method to use based on pattern_lengths
                    # Priority: mel (4-bar) > lepa (2-bar) > aicc (1-bar)
                    if 'mel' in pattern_lengths and pattern_lengths['mel'] == 4:
                        methods = ['4bar_flexStart']
                    elif 'lepa' in pattern_lengths and pattern_lengths['lepa'] == 2:
                        methods = ['2bar_flexStart']
                    elif 'aicc' in pattern_lengths and pattern_lengths['aicc'] == 1:
                        methods = ['1bar_flexStart']
                    else:
                        # Fallback: try 4-bar first
                        methods = ['4bar_flexStart']

                    loop_files = audio_export.export_stem_loops(
                        str(paths['stems_dir']),
                        str(grid_output_dir),
                        base_name,
                        str(paths['loops_dir']),
                        snippet_start=snippet_offset,
                        pattern_lengths=pattern_lengths,
                        fade_duration_ms=5.0,
                        export_format=export_format,
                        methods=methods,
                        loop_start_offset_ms=loop_start_offset_ms
                    )
                else:
                    # Detailed mode: export all three FlexStart methods
                    loop_files = audio_export.export_stem_loops(
                        str(paths['stems_dir']),
                        str(grid_output_dir),
                        base_name,
                        str(paths['loops_dir']),
                        snippet_start=snippet_offset,
                        pattern_lengths=pattern_lengths,
                        fade_duration_ms=5.0,
                        export_format=export_format,
                        loop_start_offset_ms=loop_start_offset_ms
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
        error_msg = f"Step 10 failed: {e}"
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
        description='Loop Extractor - Music Microtiming Analysis and Loop Extraction Pipeline',
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
    parser.add_argument('--onset-mode', choices=['librosa', 'drumtranscriber'], default='librosa',
                       help='Onset detection method: librosa (Step 4) or drumtranscriber (Step 11, requires DrumTranscriber)')
    parser.add_argument('--onset-threshold-drumtranscriber', type=float, default=0.5,
                       help='Minimum onset interval as fraction of 1/16th note (default: 0.5 = 1/32nd note). Only used with --onset-mode drumtranscriber')
    parser.add_argument('--loop-start-offset-ms', type=float, default=0.0,
                       help='Loop start offset in milliseconds (default: 0.0ms, use grid time exactly. Negative values start earlier but may add silence)')
    parser.add_argument('--pattern-file', help='Path to pattern lengths CSV')
    parser.add_argument('--snippet-file', help='Path to snippet offsets CSV')

    parser.add_argument('--no-audio-examples', action='store_true',
                       help='Skip audio example generation')
    parser.add_argument('--daw-ready', action='store_true',
                       help='DAW Ready mode: only export stems, loops (drum method), and MIDI (drum method)')
    parser.add_argument('--manual-start', type=float,
                       help='Manual start time in seconds (overrides snippet detection)')
    parser.add_argument('--manual-duration', type=float,
                       help='Manual duration in seconds (overrides default 30s snippet duration)')
    parser.add_argument('--export-format', choices=['wav', 'mp3'], default='wav',
                       help='Export format for stem loops (default: wav)')
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

        # Find all audio files (WAV and MP3) in the directory
        # Filter out macOS resource fork files (._filename)
        wav_files = [f for f in sorted(audio_dir.glob('*.wav')) if not f.name.startswith('._')]
        mp3_files = [f for f in sorted(audio_dir.glob('*.mp3')) if not f.name.startswith('._')]
        audio_files = sorted(wav_files + mp3_files)

        if not audio_files:
            print(f"No WAV or MP3 files found in {audio_dir}")
            sys.exit(1)

        print("=" * 80)
        print(f"Batch Processing Mode: {len(audio_files)} audio files found")
        print(f"  WAV files: {len(wav_files)}")
        print(f"  MP3 files: {len(mp3_files)}")
        print("=" * 80)

        # Track overall results
        batch_results = {
            'total_files': len(audio_files),
            'successful': [],
            'failed': [],
            'skipped': []
        }

        # Process each file
        for i, wav_file in enumerate(audio_files, start=1):
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
                    onset_mode=args.onset_mode,
                    onset_threshold_drumtranscriber=args.onset_threshold_drumtranscriber,
                    loop_start_offset_ms=args.loop_start_offset_ms,
                    skip_existing=False,
                    create_audio_examples=not args.no_audio_examples,
                    daw_ready=args.daw_ready,
                    manual_start=args.manual_start,
                    manual_duration=args.manual_duration,
                    export_format=args.export_format,
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

        # Merge all plot PDFs
        try:
            from batch_analysis.merge_plots import merge_plots
            print("\n" + "=" * 80)
            print("MERGING PLOTS")
            print("=" * 80)
            merge_plots(Path(args.output_dir))
        except ImportError as ie:
            print(f"\n⚠️  PDF merging skipped: {ie}")
            print("Install required packages: pip install PyPDF2 Pillow reportlab")
        except Exception as e:
            print(f"\n⚠️  PDF merging failed: {e}")

        # Create pattern length summary pie charts
        try:
            from batch_analysis.pattern_length_summary import create_pattern_length_summary
            create_pattern_length_summary(Path(args.output_dir))
        except ImportError as ie:
            print(f"\n⚠️  Pattern length summary skipped: {ie}")
            print("Install required packages: pip install matplotlib")
        except Exception as e:
            print(f"\n⚠️  Pattern length summary failed: {e}")

        # Create loop statistics histograms
        try:
            from batch_analysis.loop_statistics import create_loop_statistics
            create_loop_statistics(Path(args.output_dir))
        except ImportError as ie:
            print(f"\n⚠️  Loop statistics skipped: {ie}")
            print("Install required packages: pip install matplotlib numpy")
        except Exception as e:
            print(f"\n⚠️  Loop statistics failed: {e}")

        # Export LEPA data
        try:
            from batch_analysis.lepa_data_export import export_lepa_data
            export_lepa_data(Path(args.output_dir))
        except ImportError as ie:
            print(f"\n⚠️  LEPA data export skipped: {ie}")
            print("Install required packages: pip install pandas")
        except Exception as e:
            print(f"\n⚠️  LEPA data export failed: {e}")

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
            onset_mode=args.onset_mode,
            onset_threshold_drumtranscriber=args.onset_threshold_drumtranscriber,
            loop_start_offset_ms=args.loop_start_offset_ms,
            skip_existing=False,
            create_audio_examples=not args.no_audio_examples,
            daw_ready=args.daw_ready,
            manual_start=args.manual_start,
            manual_duration=args.manual_duration,
            export_format=args.export_format,
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
