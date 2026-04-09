#!/usr/bin/env python3
"""
Loop Extractor Pipeline - Main Orchestrator

Complete pipeline for music microtiming analysis and loop extraction:
1. Stem separation (Spleeter)
2. Beat detection (Beat-Transformer via subprocess)
3. Downbeat correction
3.5. Tempo plots (8-panel comparison: uncorrected vs corrected, + bar tempo CSV)
4. SongFormer music structure analysis (sections, boundaries - SOTA 70% accuracy)
4.5. Full snippet WAV extraction (with fade in/out)
5. Onset detection (librosa, from drum stem)
5.1. Drum transcription (DrumTranscriber CNN - optional, for drumtranscriber onset mode)
5.2. Filter close onsets (for drumtranscriber mode)
5.5. Pattern length detection (drum/mel/pitch methods with circular convolution)
6. Raster/grid calculations
6.1. Section anchoring (anchor onsets to SongFormer sections with FlexStart)
6.2. Filter anchored patterns (Tukey IQR outlier removal)
6.5. Raster plots
6.6. Microtiming plots
6.7. Rhythm histograms
6.8. Full song histograms
7. Anchored rhythm histograms (per-section position histograms from 6.1 data)
7.1. Anchored beat histograms (per-section IOI histograms from filtered patterns)
7.2. Anchored statistics (rhythm + beat statistics from 6.6 and 6.7 data)
8. Audio example generation
9. LEPA data export
10. MIDI export (actual onset times, one loop per method: drum, mel, pitch)
11. Stem loop export (WAV/MP3 loops for each stem, one loop per method: drum, mel, pitch)
11.1. Section extraction (extract audio sections from filtered pattern boundaries)
13. Pironio pulse clarity metrics (viterbi, entropy, peak analysis)
13.1. Pironio section metrics (pulse clarity for each extracted section)
14. Spotify audio features (danceability, energy, valence, tempo, etc.)
15. Yodfat rhythmic complexity analysis
15.1. Yodfat section analysis (rhythmic complexity for each extracted section)

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

    4. DrumTranscriber model (optional, for Step 5.1)
       - Location: drumtranscriber/model/drum_transcriber.h5
       - Download from: https://drive.google.com/file/d/1w2fIHeyr-st3sbk1PYrtGOYW6YAD1fsi/view
       - Repository: https://github.com/yoshi-man/DrumTranscriber
       - Note: Pipeline will skip Step 5.1 if model is not available

Usage:
    python main.py --audio track.wav --track-id 123 --output-dir output/
    python main.py --batch --input-dir audio/ --output-dir output/ --start-id 0 --end-id 100
"""

import argparse
from pathlib import Path
import sys
import json
import subprocess
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
from analysis import correct_bars, raster, rms_grid_histograms, onset_detection, pattern_detection, tempo_plots, anchoring, extract_sections, write_clicks_to_sectionwavs
from utils import audio_export, raster_plots, midi_export, microtiming_plots, anchored_microtiming_plots, drumtranscriber_interface
import main_pironio
import spotify_analysis
import yodfat_analysis
import songformer_analysis


def run_complete_pipeline(
    audio_file: str,
    track_id: str,
    output_dir: str,
    pattern_file: Optional[str] = None,
    snippet_offset_file: Optional[str] = None,
    onset_file: Optional[str] = None,
    onset_mode: str = 'librosa',
    onset_threshold_drumtranscriber: float = 0.5,  # Filter onsets closer than this fraction of 1/16th note (0.5 = 1/32nd)
    onset_threshold_madmom: float = 0.5,  # Madmom onset detection threshold (0.3-0.7, lower = more sensitive)
    loop_start_offset_ms: float = 0.0,  # Loop start offset in ms (0.0 = use grid time exactly, negative was adding silence)
    anchoring_mode: str = 'double',  # 'single' or 'double' anchoring
    skip_existing: bool = False,
    create_audio_examples: bool = True,
    daw_ready: bool = False,
    manual_start: Optional[float] = None,
    manual_duration: Optional[float] = None,
    export_format: str = 'wav',
    reuse_existing: bool = False,  # Reuse existing stems/beats/songformer, skip steps 1-4.5, 5.5
    all_stems: bool = False,  # Run onset/grid analysis for all 5 stems
    fullmix: bool = False,  # Also calculate on full mix (non-separated audio)
    fullmix_dir: Optional[str] = None,  # Directory with original fullmix WAVs (for reuse mode)
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
        'warnings': [],
        'errors': []
    }

    # ========================================================================
    # FULLMIX WAV LOCATING (if fullmix calculation enabled)
    # ========================================================================
    fullmix_wav_path = None
    if fullmix:
        if reuse_existing:
            # Reuse mode: look for {track_id}.wav in fullmix_dir
            if fullmix_dir is None:
                raise ValueError("--fullmix-dir is required when using --fullmix with --reuse-existing")

            fullmix_dir_path = Path(fullmix_dir)
            if not fullmix_dir_path.exists():
                raise FileNotFoundError(f"Fullmix directory not found: {fullmix_dir}")

            # Try to find matching WAV file
            fullmix_wav_path = fullmix_dir_path / f"{track_id}.wav"
            if not fullmix_wav_path.exists():
                raise FileNotFoundError(f"Fullmix WAV not found: {fullmix_wav_path}")

            if verbose:
                print(f"Fullmix WAV located: {fullmix_wav_path}")
        else:
            # Normal mode: use the input audio file directly
            fullmix_wav_path = audio_file
            if verbose:
                print(f"Fullmix WAV: using input file {fullmix_wav_path}")

    # ========================================================================
    # STEP 1: STEM SEPARATION
    # ========================================================================
    try:
        if reuse_existing and paths['npz_file'].exists():
            if verbose:
                print("\n[1/7] Stem separation - SKIPPED (reuse existing)")
            results['steps_completed'].append('stem_separation_skipped')
        elif skip_existing and paths['npz_file'].exists():
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
        if reuse_existing and paths['beats_file'].exists():
            if verbose:
                print("\n[2/7] Beat detection - SKIPPED (reuse existing)")
            results['steps_completed'].append('beat_detection_skipped')
        elif skip_existing and paths['beats_file'].exists():
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
        if reuse_existing and paths['corrected_downbeats_file'].exists():
            if verbose:
                print("\n[3] Downbeat correction - SKIPPED (reuse existing)")
            results['steps_completed'].append('correct_bars_skipped')
        elif skip_existing and paths['corrected_downbeats_file'].exists():
            if verbose:
                print("\n[3] Downbeat correction - SKIPPED (exists)")
            results['steps_completed'].append('correct_bars_skipped')
        else:
            if verbose:
                print("\n[3] Downbeat correction...")

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
        if reuse_existing and paths['tempo_csv'].exists():
            if verbose:
                print("\n[3.5] Tempo plots - SKIPPED (reuse existing)")
            results['steps_completed'].append('tempo_plots_skipped')
        elif skip_existing and paths['tempo_plots_pdf'].exists() and paths['tempo_csv'].exists():
            if verbose:
                print("\n[3.5] Tempo plots - SKIPPED (exists)")
            results['steps_completed'].append('tempo_plots_skipped')
        else:
            if daw_ready:
                if verbose:
                    print("\n[3.5] Generating tempo CSV (plots skipped in DAW mode)...")
            else:
                if verbose:
                    print("\n[3.5] Generating tempo plots...")

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
    # STEP 4: SONGFORMER MUSIC STRUCTURE ANALYSIS
    # ========================================================================
    try:
        snippet_offset_val = results['time_range'].get('actual_start', 30.0)
        snippet_dur_val = results['time_range'].get('actual_duration', 30.0)
        songformer_json = paths['songformer_dir'] / 'SF_sections.json'

        if reuse_existing and songformer_json.exists():
            if verbose:
                print("\n[4] SongFormer structure analysis - SKIPPED (reuse existing)")
            results['steps_completed'].append('songformer_skipped')
        elif skip_existing and songformer_json.exists():
            if verbose:
                print("\n[4] SongFormer structure analysis - SKIPPED (exists)")
            results['steps_completed'].append('songformer_skipped')

            # Still create plots if they don't exist
            sf_snippet_plot = paths['songformer_dir'] / 'SF_snippet_sections.png'
            if not daw_ready and not sf_snippet_plot.exists():
                if verbose:
                    print("  Creating SongFormer plots...")
                sf_plot_results = songformer_analysis.create_songformer_plots(
                    songformer_json_path=songformer_json,
                    output_dir=paths['songformer_dir'],
                    track_id=track_id,
                    snippet_start=snippet_offset_val,
                    snippet_duration=snippet_dur_val,
                    track_name=track_id,
                    downbeats_file=paths['corrected_downbeats_file'] if paths['corrected_downbeats_file'].exists() else None,
                    verbose=verbose
                )
                results['songformer_plots'] = sf_plot_results
        else:
            if verbose:
                print("\n[4] SongFormer music structure analysis...")

            songformer_results = songformer_analysis.run_songformer(
                audio_path=Path(audio_file),
                output_dir=paths['songformer_dir'],
                track_id=track_id,
                verbose=verbose
            )

            if songformer_results.get('errors'):
                results['warnings'].append(f"SongFormer: {songformer_results['errors']}")
                if verbose:
                    print(f"  ⚠ Warning: {songformer_results['errors']}")
            else:
                results['songformer_sections'] = songformer_results.get('sections', [])
                results['songformer_boundaries'] = songformer_results.get('boundaries', [])
                results['steps_completed'].append('songformer')
                if verbose:
                    num_sections = len(songformer_results.get('sections', []))
                    print(f"  ✓ SongFormer completed: {num_sections} sections detected")

                # Create plots
                if not daw_ready:
                    sf_plot_results = songformer_analysis.create_songformer_plots(
                        songformer_json_path=songformer_json,
                        output_dir=paths['songformer_dir'],
                        track_id=track_id,
                        snippet_start=snippet_offset_val,
                        snippet_duration=snippet_dur_val,
                        track_name=track_id,
                        downbeats_file=paths['corrected_downbeats_file'] if paths['corrected_downbeats_file'].exists() else None,
                        verbose=verbose
                    )
                    results['songformer_plots'] = sf_plot_results

    except Exception as e:
        error_msg = f"Step 4 failed: {e}"
        results['warnings'].append(error_msg)
        if verbose:
            print(f"  ⚠ WARNING: {e} (continuing without SongFormer)")
        # Don't raise - SongFormer is optional, continue with pipeline

    # ========================================================================
    # STEP 4.5: CREATE FULL SNIPPET WAV
    # ========================================================================
    try:
        snippet_wav_path = paths['stems_dir'] / 'full_snippet.wav'
        snippet_offset_val = results['time_range'].get('actual_start', 30.0)
        snippet_dur_val = results['time_range'].get('actual_duration', 30.0)

        if reuse_existing and snippet_wav_path.exists():
            if verbose:
                print("\n[4.5] Full snippet WAV - SKIPPED (reuse existing)")
            results['steps_completed'].append('snippet_wav_skipped')
        elif skip_existing and snippet_wav_path.exists():
            if verbose:
                print("\n[4.5] Full snippet WAV - SKIPPED (exists)")
            results['steps_completed'].append('snippet_wav_skipped')
        else:
            if verbose:
                print("\n[4.5] Creating full snippet WAV...")

            snippet_wav = spleeter_interface.create_snippet_wav(
                str(audio_file),
                str(snippet_wav_path),
                start_time=snippet_offset_val,
                duration=snippet_dur_val,
                fade_duration=0.05  # 50ms fade in/out
            )

            results['snippet_wav'] = str(snippet_wav)
            results['steps_completed'].append('snippet_wav')

            if verbose:
                print(f"  ✓ Full snippet created")

    except Exception as e:
        error_msg = f"Step 4.5 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue with pipeline

    # ========================================================================
    # STEP 5: ONSET DETECTION (for each stem)
    # ========================================================================
    # Determine which stems to process
    onset_stems = config.STEMS if all_stems else config.ONSET_STEMS
    # Append fullmix if enabled
    if fullmix:
        onset_stems = list(onset_stems) + ['fullmix']  # Convert to list and append
    results['onset_stems'] = onset_stems
    results['onset_files'] = {}

    for stem in onset_stems:
        try:
            # Get stem-specific paths
            stem_paths = config.get_stem_paths(track_id, stem, Path(output_dir))

            # Create stem output directory
            stem_paths['onsets_dir'].mkdir(parents=True, exist_ok=True)

            stem_onset_file = stem_paths['onsets_file']

            # Skip librosa for drums when using drumtranscriber mode
            # (DrumTranscriber will create the onsets in Step 5.1)
            if stem == 'drums' and onset_mode == 'drumtranscriber':
                if verbose:
                    print(f"\n[5] Onset detection ({stem}) - SKIPPED (using drumtranscriber mode)")
                results['steps_completed'].append(f'onset_detection_{stem}_skipped_drumtranscriber')
                continue

            if skip_existing and Path(stem_onset_file).exists():
                if verbose:
                    print(f"\n[5] Onset detection ({stem}) - SKIPPED (exists)")
                results['steps_completed'].append(f'onset_detection_{stem}_skipped')
                results['onset_files'][stem] = str(stem_onset_file)
            else:
                if verbose:
                    print(f"\n[5] Onset detection from {stem} stem...")

                # Detect onsets from stem (or fullmix)
                if stem == 'fullmix':
                    stem_wav = fullmix_wav_path
                else:
                    stem_wav = paths['stems_dir'] / f'{stem}.wav'

                if not stem_wav.exists():
                    raise FileNotFoundError(f"{stem.capitalize()} {'WAV' if stem == 'fullmix' else 'stem'} not found: {stem_wav}")

                # Use madmom CNN onset detection if specified
                if onset_mode == 'madmom':
                    if verbose:
                        print(f"  Using madmom CNN onset detection (threshold={onset_threshold_madmom})...")

                    # Path to madmom onset detection script
                    madmom_script = Path(__file__).parent / 'analysis' / 'onset_detection_madmom.py'

                    # Call madmom script via subprocess (uses new_beatnet_env)
                    cmd = [
                        config.MADMOM_PYTHON,
                        str(madmom_script),
                        '--audio', str(stem_wav),
                        '--output', str(stem_onset_file),
                        '--threshold', str(onset_threshold_madmom)
                    ]

                    result = subprocess.run(cmd, check=True, capture_output=not verbose, text=True)

                    if not verbose and result.stdout:
                        print(result.stdout)

                    # Read the saved CSV to get onset count
                    import pandas as pd
                    onset_df = pd.read_csv(stem_onset_file)
                    num_onsets = len(onset_df)

                    results['onset_files'][stem] = str(stem_onset_file)
                    results[f'num_onsets_{stem}'] = num_onsets
                    results['steps_completed'].append(f'onset_detection_{stem}_madmom')

                    if verbose:
                        print(f"  ✓ Detected {num_onsets} onsets with madmom")
                        print(f"  ✓ Saved to: {stem_onset_file}")
                else:
                    # Use librosa onset detection (default)
                    onsets, onset_file_path = onset_detection.detect_and_save_onsets(
                        str(stem_wav),
                        str(stem_onset_file),
                        hop_length=512,
                        backtrack=False,
                        delta=0.12,
                        refine_onsets=False,
                        min_interval_s=0.15,
                        sr=22050
                    )

                    results['onset_files'][stem] = str(onset_file_path)
                    results[f'num_onsets_{stem}'] = len(onsets)
                    results['steps_completed'].append(f'onset_detection_{stem}')

                    if verbose:
                        print(f"  ✓ Detected {len(onsets)} onsets")
                        print(f"  ✓ Saved to: {onset_file_path}")

        except Exception as e:
            error_msg = f"Step 5 ({stem}) failed: {e}"
            results['errors'].append(error_msg)
            if verbose:
                print(f"  ✗ ERROR: {e}")
            # Continue with other stems instead of raising
            continue

    # For backwards compatibility, set onset_file to drums onset file
    if 'drums' in results['onset_files']:
        onset_file = results['onset_files']['drums']
    elif onset_file is None:
        onset_file = paths['onsets_file']  # Legacy path

    # ========================================================================
    # STEP 5.1: DRUM TRANSCRIPTION (if using drumtranscriber onset mode)
    # ========================================================================
    # Run DrumTranscriber before pattern detection so we can use CNN-detected onsets
    try:
        # Check if DrumTranscriber is available and onset_mode is drumtranscriber
        if onset_mode == 'drumtranscriber':
            if not drumtranscriber_interface.DRUMTRANSCRIBER_AVAILABLE:
                if verbose:
                    print("\n[5.1] Drum transcription - SKIPPED (DrumTranscriber not available)")
                    print("      Falling back to librosa onsets from Step 5")
                results['steps_completed'].append('drumtranscriber_unavailable')
            elif daw_ready:
                # Skip in DAW mode (not essential for loop creation)
                if verbose:
                    print("\n[5.1] Drum transcription - SKIPPED (DAW mode)")
                    print("      Using librosa onsets from Step 5")
                results['steps_completed'].append('drumtranscriber_skipped_daw')
            else:
                # Always run DrumTranscriber fresh when this mode is selected
                if verbose:
                    print("\n[5.1] Drum transcription...")

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

                # Copy DrumTranscriber onsets to standard drums onset location
                # This ensures all downstream steps (anchoring, etc.) use the same file
                # Note: paths['onsets_file'] is already 4_onsets/drums/{track}_onsets.csv
                if 'onsets_csv' in transcription_results:
                    import shutil
                    drum_onset_path = paths['onsets_file']  # Use the exact path config expects
                    drum_onset_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(transcription_results['onsets_csv'], drum_onset_path)
                    onset_file = str(drum_onset_path)
                    results['onset_files']['drums'] = onset_file
                    if verbose:
                        print(f"  ✓ Transcribed {transcription_results['summary']['total_hits']} drum hits")
                        print(f"  ✓ Copied onsets to: {drum_onset_path}")
                        print(f"  ✓ Onset mode: Using DrumTranscriber onsets for all analysis")
        else:
            # Using librosa onset mode (default)
            if verbose:
                print(f"\n[5.1] Onset mode: librosa (using onsets from Step 5)")

    except Exception as e:
        error_msg = f"Step 5.1 (DrumTranscriber) failed: {e} - falling back to librosa onsets"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ⚠ WARNING: {e}")
            print(f"  Falling back to librosa onsets from Step 5")

    # ========================================================================
    # STEP 5.2: FILTER CLOSE ONSETS (if using drumtranscriber mode)
    # ========================================================================
    try:
        if onset_mode == 'drumtranscriber' and onset_file:
            # Check if tempo CSV exists (created in Step 3.5)
            if paths['tempo_csv'].exists():
                if verbose:
                    print(f"\n[5.2] Filtering close onsets...")

                # Create filtered onset file path
                onset_file_path = Path(onset_file)
                filtered_onset_file = onset_file_path.parent / f'{track_id}_onsets_filtered.csv'

                # Check if filtered file already exists
                if skip_existing and filtered_onset_file.exists():
                    if verbose:
                        print(f"  ✓ Filtered onsets - SKIPPED (exists)")
                    onset_file = str(filtered_onset_file)
                    # Also ensure standard onset file has filtered version
                    import shutil
                    shutil.copy(filtered_onset_file, paths['onsets_file'])
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

                    # Also overwrite the standard onset file so downstream steps use filtered version
                    # (stem_paths['onsets_file'] is hardcoded, so we need to replace it)
                    shutil.copy(filtered_onset_file, paths['onsets_file'])

                    if verbose:
                        print(f"  ✓ Filtered onsets saved: {filtered_onset_file.name}")
                        print(f"  ✓ Updated standard onset file with filtered version")
            else:
                if verbose:
                    print(f"\n[5.2] Onset filtering - SKIPPED (no tempo CSV yet)")
        else:
            if verbose and onset_mode == 'drumtranscriber':
                print(f"\n[5.2] Onset filtering - SKIPPED (no onsets to filter)")

    except Exception as e:
        error_msg = f"Step 5.2 (Onset filtering) failed: {e} - using unfiltered onsets"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ⚠ WARNING: {e}")
            print(f"  Using unfiltered onsets")

    # ========================================================================
    # STEP 5.5: PATTERN LENGTH DETECTION
    # ========================================================================
    try:
        # Skip pattern detection when reusing existing files (bass F0 extraction is slow)
        if reuse_existing:
            pattern_lengths = {'drum': 4, 'mel': 4, 'pitch': 4, 'lepa': 4, 'aicc': 4}
            results['pattern_lengths'] = pattern_lengths
            results['steps_completed'].append('pattern_detection_skipped')
            if verbose:
                print("\n[5.5] Pattern length detection - SKIPPED (reuse existing)")
                print(f"    Using defaults: {pattern_lengths}")
        elif verbose:
            print("\n[5.5] Pattern length detection...")

        # Load pattern lengths from file if provided
        if not reuse_existing and pattern_file and Path(pattern_file).exists():
            pattern_lengths = raster.load_pattern_lengths(pattern_file, track_id)
            if verbose:
                print(f"    Loaded from file: {pattern_lengths}")
            results['steps_completed'].append('pattern_detection_loaded')
        elif not reuse_existing:
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
                snippet_dur = manual_duration if manual_duration is not None else 30.0
                snippet = (snippet_offset, snippet_offset + snippet_dur)
            elif config.OVERVIEW_CSV.exists():
                snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
                if snippet_offset > 0:
                    snippet_dur = manual_duration if manual_duration is not None else 30.0
                    snippet = (snippet_offset, snippet_offset + snippet_dur)

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

            # Onset file should exist from Step 5
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
        error_msg = f"Step 5.5 failed: {e} - using defaults {pattern_lengths}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ⚠ WARNING: {e}")
            print(f"  Using default pattern lengths: {pattern_lengths}")

    # ========================================================================
    # STEP 6: RASTER/GRID CALCULATIONS
    # ========================================================================
    try:
        if skip_existing and paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[6] Grid calculations - SKIPPED (exists)")
            results['steps_completed'].append('raster_skipped')
        else:
            if verbose:
                print("\n[6] Raster/grid calculations...")
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

            # Onset file should exist from Step 5
            if onset_file is None:
                onset_file = paths['onsets_file']

            if not Path(onset_file).exists():
                raise FileNotFoundError(f"Onset file not found (should have been created in Step 5): {onset_file}")

            # Create comprehensive CSV
            snippet_dur = manual_duration if manual_duration is not None else 30.0
            df_comp = raster.create_comprehensive_csv(
                str(paths['corrected_downbeats_file']),
                str(onset_file),
                pattern_lengths,
                snippet_offset,
                str(paths['comprehensive_csv']),
                snippet_duration=snippet_dur
            )

            results['comprehensive_csv'] = str(paths['comprehensive_csv'])
            results['pattern_lengths'] = pattern_lengths
            results['steps_completed'].append('raster')

            if verbose:
                print(f"  ✓ Comprehensive CSV created")

    except Exception as e:
        error_msg = f"Step 6 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue to RMS if we have the CSV from before

    # ========================================================================
    # STEP 6 (additional stems): RASTER/GRID CALCULATIONS FOR OTHER STEMS
    # ========================================================================
    # Process additional stems (skip drums, already processed above)
    results['comprehensive_csvs'] = {'drums': str(paths['comprehensive_csv'])}

    for stem in onset_stems:
        if stem == 'drums':
            continue  # Already processed above

        try:
            stem_paths = config.get_stem_paths(track_id, stem, Path(output_dir))
            stem_paths['grid_dir'].mkdir(parents=True, exist_ok=True)

            stem_onset_file = results['onset_files'].get(stem)
            if not stem_onset_file or not Path(stem_onset_file).exists():
                if verbose:
                    print(f"\n[6] Grid calculations ({stem}) - SKIPPED (no onset file)")
                continue

            if skip_existing and stem_paths['comprehensive_csv'].exists():
                if verbose:
                    print(f"\n[6] Grid calculations ({stem}) - SKIPPED (exists)")
                results['comprehensive_csvs'][stem] = str(stem_paths['comprehensive_csv'])
                continue

            if verbose:
                print(f"\n[6] Raster/grid calculations ({stem})...")

            # Load snippet offset (same as drums)
            if manual_start is not None:
                snippet_offset = manual_start
            elif snippet_offset_file and Path(snippet_offset_file).exists():
                snippet_offset = raster.load_snippet_offset(snippet_offset_file, track_id)
            elif config.OVERVIEW_CSV.exists():
                snippet_offset = raster.load_snippet_offset(str(config.OVERVIEW_CSV), track_id)
            else:
                snippet_offset = 0.0

            # Create comprehensive CSV for this stem
            snippet_dur = manual_duration if manual_duration is not None else 30.0
            df_comp = raster.create_comprehensive_csv(
                str(paths['corrected_downbeats_file']),
                str(stem_onset_file),
                pattern_lengths,
                snippet_offset,
                str(stem_paths['comprehensive_csv']),
                snippet_duration=snippet_dur
            )

            results['comprehensive_csvs'][stem] = str(stem_paths['comprehensive_csv'])
            results['steps_completed'].append(f'raster_{stem}')

            if verbose:
                print(f"  ✓ Comprehensive CSV created for {stem}")

        except Exception as e:
            error_msg = f"Step 6 ({stem}) failed: {e}"
            results['errors'].append(error_msg)
            if verbose:
                print(f"  ✗ ERROR ({stem}): {e}")
            continue

    # ========================================================================
    # STEP 6.1-6.4: SECTION ANCHORING (for all stems)
    # ========================================================================
    # Check if SongFormer sections exist (common for all stems)
    sf_overlapping_csv = paths['songformer_dir'] / 'SF_overlapping_sections.csv'
    sf_timings_csv = paths['songformer_dir'] / 'SF_snippet_timings.csv'

    # Track which stems were successfully processed (for downstream steps)
    processed_stems = []

    if daw_ready:
        if verbose:
            print("\n[6.1] Section anchoring - SKIPPED (DAW ready mode)")
        results['steps_completed'].append('anchoring_skipped_daw')
    elif not sf_overlapping_csv.exists() or not sf_timings_csv.exists():
        if verbose:
            print("\n[6.1] Section anchoring - SKIPPED (no SongFormer sections)")
        results['steps_completed'].append('anchoring_skipped_no_sections')
    elif not paths['corrected_downbeats_file'].exists():
        if verbose:
            print("\n[6.1] Section anchoring - SKIPPED (no corrected downbeats)")
        results['steps_completed'].append('anchoring_skipped_no_downbeats')
    else:
        # Loop through all stems for anchoring
        # IMPORTANT: drums must be first (defined in config.STEMS) because other stems
        # use drum anchoring as reference
        onset_stems = config.STEMS if all_stems else config.ONSET_STEMS
        # Append fullmix if enabled
        if fullmix:
            onset_stems = list(onset_stems) + ['fullmix']
        results['anchoring_files'] = {}
        results['anchoring_filtered_files'] = {}

        # Track drum filtered directory for use by other stems
        drum_filtered_dir = None

        for stem in onset_stems:
            try:
                stem_paths = config.get_stem_paths(track_id, stem, Path(output_dir))
                anchoring_dir = stem_paths['anchoring_dir']
                filtered_dir = stem_paths['filtered_patterns_dir']
                stem_onset_file = stem_paths['onsets_file']

                # Check if onset file exists for this stem
                if not stem_onset_file.exists():
                    if verbose:
                        print(f"\n[6.1] Section anchoring ({stem}) - SKIPPED (no onset file)")
                    continue

                # ================================================================
                # DRUMS: Full anchoring + filtering pipeline
                # ================================================================
                if stem == 'drums':
                    # Check if anchoring already exists
                    anchoring_exists = anchoring_dir.exists() and any(anchoring_dir.glob('*.csv'))

                    if skip_existing and anchoring_exists:
                        if verbose:
                            print(f"\n[6.1] Section anchoring ({stem}) - SKIPPED (exists)")
                        # Still add to processed_stems so downstream steps can run
                        processed_stems.append(stem)
                        drum_filtered_dir = filtered_dir
                        continue

                    if verbose:
                        print(f"\n[6.1] Section anchoring ({stem})...")

                    anchoring_results = anchoring.run_anchoring(
                        corrected_downbeats_file=str(paths['corrected_downbeats_file']),
                        onset_file=str(stem_onset_file),
                        songformer_sections_csv=str(sf_overlapping_csv),
                        snippet_timings_csv=str(sf_timings_csv),
                        output_dir=str(anchoring_dir),
                        pattern_lengths=[1, 2, 4],
                        anchoring_mode=anchoring_mode,
                        verbose=verbose
                    )

                    results['anchoring_files'][stem] = anchoring_results
                    processed_stems.append(stem)

                    if verbose:
                        num_files = len(anchoring_results)
                        print(f"  ✓ Section anchoring ({stem}) completed: {num_files} files created")

                    # Generate section anchoring plots
                    from analysis import plots_anchoring
                    plots_anchoring.create_all_anchoring_plots(
                        anchoring_dir=str(anchoring_dir),
                        track_id=track_id
                    )

                    # ================================================================
                    # STEP 6.2: FILTER ANCHORED PATTERNS (drums only)
                    # ================================================================
                    if verbose:
                        print(f"\n[6.2] Filtering anchored patterns ({stem})...")

                    from analysis.filter_anchored_patterns import filter_all_anchored_patterns
                    filter_results = filter_all_anchored_patterns(
                        anchoring_dir=anchoring_dir,
                        output_dir=filtered_dir,
                        iqr_multiplier=config.IQR_MULTIPLIER_TUKEY,
                        threshold=config.RUNNING_MEAN_THRESHOLD,
                        no_of_repetitions_TH=config.NO_OF_REPETITIONS_TH,
                        verbose=verbose
                    )

                    results['anchoring_filtered_files'][stem] = filter_results
                    drum_filtered_dir = filtered_dir

                    if verbose:
                        print(f"  ✓ Filtering ({stem}) completed: {len(filter_results)} files filtered")

                # ================================================================
                # OTHER STEMS: Apply drum anchoring (no independent anchoring/filtering)
                # ================================================================
                else:
                    # Check if drum filtered patterns exist
                    if drum_filtered_dir is None or not drum_filtered_dir.exists():
                        if verbose:
                            print(f"\n[6.1-6.2] {stem} - SKIPPED (no drum anchoring available)")
                        continue

                    # Check if filtered patterns already exist for this stem
                    filtered_exists = filtered_dir.exists() and any(filtered_dir.glob('*.csv'))

                    if skip_existing and filtered_exists:
                        if verbose:
                            print(f"\n[6.1-6.2] {stem} - SKIPPED (exists)")
                        processed_stems.append(stem)
                        continue

                    if verbose:
                        print(f"\n[6.1-6.2] Applying drum anchoring to {stem}...")

                    from analysis.apply_drum_anchoring import apply_drum_anchoring_to_stem
                    stem_filter_results = apply_drum_anchoring_to_stem(
                        drum_filtered_dir=drum_filtered_dir,
                        stem_onset_file=stem_onset_file,
                        output_dir=filtered_dir,
                        stem_name=stem,
                        verbose=verbose,
                        stems_dir=paths['stems_dir']
                    )

                    results['anchoring_filtered_files'][stem] = stem_filter_results
                    processed_stems.append(stem)

                    if verbose:
                        print(f"  ✓ Applied drum anchoring to {stem}: {len(stem_filter_results)} files created")

                # ================================================================
                # STEP 6.3: PLOT FILTERED PATTERNS RASTER (all stems)
                # ================================================================
                if verbose:
                    print(f"\n[6.3] Plotting filtered patterns raster ({stem})...")

                from analysis import plots_anchoring
                plots_anchoring.create_all_anchoring_plots(
                    anchoring_dir=str(filtered_dir),
                    track_id=track_id
                )

                if verbose:
                    print(f"  ✓ Filtered patterns raster plot ({stem}) created")

                # ================================================================
                # STEP 6.4: ONSET HISTOGRAMS (all stems)
                # ================================================================
                if verbose:
                    print(f"\n[6.4] Creating onset histograms ({stem})...")

                from analysis import anchored_onset_histograms

                # For drums: create histograms for both unfiltered and filtered
                if stem == 'drums':
                    anchored_onset_histograms.create_combined_onset_histograms(
                        filtered_dir=str(anchoring_dir),
                        output_dir=str(anchoring_dir),
                        track_id=track_id,
                        verbose=verbose
                    )

                # Create histograms for filtered patterns (output to 6.2 folder)
                anchored_onset_histograms.create_combined_onset_histograms(
                    filtered_dir=str(filtered_dir),
                    output_dir=str(filtered_dir),
                    track_id=track_id,
                    verbose=verbose
                )

                if verbose:
                    if stem == 'drums':
                        print(f"  ✓ Onset histograms ({stem}) created (unfiltered + filtered)")
                    else:
                        print(f"  ✓ Onset histograms ({stem}) created (filtered)")

            except Exception as e:
                error_msg = f"Step 6.1/6.2/6.3/6.4 failed for {stem}: {e}"
                results['errors'].append(error_msg)
                if verbose:
                    print(f"  ✗ ERROR ({stem}): {e}")
                continue

        # Mark steps as completed if at least one stem was processed
        if results['anchoring_files'] or results['anchoring_filtered_files']:
            results['steps_completed'].append('anchoring')
            results['steps_completed'].append('anchoring_filtering')
            results['steps_completed'].append('filtered_patterns_plot')
            results['steps_completed'].append('onset_histograms')

    # ========================================================================
    # STEP 6.2.5: ANCHORED MICROTIMING PLOTS (per section, per stem)
    # ========================================================================
    if daw_ready:
        if verbose:
            print("\n[6.2.5] Anchored microtiming plots - SKIPPED (DAW ready mode)")
        results['steps_completed'].append('anchored_microtiming_plots_skipped_daw')
    else:
        try:
            track_dir = Path(output_dir) / track_id
            filtered_patterns_dir = track_dir / '6.2_filtered_patterns'

            if not filtered_patterns_dir.exists():
                if verbose:
                    print("\n[6.2.5] Anchored microtiming plots - SKIPPED (no filtered patterns)")
                results['steps_completed'].append('anchored_microtiming_plots_skipped')
            else:
                # Check if any anchored microtiming plots already exist
                existing_plots = list(filtered_patterns_dir.glob('*/*_microtiming.pdf'))

                if skip_existing and existing_plots:
                    if verbose:
                        print("\n[6.2.5] Anchored microtiming plots - SKIPPED (exists)")
                    results['steps_completed'].append('anchored_microtiming_plots_skipped')
                else:
                    if verbose:
                        print("\n[6.2.5] Generating anchored microtiming plots...")

                    onset_stems = config.STEMS if all_stems else config.ONSET_STEMS
                    # Append fullmix if enabled
                    if fullmix:
                        onset_stems = list(onset_stems) + ['fullmix']
                    all_plots = anchored_microtiming_plots.create_all_anchored_microtiming_plots(
                        str(filtered_patterns_dir),
                        track_id,
                        stems=onset_stems
                    )

                    results['anchored_microtiming_plots'] = all_plots
                    results['steps_completed'].append('anchored_microtiming_plots')

                    if verbose:
                        total_plots = sum(len(v) for v in all_plots.values())
                        print(f"  ✓ Anchored microtiming plots created ({total_plots} files)")

        except Exception as e:
            error_msg = f"Step 6.2.5 failed: {e}"
            results['errors'].append(error_msg)
            if verbose:
                print(f"  ✗ ERROR: {e}")
            # Don't raise - continue to next step

    # ========================================================================
    # STEP 6.5: RASTER PLOTS (for all stems)
    # ========================================================================
    if daw_ready:
        if verbose:
            print("\n[6.5] Raster plots - SKIPPED (DAW ready mode)")
        results['steps_completed'].append('raster_plots_skipped_daw')
    else:
        onset_stems = config.STEMS if all_stems else config.ONSET_STEMS
        # Append fullmix if enabled
        if fullmix:
            onset_stems = list(onset_stems) + ['fullmix']
        raster_created = False

        for stem in onset_stems:
            try:
                stem_paths = config.get_stem_paths(track_id, stem, Path(output_dir))
                comprehensive_csv = stem_paths['comprehensive_csv']

                if not comprehensive_csv.exists():
                    if verbose:
                        print(f"\n[6.5] Raster plots ({stem}) - SKIPPED (no comprehensive CSV)")
                    continue

                grid_output_dir = comprehensive_csv.parent
                raster_files_exist = (grid_output_dir / f'{track_id}_raster_comparison.png').exists()

                if skip_existing and raster_files_exist:
                    if verbose:
                        print(f"\n[6.5] Raster plots ({stem}) - SKIPPED (exists)")
                    continue

                if verbose:
                    print(f"\n[6.5] Generating raster plots ({stem})...")

                raster_plots.create_all_plots(
                    str(comprehensive_csv),
                    str(grid_output_dir),
                    track_id
                )

                raster_created = True

                if verbose:
                    print(f"  ✓ Raster plots ({stem}) created")

            except Exception as e:
                error_msg = f"Step 6.5 failed for {stem}: {e}"
                results['errors'].append(error_msg)
                if verbose:
                    print(f"  ✗ ERROR ({stem}): {e}")
                continue

        if raster_created:
            results['steps_completed'].append('raster_plots')

    # ========================================================================
    # STEP 6.6: MICROTIMING PLOTS
    # ========================================================================
    try:
        if daw_ready:
            if verbose:
                print("\n[6.6] Microtiming plots - SKIPPED (DAW ready mode)")
            results['steps_completed'].append('microtiming_plots_skipped_daw')
        elif not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[6.6] Microtiming plots - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('microtiming_plots_skipped')
        else:
            # Check if microtiming plots already exist
            grid_output_dir = paths['comprehensive_csv'].parent
            microtiming_files_exist = (grid_output_dir / f'{track_id}_microtiming_plots.pdf').exists()

            if skip_existing and microtiming_files_exist:
                if verbose:
                    print("\n[6.6] Microtiming plots - SKIPPED (exists)")
                results['steps_completed'].append('microtiming_plots_skipped')
            else:
                if verbose:
                    print("\n[6.6] Generating microtiming deviation plots...")

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
        error_msg = f"Step 6.6 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue to rhythm histograms

    # ========================================================================
    # STEP 6.7: RHYTHM HISTOGRAMS
    # ========================================================================
    try:
        from utils import rhythm_histograms

        if daw_ready:
            if verbose:
                print("\n[6.7] Rhythm histograms - SKIPPED (DAW ready mode)")
            results['steps_completed'].append('rhythm_histograms_skipped_daw')
        elif not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[6.7] Rhythm histograms - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('rhythm_histograms_skipped')
        else:
            # Define rhythm output directory based on the track directory
            track_dir = Path(output_dir) / track_id
            rhythm_output_dir = track_dir / '5.5_rhythm'

            # Check if rhythm histograms already exist
            rhythm_files_exist = (rhythm_output_dir / f'{track_id}_rhythm_histograms.pdf').exists()

            if skip_existing and rhythm_files_exist:
                if verbose:
                    print("\n[6.7] Rhythm histograms - SKIPPED (exists)")
                results['steps_completed'].append('rhythm_histograms_skipped')
            else:
                if verbose:
                    print("\n[6.7] Generating rhythm histograms...")

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
                    # comprehensive_csv is at 5_grid/drums/track_comprehensive_phases.csv
                    track_root = paths['comprehensive_csv'].parent.parent.parent
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

                    # Create simple IOI histogram (from raw onset times)
                    simple_ioi_files = beat_histograms.create_simple_ioi_histogram(
                        str(paths['onsets_file']),
                        str(paths['corrected_downbeats_file']),
                        track_id,
                        str(beat_output_dir)
                    )

                    # Create simple IOI all crosses plot (from raw onset times)
                    simple_ioi_crosses_files = beat_histograms.create_simple_ioi_all_crosses(
                        str(paths['onsets_file']),
                        str(paths['corrected_downbeats_file']),
                        track_id,
                        str(beat_output_dir)
                    )

                    # Create snippet IOI plots (filtered to snippet time range)
                    snippet_info = results.get('snippet_info', {})
                    snippet_start = snippet_info.get('usable_start_s', 0.0)
                    snippet_duration = snippet_info.get('usable_duration_s', 30.0)

                    snippet_ioi_crosses_files = beat_histograms.create_snippet_ioi_all_crosses(
                        str(paths['onsets_file']),
                        str(paths['corrected_downbeats_file']),
                        track_id,
                        str(beat_output_dir),
                        snippet_start,
                        snippet_duration
                    )

                    snippet_ioi_histogram_files = beat_histograms.create_snippet_ioi_histogram(
                        str(paths['onsets_file']),
                        str(paths['corrected_downbeats_file']),
                        track_id,
                        str(beat_output_dir),
                        snippet_start,
                        snippet_duration
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

                    # comprehensive_csv is at 5_grid/drums/track_comprehensive_phases.csv
                    # So .parent.parent.parent gets to track root
                    track_root = paths['comprehensive_csv'].parent.parent.parent

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
        error_msg = f"Step 6.7 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue to full histograms

    # ========================================================================
    # STEP 6.8: FULL SONG HISTOGRAMS
    # ========================================================================
    try:
        if verbose:
            print("\n[6.8] Creating full song histograms...")

        if daw_ready:
            if verbose:
                print("  SKIPPED (DAW ready mode)")
            results['steps_completed'].append('full_histograms_skipped_daw')
        elif not paths['comprehensive_csv'].exists():
            if verbose:
                print("  SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('full_histograms_skipped')
        else:
            # Create full song histograms
            from utils import full_histograms

            # Create 5.8_full_histograms folder (same track_root as beat_histograms)
            # comprehensive_csv is at 5_grid/drums/track_comprehensive_phases.csv
            track_root = paths['comprehensive_csv'].parent.parent.parent
            full_hist_output_dir = track_root / '5.8_full_histograms'
            full_hist_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Create full song IOI histogram (from raw onset times)
                full_ioi_files = full_histograms.create_full_song_ioi_histogram(
                    str(paths['onsets_file']),
                    str(paths['corrected_downbeats_file']),
                    track_id,
                    str(full_hist_output_dir)
                )

                if verbose:
                    print(f"  ✓ Full song IOI histogram created")

            except Exception as e:
                if verbose:
                    print(f"  ! Warning: Could not create full song IOI histogram: {e}")

            results['steps_completed'].append('full_histograms')

    except Exception as e:
        error_msg = f"Step 6.8 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")
        # Don't raise - continue to RMS analysis

    # ========================================================================
    # STEP 7: ANCHORED RHYTHM HISTOGRAMS (for all stems)
    # ========================================================================
    from analysis import anchored_rhythm_histograms

    onset_stems = config.STEMS if all_stems else config.ONSET_STEMS
    # Append fullmix if enabled
    if fullmix:
        onset_stems = list(onset_stems) + ['fullmix']
    rhythm_hist_created = False

    # Store rhythm_hist_dir per stem for use in Step 7.1
    stem_rhythm_hist_dirs = {}

    for stem in onset_stems:
        try:
            stem_paths = config.get_stem_paths(track_id, stem, Path(output_dir))
            anchoring_dir = stem_paths['anchoring_dir']
            filtered_dir = stem_paths['filtered_patterns_dir']
            rhythm_hist_dir = stem_paths['rhythm_histograms_dir']

            # Check if filtered patterns exist for this stem
            if not filtered_dir.exists() or not any(filtered_dir.glob('*.csv')):
                if verbose:
                    print(f"\n[7] Anchored rhythm histograms ({stem}) - SKIPPED (no filtered patterns)")
                continue

            if verbose:
                print(f"\n[7] Creating anchored rhythm histograms ({stem})...")

            rhythm_hist_dir.mkdir(parents=True, exist_ok=True)
            stem_rhythm_hist_dirs[stem] = rhythm_hist_dir

            # Create for unfiltered patterns (from 6.1_anchoring)
            if anchoring_dir.exists():
                anchored_rhythm_histograms.create_anchored_rhythm_histograms(
                    anchoring_dir=str(anchoring_dir),
                    output_dir=str(rhythm_hist_dir),
                    track_id=track_id,
                    verbose=verbose
                )

            # Create for filtered patterns (from 6.2_filtered_patterns)
            rhythm_results_filtered = anchored_rhythm_histograms.create_anchored_rhythm_histograms(
                anchoring_dir=str(filtered_dir),
                output_dir=str(rhythm_hist_dir),
                track_id=track_id + '_filtered',
                verbose=verbose
            )

            # Create groove pulse histograms from the filtered rhythm histograms CSV
            if rhythm_results_filtered and rhythm_results_filtered.get('csv'):
                groove_pulse_results = anchored_rhythm_histograms.create_anchored_groove_pulse_histograms(
                    rhythm_histograms_csv=rhythm_results_filtered['csv'],
                    track_id=track_id + '_filtered',
                    output_dir=str(rhythm_hist_dir),
                    verbose=verbose
                )

                # Create rhythm patterns from the groove pulse CSV
                if groove_pulse_results and groove_pulse_results.get('csv'):
                    anchored_rhythm_histograms.create_anchored_rhythm_patterns(
                        groove_pulse_csv=groove_pulse_results['csv'],
                        track_id=track_id + '_filtered',
                        output_dir=str(rhythm_hist_dir),
                        verbose=verbose
                    )

            rhythm_hist_created = True

            if verbose:
                print(f"  ✓ Anchored rhythm histograms ({stem}) created")

        except Exception as e:
            error_msg = f"Step 7 failed for {stem}: {e}"
            results['errors'].append(error_msg)
            if verbose:
                print(f"  ✗ ERROR ({stem}): {e}")
            continue

    if rhythm_hist_created:
        results['steps_completed'].append('anchored_rhythm_histograms')

    # ========================================================================
    # STEP 7.1: ANCHORED BEAT HISTOGRAMS (for all stems)
    # ========================================================================
    from utils import anchored_beat_histograms

    onset_stems = config.STEMS if all_stems else config.ONSET_STEMS
    # Append fullmix if enabled
    if fullmix:
        onset_stems = list(onset_stems) + ['fullmix']
    beat_hist_created = False

    for stem in onset_stems:
        try:
            stem_paths = config.get_stem_paths(track_id, stem, Path(output_dir))
            filtered_dir = stem_paths['filtered_patterns_dir']
            beat_hist_dir = stem_paths['beat_histograms_dir']
            rhythm_hist_dir = stem_paths['rhythm_histograms_dir']

            # Check if filtered patterns exist for this stem
            if not filtered_dir.exists() or not any(filtered_dir.glob('*.csv')):
                if verbose:
                    print(f"\n[7.1] Anchored beat histograms ({stem}) - SKIPPED (no filtered patterns)")
                continue

            if verbose:
                print(f"\n[7.1] Creating anchored beat histograms ({stem})...")

            beat_hist_dir.mkdir(parents=True, exist_ok=True)

            # Create from filtered patterns (6.2_filtered_patterns -> 6.7_anchored_beat_histograms)
            anchored_beat_histograms.create_anchored_beat_histograms(
                filtered_patterns_dir=str(filtered_dir),
                track_id=track_id,
                output_dir=str(beat_hist_dir)
            )

            anchored_beat_histograms.create_anchored_beat_histograms_all_onsets(
                filtered_patterns_dir=str(filtered_dir),
                track_id=track_id,
                output_dir=str(beat_hist_dir)
            )

            # Create binary beat patterns from aggregated CSV
            aggregated_csv = beat_hist_dir / f'{track_id}_anchored_beat_histograms.csv'
            if aggregated_csv.exists():
                pattern_results = anchored_beat_histograms.create_anchored_beat_patterns(
                    beat_histograms_csv=str(aggregated_csv),
                    track_id=track_id,
                    output_dir=str(beat_hist_dir)
                )
                if verbose:
                    print(f"    Created {pattern_results.get('plots_created', 0)} beat pattern plots")

            # Create groove pulse beat histograms (filtered by groove positions from 6.6)
            groove_pulse_csv = rhythm_hist_dir / f'{track_id}_filtered_anchored_groove_pulse_histograms.csv'
            if groove_pulse_csv.exists():
                groove_beat_results = anchored_beat_histograms.create_anchored_groove_pulse_beat_histograms(
                    filtered_patterns_dir=str(filtered_dir),
                    track_id=track_id,
                    output_dir=str(beat_hist_dir),
                    groove_pulse_csv=str(groove_pulse_csv)
                )
                if verbose:
                    print(f"    Created {groove_beat_results.get('plots_created', 0)} groove pulse beat histogram plots")

                groove_beat_results_all = anchored_beat_histograms.create_anchored_groove_pulse_beat_histograms_all_onsets(
                    filtered_patterns_dir=str(filtered_dir),
                    track_id=track_id,
                    output_dir=str(beat_hist_dir),
                    groove_pulse_csv=str(groove_pulse_csv)
                )
                if verbose:
                    print(f"    Created {groove_beat_results_all.get('plots_created', 0)} groove pulse scatter plots")

            beat_hist_created = True

            if verbose:
                print(f"  ✓ Anchored beat histograms ({stem}) created")

        except Exception as e:
            error_msg = f"Step 7.1 failed for {stem}: {e}"
            results['errors'].append(error_msg)
            if verbose:
                print(f"  ✗ ERROR ({stem}): {e}")
            continue

    if beat_hist_created:
        results['steps_completed'].append('anchored_beat_histograms')

    # ========================================================================
    # STEP 7.2: ANCHORED STATISTICS (for all stems)
    # ========================================================================
    if daw_ready:
        if verbose:
            print("\n[7.2] Anchored statistics - SKIPPED (DAW ready mode)")
        results['steps_completed'].append('anchored_statistics_skipped_daw')
    else:
        from batch_analysis import anchored_rhythm_statistics

        onset_stems = config.STEMS if all_stems else config.ONSET_STEMS
        # Append fullmix if enabled
        if fullmix:
            onset_stems = list(onset_stems) + ['fullmix']
        stats_created = False

        for stem in onset_stems:
            try:
                stem_paths = config.get_stem_paths(track_id, stem, Path(output_dir))
                rhythm_hist_dir = stem_paths['rhythm_histograms_dir']
                beat_hist_dir = stem_paths['beat_histograms_dir']

                # Check if rhythm histograms exist for this stem
                rhythm_csv = rhythm_hist_dir / f'{track_id}_filtered_anchored_rhythm_histograms.csv'
                if not rhythm_csv.exists():
                    if verbose:
                        print(f"\n[7.2] Anchored statistics ({stem}) - SKIPPED (no rhythm histograms)")
                    continue

                if verbose:
                    print(f"\n[7.2] Calculating anchored statistics ({stem})...")

                # Output directory for statistics (stem-specific)
                stats_dir = track_dir / '6.8_anchored_statistics' / stem
                stats_dir.mkdir(parents=True, exist_ok=True)

                # Calculate anchored rhythm statistics
                try:
                    anchored_rhythm_statistics.anchored_statistics_for_track_stem(
                        rhythm_hist_dir=rhythm_hist_dir,
                        stats_dir=stats_dir,
                        track_id=track_id
                    )
                    if verbose:
                        print(f"  ✓ Anchored rhythm statistics ({stem}) calculated")
                except Exception as e:
                    if verbose:
                        print(f"  ! Warning: Could not calculate anchored rhythm statistics ({stem}): {e}")

                # Calculate anchored beat statistics
                try:
                    anchored_rhythm_statistics.anchored_beat_statistics_for_track_stem(
                        beat_hist_dir=beat_hist_dir,
                        stats_dir=stats_dir,
                        track_id=track_id
                    )
                    if verbose:
                        print(f"  ✓ Anchored beat statistics ({stem}) calculated")
                except Exception as e:
                    if verbose:
                        print(f"  ! Warning: Could not calculate anchored beat statistics ({stem}): {e}")

                stats_created = True

            except Exception as e:
                error_msg = f"Step 7.2 failed for {stem}: {e}"
                results['errors'].append(error_msg)
                if verbose:
                    print(f"  ✗ ERROR ({stem}): {e}")
                continue

        if stats_created:
            results['steps_completed'].append('anchored_statistics')

    # ========================================================================
    # STEP 7 (OLD): RMS ANALYSIS - COMMENTED OUT FOR FUTURE REFERENCE
    # ========================================================================
    # try:
    #     if daw_ready:
    #         if verbose:
    #             print("\n[7] RMS analysis - SKIPPED (DAW ready mode)")
    #         results['steps_completed'].append('rms_skipped_daw')
    #     elif not paths['comprehensive_csv'].exists():
    #         if verbose:
    #             print("\n[7] RMS analysis - SKIPPED (no comprehensive CSV)")
    #         results['steps_completed'].append('rms_skipped')
    #     elif skip_existing and paths['rms_summary'].exists():
    #         if verbose:
    #             print("\n[7] RMS analysis - SKIPPED (exists)")
    #         results['steps_completed'].append('rms_analysis_skipped')
    #     else:
    #         if verbose:
    #             print("\n[7] RMS histogram analysis...")
    #
    #         rms_values = rms_grid_histograms.calculate_rms_from_csv(
    #             str(paths['comprehensive_csv'])
    #         )
    #
    #         if rms_values:
    #             # Save RMS summary as JSON
    #             with open(paths['rms_summary'], 'w') as f:
    #                 # Convert numpy types to Python types for JSON
    #                 rms_json = {k: float(v) if not isinstance(v, dict) else v
    #                            for k, v in rms_values.items()}
    #                 json.dump(rms_json, f, indent=2)
    #
    #             results['rms_values'] = rms_values
    #             results['steps_completed'].append('rms_analysis')
    #
    #             if verbose:
    #                 print(f"  ✓ RMS calculated:")
    #                 print(f"    Uncorrected: {rms_values['uncorrected_ms']:.2f}ms")
    #                 print(f"    Per-snippet: {rms_values['per_snippet_ms']:.2f}ms")
    #                 print(f"    Drum method: {rms_values['drum_ms']:.2f}ms")
    #         else:
    #             if verbose:
    #                 print(f"  ⚠️  RMS calculation returned no values")
    #
    # except Exception as e:
    #     error_msg = f"Step 7 failed: {e}"
    #     results['errors'].append(error_msg)
    #     if verbose:
    #         print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 8: AUDIO EXAMPLES
    # ========================================================================
    if create_audio_examples:
        try:
            if not paths['comprehensive_csv'].exists():
                if verbose:
                    print("\n[8] Audio examples - SKIPPED (no comprehensive CSV)")
                results['steps_completed'].append('audio_examples_skipped')
            else:
                # Check if audio examples already exist
                audio_files_exist = (
                    (paths['audio_examples_dir'] / 'drum.mp3').exists() or
                    (paths['audio_examples_dir'] / 'uncorrected.mp3').exists()
                )

                if skip_existing and audio_files_exist:
                    if verbose:
                        print("\n[8] Audio examples - SKIPPED (exists)")
                    results['steps_completed'].append('audio_examples_skipped')
                else:
                    if verbose:
                        print("\n[8] Audio examples...")

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
                        export_format=export_format,
                        track_id=track_id
                    )

                    results['steps_completed'].append('audio_examples')

                    if verbose:
                        print(f"  ✓ Audio examples created")

        except Exception as e:
            error_msg = f"Step 8 failed: {e}"
            results['errors'].append(error_msg)
            if verbose:
                print(f"  ✗ ERROR: {e}")
    else:
        if verbose:
            print("\n[8] Audio examples - SKIPPED (disabled)")
        results['steps_completed'].append('audio_examples_disabled')

    # ========================================================================
    # STEP 9: LEPA DATA EXPORT
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[9] LEPA export - SKIPPED (no comprehensive CSV)")
            results['steps_completed'].append('lepa_export_skipped')
        else:
            # Define LEPA output directory
            lepa_output_dir = Path(output_dir) / track_id / '10_output_for_lepa'

            # Check if LEPA export already exists (check for L1 file)
            lepa_file_exists = (lepa_output_dir / f'{track_id}_bar_durations_L1.csv').exists()

            if skip_existing and lepa_file_exists:
                if verbose:
                    print("\n[9] LEPA export - SKIPPED (exists)")
                results['steps_completed'].append('lepa_export_skipped')
            else:
                if verbose:
                    print("\n[9] Exporting LEPA bar duration data...")

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
        error_msg = f"Step 9 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 10: MIDI EXPORT
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[10] MIDI export - SKIPPED (no comprehensive CSV)")
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
                    print("\n[10] MIDI export - SKIPPED (exists)")
                results['steps_completed'].append('midi_export_skipped')
            else:
                if verbose:
                    print("\n[10] MIDI export...")

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
                        print("\n  [10] MIDI export (drum method + bass pitch)...")
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
                        print("\n  [10a] Onset-based MIDI (drum hits + FlexStart grid)...")
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
                        print("\n  [10b] Bass pitch MIDI (all methods + FlexStart)...")
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
        error_msg = f"Step 10 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 11: STEM LOOP EXPORT
    # ========================================================================
    try:
        if not paths['comprehensive_csv'].exists():
            if verbose:
                print("\n[11] Stem loop export - SKIPPED (no comprehensive CSV)")
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
                    print("\n[11] Stem loop export - SKIPPED (exists)")
                results['steps_completed'].append('loops_skipped')
            else:
                if verbose:
                    print("\n[11] Stem loop export...")

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
        error_msg = f"Step 11 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 11.1: SECTION EXTRACTION (for all stems)
    # ========================================================================
    onset_stems = config.STEMS if all_stems else config.ONSET_STEMS
    # Append fullmix if enabled
    if fullmix:
        onset_stems = list(onset_stems) + ['fullmix']
    sections_extracted = False
    results['sections_extracted'] = {}

    for stem in onset_stems:
        try:
            stem_paths = config.get_stem_paths(track_id, stem, Path(output_dir))
            sections_dir = stem_paths['sections_dir']
            filtered_dir = stem_paths['filtered_patterns_dir']

            # Check if sections already exist
            existing_sections = list(sections_dir.glob('*_section.wav')) if sections_dir.exists() else []

            if skip_existing and len(existing_sections) > 0:
                if verbose:
                    print(f"\n[11.1] Section extraction ({stem}) - SKIPPED ({len(existing_sections)} exist)")
                continue
            elif not filtered_dir.exists() or not any(filtered_dir.glob('*.csv')):
                if verbose:
                    print(f"\n[11.1] Section extraction ({stem}) - SKIPPED (no filtered patterns)")
                continue

            if verbose:
                print(f"\n[11.1] Extracting sections from filtered patterns ({stem})...")

            # Use stem-specific audio file (or fullmix)
            if stem == 'fullmix':
                stem_audio_path = fullmix_wav_path
            else:
                stem_audio_path = paths['stems_dir'] / f'{stem}.wav'

            if not stem_audio_path.exists():
                if verbose:
                    print(f"  ⚠️  {'Fullmix' if stem == 'fullmix' else 'Stem'} audio not found: {stem_audio_path.name}, skipping")
                continue

            extracted = extract_sections.extract_all_sections(
                filtered_dir=filtered_dir,
                audio_path=stem_audio_path,
                output_dir=sections_dir,
                fade_duration=0.05,  # 50ms fade in/out
                sr=44100,
                verbose=verbose
            )

            results['sections_extracted'][stem] = len(extracted)
            sections_extracted = True

            if verbose:
                print(f"  ✓ Extracted {len(extracted)} sections ({stem})")

        except Exception as e:
            error_msg = f"Step 11.1 failed for {stem}: {e}"
            results['errors'].append(error_msg)
            if verbose:
                print(f"  ✗ ERROR ({stem}): {e}")
            continue

    if sections_extracted:
        results['steps_completed'].append('section_extraction')

    # ========================================================================
    # STEP 11.2: SECTION CLICK TRACKS (rhythm pattern clicks on section WAVs)
    # ========================================================================
    try:
        track_output_dir = Path(output_dir) / track_id
        rhythm_hist_dir = track_output_dir / '6.6_anchored_rhythm_histograms'

        # Check if rhythm histograms exist (prerequisite)
        if not rhythm_hist_dir.exists():
            if verbose:
                print(f"\n[11.2] Section click tracks - SKIPPED (no 6.6_anchored_rhythm_histograms)")
        else:
            # Check if click WAVs already exist
            sections_dir = track_output_dir / '9.1_sections'
            existing_clicks = list(sections_dir.rglob('*_section_clicks.wav')) if sections_dir.exists() else []

            if skip_existing and len(existing_clicks) > 0:
                if verbose:
                    print(f"\n[11.2] Section click tracks - SKIPPED ({len(existing_clicks)} exist)")
            else:
                if verbose:
                    print(f"\n[11.2] Writing rhythm pattern click tracks onto section WAVs...")

                created = write_clicks_to_sectionwavs.write_clicks_to_section_wavs(
                    track_dir=track_output_dir,
                    stems=onset_stems,
                    sr=44100,
                    click_volume_db=0.0,
                    verbose=verbose,
                )

                results['section_click_tracks'] = len(created)
                if created:
                    results['steps_completed'].append('section_click_tracks')

    except Exception as e:
        error_msg = f"Step 11.2 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 13: PIRONIO PULSE CLARITY METRICS
    # ========================================================================
    try:
        pironio_json = paths['pironio_dir'] / f'{track_id}_pironio_metrics.json'
        snippet_wav_path = paths['stems_dir'] / 'full_snippet.wav'

        if reuse_existing and pironio_json.exists():
            if verbose:
                print("\n[13] Pironio pulse clarity - SKIPPED (reuse existing)")
            results['steps_completed'].append('pironio_skipped')
        elif skip_existing and pironio_json.exists():
            if verbose:
                print("\n[13] Pironio pulse clarity - SKIPPED (exists)")
            results['steps_completed'].append('pironio_skipped')
        elif not snippet_wav_path.exists():
            if verbose:
                print("\n[13] Pironio pulse clarity - SKIPPED (no snippet WAV)")
            results['steps_completed'].append('pironio_no_snippet')
        else:
            if verbose:
                print("\n[13] Computing Pironio pulse clarity metrics...")

            pironio_results = main_pironio.run_pironio_analysis(
                audio_file=str(snippet_wav_path),
                output_dir=str(Path(output_dir) / track_id),
                track_id=track_id,
                downbeat_model=True,
                compute_slow_metrics=True,  # All 8 metrics
                verbose=verbose
            )

            results['pironio'] = pironio_results.get('metrics', {})
            results['pironio_json'] = pironio_results.get('output_json')
            results['steps_completed'].append('pironio')

            if verbose:
                num_metrics = len(pironio_results.get('metrics', {}))
                print(f"  ✓ Computed {num_metrics} pulse clarity metrics")

    except Exception as e:
        error_msg = f"Step 13 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 13.1: PIRONIO SECTION METRICS
    # ========================================================================
    try:
        sections_dir = paths['sections_dir']
        pironio_sections_json = paths['pironio_dir'] / f'{track_id}_pironio_sections.json'

        # Check if sections exist
        section_wavs = list(sections_dir.glob('*_section.wav')) if sections_dir.exists() else []

        if reuse_existing and pironio_sections_json.exists():
            if verbose:
                print("\n[13.1] Pironio section metrics - SKIPPED (reuse existing)")
            results['steps_completed'].append('pironio_sections_skipped')
        elif skip_existing and pironio_sections_json.exists():
            if verbose:
                print("\n[13.1] Pironio section metrics - SKIPPED (exists)")
            results['steps_completed'].append('pironio_sections_skipped')
        elif len(section_wavs) == 0:
            if verbose:
                print("\n[13.1] Pironio section metrics - SKIPPED (no sections)")
            results['steps_completed'].append('pironio_sections_no_input')
        else:
            if verbose:
                print(f"\n[13.1] Computing Pironio metrics for {len(section_wavs)} sections...")

            # Run via subprocess (requires madmom in new_beatnet_env)
            run_script = Path(__file__).parent / "run_pironio.py"

            cmd = [
                config.BEAT_DETECTION_PYTHON,
                str(run_script),
                '--sections-dir', str(sections_dir),
                '--output-dir', str(paths['pironio_dir']),
                '--track-id', track_id
            ]

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=not verbose,
                text=True
            )

            if not verbose and result.stdout:
                print(result.stdout)

            results['pironio_sections_json'] = str(pironio_sections_json)
            results['steps_completed'].append('pironio_sections')

            if verbose:
                print(f"  ✓ Computed Pironio metrics for {len(section_wavs)} sections")

    except Exception as e:
        error_msg = f"Step 13.1 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 14: SPOTIFY AUDIO FEATURES
    # ========================================================================
    try:
        spotify_json = paths['spotify_dir'] / f'{track_id}_spotify_features.json'

        if skip_existing and spotify_json.exists():
            if verbose:
                print("\n[14] Spotify analysis - SKIPPED (exists)")
            results['steps_completed'].append('spotify_skipped')
        else:
            # Extract track name and artist from track_id if possible
            # Expected format: "17_Panini - Lil Nas X" -> track="Panini", artist="Lil Nas X"
            track_name_parsed = track_id
            artist_parsed = None

            if " - " in track_id:
                parts = track_id.split(" - ", 1)
                # Remove leading number if present (e.g., "17_Panini" -> "Panini")
                track_part = parts[0]
                if "_" in track_part:
                    track_name_parsed = track_part.split("_", 1)[1]
                else:
                    track_name_parsed = track_part
                artist_parsed = parts[1] if len(parts) > 1 else None

            # Skip Spotify API audio features (requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)
            # Uncomment below if you have credentials set up:
            # if verbose:
            #     print(f"\n[Step 13] Spotify audio features analysis...")
            #
            # spotify_results = spotify_analysis.run_spotify_analysis(
            #     track_name=track_name_parsed,
            #     output_dir=str(Path(output_dir) / track_id),
            #     track_id=track_id,
            #     artist=artist_parsed,
            #     verbose=verbose
            # )
            #
            # results['spotify'] = spotify_results.get('audio_features', {})
            # results['spotify_json'] = spotify_results.get('output_json')
            #
            # if spotify_results.get('errors'):
            #     results['steps_completed'].append('spotify_with_errors')
            # else:
            #     results['steps_completed'].append('spotify')

            # Run sections analysis using local groove-data
            snippet_start_s = results['time_range'].get('actual_start', 30.0)
            snippet_dur_s = results['time_range'].get('actual_duration', 30.0)

            sections_results = spotify_analysis.run_spotify_sections_analysis(
                output_dir=str(Path(output_dir) / track_id),
                track_id=track_id,
                snippet_start=snippet_start_s,
                snippet_duration=snippet_dur_s,
                track_name=f"{track_name_parsed} - {artist_parsed}" if artist_parsed else track_name_parsed,
                verbose=verbose
            )

            results['spotify_sections'] = sections_results.get('sections', [])
            results['spotify_sections_plot'] = sections_results.get('output_plot')

            if sections_results.get('errors'):
                results['steps_completed'].append('spotify_sections_with_errors')
            else:
                results['steps_completed'].append('spotify_sections')

    except Exception as e:
        error_msg = f"Step 14 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 15: YODFAT RHYTHMIC COMPLEXITY ANALYSIS
    # ========================================================================
    # Uses full_snippet.wav (same as Step 12 Pironio)
    # Calculates onset cross-correlation at quarter/half/full bar segments
    # High CC = low rhythmic complexity (more repetitive)
    # Low CC = high rhythmic complexity (more varied)
    # ========================================================================
    try:
        if not daw_ready:
            # Check for full_snippet.wav (created in Step 3.6)
            snippet_wav = paths['stems_dir'] / 'full_snippet.wav'

            if snippet_wav.exists():
                if verbose:
                    print(f"\n[15] Yodfat rhythmic complexity analysis...")

                yodfat_results = yodfat_analysis.run_yodfat_analysis(
                    audio_file=str(snippet_wav),
                    output_dir=str(Path(output_dir) / track_id),
                    track_id=track_id,
                    verbose=verbose
                )

                results['yodfat'] = yodfat_results.get('metrics', {})
                results['yodfat_json'] = yodfat_results.get('output_json')

                if yodfat_results.get('errors'):
                    results['steps_completed'].append('yodfat_with_errors')
                else:
                    results['steps_completed'].append('yodfat')
            else:
                if verbose:
                    print(f"\n[15] Skipping Yodfat analysis - full_snippet.wav not found")
                results['errors'].append("Step 15 skipped: full_snippet.wav not found")

    except Exception as e:
        error_msg = f"Step 15 failed: {e}"
        results['errors'].append(error_msg)
        if verbose:
            print(f"  ✗ ERROR: {e}")

    # ========================================================================
    # STEP 15.1: YODFAT SECTION ANALYSIS
    # ========================================================================
    try:
        sections_dir = paths['sections_dir']
        yodfat_sections_json = paths['yodfat_dir'] / f'{track_id}_yodfat_sections.json'

        # Check if sections exist
        section_wavs = list(sections_dir.glob('*_section.wav')) if sections_dir.exists() else []

        if reuse_existing and yodfat_sections_json.exists():
            if verbose:
                print("\n[15.1] Yodfat section analysis - SKIPPED (reuse existing)")
            results['steps_completed'].append('yodfat_sections_skipped')
        elif skip_existing and yodfat_sections_json.exists():
            if verbose:
                print("\n[15.1] Yodfat section analysis - SKIPPED (exists)")
            results['steps_completed'].append('yodfat_sections_skipped')
        elif len(section_wavs) == 0:
            if verbose:
                print("\n[15.1] Yodfat section analysis - SKIPPED (no sections)")
            results['steps_completed'].append('yodfat_sections_no_input')
        else:
            if verbose:
                print(f"\n[15.1] Computing Yodfat metrics for {len(section_wavs)} sections...")

            yodfat_section_results = yodfat_analysis.run_yodfat_section_analysis(
                sections_dir=str(sections_dir),
                output_dir=str(paths['yodfat_dir']),
                track_id=track_id,
                verbose=verbose
            )

            results['yodfat_sections_json'] = yodfat_section_results.get('output_json')
            results['steps_completed'].append('yodfat_sections')

            if verbose:
                print(f"  ✓ Computed Yodfat metrics for {len(section_wavs)} sections")

    except Exception as e:
        error_msg = f"Step 15.1 failed: {e}"
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
    parser.add_argument('--onset-mode', choices=['librosa', 'drumtranscriber', 'madmom'], default='librosa',
                       help='Onset detection method: librosa (Step 5), drumtranscriber (Step 5.1), or madmom (CNN-based)')
    parser.add_argument('--onset-threshold-drumtranscriber', type=float, default=0.5,
                       help='Minimum onset interval as fraction of 1/16th note (default: 0.5 = 1/32nd note). Only used with --onset-mode drumtranscriber')
    parser.add_argument('--onset-threshold-madmom', type=float, default=0.5,
                       help='Onset detection threshold for madmom (default: 0.5, range: 0.3-0.7). Lower = more sensitive. Only used with --onset-mode madmom')
    parser.add_argument('--anchoring-mode', choices=['single', 'double'], default='double',
                       help='Grid anchoring mode: single (start only) or double (start + end, default)')
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
    parser.add_argument('--reuse-existing', action='store_true',
                       help='Reuse existing stems, beats, and SongFormer files. Skips steps 1-4.5 and 5.5 (pattern detection). Useful for re-running analysis with different onset/anchoring parameters.')
    parser.add_argument('--all-stems', action='store_true',
                       help='Run onset detection and downstream analysis for all 5 stems (vocals, drums, bass, piano, other). Default: drums only.')
    parser.add_argument('--fullmix', action='store_true',
                       help='Also calculate onset detection and analysis on full mix (non-separated audio).')
    parser.add_argument('--fullmix-dir', default=None,
                       help='Directory containing original full mix WAV files (required when using --fullmix with --reuse-existing). Files should be named as {track_id}.wav.')

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

        # When reusing existing files, look for track folders in output directory
        # Each track folder should have 1_stems/ with the original stems
        if args.reuse_existing:
            # Look for existing track folders (directories that contain 1_stems/)
            track_folders = []
            for d in sorted(audio_dir.iterdir()):
                if d.is_dir() and (d / '1_stems').exists():
                    # Find a representative audio file (drums.wav or any stem)
                    stems_dir = d / '1_stems'
                    drums_wav = stems_dir / 'drums.wav'
                    if drums_wav.exists():
                        track_folders.append((d.name, drums_wav))

            if not track_folders:
                print(f"No existing track folders found in {audio_dir}")
                print("  (Expected folders with 1_stems/ subdirectory)")
                sys.exit(1)

            print("=" * 80)
            print(f"Batch Processing Mode (REUSE EXISTING): {len(track_folders)} track folders found")
            print("  Skipping: stems, beats, downbeat correction, tempo plots, SongFormer,")
            print("            pattern detection, onset detection, Pironio, Yodfat")
            print("  Running: anchoring, filtering, histograms, statistics")
            print("=" * 80)

            audio_files = track_folders  # List of (track_id, audio_path) tuples
        else:
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
        for i, item in enumerate(audio_files, start=1):
            # Handle both reuse mode (tuple) and normal mode (Path)
            if args.reuse_existing:
                track_id, wav_file = item
            else:
                wav_file = item
                track_id = wav_file.stem

            print(f"\n{'=' * 80}")
            print(f"Processing [{i}/{len(audio_files)}]: {track_id}")
            print(f"File: {wav_file.name if hasattr(wav_file, 'name') else wav_file}")
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
                    onset_threshold_madmom=args.onset_threshold_madmom,
                    loop_start_offset_ms=args.loop_start_offset_ms,
                    anchoring_mode=args.anchoring_mode,
                    skip_existing=False,
                    create_audio_examples=not args.no_audio_examples,
                    daw_ready=args.daw_ready,
                    manual_start=args.manual_start,
                    manual_duration=args.manual_duration,
                    export_format=args.export_format,
                    reuse_existing=args.reuse_existing,
                    all_stems=args.all_stems,
                    fullmix=args.fullmix,
                    fullmix_dir=args.fullmix_dir,
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

        # Detect available stems for batch analysis
        from batch_analysis.collect_data import detect_available_stems
        batch_track_dirs = sorted([
            d for d in Path(args.output_dir).iterdir()
            if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis',
                                              'snippet_ratio_batch_analysis', 'collected_data']
        ])
        available_stems = detect_available_stems(batch_track_dirs) if batch_track_dirs else ['drums']
        print(f"\nDetected stems for batch analysis: {available_stems}")

        # Merge all plot PDFs (per stem)
        try:
            from batch_analysis.merge_plots import merge_plots
            print("\n" + "=" * 80)
            print("MERGING PLOTS")
            print("=" * 80)
            for stem in available_stems:
                print(f"\n--- Merging plots for {stem} ---")
                merge_plots(Path(args.output_dir), stem=stem)
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

        # Create snippet ratio diagrams (section coverage analysis) - per stem
        try:
            from batch_analysis.snippet_ratio_diagrams import create_snippet_ratio_diagrams
            for stem in available_stems:
                print(f"\n--- Snippet ratio diagrams for {stem} ---")
                create_snippet_ratio_diagrams(Path(args.output_dir), stem=stem)
        except ImportError as ie:
            print(f"\n⚠️  Snippet ratio diagrams skipped: {ie}")
            print("Install required packages: pip install matplotlib numpy")
        except Exception as e:
            print(f"\n⚠️  Snippet ratio diagrams failed: {e}")

        # Step 21: Repetitions per section analysis - per stem
        try:
            from batch_analysis.repetitions_per_section import create_repetitions_diagrams
            for stem in available_stems:
                print(f"\n--- Repetitions per section for {stem} ---")
                create_repetitions_diagrams(Path(args.output_dir), stem=stem)
        except ImportError as ie:
            print(f"\n⚠️  Repetitions per section skipped: {ie}")
            print("Install required packages: pip install matplotlib numpy")
        except Exception as e:
            print(f"\n⚠️  Repetitions per section failed: {e}")

        # Step 22: Collect aggregated data - per stem
        try:
            from batch_analysis.collect_data import create_collected_data, collect_pironio_yodfat_data, collect_spotify_data
            for stem in available_stems:
                create_collected_data(Path(args.output_dir), stem=stem)
            # Pironio/Yodfat and Spotify are not stem-specific
            collect_pironio_yodfat_data(Path(args.output_dir))
            collect_spotify_data(Path(args.output_dir))
        except ImportError as ie:
            print(f"\n⚠️  Collect data skipped: {ie}")
        except Exception as e:
            print(f"\n⚠️  Collect data failed: {e}")

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
            onset_threshold_madmom=args.onset_threshold_madmom,
            loop_start_offset_ms=args.loop_start_offset_ms,
            anchoring_mode=args.anchoring_mode,
            skip_existing=False,
            create_audio_examples=not args.no_audio_examples,
            daw_ready=args.daw_ready,
            manual_start=args.manual_start,
            manual_duration=args.manual_duration,
            export_format=args.export_format,
            reuse_existing=args.reuse_existing,
            all_stems=args.all_stems,
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
