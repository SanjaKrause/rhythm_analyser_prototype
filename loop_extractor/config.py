"""
Configuration for AP2 Analysis Pipeline.

This module contains all configuration parameters for the complete AP2 workflow.
Main environment: AEinBOX_13_3
Subprocess environment: new_beatnet_env (for Beat-Transformer only)
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Main configuration class for AP2 pipeline."""

    # ============================================================================
    # ENVIRONMENT SETTINGS
    # ============================================================================

    # Main environment (runs spleeter, analysis, everything except beat detection)
    MAIN_ENV = "loop_extractor_main"  # Updated for ARM Mac

    # Subprocess environment (only for Beat-Transformer)
    BEAT_DETECTION_ENV = "new_beatnet_env"

    # Path to conda/python executables
    BEAT_DETECTION_PYTHON = "/Users/alexk/miniforge3/envs/new_beatnet_env/bin/python"

    # ============================================================================
    # DIRECTORY PATHS
    # ============================================================================

    # Project root
    PROJECT_ROOT = Path("/Users/alexk/mastab/loop_extractor_python/loop_extractor")

    # Beat-Transformer model directory
    BEAT_TRANSFORMER_DIR = Path("/Users/alexk/mastab/loop_extractor_python/Beat-Transformer")
    BEAT_TRANSFORMER_CHECKPOINT = BEAT_TRANSFORMER_DIR / "checkpoint" / "fold_4_trf_param.pt"
    BEAT_TRANSFORMER_CODE = BEAT_TRANSFORMER_DIR / "code"

    # Default input/output directories (within project)
    DEFAULT_INPUT_DIR = PROJECT_ROOT / "input_wav"
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_analysis"

    # Overview CSV with snippet offsets (optional - comment out if not available)
    OVERVIEW_CSV = Path("/Users/alexk/mastab/main_project/AP_1/corrected_shift_results.csv")

    # ============================================================================
    # STEP 1: STEM SEPARATION (Spleeter)
    # ============================================================================

    # Spleeter model
    SPLEETER_MODEL = "spleeter:5stems"
    SPLEETER_OUTPUT_CODEC = "wav"

    # Mel-spectrogram parameters (for Beat-Transformer input)
    MEL_SR = 44100              # Sample rate
    MEL_N_FFT = 4096            # FFT window size
    MEL_HOP_LENGTH = 1024       # STFT hop length
    MEL_N_MELS = 128            # Number of mel bins
    MEL_FMIN = 30               # Minimum frequency (Hz)
    MEL_FMAX = 11000            # Maximum frequency (Hz)

    # Stems order
    STEMS = ["vocals", "drums", "bass", "piano", "other"]

    # ============================================================================
    # STEP 2: BEAT DETECTION (Beat-Transformer)
    # ============================================================================

    # Beat-Transformer model parameters
    BT_ATTN_LEN = 5
    BT_INSTR = 5                # 5 stems
    BT_NTOKEN = 2               # beat, downbeat
    BT_DMODEL = 256
    BT_NHEAD = 8
    BT_D_HID = 1024
    BT_NLAYERS = 9
    BT_NORM_FIRST = True
    BT_FOLD = 4                 # Which checkpoint fold to use

    # Madmom DBN parameters
    DBN_MIN_BPM = 55.0
    DBN_MAX_BPM = 215.0
    DBN_FPS = MEL_SR / MEL_HOP_LENGTH  # ~43.066 Hz
    DBN_TRANSITION_LAMBDA = 100
    DBN_OBSERVATION_LAMBDA = 6
    DBN_THRESHOLD = 0.2
    DBN_BEATS_PER_BAR = [3, 4]  # Support 3/4 and 4/4 time signatures

    # ============================================================================
    # STEP 3: CORRECT BARS
    # ============================================================================

    # Correction algorithm parameters
    CORRECT_BARS_OUTLIER_PERCENT = 10.0      # ±10% tolerance for usable tempo
    CORRECT_BARS_MULT_MATCH_TOL = 0.10       # 10% tolerance for factor-of-two matching
    CORRECT_BARS_BPM_THRESHOLD = 135.0       # BPM threshold for dominant tempo adjustment
    CORRECT_BARS_DOUBLE_BPM_ADJUSTED = True  # Enable BPM threshold adjustment
    CORRECT_BARS_SNIPPET_DURATION_S = 30.0   # Duration for snippet visualization

    # ============================================================================
    # STEP 4: RASTER/GRID CALCULATIONS
    # ============================================================================

    # Grid parameters
    GRID_POSITIONS_PER_BAR = 16  # 16th note grid

    # ============================================================================
    # STEP 5: RMS HISTOGRAMS
    # ============================================================================

    # RMS calculation (already implemented in analysis/rms_grid_histograms.py)
    # Uses GRID_POSITIONS_PER_BAR from above

    # ============================================================================
    # STEP 6: AUDIO EXAMPLES
    # ============================================================================

    # Audio export parameters
    AUDIO_EXPORT_FORMAT = "mp3"
    AUDIO_EXPORT_BITRATE = "192k"
    CLICK_TRACK_FREQUENCY = 3000  # Hz for click sound (3000 = sharp, clicky sound)
    CLICK_TRACK_DURATION = 0.050  # 50ms click

    # ============================================================================
    # BATCH PROCESSING
    # ============================================================================

    # Track ID range for batch processing
    USE_ID_RANGE = True
    START_ID = 0
    END_ID = 120

    # Skip existing files
    SKIP_EXISTING = True

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    @classmethod
    def get_output_paths(cls, track_name: str, base_output_dir: Optional[Path] = None, daw_ready: bool = False) -> dict:
        """
        Generate standardized output paths for a track.

        Parameters
        ----------
        track_name : str
            Name of the track (without extension)
        base_output_dir : Path, optional
            Base output directory. If None, uses DEFAULT_OUTPUT_DIR
        daw_ready : bool
            If True, uses 'drum' suffix for MIDI and loops folders

        Returns
        -------
        dict
            Dictionary with paths for each processing step:
            - stems_dir: Directory containing 5 stem WAV files
            - npz_file: Mel-spectrogram NPZ file
            - beats_file: Beat detection output
            - corrected_downbeats_file: Corrected downbeats
            - onsets_file: Onset detection CSV
            - comprehensive_csv: Comprehensive phases CSV
            - rms_summary: RMS summary
            - audio_examples_dir: Directory for audio examples
            - midi_dir: Directory for MIDI export
        """
        if base_output_dir is None:
            base_output_dir = cls.DEFAULT_OUTPUT_DIR

        track_dir = base_output_dir / track_name

        # Use different folder names for DAW ready mode
        midi_folder = '8_midi_drum' if daw_ready else '8_midi'
        loops_folder = '9_loops_drum' if daw_ready else '9_loops'

        return {
            # Step 1: Stem separation
            'stems_dir': track_dir / '1_stems',
            'npz_file': track_dir / '1_stems' / f'{track_name}_5stems.npz',

            # Step 2: Beat detection
            'beats_file': track_dir / '2_beats' / f'{track_name}_output.txt',

            # Step 3: Corrected downbeats
            'corrected_downbeats_file': track_dir / '3_corrected' / f'{track_name}_downbeats_corrected.txt',
            'corrected_summary_csv': track_dir / '3_corrected' / f'{track_name}_downbeat_tempos.csv',
            'corrected_plots_pdf': track_dir / '3_corrected' / f'{track_name}_tempo_plots.pdf',

            # Step 3.5: Tempo plots
            'tempo_plots_dir': track_dir / '3.5_tempo_plots',
            'tempo_plots_pdf': track_dir / '3.5_tempo_plots' / f'{track_name}_tempo_plots.pdf',
            'tempo_csv': track_dir / '3.5_tempo_plots' / f'{track_name}_bar_tempos.csv',

            # Step 4: Onset detection
            'onsets_file': track_dir / '4_onsets' / f'{track_name}_onsets.csv',

            # Step 5: Raster/grid calculations
            'comprehensive_csv': track_dir / '5_grid' / f'{track_name}_comprehensive_phases.csv',

            # Step 6: RMS analysis
            'rms_summary': track_dir / '6_rms' / f'{track_name}_rms_summary.json',

            # Step 7: Audio examples
            'audio_examples_dir': track_dir / '7_audio_examples',

            # Step 8: MIDI export
            'midi_dir': track_dir / midi_folder,

            # Step 9: Stem loops
            'loops_dir': track_dir / loops_folder,
        }

    @classmethod
    def create_output_directories(cls, track_name: str, base_output_dir: Optional[Path] = None, daw_ready: bool = False):
        """Create all necessary output directories for a track."""
        paths = cls.get_output_paths(track_name, base_output_dir, daw_ready=daw_ready)

        for key, path in paths.items():
            if key.endswith('_dir'):
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_environment(cls) -> dict:
        """
        Validate that all required paths and environments exist.

        Returns
        -------
        dict
            Validation results with 'valid' boolean and 'errors' list
        """
        errors = []

        # Check Beat-Transformer paths
        if not cls.BEAT_TRANSFORMER_DIR.exists():
            errors.append(f"Beat-Transformer directory not found: {cls.BEAT_TRANSFORMER_DIR}")

        if not cls.BEAT_TRANSFORMER_CHECKPOINT.exists():
            errors.append(f"Beat-Transformer checkpoint not found: {cls.BEAT_TRANSFORMER_CHECKPOINT}")

        if not cls.BEAT_TRANSFORMER_CODE.exists():
            errors.append(f"Beat-Transformer code directory not found: {cls.BEAT_TRANSFORMER_CODE}")

        # Check Python executable for beat detection
        if not Path(cls.BEAT_DETECTION_PYTHON).exists():
            errors.append(f"Beat detection Python not found: {cls.BEAT_DETECTION_PYTHON}")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }


# Convenience instance for importing
config = Config()
