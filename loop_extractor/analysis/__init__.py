"""
Analysis module for music timing and rhythm analysis.

This module contains core analysis functions for microtiming and rhythm analysis.
All functions in this module use the base environment (numpy, pandas, matplotlib).

Modules:
    tempo: Tempo calculation functions
    microtiming: Microtiming deviation analysis
    raster: Raster plot generation and phase calculations
    rms_grid_histograms: RMS histogram analysis
    correct_bars: Downbeat correction
    onset_detection: Onset detection using librosa
"""

from . import tempo
from . import microtiming
from . import raster
from . import rms_grid_histograms
from . import correct_bars
from . import onset_detection

__all__ = ["tempo", "microtiming", "raster", "rms_grid_histograms", "correct_bars", "onset_detection"]
