"""
AP_2_code: Music Analysis Toolkit

A Python package for analyzing microtiming, tempo, and rhythm in music.
Extracted and refactored from AP2 analysis notebooks.

Modules:
    analysis: Core analysis functions (tempo, microtiming, RMS calculations)
    beat_detection: Beat-Transformer integration (requires new_beatnet_env)
    stem_separation: Spleeter integration (requires AEinBOX env)
    utils: Utility functions and helpers
    config: Configuration management
"""

__version__ = "0.1.0"
__author__ = "AP2 Analysis Project"

# Core analysis imports (base environment)
from .analysis import tempo, microtiming, raster, histograms

# Utility imports
from .utils import file_io, data_processing

__all__ = [
    "tempo",
    "microtiming",
    "raster",
    "histograms",
    "file_io",
    "data_processing",
]
