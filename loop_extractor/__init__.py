"""
Loop Extractor: Music Analysis and Loop Extraction Toolkit

A Python package for analyzing microtiming, tempo, and rhythm in music,
and extracting perfectly looping audio stems.

Modules:
    analysis: Core analysis functions (tempo, microtiming, RMS calculations)
    beat_detection: Beat-Transformer integration (requires new_beatnet_env)
    stem_separation: Spleeter integration (requires loop_extractor_main env)
    utils: Utility functions and helpers
    config: Configuration management
"""

__version__ = "0.1.0"
__author__ = "Alexander Krause, TU Berlin"

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
