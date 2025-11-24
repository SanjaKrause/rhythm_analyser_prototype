"""
Stem separation module using Spleeter.

⚠️  ENVIRONMENT REQUIREMENT: AEinBOX_13_3
This module requires the AEinBOX_13_3 conda environment.

Dependencies:
    - spleeter
    - tensorflow
    - librosa
    - numpy

Modules:
    spleeter_interface: Spleeter 5-stem separation and mel-spectrogram conversion
"""

from . import spleeter_interface

__all__ = ["spleeter_interface"]
