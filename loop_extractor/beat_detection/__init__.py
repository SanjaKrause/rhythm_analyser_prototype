"""
Beat detection module using Beat-Transformer.

This module provides a wrapper that runs Beat-Transformer in the new_beatnet_env
environment via subprocess, allowing the main pipeline to run in AEinBOX_13_3.

Architecture:
    - transformer.py: Wrapper running in AEinBOX_13_3 (calls subprocess)
    - run_transformer.py: Standalone script running in new_beatnet_env

Environment Requirements:
    - Main: AEinBOX_13_3 (for transformer.py)
    - Subprocess: new_beatnet_env (for run_transformer.py with madmom)

Modules:
    transformer: Beat-Transformer subprocess wrapper
"""

from . import transformer

__all__ = ["transformer"]
