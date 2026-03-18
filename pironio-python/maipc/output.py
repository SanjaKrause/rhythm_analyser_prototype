import warnings
from typing import Union, List

import numpy as np
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

from .decorators import process_multiple
from .outputs.utils import *


@process_multiple
def peak_average(audio_path: Union[str, List[str]], downbeat_model: bool = True) -> float:
    """
    Peak average pulse clarity metric

    Args:
    audio_path: string or list of strings representing audio paths
    downbeat_model: whether to use the downbeat model

    Returns: pulse clarity metric
    """

    if downbeat_model:
        rnn = RNNDownBeatProcessor()
        beat_activations = np.array([beat_prob + downbeat_prob
                                     for beat_prob, downbeat_prob in rnn(audio_path)])
    else:
        rnn = RNNBeatProcessor()
        beat_activations = rnn(audio_path)

    peak_values = get_peaks(beat_activations)[:, 1]

    return np.mean(peak_values)


@process_multiple
def RNN_entropy(audio_path: Union[str, List[str]], downbeat_model: bool = True) -> float:
    """
    RNN entropy pulse clarity metric

    Args:
    audio_path: string or list of strings representing audio paths
    downbeat_model: whether to use the downbeat model

    Returns: pulse clarity metric
    """

    if downbeat_model:
        rnn = RNNDownBeatProcessor()
        beat_activations = np.array([beat_prob + downbeat_prob
                                     for beat_prob, downbeat_prob in rnn(audio_path)])
    else:
        rnn = RNNBeatProcessor()
        beat_activations = rnn(audio_path)

    peak_moments = get_peaks(beat_activations)[:, 0] / 100

    return KDE_entropy(peak_moments)


@process_multiple
def DBN_entropy(audio_path: Union[str, List[str]], downbeat_model: bool = True) -> float:
    """
    DBN entropy pulse clarity metric

    Args:
    audio_path: string or list of strings representing audio paths
    downbeat_model: whether to use the downbeat model

    Returns: pulse clarity metric
    """
    rnn = RNNDownBeatProcessor() if downbeat_model else RNNBeatProcessor()

    if downbeat_model:
        dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)

        # madmom produces a DeprecationWarning when creating an array from list of tuples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beat_moments = dbn(rnn(audio_path))[:, 0]
    else:
        dbn = DBNBeatTrackingProcessor(fps=100)
        beat_moments = dbn(rnn(audio_path))

    return KDE_entropy(beat_moments)
