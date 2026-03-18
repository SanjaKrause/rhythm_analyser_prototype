from typing import List, Tuple

import numpy as np
import KDEpy 
from scipy.stats import entropy as entropy

def KDE_entropy(beats: List, count: int = 400, bw: int = 5,
                min_delta: float = 60000/320, max_delta: float = 60000/8, mult=1000.):
    """
    Compute the entropy value of a gaussian KDE fitted over the inter-beat distribution
    Args:
        beats: audio beats (in miliseconds)
        count: number of points to evaluate over the fitted KDE
        bw: bandwith of the gaussian kernel
        min_delta: minimum distance between taps considered for evaluating
        max_delta: maximum distance between taps considered for evaluating
        mult: factor to multiply the beats in order to have them in miliseconds

    Returns: entropy estimation value
    """
    if len(beats) in [0,1]:
        return np.nan 
    
    beat_diffs = np.diff(beats)*mult
    sample_points = np.linspace(min_delta, max_delta, count)

    estimations = (
                    KDEpy.NaiveKDE(kernel='gaussian', bw=bw)
                         .fit(beat_diffs).evaluate(sample_points)
    )
    
    return entropy(estimations)

def get_peaks(activation: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """
    Simple peak picking process for a beat activation function
    Args:
        activation: beat activation values
        epsilon: threshold

    Returns: array of peaks (tuples), represented as index of the peak and its activation value
    """
    peaks = []
    for i in range(1, len(activation) - 1):
        if activation[i]>activation[i - 1] and activation[i]>activation[i + 1] and activation[i]>=epsilon:
            peaks.append((i, activation[i]))
    return np.array(peaks)


def entropyPeaks(track_peaks: np.ndarray) -> float:
    """
    Return the entropy of the peak moments interpreted as beats
    Args:
        track_peaks: peaks of activation function

    Returns: entropy value
    """

    times = [t/100 for t, p in track_peaks]
    return KDE_entropy(times)