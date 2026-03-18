from typing import Tuple

from madmom.ml.hmm import HiddenMarkovModel
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
import numpy as np

from .viterbi import viterbi_matrix


class DBNBeat_VM(DBNBeatTrackingProcessor):
    """
    DBN for a beat-tracking model. Used to obtain the viterbi matrix for RNN activations from a given audio
    """

    def process_offline(self, activations: np.ndarray) -> np.ndarray:
        """
        Get the viterbi matrix for the activations of the RNN
        Args:
            activations: Beat activation function.

        Returns: viterbi matrix
        """
        # init the beats to return and the offset
        beats = np.empty(0, dtype=np.int)
        first = 0
        # use only the activations > threshold
        if self.threshold:
            activations, first = threshold_activations(activations,
                                                       self.threshold)

        # get the last state distribution from viterbi
        matrix, _ = viterbi_matrix(self.tm, self.om,
                                   activations)

        return matrix


def _process_dbn(process_tuple: Tuple[HiddenMarkovModel, np.ndarray]):
    """
    Extract the best path through the state space in an observation sequence.

    This proxy function is necessary to process different sequences in parallel
    using the multiprocessing module.

    Args:
        process_tuple: tuple with an HMM and observations

    Returns: viterbi matrix for the observations using the HMM
    """
    # pylint: disable=no-name-in-module
    return viterbi_matrix(process_tuple[0].transition_model, process_tuple[0].observation_model,
                          process_tuple[1])


class DBNDownBeat_VM(DBNDownBeatTrackingProcessor):
    """
   DBN for a downbeat-tracking model. Used to obtain the viterbi matrix for RNN activations from a given audio
   """
    def process(self, activations: np.ndarray) -> np.ndarray:
        """
        Process the activations of the RNN and get the correponding viterbi matrix
        Args:
            activations: RNN activations for an audio

        Returns: Viterbi matrix

        """
        # pylint: disable=arguments-differ
        import itertools as it
        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            activations, first = threshold_activations(activations,
                                                       self.threshold)
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return np.empty((0, 2))

        # parallel processing of the distinct beats per bar HMM
        results = list(self.map(_process_dbn, zip(self.hmms,
                                                  it.repeat(activations))))

        # choose the best HMM (highest log probability)
        best = np.argmax(np.asarray([prob for matrix, prob in results]))
        # the viterbi matrix for the best HMM
        matrix, _ = results[best]
        return matrix


def threshold_activations(activations: np.ndarray, threshold: float) -> Tuple[np.ndarray, int]:
    """
    Threshold activations to include only the main segment exceeding the given
    threshold (i.e. first to last time/index exceeding the threshold).

    Args:
        activations: Activations to be thresholded.
        threshold: Threshold value.

    Returns: Thresholded activations and the index of the first activation exceeding the threshold.
    """
    first = last = 0
    # use only the activations > threshold
    idx = np.nonzero(activations >= threshold)[0]
    if idx.any():
        first = max(first, np.min(idx))
        last = min(len(activations), np.max(idx) + 1)
    # return thresholded activations segment and first index
    return activations[first:last], first
