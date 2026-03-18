from typing import Union, List

import numpy as np
from scipy.signal import correlate, find_peaks
from madmom.models import DOWNBEATS_BLSTM, BEATS_BLSTM

from .decorators import process_multiple
from .RNN.audio import B_audio_processor, DB_audio_processor
from .RNN.processing import get_blstms
from .RNN.correlation_utils import self_correlate, normalize_corr

DBRNN = DOWNBEATS_BLSTM[-1]
BRNN = BEATS_BLSTM[-1]


@process_multiple
def neurons_cross_correlation(audio_path: Union[str, List[str]], downbeat_model: bool = True) -> float:
    """
    Neurons cross correlation pulse clarity metric

    Args:
    audio_path: string or list of strings representing audio paths
    downbeat_model: whether to use the downbeat model

    Returns: pulse clarity metric
    """
    from .RNN.values_arrangement import get_outputs_by_neuron

    rnn_file = DBRNN if downbeat_model else BRNN
    blstms = get_blstms(rnn_file)
    input_processor = DB_audio_processor() if downbeat_model else B_audio_processor()

    rnn_input = input_processor(audio_path)
    outputs = get_outputs_by_neuron(rnn_input, blstms)

    cross_correlation_sum = 0

    for i in range(1, 150):
        for j in range(i):
            neuron_x = outputs[i]
            neuron_y = outputs[j]

            cross_correlation_sum += np.max(np.absolute(correlate(neuron_x, neuron_y,
                                                                  mode='full',
                                                                  method='fft')))

    return cross_correlation_sum


@process_multiple
def cell_states_precision(audio_path: Union[str, List[str]], downbeat_model: bool = True) -> float:
    """
    Cell states precision pulse clarity metric

    Args:
        audio_path: string or list of strings representing audio paths
        downbeat_model: whether to use the downbeat model

    Returns: pulse clarity metric
    """
    from .RNN.values_arrangement import mean_cell_state_by_layer

    rnn_file = DBRNN if downbeat_model else BRNN
    blstms = get_blstms(rnn_file)

    input_processor = DB_audio_processor() if downbeat_model else B_audio_processor()

    rnn_input = input_processor(audio_path)
    CS_time_series = mean_cell_state_by_layer(rnn_input, blstms)[2]  # Consider only the last layer

    pos = np.clip(CS_time_series, a_min=0, a_max=1)
    neg = np.multiply(np.clip(CS_time_series, a_min=-1, a_max=0), -1)

    widths = []
    for series in [pos, neg]:
        peaks, props = find_peaks(pos, height=0.01, width=np.std(series),
                                  prominence=0.05, distance=10)
        num_peaks = len(peaks)
        peaks_widths = np.array([props['right_bases'][p] - props['left_bases'][p]
                                 for p in range(num_peaks)])

        widths.append(peaks_widths)

    return np.mean(np.concatenate(widths))


@process_multiple
def autocorrelation_periodicity(audio_path: Union[str, List[str]], downbeat_model: bool = True) -> float:
    """
    Autocorrelation periodicity pulse clarity metric

    Args:
        audio_path: string or list of strings representing audio paths
        downbeat_model: whether to use the downbeat model

    Returns: pulse clarity metric
    """
    from .RNN.values_arrangement import get_outputs_by_neuron

    rnn_file = DBRNN if downbeat_model else BRNN
    blstms = get_blstms(rnn_file)

    input_processor = DB_audio_processor() if downbeat_model else B_audio_processor()

    rnn_input = input_processor(audio_path)
    outputs = get_outputs_by_neuron(rnn_input, blstms)

    num_neurons = len(outputs)
    num_frames = len(outputs[0])

    self_correlations = [self_correlate(output)[num_frames - 1:] for output in outputs]

    left_lim = 60000 // (330 * 10)
    right_lim = 60000 // (40 * 10)
    normalized_corrs = [normalize_corr(sc)[left_lim:right_lim] for sc in self_correlations]

    return np.mean(np.max(normalized_corrs, axis=1))
