from typing import List

import numpy as np
from madmom.ml.nn.layers import BidirectionalLayer

from .processing import layers_values, get_bidirectional_values, neurons_values


def neuron_time_series(track_frames: List[np.ndarray], neuron_number: int) -> List:
	"""
	Get the values time series for a specific neuron
	Args:
		track_frames: time series of all neurons for a track
		neuron_number: neuron to get

	Returns: activation values for a neuron
	"""
	num_frames = len(track_frames)
	time_series = [track_frames[frame][neuron_number] for frame in range(num_frames)]
	return time_series

def get_outputs_by_neuron(RNN_input: List, blstms: List[BidirectionalLayer]) -> List[List[float]]:
	"""
	Get neurons values series for an input
	Args:
		RNN_input: input for the NN
		blstms: list of bidirectional LSTM layers

	Returns: values series for each neuron
	"""
	outputs_by_layer = layers_values(RNN_input, blstms, 'output')
	num_frames = len(outputs_by_layer[0]['forward'])

	outputs_by_frame = []
	for frame in range(num_frames):
		all_layers = []
		for layer in range(0,3):
			all_layers.append(outputs_by_layer[layer]['forward'][frame])
			all_layers.append(outputs_by_layer[layer]['backward'][frame])

		outputs_by_frame.append(np.concatenate(all_layers))

	outputs_by_neuron = [neuron_time_series(outputs_by_frame, neuron) for neuron in range(150)]
	return outputs_by_neuron


def mean_cell_state_by_layer(RNN_input: List, blstms: List[BidirectionalLayer]) -> np.ndarray:
	"""
	Get the mean cell state value for each frame, in each layer
	Args:
		RNN_input: input for the NN
		blstms: list of bidirectional LSTM layers

	Returns: mean value per frame, for each layer
	"""

	cell_states_by_layer = layers_values(RNN_input, blstms, 'cell_state')

	num_frames = len(cell_states_by_layer[0]['forward'])
	num_layers = len(cell_states_by_layer)

	mean_by_layer = np.ndarray((num_layers, num_frames))

	for layer in range(num_layers):
		values_for_layer = np.ndarray(num_frames)
		for frame in range(num_frames):
			frame_values = np.concatenate([cell_states_by_layer[layer]['forward'][frame],
										   cell_states_by_layer[layer]['backward'][frame]])
			values_for_layer[frame] = np.mean(frame_values)

		mean_by_layer[layer] = values_for_layer

	return mean_by_layer





