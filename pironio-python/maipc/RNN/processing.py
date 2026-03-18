from typing import List, Dict

import madmom as md
import numpy as np

NN_DTYPE = np.float32


def get_blstms(RNN: str) -> List[md.ml.nn.layers.BidirectionalLayer]:
	"""
	Get the input and bidirectional LSTM layers of a trained model
	Args:
		RNN: path to madmom trained NN
	Returns: list of layers
	"""
	rnn = md.ml.nn.NeuralNetwork.load(RNN)
	return rnn.layers[:3]

def layers_values(RNN_input: List[float], blstms: List[md.ml.nn.layers.BidirectionalLayer],
				  ppty_name: str) -> List[Dict[str, List[np.ndarray]]]:
	"""
	Get internal value activations for an input
	Args:
		RNN_input: input for the RNN
		blstms: list of the bidirectional layers of the network
		ppty_name: the type of values to get

	Returns: values organized by layer, direction (fwd/bwd) and frame
	"""
	layer_input = RNN_input
	layer_values = []
	for bi_layer in blstms:
		layer_input, values = get_bidirectional_values(bi_layer, layer_input, ppty_name) 
		layer_values.append(values)

	return layer_values

def get_bidirectional_values(bi_layer: md.ml.nn.layers.BidirectionalLayer,
							 layer_input: List[float], ppty_name: str) -> Dict[str, List[np.ndarray]]:
	"""
	Get the activation values for the forward and backward layer of a bidirectional layer
	Args:
		bi_layer: bidirectional layer
		layer_input: input to process by the layer
		ppty_name: the type of values to get

	Returns: dictionary with forward and backward layer activation values
	"""
	fwd, fwd_values = neurons_values(bi_layer.fwd_layer, layer_input, ppty_name)
	# also activate with reverse input
	bwd, bwd_values = neurons_values(bi_layer.bwd_layer, layer_input, ppty_name)
	# stack data
	output = np.hstack((fwd, bwd[::-1]))
	return output , {'forward': fwd_values, 'backward': bwd_values}	



def neurons_values(lstm_layer: md.ml.nn.layers.LSTMLayer, data: List[float], ppty_name: str) -> List[np.ndarray]:
	"""
	Get the activation values for a LSTM layer
	Args:
		lstm_layer: LSTM layer
		data: data to process
		ppty_name: the type of values to get

	Returns: List where each position is the activation value for a frame
	"""
	# init arrays
	size = len(data)
	# output matrix for the whole sequence
	out = np.zeros((size, lstm_layer.cell.bias.size), dtype=NN_DTYPE)
	# output list of internal values
	ppty_values = {'cell_state': [], 'output': []}
	# process the input data
	for i in range(size):
		# cache input data
		data_ = data[i]
		# input gate:
		# operate on current data, previous output and state
		ig = lstm_layer.input_gate.activate(data_, lstm_layer._prev, lstm_layer._state)
		# forget gate:
		# operate on current data, previous output and state
		fg = lstm_layer.forget_gate.activate(data_, lstm_layer._prev, lstm_layer._state)
		# cell:
		# operate on current data and previous output
		cell = lstm_layer.cell.activate(data_, lstm_layer._prev)
		# internal state:
		# weight the cell with the input gate
		# and add the previous state weighted by the forget gate
		lstm_layer._state = cell * ig + lstm_layer._state * fg
		# output gate:
		# operate on current data, previous output and current state
		og = lstm_layer.output_gate.activate(data_, lstm_layer._prev, lstm_layer._state)
		# output:
		# apply activation function to state and weight by output gate
		out[i] = lstm_layer.activation_fn(lstm_layer._state) * og
		# set reference to current output
		lstm_layer._prev = out[i]

		# store internal values
		ppty_values['cell_state'].append(cell)
		ppty_values['output'].append(out[i]) 

	return out, ppty_values[ppty_name]