from typing import Union, List

from .DBN.classes import DBNBeat_VM, DBNDownBeat_VM
from .DBN.v_metrics import viterbi_metrics
from .decorators import process_multiple

from madmom.features.beats import RNNBeatProcessor
from madmom.features.downbeats import RNNDownBeatProcessor

@process_multiple
def viterbi_max(audio_path: Union[str, List[str]], downbeat_model: bool = True) -> float:
	""" 
	Viterbi max pulse clarity metric for an audio file
	
	Input:
		audio_path: string or list of strings for audio file path
		downbeat_model: boolean. Wether to use downbeat or beat model  
	Output: 
		Viterbi max pulse clarity metric
	"""
	rnn = RNNDownBeatProcessor() if downbeat_model else RNNBeatProcessor()
	dbn = DBNDownBeat_VM(beats_per_bar=[3,4], fps=100) if downbeat_model else DBNBeat_VM(fps=100)

	activations = rnn(audio_path)
	metrics = viterbi_metrics(dbn(activations))[0]

	return metrics['max_over_frames']

@process_multiple
def viterbi_entropy(audio_path: Union[str, List[str]], downbeat_model: bool = True) -> float:
	""" 
	Viterbi entropy pulse clarity metric for an audio file
	
	Input:
		audio_path: string or list of strings for audio file path
		downbeat_model: boolean. Wether to use downbeat or beat model  
	Output: 
		Viterbi max pulse clarity metric
	"""
	rnn = RNNDownBeatProcessor() if downbeat_model else RNNBeatProcessor()
	dbn = DBNDownBeat_VM(beats_per_bar=[3,4], fps=100) if downbeat_model else DBNBeat_VM(fps=100)

	activations = rnn(audio_path)
	metrics = viterbi_metrics(dbn(activations))[0]

	return metrics['entropy']

