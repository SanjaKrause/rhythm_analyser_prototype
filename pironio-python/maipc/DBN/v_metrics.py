from typing import Dict, List

from scipy.stats import entropy
import numpy as np


def metrics_for_column(column: np.ndarray, number_of_frames: int = 1) -> Dict[str, float]:
	"""
	For a given column of the viterbi matrix, compute the confidence metrics

	Args:
		column: viterbi matrix column
		number_of_frames: number of frames considered for averaging

	Returns: Dictionary with names of metrics for keys and value asociated.
	"""
	metrics = {"max_over_frames": np.max(column)/number_of_frames,
			   "entropy": entropy(column)}

	return metrics



def viterbi_metrics(matrix: np.ndarray, section: slice =slice(-1,-2,-1)) -> List[Dict[str, int]]:
	"""
	for a slice of columns of a Viterbi matrix, compute the Viterbi metrics
	Args:
		matrix: Viterbi matrix
		section: slice of columns to consider

	Returns: list of viterbi metrics for each column in slice
	"""
	selection = matrix[section]
	metrics_for_selection=[metrics_for_column(column, len(matrix)) 
						   for column in selection]

	return metrics_for_selection

