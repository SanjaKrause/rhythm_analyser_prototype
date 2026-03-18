import numpy as np
from scipy.signal import correlate

def self_correlate(time_series: np.ndarray) -> np.ndarray:
	"""
	Correlate a time series with itself, considering all possible offsets
	Args:
		time_series: time series to selfcorrelate

	Returns: correlation value for each offset
	"""
	return correlate(time_series, time_series, mode='full', method='fft')
    
def normalize_corr(self_corrs: np.ndarray) -> np.ndarray:
	"""
	Normalize selfcorrelation values by the respective number of frames considered for the correlation computaion
	Args:
		self_corrs: selfcorrelation values

	Returns: normalized selfcorrelation values
	"""
	num_frames = len(self_corrs)
	return np.divide(self_corrs, np.array([num_frames - lag for lag in range(num_frames)]))