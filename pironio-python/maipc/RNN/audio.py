from functools import partial

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
									  SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, Processor, SequentialProcessor
import numpy as np


def B_audio_processor() -> SequentialProcessor:
	"""
	Get audio processor for beat model RNN
	Returns: audio preprocessor
	"""
	frame_sizes = [1024, 2048, 4096]
	num_bands = 6
	# define pre-processing chain
	sig = SignalProcessor(num_channels=1, sample_rate=44100)
	# process the multi-resolution spec & diff in parallel
	multi = ParallelProcessor([])
	for frame_size in frame_sizes:
		frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
		stft = ShortTimeFourierTransformProcessor()  # caching FFT window
		filt = FilteredSpectrogramProcessor(num_bands=num_bands, fmin=30,
		                                    fmax=17000, norm_filters=True)
		spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
		diff = SpectrogramDifferenceProcessor(
		    diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
		# process each frame size with spec and diff sequentially
		multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
	# stack the features and processes everything sequentially
	pre_processor = SequentialProcessor((sig, multi, np.hstack))

	return pre_processor


def DB_audio_processor() -> SequentialProcessor:
	"""
	Get audio processor for downbeat model RNN
	Returns: audio preprocessor
	"""
	# define pre-processing chain
	sig = SignalProcessor(num_channels=1, sample_rate=44100)
	# process the multi-resolution spec & diff in parallel
	multi = ParallelProcessor([])
	frame_sizes = [1024, 2048, 4096]
	num_bands = [3, 6, 12] 
	for frame_size, num_bands in zip(frame_sizes, num_bands):
		frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
		stft = ShortTimeFourierTransformProcessor()  # caching FFT window
		filt = FilteredSpectrogramProcessor(
			num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
		spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
		diff = SpectrogramDifferenceProcessor(
			diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
		# process each frame size with spec and diff sequentially
		multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
	# stack the features and processes everything sequentially
	pre_processor = SequentialProcessor((sig, multi, np.hstack))

	return pre_processor


