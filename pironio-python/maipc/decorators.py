from functools import wraps 

def process_multiple(metric_fn):
	"""
	Wrapper for enabling metrics to process a single audio or a list of them
	Args:
		metric_fn: metric function to wrap

	Returns: wrapped function
	"""
	@wraps(metric_fn)
	def flexible_metric_fn(audio_path, downbeat_model=True):
		if type(audio_path) == list:
			return [metric_fn(audio, downbeat_model) for audio in audio_path]
		else:
			return metric_fn(audio_path, downbeat_model)

	return flexible_metric_fn 