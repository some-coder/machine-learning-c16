import numpy as np


class Loss:

	@staticmethod
	def compute(actual: np.ndarray, expected: np.ndarray) -> float:
		"""
		Calculates the loss metric given actual and expected predictions.

		:param actual: A column vector of mdl predictions.
		:param expected: A column vector of ground truth predictions.
		:return: The loss metric given the two value sets.
		"""
		pass
